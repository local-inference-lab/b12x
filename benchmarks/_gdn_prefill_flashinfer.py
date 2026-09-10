"""Benchmark-local CuTe preparation for FlashInfer's SM120 GDN prefill API."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.intrinsics import warp_reduce
from b12x._lib.utils import current_cuda_stream, make_ptr


def _add(a: Float32, b: Float32) -> Float32:
    return a + b


def _ptr(tensor):
    dtype = {torch.bfloat16: BFloat16, torch.float32: Float32,
             torch.int32: Int32, torch.int64: Int64}[tensor.dtype]
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=dtype.width // 8)


class _NormalizeGates:
    def __init__(self, key_heads, value_heads):
        self.key_heads = key_heads
        self.value_heads = value_heads

    @cute.jit
    def __call__(self, q: cute.Pointer, k: cute.Pointer, a: cute.Pointer, b: cute.Pointer,
                 A: cute.Pointer, bias: cute.Pointer, qn: cute.Pointer, kn: cute.Pointer,
                 alpha: cute.Pointer, beta: cute.Pointer, tokens: Int32, stream: cuda.CUstream):
        self.kernel(q, k, a, b, A, bias, qn, kn, alpha, beta).launch(
            grid=(tokens, self.value_heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Pointer, k: cute.Pointer, a: cute.Pointer, b: cute.Pointer,
               A: cute.Pointer, bias: cute.Pointer, qn: cute.Pointer, kn: cute.Pointer,
               alpha: cute.Pointer, beta: cute.Pointer):
        token, head, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if head % Int32(3) == Int32(0):
            base = (token.to(Int64) * Int64(self.key_heads) + (head // Int32(3)).to(Int64)) * Int64(128)
            qv = cute.make_rmem_tensor((4,), Float32)
            kv = cute.make_rmem_tensor((4,), Float32)
            qs, ks = Float32(0), Float32(0)
            for i in cutlass.range_constexpr(4):
                qv[i] = Float32(q[base + (lane + Int32(32*i)).to(Int64)])
                kv[i] = Float32(k[base + (lane + Int32(32*i)).to(Int64)])
                qs += qv[i] * qv[i]
                ks += kv[i] * kv[i]
            qr = cute.math.rsqrt(warp_reduce(qs, _add, 32) + Float32(1e-6), fastmath=False)
            kr = cute.math.rsqrt(warp_reduce(ks, _add, 32) + Float32(1e-6), fastmath=False)
            for i in cutlass.range_constexpr(4):
                qn[base + (lane + Int32(32*i)).to(Int64)] = BFloat16(qv[i] * qr)
                kn[base + (lane + Int32(32*i)).to(Int64)] = BFloat16(kv[i] * kr)
        if lane == Int32(0):
            offset = token.to(Int64) * Int64(self.value_heads) + head.to(Int64)
            z = Float32(a[offset]) + Float32(bias[head])
            softplus = z
            if z <= Float32(20):
                softplus = cute.math.log1p(cute.math.exp(z, fastmath=False), fastmath=False)
            alpha[offset] = cute.math.exp(-cute.math.exp(Float32(A[head]), fastmath=False) * softplus,
                                          fastmath=False)
            activated = Float32(1) / (Float32(1) + cute.math.exp(-Float32(b[offset]), fastmath=False))
            beta[offset] = Float32(BFloat16(activated))


class _PoolTransfer:
    def __init__(self, heads, scatter):
        self.heads = heads
        self.scatter = scatter

    @cute.jit
    def __call__(self, pool: cute.Pointer, compact: cute.Pointer, indices: cute.Pointer,
                 cu32: cute.Pointer, cu64: cute.Pointer, seqs: Int32, stream: cuda.CUstream):
        self.kernel(pool, compact, indices, cu32, cu64, seqs).launch(
            grid=(64, self.heads, seqs), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, pool: cute.Pointer, compact: cute.Pointer, indices: cute.Pointer,
               cu32: cute.Pointer, cu64: cute.Pointer, seqs: Int32):
        block, head, seq = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        element = (block * Int32(256) + lane).to(Int64)
        source = (indices[seq].to(Int64) * Int64(self.heads) + head.to(Int64)) * Int64(16384) + element
        target = (seq.to(Int64) * Int64(self.heads) + head.to(Int64)) * Int64(16384) + element
        if cutlass.const_expr(self.scatter):
            pool[source] = compact[target]
        else:
            compact[target] = pool[source]
            if block == Int32(0) and head == Int32(0) and lane == Int32(0):
                cu64[seq] = cu32[seq].to(Int64)
                if seq == seqs - Int32(1):
                    cu64[seqs] = cu32[seqs].to(Int64)


class FlashInferArm:
    """Raw projections and pooled state to recurrence output and pooled final state.

    This adapter is restricted to valid, non-null final-state-only benchmark
    cases. It neither replaces nor alters the installed FlashInfer kernels.
    """

    def __init__(self, case, tensors, *, use_cp):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        self.fn = chunk_gated_delta_rule
        self.use_cp = use_cp
        self.case = case
        self.tensors = tensors
        self.q = torch.empty_like(tensors["q"][:case.tokens])
        self.k = torch.empty_like(tensors["k"][:case.tokens])
        self.v = tensors["v"][:case.tokens]
        self.output = tensors["output"][:case.tokens]
        self.alpha = torch.empty_like(tensors["raw_g"][:case.tokens], dtype=torch.float32)
        self.beta = torch.empty_like(self.alpha)
        shape = (len(case.lengths), case.value_heads, 128, 128)
        self.initial = torch.empty(shape, dtype=torch.float32, device=self.q.device)
        self.final = torch.empty_like(self.initial)
        self.cu = torch.empty(len(case.lengths)+1, dtype=torch.int64, device=self.q.device)
        self.norm_args = tuple(_ptr(t) for t in (
            tensors["q"], tensors["k"], tensors["raw_g"], tensors["raw_beta"], tensors["A_log"],
            tensors["dt_bias"], self.q, self.k, self.alpha, self.beta))
        self.gather_args = tuple(_ptr(t) for t in (
            tensors["recurrent_state"], self.initial, tensors["initial_state_indices"],
            tensors["cu_seqlens"], self.cu))
        self.scatter_args = tuple(_ptr(t) for t in (
            tensors["recurrent_state"], self.final, tensors["final_state_indices"],
            tensors["cu_seqlens"], self.cu))
        stream = current_cuda_stream()
        self.normalize = cute.compile(_NormalizeGates(case.key_heads, case.value_heads),
                                      *self.norm_args, Int32(case.tokens), stream)
        self.gather = cute.compile(_PoolTransfer(case.value_heads, False),
                                   *self.gather_args, Int32(len(case.lengths)), stream)
        self.scatter = cute.compile(_PoolTransfer(case.value_heads, True),
                                    *self.scatter_args, Int32(len(case.lengths)), stream)

    @property
    def buffers(self):
        return (self.q, self.k, self.alpha, self.beta, self.initial, self.final, self.cu)

    def poison(self):
        for tensor in self.buffers:
            tensor.fill_(-1 if tensor.dtype == torch.int64 else float("nan"))

    def __call__(self):
        stream = current_cuda_stream()
        self.normalize(*self.norm_args, Int32(self.case.tokens), stream)
        self.gather(*self.gather_args, Int32(len(self.case.lengths)), stream)
        self.fn(self.q, self.k, self.v, g=self.alpha, beta=self.beta,
                scale=128**-0.5, initial_state=self.initial, output_final_state=True,
                cu_seqlens=self.cu, use_qk_l2norm_in_kernel=False,
                output=self.output, output_state=self.final, use_cp=self.use_cp)
        self.scatter(*self.scatter_args, Int32(len(self.case.lengths)), stream)
