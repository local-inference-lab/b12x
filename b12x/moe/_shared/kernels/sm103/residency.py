"""CuTe route compaction, MXFP8 activation boundaries, and ordered finalization."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import pow2_ceil_ue8m0, ue8m0_to_output_scale
from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm


@dsl_user_op
def ordered_fma(a, b, c, *, loc=None, ip=None):
    """One round-to-nearest FP32 FMA, independent of compiler contraction."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


class PartitionRoutes:
    """A deterministic single-CTA baseline with no atomics or reset launch.

    Compact rows retain original route indices. Inactive capacity is never read
    by expert GEMMs. The serial scan is deliberately independently benchmarkable.
    """
    def __init__(self, experts, capacity):
        self.experts, self.capacity = experts, capacity

    @cute.jit
    def __call__(self, ids: cute.Pointer, mapping: cute.Pointer, local_ids: cute.Pointer,
                 indices: cute.Pointer, counts: cute.Pointer, live: Int32, stream: cuda.CUstream):
        self.kernel(ids, mapping, local_ids, indices, counts, live).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, mapping: cute.Pointer, local_ids: cute.Pointer,
               indices: cute.Pointer, counts: cute.Pointer, live: Int32):
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            hot, cold = Int32(0), Int32(0)
            for route in cutlass.range(live):
                expert = Int64(ids[Int64(route)])
                if (expert >= 0) & (expert < self.experts):
                    tier = mapping[expert * 2]
                    row = mapping[expert * 2 + 1]
                    slot = Int64(hot)
                    if tier == 1:
                        slot = Int64(self.capacity) + cold
                        cold += 1
                    else:
                        hot += 1
                    local_ids[slot] = row
                    indices[slot] = route
            counts[0], counts[4] = hot, cold


class OrderedFinalize:
    def __init__(self, hidden, experts):
        self.hidden, self.experts = hidden, experts

    @cute.jit
    def __call__(self, rows: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer,
                 out: cute.Pointer, tokens: Int32, top_k: Int32, stream: cuda.CUstream):
        self.kernel(rows, ids, weights, out, top_k).launch(
            grid=(tokens, cute.ceil_div(self.hidden, 128), 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, rows: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer,
               out: cute.Pointer, top_k: Int32):
        tid, _, _ = cute.arch.thread_idx()
        token, block, _ = cute.arch.block_idx()
        col = block * 128 + tid
        if col < self.hidden:
            acc = Float32(0)
            for rank in cutlass.range(top_k):
                route = Int64(token) * top_k + rank
                expert = Int64(ids[route])
                if (expert >= 0) & (expert < self.experts):
                    acc = ordered_fma(Float32(weights[route]), Float32(rows[route * self.hidden + col]), acc)
            out[Int64(token) * self.hidden + col] = acc.to(cutlass.BFloat16)


class QuantizeMxRoutes:
    """MXFP8 K32 activations; FC1 stays FP32 through the gated activation."""
    def __init__(self, k, experts, *, activation=False, gate_first=False, limit=None):
        self.k, self.experts = k, experts
        self.activation, self.gate_first, self.limit = activation, gate_first, limit

    @cute.jit
    def __call__(self, x: cute.Pointer, ids: cute.Pointer, q: cute.Pointer,
                 sf: cute.Pointer, live: Int32, top_k: Int32, stream: cuda.CUstream):
        self.kernel(x, ids, q, sf, top_k).launch(
            grid=(live, cute.ceil_div(self.k // 32, 128), 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, ids: cute.Pointer, q: cute.Pointer,
               sf: cute.Pointer, top_k: Int32):
        tid, _, _ = cute.arch.thread_idx()
        route, block, _ = cute.arch.block_idx()
        group = block * 128 + tid
        row = Int64(route)
        if group < self.k // 32:
            expert = Int64(ids[row])
            valid = (expert >= 0) & (expert < self.experts)
            values = cute.make_rmem_tensor(32, Float32)
            maximum = Float32(0)
            for j in cutlass.range_constexpr(32):
                value = Float32(0)
                if valid:
                    if cutlass.const_expr(self.activation):
                        offset = row * (2 * self.k) + group * 32 + j
                        up, gate = Float32(x[offset]), Float32(x[offset + self.k])
                        if cutlass.const_expr(self.gate_first):
                            gate, up = up, gate
                        if cutlass.const_expr(self.limit is not None):
                            gate = cute.arch.fmin(gate, Float32(self.limit))
                            up = cute.arch.fmax(cute.arch.fmin(up, Float32(self.limit)), Float32(-self.limit))
                        value = (gate / (1.0 + cute.math.exp(-gate))) * up
                    else:
                        value = Float32(x[(row // top_k) * self.k + group * 32 + j])
                values[j] = value
                maximum = cute.arch.fmax(maximum, cute.math.abs(value))
            # Match the shared b12x/FlashInfer power-of-two scale boundary.
            _, scale = pow2_ceil_ue8m0(maximum / 448.0)
            if maximum == Float32(0):
                scale = cutlass.Uint32(127)
            inverse = ue8m0_to_output_scale(scale)
            for j in cutlass.range_constexpr(32):
                q[row * self.k + group * 32 + j] = (values[j] * inverse).to(cutlass.Float8E4M3FN)
            for padding_row in cutlass.range_constexpr(128):
                offset = row * (128 * (self.k // 32)) + (group // 4) * 512
                offset += (padding_row % 32) * 16 + (padding_row // 32) * 4 + group % 4
                sf[offset] = scale.to(Uint8)


class RoutedMxGemm(BlockscaledGemm):
    def __init__(self, n, k, experts, capacity, *, fc1):
        super().__init__(n, k, experts, recipe="w4a8_mx", routed_capacity=capacity,
                         c_dtype=cutlass.Float32 if fc1 else cutlass.BFloat16,
                         alpha_is_one=True)

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, sa: cute.Pointer, sb: cute.Pointer,
                 c: cute.Pointer, ids: cute.Pointer, indices: cute.Pointer, count: cute.Pointer,
                 alpha: cute.Pointer, live: Int32, stream: cuda.CUstream):
        self._launch(a, b, sa, sb, c, ids, alpha, live, Int64(self.k), Int64(self.n), Int64(1),
                     stream, route_indices=indices, route_count=count)
