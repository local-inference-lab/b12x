"""CuTe KDA decode with FP32 recurrence and caller-owned checkpoint storage.

Eight lanes cooperate on each value row. Token counts, bound capacities, and
all tensor strides are runtime arguments; compiled objects depend on planned
capacity, model geometry, parameter types, and device identity only.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from ._cute_kernels import _add, _fake_pointer, _numeric_type, _pointer


class KdaRecurrence:
    def __init__(self, heads, block_v, qk_l2norm, null_index, validate, state_type):
        self.heads = heads
        self.block_v = block_v
        self.qk_l2norm = qk_l2norm
        self.null_index = null_index
        self.validate = validate
        self.state_type = state_type

    @cute.jit
    def __call__(
        self,
        ptrs: tuple,
        strides: tuple,
        capacities: tuple,
        scale: Float32,
        lower_bound: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(ptrs, strides, capacities, scale, lower_bound).launch(
            grid=(capacities[0] * self.heads * (128 // self.block_v), 1, 1),
            block=(self.block_v * 8, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ptrs: tuple,
        strides: tuple,
        capacities: tuple,
        scale: Float32,
        lower_bound: Float32,
    ):
        (
            mixed,
            raw_g,
            beta,
            alog,
            bias,
            pool,
            starts,
            accepted,
            indices,
            nseq,
            out,
            error,
        ) = ptrs
        sm, sg, sb, sbh, sd, ss, si, sj, so = strides
        seq_capacity, columns = capacities
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = (Int32(bid) % (128 // self.block_v)) * self.block_v + Int32(tid) // 8
        head = (Int32(bid) // (128 // self.block_v)) % self.heads
        request = Int32(bid) // (self.heads * (128 // self.block_v))
        lane = Int32(tid) % 8
        valid = request < cutlass.min(Int32(nseq[0]), seq_capacity)
        if cutlass.const_expr(self.validate):
            valid = valid & (Int32(error[0]) == 0)
        if valid:
            start, end = Int32(starts[request]), Int32(starts[request + 1])
            if start < end:
                index_base = request.to(Int64) * si
                source = indices[index_base + (Int64(accepted[request]) - 1) * sj].to(
                    Int64
                )
                active = Int32(1)
                if cutlass.const_expr(self.null_index is not None):
                    if source == Int64(self.null_index):
                        active = Int32(0)
                state = cute.make_rmem_tensor((16,), Float32)
                head_row = head.to(Int64) * (128 * 128) + row.to(Int64) * 128
                for j in cutlass.range_constexpr(16):
                    state[j] = Float32(0.0)
                    if active != 0:
                        state[j] = Float32(
                            pool[source * ss + head_row + lane.to(Int64) + j * 8]
                        )
                token = start
                while token < cutlass.min(end, start + columns):
                    output_base = (
                        token.to(Int64) * so + head.to(Int64) * 128 + row.to(Int64)
                    )
                    if active == 0:
                        if lane == 0:
                            out[output_base] = BFloat16(0.0)
                    else:
                        q = cute.make_rmem_tensor((16,), Float32)
                        k = cute.make_rmem_tensor((16,), Float32)
                        qsum, ksum = Float32(0.0), Float32(0.0)
                        base = token.to(Int64) * sm + head.to(Int64) * 128
                        for j in cutlass.range_constexpr(16):
                            col = lane.to(Int64) + j * 8
                            q[j] = Float32(mixed[base + col])
                            k[j] = Float32(mixed[base + self.heads * 128 + col])
                            qsum += q[j] * q[j]
                            ksum += k[j] * k[j]
                        qnorm, knorm = Float32(1.0), Float32(1.0)
                        if cutlass.const_expr(self.qk_l2norm):
                            qnorm = cute.math.rsqrt(
                                warp_reduce(qsum, _add, 8) + Float32(1e-6),
                                fastmath=False,
                            )
                            knorm = cute.math.rsqrt(
                                warp_reduce(ksum, _add, 8) + Float32(1e-6),
                                fastmath=False,
                            )
                        gate_scale = cute.math.exp(Float32(alog[head]), fastmath=False)
                        b = cute.arch.rcp_approx(
                            Float32(1.0)
                            + cute.math.exp(
                                -Float32(
                                    beta[token.to(Int64) * sb + head.to(Int64) * sbh]
                                ),
                                fastmath=False,
                            )
                        )
                        dot = Float32(0.0)
                        for j in cutlass.range_constexpr(16):
                            col = lane.to(Int64) + j * 8
                            q[j] = q[j] * qnorm * scale
                            k[j] = k[j] * knorm
                            g = Float32(
                                raw_g[token.to(Int64) * sg + head.to(Int64) * 128 + col]
                            )
                            g += Float32(bias[head.to(Int64) * sd + col])
                            log_decay = lower_bound * cute.arch.rcp_approx(
                                Float32(1.0)
                                + cute.math.exp(-gate_scale * g, fastmath=False)
                            )
                            state[j] *= cute.math.exp(log_decay, fastmath=False)
                            dot += state[j] * k[j]
                        dot = warp_reduce(dot, _add, 8)
                        v = Float32(mixed[base + self.heads * 256 + row.to(Int64)])
                        delta = (v - dot) * b
                        result = Float32(0.0)
                        destination = indices[
                            index_base + (token - start).to(Int64) * sj
                        ].to(Int64)
                        write_state = Int32(1)
                        if cutlass.const_expr(self.null_index is not None):
                            if destination == Int64(self.null_index):
                                write_state = Int32(0)
                        for j in cutlass.range_constexpr(16):
                            state[j] += delta * k[j]
                            result += state[j] * q[j]
                            if write_state != 0:
                                pool[
                                    destination * ss + head_row + lane.to(Int64) + j * 8
                                ] = self.state_type(state[j])
                        result = warp_reduce(result, _add, 8)
                        if lane == 0:
                            out[output_base] = BFloat16(result)
                    token += 1


class KdaNorm:
    def __init__(self, heads, validate):
        self.heads = heads
        self.validate = validate

    @cute.jit
    def __call__(
        self,
        ptrs: tuple,
        strides: tuple,
        capacity: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(ptrs, strides, capacity, eps).launch(
            grid=((capacity * self.heads + 7) // 8, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ptrs: tuple, strides: tuple, capacity: Int32, eps: Float32):
        out, z, weight, ntokens, error = ptrs
        so, sz = strides
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = Int32(bid) * 8 + Int32(tid) // 32
        lane = Int32(tid) % 32
        token, head = row // self.heads, row % self.heads
        if token < capacity:
            ob = token.to(Int64) * so + head.to(Int64) * 128
            zb = token.to(Int64) * sz + head.to(Int64) * 128
            failed = Int32(0)
            if cutlass.const_expr(self.validate):
                failed = Int32(error[0])
            if failed != 0:
                for j in cutlass.range_constexpr(4):
                    out[ob + lane.to(Int64) + j * 32] = BFloat16(float("nan"))
            elif token >= Int32(ntokens[0]):
                for j in cutlass.range_constexpr(4):
                    out[ob + lane.to(Int64) + j * 32] = BFloat16(0.0)
            else:
                values = cute.make_rmem_tensor((4,), Float32)
                squares = Float32(0.0)
                for j in cutlass.range_constexpr(4):
                    values[j] = Float32(out[ob + lane.to(Int64) + j * 32])
                    squares += values[j] * values[j]
                inv = cute.math.rsqrt(
                    warp_reduce(squares, _add) / Float32(128.0) + eps, fastmath=False
                )
                for j in cutlass.range_constexpr(4):
                    col = lane.to(Int64) + j * 32
                    gate = cute.arch.rcp_approx(
                        Float32(1.0)
                        + cute.math.exp(-Float32(z[zb + col]), fastmath=False)
                    )
                    out[ob + col] = BFloat16(
                        values[j] * inv * Float32(weight[col]) * gate
                    )


_CACHE = {}
_WARMED = set()


def compile_kernels(key):
    (
        device,
        max_tokens,
        max_seqs,
        max_slots,
        columns,
        heads,
        block_v,
        qknorm,
        null,
        validate,
        *dtypes,
    ) = key
    types = tuple(_numeric_type(dtype) for dtype in dtypes)
    state_type, index_type, alog_type, bias_type, norm_type = types
    recurrence = KdaRecurrence(heads, block_v, qknorm, null, validate, state_type)
    norm = KdaNorm(heads, validate)
    raise_if_kernel_resolution_frozen("cute.compile", target=recurrence, cache_key=key)
    raw = b12x_compile(
        recurrence,
        tuple(
            _fake_pointer(t)
            for t in (
                BFloat16,
                BFloat16,
                BFloat16,
                alog_type,
                bias_type,
                state_type,
                Int32,
                Int32,
                index_type,
                Int32,
                BFloat16,
                Int32,
            )
        ),
        (Int64(1),) * 9,
        (Int32(1), Int32(1)),
        Float32(1.0),
        Float32(-5.0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "sequence.gdn_decode.kda_recurrence", 1, key
        ),
    )
    raw_norm = b12x_compile(
        norm,
        tuple(_fake_pointer(t) for t in (BFloat16, BFloat16, norm_type, Int32, Int32)),
        (Int64(1), Int64(1)),
        Int32(1),
        Float32(1e-6),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.gdn_decode.kda_norm", 1, key),
    )
    _CACHE[key] = raw, raw_norm
    return raw, raw_norm


def _execute(
    inputs,
    pool,
    output,
    duplicate_slots,
    error_code,
    geometry,
    qknorm,
    validate,
    scale,
    lower_bound,
    eps,
):
    (
        mixed,
        raw_g,
        beta,
        z,
        alog,
        bias,
        weight,
        starts,
        accepted,
        indices,
        nseq,
        ntokens,
    ) = inputs
    max_tokens, max_seqs, max_slots, columns, heads, block_v, null, table_size = (
        geometry
    )
    key = (
        output.device.index,
        max_tokens,
        max_seqs,
        max_slots,
        columns,
        heads,
        block_v,
        qknorm,
        None if null == -1 else null,
        validate,
        pool.dtype,
        indices.dtype,
        alog.dtype,
        bias.dtype,
        weight.dtype,
    )
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _WARMED:
        raise RuntimeError("CuTe KDA must be warm-run before CUDA graph capture")
    raw, norm = _CACHE[key] if key in _CACHE else compile_kernels(key)
    seq_capacity, live_columns = indices.shape
    token_capacity = output.shape[0]
    if validate:
        from . import _kernels as metadata

        metadata._reset_validation_kernel[((table_size + 255) // 256,)](
            duplicate_slots,
            error_code,
            TABLE_SIZE=table_size,
            BLOCK=256,
            num_warps=1,
            num_stages=1,
        )
        metadata._validate_packed_metadata_kernel[(seq_capacity,)](
            starts,
            accepted,
            nseq,
            ntokens,
            error_code,
            token_capacity,
            seq_capacity,
            live_columns,
            num_warps=1,
            num_stages=1,
        )
        metadata._validate_active_state_slots_kernel[(seq_capacity * live_columns,)](
            starts,
            accepted,
            indices,
            nseq,
            duplicate_slots,
            error_code,
            seq_capacity,
            live_columns,
            stride_indices_request=indices.stride(0),
            stride_indices_column=indices.stride(1),
            MAX_STATE_SLOTS=max_slots,
            TABLE_SIZE=table_size,
            HAS_NULL_STATE_INDEX=null != -1,
            NULL_STATE_INDEX=null,
            num_warps=1,
            num_stages=1,
        )

    def ptr(t):
        return _pointer(t, _numeric_type(t.dtype))

    raw(
        tuple(
            ptr(t)
            for t in (
                mixed,
                raw_g,
                beta,
                alog,
                bias,
                pool,
                starts,
                accepted,
                indices,
                nseq,
                output,
                error_code,
            )
        ),
        (
            mixed.stride(0),
            raw_g.stride(0),
            beta.stride(0),
            beta.stride(1),
            bias.stride(0),
            pool.stride(0),
            indices.stride(0),
            indices.stride(1),
            output.stride(0),
        ),
        (seq_capacity, live_columns),
        scale,
        lower_bound,
        current_cuda_stream(),
    )
    norm(
        tuple(ptr(t) for t in (output, z, weight, ntokens, error_code)),
        (output.stride(0), z.stride(0)),
        token_capacity,
        eps,
        current_cuda_stream(),
    )
    if not capturing:
        _WARMED.add(key)


@torch.library.custom_op(
    "b12x::cute_kda_decode",
    mutates_args=("pool", "output", "duplicate_slots", "error_code"),
)
def _op(
    inputs: list[torch.Tensor],
    pool: torch.Tensor,
    output: torch.Tensor,
    duplicate_slots: torch.Tensor,
    error_code: torch.Tensor,
    geometry: list[int],
    qknorm: bool,
    validate: bool,
    scale: float,
    lower_bound: float,
    eps: float,
) -> None:
    with torch.cuda.device(pool.device):
        _execute(
            inputs,
            pool,
            output,
            duplicate_slots,
            error_code,
            geometry,
            qknorm,
            validate,
            scale,
            lower_bound,
            eps,
        )


@_op.register_fake
def _fake(
    inputs,
    pool,
    output,
    duplicate_slots,
    error_code,
    geometry,
    qknorm,
    validate,
    scale,
    lower_bound,
    eps,
):
    return None


def run(binding, *, scale, lower_bound, eps):
    caps = binding.plan.caps
    _op(
        [
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.z,
            binding.A_log,
            binding.dt_bias,
            binding.norm_weight,
            binding.query_start_loc,
            binding.num_accepted_tokens,
            binding.state_indices,
            binding.num_seqs,
            binding.num_tokens,
        ],
        binding.recurrent_state,
        binding.output,
        binding.duplicate_slots,
        binding.error_code,
        [
            caps.max_tokens,
            caps.max_seqs,
            caps.max_state_slots,
            caps.state_index_columns,
            caps.value_heads,
            binding.plan.recurrent_block_v,
            -1 if caps.null_state_index is None else caps.null_state_index,
            binding.plan.duplicate_table_size,
        ],
        caps.qk_l2norm,
        caps.kda_metadata_validation == "transactional",
        scale,
        lower_bound,
        eps,
    )
