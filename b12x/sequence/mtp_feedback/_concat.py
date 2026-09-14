"""CuTe RMS normalization, position masking and concatenated feedback projection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch

from b12x._lib.architecture import UnsupportedArchitectureError, supports_architecture
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.intrinsics import block_reduce, warp_reduce
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import materialize_scratch_view
from b12x._lib.utils import current_cuda_stream, make_ptr
from ._cute_prefill import compile_mtp_prefill_bf16_gemm


def _add(left: c.Float32, right: c.Float32) -> c.Float32:
    return left + right


class NormalizeConcat:
    """Independent RMS groups with zero-position masking on embeddings."""

    def __init__(self, hidden: int, block_h: int, warps: int, streams=1, capacity=0):
        self.hidden = hidden
        self.threads = warps * 32
        self.warps = warps
        self.items = cute.ceil_div(block_h, self.threads)
        self.streams = streams
        self.capacity = capacity

    @cute.jit
    def __call__(
        self,
        embedding,
        state,
        e_weight,
        h_weight,
        positions,
        output,
        eps: c.Float32,
        tokens: c.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            embedding, state, e_weight, h_weight, positions, output, eps
        ).launch(
            grid=(tokens, self.streams + 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, embedding, state, e_weight, h_weight, positions, output, eps: c.Float32
    ):
        token, group, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        base = c.Int64(token) * self.hidden
        output_base = c.Int64(token) * (2 * self.hidden) + c.Int64(group) * self.hidden
        state_base = base
        if c.const_expr(self.capacity > 0):
            state_base = (
                c.Int64(token) * self.streams + c.Int64(group) - 1
            ) * self.hidden
            output_base = base
            if group > 0:
                output_base = c.Int64(self.capacity) * self.hidden + state_base
        keep_embedding = positions[c.Int64(token)] != 0
        total = c.Float32(0)
        for item in c.range_constexpr(self.items):
            column = thread + item * self.threads
            if column < self.hidden:
                value = c.Float32(0)
                if group == 0:
                    if keep_embedding:
                        value = c.Float32(embedding[base + c.Int64(column)])
                else:
                    value = c.Float32(state[state_base + c.Int64(column)])
                total += value * value
        allocator = c.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            c.Float32,
            cute.make_layout((1, self.warps)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            c.Float32, cute.make_layout((1,)), byte_alignment=4
        )
        total = block_reduce(warp_reduce(total, _add), _add, reduction, c.Float32(0))
        if thread == 0:
            inverse[0] = cute.math.rsqrt(total / self.hidden + eps, fastmath=True)
        cute.arch.sync_threads()
        for item in c.range_constexpr(self.items):
            column = thread + item * self.threads
            if column < self.hidden:
                value = c.Float32(0)
                weight = c.Float32(0)
                if group == 0:
                    if keep_embedding:
                        value = c.Float32(embedding[base + c.Int64(column)])
                    weight = c.Float32(e_weight[c.Int64(column)])
                else:
                    value = c.Float32(state[state_base + c.Int64(column)])
                    weight = c.Float32(h_weight[c.Int64(column)])
                output[output_base + c.Int64(column)] = c.BFloat16(
                    value * inverse[0] * weight
                )


def _pointer(tensor, dtype=c.BFloat16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


@lru_cache(maxsize=None)
def compile_norm(
    hidden,
    block_h,
    warps,
    position_dtype,
    device_index,
    architecture,
    streams=1,
    capacity=0,
):
    kernel = NormalizeConcat(hidden, block_h, warps, streams, capacity)
    position_type = c.Int64 if position_dtype == torch.int64 else c.Int32

    def ptr(dtype):
        return make_ptr(
            dtype, 16, cute.AddressSpace.gmem, assumed_align=dtype.width // 8
        )

    with torch.cuda.device(device_index):
        return compile_cute(
            kernel,
            ptr(c.BFloat16),
            ptr(c.BFloat16),
            ptr(c.BFloat16),
            ptr(c.BFloat16),
            ptr(position_type),
            ptr(c.BFloat16),
            c.Float32(1e-6),
            c.Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "sequence.mtp_feedback.rms_concat",
                2,
                ("hidden", hidden),
                ("block_h", block_h),
                ("warps", warps),
                ("position_dtype", str(position_dtype)),
                ("architecture", architecture),
                ("streams", streams),
                ("capacity", capacity),
            ),
        )


@dataclass(frozen=True)
class BoundLaunches:
    norm: object
    norm_args: tuple
    concatenated: torch.Tensor


@dataclass
class BackendPlan:
    rows: int
    norms: dict
    projection: object
    warmed: bool = False

    def bind(
        self,
        planned,
        *,
        scratch,
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        combined_fc_weight,
        positions,
        output,
        tokens,
    ):
        from ._impl import Binding, _overlaps, _require_tensor

        caps = planned.caps
        if not isinstance(token_embedding, torch.Tensor) or token_embedding.ndim != 2:
            raise ValueError("token_embedding must have shape [rows, hidden_size]")
        live = planned._live_tokens(
            token_embedding.shape[0] if tokens is None else tokens
        )
        available = int(token_embedding.shape[0])
        if not live <= available <= caps.max_tokens:
            raise ValueError(
                "RMS-concat input rows must cover live tokens within capacity"
            )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or not live <= output.shape[0] <= caps.max_tokens
        ):
            raise ValueError(
                "RMS-concat output rows must cover live tokens within capacity"
            )
        shapes = {
            "token_embedding": (available, caps.hidden_size),
            "multi_state": (available, caps.hidden_size),
            "token_norm_weight": (caps.hidden_size,),
            "state_norm_weight": (caps.hidden_size,),
            "combined_fc_weight": (caps.hidden_size, 2 * caps.hidden_size),
            "output": (output.shape[0], caps.hidden_size),
        }
        tensors = dict(
            token_embedding=token_embedding,
            multi_state=multi_state,
            token_norm_weight=token_norm_weight,
            state_norm_weight=state_norm_weight,
            combined_fc_weight=combined_fc_weight,
            output=output,
        )
        for name, shape in shapes.items():
            if not isinstance(tensors[name], torch.Tensor):
                raise TypeError(f"{name} must be a tensor for RMS-concat feedback")
            _require_tensor(name, tensors[name], shape=shape, caps=caps)
        if (
            not isinstance(positions, torch.Tensor)
            or positions.shape != (available,)
            or positions.dtype not in (torch.int32, torch.int64)
            or positions.device != caps.device
            or not positions.is_contiguous()
        ):
            raise ValueError(
                "positions must be contiguous Int32/Int64 positions on the plan device"
            )
        if output.data_ptr() % 16 or combined_fc_weight.data_ptr() % 16:
            raise ValueError(
                "RMS-concat projection weights and output must be 16-byte aligned"
            )
        storage = scratch_tensor(
            scratch, planned.scratch_specs(), owner="RMS-concat feedback"
        )
        concatenated, _ = materialize_scratch_view(
            storage,
            offset_bytes=0,
            shape=(self.rows, 2 * caps.hidden_size),
            dtype=caps.dtype,
        )
        readonly = [
            (name, value) for name, value in tensors.items() if name != "output"
        ] + [("positions", positions)]
        for name, mutable in (("scratch", storage), ("output", output)):
            for source_name, source in readonly:
                if _overlaps(mutable, source):
                    raise ValueError(f"{name} must not overlap read-only {source_name}")
        if _overlaps(storage, output):
            raise ValueError("scratch and output must not overlap")
        position_type = c.Int64 if positions.dtype == torch.int64 else c.Int32
        bound = BoundLaunches(
            self.norms[positions.dtype],
            tuple(
                _pointer(t)
                for t in (
                    token_embedding,
                    multi_state,
                    token_norm_weight,
                    state_norm_weight,
                )
            )
            + (_pointer(positions, position_type), _pointer(concatenated)),
            concatenated,
        )
        return Binding(
            plan=planned,
            tokens=live,
            scratch=storage,
            token_normalized=concatenated[:live, : caps.hidden_size],
            state_partial_sums=None,
            state_normalized=concatenated[:live, caps.hidden_size :],
            token_path=None,
            token_embedding=token_embedding[:live],
            multi_state=multi_state[:live],
            token_norm_weight=token_norm_weight,
            state_norm_weight=state_norm_weight,
            embedding_fc_weight=None,
            hidden_fc_weight=None,
            output=output[:live],
            combined_fc_weight=combined_fc_weight,
            positions=positions[:live],
            _backend_binding=bound,
        )

    def run(self, binding, *, eps):
        if torch.cuda.is_current_stream_capturing() and not self.warmed:
            raise RuntimeError(
                "RMS-concat feedback must be warm-run before graph capture"
            )
        bound = binding._backend_binding
        run_compiled(
            bound.norm,
            (
                *bound.norm_args,
                c.Float32(eps),
                c.Int32(binding.tokens),
                current_cuda_stream(),
            ),
        )
        self.projection(
            bound.concatenated,
            binding.combined_fc_weight,
            binding.output.reshape(-1),
            live_rows=binding.tokens,
        )
        self.warmed = True
        return binding.output


def plan(caps, resolution):
    from ._impl import Plan

    device = resolution.device
    if device is None or not supports_architecture(
        device.compute_capability, ("sm103a", "sm120a", "sm121a")
    ):
        raise UnsupportedArchitectureError(
            "RMS-concat feedback requires an implemented Blackwell architecture"
        )
    rows = ((caps.max_tokens + 15) // 16) * 16
    config = resolution.config
    norms = {
        dtype: compile_norm(
            caps.hidden_size,
            config.norm_block_h,
            config.norm_num_warps,
            dtype,
            caps.device.index,
            device.compute_capability,
        )
        for dtype in (torch.int32, torch.int64)
    }
    projection = compile_mtp_prefill_bf16_gemm(
        rows,
        caps.hidden_size,
        2 * caps.hidden_size,
        device=caps.device,
        streams=1,
        add_token_path=False,
    )
    backend = BackendPlan(rows, norms, projection)
    spec = scratch_buffer_spec(
        "mtp_feedback", nbytes=rows * 2 * caps.hidden_size * 2, device=caps.device
    )
    return Plan(
        caps=caps,
        token_normalized_offset_bytes=None,
        state_partial_sums_offset_bytes=None,
        state_normalized_offset_bytes=None,
        token_path_offset_bytes=None,
        _scratch_specs=(spec,),
        token_projection_rows=rows,
        state_projection_rows=rows,
        norm_block_h=config.norm_block_h,
        norm_block_s=config.norm_block_s,
        norm_num_warps=config.norm_num_warps,
        policy_resolution=resolution,
        _backend_plan=backend,
    )
