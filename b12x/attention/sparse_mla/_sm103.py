"""Planned GLM sparse attention using ordinary FP8 and BF16 warp MMA.

Decode partitions the planned selection width into fixed splits. Extend uses
the shared multigroup prefill pipeline. All storage belongs to the public plan;
bound row counts and physical pool strides are runtime launch arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from .._shared.mla.kernel import (
    UnifiedDecodeKernel,
    _cache_base_tensor,
    _cache_block_stride_bytes,
    _to_cute,
)
from .._shared.mla.merge import (
    SparseMLASplitDecodeMergeKernel,
    SparseMLASplitDecodeSinkMergeKernel,
)
from .._shared.mla.prefill_mg import UnifiedPrefillMGKernel
from .._shared.mla.smem import make_smem_layout
from .._shared.mla.smem_mg import make_smem_layout_mg

_CACHE = {}
_PLANS = {}


class SplitLse:
    """Emit the selected-token LSE in the public sparse-MLA convention."""

    def __init__(self, num_splits, natural):
        self.num_splits = num_splits
        self.natural = natural

    @cute.jit
    def __call__(
        self, partial: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(partial, output).launch(
            grid=(output.shape[0], output.shape[1], 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, partial: cute.Tensor, output: cute.Tensor):
        row, head, _ = cute.arch.block_idx()
        row, head = Int64(row), Int64(head)
        lane = cute.arch.lane_idx()
        maximum = Float32(-Float32.inf)
        index = Int32(lane)
        while index < self.num_splits:
            maximum = cutlass.max(maximum, Float32(partial[row, head, index]))
            index += 32
        for offset in (16, 8, 4, 2, 1):
            maximum = cutlass.max(
                maximum, cute.arch.shuffle_sync_bfly(maximum, offset=offset)
            )
        total = Float32(0.0)
        index = Int32(lane)
        while index < self.num_splits:
            value = Float32(partial[row, head, index])
            if value != Float32(-Float32.inf):
                total += cute.math.exp2(value - maximum, fastmath=True)
            index += 32
        for offset in (16, 8, 4, 2, 1):
            total += cute.arch.shuffle_sync_bfly(total, offset=offset)
        if lane == 0:
            result = Float32(-Float32.inf)
            if total > Float32(0.0):
                result = maximum + cute.math.log2(total, fastmath=True)
            if cutlass.const_expr(self.natural):
                result *= Float32(math.log(2.0))
            output[row, head] = result


class ConvertLse:
    @cute.jit
    def __call__(self, lse: cute.Tensor, stream: cuda.CUstream):
        self.kernel(lse).launch(
            grid=(cute.ceil_div(Int64(lse.shape[0]) * Int64(lse.shape[1]), 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, lse: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = Int64(bid) * 128 + Int64(tid)
        if index < Int64(lse.shape[0]) * Int64(lse.shape[1]):
            row, head = index // lse.shape[1], index % lse.shape[1]
            lse[row, head] = Float32(lse[row, head]) * Float32(math.log(2.0))


@dataclass
class Runtime:
    plan: object
    binding: object
    sink: torch.Tensor | None
    output: torch.Tensor
    lse: torch.Tensor
    plan_key: str
    prepared: object | None = None


def bind(plan, binding, sink):
    from .._shared.mla.api import (
        _is_supported_packed_kv_cache_view,
        _validate_tensor_storage_bounds,
    )

    cache = binding.kv_cache
    if not cache.is_contiguous() and not _is_supported_packed_kv_cache_view(
        cache,
        page_size=plan.caps.page_size,
    ):
        raise ValueError(
            "sparse MLA requires contiguous or packed page-strided KV records"
        )
    _validate_tensor_storage_bounds(cache, name="sparse MLA KV cache")
    rows = binding.q.shape[0]
    scratch = binding.scratch
    key = repr((plan.caps, plan.policy_resolution.config))
    _PLANS.setdefault(key, plan)
    return Runtime(
        plan, binding, sink, scratch.output_buffer[:rows], scratch.final_lse[:rows], key
    )


def launch_specs(runtime):
    """Build kernel objects and launch arguments without device allocations."""
    caps = runtime.plan.caps
    config = runtime.plan.policy_resolution.config
    bound = runtime.binding
    scratch = bound.scratch
    traits = caps.cache_traits
    q, kv, indices, lengths = (
        bound.q,
        bound.kv_cache,
        bound.selected_indices,
        bound.nsa_cache_seqlens_int32,
    )
    heads = caps.num_q_heads
    rows = q.shape[0]
    q_c = _to_cute(q, cutlass.BFloat16, dynamic_layout=True)
    # The kernel consumes a raw pool base and an Int64 physical page stride.
    # A one-byte pointer view avoids a 32-bit dynamic DLPack extent for pools
    # larger than 2 GiB; all selected-row addressing remains in the kernel.
    kv_c = _to_cute(_cache_base_tensor(kv)[:1], cutlass.Uint8)
    index_c = _to_cute(indices, cutlass.Int32, align=4, dynamic_layout=True)
    length_c = _to_cute(lengths, cutlass.Int32, align=4, dynamic_layout=True)
    output_c = _to_cute(runtime.output, cutlass.BFloat16, dynamic_layout=True)
    lse_c = _to_cute(runtime.lse, cutlass.Float32, align=4, dynamic_layout=True)
    sink_c = _to_cute(
        runtime.sink if runtime.sink is not None else scratch.sm_scale_tensor,
        cutlass.Float32,
        align=4,
        dynamic_layout=True,
    )
    scale = Float32(caps.softmax_scale * math.log2(math.e))
    latent = Float32(caps.latent_scale)
    stride = cutlass.Int64(
        _cache_block_stride_bytes(
            kv,
            page_size=caps.page_size,
            model_type=traits.model_type,
            record_bytes=traits.kv_gmem_stride,
        )
    )
    count = Int32(rows)
    # The stream is attached at execution so a warmed binding can be captured
    # on another stream without retaining its warmup stream.
    launches = []
    segments = [(heads // 16, 16, 0)] if heads >= 16 else []
    if heads % 16:
        segments.append((1, heads % 16, heads // 16))
    if caps.mode == "decode":
        splits = config.num_splits
        chunks_per_split = ((caps.max_width + 63) // 64 + splits - 1) // splits
        partial = scratch.tmp_output
        partial_lse = scratch.tmp_lse
        partial_c = _to_cute(partial, cutlass.BFloat16, dynamic_layout=True)
        partial_lse_c = _to_cute(
            partial_lse, cutlass.Float32, align=4, dynamic_layout=True
        )
        args = (
            q_c,
            kv_c,
            index_c,
            partial_c,
            partial_lse_c,
            scale,
            latent,
            length_c,
            stride,
            count,
        )
        for blocks, valid, offset in segments:
            kernel = UnifiedDecodeKernel(
                traits,
                make_smem_layout(traits),
                caps.page_size,
                chunks_per_split,
                h_blocks=blocks,
                num_splits=splits,
                num_heads=heads,
                q_head_dim=caps.head_dim,
                topk=caps.max_width,
                extra_topk=0,
                q_stride=q.stride(),
                swa_indices_stride0=indices.stride(0),
                extra_indices_stride0=indices.stride(0),
                mid_out_stride=partial.stride(),
                mid_lse_stride=partial_lse.stride(),
                valid_hpb=valid,
                head_block_offset=offset,
                per_token_len=True,
                vector_q=True,
                block_scaled_mma=False,
            )
            launches.append((f"decode_h{valid}_o{offset}", kernel.call_pertok, args))
        control_c = _to_cute(
            scratch.num_chunks_ptr, cutlass.Int32, align=4, dynamic_layout=True
        )
        if runtime.sink is None:
            merge = SparseMLASplitDecodeMergeKernel(splits)
            merge_args = (partial_c, partial_lse_c, control_c, output_c)
        else:
            merge = SparseMLASplitDecodeSinkMergeKernel(splits)
            merge_args = (partial_c, partial_lse_c, control_c, sink_c, output_c)
        launches.append(("merge", merge, merge_args))
        if caps.return_lse:
            launches.append(
                (
                    "lse",
                    SplitLse(splits, caps.lse_scale == "natural"),
                    (partial_lse_c, lse_c),
                )
            )
    else:
        for blocks, valid, offset in segments:
            kernel = UnifiedPrefillMGKernel(
                traits,
                make_smem_layout_mg(traits, 1),
                caps.page_size,
                (caps.max_width + 63) // 64,
                replicate_h=blocks,
                num_heads=heads,
                q_stride=q.stride(),
                indices_stride0=indices.stride(0),
                output_stride=runtime.output.stride(),
                out_lse_stride=runtime.lse.stride(),
                has_sink=runtime.sink is not None,
                topk=caps.max_width,
                head_offset=offset * 16,
                valid_hpb=valid,
                block_scaled_mma=False,
            )
            args = (
                q_c,
                kv_c,
                index_c,
                length_c,
                sink_c,
                output_c,
                lse_c,
                scale,
                latent,
                stride,
                count,
            )
            launches.append((f"prefill_h{valid}_o{offset}", kernel, args))
        if caps.return_lse and caps.lse_scale == "natural":
            launches.append(("lse", ConvertLse(), (lse_c,)))
    return launches


def prepare(runtime):
    caps = runtime.plan.caps
    config = runtime.plan.policy_resolution.config
    bound = runtime.binding
    key = (
        caps,
        config,
        tuple(bound.q.stride()),
        tuple(bound.selected_indices.stride()),
        tuple(runtime.output.stride()),
        tuple(bound.scratch.tmp_output.stride()) if caps.mode == "decode" else (),
    )
    launches = launch_specs(runtime)
    compiled = _CACHE.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm sparse MLA before CUDA graph capture")
        raise_if_kernel_resolution_frozen("sparse MLA warp compilation", cache_key=key)
        compiled = tuple(
            b12x_compile(
                kernel,
                *args,
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    f"attention.sparse_mla.warp.{name}", 1, key
                ),
            )
            for name, kernel, args in launches
        )
        _CACHE[key] = compiled
    runtime.prepared = tuple(
        (kernel, args) for kernel, (_, _, args) in zip(compiled, launches, strict=True)
    )


def _run_bound(runtime):
    with torch.cuda.device(runtime.output.device):
        if runtime.output.shape[0] == 0:
            return (
                (runtime.output, runtime.lse)
                if runtime.plan.caps.return_lse
                else runtime.output
            )
        if runtime.prepared is None:
            prepare(runtime)
        stream = current_cuda_stream()
        for kernel, args in runtime.prepared:
            kernel(*args, stream)
    return (
        (runtime.output, runtime.lse)
        if runtime.plan.caps.return_lse
        else runtime.output
    )


@torch.library.custom_op("b12x::sparse_mla_warp", mutates_args=("scratch",))
def _run_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
    scratch: torch.Tensor,
    sink: torch.Tensor | None,
    plan_key: str,
) -> None:
    # The registry contains static plans only. Reconstructing views consumes
    # the already-resolved config and allocates no CUDA storage. The custom op
    # keeps CuTe launch objects opaque to torch.compile's serving graph.
    plan = _PLANS[plan_key]
    binding = plan.bind(
        scratch=scratch,
        q=q,
        kv_cache=kv,
        selected_indices=selected,
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=active,
    )
    _run_bound(bind(plan, binding, sink))


@_run_op.register_fake
def _run_op_fake(q, kv, selected, lengths, active, scratch, sink, plan_key):
    return None


def run(runtime):
    bound = runtime.binding
    _run_op(
        bound.q,
        bound.kv_cache,
        bound.selected_indices,
        bound.cache_seqlens_int32,
        bound.nsa_cache_seqlens_int32,
        bound.scratch.shared_scratch,
        runtime.sink,
        runtime.plan_key,
    )
    return (
        (runtime.output, runtime.lse)
        if runtime.plan.caps.return_lse
        else runtime.output
    )


def clear_caches():
    _CACHE.clear()
