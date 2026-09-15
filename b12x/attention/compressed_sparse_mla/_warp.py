"""Planned ordinary-MMA execution over DeepSeek SWA and indexed caches.

Selection staging owns only metadata. Native compressed KV records flow into
shared memory in the unified CuTe attention kernels. Live counts, pool sizes,
page strides and softmax scale are runtime arguments, never compile keys.
"""

from dataclasses import replace
import math
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import make_ptr
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from .._shared.mla.kernel import UnifiedDecodeKernel, _cache_base_tensor, _to_cute
from .._shared.mla.prefill_mg import UnifiedPrefillMGKernel
from .._shared.mla.smem import make_smem_layout
from .._shared.mla.smem_mg import make_smem_layout_mg
from .._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    make_unified_traits,
)
from .._shared.mla.merge import (
    SparseMLASplitDecodeMergeKernel,
    SparseMLASplitDecodeSinkMergeKernel,
)
from ..sparse_mla._sm103 import SplitLse, ConvertLse



class PrepareSelections:
    """Pad a live selection into planned storage and validate physical slots."""

    def __init__(self, capacity, width, page_size):
        self.capacity, self.width, self.page_size = capacity, width, page_size

    @cute.jit
    def __call__(
        self,
        indices: cute.Pointer,
        lengths: cute.Pointer,
        table: cute.Pointer,
        output: cute.Pointer,
        out_lengths: cute.Pointer,
        zero_record: cute.Pointer,
        rows: c.Int32,
        width: c.Int32,
        input_stride: c.Int64,
        num_pages: c.Int64,
        mapped: c.Boolean,
        table_width: c.Int32,
        table_stride: c.Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            indices,
            lengths,
            table,
            output,
            out_lengths,
            zero_record,
            width,
            input_stride,
            num_pages,
            mapped,
            table_width,
            table_stride,
        ).launch(
            grid=(rows, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        indices,
        lengths,
        table,
        output,
        out_lengths,
        zero_record,
        width: c.Int32,
        input_stride: c.Int64,
        num_pages: c.Int64,
        mapped: c.Boolean,
        table_width: c.Int32,
        table_stride: c.Int64,
    ):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        row = c.Int64(row)
        col = c.Int64(thread)
        # The aligned split-control slot owns a 1-KiB region. Fixed-split
        # execution uses it as a benign source for masked empty-pool reads.
        zeros = cute.make_tensor(zero_record, cute.make_layout(256))
        if row == 0:
            zeros[thread] = c.Int32(0)
        x = cute.make_tensor(
            indices, cute.make_layout(c.Int64(self.capacity) * input_stride)
        )
        ln = cute.make_tensor(lengths, cute.make_layout(self.capacity))
        pt = cute.make_tensor(
            table,
            cute.make_layout(
                c.Int64(self.capacity) * table_stride + c.Int64(table_width)
            ),
        )
        dest = cute.make_tensor(
            output, cute.make_layout(c.Int64(self.capacity) * self.width)
        )
        dest_ln = cute.make_tensor(out_lengths, cute.make_layout(self.capacity))
        length = c.Int32(0)
        if width > 0:
            length = c.Int32(ln[row])
            length = c.max(c.Int32(0), c.min(length, width))
        bound = c.Int32(0)
        while col < self.width:
            slot = c.Int64(-1)
            if col < length:
                logical = c.Int64(x[row * input_stride + col])
                if logical >= 0:
                    slot = logical
                    if mapped:
                        page = logical // c.Int64(self.page_size)
                        slot = c.Int64(-1)
                        if page < table_width:
                            pid = c.Int64(pt[row * table_stride + page])
                            if (pid >= 0) & (pid < num_pages):
                                slot = pid * c.Int64(
                                    self.page_size
                                ) + logical % c.Int64(self.page_size)
                    if (
                        (slot < 0)
                        | (slot >= num_pages * c.Int64(self.page_size))
                        | (slot > 2147483647)
                    ):
                        slot = c.Int64(-1)
            dest[row * c.Int64(self.width) + col] = slot.to(c.Int32)
            if slot >= 0:
                bound = c.Int32(col + 1)
            col += c.Int64(256)
        for offset in (16, 8, 4, 2, 1):
            bound = c.max(bound, cute.arch.shuffle_sync_bfly(bound, offset=offset))
        shared = cutlass_utils.SmemAllocator().allocate_tensor(
            c.Int32, cute.make_layout(8), 16
        )
        lane = cute.arch.lane_idx()
        warp = thread // 32
        if lane == 0:
            shared[warp] = bound
        cute.arch.barrier()
        if warp == 0:
            bound = c.Int32(0)
            if lane < 8:
                bound = shared[lane]
            for offset in (16, 8, 4, 2, 1):
                bound = c.max(bound, cute.arch.shuffle_sync_bfly(bound, offset=offset))
            if lane == 0:
                dest_ln[row] = bound


def traits_for(plan):
    caps, config = plan.caps, plan.execution_config
    if caps.cache_format == "deepseek_v4":
        return make_unified_traits(
            ModelType.DSV4,
            ComputeMode.BF16 if caps.mode == "extend" else ComputeMode.FP8,
            ScaleFormat.UE8M0_BYTE,
            fp8_rope=False,
        )
    traits = make_unified_traits(
        ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False
    )
    if config.v41_compute_mode == "fp8":
        traits = replace(traits, fp8_internal=True)
        if caps.mode == "decode":
            traits = replace(
                traits,
                compute_mode=ComputeMode.FP8,
                q_nope_stride=528,
                kv_smem_stride=624,
            )
    return traits


def launch_specs(
    plan,
    scratch,
    q,
    swa,
    selected,
    lengths,
    indexed,
    indexed_selected,
    indexed_lengths,
    table,
    sink,
    output,
    sm_scale,
):
    caps, config = plan.caps, plan.execution_config
    rows = q.shape[0]
    padded_swa, padded_indexed = (
        scratch.staged_swa_indices,
        scratch.staged_indexed_indices,
    )
    lens_swa, lens_indexed = scratch.staged_swa_lengths, scratch.staged_indexed_lengths
    dummy = scratch.num_chunks_ptr
    specs = []

    def metadata_pointer(value):
        tensor = value if value is not None and value.numel() else dummy
        from torch._subclasses.fake_tensor import FakeTensor
        return make_ptr(
            c.Int32, 0 if isinstance(tensor, FakeTensor) else tensor.data_ptr(),
            cute.AddressSpace.gmem, assumed_align=4
        )

    for name, source, counts, pages, dest, dest_ln, page_size, mapping in (
        ("swa", selected, lengths, swa, padded_swa, lens_swa, caps.swa_page_size, None),
        (
            "indexed",
            indexed_selected,
            indexed_lengths,
            indexed,
            padded_indexed,
            lens_indexed,
            caps.indexed_page_size,
            table,
        ),
    ):
        width = source.shape[1] if source is not None else 0
        args = tuple(
            metadata_pointer(value)
            for value in (source, counts, mapping, dest, dest_ln, dummy)
        )
        args += (
            c.Int32(rows),
            c.Int32(width),
            c.Int64(width),
            c.Int64(pages.shape[0] if pages is not None else 0),
            c.Boolean(mapping is not None),
            c.Int32(mapping.shape[1] if mapping is not None else 0),
            c.Int64(mapping.stride(0) if mapping is not None else 0),
        )
        specs.append(
            (
                "prepare_" + name,
                PrepareSelections(caps.max_q_rows, dest.shape[1], page_size),
                args,
            )
        )

    def tc(value, dtype, align=16):
        return _to_cute(value, dtype, align=align, dynamic_layout=True)

    q_c, out_c = tc(q, c.BFloat16), tc(output, c.BFloat16)
    empty_cache = dummy.view(torch.uint8).narrow(0, 0, 1)
    swa_c = _to_cute(
        _cache_base_tensor(swa).narrow(0, 0, 1) if swa.numel() else empty_cache, c.Uint8
    )
    index_c = (
        _to_cute(_cache_base_tensor(indexed).narrow(0, 0, 1), c.Uint8)
        if indexed is not None and indexed.numel()
        else _to_cute(empty_cache, c.Uint8)
    )
    si, ii = tc(padded_swa.narrow(0, 0, rows), c.Int32, 4), tc(padded_indexed.narrow(0, 0, rows), c.Int32, 4)
    sl, il = tc(lens_swa.narrow(0, 0, rows), c.Int32, 4), tc(lens_indexed.narrow(0, 0, rows), c.Int32, 4)
    sink_c = tc(sink if sink is not None else scratch.sm_scale_tensor, c.Float32, 4)
    lse_c = tc(scratch.final_lse.narrow(0, 0, rows), c.Float32, 4)
    stride_swa = c.Int64(swa.stride(0) * swa.element_size())
    stride_indexed = c.Int64(
        indexed.stride(0) * indexed.element_size() if indexed is not None else 0
    )
    scale, latent, count = (
        c.Float32(sm_scale * math.log2(math.e)),
        c.Float32(1),
        c.Int32(rows),
    )
    traits = traits_for(plan)
    has_extra = caps.indexed_width > 0
    main_tiles = padded_swa.shape[1] // 64
    total_tiles = main_tiles + (padded_indexed.shape[1] // 64 if has_extra else 0)
    heads = caps.num_q_heads
    segments = [(heads // 16, 16, 0)] if heads >= 16 else []
    if heads % 16:
        segments.append((1, heads % 16, heads // 16))
    if caps.mode == "decode":
        splits = config.max_chunks_per_row
        partial, partial_lse = scratch.tmp_output.narrow(0, 0, rows), scratch.tmp_lse.narrow(0, 0, rows)
        pc, pl = tc(partial, c.BFloat16), tc(partial_lse, c.Float32, 4)
        for blocks, valid, offset in segments:
            kernel = UnifiedDecodeKernel(
                traits,
                make_smem_layout(traits),
                caps.swa_page_size,
                (total_tiles + splits - 1) // splits,
                h_blocks=blocks,
                num_splits=splits,
                num_heads=heads,
                q_head_dim=512,
                topk=padded_swa.shape[1],
                extra_topk=padded_indexed.shape[1] if has_extra else 0,
                q_stride=(heads * 512, 512, 1),
                swa_indices_stride0=padded_swa.shape[1],
                extra_indices_stride0=padded_indexed.shape[1],
                mid_out_stride=scratch.tmp_output.stride(),
                mid_lse_stride=scratch.tmp_lse.stride(),
                has_extra=has_extra,
                pbs_extra=caps.indexed_page_size,
                valid_hpb=valid,
                head_block_offset=offset,
                per_token_len=True,
                vector_q=True,
                block_scaled_mma=False,
            )
            args = (q_c, swa_c, si, pc, pl, scale, latent, sl, stride_swa)
            if has_extra:
                args += (index_c, ii, il, c.Int32(main_tiles), stride_indexed)
            specs.append(
                (
                    f"decode_h{valid}_o{offset}",
                    kernel.call_extra_pertok if has_extra else kernel.call_pertok,
                    args + (count,),
                )
            )
        control = tc(dummy, c.Int32, 4)
        specs.append(
            ("merge", SparseMLASplitDecodeMergeKernel(splits), (pc, pl, control, out_c))
        )
        specs.append(
            (
                "merge_sink",
                SparseMLASplitDecodeSinkMergeKernel(splits),
                (pc, pl, control, sink_c, out_c),
            )
        )
        for natural in (False, True):
            for has_sink in (False, True):
                lse_kernel = SplitLse(splits, natural, has_sink)
                specs.append(
                    (
                        f"lse_{int(natural)}_sink{int(has_sink)}",
                        lse_kernel.call_sink if has_sink else lse_kernel,
                        (pl, lse_c, sink_c) if has_sink else (pl, lse_c),
                    )
                )
    else:
        for blocks, valid, offset in segments:
            for has_sink in (False, True):
                kernel = UnifiedPrefillMGKernel(
                    traits,
                    make_smem_layout_mg(traits, 1),
                    caps.swa_page_size,
                    total_tiles,
                    replicate_h=blocks,
                    num_heads=heads,
                    q_stride=(heads * 512, 512, 1),
                    indices_stride0=padded_swa.shape[1],
                    output_stride=(heads * 512, 512, 1),
                    out_lse_stride=(heads, 1),
                    has_sink=has_sink,
                    topk=padded_swa.shape[1],
                    has_extra=has_extra,
                    pbs_extra=caps.indexed_page_size,
                    num_main_tiles=main_tiles,
                    extra_topk=padded_indexed.shape[1],
                    extra_indices_stride0=padded_indexed.shape[1],
                    head_offset=offset * 16,
                    valid_hpb=valid,
                    block_scaled_mma=False,
                )
                args = (
                    q_c,
                    swa_c,
                    si,
                    sl,
                    sink_c,
                    out_c,
                    lse_c,
                    scale,
                    latent,
                    stride_swa,
                )
                if has_extra:
                    args += (index_c, ii, il, stride_indexed)
                specs.append(
                    (
                        f"prefill_h{valid}_o{offset}_sink{int(has_sink)}",
                        kernel.call_dual if has_extra else kernel,
                        args + (count,),
                    )
                )
        specs.append(("lse_natural", ConvertLse(), (lse_c,)))
    return specs


def compile_launches(key, specs, *, offline=False, artifact_dir=None):
    compiled = {}
    for name, kernel, args in specs:
        options = ""
        if artifact_dir is not None:
            directory = Path(artifact_dir) / name
            directory.mkdir(exist_ok=False, parents=True)
            options = f" --keep-ptx --keep-cubin --dump-dir={directory}"
        if offline:
            fn = cute.compile(
                kernel,
                *args,
                cuda.CUstream(0),
                options="--gpu-arch=sm_103a" + options,
                no_jit_engine=True,
            )
        else:
            fn = b12x_compile(
                kernel,
                *args,
                current_cuda_stream(),
                options=options,
                compile_spec=KernelCompileSpec.from_key(
                    "attention.compressed_sparse_mla.warp." + name, 1, key
                ),
            )
        if artifact_dir is not None:
            (directory / (name + ".mlir")).write_text(str(fn.ir_module))
        compiled[name] = fn
    return compiled


def execute(
    plan,
    scratch,
    q,
    swa,
    selected,
    lengths,
    indexed,
    indexed_selected,
    indexed_lengths,
    table,
    sink,
    output,
    sm_scale,
    natural,
    *, compiled,
):
    if q.shape[0] == 0:
        return
    specs = launch_specs(
        plan,
        scratch,
        q,
        swa,
        selected,
        lengths,
        indexed,
        indexed_selected,
        indexed_lengths,
        table,
        sink,
        output,
        sm_scale,
    )
    stream = current_cuda_stream()
    for name, _, args in specs:
        if name.startswith("merge") and (name == "merge_sink") != (sink is not None):
            continue
        if name.startswith("prefill") and name.endswith("sink1") != (sink is not None):
            continue
        if name.startswith(("lse_0", "lse_1")) and name != (
            f"lse_{int(natural)}_sink{int(sink is not None)}"
        ):
            continue
        if name == "lse_natural" and not natural:
            continue
        from b12x._lib.compiler import run_compiled
        run_compiled(compiled[name], (*args, stream))



@torch.library.custom_op("b12x::compressed_mla_warp", mutates_args=("storage", "out"))
def _run_op(
    storage: torch.Tensor, q: torch.Tensor, swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor, indexed_indices: torch.Tensor | None,
    indexed_lengths: torch.Tensor | None, indexed_page_table: torch.Tensor | None,
    swa_cache: torch.Tensor, indexed_cache: torch.Tensor | None,
    sink: torch.Tensor | None, out: torch.Tensor | None, scale: float,
    swa_page: int, indexed_page: int | None, return_lse: bool, lse_scale: str,
    cache_format: str, plan_handle: int,
) -> None:
    from b12x.preparation import plan_from_handle, require_prepared

    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "attention.compressed_sparse_mla", q.device)
    binding = state.bind(
        plan, scratch=storage, q=q, swa_indices=swa_indices,
        swa_lengths=swa_lengths, indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths, indexed_page_table=indexed_page_table,
    )
    state.run(
        binding, swa_k_cache=swa_cache, indexed_k_cache=indexed_cache,
        attn_sink=sink, out=out, sm_scale=scale, swa_page_size=swa_page,
        indexed_page_size=indexed_page, return_lse=return_lse,
        lse_scale=lse_scale, cache_format=cache_format,
    )


@_run_op.register_fake
def _run_fake(*args, **kwargs):
    return None


def run_opaque(
    binding, *, swa_k_cache, indexed_k_cache=None, attn_sink=None, out=None,
    sm_scale=512**-0.5, swa_page_size=None, indexed_page_size=None,
    return_lse=False, lse_scale="base2", cache_format=None,
):
    """Expose scratch/output mutation while keeping native launch metadata opaque."""
    query = binding.plan.query
    torch.ops.b12x.compressed_mla_warp(
        binding.scratch.shared_scratch, binding.q, binding.swa_indices,
        binding.swa_lengths, binding.indexed_indices, binding.indexed_lengths,
        binding.indexed_page_table, swa_k_cache, indexed_k_cache, attn_sink, out,
        float(sm_scale), query.swa_page_size if swa_page_size is None else swa_page_size,
        query.indexed_page_size if indexed_page_size is None else indexed_page_size,
        return_lse, lse_scale, query.cache_format if cache_format is None else cache_format,
        binding.plan.handle,
    )
    rows = binding.q.shape[0]
    output = binding.scratch.output_buffer[:rows] if out is None else out
    return (output, binding.scratch.final_lse[:rows]) if return_lse else output
