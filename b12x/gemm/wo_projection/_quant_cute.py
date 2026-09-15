from __future__ import annotations

import functools
from b12x._lib.compiler import run_compiled
from b12x._lib.compile_plan import attach_programs
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    bfloat2_to_float2_scaled,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_nc_u32,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from b12x._lib.utils import cuda_stream_to_int, current_cuda_stream, make_ptr

_THREADS = 256
_GRID_CTAS_PER_SM = 4


@dsl_user_op
def _invalid_position(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


class _WOQuantCuTeLaunch:
    """Tiled MXFP8 quantizer for the WO activation layouts.

    Live rows and all pool offsets are runtime arguments. Every launch writes
    unity into physical scale padding, so scratch may be reused or poisoned
    without a separate initialization kernel.

    Mirrors the dense `_MXFP8RowsQuantLaunch` subgroup scheme (four 8-lane
    subgroups per warp, one 32-value scale block each, four adjacent values
    per lane) over two WO-specific addressings:

    - grouped (`mode="grouped"`): BF16 source rows `[m, groups * group_width]`
      (flat contiguous attention output), optionally applying inverse RoPE to
      the trailing `rope_dim` of every `head_dim` block; writes the grouped
      dense-GEMM operand (values physical `[g, m, group_width]`, per-group
      scale_rows / swizzled scale_mma).
    - group-major (`mode="group_major"`): BF16 source physical
      `[g, m, rank]` (the WO-A output) regathered as flat group-major rows
      `[m, groups * rank]`; writes the singleton-group dense-GEMM operand.
    """

    def __init__(
        self,
        mode: str,
        total_k: int,
        span: int,
        source_type: type[cutlass.Numeric],
        inv_rope: bool,
        head_dim: int,
        nope_dim: int,
        rope_dim: int,
        positions_type: type[cutlass.Numeric],
        cos_sin_type: type[cutlass.Numeric],
        threads: int,
    ) -> None:
        if mode not in ("grouped", "group_major"):
            raise ValueError(f"unsupported WO quant mode {mode!r}")
        if (total_k <= 0 or total_k >= 2**31 or total_k % 128 or span <= 0
                or total_k % span or span % (128 if mode == "grouped" else 4)):
            raise ValueError("WO quantization requires K divisible by 128 and aligned group spans")
        if inv_rope and (mode != "grouped" or head_dim != nope_dim + rope_dim
                or nope_dim < 0 or rope_dim <= 0 or nope_dim % 4 or rope_dim % 4
                or span % head_dim):
            raise ValueError("WO inverse RoPE requires complete heads and dimensions divisible by four")
        if threads <= 0 or threads > 1024 or threads % 32:
            raise ValueError("WO quantization threads must be a multiple of 32 in [32,1024]")
        self._mode = mode
        self._total_k = int(total_k)
        # Per-group width: group_width for grouped, rank for group-major.
        self._span = int(span)
        self._groups = self._total_k // self._span
        self._groups_k = self._total_k // 32
        self._span_groups_k = self._span // 32
        self._source_type = source_type
        self._inv_rope = bool(inv_rope)
        self._head_dim = int(head_dim)
        self._nope_dim = int(nope_dim)
        self._rope_dim = int(rope_dim)
        self._positions_type = positions_type
        self._cos_sin_type = cos_sin_type
        self._threads = int(threads)
        self._warps_per_cta = self._threads // 32

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        m: Int32,
        source_row_stride: Int64,
        cos_sin_len: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        source = cute.make_tensor(
            source_ptr,
            cute.make_layout((Int64(m) * source_row_stride,)),
        )
        positions = cute.make_tensor(positions_ptr, cute.make_layout((m,)))
        cos_sin = cute.make_tensor(cos_sin_ptr, cute.make_layout((cos_sin_len,)))
        values_u32 = cute.make_tensor(
            values_ptr,
            cute.make_layout((Int64(m) * (self._total_k // 4),)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_layout((Int64(m) * self._groups_k,)),
        )
        scale_mma = cute.make_tensor(
            scale_mma_ptr,
            cute.make_layout((((Int64(m) + 127) // 128) * (self._total_k // 128) * 512,)),
        )
        self.kernel(
            source, positions, cos_sin, values_u32, scale_rows, scale_mma, m, source_row_stride
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        positions: cute.Tensor,
        cos_sin: cute.Tensor,
        values_u32: cute.Tensor,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        m: Int32,
        source_row_stride: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        # Four 8-lane subgroups per warp each quantize one 32-value block;
        # every lane owns four adjacent values. total_k % 128 == 0, so all
        # four blocks of a task exist and the warp never diverges at the
        # butterfly reductions.
        warp = Int32(tidx) // Int32(32)
        lane = Int32(tidx) % Int32(32)
        subgroup = lane // Int32(8)
        lane8 = lane % Int32(8)
        group_tiles = Int32(self._groups_k // 4)
        task = Int64(bidx) * self._warps_per_cta + Int64(warp)
        total_tasks = Int64(m) * Int64(group_tiles)
        while task < total_tasks:
            row = task // Int64(group_tiles)
            block = Int32(task % Int64(group_tiles)) * Int32(4) + subgroup
            k0 = block * Int32(32) + lane8 * Int32(4)
            g = k0 // Int32(self._span)
            inner0 = k0 - g * Int32(self._span)

            if cutlass.const_expr(self._mode == "group_major"):
                # Source physical [g, m, span]: regather group-major flat k.
                src0 = Int64(g) * (Int64(m) * self._span) + row * self._span + Int64(inner0)
            else:
                src0 = row * source_row_stride + Int64(k0)

            # Pure SSA scalars (no rmem array): divergent updates under the
            # rope branch then merge in registers instead of spilling.
            v0 = cutlass.Float32(source[src0 + Int32(0)])
            v1 = cutlass.Float32(source[src0 + Int32(1)])
            v2 = cutlass.Float32(source[src0 + Int32(2)])
            v3 = cutlass.Float32(source[src0 + Int32(3)])

            if cutlass.const_expr(self._inv_rope):
                # Branchless: every lane loads (clamped) cos/sin and computes
                # the rotation; a full-width bit mask selects rotated vs raw
                # values, so no divergent path lengthens the task loop and
                # -0.0 payloads survive untouched on nope lanes.
                head_d0 = k0 % Int32(self._head_dim)
                is_rope = head_d0 >= Int32(self._nope_dim)
                pos = Int64(positions[row])
                if pos < Int64(0) or pos >= cute.size(cos_sin) // self._rope_dim:
                    _invalid_position()
                    pos = Int64(0)
                half_rope = Int32(self._rope_dim // 2)
                cs_base = pos * self._rope_dim
                rl_half0 = (head_d0 - Int32(self._nope_dim)) // Int32(2)
                if rl_half0 < Int32(0):
                    rl_half0 = Int32(0)
                if cutlass.const_expr(self._cos_sin_type == cutlass.BFloat16):
                    # rl_half0 is even (head_d0 and nope_dim are multiples of
                    # 4), so each cos/sin pair sits on one aligned 4B word.
                    cos_w = ld_global_nc_u32(
                        get_ptr_as_int64(cos_sin, cs_base + rl_half0)
                    )
                    sin_w = ld_global_nc_u32(
                        get_ptr_as_int64(cos_sin, cs_base + rl_half0 + half_rope)
                    )
                    cos0, cos1 = bfloat2_to_float2_scaled(cos_w, cutlass.Float32(1.0))
                    sin0, sin1 = bfloat2_to_float2_scaled(sin_w, cutlass.Float32(1.0))
                else:
                    cos0 = cutlass.Float32(cos_sin[cs_base + rl_half0])
                    sin0 = cutlass.Float32(cos_sin[cs_base + rl_half0 + half_rope])
                    cos1 = cutlass.Float32(cos_sin[cs_base + rl_half0 + Int32(1)])
                    sin1 = cutlass.Float32(
                        cos_sin[cs_base + rl_half0 + Int32(1) + half_rope]
                    )
                r0 = v0 * cos0 + v1 * sin0
                r1 = v1 * cos0 - v0 * sin0
                r2 = v2 * cos1 + v3 * sin1
                r3 = v3 * cos1 - v2 * sin1
                mask = Uint32(0) - Uint32(is_rope)
                keep = mask ^ Uint32(0xFFFFFFFF)
                v0 = cutlass.Uint32(
                    (r0.bitcast(Uint32) & mask) | (v0.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v1 = cutlass.Uint32(
                    (r1.bitcast(Uint32) & mask) | (v1.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v2 = cutlass.Uint32(
                    (r2.bitcast(Uint32) & mask) | (v2.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v3 = cutlass.Uint32(
                    (r3.bitcast(Uint32) & mask) | (v3.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)

            max_abs = fabs_f32(v0)
            max_abs = fmax_f32(max_abs, fabs_f32(v1))
            max_abs = fmax_f32(max_abs, fabs_f32(v2))
            max_abs = fmax_f32(max_abs, fabs_f32(v3))
            for shift in cutlass.range_constexpr(3):
                max_abs = fmax_f32(
                    max_abs,
                    cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                )

            _, scale_byte = pow2_ceil_ue8m0(
                max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
            )
            if max_abs == cutlass.Float32(0.0):
                scale_byte = Uint32(127)
            inv_scale = ue8m0_to_output_scale(scale_byte)
            payload = cvt_f32x4_to_e4m3x4(
                v0 * inv_scale,
                v1 * inv_scale,
                v2 * inv_scale,
                v3 * inv_scale,
            )

            if cutlass.const_expr(self._mode == "grouped"):
                # Values physical [g, m, span]: transpose grouped rows.
                word = (
                    Int64(g) * (Int64(m) * (self._span // 4))
                    + row * (self._span // 4)
                    + Int64(inner0 // Int32(4))
                )
            else:
                word = row * (self._total_k // 4) + Int64(k0 // Int32(4))
            values_u32[word] = payload

            if lane8 == Int32(0):
                if cutlass.const_expr(self._mode == "grouped"):
                    chunk = inner0 // Int32(32)
                    span_gk = Int32(self._span_groups_k)
                    scale_rows[Int64(g) * (Int64(m) * Int64(span_gk)) + row * Int64(span_gk) + Int64(chunk)] = Uint8(
                        scale_byte
                    )
                    span_tiles_k = Int32((self._span_groups_k + 3) // 4)
                    m_tiles = (Int64(m) + 127) // 128
                    self._store_scale_mma(
                        scale_mma,
                        Int64(g) * (m_tiles * Int64(span_tiles_k) * 512),
                        span_tiles_k,
                        row,
                        chunk,
                        scale_byte,
                    )
                else:
                    chunk = block
                    scale_rows[row * self._groups_k + Int64(chunk)] = Uint8(scale_byte)
                    self._store_scale_mma(
                        scale_mma,
                        Int64(0),
                        Int32((self._groups_k + 3) // 4),
                        row,
                        chunk,
                        scale_byte,
                    )
            task += Int64(gdim) * self._warps_per_cta

        # Only scale padding is materialized; invalid rows never read source
        # values or positions. Logical and padded scale writes are disjoint.
        padded_m = ((Int64(m) + 127) // 128) * 128
        pad_rows = padded_m - Int64(m)
        index = Int64(bidx) * self._threads + Int64(tidx)
        while index < pad_rows * self._groups_k:
            row = Int64(m) + index // self._groups_k
            chunk = Int32(index % self._groups_k)
            if cutlass.const_expr(self._mode == "grouped"):
                group = chunk // self._span_groups_k
                inner_chunk = chunk % self._span_groups_k
                tiles_k = Int32(self._span_groups_k // 4)
                base = Int64(group) * (padded_m // 128) * Int64(tiles_k) * 512
            else:
                inner_chunk = chunk
                tiles_k = Int32(self._groups_k // 4)
                base = Int64(0)
            self._store_scale_mma(scale_mma, base, tiles_k, row, inner_chunk, Uint32(127))
            index += Int64(gdim) * self._threads

    @cute.jit
    def _store_scale_mma(
        self,
        scale_mma: cute.Tensor,
        base: Int64,
        tiles_k: Int32,
        row: Int64,
        chunk: Int32,
        scale_byte: Uint32,
    ) -> None:
        row32 = row % Int32(32)
        row4 = (row // Int32(32)) % Int32(4)
        tile_m = row // Int32(128)
        k4 = chunk % Int32(4)
        tile_k = chunk // Int32(4)
        offset = (
            base
            + row32 * Int32(16)
            + row4 * Int32(4)
            + tile_m * (Int64(tiles_k) * 512)
            + k4
            + Int64(tile_k) * 512
        )
        scale_mma[offset] = Uint8(scale_byte)


def _cutlass_source_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(f"WO CuTe quantizer requires BF16/FP16 input, got {dtype}")


def _cutlass_positions_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.int64:
        return cutlass.Int64
    if dtype == torch.int32:
        return cutlass.Int32
    raise TypeError(f"WO CuTe quantizer positions must be int32/int64, got {dtype}")


def _cutlass_cos_sin_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"WO CuTe quantizer cos/sin must be bf16/fp32, got {dtype}")


def _get_compiled_wo_quant(
    mode: str,
    total_k: int,
    span: int,
    source_dtype: torch.dtype,
    inv_rope: bool,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    positions_dtype: torch.dtype,
    cos_sin_dtype: torch.dtype,
    device_ordinal: int | None = None,
    architecture: str | None = None,
) -> Callable:
    if device_ordinal is None:
        device_ordinal = torch.cuda.current_device()
    if architecture is None:
        props = torch.cuda.get_device_properties(device_ordinal)
        architecture = f"sm_{props.major}{props.minor}a"
    return _compile_wo_quant(mode, total_k, span, source_dtype, inv_rope,
                             head_dim, nope_dim, rope_dim, positions_dtype,
                             cos_sin_dtype, device_ordinal, architecture)


from b12x._lib.program_cache import program_cache


@program_cache
def _compile_wo_quant(
    mode: str,
    total_k: int,
    span: int,
    source_dtype: torch.dtype,
    inv_rope: bool,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    positions_dtype: torch.dtype,
    cos_sin_dtype: torch.dtype,
    device_ordinal: int,
    architecture: str,
) -> Callable:
    source_type = _cutlass_source_type(source_dtype)
    positions_type = _cutlass_positions_type(positions_dtype)
    cos_sin_type = _cutlass_cos_sin_type(cos_sin_dtype)
    launch = _WOQuantCuTeLaunch(
        mode,
        total_k,
        span,
        source_type,
        inv_rope,
        head_dim,
        nope_dim,
        rope_dim,
        positions_type,
        cos_sin_type,
        _THREADS,
    )
    cache_key = (
        mode,
        int(total_k),
        int(span),
        str(source_dtype),
        bool(inv_rope),
        int(head_dim),
        int(nope_dim),
        int(rope_dim),
        str(positions_dtype),
        str(cos_sin_dtype),
        _THREADS,
        device_ordinal,
        architecture,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=cache_key,
    )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("WO quantization must be prewarmed before graph capture")
    raw = b12x_compile(
        launch,
        make_ptr(source_type, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(positions_type, 16, cute.AddressSpace.gmem, assumed_align=positions_type.width // 8),
        make_ptr(cos_sin_type, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        Int64(1),
        Int64(1),
        Int32(1),
        current_cuda_stream(),
        options=f"--gpu-arch={architecture}",
        compile_spec=KernelCompileSpec.from_key(
            "gemm.wo_quant_cute",
            4,
            cache_key,
        ),
    )

    sm_count = torch.cuda.get_device_properties(device_ordinal).multi_processor_count

    def launch_tensors(
        source: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        values: torch.Tensor,
        scale_rows: torch.Tensor,
        scale_mma: torch.Tensor,
        m: int,
        stream: object = None,
    ) -> None:
        groups_per_warp_tile = 4
        total_tasks = m * (total_k // 32 // groups_per_warp_tile)
        warps_per_cta = _THREADS // 32
        natural_grid = max(1, (total_tasks + warps_per_cta - 1) // warps_per_cta)
        grid_x = min(natural_grid, sm_count * _GRID_CTAS_PER_SM)
        run_compiled(raw, (
            make_ptr(
                source_type,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                positions_type,
                positions.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=positions_type.width // 8,
            ),
            make_ptr(
                cos_sin_type,
                cos_sin.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint32,
                values.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scale_rows.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scale_mma.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            Int32(m),
            Int64(_grouped_source_stride(source, m, total_k) if mode == "grouped" else total_k),
            Int64(cos_sin.numel()),
            Int32(grid_x),
            current_cuda_stream() if stream is None else cuda.CUstream(cuda_stream_to_int(stream)),
        ))

    return attach_programs(launch_tensors, raw)


_get_compiled_wo_quant.cache_clear = _compile_wo_quant.cache_clear
_get_compiled_wo_quant.cache_info = _compile_wo_quant.cache_info


def quantize_wo_grouped_rows_cute(
    source_flat: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    m: int,
    groups: int,
    group_width: int,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    head_dim: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
    stream: object = None,
    compiled=None,
) -> None:
    """Quantize flat `[m, groups*group_width]` rows into the grouped WO-A
    MXFP8 operand, optionally applying inverse RoPE first."""

    _validate_storage(source_flat, values, scale_rows, scale_mma,
                      m=m, total_k=groups * group_width, span=group_width, mode="grouped")
    inv_rope = positions is not None
    if inv_rope:
        if cos_sin_cache is None:
            raise ValueError("WO inverse RoPE requires a cos/sin cache")
        _validate_rope(source_flat, positions, cos_sin_cache, m, head_dim, nope_dim, rope_dim)
        for read in (positions, cos_sin_cache):
            for write in (values, scale_rows, scale_mma):
                _check_disjoint((read, write))
        pos_t: torch.Tensor = positions
        cs_t: torch.Tensor = cos_sin_cache
    else:
        if cos_sin_cache is not None:
            raise ValueError("WO inverse RoPE requires positions")
        pos_t = source_flat
        cs_t = source_flat
    if m:
        with torch.cuda.device(source_flat.device):
            fn = compiled
            if fn is None:
                fn = _get_compiled_wo_quant(
                "grouped", groups * group_width, group_width, source_flat.dtype,
                inv_rope, head_dim if inv_rope else 0, nope_dim if inv_rope else 0,
                rope_dim if inv_rope else 0,
                pos_t.dtype if inv_rope else torch.int64,
                cs_t.dtype if inv_rope else torch.bfloat16,
                source_flat.device.index,
            )
            fn(source_flat, pos_t, cs_t, values, scale_rows, scale_mma, m, stream)


def quantize_wo_group_major_rows_cute(
    source_gmr: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    m: int,
    groups: int,
    rank: int,
    stream: object = None,
    compiled=None,
) -> None:
    """Quantize the WO-A output (physical `[groups, m, rank]`) as group-major
    flat `[m, groups*rank]` MXFP8 rows for WO-B."""

    _validate_storage(source_gmr, values, scale_rows, scale_mma,
                      m=m, total_k=groups * rank, span=rank, mode="group_major")
    if m:
        with torch.cuda.device(source_gmr.device):
            fn = compiled
            if fn is None:
                fn = _get_compiled_wo_quant(
                "group_major", groups * rank, rank, source_gmr.dtype, False,
                0, 0, 0, torch.int64, torch.bfloat16, source_gmr.device.index,
            )
            fn(source_gmr, source_gmr, source_gmr, values, scale_rows, scale_mma, m, stream)


def _grouped_source_stride(source, m, total_k):
    if source.is_contiguous():
        return total_k
    if (source.ndim in (2, 3) and source.shape[0] == m
            and source.stride(-1) == 1 and source.stride(0) >= total_k
            and (source.ndim == 2 or source.stride(1) == source.shape[2])):
        return source.stride(0)
    raise ValueError("WO input requires contiguous columns and nonoverlapping rows")


def _validate_storage(source, values, scale_rows, scale_mma, *, m, total_k, span, mode):
    if (m < 0 or m >= 2**31 or total_k <= 0 or total_k >= 2**31
            or total_k % 128 or span <= 0 or total_k % span
            or span % (128 if mode == "grouped" else 4)):
        raise ValueError("WO quantization requires valid rows, K, and group spans")
    groups = total_k // span
    if source.dtype not in (torch.bfloat16, torch.float16) or source.numel() != m * total_k:
        raise ValueError("WO input must contain complete BF16/FP16 rows")
    if mode == "grouped":
        _grouped_source_stride(source, m, total_k)
        source_layout = True
        physical_values = values.permute(2, 0, 1) if values.ndim == 3 else values
        value_groups, value_k = groups, span
    else:
        source_layout = (source.ndim == 3 and tuple(source.shape) == (m, span, groups)
                         and source.permute(2, 0, 1).is_contiguous())
        physical_values = values
        value_groups, value_k = 1, total_k
    if not source_layout or not physical_values.is_contiguous():
        raise ValueError("WO quantization requires contiguous physical value storage")
    tiles_m, tiles_k = (m + 127) // 128, value_k // 128
    expected_scales = (32, 4, tiles_m, 4, tiles_k, value_groups)
    if (tuple(scale_mma.shape) != expected_scales
            or not scale_mma.permute(5, 2, 4, 0, 1, 3).is_contiguous()
            or not scale_rows.is_contiguous()):
        raise ValueError("WO quantization requires compact row scales and F8_128x4 MMA scales")
    for tensor, dtypes, count in (
        (values, (torch.float8_e4m3fn, torch.uint8), m * total_k),
        (scale_rows, (torch.float8_e8m0fnu, torch.uint8), m * (total_k // 32)),
        (scale_mma, (torch.float8_e8m0fnu, torch.uint8), tiles_m * (total_k // 128) * 512),
    ):
        if tensor.device != source.device or tensor.dtype not in dtypes or tensor.numel() != count:
            raise ValueError("WO quantizer outputs must match input device, byte dtype, and live capacity")
    tensors = (source, values, scale_rows, scale_mma)
    if any(t.numel() and t.data_ptr() % 16 for t in tensors):
        raise ValueError("WO quantizer values and scales require 16-byte alignment")
    _check_disjoint(tensors)


def _check_disjoint(tensors):
    from ._execution import _span
    spans = [_span(t) for t in tensors]
    for i, (lo, hi) in enumerate(spans):
        if any(lo < end and start < hi for start, end in spans[:i]):
            raise ValueError("WO input and output buffers must not overlap")


def _validate_rope(source, positions, cos_sin, m, head_dim, nope_dim, rope_dim):
    if (head_dim != nope_dim + rope_dim or nope_dim < 0 or rope_dim <= 0
            or nope_dim % 4 or rope_dim % 4):
        raise ValueError("WO inverse RoPE requires head dimensions divisible by four")
    if (positions.device != source.device or positions.dtype not in (torch.int32, torch.int64)
            or tuple(positions.shape) != (m,) or not positions.is_contiguous()
            or positions.numel() and positions.data_ptr() % positions.element_size()):
        raise ValueError("WO positions must be aligned contiguous Int32/Int64 rows on the input device")
    if (cos_sin.device != source.device or cos_sin.dtype not in (torch.bfloat16, torch.float32)
            or cos_sin.ndim != 2 or cos_sin.shape[1] != rope_dim or cos_sin.shape[0] <= 0
            or not cos_sin.is_contiguous() or cos_sin.data_ptr() % 4):
        raise ValueError("WO cos/sin cache must be contiguous BF16/FP32 [positions,rope_dim]")


__all__ = [
    "quantize_wo_group_major_rows_cute",
    "quantize_wo_grouped_rows_cute",
]
