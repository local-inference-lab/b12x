from __future__ import annotations

import functools
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Uint8, Uint32

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    max_abs_32,
    pow2_ceil_ue8m0,
    quantize_block_fp8_mx,
    ue8m0_to_output_scale,
)
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from b12x._lib.utils import current_cuda_stream, make_ptr


_THREADS = 256
_GRID_CTAS_PER_SM = 4
_WARP_SUBGROUP_WIDTH = 4

_SCALE_VEC_SIZE = 32
_SCALE_ROW_TILE = 128
_SCALE_K_TILE = 4
_KERNEL_ALIGN_BYTES = 16
# Guard the kernel's explicit Int32 offset arithmetic (scale_mma offset =
# row32*16 + row4*4 + tile_m*T*512 + k4 + tile_k*512, where T=ceil(G/4)).
# 2^31 - 1 is the max positive Int32; keep a safety margin for intermediate
# products before the final addition.
_INT32_MAX = 2 ** 31 - 1


class _MXFP8RowsQuantLaunch:
    def __init__(
        self,
        k: int,
        source_type: type[cutlass.Numeric],
        subgroup_width: int,
        threads: int,
        trellis_native_mma_order: bool,
    ) -> None:
        self._k = int(k)
        self._groups_k = self._k // 32
        self._source_type = source_type
        self._subgroup_width = int(subgroup_width)
        self._threads = int(threads)
        self._warps_per_cta = self._threads // 32
        self._trellis_native_mma_order = bool(trellis_native_mma_order)

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        m: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        source = cute.make_tensor(
            source_ptr,
            cute.make_ordered_layout((m, self._k), order=(1, 0)),
        )
        values_u32 = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout((m, self._k // 4), order=(1, 0)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((m, self._groups_k), order=(1, 0)),
        )
        scale_mma = cute.make_tensor(
            scale_mma_ptr,
            cute.make_layout((max(512, ((self._groups_k + 3) // 4) * 512),)),
        )
        self.kernel(source, values_u32, scale_rows, scale_mma, m).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        values_u32: cute.Tensor,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        m: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._subgroup_width == 4):
            # Eight 4-lane subgroups per warp each quantize one 32-value block.
            # Each lane owns eight adjacent values and emits two packed words.
            warp = Int32(tidx) // Int32(32)
            lane = Int32(tidx) % Int32(32)
            subgroup = lane // Int32(4)
            lane8 = lane % Int32(4)
            group_tiles = Int32((self._groups_k + 7) // 8)
            task = Int32(bidx) * Int32(self._warps_per_cta) + warp
            total_tasks = m * group_tiles
            while task < total_tasks:
                row = task // group_tiles
                group = (task % group_tiles) * Int32(8) + subgroup
                if group < Int32(self._groups_k):
                    values = cute.make_rmem_tensor((8,), cutlass.Float32)
                    k0 = group * Int32(32) + lane8 * Int32(8)
                    for elem in cutlass.range_constexpr(8):
                        values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])

                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 8):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
                    for shift in cutlass.range_constexpr(2):
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
                    word0 = group * Int32(8) + lane8 * Int32(2)
                    values_u32[row, word0] = cvt_f32x4_to_e4m3x4(
                        values[0] * inv_scale,
                        values[1] * inv_scale,
                        values[2] * inv_scale,
                        values[3] * inv_scale,
                    )
                    values_u32[row, word0 + Int32(1)] = cvt_f32x4_to_e4m3x4(
                        values[4] * inv_scale,
                        values[5] * inv_scale,
                        values[6] * inv_scale,
                        values[7] * inv_scale,
                    )

                    if lane8 == Int32(0):
                        self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                task += Int32(gdim) * Int32(self._warps_per_cta)
        elif cutlass.const_expr(self._subgroup_width == 8):
            # Four 8-lane subgroups per warp each quantize one 32-value block.
            # Every lane owns four adjacent values, giving coalesced 128-value
            # warp loads/stores and a cheap width-8 butterfly max reduction.
            warp = Int32(tidx) // Int32(32)
            lane = Int32(tidx) % Int32(32)
            subgroup = lane // Int32(8)
            lane4 = lane % Int32(8)
            group_tiles = Int32((self._groups_k + 3) // 4)
            task = Int32(bidx) * Int32(self._warps_per_cta) + warp
            total_tasks = m * group_tiles
            while task < total_tasks:
                row = task // group_tiles
                group = (task % group_tiles) * Int32(4) + subgroup
                if group < Int32(self._groups_k):
                    values = cute.make_rmem_tensor((4,), cutlass.Float32)
                    k0 = group * Int32(32)
                    if cutlass.const_expr(self._trellis_native_mma_order):
                        # Match the direct native-trellis B assignment.  Each
                        # output word contains one native EXL lane's four K
                        # values; the eight words are a permutation wholly
                        # within this K32 scale group.
                        r = Int32(0)
                        tile = Int32(0)
                        if lane4 < Int32(4):
                            r = ((lane4 & Int32(1)) << Int32(1)) | (
                                lane4 >> Int32(1)
                            )
                        else:
                            q = lane4 - Int32(4)
                            r = (
                                ((Int32(1) - (q & Int32(1))) << Int32(1))
                                | (q >> Int32(1))
                            )
                            tile = Int32(16)
                        base = k0 + tile + (r << Int32(1))
                        values[0] = cutlass.Float32(source[row, base])
                        values[1] = cutlass.Float32(source[row, base + Int32(1)])
                        values[2] = cutlass.Float32(source[row, base + Int32(8)])
                        values[3] = cutlass.Float32(source[row, base + Int32(9)])
                    else:
                        k0 += lane4 * Int32(4)
                        for elem in cutlass.range_constexpr(4):
                            values[elem] = cutlass.Float32(
                                source[row, k0 + Int32(elem)]
                            )

                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
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
                        values[0] * inv_scale,
                        values[1] * inv_scale,
                        values[2] * inv_scale,
                        values[3] * inv_scale,
                    )
                    values_u32[row, group * Int32(8) + lane4] = payload

                    if lane4 == Int32(0):
                        self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                task += Int32(gdim) * Int32(self._warps_per_cta)
        else:
            block = Int32(bidx) * Int32(self._threads) + Int32(tidx)
            total_blocks = m * Int32(self._groups_k)
            while block < total_blocks:
                row = block // Int32(self._groups_k)
                group = block % Int32(self._groups_k)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                k0 = group * Int32(32)
                for elem in cutlass.range_constexpr(32):
                    values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])

                max_abs = max_abs_32(values)
                payload, scale_byte = quantize_block_fp8_mx(values, max_abs)
                if max_abs == cutlass.Float32(0.0):
                    scale_byte = Uint32(127)

                word0 = group * Int32(8)
                for word in cutlass.range_constexpr(8):
                    values_u32[row, word0 + Int32(word)] = payload[word]
                self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                block += Int32(gdim) * Int32(self._threads)

    @cute.jit
    def _store_scale(
        self,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        row: Int32,
        group: Int32,
        scale_byte: Uint32,
    ) -> None:
        scale_u8 = Uint8(scale_byte)
        scale_rows[row, group] = scale_u8
        row32 = row % Int32(32)
        row4 = (row // Int32(32)) % Int32(4)
        tile_m = row // Int32(128)
        k4 = group % Int32(4)
        tile_k = group // Int32(4)
        scale_mma_offset = (
            row32 * Int32(16)
            + row4 * Int32(4)
            + tile_m * Int32(((self._groups_k + 3) // 4) * 512)
            + k4
            + tile_k * Int32(512)
        )
        scale_mma[scale_mma_offset] = scale_u8


@functools.cache
def _get_compiled_mxfp8_rows_quant(
    k: int,
    source_dtype: torch.dtype,
    subgroup_width: int,
    threads: int,
    value_order: str,
    device: torch.device,
) -> Callable:
    k = int(k)
    if device.type != "cuda":
        raise ValueError(f"MXFP8 CuTe quantizer requires a CUDA device, got {device}")
    device_index = (
        torch.cuda.current_device()
        if device.index is None
        else int(device.index)
    )
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"MXFP8 CuTe quantizer requires K divisible by 32, got {k}")
    if source_dtype == torch.bfloat16:
        source_type = cutlass.BFloat16
        source_dtype_name = "bf16"
    elif source_dtype == torch.float16:
        source_type = cutlass.Float16
        source_dtype_name = "fp16"
    else:
        raise TypeError(
            f"CuTe MXFP8 quantizer requires BF16 or FP16 input, got {source_dtype}"
        )
    if subgroup_width not in (0, 4, 8):
        raise ValueError(
            f"MXFP8 CuTe quantizer subgroup width must be 0, 4, or 8, got {subgroup_width}"
        )
    if threads <= 0 or threads % 32 != 0:
        raise ValueError(
            f"MXFP8 CuTe quantizer threads must be a positive multiple of 32, got {threads}"
        )
    if value_order not in {"linear", "trellis_native_mma"}:
        raise ValueError(
            "MXFP8 CuTe quantizer value_order must be 'linear' or "
            f"'trellis_native_mma', got {value_order!r}"
        )
    if value_order == "trellis_native_mma" and subgroup_width != 8:
        raise ValueError(
            "trellis_native_mma MXFP8 ordering requires subgroup_width=8"
        )
    launch = _MXFP8RowsQuantLaunch(
        k,
        source_type,
        subgroup_width,
        threads,
        value_order == "trellis_native_mma",
    )
    cache_key = (
        k,
        source_dtype_name,
        int(subgroup_width),
        int(threads),
        value_order,
        device_index,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=cache_key,
    )
    raw = b12x_compile(
        launch,
        make_ptr(source_type, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.mxfp8_quant_cute",
            4,
            cache_key,
        ),
    )

    def launch_tensors(
        source: torch.Tensor,
        values: torch.Tensor,
        scale_rows: torch.Tensor,
        scale_mma: torch.Tensor,
    ) -> None:
        if subgroup_width:
            groups_per_warp = 32 // subgroup_width
            total_tasks = int(source.shape[0]) * (
                (k // 32 + groups_per_warp - 1) // groups_per_warp
            )
            warps_per_cta = threads // 32
            natural_grid = max(1, (total_tasks + warps_per_cta - 1) // warps_per_cta)
            sm_count = torch.cuda.get_device_properties(
                source.device
            ).multi_processor_count
            grid_x = min(natural_grid, sm_count * _GRID_CTAS_PER_SM)
        else:
            total_blocks = int(source.shape[0]) * (k // 32)
            grid_x = max(1, (total_blocks + threads - 1) // threads)
        raw(
            make_ptr(
                source_type,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
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
            int(source.shape[0]),
            grid_x,
            current_cuda_stream(),
        )

    return launch_tensors


def _backing_span(tensor: torch.Tensor) -> int:
    """Bytes from ``tensor.data_ptr()`` to the end of its untyped storage.

    Uses ``storage_offset`` + ``nbytes`` of the *storage* (not ``numel``,
    which zero-stride expansions spoof) so a one-element backing allocation
    expanded to a large logical shape is correctly detected as too short.
    """
    untyped = tensor.untyped_storage()
    # storage_offset is in elements of the tensor's current dtype; convert
    # to bytes using the element size.
    elem_size = tensor.element_size()
    offset_bytes = tensor.storage_offset() * elem_size
    return len(untyped) - offset_bytes


def _check_alignment(name: str, tensor: torch.Tensor) -> None:
    ptr = tensor.data_ptr()
    if ptr % _KERNEL_ALIGN_BYTES != 0:
        raise ValueError(
            f"{name} base pointer must be {_KERNEL_ALIGN_BYTES}-byte aligned "
            f"for the CuTe kernel, got data_ptr()=0x{ptr:x} "
            f"(offset {ptr % _KERNEL_ALIGN_BYTES})"
        )


def _check_device(name: str, tensor: torch.Tensor, ref_device: torch.device) -> None:
    if tensor.device != ref_device:
        raise ValueError(
            f"{name} must be on the source/launch device {ref_device}, "
            f"got {tensor.device}"
        )


def _check_no_alias(
    name_a: str, ptr_a: int, span_a: int,
    name_b: str, ptr_b: int, span_b: int,
) -> None:
    """Reject overlapping byte ranges ``[ptr, ptr+span)``."""
    if ptr_a < ptr_b + span_b and ptr_b < ptr_a + span_a:
        raise ValueError(
            f"{name_a} and {name_b} have overlapping storage "
            f"(0x{ptr_a:x}+{span_a} vs 0x{ptr_b:x}+{span_b})"
        )


def validate_mxfp8_rows_source(source: torch.Tensor) -> None:
    """Validate the complete raw-pointer contract for an MXFP8 source."""
    if not isinstance(source, torch.Tensor):
        raise TypeError(
            f"source must be a torch.Tensor, got {type(source).__name__}"
        )
    if source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"CuTe MXFP8 quantizer requires BF16 or FP16 input, got {source.dtype}"
        )
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("CuTe MXFP8 quantizer requires contiguous [M,K] input")
    if not source.is_cuda:
        raise ValueError("source must be on CUDA")
    m, k = (int(dim) for dim in source.shape)
    _compute_mxfp8_rows_geometry(m, k)
    _check_alignment("source", source)
    required_bytes = m * k * source.element_size()
    span = _backing_span(source)
    if span < required_bytes:
        raise ValueError(
            f"source backing storage is {span} bytes from data_ptr() but the "
            f"kernel reads {required_bytes} bytes"
        )


def _compute_mxfp8_rows_geometry(
    m: int,
    k: int,
) -> dict[str, int]:
    """Compute exact destination spans and prove every Int32 kernel index."""
    if m <= 0:
        raise ValueError(f"MXFP8 row quantizer requires positive M, got {m}")
    if k <= 0 or k % _SCALE_VEC_SIZE != 0:
        raise ValueError(
            f"MXFP8 row quantizer requires K divisible by {_SCALE_VEC_SIZE}, got {k}"
        )

    g = k // _SCALE_VEC_SIZE
    m_tiles = (m + _SCALE_ROW_TILE - 1) // _SCALE_ROW_TILE
    k_tiles = (g + _SCALE_K_TILE - 1) // _SCALE_K_TILE
    values_bytes = m * k
    scale_rows_bytes = m * g
    scale_mma_bytes = m_tiles * k_tiles * 512

    # The row permutation is not monotonic within a 128-row tile. Compute its
    # exact maximum over the live rows in the final tile rather than assuming
    # the final logical row has the largest address.
    live_rows_in_last_tile = (m - 1) % _SCALE_ROW_TILE + 1
    max_row_offset = max(
        (row % 32) * 16 + (row // 32) * 4
        for row in range(live_rows_in_last_tile)
    )
    last_group = g - 1
    scale_mma_max_offset = (
        ((m - 1) // _SCALE_ROW_TILE) * k_tiles * 512
        + (last_group // _SCALE_K_TILE) * 512
        + max_row_offset
        + last_group % _SCALE_K_TILE
    )

    # All dynamic coordinates, task counters, and explicit scale offsets in
    # the kernel are Int32. Python integers make these products overflow-safe.
    int32_maxima = {
        "M": m,
        "source element index": values_bytes - 1,
        "values word index": m * (k // 4) - 1,
        "scale_rows element index": scale_rows_bytes - 1,
        "task count": m * g,
        "scale_mma byte offset": scale_mma_max_offset,
    }
    overflow = {
        name: value for name, value in int32_maxima.items() if value > _INT32_MAX
    }
    if overflow:
        details = ", ".join(f"{name}={value}" for name, value in overflow.items())
        raise ValueError(
            f"MXFP8 row quantizer Int32 index range exceeded for M={m}, K={k}: "
            f"{details}"
        )
    if scale_mma_max_offset >= scale_mma_bytes:
        raise RuntimeError(
            "internal MXFP8 scale geometry exceeds its canonical backing span"
        )
    return {
        "g": g,
        "m_tiles": m_tiles,
        "k_tiles": k_tiles,
        "values_bytes": values_bytes,
        "scale_rows_bytes": scale_rows_bytes,
        "scale_mma_bytes": scale_mma_bytes,
        "scale_mma_max_offset": scale_mma_max_offset,
    }


def _validate_mxfp8_rows_destination(
    *,
    name: str,
    tensor: torch.Tensor,
    ref_device: torch.device,
    expected_dtype: torch.dtype,
    expected_shape: tuple[int, ...],
    expected_strides: tuple[int, ...] | None,
    required_bytes: int,
    is_scale_mma: bool = False,
) -> int:
    """Validate one caller-owned MXFP8 row-quantizer destination tensor.

    Enforces dtype, CUDA device equality, canonical contiguous layout/stride,
    exact capacity measured against untyped backing storage (not ``numel``),
    16-byte ``data_ptr()`` alignment, and returns the base ``data_ptr()`` for
    alias checking.

    ``expected_strides`` is the exact required stride tuple; ``None`` means
    "contiguous for this shape" (computed from shape).
    ``is_scale_mma`` enables the canonical permuted-stride contract instead
    of plain contiguity.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name} must have dtype {expected_dtype}, got {tensor.dtype}"
        )
    _check_device(name, tensor, ref_device)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    # Stride contract
    if is_scale_mma:
        # scale_mma is a permuted view of physical [1, m_tiles, k_tiles, 32, 4, 4].
        # Canonical strides: (16, 4, k_tiles*512, 1, 512, m_tiles*k_tiles*512).
        # Verify exact stride match.
        actual_strides = tuple(tensor.stride())
        m_tiles = expected_shape[2]
        k_tiles = expected_shape[4]
        canonical_strides = (
            16, 4, k_tiles * 512, 1, 512, m_tiles * k_tiles * 512,
        )
        if actual_strides != canonical_strides:
            raise ValueError(
                f"{name} has noncanonical strides {actual_strides}; "
                f"expected the canonical MMA-swizzled strides {canonical_strides}"
            )
    else:
        if expected_strides is not None:
            actual_strides = tuple(tensor.stride())
            if actual_strides != expected_strides:
                raise ValueError(
                    f"{name} has strides {actual_strides}, "
                    f"expected {expected_strides}"
                )
        else:
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    # Reject zero-stride expansions: any stride of 0 in a non-singleton dim
    # means the logical shape is larger than the physical backing.
    for i, (s, dim) in enumerate(zip(tensor.stride(), tensor.shape, strict=True)):
        if dim > 1 and s == 0:
            raise ValueError(
                f"{name} has a zero stride at dim {i} (dim size {dim}), "
                "which indicates a noncanonical expansion"
            )
    # Alignment
    _check_alignment(name, tensor)
    # Backing span: bytes from data_ptr() to end of storage.  This catches
    # one-byte allocations expanded to exact logical shapes.
    span = _backing_span(tensor)
    if span < required_bytes:
        raise ValueError(
            f"{name} backing storage is {span} bytes from data_ptr() but the "
            f"kernel writes {required_bytes} bytes"
        )
    return tensor.data_ptr()


def validate_mxfp8_rows_destinations(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
) -> None:
    """Validate every caller-owned MXFP8 row-quantizer destination before launch.

    This is the single reusable exact destination contract for the row
    quantizer.  It checks:
      - Source is CUDA (the launch device).
      - Each destination's dtype, shape, device equality, canonical stride/
        contiguity, 16-byte alignment, and sufficient backing bytes from
        ``data_ptr()`` to the end of untyped storage.
      - No pairwise overlap among source, values, scale_rows, scale_mma
        write regions (rejects aliases).
      - Int32 safety of the scale_mma offset arithmetic.

    Canonical producer behavior is preserved: allocator-produced tensors
    from ``empty_mxfp8_rows_for_dense_gemm`` and disjoint aligned slices of
    a shared scratch arena remain accepted.
    """
    validate_mxfp8_rows_source(source)
    ref_device = source.device

    m = int(source.shape[0])
    k = int(source.shape[1])
    geom = _compute_mxfp8_rows_geometry(m, k)
    g = geom["g"]
    m_tiles = geom["m_tiles"]
    k_tiles = geom["k_tiles"]

    # values: [M, K] float8_e4m3fn, contiguous, M*K bytes
    values_ptr = _validate_mxfp8_rows_destination(
        name="values",
        tensor=values,
        ref_device=ref_device,
        expected_dtype=torch.float8_e4m3fn,
        expected_shape=(m, k),
        expected_strides=(k, 1),
        required_bytes=geom["values_bytes"],
    )

    # scale_rows: [1, M, G] float8_e8m0fnu, contiguous, M*G bytes
    scale_rows_ptr = _validate_mxfp8_rows_destination(
        name="scale_rows",
        tensor=scale_rows,
        ref_device=ref_device,
        expected_dtype=torch.float8_e8m0fnu,
        expected_shape=(1, m, g),
        expected_strides=(m * g, g, 1),
        required_bytes=geom["scale_rows_bytes"],
    )

    # scale_mma: [32, 4, ceil(M/128), 4, ceil(G/4), 1] float8_e8m0fnu,
    # canonical permuted strides, ceil(M/128)*ceil(G/4)*512 bytes
    expected_scale_mma_shape = (32, 4, m_tiles, 4, k_tiles, 1)
    scale_mma_ptr = _validate_mxfp8_rows_destination(
        name="scale_mma",
        tensor=scale_mma,
        ref_device=ref_device,
        expected_dtype=torch.float8_e8m0fnu,
        expected_shape=expected_scale_mma_shape,
        expected_strides=None,
        required_bytes=geom["scale_mma_bytes"],
        is_scale_mma=True,
    )

    # Alias checks: reject overlapping write regions.
    source_ptr = source.data_ptr()
    source_bytes = source.element_size() * source.numel()
    _check_no_alias("source", source_ptr, source_bytes,
                     "values", values_ptr, geom["values_bytes"])
    _check_no_alias("source", source_ptr, source_bytes,
                     "scale_rows", scale_rows_ptr, geom["scale_rows_bytes"])
    _check_no_alias("source", source_ptr, source_bytes,
                     "scale_mma", scale_mma_ptr, geom["scale_mma_bytes"])
    _check_no_alias("values", values_ptr, geom["values_bytes"],
                     "scale_rows", scale_rows_ptr, geom["scale_rows_bytes"])
    _check_no_alias("values", values_ptr, geom["values_bytes"],
                     "scale_mma", scale_mma_ptr, geom["scale_mma_bytes"])
    _check_no_alias("scale_rows", scale_rows_ptr, geom["scale_rows_bytes"],
                     "scale_mma", scale_mma_ptr, geom["scale_mma_bytes"])


def quantize_mxfp8_rows_cute(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    value_order: str = "linear",
) -> None:
    """Quantize contiguous BF16 rows into dense-GEMM MXFP8 layouts.

    ``trellis_native_mma`` applies the fixed within-K32 byte permutation used
    by direct native-trellis E4M3 B fragments.  It changes neither values nor
    scale groups and avoids a separate activation transpose kernel.
    """

    validate_mxfp8_rows_destinations(source, values, scale_rows, scale_mma)
    if value_order == "trellis_native_mma":
        subgroup_width = 8
    else:
        subgroup_width = _WARP_SUBGROUP_WIDTH if int(source.shape[0]) > 8 else 0
    with torch.cuda.device(source.device):
        _get_compiled_mxfp8_rows_quant(
            int(source.shape[1]),
            source.dtype,
            subgroup_width,
            _THREADS,
            value_order,
            source.device,
        )(
            source,
            values,
            scale_rows,
            scale_mma,
        )
