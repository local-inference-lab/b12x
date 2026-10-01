"""Scale packing for gemm.mgroup_fp8_gemm: plain f32 contract scales to the
dense-GEMM UE8M0 MMA layout.

The fork/xcheck contract hands the op plain f32 block scales whose values are
exact powers of two (``per_token_cast_to_fp8(use_ue8m0=True)``), so widening
granularity (masked/SFB gran-128 -> gran-32 by repeating each byte across
four K32 blocks) and the f32->UE8M0 exponent-byte cast are both exact.
Run paths pack with a single Triton kernel per operand into a fresh
caching-allocator buffer per call (``pack_grouped_scales_fast``; the
``masked_m`` variant skips groups with zero live rows — the GEMM never reads
their atoms). The torch composition (``pack_grouped_scales``) stays as the
allocating reference the GPU tests diff against.
"""
from __future__ import annotations

import torch

from .._shared.wo_mxfp8 import pack_mxfp8_scales_for_dense_gemm


def _check_f32_scales(name: str, scales: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(scales, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not scales.is_cuda or scales.dtype != torch.float32:
        raise ValueError(f"{name} must be a CUDA f32 tensor")
    if tuple(scales.shape) != shape:
        raise ValueError(f"{name} shape {tuple(scales.shape)} differs from {shape}")
    if not scales.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _f32_pow2_to_ue8m0(scales_f32: torch.Tensor) -> torch.Tensor:
    """Cast exact power-of-two f32 scales to UE8M0 bytes (exponent field)."""
    bits = scales_f32.view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8).view(torch.float8_e8m0fnu)


def expand_to_gran32(scales_f32: torch.Tensor, *, gran: int) -> torch.Tensor:
    """(... , k/gran) f32 power-of-two scales -> (..., k/32) UE8M0 byte tensor."""
    if gran == 128:
        scales_f32 = scales_f32.repeat_interleave(4, dim=-1)
    elif gran != 32:
        raise ValueError(f"scale granularity must be 32 or 128, got {gran}")
    return _f32_pow2_to_ue8m0(scales_f32)


def pack_grouped_scales(
    scales_f32: torch.Tensor, *, rows: int, k: int, num_groups: int, gran: int
) -> torch.Tensor:
    """Plain f32 (..., k/gran) scales to the dense-GEMM MMA atom layout."""
    expected = (
        (rows, k // gran) if num_groups == 1 else (num_groups, rows, k // gran)
    )
    _check_f32_scales("scales", scales_f32, expected)
    rows_e8m0 = expand_to_gran32(scales_f32, gran=gran)
    return pack_mxfp8_scales_for_dense_gemm(
        rows_e8m0, m=rows, k=k, num_groups=num_groups
    )


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit(do_not_specialize=["rows", "row_atoms"],
            do_not_specialize_on_alignment=["rows", "row_atoms"])
def _pack_ue8m0_mma_kernel(
    src_ptr,  # f32 (G, rows, k_src) contiguous, exact powers of two
    dst_ptr,  # uint8 MMA-atom buffer
    rows,
    k_src,  # source scale columns per row (k / gran)
    row_atoms,  # ceil(rows / 128)
    k_atoms,  # ceil((k / 32) / 4)
    masked_ptr,  # nullable int32 (G,) per-group live row counts
    REP: tl.constexpr,  # gran-128 sources repeat across four K32 blocks
    HAS_MASK: tl.constexpr,
    K_ATOMS_PER_PASS: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over G * row_atoms
    g = pid_m // row_atoms
    if HAS_MASK:
        # Dead groups (masked_m[g] == 0) pack nothing: the GEMM's skip path
        # never reads their scale atoms, so stale dst bytes are unreachable.
        if tl.load(masked_ptr + g) <= 0:
            return
    mt = pid_m % row_atoms
    lr = tl.arange(0, 128)
    r = mt * 128 + lr
    in_m = r < rows
    # 1.0f has exponent field 127, the layout's pad byte.
    src_base = (
        src_ptr
        + g.to(tl.int64) * (rows.to(tl.int64) * k_src)
        + r.to(tl.int64)[:, None] * k_src
    )
    # MMA atom layout (32, 4, row_atoms, 4, k_atoms, G) with strides
    # (16, 4, k_atoms*512, 1, 512, row_atoms*k_atoms*512).
    dst_base = (
        dst_ptr
        + pid_m.to(tl.int64) * k_atoms * 512
        + (lr % 32)[:, None] * 16
        + (lr // 32)[:, None] * 4
    )
    for kt0 in range(0, k_atoms, K_ATOMS_PER_PASS):
        jj = tl.arange(0, K_ATOMS_PER_PASS * 4)
        kk = kt0 * 4 + jj  # gran-32 column
        in_k = kk < k_src * REP
        vals = tl.load(
            src_base + (kk // REP)[None, :],
            mask=in_m[:, None] & in_k[None, :],
            other=1.0,
        )
        e8m0 = ((vals.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
        # Pad lanes (out-of-range row/col) carry the loaded 1.0f pad byte.
        tl.store(dst_base + (kt0 + jj // 4)[None, :] * 512 + (jj % 4)[None, :], e8m0)


def pack_grouped_scales_into(
    scales_f32: torch.Tensor,
    dst: torch.Tensor,
    *,
    rows: int,
    k: int,
    num_groups: int,
    gran: int,
    masked_m: torch.Tensor | None = None,
    compact128: bool = False,
) -> torch.Tensor:
    """Pack plain f32 (..., k/gran) scales into the MMA-layout buffer.

    One kernel launch, no allocation. Every byte of the live atom region is
    written, including the 127 pad byte, so stale bytes from earlier uses of
    ``dst`` never leak. With ``masked_m`` (masked mode), groups whose live
    row count is zero are skipped outright — legal because the GEMM skips
    all their tiles and never reads those atoms. ``compact128`` stores four
    consecutive gran-128 exponent bytes per word instead of repeating each byte.
    """
    expected = (
        (rows, k // gran) if num_groups == 1 else (num_groups, rows, k // gran)
    )
    _check_f32_scales("scales", scales_f32, expected)
    if compact128:
        if gran != 128 or k % 128:
            raise ValueError("compact128 requires gran-128 sources and K divisible by 128")
        rep = 1
    elif gran == 128:
        rep = 4
    elif gran == 32:
        rep = 1
    else:
        raise ValueError(f"scale granularity must be 32 or 128, got {gran}")
    if k % 32:
        raise ValueError(f"K must be divisible by 32, got {k}")
    if masked_m is not None:
        if (
            not isinstance(masked_m, torch.Tensor)
            or masked_m.dtype != torch.int32
            or not masked_m.is_cuda
            or tuple(masked_m.shape) != (num_groups,)
            or not masked_m.is_contiguous()
        ):
            raise ValueError("masked_m must be a contiguous CUDA int32 (G,) tensor")
    row_atoms = -(-rows // 128)
    k_atoms = -(-k // 512) if compact128 else -(-k // 128)
    required = num_groups * row_atoms * k_atoms * 512
    if not isinstance(dst, torch.Tensor) or dst.dtype != torch.uint8:
        raise ValueError("dst must be a uint8 workspace tensor")
    if not dst.is_cuda or not dst.is_contiguous() or dst.numel() < required:
        raise ValueError(
            f"dst workspace too small or invalid: need {required} bytes, "
            f"got {dst.numel() if isinstance(dst, torch.Tensor) else 'n/a'}"
        )
    from b12x._lib.compile_plan import compile_only_launches_enabled, launch_triton

    k_per_pass = 2 if k_atoms % 2 == 0 else 1
    compiled = launch_triton(
        _pack_ue8m0_mma_kernel, (num_groups * row_atoms,),
        scales_f32, dst, rows, k // gran, row_atoms, k_atoms,
        masked_m if masked_m is not None else dst,
        REP=rep, HAS_MASK=masked_m is not None, K_ATOMS_PER_PASS=k_per_pass,
    )
    return compiled if compile_only_launches_enabled() else dst


def pack_grouped_scales_fast(
    scales_f32: torch.Tensor, *, rows: int, k: int, num_groups: int, gran: int,
    masked_m: torch.Tensor | None = None,
    compact128: bool = False,
) -> torch.Tensor:
    """Allocate the exact MMA-layout buffer and pack in one kernel launch.

    The allocation rides the torch caching allocator (sub-10 us warm, reused
    across calls, CUDA-graph capture-safe); the pack itself is a single
    Triton kernel instead of the torch reference's op chain.
    """
    row_atoms = -(-rows // 128)
    k_atoms = -(-k // 512) if compact128 else -(-k // 128)
    dst = torch.empty(
        num_groups * row_atoms * k_atoms * 512,
        dtype=torch.uint8, device=scales_f32.device,
    )
    return pack_grouped_scales_into(
        scales_f32, dst, rows=rows, k=k, num_groups=num_groups, gran=gran,
        masked_m=masked_m, compact128=compact128,
    )


@triton.jit
def _zero_label_padding_kernel(d_ptr, labels_ptr, n, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    if label < 0:
        offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            d_ptr + row.to(tl.int64) * n + offs,
            tl.zeros((BLOCK_N,), dtype=tl.bfloat16),
            mask=offs < n,
        )


def zero_label_padding_rows(d: torch.Tensor, labels: torch.Tensor, *, n: int | None = None):
    """Zero contiguous-mode padding rows (label -1) of D in the live span.

    The grouped kernel never writes valid data to those rows (they compute
    against a clamped group); this pass enforces the op's zero-fill guarantee
    without allocations during the run.
    """
    n = d.shape[1] if n is None else n
    block_n = 4096 if n >= 4096 else 1024
    _zero_label_padding_kernel[(d.shape[0], triton.cdiv(n, block_n))](
        d, labels, n, BLOCK_N=block_n
    )
    return d
