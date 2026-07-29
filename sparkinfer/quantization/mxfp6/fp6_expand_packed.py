"""One-pass GPU expansion of 3:4-packed MX-FP6 codes to byte-containers.

Kernel counterpart of :func:`sparkinfer._lib.fp6.expand_mxfp6_packed_to_bytes`
for the LARGE-M packed-B regime fix: at prefill M the packed-B GEMM streams
lose 1.27-1.28x to the expanded-B kernel (Phase A microbench, every Behemoth
TP=2 shard), so the dense linear expands the packed weight into a shared
scratch buffer per call and runs the expanded-B kernel instead. That only
pays if the expansion itself is bandwidth-cheap: the torch reference makes
several int32 passes with large temporaries (~1 ms+ on a 14336x12288 shard),
while this kernel does one coalesced pass (read 3 bytes/4 codes, write 4)
— ~0.2 ms at the same size, against a measured multi-ms GEMM saving.

Layout contract: input is the flat contiguous ``(N, 3K/4)`` packed weight,
output the flat contiguous ``(N, K)`` byte-container weight (one code in bits
[5:0] of each byte) — bit-identical to ``expand_mxfp6_packed_to_bytes`` (both
are pure bit rearrangements). Each thread handles 48 packed bytes -> 16
four-code groups -> 64 output bytes, so all loads are 16-byte vectors and all
stores 8-byte words. ``K % 128 == 0`` makes every row a multiple of 96 packed
bytes, hence the flat size is always a multiple of 48.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import Int32, Int64, Uint32, Uint64

from sparkinfer._lib.compiler import KernelCompileSpec, compile as sparkinfer_compile
from sparkinfer._lib.intrinsics import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_u64,
)
from sparkinfer._lib.runtime_control import raise_if_kernel_resolution_frozen
from sparkinfer._lib.utils import current_cuda_stream

_THREADS = 256
_UNIT_PACKED_BYTES = 48  # 3 x 16-byte vector loads
_UNIT_OUT_BYTES = 64     # 16 groups x 4 codes

_KERNEL_CACHE: Dict[Tuple, object] = {}


@cute.jit
def _expand_group_word(bits: Uint32) -> Uint32:
    """24 packed bits (one 4-code group) -> 4 byte-containers in one u32.

    ``bits`` holds ``b0 | b1<<8 | b2<<16``; codes are the four 6-bit fields.
    The output word's little-endian bytes are ``c0,c1,c2,c3`` — exactly the
    reference's ``stack((c0..c3), -1)`` byte order.
    """
    return (
        (bits & Uint32(0x3F))
        | (((bits >> Uint32(6)) & Uint32(0x3F)) << Uint32(8))
        | (((bits >> Uint32(12)) & Uint32(0x3F)) << Uint32(16))
        | (((bits >> Uint32(18)) & Uint32(0x3F)) << Uint32(24))
    )


class ExpandPackedKernel:
    """One thread per 48-byte packed unit, fully coalesced in and out."""

    @cute.jit
    def __call__(
        self,
        packed_flat: cute.Tensor,
        out_flat: cute.Tensor,
        grid: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.kernel(packed_flat, out_flat).launch(
            grid=(grid, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mPacked: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        units = Int32(mPacked.shape[0]) // Int32(_UNIT_PACKED_BYTES)
        idx = bidx * Int32(_THREADS) + tidx
        if idx < units:
            # Unit id * stride in Int64 per the 64-bit offset guideline
            # (large shards approach 2^31 flat bytes).
            in_base = Int64(idx) * Int64(_UNIT_PACKED_BYTES)
            out_base = Int64(idx) * Int64(_UNIT_OUT_BYTES)
            w0, w1, w2, w3 = ld_global_v4_u32(get_ptr_as_int64(mPacked, in_base))
            w4, w5, w6, w7 = ld_global_v4_u32(
                get_ptr_as_int64(mPacked, in_base + Int64(16))
            )
            w8, w9, w10, w11 = ld_global_v4_u32(
                get_ptr_as_int64(mPacked, in_base + Int64(32))
            )
            ws = (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11)
            for t in cutlass.range_constexpr(4):
                # 12 consecutive packed bytes = words (a, b, c) little-endian;
                # split into four 24-bit groups spanning the word boundaries.
                a = ws[3 * t]
                b = ws[3 * t + 1]
                c = ws[3 * t + 2]
                g0 = a & Uint32(0x00FFFFFF)
                g1 = (a >> Uint32(24)) | ((b & Uint32(0xFFFF)) << Uint32(8))
                g2 = (b >> Uint32(16)) | ((c & Uint32(0xFF)) << Uint32(16))
                g3 = c >> Uint32(8)
                o0 = _expand_group_word(g0)
                o1 = _expand_group_word(g1)
                o2 = _expand_group_word(g2)
                o3 = _expand_group_word(g3)
                lo = Uint64(o0) | (Uint64(o1) << Uint64(32))
                hi = Uint64(o2) | (Uint64(o3) << Uint64(32))
                st_global_u64(
                    get_ptr_as_int64(mOut, out_base + Int64(t * 16)), lo
                )
                st_global_u64(
                    get_ptr_as_int64(mOut, out_base + Int64(t * 16 + 8)), hi
                )


def compile_fp6_expand_packed(packed_bytes: int):
    """Compile the packed->bytes expansion for a flat packed size.

    Returns ``launch(packed_flat_u8, out_flat_u8)`` where ``packed_flat_u8``
    is the contiguous flat view of the ``(N, 3K/4)`` packed weight and
    ``out_flat_u8`` a flat uint8 buffer of at least ``packed_bytes * 4 // 3``
    bytes (both 16-byte aligned; torch allocations always are).
    """
    assert packed_bytes > 0 and packed_bytes % _UNIT_PACKED_BYTES == 0, (
        f"flat packed size must be a positive multiple of {_UNIT_PACKED_BYTES}, "
        f"got {packed_bytes} (guaranteed by K % 128 == 0)"
    )
    cache_key = (packed_bytes,)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    out_bytes = packed_bytes * 4 // 3
    packed_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (packed_bytes,), assumed_align=16
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (out_bytes,), assumed_align=16
    )
    units = packed_bytes // _UNIT_PACKED_BYTES
    grid = (units + _THREADS - 1) // _THREADS
    kernel = ExpandPackedKernel()
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = sparkinfer_compile(
        kernel,
        packed_fake,
        out_fake,
        grid,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.fp6_expand_packed",
            1,
            cache_key,
        ),
    )

    def launch(packed_flat, out_flat):
        raw(packed_flat, out_flat[:out_bytes], current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch
