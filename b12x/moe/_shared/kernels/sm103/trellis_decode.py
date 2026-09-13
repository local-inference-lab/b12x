"""Decode one lane of the existing circular t256 weight representation.

The result preserves the checkpoint's FP16 rounding. Callers choose whether
to store diagnostic tiles or stage these eight values directly for MMA.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

from b12x._lib.intrinsics import (
    fp8x4_e4m3_to_half2x2,
    packed_decode_sqg_fp16_d3l_to_half2x4,
    packed_decode_trellis_sqg_direct_lut_to_e4m3x8,
    packed_dequant_trellis_stream_to_half2x4,
    trellis_align_stream_u32x2,
)
from ..trellis_ring import trellis256_lane_geom_bits
from ...trellis_codebooks import normalize_codebook, validate_codebook_bits


def decoder_contract(bits: int, codebook: str) -> str:
    codebook = normalize_codebook(codebook)
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError("t256 reconstruction requires K2 through K6")
    validate_codebook_bits(codebook, bits)
    return codebook


@cute.jit
def decode_lane(
    source: cute.Tensor,
    tile: Int64,
    lane: Int32,
    lut: cute.Pointer,
    bits: cutlass.Constexpr,
    codebook: cutlass.Constexpr,
):
    """Return eight FP16 values in native lane order without global scratch."""
    ia, ib, shift, span = trellis256_lane_geom_bits(lane, 0, 8, bits)
    middle = (ia + 1) % (8 * bits)
    base = Int64(tile) * Int64(8 * bits)
    # At K5/K6 the lane's eight windows can span three circular words.
    lo, hi = trellis_align_stream_u32x2(
        source[base + Int64(ia)],
        source[base + Int64(middle)],
        source[base + Int64(ib)],
        shift,
        span,
    )
    if cutlass.const_expr(codebook == "mcg"):
        h0, h1, h2, h3 = packed_dequant_trellis_stream_to_half2x4(lo, hi, bits)
    else:
        second = hi
        if cutlass.const_expr(bits != 6):
            second = Uint32(((Uint64(hi) << 32) | Uint64(lo)) >> (4 * bits))
        if cutlass.const_expr(codebook == "sqg_fp16"):
            h0, h1, h2, h3 = packed_decode_sqg_fp16_d3l_to_half2x4(
                lo, second, lut.toint(), bits
            )
        else:
            e0, e1 = packed_decode_trellis_sqg_direct_lut_to_e4m3x8(
                lo, second, lut.toint(), bits, rate_indexed=True
            )
            h0, h1 = fp8x4_e4m3_to_half2x2(e0)
            h2, h3 = fp8x4_e4m3_to_half2x2(e1)
    words = cute.make_rmem_tensor(4, Uint32)
    words[0], words[1], words[2], words[3] = h0, h1, h2, h3
    return cute.recast_tensor(words, cutlass.Float16)


@cute.jit
def tile_coordinates(lane: Int32, value: cutlass.Constexpr):
    """Map native lane/value order to the logical (N,K) tile axes."""
    row = 2 * (lane // 8) + ((lane >> 2) & 1) + (8 if value >= 4 else 0)
    col = 2 * (lane % 4) + (value % 2) + (8 if value % 4 >= 2 else 0)
    return row, col
