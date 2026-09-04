"""Shared SQG-XOR-Cheb-T12 native-tile decode for W4A8 kernels.

The mixin extracts overlapping L16 trellis windows from the packed native
payload with the t256 ring geometry, decodes them through the modal T12
staircase, and maps the resulting E4M3 bytes into ``m16n8k32`` MMA B
fragments (including the butterfly pairing of adjacent N16 tiles).  Kernel
classes that inherit it provide the payload addressing (``_tile_base``-style
helpers) against their own task coordinates.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
)

_PAIR_U32_PER_K16 = 384  # 8 N16 tiles * 8 u32/bit * (low_bits + high_bits=6)


class _NativeTrellisDecode:
    """Shared native-tile window extraction and E4M3 fragment mapping."""

    @cute.jit
    def _decode_windows(
        self,
        win_a: Uint32,
        win_b: Uint32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int64,
    ):
        return packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            win_a,
            win_b,
            rank_lut_addr,
            int(bits),
        )

    @cute.jit
    def _lane_geom(self, lane: Int32, bits: cutlass.Constexpr[int]):
        bits_i32 = Int32(int(bits))
        ring_u32 = Int32(8 * int(bits))
        t_offset = Int32(8) * lane
        b1 = (t_offset + Int32(257)) * bits_i32
        b0 = b1 - Int32(16)
        b2 = b1 + Int32(7 * int(bits))
        i0 = b0 >> Int32(5)
        i2 = (b2 - Int32(1)) >> Int32(5)
        ia = i0 - ring_u32 * (i0 >= ring_u32).to(Int32)
        ib = i2 - ring_u32 * (i2 >= ring_u32).to(Int32)
        s2 = (i2 + Int32(1)) * Int32(32) - b2
        return ia, ib, s2

    @cute.jit
    def _decode_tile_at(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        base: Int64,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ) -> Uint32:
        ia, ib, s2 = self._lane_geom(lane, bits)
        a = Uint32(trellis[base + Int64(ia)])
        b = Uint32(trellis[base + Int64(ib)])
        merged = (Int64(a) << Int64(32)) | Int64(b)
        win_a = Uint32(merged >> Int64(s2))
        win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
        lo, hi = self._decode_windows(
            win_a,
            win_b,
            bits,
            rank_lut_smem_addr,
        )
        value = lo
        if n_high != Int32(0):
            value = hi
        return value

    @cute.jit
    def _decode_tile_both(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        base: Int64,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        """Decode both adjacent N8 halves of one native N16 tile once."""

        ia, ib, s2 = self._lane_geom(lane, bits)
        a = Uint32(trellis[base + Int64(ia)])
        b = Uint32(trellis[base + Int64(ib)])
        merged = (Int64(a) << Int64(32)) | Int64(b)
        win_a = Uint32(merged >> Int64(s2))
        win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
        return self._decode_windows(
            win_a,
            win_b,
            bits,
            rank_lut_smem_addr,
        )

    @cute.jit
    def _decode_tile_both_geom(
        self,
        trellis: cute.Tensor,
        base: Int64,
        ia: Int32,
        ib: Int32,
        s2: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        a = Uint32(trellis[base + Int64(ia)])
        b = Uint32(trellis[base + Int64(ib)])
        merged = (Int64(a) << Int64(32)) | Int64(b)
        win_a = Uint32(merged >> Int64(s2))
        win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
        return self._decode_windows(
            win_a,
            win_b,
            bits,
            rank_lut_smem_addr,
        )

    @cute.jit
    def _pair_native_words(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        base0: Int64,
        base1: Int64,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        e0 = self._decode_tile_at(
            trellis, lane, base0, n_high, bits, rank_lut_smem_addr
        )
        e1 = self._decode_tile_at(
            trellis, lane, base1, n_high, bits, rank_lut_smem_addr
        )
        c = lane & Int32(3)
        own = e0
        send = e1
        if c >= Int32(2):
            own = e1
            send = e0
        peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))
        return own, peer

    @cute.jit
    def _pair_native_words_both(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        base0: Int64,
        base1: Int64,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        ia, ib, s2 = self._lane_geom(lane, bits)
        e0_lo, e0_hi = self._decode_tile_both_geom(
            trellis, base0, ia, ib, s2, bits, rank_lut_smem_addr
        )
        e1_lo, e1_hi = self._decode_tile_both_geom(
            trellis, base1, ia, ib, s2, bits, rank_lut_smem_addr
        )
        c = lane & Int32(3)
        own_lo = e0_lo
        own_hi = e0_hi
        send_lo = e1_lo
        send_hi = e1_hi
        if c >= Int32(2):
            own_lo = e1_lo
            own_hi = e1_hi
            send_lo = e0_lo
            send_hi = e0_hi
        peer_lo = Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2))
        peer_hi = Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2))
        return own_lo, peer_lo, own_hi, peer_hi

    @cute.jit
    def _a_fragment(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        route: Int32,
        k32: Int32,
        lane: Int32,
    ):
        c = lane & Int32(3)
        g = lane >> Int32(2)
        a0 = Uint32(0)
        a1 = Uint32(0)
        a2 = Uint32(0)
        a3 = Uint32(0)
        if g == Int32(0):
            word0 = k32 * Int32(8) + c * Int32(2)
            a0 = Uint32(values[route, word0])
            a2 = Uint32(values[route, word0 + Int32(1)])

        sf = Uint32(127)
        if g == Int32(0):
            if (lane & Int32(1)) == Int32(0):
                sf = Uint32(scale_rows[route, k32])
        return a0, a1, a2, a3, sf * Uint32(0x01010101)
