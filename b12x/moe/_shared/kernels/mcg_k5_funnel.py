"""Procedural MCG-to-E4M3 decode shared by monolithic and split P8 kernels.

The constants and state construction are ported from ExLlamaV3's procedural
MCG decoder. Sol's change is the alpha-2 half2 compander followed by a native
ties-to-even finite E4M3 conversion, returned directly in QMMA B-register
order. No weight matrix and no runtime codebook table are materialized.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from b12x._lib.intrinsics import ld_shared_u32
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_trellis_pair_words_both as _sqg_pair_words_both,
)


@dsl_user_op
def packed_decode_mcg2_to_e4m3x8(win_a, win_b, bits: int, *, loc=None, ip=None):
    """Decode eight sliding-window states to two packed E4M3x4 words."""
    bits = int(bits)
    if bits not in (3, 4, 5):
        raise ValueError(f"P8 MCG supports K3/K4/K5 streams, got K{bits}")
    asm = """
        {
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi, M;
            .reg .b32 h01,h23,h45,h67;
            .reg .b16 e01,e23,e45,e67;
            mov.b32 M, 0xCBAC1FED;
            and.b32 w7, $2, 0xffff;
            shr.u32 w6, $2, __B1__; and.b32 w6, w6, 0xffff;
            shr.u32 w5, $2, __B2__; and.b32 w5, w5, 0xffff;
            shr.u32 w4, $2, __B3__; and.b32 w4, w4, 0xffff;
            and.b32 w3, $3, 0xffff;
            shr.u32 w2, $3, __B1__; and.b32 w2, w2, 0xffff;
            shr.u32 w1, $3, __B2__; and.b32 w1, w1, 0xffff;
            shr.u32 w0, $3, __B3__; and.b32 w0, w0, 0xffff;
            mul.lo.u32 w0, w0, M; lop3.b32 w0, w0, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w1, w1, M; lop3.b32 w1, w1, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w2, w2, M; lop3.b32 w2, w2, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w3, w3, M; lop3.b32 w3, w3, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w4, w4, M; lop3.b32 w4, w4, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w5, w5, M; lop3.b32 w5, w5, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w6, w6, M; lop3.b32 w6, w6, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            mul.lo.u32 w7, w7, M; lop3.b32 w7, w7, 0x8FFF8FFF, 0x3B603B60, 0x6a;
            prmt.b32 lo, w0, w1, 0x5410; prmt.b32 hi, w0, w1, 0x7632; add.rn.f16x2 h01, lo, hi;
            prmt.b32 lo, w2, w3, 0x5410; prmt.b32 hi, w2, w3, 0x7632; add.rn.f16x2 h23, lo, hi;
            prmt.b32 lo, w4, w5, 0x5410; prmt.b32 hi, w4, w5, 0x7632; add.rn.f16x2 h45, lo, hi;
            prmt.b32 lo, w6, w7, 0x5410; prmt.b32 hi, w6, w7, 0x7632; add.rn.f16x2 h67, lo, hi;
            add.rn.f16x2 h01, h01, h01;
            add.rn.f16x2 h23, h23, h23;
            add.rn.f16x2 h45, h45, h45;
            add.rn.f16x2 h67, h67, h67;
            cvt.rn.satfinite.e4m3x2.f16x2 e01, h01;
            cvt.rn.satfinite.e4m3x2.f16x2 e23, h23;
            cvt.rn.satfinite.e4m3x2.f16x2 e45, h45;
            cvt.rn.satfinite.e4m3x2.f16x2 e67, h67;
            mov.b32 $0, {e01, e23};
            mov.b32 $1, {e45, e67};
        }
    """
    asm = (
        asm.replace("__B1__", str(bits))
        .replace("__B2__", str(2 * bits))
        .replace("__B3__", str(3 * bits))
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(win_a).ir_value(loc=loc, ip=ip), Uint32(win_b).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _mcg_funnel_right(lo, hi, shift, *, loc=None, ip=None):
    result = llvm.inline_asm(T.i32(),
        [Uint32(lo).ir_value(loc=loc,ip=ip),Uint32(hi).ir_value(loc=loc,ip=ip),Uint32(shift).ir_value(loc=loc,ip=ip)],
        "shf.r.wrap.b32 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=False,is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
    return Uint32(result)


@cute.jit
def _mcg_decode_both(smem_base, base, ia, ib, s2, bits):
    a = Uint32(ld_shared_u32(smem_base + ((base + ia) << Int32(2))))
    b = Uint32(ld_shared_u32(smem_base + ((base + ib) << Int32(2))))
    if cutlass.const_expr(int(bits) == 5):
        # A lane's eight overlapping L16 windows span 16 + 7*bits bits: 37 at
        # K3, 44 at K4, 51 at K5.  With the ring geometry's funnel alignment the
        # K3 and K4 spans always fit in the two ring words (ia, ib), but at K5
        # half the lanes start deep enough into their first word that the span
        # crosses into a third word, which the two-word merge skipped (device
        # closure: input carriers exact, middle payload 4030/4096 bytes wrong).
        # Read the middle ring word too and take each 32-bit window from the
        # 64-bit pair that contains it: (mid:last) covers bits [0, 64) of the
        # 96-bit read and (first:mid) covers [32, 96).  Two-word lanes keep
        # (first:last) as the low pair with a zero top word, which reproduces
        # the original arithmetic exactly.
        ring = Int32(8 * int(bits))
        im = ia + Int32(1)
        im = im - ring * (im >= ring).to(Int32)
        m = Uint32(ld_shared_u32(smem_base + ((base + im) << Int32(2))))
        mid = a
        if ib != im:
            mid = m
        # CPU basis proof covers all32K5lane geometries. The second funnel's
        # upper word is always a; m aliases b in the two-word case.
        win_a = _mcg_funnel_right(b, mid, s2)
        win_b = _mcg_funnel_right(m, a, s2 + Int32(20))
        return packed_decode_mcg2_to_e4m3x8(win_a, win_b, int(bits))

    merged = (Int64(a) << Int64(32)) | Int64(b)
    return packed_decode_mcg2_to_e4m3x8(
        Uint32(merged >> Int64(s2)),
        Uint32(merged >> Int64(s2 + Int32(4 * int(bits)))),
        int(bits),
    )


@cute.jit
def _mcg_pair_words_both(smem_base, lane, base0, base1, ia, ib, s2, bits):
    e0_lo, e0_hi = _mcg_decode_both(smem_base, base0, ia, ib, s2, bits)
    e1_lo, e1_hi = _mcg_decode_both(smem_base, base1, ia, ib, s2, bits)
    c = lane & Int32(3)
    own_lo, own_hi, send_lo, send_hi = e0_lo, e0_hi, e1_lo, e1_hi
    if c >= Int32(2):
        own_lo, own_hi, send_lo, send_hi = e1_lo, e1_hi, e0_lo, e0_hi
    return (
        own_lo,
        Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2)),
        own_hi,
        Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2)),
    )


@cute.jit
def w4a8_trellis_pair_words_dispatch(
    smem_base,
    lane,
    base0,
    base1,
    ia,
    ib,
    s2,
    bits,
    lut_addr,
    lut_in_smem=True,
    direct_lut=False,
):
    """Dispatch SQG or table-free MCG using compile-time LUT mode flags."""
    if cutlass.const_expr(not lut_in_smem and not direct_lut):
        return _mcg_pair_words_both(smem_base, lane, base0, base1, ia, ib, s2, bits)
    return _sqg_pair_words_both(
        smem_base,
        lane,
        base0,
        base1,
        ia,
        ib,
        s2,
        bits,
        lut_addr,
        lut_in_smem,
        direct_lut,
    )
