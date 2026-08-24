"""CuTeDSL cooperative producer for native KVarN MLA decode."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    f16x2_to_f32x2,
    fp8x4_e4m3_to_bfloat2x2,
    get_ptr_as_int64,
    ld_global_b16,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_u32,
    st_shared_bf16_from_f32,
    st_shared_f32,
    st_shared_u16,
    st_shared_u32,
    st_shared_v4_u32,
)

_GROUP = 64
_TILE = 64
_META_SCALE_OFFSET = _TILE * 4
_LATENT_DIM = 512
_ROPE_DIM = 64
_K5_S_COL_OFFSET = 20_480
_K5_ZP_OFFSET = 21_504
_K5_S_ROW_OFFSET = 22_528

@dsl_user_op
def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load one packed payload byte from an arbitrary global address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )

@dsl_user_op
def _pack_f32x2_to_bfloat2_rn(
    x0: Float32, x1: Float32, *, loc=None, ip=None
) -> Uint32:
    """Pack two f32 values with the scalar store's non-saturating RN semantics."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_K5_ROPE_OFFSET = 22_656
_PRODUCER_THREADS = 128


@cute.jit
def _load_f16(cache_u8: cute.Tensor, byte_offset: Int64) -> Float32:
    bits = ld_global_b16(get_ptr_as_int64(cache_u8, byte_offset))
    value, _ = f16x2_to_f32x2(bits)
    return value


@cute.jit
def io_issue_kvarn_k5_gather(
    cache_u8: cute.Tensor,
    topk_indices: cute.Tensor,
    block_to_pool_slot: cute.Tensor,
    latent_pool: cute.Tensor,
    rope_pool: cute.Tensor,
    kv_bf16_dst_addr: Int32,
    meta_dst_addr: Int32,
    kv_rope_dst_addr: Int32,
    token_idx_view: cute.Tensor,
    full_mbar_ptr,
    g_start: Int32,
    g_end: Int32,
    io_lane: Int32,
    num_blocks: Int32,
    num_pool_slots: Int32,
    *,
    cache_block_stride: cutlass.Constexpr,
    packed_bits: cutlass.Constexpr = 5,
    s_col_offset: cutlass.Constexpr = _K5_S_COL_OFFSET,
    zp_offset: cutlass.Constexpr = _K5_ZP_OFFSET,
    s_row_offset: cutlass.Constexpr = _K5_S_ROW_OFFSET,
    rope_offset: cutlass.Constexpr = _K5_ROPE_OFFSET,
    exact_fp8: cutlass.Constexpr = False,
    exact_pool_only: cutlass.Constexpr = False,
    exact_fast_io: cutlass.Constexpr = False,
    kv_smem_stride: cutlass.Constexpr,
    rope_smem_stride: cutlass.Constexpr,
    io_threads: cutlass.Constexpr = _PRODUCER_THREADS,
):
    """Stage one BI64 tile directly into the native shared-memory format.

    ``exact_fp8=False`` reconstructs packed K4/K5 and exact rows as BF16.
    ``exact_fp8=True`` copies exact E4M3 rows and writes unit inline scales.
    ``exact_pool_only=True`` rejects packed rows without changing the exact source;
    this lets the H16 prototype reuse the exact producer with BF16 math.
    ``exact_fast_io=True`` selects the vectorized exact-only producer while the
    default retains its scalar A/B baseline. Non-exact rows are invalidated and
    zero-filled. Map value ``-1`` alone authorizes packed rows in mixed mode.
    Every other out-of-range map or physical index is invalid. No selected-row
    global stage is materialized, and candidate order is preserved.
    """
    meta_entry = io_lane
    while meta_entry < Int32(_TILE):
        cand = g_start + meta_entry
        physical = Int32(-1)
        if cand < g_end:
            physical = Int32(topk_indices[cand])
        valid = physical >= Int32(0) and physical < num_blocks * Int32(_GROUP)
        safe = physical
        if not valid:
            safe = Int32(0)
        block = safe >> Int32(6)
        pool_slot = Int32(-2)
        if num_blocks > Int32(0):
            pool_slot = Int32(block_to_pool_slot[block])
        packed = valid and pool_slot == Int32(-1)
        if cutlass.const_expr(exact_fp8 or exact_pool_only):
            # Exact-only specializations never authorize the mixed-mode -1 map
            # sentinel. Capture/profile may deliberately present all-unmapped
            # ownership; those rows are neutral rather than live zero-K logits.
            packed = False
        exact = valid and pool_slot >= Int32(0) and pool_slot < num_pool_slots
        status = pool_slot
        if not (packed or exact):
            physical = Int32(-1)
            status = Int32(-2)
        token_idx_view[meta_entry] = physical
        st_shared_u32(meta_dst_addr + meta_entry * Int32(4), Uint32(status))
        if cutlass.const_expr(not exact_fp8 or not exact_fast_io):
            row_scale = Float32(0.0)
            if packed:
                base = Int64(block) * Int64(cache_block_stride)
                token = safe & Int32(63)
                row_scale = _load_f16(
                    cache_u8,
                    base + Int64(s_row_offset) + token.to(Int64) * Int64(2),
                )
            st_shared_f32(
                meta_dst_addr + Int32(_META_SCALE_OFFSET) + meta_entry * Int32(4),
                row_scale,
            )
        meta_entry += Int32(io_threads)

    cute.arch.barrier(barrier_id=4, number_of_threads=io_threads)
    if cutlass.const_expr(exact_fp8):
        if cutlass.const_expr(exact_fast_io):
            fast_scale_entry = io_lane
            while fast_scale_entry < Int32(_TILE):
                st_shared_v4_u32(
                    kv_bf16_dst_addr
                    + fast_scale_entry * Int32(kv_smem_stride)
                    + Int32(512),
                    Uint32(0x3F800000),
                    Uint32(0x3F800000),
                    Uint32(0x3F800000),
                    Uint32(0x3F800000),
                )
                fast_scale_entry += Int32(io_threads)
        else:
            i = io_lane
            while i < Int32(_TILE * 4):
                slow_scale_entry = i // Int32(4)
                scale_group = i - slow_scale_entry * Int32(4)
                st_shared_f32(
                    kv_bf16_dst_addr
                    + slow_scale_entry * Int32(kv_smem_stride)
                    + Int32(512)
                    + scale_group * Int32(4),
                    Float32(1.0),
                )
                i += Int32(io_threads)

    warp = io_lane >> Int32(5)
    lane = io_lane & Int32(31)
    entry = warp
    while entry < Int32(_TILE):
        physical = Int32(token_idx_view[entry])
        valid = physical >= Int32(0)
        safe = physical
        if not valid:
            safe = Int32(0)
        block = safe >> Int32(6)
        token = safe & Int32(63)
        pool_slot = Int32(ld_shared_u32(meta_dst_addr + entry * Int32(4)))
        packed = valid and pool_slot == Int32(-1)
        exact = valid and pool_slot >= Int32(0) and pool_slot < num_pool_slots
        base = Int64(block) * Int64(cache_block_stride)
        token_bit = token * Int32(packed_bits)
        token_byte = token_bit >> Int32(3)
        shift = token_bit & Int32(7)
        s_row = Float32(0.0)
        if cutlass.const_expr(not exact_fp8):
            s_row = ld_shared_f32(
                meta_dst_addr + Int32(_META_SCALE_OFFSET) + entry * Int32(4)
            )
        if exact:
            if cutlass.const_expr(exact_fp8 and exact_fast_io):
                # Exact rows are 512-byte aligned. One 128-bit load/store per
                # lane copies the full E4M3 row with a single warp.
                dim16 = lane * Int32(16)
                while dim16 < Int32(_LATENT_DIM):
                    src = (
                        pool_slot.to(Int64) * Int64(_GROUP * _LATENT_DIM)
                        + token.to(Int64) * Int64(_LATENT_DIM)
                        + dim16.to(Int64)
                    )
                    v0, v1, v2, v3 = ld_global_nc_v4_u32(
                        get_ptr_as_int64(latent_pool, src)
                    )
                    st_shared_v4_u32(
                        kv_bf16_dst_addr + entry * Int32(kv_smem_stride) + dim16,
                        v0, v1, v2, v3,
                    )
                    dim16 += Int32(512)
            else:
                dim4 = lane * Int32(4)
                while dim4 < Int32(_LATENT_DIM):
                    src = (
                        pool_slot.to(Int64) * Int64(_GROUP * _LATENT_DIM)
                        + token.to(Int64) * Int64(_LATENT_DIM)
                        + dim4.to(Int64)
                    )
                    raw = ld_global_nc_u32(get_ptr_as_int64(latent_pool, src))
                    dst = kv_bf16_dst_addr + entry * Int32(kv_smem_stride)
                    if cutlass.const_expr(exact_fp8):
                        st_shared_u32(dst + dim4, raw)
                    else:
                        bf0, bf1 = fp8x4_e4m3_to_bfloat2x2(raw)
                        bf16_dst = dst + dim4 * Int32(2)
                        st_shared_u32(bf16_dst, bf0)
                        st_shared_u32(bf16_dst + Int32(4), bf1)
                    dim4 += Int32(128)
        elif cutlass.const_expr(exact_fp8):
            if cutlass.const_expr(exact_fast_io):
                dim16 = lane * Int32(16)
                while dim16 < Int32(_LATENT_DIM):
                    st_shared_v4_u32(
                        kv_bf16_dst_addr + entry * Int32(kv_smem_stride) + dim16,
                        Uint32(0), Uint32(0), Uint32(0), Uint32(0),
                    )
                    dim16 += Int32(512)
            else:
                dim4 = lane * Int32(4)
                while dim4 < Int32(_LATENT_DIM):
                    st_shared_u32(
                        kv_bf16_dst_addr + entry * Int32(kv_smem_stride) + dim4,
                        Uint32(0),
                    )
                    dim4 += Int32(128)
        else:
            # Adjacent dimensions share packed f16 affine loads and one packed
            # non-saturating RN BF16 store. Invalid/exact rows never dereference
            # cache block zero, retaining empty-cache safety.
            dim = lane * Int32(2)
            while dim < Int32(_LATENT_DIM):
                value0 = Float32(0.0)
                value1 = Float32(0.0)
                if packed:
                    byte_position0 = (
                        dim * Int32((_GROUP * packed_bits) // 8) + token_byte
                    )
                    byte00 = _ld_global_nc_u8(
                        get_ptr_as_int64(cache_u8, base + byte_position0.to(Int64))
                    )
                    byte01 = Uint32(0)
                    if shift > Int32(8 - packed_bits):
                        byte01 = _ld_global_nc_u8(
                            get_ptr_as_int64(
                                cache_u8,
                                base + byte_position0.to(Int64) + Int64(1),
                            )
                        )
                    code0 = (
                        (byte00 | (byte01 << Uint32(8))) >> Uint32(shift)
                    ) & Uint32((1 << packed_bits) - 1)

                    dim1 = dim + Int32(1)
                    byte_position1 = (
                        dim1 * Int32((_GROUP * packed_bits) // 8) + token_byte
                    )
                    byte10 = _ld_global_nc_u8(
                        get_ptr_as_int64(cache_u8, base + byte_position1.to(Int64))
                    )
                    byte11 = Uint32(0)
                    if shift > Int32(8 - packed_bits):
                        byte11 = _ld_global_nc_u8(
                            get_ptr_as_int64(
                                cache_u8,
                                base + byte_position1.to(Int64) + Int64(1),
                            )
                        )
                    code1 = (
                        (byte10 | (byte11 << Uint32(8))) >> Uint32(shift)
                    ) & Uint32((1 << packed_bits) - 1)

                    s_col0, s_col1 = f16x2_to_f32x2(
                        ld_global_nc_u32(
                            get_ptr_as_int64(
                                cache_u8,
                                base + Int64(s_col_offset)
                                + dim.to(Int64) * Int64(2),
                            )
                        )
                    )
                    zero0, zero1 = f16x2_to_f32x2(
                        ld_global_nc_u32(
                            get_ptr_as_int64(
                                cache_u8,
                                base + Int64(zp_offset)
                                + dim.to(Int64) * Int64(2),
                            )
                        )
                    )
                    value0 = (Float32(code0) * s_col0 + zero0) * s_row
                    value1 = (Float32(code1) * s_col1 + zero1) * s_row
                dst = (
                    kv_bf16_dst_addr
                    + entry * Int32(kv_smem_stride)
                    + dim * Int32(2)
                )
                st_shared_u32(
                    dst, _pack_f32x2_to_bfloat2_rn(value0, value1)
                )
                dim += Int32(64)
        if cutlass.const_expr(exact_fp8 and exact_fast_io):
            # Reuse this row's physical/map metadata for its BF16 RoPE pair.
            # The exact specialization has one producer warp; folding RoPE into
            # the latent pass removes a second 64-row metadata/division loop.
            rope_dim = lane * Int32(2)
            rope_dst = (
                kv_rope_dst_addr
                + entry * Int32(rope_smem_stride * 2)
                + rope_dim * Int32(2)
            )
            if exact:
                exact_rope_offset = (
                    (pool_slot.to(Int64) * Int64(_GROUP) + token.to(Int64))
                    * Int64(_ROPE_DIM)
                    + rope_dim.to(Int64)
                )
                rope_bits = ld_global_nc_u32(
                    get_ptr_as_int64(rope_pool, exact_rope_offset)
                )
                st_shared_u32(rope_dst, rope_bits)
            else:
                st_shared_u32(rope_dst, Uint32(0))
        entry += Int32(io_threads // 32)

    if cutlass.const_expr(not exact_fp8 or not exact_fast_io):
        entry = warp
        while entry < Int32(_TILE):
            physical = Int32(token_idx_view[entry])
            valid = physical >= Int32(0)
            safe = physical
            if not valid:
                safe = Int32(0)
            block = safe >> Int32(6)
            token = safe & Int32(63)
            pool_slot = Int32(ld_shared_u32(meta_dst_addr + entry * Int32(4)))
            packed = valid and pool_slot == Int32(-1)
            exact = valid and pool_slot >= Int32(0) and pool_slot < num_pool_slots
            base = Int64(block) * Int64(cache_block_stride)
            if cutlass.const_expr(exact_fp8):
                dim = lane
                while dim < Int32(_ROPE_DIM):
                    dst = (
                        kv_rope_dst_addr
                        + entry * Int32(rope_smem_stride * 2)
                        + dim * Int32(2)
                    )
                    if packed:
                        st_shared_u16(dst, Uint32(0))
                    elif exact:
                        st_shared_bf16_from_f32(
                            dst, Float32(rope_pool[pool_slot, token, dim])
                        )
                    else:
                        st_shared_u16(dst, Uint32(0))
                    dim += Int32(32)
            else:
                dim = lane * Int32(2)
                dst = (
                    kv_rope_dst_addr
                    + entry * Int32(rope_smem_stride * 2)
                    + dim * Int32(2)
                )
                if packed:
                    bits = ld_global_nc_u32(
                        get_ptr_as_int64(
                            cache_u8,
                            base
                            + Int64(rope_offset)
                            + token.to(Int64) * Int64(_ROPE_DIM * 2)
                            + dim.to(Int64) * Int64(2),
                        )
                    )
                    st_shared_u32(dst, bits)
                elif exact:
                    exact_rope_offset = (
                        (pool_slot.to(Int64) * Int64(_GROUP) + token.to(Int64))
                        * Int64(_ROPE_DIM)
                        + dim.to(Int64)
                    )
                    bits = ld_global_nc_u32(
                        get_ptr_as_int64(rope_pool, exact_rope_offset)
                    )
                    st_shared_u32(dst, bits)
                else:
                    st_shared_u32(dst, Uint32(0))
            entry += Int32(io_threads // 32)

    cute.arch.barrier(barrier_id=4, number_of_threads=io_threads)
    cute.arch.fence_acq_rel_cta()
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive(full_mbar_ptr)


__all__ = ["io_issue_kvarn_k5_gather"]
