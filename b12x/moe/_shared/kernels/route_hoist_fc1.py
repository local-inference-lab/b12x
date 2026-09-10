"""Opt-in P8 M1 N128 owner for the H128 scale-sandwich boundary.
Donor image c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed.
Donor w4a8_phase1.py sha256
7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c.

N64/N32 tasks preserve full ordered K4096 and complete output K32 groups.
Only owned B/SFB spans are staged at their original N128 shared offsets.
This changes copy traffic without changing MMA, scale, or output arithmetic.
Device arithmetic closure and graph replay validation are required.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    atomic_add_global_i32,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    e2m1x8_to_qmma_e2m1x8,
    fabs_f32,
    get_ptr_as_int64,
    ld_shared_bf16_to_f32,
    ld_shared_f32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
    cvt_f32x4_to_e4m3x4,
    shared_ptr_to_u32,
    st_shared_u32,
    st_shared_f32,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_stage_trellis_b_tile,
    _w4a8_trellis_lane_geom,
    _w4a8_trellis_permute_k32,
)
from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.w4a8_mcg_decode import (
    w4a8_trellis_pair_words_dispatch,
)



from b12x.moe._shared.kernels.w4a8_phase1 import W4A8MaterializedPhase1Kernel


@dsl_user_op
def _p8_pack_f32x2_to_half2(x0, x1, *, loc=None, ip=None):
    """Pack FP32 to FP16x2 with the FP16-store-before-Hadamard reference in p8_coupled_scales.py's RN storage cast."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p8_ld_shared_f16_to_f32(addr, *, loc=None, ip=None):
    """Load one shared FP16 element and widen exactly to FP32."""

    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "{.reg .b16 tmp; ld.shared.b16 tmp, [$1]; cvt.f32.f16 $0, tmp;}",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class P8H128NarrowFC1Kernel(W4A8MaterializedPhase1Kernel):
    p8_tile_major = False
    p8_pipeline_stages = 2
    p8_n8_per_warp = 4
    p8_a_swizzle_rotate = False
    p8_warp_quant = False
    p8_exact_staging = False
    p8_broadcast_a = False
    # Set by the owning runtime before compilation. TP4 remains the default.
    p8_intermediate = 512
    tile_m = 16
    source_tile_m = 16
    mma_m_blocks = 1
    owned_row_groups = 4
    scale_sandwich = False
    diagnostic_raw_fc1 = False
    p8_row_interleaved = False

    def __init__(self, tile_n: int, *, trellis_bits: int = 4):
        if tile_n not in (32, 64):
            raise ValueError("P8 narrow FC1 supports N32/N64 only")
        if int(trellis_bits) not in (3, 4, 5):
            raise ValueError("P8 narrow FC1 supports K3, K4 or K5 streams")
        self.trellis_bits = int(trellis_bits)
        # A staged N128xK64 trellis B half is 4 k16 x 8 n16 windows of 8*bits
        # u32 words = 1024*bits bytes; the inherited constant is the K4 value.
        self.b_payload_bytes = 1024 * self.trellis_bits
        self.owned_n = int(tile_n)
        self.num_warps = self.owned_n // 32
        self.threads_per_cta = self.num_warps * 32
        self.subtiles = 128 // self.owned_n
        # Preserve the donor's N128 carrier and shared-memory address layout.
        self.a_payload_bytes = self.tile_m * 128
        self.a_scale_bytes = self.tile_m * 4
        self.sfa_offset = self.a_payload_bytes
        self.gate_b_offset = self.sfa_offset + self.a_scale_bytes
        self.up_b_offset = self.gate_b_offset + self.b_payload_bytes
        self.gate_sfb_offset = self.up_b_offset + self.b_payload_bytes
        self.up_sfb_offset = self.gate_sfb_offset + self.sfb_bytes
        self.stage_bytes = self.up_sfb_offset + self.sfb_bytes
        self.shared_bytes = max(2 * self.stage_bytes, self.tile_m * 128 * 2)
        self.shared_words = (self.shared_bytes + 3) // 4
        self.fast_math = False
        self.source_halves = 1
        self.deterministic_output = True
        self.num_topk = 8
        self.is_situ = False
        self.w4a8_trellis = True
        self.trellis_coupled = False
        self.trellis_direct_lut = False
        self.trellis_codebook = "mcg"
        self.trellis_scaled = True
        self.trellis_identity_boundary = True
        self.trellis_lut_offset = self.shared_bytes

    @cute.jit
    def _scale_fc1_after_h128(
        self,
        value: cutlass.Float32,
        scale_component: cute.Tensor,
        expert_idx: Int32,
        local_col: Int32,
        projection_slot: Int32,
    ) -> cutlass.Float32:
        """Apply private gate/up svh after the owning H128 transform.

        Packed layout is ``gate_up_suh[H] | [E,gate_svh|up_svh|down_suh] |
        down_svh[H]``.  The N128 kernel owns the adjacent halves and performs
        H128 immediately before this helper.
        """

        hidden = Int32(4096)
        intermediate = Int32(self.p8_intermediate)
        scale_idx = (
            hidden
            + expert_idx * Int32(3 * self.p8_intermediate)
            + projection_slot * intermediate
            + local_col
        )
        return value * scale_component[scale_idx].to(cutlass.Float32)

    @cute.jit
    def _scale_down_before_h128(
        self,
        value: cutlass.Float32,
        scale_component: cute.Tensor,
        expert_idx: Int32,
        local_col: Int32,
    ) -> cutlass.Float32:
        """Apply private down suh with FP32 multiply followed by FP16 store before H128 (p8_coupled_scales.py)."""

        scale_idx = (
            Int32(4096)
            + expert_idx * Int32(3 * self.p8_intermediate)
            + Int32(2 * self.p8_intermediate)
            + local_col
        )
        scaled = value * scale_component[scale_idx].to(cutlass.Float32)
        # The following H128 owner must consume this rounded FP16 value.  The
        # E4M3 block amax is computed only after that H128.
        return cutlass.Float16(scaled).to(cutlass.Float32)


    @cute.jit
    def _stage_owned_trellis_b(
        self, tr_u32: cute.Tensor, smem_base: Int32, tile_base: Int64,
        k16_stride_u32: Int32, bits: cutlass.Constexpr, tidx: Int32,
        tcnt: cutlass.Constexpr, k16_rows: cutlass.Constexpr, subtile: Int32,
    ):
        # One vector copies four u32 words (16 bytes). Retain the full N128
        # row stride so the existing decoder's logical warp addresses agree.
        full_chunks = 16 * bits
        owned_chunks = (self.owned_n // 16) * (2 * bits)
        transfers = k16_rows * owned_chunks
        limit = Int64(tr_u32.shape[0]) - Int64(4)
        for i in cutlass.range_constexpr((transfers + tcnt - 1) // tcnt):
            idx = tidx + Int32(i * tcnt)
            if idx < Int32(transfers):
                k16_local = idx // Int32(owned_chunks)
                local_chunk = idx - k16_local * Int32(owned_chunks)
                chunk = subtile * Int32(owned_chunks) + local_chunk
                src = (
                    tile_base + Int64(k16_local) * Int64(k16_stride_u32)
                    + Int64(chunk * Int32(4))
                )
                if cutlass.const_expr(self.p8_exact_staging):
                    # Opt-in only after host validation of the complete
                    # E288/H4096/I512 TP4 stream extent and N128 ownership.
                    cp_async4_shared_global(
                        smem_base + (k16_local * Int32(full_chunks) + chunk)
                        * Int32(16),
                        get_ptr_as_int64(tr_u32, src),
                    )
                elif src >= Int64(0) and src <= limit:
                    cp_async4_shared_global(
                        smem_base + (k16_local * Int32(full_chunks) + chunk)
                        * Int32(16),
                        get_ptr_as_int64(tr_u32, src),
                    )

    @cute.jit
    def _stage_owned_sfb(
        self, scales: cute.Tensor, dst_base: Int32, tile_word_base: Int64,
        packed_half: Int32, tid: Int32, subtile: Int32,
    ):
        # Each owned N8 needs eight u32 scale words. The K128 word remains
        # intact: the consumer still selects its original K32 byte.
        owned_vectors = self.owned_n // 4
        for i in cutlass.range_constexpr(
            (owned_vectors + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(owned_vectors):
                vector = subtile * Int32(owned_vectors) + idx
                src_word = (
                    tile_word_base + Int64(packed_half * Int32(128))
                    + Int64(vector * Int32(4))
                )
                cp_async4_shared_global(
                    dst_base + vector * Int32(16),
                    get_ptr_as_int64(scales, src_word),
                )

    @cute.jit
    def _cache_route_tokens(self, token_map: cute.Tensor, source_m_tile: Int32, m_half: Int32, valid_rows: Int32, tid: Int32):
        count = ( (1 if self.p8_broadcast_a else self.tile_m) * 4 + self.threads_per_cta - 1) // self.threads_per_cta
        cached = cute.make_rmem_tensor((count,), Int32)
        base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(self.tile_m)
        for i in cutlass.range_constexpr(count):
            idx = tid + Int32(i * self.threads_per_cta)
            tok = Int32(0)
            if idx < Int32((1 if self.p8_broadcast_a else self.tile_m) * 4):
                row = idx >> Int32(2)
                if row < valid_rows:
                    tok = token_map[base + row].to(Int32)
                    if cutlass.const_expr(self.deterministic_output):
                        tok = tok // Int32(self.num_topk)
            cached[i] = tok
        scale_tok = Int32(0)
        if tid < Int32(1 if self.p8_broadcast_a else self.tile_m):
            if tid < valid_rows:
                scale_tok = token_map[base + tid].to(Int32)
                if cutlass.const_expr(self.deterministic_output):
                    scale_tok = scale_tok // Int32(self.num_topk)
        return cached, scale_tok

    @cute.jit
    def _stage_slice(self, packed_a_u32: cute.Tensor, scale_storage: cute.Tensor, w13_rp: cute.Tensor, w13_sfb_rp: cute.Tensor, token_map: cute.Tensor, cached_payload_tokens: cute.Tensor, cached_scale_token: Int32, smem_base: Int32, tid: Int32, source_m_tile: Int32, m_half: Int32, expert_idx: Int32, output_tile: Int32, valid_rows: Int32, k64_slice: Int32, input_k128_tiles: Int32, intermediate_tiles: Int32, packed_w13_tiles: Int32, subtile: Int32):
        stage = k64_slice % Int32(self.p8_pipeline_stages)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        gate_b_base = stage_base + Int32(self.gate_b_offset)
        up_b_base = stage_base + Int32(self.up_b_offset)
        gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
        up_sfb_base = stage_base + Int32(self.up_sfb_offset)
        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(self.tile_m)
        words_per_token = input_k128_tiles * Int32(32)
        for i in cutlass.range_constexpr(((1 if self.p8_broadcast_a else self.tile_m) * 4 + self.threads_per_cta - 1) // self.threads_per_cta):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32((1 if self.p8_broadcast_a else self.tile_m) * 4):
                row = idx >> Int32(2)
                vec = idx & Int32(3)
                tok = Int32(0)
                tok = cached_payload_tokens[i]
                physical_vec = vec ^ row & Int32(7)
                if cutlass.const_expr(self.p8_a_swizzle_rotate):
                    physical_vec = vec ^ ((row & Int32(3)) << Int32(1) | row >> Int32(2) & Int32(1))
                src_word = tok * words_per_token + k64_slice * Int32(16) + (vec << Int32(2))
                cp_async4_shared_global(a_base + row * Int32(128) + (physical_vec << Int32(4)), get_ptr_as_int64(packed_a_u32, src_word))
        if tid < Int32(1 if self.p8_broadcast_a else self.tile_m):
            tok = Int32(0)
            tok = cached_scale_token
            sf_src = tok * input_k128_tiles * Int32(4) + (k64_slice >> Int32(1)) * Int32(4)
            cp_async_u32_shared_global(sfa_base + (tid << Int32(2)), get_ptr_as_int64(scale_storage, sf_src))
        k128_slice = k64_slice >> Int32(1)
        k_half = k64_slice & Int32(1)
        input_k128_count = input_k128_tiles
        up_packed_tile = output_tile >> Int32(1)
        up_packed_half = output_tile & Int32(1)
        gate_tile = output_tile + intermediate_tiles
        gate_packed_tile = gate_tile >> Int32(1)
        gate_packed_half = gate_tile & Int32(1)
        up_tile = (expert_idx * packed_w13_tiles + up_packed_tile) * input_k128_count + k128_slice
        gate_tile_idx = (expert_idx * packed_w13_tiles + gate_packed_tile) * input_k128_count + k128_slice
        if cutlass.const_expr(self.w4a8_trellis):
            tr_n16_cnt = intermediate_tiles * Int32(8)
            tr_k16_stride = tr_n16_cnt * Int32(8 * self.trellis_bits)
            tr_eu = Int64(input_k128_tiles * Int32(8)) * Int64(tr_k16_stride)
            tr_w13_half = Int64(w13_rp.shape[0]) >> Int64(1)
            tr_common = Int64(expert_idx) * tr_eu + Int64(k64_slice * Int32(4)) * Int64(tr_k16_stride) + Int64(output_tile * Int32(8)) * Int64(8 * self.trellis_bits)
            self._stage_owned_trellis_b(w13_rp, gate_b_base, tr_common, tr_k16_stride, self.trellis_bits, tid, self.threads_per_cta, 4, subtile)
            self._stage_owned_trellis_b(w13_rp, up_b_base, tr_common + tr_w13_half, tr_k16_stride, self.trellis_bits, tid, self.threads_per_cta, 4, subtile)
            if cutlass.const_expr(self.trellis_scaled):
                self._stage_owned_sfb(w13_sfb_rp, gate_sfb_base, Int64(gate_tile_idx) * Int64(256), gate_packed_half, tid, subtile)
                self._stage_owned_sfb(w13_sfb_rp, up_sfb_base, Int64(up_tile) * Int64(256), up_packed_half, tid, subtile)

    @cute.jit
    def _activated_value(self, gate, up, alpha_value):
        # Match dynamic.py _gated_activation_value, including comparisons and
        # the post-activation BF16 store in _run_task.
        gate = alpha_value * gate
        up = alpha_value * up
        if gate > cutlass.Float32(10.0):
            gate = cutlass.Float32(10.0)
        if up > cutlass.Float32(10.0):
            up = cutlass.Float32(10.0)
        if up < cutlass.Float32(-10.0):
            up = cutlass.Float32(-10.0)
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=False)
        )
        return gate * sigmoid * up

    @cute.jit
    def _run_task(self, packed_a_u32: cute.Tensor, scale_storage: cute.Tensor, w13_rp: cute.Tensor, w13_sfb_rp: cute.Tensor, intermediate_u32: cute.Tensor, token_map: cute.Tensor, alpha: cute.Tensor, input_global_scale: cute.Tensor, trellis_lut: cute.Tensor, trellis_rotations: cute.Tensor, smem_base: Int32, tid: Int32, warp_idx: Int32, source_m_tile: Int32, m_half: Int32, expert_idx: Int32, output_tile: Int32, valid_rows: Int32, rows_capacity: Int32, input_k128_tiles: Int32, intermediate_tiles: Int32, packed_w13_tiles: Int32, subtile: Int32):
        cached_payload_tokens, cached_scale_token = self._cache_route_tokens(token_map, source_m_tile, m_half, valid_rows, tid)
        warp_idx = warp_idx + subtile * Int32(self.num_warps)
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        if cutlass.const_expr(self.w4a8_trellis):
            tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(lane, self.trellis_bits)
            trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))
            if cutlass.const_expr(self.trellis_direct_lut):
                trellis_lut_addr = trellis_lut.iterator.toint()
        self._stage_slice(packed_a_u32, scale_storage, w13_rp, w13_sfb_rp, token_map, cached_payload_tokens, cached_scale_token, smem_base, tid, source_m_tile, m_half, expert_idx, output_tile, valid_rows, Int32(0), input_k128_tiles, intermediate_tiles, packed_w13_tiles, subtile)
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.p8_pipeline_stages == 3):
            self._stage_slice(packed_a_u32, scale_storage, w13_rp, w13_sfb_rp, token_map, cached_payload_tokens, cached_scale_token, smem_base, tid, source_m_tile, m_half, expert_idx, output_tile, valid_rows, Int32(1), input_k128_tiles, intermediate_tiles, packed_w13_tiles, subtile)
            cute.arch.cp_async_commit_group()
        gate_acc = tuple((tuple((cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(self.p8_n8_per_warp))) for _blk in range(self.mma_m_blocks)))
        up_acc = tuple((tuple((cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(self.p8_n8_per_warp))) for _blk in range(self.mma_m_blocks)))
        for blk in cutlass.range_constexpr(self.mma_m_blocks):
            for nt in cutlass.range_constexpr(self.p8_n8_per_warp):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)
        input_k64_tiles = input_k128_tiles * Int32(2)
        k64_slice = Int32(0)
        while k64_slice < input_k64_tiles:
            stage = k64_slice % Int32(self.p8_pipeline_stages)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)
            gate_b_base = stage_base + Int32(self.gate_b_offset)
            up_b_base = stage_base + Int32(self.up_b_offset)
            gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
            up_sfb_base = stage_base + Int32(self.up_sfb_offset)
            next_slice = k64_slice + Int32(self.p8_pipeline_stages - 1)
            if next_slice < input_k64_tiles:
                self._stage_slice(packed_a_u32, scale_storage, w13_rp, w13_sfb_rp, token_map, cached_payload_tokens, cached_scale_token, smem_base, tid, source_m_tile, m_half, expert_idx, output_tile, valid_rows, next_slice, input_k128_tiles, intermediate_tiles, packed_w13_tiles, subtile)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.p8_pipeline_stages - 1)
            cute.arch.fence_proxy('async.shared', space='cta')
            cute.arch.sync_threads()
            scale_shift = Uint32(k64_slice & Int32(1)) * Uint32(16)
            asc = cute.make_rmem_tensor((self.mma_m_blocks,), Uint32)
            for blk in cutlass.range_constexpr(self.mma_m_blocks):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                if cutlass.const_expr(self.p8_broadcast_a):
                    sf_row = Int32(0)
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2))) >> scale_shift
            for kb in cutlass.range_constexpr(2):
                u_phys = Int32(kb * 2) + (c >> Int32(1)) ^ q
                if cutlass.const_expr(self.p8_a_swizzle_rotate):
                    u_phys = Int32(kb * 2) + (c >> Int32(1)) ^ ((q & Int32(3)) << Int32(1) | q >> Int32(2))
                if cutlass.const_expr(self.p8_broadcast_a):
                    u_phys = Int32(kb * 2) + (c >> Int32(1))
                a_frag = cute.make_rmem_tensor((self.mma_m_blocks, 4), Uint32)
                for blk in cutlass.range_constexpr(self.mma_m_blocks):
                    a_lo = a_base + Int32(blk * 16 * 128) + (q << Int32(7)) + (u_phys << Int32(4)) + ((c & Int32(1)) << Int32(3))
                    if cutlass.const_expr(self.p8_broadcast_a):
                        a_lo = a_base + (u_phys << Int32(4)) + ((c & Int32(1)) << Int32(3))
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    if cutlass.const_expr(self.p8_broadcast_a):
                        a1, a3 = (a0, a2)
                    else:
                        a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * 128))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3
                gate_b0 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)
                gate_b1 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)
                up_b0 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)
                up_b1 = cute.make_rmem_tensor((self.p8_n8_per_warp,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    for th in cutlass.range_constexpr(self.p8_n8_per_warp // 2):
                        tr_n16 = warp_idx * Int32(self.p8_n8_per_warp // 2) + Int32(th)
                        tr_b0 = (Int32(kb * 16) + tr_n16) * Int32(8 * self.trellis_bits)
                        g_lo0, g_lo1, g_hi0, g_hi1 = w4a8_trellis_pair_words_dispatch(gate_b_base, lane, tr_b0, tr_b0 + Int32(64 * self.trellis_bits), tr_ia, tr_ib, tr_s2, self.trellis_bits, trellis_lut_addr, not self.trellis_direct_lut and self.trellis_codebook != 'mcg', self.trellis_direct_lut)
                        gate_b0[th * 2] = g_lo0
                        gate_b1[th * 2] = g_lo1
                        gate_b0[th * 2 + 1] = g_hi0
                        gate_b1[th * 2 + 1] = g_hi1
                        u_lo0, u_lo1, u_hi0, u_hi1 = w4a8_trellis_pair_words_dispatch(up_b_base, lane, tr_b0, tr_b0 + Int32(64 * self.trellis_bits), tr_ia, tr_ib, tr_s2, self.trellis_bits, trellis_lut_addr, not self.trellis_direct_lut and self.trellis_codebook != 'mcg', self.trellis_direct_lut)
                        up_b0[th * 2] = u_lo0
                        up_b1[th * 2] = u_lo1
                        up_b0[th * 2 + 1] = u_hi0
                        up_b1[th * 2 + 1] = u_hi1
                else:
                    gw0, gw1, gw2, gw3 = ld_shared_v4_u32(gate_b_base + ((Int32(kb * 4) + warp_idx) * Int32(32) + lane << Int32(4)))
                    uw0, uw1, uw2, uw3 = ld_shared_v4_u32(up_b_base + ((Int32(kb * 4) + warp_idx) * Int32(32) + lane << Int32(4)))
                    gate_words = cute.make_rmem_tensor((4,), Uint32)
                    up_words = cute.make_rmem_tensor((4,), Uint32)
                    gate_words[0] = gw0
                    gate_words[1] = gw1
                    gate_words[2] = gw2
                    gate_words[3] = gw3
                    up_words[0] = uw0
                    up_words[1] = uw1
                    up_words[2] = uw2
                    up_words[3] = uw3
                    for nt in cutlass.range_constexpr(4):
                        wb0, wb1 = e2m1x8_to_qmma_e2m1x8(gate_words[nt])
                        gate_b0[nt] = wb0
                        gate_b1[nt] = wb1
                        wb0, wb1 = e2m1x8_to_qmma_e2m1x8(up_words[nt])
                        up_b0[nt] = wb0
                        up_b1[nt] = wb1
                for nt in cutlass.range_constexpr(self.p8_n8_per_warp):
                    n8 = warp_idx * Int32(self.p8_n8_per_warp) + Int32(nt)
                    gb0 = gate_b0[nt]
                    gb1 = gate_b1[nt]
                    ub0 = up_b0[nt]
                    ub1 = up_b1[nt]
                    gate_sfb = Uint32(2139062143)
                    up_sfb = Uint32(2139062143)
                    if cutlass.const_expr(not self.w4a8_trellis or self.trellis_scaled):
                        gate_sfb = ld_shared_u32(gate_sfb_base + (n8 * Int32(8) + q << Int32(2))) >> scale_shift
                        up_sfb = ld_shared_u32(up_sfb_base + (n8 * Int32(8) + q << Int32(2))) >> scale_shift
                    for blk in cutlass.range_constexpr(self.mma_m_blocks):
                        gate_fragment = gate_acc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e4m3(gate_fragment[0], gate_fragment[1], gate_fragment[2], gate_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], gb0, gb1, asc[blk], gate_sfb, bid_a=kb, bid_b=kb)
                        else:
                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e2m1(gate_fragment[0], gate_fragment[1], gate_fragment[2], gate_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], gb0, gb1, asc[blk], gate_sfb, bid_a=kb, bid_b=kb)
                        gate_fragment[0] = g0
                        gate_fragment[1] = g1
                        gate_fragment[2] = g2
                        gate_fragment[3] = g3
                        up_fragment = up_acc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e4m3(up_fragment[0], up_fragment[1], up_fragment[2], up_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], ub0, ub1, asc[blk], up_sfb, bid_a=kb, bid_b=kb)
                        else:
                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e2m1(up_fragment[0], up_fragment[1], up_fragment[2], up_fragment[3], a_frag[blk, 0], a_frag[blk, 1], a_frag[blk, 2], a_frag[blk, 3], ub0, ub1, asc[blk], up_sfb, bid_a=kb, bid_b=kb)
                        up_fragment[0] = u0
                        up_fragment[1] = u1
                        up_fragment[2] = u2
                        up_fragment[3] = u3
            cute.arch.sync_threads()
            k64_slice += Int32(1)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy('async.shared', space='cta')
        cute.arch.sync_threads()
        alpha_value = alpha[expert_idx].to(cutlass.Float32) * input_global_scale[expert_idx].to(cutlass.Float32)
        epilogue_base = smem_base
        projection_row_stride = Int32(self.tile_n)
        up_epilogue_base = epilogue_base + Int32(self.tile_m * self.tile_n * 2)
        if cutlass.const_expr(self.p8_row_interleaved):
            projection_row_stride = Int32(2 * self.tile_n)
            up_epilogue_base = epilogue_base + Int32(self.tile_n * 2)
        full_output_base = up_epilogue_base + Int32(self.tile_m * self.tile_n * 2)
        if cutlass.const_expr(self.p8_row_interleaved):
            full_output_base = epilogue_base
        col_base = warp_idx * Int32(8 * self.p8_n8_per_warp) + (c << Int32(1))
        for nt in cutlass.range_constexpr(self.p8_n8_per_warp):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(self.mma_m_blocks):
                gate_fragment = gate_acc[blk][nt]
                up_fragment = up_acc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                if cutlass.const_expr(self.scale_sandwich):
                    st_shared_u32(epilogue_base + (row_lo * projection_row_stride + col) * Int32(2), _p8_pack_f32x2_to_half2(alpha_value * gate_fragment[0], alpha_value * gate_fragment[1]))
                    st_shared_u32(epilogue_base + (row_hi * projection_row_stride + col) * Int32(2), _p8_pack_f32x2_to_half2(alpha_value * gate_fragment[2], alpha_value * gate_fragment[3]))
                    st_shared_u32(up_epilogue_base + (row_lo * projection_row_stride + col) * Int32(2), _p8_pack_f32x2_to_half2(alpha_value * up_fragment[0], alpha_value * up_fragment[1]))
                    st_shared_u32(up_epilogue_base + (row_hi * projection_row_stride + col) * Int32(2), _p8_pack_f32x2_to_half2(alpha_value * up_fragment[2], alpha_value * up_fragment[3]))
                else:
                    act0 = self._activated_value(gate_fragment[0], up_fragment[0], alpha_value)
                    act1 = self._activated_value(gate_fragment[1], up_fragment[1], alpha_value)
                    act2 = self._activated_value(gate_fragment[2], up_fragment[2], alpha_value)
                    act3 = self._activated_value(gate_fragment[3], up_fragment[3], alpha_value)
                    st_shared_u32(epilogue_base + (row_lo * projection_row_stride + col) * Int32(2), pack_f32x2_to_bfloat2(act0, act1))
                    st_shared_u32(epilogue_base + (row_hi * projection_row_stride + col) * Int32(2), pack_f32x2_to_bfloat2(act2, act3))
        cute.arch.sync_threads()
        if cutlass.const_expr(self.diagnostic_raw_fc1):
            capture_words_per_row = intermediate_tiles * Int32(32)
            if tid < Int32(64) and valid_rows > Int32(0):
                capture_row = source_m_tile * Int32(self.source_tile_m) + output_tile
                capture_base = capture_row * capture_words_per_row
                gate_pair = ld_shared_u32(epilogue_base + tid * Int32(4))
                up_pair = ld_shared_u32(up_epilogue_base + tid * Int32(4))
                intermediate_u32[capture_base + tid] = gate_pair
                intermediate_u32[capture_base + Int32(64) + tid] = up_pair
            if tid == Int32(0):
                trace_slot = source_m_tile * Int32(4) + output_tile
                trace_base = rows_capacity * capture_words_per_row
                trace_value = expert_idx & Int32(511) | (source_m_tile & Int32(7)) << Int32(9) | (output_tile & Int32(3)) << Int32(12)
                intermediate_u32[trace_base + trace_slot] = trace_value
                intermediate_u32[trace_base + Int32(256) + source_m_tile * Int32(8) + output_tile] = Int32(self.p8_intermediate)
                atomic_add_global_i32(get_ptr_as_int64(intermediate_u32, trace_base + Int32(32) + trace_slot), Int32(1))
            cute.arch.sync_threads()
            return
        if cutlass.const_expr(self.full_coupled):
            for row_group in cutlass.range_constexpr(self.owned_row_groups):
                hrow = warp_idx + Int32(row_group * (16 // self.p8_n8_per_warp))
                if hrow < valid_rows:
                    activated = cute.make_rmem_tensor((4,), cutlass.Float32)
                    for segment in cutlass.range_constexpr(2):
                        raw_idx = lane * Int32(4)
                        raw_values = cute.make_rmem_tensor((4,), cutlass.Float32)
                        for component in cutlass.range_constexpr(4):
                            index = raw_idx + Int32(component)
                            atom = index >> Int32(5)
                            within = index & Int32(31)
                            source_col = Int32(segment * 64) + (atom >> Int32(1)) * Int32(32) + within
                            source_addr = epilogue_base + (hrow * projection_row_stride + source_col) * Int32(2)
                            if atom & Int32(1) != Int32(0):
                                source_addr = up_epilogue_base + (hrow * projection_row_stride + source_col) * Int32(2)
                            raw_values[component] = _p8_ld_shared_f16_to_f32(source_addr)
                        p0, p1, p2, p3 = _w4a8_had128_quad(raw_values[0], raw_values[1], raw_values[2], raw_values[3], lane)
                        first = raw_idx >> Int32(5) & Int32(1)
                        second = raw_idx + Int32(1) >> Int32(5) & Int32(1)
                        third = raw_idx + Int32(2) >> Int32(5) & Int32(1)
                        fourth = raw_idx + Int32(3) >> Int32(5) & Int32(1)
                        base_col = output_tile * Int32(128) + Int32(segment * 64) + (raw_idx >> Int32(6)) * Int32(32) + (raw_idx & Int32(31))
                        p0 = self._scale_fc1_after_h128(p0, trellis_rotations, expert_idx, base_col, first)
                        p1 = self._scale_fc1_after_h128(p1, trellis_rotations, expert_idx, base_col + Int32(1), second)
                        p2 = self._scale_fc1_after_h128(p2, trellis_rotations, expert_idx, base_col + Int32(2), third)
                        p3 = self._scale_fc1_after_h128(p3, trellis_rotations, expert_idx, base_col + Int32(3), fourth)
                        p0, p1, p2, p3 = _w4a8_had128_quad(p0, p1, p2, p3, lane)
                        pre_base = output_tile * Int32(256) + Int32(segment * 128) + raw_idx
                        p0 *= self._coupled_sign(trellis_rotations, pre_base)
                        p1 *= self._coupled_sign(trellis_rotations, pre_base + Int32(1))
                        p2 *= self._coupled_sign(trellis_rotations, pre_base + Int32(2))
                        p3 *= self._coupled_sign(trellis_rotations, pre_base + Int32(3))
                        a0 = self._coupled_activation(p0, p1)
                        a1 = self._coupled_activation(p2, p3)
                        activation_col = Int32(segment * 64) + lane * Int32(2)
                        post_base = Int32(2 * self.p8_intermediate) + output_tile * Int32(128) + activation_col
                        a0 *= self._coupled_sign(trellis_rotations, post_base)
                        a1 *= self._coupled_sign(trellis_rotations, post_base + Int32(1))
                        activated[segment * 2] = a0
                        activated[segment * 2 + 1] = a1
                    hcol = lane * Int32(4)
                    src0 = (lane & Int32(15)) << Int32(1)
                    src1 = src0 + Int32(1)
                    first0 = cute.arch.shuffle_sync(activated[0], src0)
                    first1 = cute.arch.shuffle_sync(activated[1], src0)
                    first2 = cute.arch.shuffle_sync(activated[0], src1)
                    first3 = cute.arch.shuffle_sync(activated[1], src1)
                    second0 = cute.arch.shuffle_sync(activated[2], src0)
                    second1 = cute.arch.shuffle_sync(activated[3], src0)
                    second2 = cute.arch.shuffle_sync(activated[2], src1)
                    second3 = cute.arch.shuffle_sync(activated[3], src1)
                    a0 = first0
                    a1 = first1
                    a2 = first2
                    a3 = first3
                    if lane >= Int32(16):
                        a0 = second0
                        a1 = second1
                        a2 = second2
                        a3 = second3
                    a0, a1, a2, a3 = _w4a8_had128_quad(a0, a1, a2, a3, lane)
                    local_col = output_tile * Int32(128) + hcol
                    a0 *= trellis_rotations[Int32(4096) + expert_idx * Int32(3 * self.p8_intermediate) + Int32(2 * self.p8_intermediate) + local_col].to(cutlass.Float32)
                    a1 *= trellis_rotations[Int32(4096) + expert_idx * Int32(3 * self.p8_intermediate) + Int32(2 * self.p8_intermediate) + local_col + Int32(1)].to(cutlass.Float32)
                    a2 *= trellis_rotations[Int32(4096) + expert_idx * Int32(3 * self.p8_intermediate) + Int32(2 * self.p8_intermediate) + local_col + Int32(2)].to(cutlass.Float32)
                    a3 *= trellis_rotations[Int32(4096) + expert_idx * Int32(3 * self.p8_intermediate) + Int32(2 * self.p8_intermediate) + local_col + Int32(3)].to(cutlass.Float32)
                    a0, a1, a2, a3 = _w4a8_had128_quad(a0, a1, a2, a3, lane)
                    addr = full_output_base + (hrow * Int32(128) + hcol) * Int32(4)
                    st_shared_f32(addr, a0)
                    st_shared_f32(addr + Int32(4), a1)
                    st_shared_f32(addr + Int32(8), a2)
                    st_shared_f32(addr + Int32(12), a3)
            cute.arch.sync_threads()
        elif cutlass.const_expr(self.scale_sandwich):
            for row_group in cutlass.range_constexpr(self.owned_row_groups):
                hrow = warp_idx + Int32(row_group * 4)
                if hrow < valid_rows:
                    hcol = lane * Int32(4)
                    goff = epilogue_base + (hrow * Int32(128) + hcol) * Int32(2)
                    uoff = up_epilogue_base + (hrow * Int32(128) + hcol) * Int32(2)
                    g0 = _p8_ld_shared_f16_to_f32(goff)
                    g1 = _p8_ld_shared_f16_to_f32(goff + Int32(2))
                    g2 = _p8_ld_shared_f16_to_f32(goff + Int32(4))
                    g3 = _p8_ld_shared_f16_to_f32(goff + Int32(6))
                    u0 = _p8_ld_shared_f16_to_f32(uoff)
                    u1 = _p8_ld_shared_f16_to_f32(uoff + Int32(2))
                    u2 = _p8_ld_shared_f16_to_f32(uoff + Int32(4))
                    u3 = _p8_ld_shared_f16_to_f32(uoff + Int32(6))
                    g0, g1, g2, g3 = _w4a8_had128_quad(g0, g1, g2, g3, lane)
                    u0, u1, u2, u3 = _w4a8_had128_quad(u0, u1, u2, u3, lane)
                    local_col = output_tile * Int32(128) + hcol
                    g0 = self._scale_fc1_after_h128(g0, trellis_rotations, expert_idx, local_col, Int32(0))
                    g1 = self._scale_fc1_after_h128(g1, trellis_rotations, expert_idx, local_col + Int32(1), Int32(0))
                    g2 = self._scale_fc1_after_h128(g2, trellis_rotations, expert_idx, local_col + Int32(2), Int32(0))
                    g3 = self._scale_fc1_after_h128(g3, trellis_rotations, expert_idx, local_col + Int32(3), Int32(0))
                    u0 = self._scale_fc1_after_h128(u0, trellis_rotations, expert_idx, local_col, Int32(1))
                    u1 = self._scale_fc1_after_h128(u1, trellis_rotations, expert_idx, local_col + Int32(1), Int32(1))
                    u2 = self._scale_fc1_after_h128(u2, trellis_rotations, expert_idx, local_col + Int32(2), Int32(1))
                    u3 = self._scale_fc1_after_h128(u3, trellis_rotations, expert_idx, local_col + Int32(3), Int32(1))
                    g0 = cutlass.Float16(g0).to(cutlass.Float32)
                    g1 = cutlass.Float16(g1).to(cutlass.Float32)
                    g2 = cutlass.Float16(g2).to(cutlass.Float32)
                    g3 = cutlass.Float16(g3).to(cutlass.Float32)
                    u0 = cutlass.Float16(u0).to(cutlass.Float32)
                    u1 = cutlass.Float16(u1).to(cutlass.Float32)
                    u2 = cutlass.Float16(u2).to(cutlass.Float32)
                    u3 = cutlass.Float16(u3).to(cutlass.Float32)
                    one = cutlass.Float32(1.0)
                    a0 = self._activated_value(g0, u0, one)
                    a1 = self._activated_value(g1, u1, one)
                    a2 = self._activated_value(g2, u2, one)
                    a3 = self._activated_value(g3, u3, one)
                    a0 = cutlass.Float16(a0).to(cutlass.Float32)
                    a1 = cutlass.Float16(a1).to(cutlass.Float32)
                    a2 = cutlass.Float16(a2).to(cutlass.Float32)
                    a3 = cutlass.Float16(a3).to(cutlass.Float32)
                    a0 = self._scale_down_before_h128(a0, trellis_rotations, expert_idx, local_col)
                    a1 = self._scale_down_before_h128(a1, trellis_rotations, expert_idx, local_col + Int32(1))
                    a2 = self._scale_down_before_h128(a2, trellis_rotations, expert_idx, local_col + Int32(2))
                    a3 = self._scale_down_before_h128(a3, trellis_rotations, expert_idx, local_col + Int32(3))
                    a0, a1, a2, a3 = _w4a8_had128_quad(a0, a1, a2, a3, lane)
                    st_shared_u32(goff, _p8_pack_f32x2_to_half2(a0, a1))
                    st_shared_u32(goff + Int32(4), _p8_pack_f32x2_to_half2(a2, a3))
            cute.arch.sync_threads()
        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(self.tile_m)
        words_per_row = intermediate_tiles * Int32(32)
        if cutlass.const_expr(self.p8_warp_quant):
            block = lane >> Int32(3)
            word = lane & Int32(7)
            for row_group in cutlass.range_constexpr(self.owned_row_groups):
                row = warp_idx + Int32(row_group * self.num_warps)
                if row < valid_rows:
                    values4 = cute.make_rmem_tensor((4,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    for elem in cutlass.range_constexpr(4):
                        i = word * Int32(4) + Int32(elem)
                        pi = i & Int32(1) | i >> Int32(2) & Int32(2) | i & Int32(4) ^ i >> Int32(2) & Int32(4) | (i & Int32(2)) << Int32(2) | i & Int32(16)
                        value = ld_shared_f32(full_output_base + (row * Int32(128) + block * Int32(32) + pi) * Int32(4))
                        values4[elem] = value
                        magnitude = fabs_f32(value)
                        if magnitude > block_max:
                            block_max = magnitude
                    for offset in cutlass.range_constexpr(3):
                        other = cute.arch.shuffle_sync_bfly(block_max, offset=1 << offset)
                        if other > block_max:
                            block_max = other
                    _, scale_byte = pow2_ceil_ue8m0(block_max * cutlass.Float32(1.0 / 448.0))
                    inv_scale = ue8m0_to_output_scale(scale_byte)
                    payload = cvt_f32x4_to_e4m3x4(values4[0] * inv_scale, values4[1] * inv_scale, values4[2] * inv_scale, values4[3] * inv_scale)
                    dst = (physical_row_base + row) * words_per_row + output_tile * Int32(32) + block * Int32(8) + word
                    intermediate_u32[dst] = payload
                    if word == Int32(0):
                        scale_bytes = cute.recast_tensor(intermediate_u32, cutlass.Uint8)
                        sf_base = rows_capacity * words_per_row
                        scale_bytes[(sf_base + output_tile * rows_capacity + physical_row_base + row) * Int32(4) + block] = cutlass.Uint8(scale_byte & Uint32(255))
        elif tid < valid_rows:
            scale_word = Uint32(0)
            for local_block in cutlass.range_constexpr(self.owned_n // 32):
                block = Int32(local_block) + subtile * Int32(self.owned_n // 32)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem in cutlass.range_constexpr(32):
                    element_index = tid * Int32(self.tile_n) + Int32(block * 32 + elem)
                    value_addr = epilogue_base + element_index * Int32(2)
                    if cutlass.const_expr(self.full_coupled):
                        value_addr = full_output_base + element_index * Int32(4)
                    value = ld_shared_bf16_to_f32(value_addr)
                    if cutlass.const_expr(self.full_coupled):
                        value = ld_shared_f32(value_addr)
                    elif cutlass.const_expr(self.scale_sandwich):
                        value = _p8_ld_shared_f16_to_f32(value_addr)
                    values[elem] = value
                    abs_value = fabs_f32(value)
                    if abs_value > block_max:
                        block_max = abs_value
                if cutlass.const_expr(self.w4a8_trellis):
                    payload, scale_byte = quantize_block_fp8_mx(_w4a8_trellis_permute_k32(values), block_max)
                else:
                    payload, scale_byte = quantize_block_fp8_mx(values, block_max)
                dst_word = (physical_row_base + tid) * words_per_row + output_tile * Int32(32) + Int32(block * 8)
                for word in cutlass.range_constexpr(8):
                    intermediate_u32[dst_word + Int32(word)] = payload[word]
                sf_base = rows_capacity * words_per_row
                scale_bytes = cute.recast_tensor(intermediate_u32, cutlass.Uint8)
                scale_bytes[(sf_base + output_tile * rows_capacity + physical_row_base + tid) * Int32(4) + block] = cutlass.Uint8(scale_byte & Uint32(255))
        cute.arch.sync_threads()


    @cute.kernel
    def kernel(
        self, packed_a_u32: cute.Tensor, scale_storage: cute.Tensor,
        w13_rp: cute.Tensor, w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor, token_map: cute.Tensor,
        task_expert: cute.Tensor, task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor, alpha: cute.Tensor,
        input_global_scale: cute.Tensor, trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor, input_k128_tiles: Int32,
        intermediate_tiles: Int32, packed_w13_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        _, _, gdimz = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        rows_capacity = Int32(token_map.shape[0])
        tasks_per_route = intermediate_tiles * Int32(self.subtiles)
        task = Int32(bidz)
        # task_expert is the flattened top-k routing tensor. MTP verification
        # needs all token routes, not just the first token's eight routes.
        while task < Int32(task_expert.shape[0]) * tasks_per_route:
            if cutlass.const_expr(self.p8_tile_major):
                route = task % Int32(task_expert.shape[0])
                tile = task // Int32(task_expert.shape[0])
            else:
                route = task // tasks_per_route
                tile = task % tasks_per_route
            output_tile = tile // Int32(self.subtiles)
            subtile = tile % Int32(self.subtiles)
            expert = task_expert[route].to(Int32)
            if expert >= Int32(0) and expert < Int32(288):
                self._run_task(
                    packed_a_u32, scale_storage, w13_rp, w13_sfb_rp,
                    intermediate_u32, token_map, alpha, input_global_scale,
                    trellis_lut, trellis_rotations, smem_base, tid, warp_idx,
                    route, Int32(0), expert, output_tile, Int32(1),
                    rows_capacity, input_k128_tiles, intermediate_tiles,
                    packed_w13_tiles, subtile,
                )
            task += Int32(gdimz)


class P8H128FC1Kernel(P8H128NarrowFC1Kernel):
    """One-CTA N128 owner for the H128 scale-sandwich boundary."""

    def __init__(self, *, full_coupled: bool = False, trellis_bits: int = 4):
        if int(trellis_bits) not in (3, 4, 5):
            raise ValueError("P8 H128 FC1 supports K3, K4 or K5 streams")
        self.trellis_bits = int(trellis_bits)
        self.b_payload_bytes = 1024 * self.trellis_bits
        self.scale_sandwich = True
        self.full_coupled = bool(full_coupled)
        self.owned_n = 128
        self.num_warps = 4
        self.threads_per_cta = 128
        self.subtiles = 1
        self.a_payload_bytes = self.tile_m * 128
        self.a_scale_bytes = self.tile_m * 4
        self.sfa_offset = self.a_payload_bytes
        self.gate_b_offset = self.sfa_offset + self.a_scale_bytes
        self.up_b_offset = self.gate_b_offset + self.b_payload_bytes
        self.gate_sfb_offset = self.up_b_offset + self.b_payload_bytes
        self.up_sfb_offset = self.gate_sfb_offset + self.sfb_bytes
        self.stage_bytes = self.up_sfb_offset + self.sfb_bytes
        self.shared_bytes = max(
            2 * self.stage_bytes,
            # Full coupling keeps gate FP16, up FP16 and transformed FP32
            # disjoint. An in-place FP32 row spans two physical FP16 rows and
            # races neighboring warps that have not consumed them yet.
            self.tile_m * 128 * (8 if self.full_coupled else 4),
        )
        self.shared_words = (self.shared_bytes + 3) // 4
        self.fast_math = False
        self.source_halves = 1
        self.deterministic_output = True
        self.num_topk = 8
        self.is_situ = False
        self.w4a8_trellis = True
        self.trellis_coupled = False
        self.trellis_direct_lut = False
        self.trellis_codebook = "mcg"
        self.trellis_scaled = True
        self.trellis_identity_boundary = not self.full_coupled
        self.trellis_lut_offset = self.shared_bytes

    @cute.jit
    def _coupled_sign(
        self,
        scale_component: cute.Tensor,
        sign_idx: Int32,
    ) -> cutlass.Float32:
        # Packed scales occupy H + E*3I + H FP16 elements. Fixed signs are
        # shared by every expert on this TP rank and occupy pre[2I]|post[I].
        sign_base = Int32(4096 + 288 * 3 * self.p8_intermediate + 4096)
        return scale_component[sign_base + sign_idx].to(cutlass.Float32)

    @cute.jit
    def _coupled_activation(self, gate, up):
        # GLM-5.3 Flash target activation: capped SiLU, not the archived
        # synthetic SiTU fixture. The draw-0 coupled candidate still owns the
        # joint H128 boundary around this target nonlinearity.
        if gate > cutlass.Float32(10.0):
            gate = cutlass.Float32(10.0)
        if up > cutlass.Float32(10.0):
            up = cutlass.Float32(10.0)
        if up < cutlass.Float32(-10.0):
            up = cutlass.Float32(-10.0)
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=False)
        )
        return gate * sigmoid * up


class P8H128FC1RawCaptureKernel(P8H128FC1Kernel):
    """Diagnostic-only M1 owner that stops after raw FP16 gate/up capture."""

    diagnostic_raw_fc1 = True

    def __init__(self):
        super().__init__(full_coupled=True)
