"""Opt-in P8 M1 narrow FC1, derived from pinned B12X phase1.
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

from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    e2m1x8_to_qmma_e2m1x8,
    fabs_f32,
    get_ptr_as_int64,
    ld_shared_bf16_to_f32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_shared_u32,
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


class P8NarrowFC1Kernel(W4A8MaterializedPhase1Kernel):
    tile_m = 16
    source_tile_m = 16

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
        down_svh[H]``.  This helper intentionally does not perform H128: the
        N64 path needs an explicit cross-CTA owner for the adjacent half first.
        """

        hidden = Int32(4096)
        intermediate = Int32(512)
        scale_idx = (
            hidden
            + expert_idx * Int32(3 * 512)
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
            + expert_idx * Int32(3 * 512)
            + Int32(2 * 512)
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
                if src >= Int64(0) and src <= limit:
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
    def _stage_slice(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        token_map: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        k64_slice: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        packed_w13_tiles: Int32,
        subtile: Int32,
    ):
        stage = k64_slice & Int32(1)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        gate_b_base = stage_base + Int32(self.gate_b_offset)
        up_b_base = stage_base + Int32(self.up_b_offset)
        gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
        up_sfb_base = stage_base + Int32(self.up_sfb_offset)

        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        words_per_token = input_k128_tiles * Int32(32)

        # Gather the shared input representation through the published route
        # map.  Invalid tail rows may read token zero safely; no output is
        # published for them.
        for i in cutlass.range_constexpr(
            (self.tile_m * 4 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 4):
                row = idx >> Int32(2)
                vec = idx & Int32(3)
                tok = Int32(0)
                if row < valid_rows:
                    tok = token_map[physical_row_base + row].to(Int32)
                    if cutlass.const_expr(self.deterministic_output):
                        tok = tok // Int32(self.num_topk)
                physical_vec = vec ^ (row & Int32(7))
                src_word = (
                    tok * words_per_token + k64_slice * Int32(16) + (vec << Int32(2))
                )
                cp_async4_shared_global(
                    a_base + row * Int32(128) + (physical_vec << Int32(4)),
                    get_ptr_as_int64(packed_a_u32, src_word),
                )

        if tid < Int32(self.tile_m):
            tok = Int32(0)
            if tid < valid_rows:
                tok = token_map[physical_row_base + tid].to(Int32)
                if cutlass.const_expr(self.deterministic_output):
                    tok = tok // Int32(self.num_topk)
            sf_src = tok * input_k128_tiles * Int32(4) + (
                k64_slice >> Int32(1)
            ) * Int32(4)
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(scale_storage, sf_src),
            )

        k128_slice = k64_slice >> Int32(1)
        k_half = k64_slice & Int32(1)
        input_k128_count = input_k128_tiles

        up_packed_tile = output_tile >> Int32(1)
        up_packed_half = output_tile & Int32(1)
        gate_tile = output_tile + intermediate_tiles
        gate_packed_tile = gate_tile >> Int32(1)
        gate_packed_half = gate_tile & Int32(1)

        up_tile = (
            expert_idx * packed_w13_tiles + up_packed_tile
        ) * input_k128_count + k128_slice
        gate_tile_idx = (
            expert_idx * packed_w13_tiles + gate_packed_tile
        ) * input_k128_count + k128_slice

        if cutlass.const_expr(self.w4a8_trellis):
            # Projection-major [proj][E][K16][N16] trellis windows (the
            # prepared QSRT layout, shared with the micro kernel); stage
            # the four K16 rows of this K64 epoch for gate (projection 0)
            # and up (projection 1). Only the owned N64/N32 spans are copied;
            # their original N128 shared-memory offsets remain unchanged.
            tr_n16_cnt = intermediate_tiles * Int32(8)
            tr_k16_stride = tr_n16_cnt * Int32(8 * self.trellis_bits)
            tr_eu = Int64(input_k128_tiles * Int32(8)) * Int64(tr_k16_stride)
            tr_w13_half = Int64(w13_rp.shape[0]) >> Int64(1)
            tr_common = (
                Int64(expert_idx) * tr_eu
                + Int64(k64_slice * Int32(4)) * Int64(tr_k16_stride)
                + Int64(output_tile * Int32(8))
                * Int64(8 * self.trellis_bits)
            )
            self._stage_owned_trellis_b(
                w13_rp,
                gate_b_base,
                tr_common,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
                subtile,
            )
            self._stage_owned_trellis_b(
                w13_rp,
                up_b_base,
                tr_common + tr_w13_half,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
                subtile,
            )
            if cutlass.const_expr(self.trellis_scaled):
                self._stage_owned_sfb(
                    w13_sfb_rp,
                    gate_sfb_base,
                    Int64(gate_tile_idx) * Int64(256),
                    gate_packed_half,
                    tid,
                    subtile,
                )
                self._stage_owned_sfb(
                    w13_sfb_rp,
                    up_sfb_base,
                    Int64(up_tile) * Int64(256),
                    up_packed_half,
                    tid,
                    subtile,
                )

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
    def _run_task(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        rows_capacity: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        packed_w13_tiles: Int32,
        subtile: Int32,
    ):
        warp_idx = warp_idx + subtile * Int32(self.num_warps)
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        if cutlass.const_expr(self.w4a8_trellis):
            tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(
                lane, self.trellis_bits
            )
            trellis_lut_addr = Int64(
                smem_base + Int32(self.trellis_lut_offset)
            )
            if cutlass.const_expr(self.trellis_direct_lut):
                trellis_lut_addr = trellis_lut.iterator.toint()

        self._stage_slice(
            packed_a_u32,
            scale_storage,
            w13_rp,
            w13_sfb_rp,
            token_map,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            valid_rows,
            Int32(0),
            input_k128_tiles,
            intermediate_tiles,
            packed_w13_tiles,
            subtile,
        )
        cute.arch.cp_async_commit_group()

        # Keep each MMA's four accumulator registers as an independent
        # fragment.  A single 4x4x4 mutable tensor makes the 4.6 lowering pack
        # and unpack the complete live accumulator set around loop-carried
        # values even though every MMA consumes exactly four adjacent values.
        gate_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(1)
        )
        up_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(1)
        )
        for blk in cutlass.range_constexpr(1):
            for nt in cutlass.range_constexpr(4):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)

        input_k64_tiles = input_k128_tiles * Int32(2)
        k64_slice = Int32(0)
        while k64_slice < input_k64_tiles:
            stage = k64_slice & Int32(1)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)
            gate_b_base = stage_base + Int32(self.gate_b_offset)
            up_b_base = stage_base + Int32(self.up_b_offset)
            gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
            up_sfb_base = stage_base + Int32(self.up_sfb_offset)

            next_slice = k64_slice + Int32(1)
            if next_slice < input_k64_tiles:
                self._stage_slice(
                    packed_a_u32,
                    scale_storage,
                    w13_rp,
                    w13_sfb_rp,
                    token_map,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    valid_rows,
                    next_slice,
                    input_k128_tiles,
                    intermediate_tiles,
                    packed_w13_tiles,
                    subtile,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            scale_shift = Uint32(k64_slice & Int32(1)) * Uint32(16)
            asc = cute.make_rmem_tensor((1,), Uint32)
            for blk in cutlass.range_constexpr(1):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2))) >> scale_shift

            for kb in cutlass.range_constexpr(2):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((1, 4), Uint32)
                for blk in cutlass.range_constexpr(1):
                    a_lo = (
                        a_base
                        + Int32(blk * 16 * 128)
                        + (q << Int32(7))
                        + (u_phys << Int32(4))
                        + ((c & Int32(1)) << Int32(3))
                    )
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * 128))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3

                gate_b0 = cute.make_rmem_tensor((4,), Uint32)
                gate_b1 = cute.make_rmem_tensor((4,), Uint32)
                up_b0 = cute.make_rmem_tensor((4,), Uint32)
                up_b1 = cute.make_rmem_tensor((4,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    # Each warp owns adjacent n8 fragments: decode both
                    # halves of its two N16 tiles per K32 with no waste.
                    for th in cutlass.range_constexpr(2):
                        tr_n16 = warp_idx * Int32(2) + Int32(th)
                        tr_b0 = (Int32(kb * 16) + tr_n16) * Int32(
                            8 * self.trellis_bits
                        )
                        g_lo0, g_lo1, g_hi0, g_hi1 = (
                            w4a8_trellis_pair_words_dispatch(
                                gate_b_base,
                                lane,
                                tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia,
                                tr_ib,
                                tr_s2,
                                self.trellis_bits,
                                trellis_lut_addr,
                                not self.trellis_direct_lut
                                and self.trellis_codebook != "mcg",
                                self.trellis_direct_lut,
                            )
                        )
                        gate_b0[th * 2] = g_lo0
                        gate_b1[th * 2] = g_lo1
                        gate_b0[th * 2 + 1] = g_hi0
                        gate_b1[th * 2 + 1] = g_hi1
                        u_lo0, u_lo1, u_hi0, u_hi1 = (
                            w4a8_trellis_pair_words_dispatch(
                                up_b_base,
                                lane,
                                tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia,
                                tr_ib,
                                tr_s2,
                                self.trellis_bits,
                                trellis_lut_addr,
                                not self.trellis_direct_lut
                                and self.trellis_codebook != "mcg",
                                self.trellis_direct_lut,
                            )
                        )
                        up_b0[th * 2] = u_lo0
                        up_b1[th * 2] = u_lo1
                        up_b0[th * 2 + 1] = u_hi0
                        up_b1[th * 2 + 1] = u_hi1
                else:
                    gw0, gw1, gw2, gw3 = ld_shared_v4_u32(
                        gate_b_base
                        + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                    )
                    uw0, uw1, uw2, uw3 = ld_shared_v4_u32(
                        up_b_base
                        + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                    )
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

                for nt in cutlass.range_constexpr(4):
                    n8 = warp_idx * Int32(4) + Int32(nt)
                    gb0 = gate_b0[nt]
                    gb1 = gate_b1[nt]
                    ub0 = up_b0[nt]
                    ub1 = up_b1[nt]
                    gate_sfb = Uint32(0x7F7F7F7F)
                    up_sfb = Uint32(0x7F7F7F7F)
                    if cutlass.const_expr(
                        not self.w4a8_trellis or self.trellis_scaled
                    ):
                        gate_sfb = (
                            ld_shared_u32(
                                gate_sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                            )
                            >> scale_shift
                        )
                        up_sfb = (
                            ld_shared_u32(
                                up_sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                            )
                            >> scale_shift
                        )
                    for blk in cutlass.range_constexpr(1):
                        gate_fragment = gate_acc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                gate_fragment[0],
                                gate_fragment[1],
                                gate_fragment[2],
                                gate_fragment[3],
                                a_frag[blk, 0],
                                a_frag[blk, 1],
                                a_frag[blk, 2],
                                a_frag[blk, 3],
                                gb0,
                                gb1,
                                asc[blk],
                                gate_sfb,
                                bid_a=kb,
                                bid_b=kb,
                            )
                        else:
                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                gate_fragment[0],
                            gate_fragment[1],
                            gate_fragment[2],
                            gate_fragment[3],
                            a_frag[blk, 0],
                            a_frag[blk, 1],
                            a_frag[blk, 2],
                            a_frag[blk, 3],
                            gb0,
                            gb1,
                            asc[blk],
                            gate_sfb,
                            bid_a=kb,
                            bid_b=kb,
                        )
                        gate_fragment[0] = g0
                        gate_fragment[1] = g1
                        gate_fragment[2] = g2
                        gate_fragment[3] = g3
                        up_fragment = up_acc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                up_fragment[0],
                                up_fragment[1],
                                up_fragment[2],
                                up_fragment[3],
                                a_frag[blk, 0],
                                a_frag[blk, 1],
                                a_frag[blk, 2],
                                a_frag[blk, 3],
                                ub0,
                                ub1,
                                asc[blk],
                                up_sfb,
                                bid_a=kb,
                                bid_b=kb,
                            )
                        else:
                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                up_fragment[0],
                            up_fragment[1],
                            up_fragment[2],
                            up_fragment[3],
                            a_frag[blk, 0],
                            a_frag[blk, 1],
                            a_frag[blk, 2],
                            a_frag[blk, 3],
                            ub0,
                            ub1,
                            asc[blk],
                            up_sfb,
                            bid_a=kb,
                            bid_b=kb,
                        )
                        up_fragment[0] = u0
                        up_fragment[1] = u1
                        up_fragment[2] = u2
                        up_fragment[3] = u3

            cute.arch.sync_threads()
            k64_slice += Int32(1)

        # The pipeline region aliases the activation staging tile below.  A
        # wait-group(1) is sufficient while consuming alternating stages, but
        # the final (possibly empty) committed group must be fully retired
        # before those shared addresses are repurposed by ordinary stores.
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        alpha_value = alpha[expert_idx].to(cutlass.Float32) * input_global_scale[
            expert_idx
        ].to(cutlass.Float32)
        epilogue_base = smem_base
        col_base = warp_idx * Int32(32) + (c << Int32(1))
        # Keep gate/up FP32 through clipped SwiGLU. Round only its result.
        for nt in cutlass.range_constexpr(4):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(1):
                gate_fragment = gate_acc[blk][nt]
                up_fragment = up_acc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                act0 = self._activated_value(
                    gate_fragment[0], up_fragment[0], alpha_value
                )
                act1 = self._activated_value(
                    gate_fragment[1], up_fragment[1], alpha_value
                )
                act2 = self._activated_value(
                    gate_fragment[2], up_fragment[2], alpha_value
                )
                act3 = self._activated_value(
                    gate_fragment[3], up_fragment[3], alpha_value
                )
                st_shared_u32(
                    epilogue_base + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                    pack_f32x2_to_bfloat2(act0, act1),
                )
                st_shared_u32(
                    epilogue_base + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                    pack_f32x2_to_bfloat2(act2, act3),
                )

        cute.arch.sync_threads()

        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        words_per_row = intermediate_tiles * Int32(32)
        if tid < valid_rows:
            scale_word = Uint32(0)
            for local_block in cutlass.range_constexpr(self.num_warps):
                block = Int32(local_block) + subtile * Int32(self.num_warps)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem in cutlass.range_constexpr(32):
                    value = ld_shared_bf16_to_f32(
                        epilogue_base
                        + (tid * Int32(self.tile_n) + Int32(block * 32 + elem))
                        * Int32(2)
                    )
                    values[elem] = value
                    abs_value = fabs_f32(value)
                    if abs_value > block_max:
                        block_max = abs_value
                if cutlass.const_expr(self.w4a8_trellis):
                    payload, scale_byte = quantize_block_fp8_mx(
                        _w4a8_trellis_permute_k32(values), block_max
                    )
                else:
                    payload, scale_byte = quantize_block_fp8_mx(
                        values, block_max
                    )
                dst_word = (
                    (physical_row_base + tid) * words_per_row
                    + output_tile * Int32(32)
                    + Int32(block * 8)
                )
                for word in cutlass.range_constexpr(8):
                    intermediate_u32[dst_word + Int32(word)] = payload[word]
                # Different narrow CTAs own different bytes of one K128 scale
                # word. Byte stores prevent a read/modify/write race.
                sf_base = rows_capacity * words_per_row
                scale_bytes = cute.recast_tensor(intermediate_u32, cutlass.Uint8)
                scale_bytes[
                    (sf_base + output_tile * rows_capacity + physical_row_base + tid)
                    * Int32(4) + block
                ] = cutlass.Uint8(scale_byte & Uint32(0xFF))

        # Only the first 64 threads perform the row-wise quantize/store.  The
        # other warps must not advance the persistent task loop and overwrite
        # the aliased shared activation tile while those threads still read it.
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
        while task < Int32(8) * tasks_per_route:
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
