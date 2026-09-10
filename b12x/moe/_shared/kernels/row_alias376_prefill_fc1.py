"""Exact full-coupled P8 M64/N128 FC1 for materialized prefill.

This specializes the CPU/static-validated N128 owner used by decode.  It
changes only M ownership and grouped-task traversal: four M16 MMA fragments
cover an M64 tile, while every H128 transform remains inside one CTA.  The
procedural-MCG stream, E4M3/UE8M0/32 operands and native mxf8f6f4 MMA are
unchanged. The stored rate comes from ``trellis_bits`` and controls B-operand
staging. The decode reference is ``p8_h128_fc1.py:P8H128FC1Kernel``; its numerical
boundary is specified by ``trellismx/p8_coupled_scales.py``. Decode-only tests do
not qualify this grouped prefill path.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32

from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x.moe._shared.kernels.row_alias376_fc1 import P8H128FC1Kernel


class P8CoupledPrefillFC1Kernel(P8H128FC1Kernel):
    """One M64/N128 CTA with exact draw0/capped-SiLU coupled boundaries."""

    p8_row_interleaved = True
    tile_m = 64
    source_tile_m = 64
    mma_m_blocks = 4
    owned_row_groups = 16

    def __init__(self, *, trellis_bits: int = 4) -> None:
        if int(trellis_bits) not in (3, 4, 5):
            raise ValueError("coupled P8 prefill FC1 supports K3, K4 or K5 streams")
        super().__init__(full_coupled=True, trellis_bits=int(trellis_bits))
        # GateFP16+upFP16 and transformedFP32 each occupy512 bytes/row.
        # Their lifetimes are disjoint within a single warp-owned row.
        self.shared_bytes = max(2 * self.stage_bytes, self.tile_m * 128 * 4)
        self.shared_words = (self.shared_bytes + 3) // 4
        self.trellis_lut_offset = self.shared_bytes
        if (self.tile_m, self.owned_n, self.source_tile_m) != (64, 128, 64):
            raise ValueError("coupled P8 prefill FC1 requires exact M64/N128")

    @cute.jit
    def __call__(
        self,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_w13_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """FC1-only376-CTA grid for188-SM target; other phase grids unchanged."""

        self.kernel(
            cute.recast_tensor(packed_a_storage, cutlass.Uint32),
            scale_storage,
            w13_rp,
            w13_sfb_rp,
            intermediate_u32,
            token_map,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            alpha,
            input_global_scale,
            trellis_lut,
            scale_component,
            input_k128_tiles,
            intermediate_tiles,
            packed_w13_tiles,
        ).launch(
            grid=(1, 1, 376),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_w13_tiles: cutlass.Int32,
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
        num_experts = Int32(expert_tile_base.shape[0] - 1)
        source_m_tiles = expert_tile_base[num_experts].to(Int32)
        task_tail = source_m_tiles * intermediate_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % intermediate_tiles
            source_m_tile = task_slot // intermediate_tiles
            phase1_meta = source_m_tile * intermediate_tiles
            expert_idx = task_expert[phase1_meta].to(Int32)
            valid_rows = task_valid_rows[phase1_meta].to(Int32)
            if valid_rows > Int32(self.tile_m):
                valid_rows = Int32(self.tile_m)
            if valid_rows > Int32(0):
                self._run_task(
                    packed_a_u32,
                    scale_storage,
                    w13_rp,
                    w13_sfb_rp,
                    intermediate_u32,
                    token_map,
                    alpha,
                    input_global_scale,
                    trellis_lut,
                    scale_component,
                    smem_base,
                    tid,
                    warp_idx,
                    source_m_tile,
                    Int32(0),
                    expert_idx,
                    output_tile,
                    valid_rows,
                    rows_capacity,
                    input_k128_tiles,
                    intermediate_tiles,
                    packed_w13_tiles,
                    Int32(0),
                )
            task_slot += Int32(gdimz)


__all__ = ["P8CoupledPrefillFC1Kernel"]
