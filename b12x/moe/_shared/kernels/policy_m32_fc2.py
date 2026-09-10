"""Exact full-coupled P8 M32/N128 FC2 for materialized prefill.

The kernel retains the native K4 procedural-MCG -> E4M3/UE8M0/32 MMA path,
keeps the ordered K=512 result in FP32, applies the physical FP16 boundary,
then owns H128 plus shared down_svh inside one CTA.  It scatters unweighted
FP32 route values by the deterministic pair index for the existing top-k/H512
reducer.  Device closure remains a separate acceptance gate.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32

from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x.moe._shared.kernels.p8_small_m import P8SmallMPhase2Kernel


class P8CoupledPrefillFC2Kernel(P8SmallMPhase2Kernel):
    """One grouped M32/N128 FC2 owner with deterministic FP32 scatter."""

    tile_m = 32
    source_tile_m = 32
    mma_m_blocks = 2
    owned_row_groups = 8
    prefill_grouped = True

    a_payload_bytes = tile_m * 128
    a_scale_bytes = tile_m * 4
    a_stage_bytes = a_payload_bytes + a_scale_bytes
    a_storage_bytes = 2 * a_stage_bytes
    b_storage_offset = ((a_storage_bytes + 1023) // 1024) * 1024
    # Class constants are the K4 case; __init__ rescales B and everything after it
    # from the stored rate, and at K4 the recomputed values equal these exactly.
    b_stage_bytes = 128 * 128 // 2
    b_storage_bytes = 2 * b_stage_bytes
    sfb_storage_offset = b_storage_offset + b_storage_bytes
    sfb_stage_bytes = 16 * 8 * 4
    shared_bytes = sfb_storage_offset + 2 * sfb_stage_bytes
    shared_words = (shared_bytes + 3) // 4

    def __init__(self, *, trellis_bits: int = 4) -> None:
        if int(trellis_bits) not in (3, 4, 5):
            raise ValueError("coupled P8 prefill FC2 supports K3, K4 or K5 streams")
        super().__init__(scale_sandwich=True, full_coupled=True, trellis_bits=int(trellis_bits))
        if (self.tile_m, self.tile_n, self.source_tile_m) != (32, 128, 32):
            raise ValueError("coupled P8 prefill FC2 requires exact M32/N128")
        # The parent recomputes b_stage/b_storage/sfb_offset/shared from the rate using this
        # subclass's own b_storage_offset; re-derive shared_words so the launch matches.
        self.shared_words = (self.shared_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Extend the 16-argument parent launch ABI with scale_component."""

        self.kernel(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            scatter_output,
            token_map,
            token_weights,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            down_alpha,
            global_scale,
            trellis_lut,
            scale_component,
            intermediate_tiles,
            packed_output_tiles,
        ).launch(
            grid=(1, 1, 376),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
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
        output_tiles = packed_output_tiles * Int32(2)
        task_tail = source_m_tiles * output_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % output_tiles
            source_m_tile = task_slot // output_tiles
            phase1_meta = source_m_tile * intermediate_tiles
            expert_idx = task_expert[phase1_meta].to(Int32)
            valid_rows = task_valid_rows[phase1_meta].to(Int32)
            if valid_rows > Int32(self.tile_m):
                valid_rows = Int32(self.tile_m)
            if valid_rows > Int32(0):
                self._run_task(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    scatter_output,
                    token_map,
                    token_weights,
                    down_alpha,
                    global_scale,
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
                    intermediate_tiles,
                    packed_output_tiles,
                )
            task_slot += Int32(gdimz)


__all__ = ["P8CoupledPrefillFC2Kernel"]
