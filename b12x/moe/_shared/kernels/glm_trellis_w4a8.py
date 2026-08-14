"""Route-packed GLM SQG W4A8 projection primitives.

Unlike Kimi's coupled P24/P33 runtime profile, GLM keeps an independent K3/K4
decision for every expert tensor. This module therefore stores one native
trellis pool per rate and one global-expert-to-pool-slot map per projection.
Route packing groups rows by expert; one warp then decodes a weight fragment
once and reuses it across a complete M64 route block.

The primitive intentionally stops at the transformed projection output. The
caller owns the exact GLM ``suh``/Hadamard input transform and the expert-local
``svh`` output transform. Keeping those operations explicit lets the same
projection serve gate, up, and (if quality permits) down without inventing a
Kimi pair-mode contract for GLM.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
    shared_ptr_to_u32,
)
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_lut
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
from b12x.gemm.trellis_linear.w4a8 import (
    _TrellisW4A8DenseLaunch,
    _validate_quantized,
)


_GLM_ROUTE_BLOCK_ROWS = 128
_GLM_TRELLIS_CODEBOOK = "sqg_xor_cheb_t12"


@dataclass(frozen=True)
class GLMRoutePackedW4A8Projection:
    """Prepared native-trellis pools for one independently rated projection.

    Each pool concatenates complete native tensors in slot order. The maps
    translate a logical expert ID to its rate-local slot, or ``-1`` when that
    expert uses the other rate. Exactly one map must own every active expert;
    the materializer is responsible for sealing that partition.
    """

    size_k: int
    size_n: int
    num_experts: int
    trellis_k3: torch.Tensor
    trellis_k4: torch.Tensor
    expert_slots_k3: torch.Tensor
    expert_slots_k4: torch.Tensor
    trellis_codebook: str = _GLM_TRELLIS_CODEBOOK


def prepare_glm_route_packed_w4a8_projection(
    trellis_by_expert: Sequence[torch.Tensor],
    bits_by_expert: Sequence[int],
    *,
    size_k: int,
    size_n: int,
) -> GLMRoutePackedW4A8Projection:
    """Build rate-local native pools without coupling projection rate choices."""

    tensors = tuple(trellis_by_expert)
    rates = tuple(int(bits) for bits in bits_by_expert)
    if not tensors or len(tensors) != len(rates):
        raise ValueError("GLM projection preparation requires one rate per expert")
    if any(bits not in (3, 4) for bits in rates):
        raise ValueError("GLM projection preparation supports only K3 and K4")
    device = tensors[0].device
    expected_prefix = (int(size_k) // 16, int(size_n) // 16)
    for expert, (tensor, bits) in enumerate(zip(tensors, rates, strict=True)):
        expected_shape = (*expected_prefix, 16 * bits)
        if (
            tensor.dtype != torch.int16
            or tensor.device != device
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != expected_shape
        ):
            raise ValueError(
                f"expert {expert} K{bits} trellis must be contiguous int16 "
                f"{expected_shape} on {device}"
            )

    pools: dict[int, list[torch.Tensor]] = {3: [], 4: []}
    slots = {
        3: torch.full((len(tensors),), -1, dtype=torch.int32, device=device),
        4: torch.full((len(tensors),), -1, dtype=torch.int32, device=device),
    }
    for expert, (tensor, bits) in enumerate(zip(tensors, rates, strict=True)):
        slots[bits][expert] = len(pools[bits])
        pools[bits].append(tensor)

    def stack_rate(bits: int) -> torch.Tensor:
        if pools[bits]:
            return torch.stack(pools[bits]).contiguous()
        return torch.empty((0,), dtype=torch.int16, device=device)

    return GLMRoutePackedW4A8Projection(
        size_k=int(size_k),
        size_n=int(size_n),
        num_experts=len(tensors),
        trellis_k3=stack_rate(3),
        trellis_k4=stack_rate(4),
        expert_slots_k3=slots[3],
        expert_slots_k4=slots[4],
    )


class _GLMRoutePackedW4A8ProjectionLaunch(_TrellisW4A8DenseLaunch):
    """One-warp M64xN8 projection over expert-homogeneous route blocks."""

    def __init__(
        self,
        *,
        size_k: int,
        size_n: int,
        trellis_bits: int,
        topk: int,
        shared_input: bool,
        route_block_rows: int = _GLM_ROUTE_BLOCK_ROWS,
    ) -> None:
        super().__init__(
            size_k=size_k,
            size_n=size_n,
            trellis_bits=trellis_bits,
            pair_kind=None,
            rate_axis=None,
            trellis_codebook=_GLM_TRELLIS_CODEBOOK,
            m_tile_rows=route_block_rows,
        )
        self.topk = int(topk)
        self.shared_input = bool(shared_input)
        self.route_block_rows = int(route_block_rows)
        if self.topk <= 0:
            raise ValueError("GLM route-packed W4A8 topk must be positive")
        if self.route_block_rows != _GLM_ROUTE_BLOCK_ROWS:
            raise ValueError(
                "GLM route-packed W4A8 requires fixed 128-row route blocks"
            )

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        trellis_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        packed_route_indices_ptr: cute.Pointer,
        block_expert_ids_ptr: cute.Pointer,
        expert_slots_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        input_rows: Int32,
        routes: Int32,
        packed_routes: Int32,
        route_blocks: Int32,
        num_experts: Int32,
        pool_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout((input_rows, self.size_k // 4), order=(1, 0)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((input_rows, self.size_k // 32), order=(1, 0)),
        )
        trellis = cute.make_tensor(
            trellis_ptr,
            cute.make_layout((Int64(pool_experts) * Int64(self.trellis_words),)),
        )
        rank_lut = cute.make_tensor(rank_lut_ptr, cute.make_layout((4096,)))
        packed_route_indices = cute.make_tensor(
            packed_route_indices_ptr, cute.make_layout((packed_routes,))
        )
        block_expert_ids = cute.make_tensor(
            block_expert_ids_ptr, cute.make_layout((route_blocks,))
        )
        expert_slots = cute.make_tensor(
            expert_slots_ptr, cute.make_layout((num_experts,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((routes, self.size_n), order=(1, 0)),
        )
        self.kernel(
            values,
            scale_rows,
            trellis,
            rank_lut,
            packed_route_indices,
            block_expert_ids,
            expert_slots,
            output,
            routes,
            packed_routes,
            num_experts,
            pool_experts,
        ).launch(
            grid=(route_blocks, self.size_n // 8, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _run_rate(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis: cute.Tensor,
        rank_lut: cute.Tensor,
        packed_route_indices: cute.Tensor,
        output: cute.Tensor,
        lane: Int32,
        block: Int32,
        n_base: Int32,
        c: Int32,
        g: Int32,
        tensor_base: Int64,
        routes: Int32,
        packed_routes: Int32,
        bits: cutlass.Constexpr[int],
    ) -> None:
        accumulators = tuple(
            cute.make_rmem_tensor((4,), Float32) for _ in range(self.m_groups)
        )
        for group in cutlass.range_constexpr(self.m_groups):
            accumulators[group].fill(0.0)

        rank_lut_addr = get_ptr_as_int64(rank_lut, Int32(0))
        k32 = Int32(0)
        while k32 < Int32(self.size_k // 32):
            n16 = n_base // Int32(16)
            n_high = ((n_base & Int32(15)) >= Int32(8)).to(Int32)
            b0, b1 = self._decode_k32_bits_at_base(
                trellis,
                lane,
                k32,
                n16,
                Int32(0),
                n_high,
                bits,
                rank_lut_addr,
                tensor_base,
            )

            word0 = k32 * Int32(8) + c * Int32(2)
            packed_base = block * Int32(self.route_block_rows)
            for group in cutlass.range_constexpr(self.m_groups):
                group_base = packed_base + Int32(group * 16)
                packed_lo = group_base + g
                packed_hi = packed_lo + Int32(8)
                route_lo = routes
                route_hi = routes
                if packed_lo < packed_routes:
                    route_lo = packed_route_indices[packed_lo].to(Int32)
                if packed_hi < packed_routes:
                    route_hi = packed_route_indices[packed_hi].to(Int32)
                valid_lo = (route_lo >= Int32(0)) & (route_lo < routes)
                valid_hi = (route_hi >= Int32(0)) & (route_hi < routes)
                input_lo = route_lo
                input_hi = route_hi
                if cutlass.const_expr(self.shared_input):
                    input_lo = route_lo // Int32(self.topk)
                    input_hi = route_hi // Int32(self.topk)

                a0 = Uint32(0)
                a1 = Uint32(0)
                a2 = Uint32(0)
                a3 = Uint32(0)
                if valid_lo:
                    a0 = Uint32(values[input_lo, word0])
                    a2 = Uint32(values[input_lo, word0 + Int32(1)])
                if valid_hi:
                    a1 = Uint32(values[input_hi, word0])
                    a3 = Uint32(values[input_hi, word0 + Int32(1)])

                scale_route = route_lo
                scale_valid = valid_lo
                if (lane & Int32(1)) != Int32(0):
                    scale_route = route_hi
                    scale_valid = valid_hi
                scale_input = scale_route
                if cutlass.const_expr(self.shared_input):
                    scale_input = scale_route // Int32(self.topk)
                sf = Uint32(127)
                if scale_valid:
                    sf = Uint32(scale_rows[scale_input, k32])
                sfa = sf * Uint32(0x01010101)

                frag = accumulators[group]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    frag[0],
                    frag[1],
                    frag[2],
                    frag[3],
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    sfa,
                    Uint32(0x7F7F7F7F),
                )
                frag[0] = d0
                frag[1] = d1
                frag[2] = d2
                frag[3] = d3
            k32 += Int32(1)

        col = n_base + c * Int32(2)
        packed_base = block * Int32(self.route_block_rows)
        for group in cutlass.range_constexpr(self.m_groups):
            group_base = packed_base + Int32(group * 16)
            packed_lo = group_base + g
            packed_hi = packed_lo + Int32(8)
            route_lo = routes
            route_hi = routes
            if packed_lo < packed_routes:
                route_lo = packed_route_indices[packed_lo].to(Int32)
            if packed_hi < packed_routes:
                route_hi = packed_route_indices[packed_hi].to(Int32)
            frag = accumulators[group]
            if route_lo >= Int32(0) and route_lo < routes:
                output[route_lo, col] = cutlass.Float16(frag[0])
                output[route_lo, col + Int32(1)] = cutlass.Float16(frag[1])
            if route_hi >= Int32(0) and route_hi < routes:
                output[route_hi, col] = cutlass.Float16(frag[2])
                output[route_hi, col + Int32(1)] = cutlass.Float16(frag[3])

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis: cute.Tensor,
        rank_lut: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        expert_slots: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        packed_routes: Int32,
        num_experts: Int32,
        pool_experts: Int32,
    ) -> None:
        lane = cute.arch.lane_idx()
        block_idx, n_idx, _ = cute.arch.block_idx()
        block = Int32(block_idx)
        n_base = Int32(n_idx) * Int32(8)
        c = lane & Int32(3)
        g = lane >> Int32(2)

        expert = block_expert_ids[block].to(Int32)
        slot = Int32(-1)
        selected = Int32(0)
        if expert >= Int32(0) and expert < num_experts:
            slot = expert_slots[expert].to(Int32)
            if slot >= Int32(0) and slot < pool_experts:
                selected = Int32(1)
        tensor_base = Int64(0)
        if selected != Int32(0):
            tensor_base = Int64(slot) * Int64(self.trellis_words)

        accumulators = tuple(
            cute.make_rmem_tensor((4,), Float32) for _ in range(self.m_groups)
        )
        for group in cutlass.range_constexpr(self.m_groups):
            accumulators[group].fill(0.0)

        rank_lut_addr = get_ptr_as_int64(rank_lut, Int32(0))
        k32 = Int32(0)
        if selected == Int32(0):
            k32 = Int32(self.size_k // 32)
        while k32 < Int32(self.size_k // 32):
            n16 = n_base // Int32(16)
            n_high = ((n_base & Int32(15)) >= Int32(8)).to(Int32)
            if cutlass.const_expr(self.trellis_bits == 3):
                b0, b1 = self._decode_k32_bits_at_base(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(0),
                    n_high,
                    3,
                    rank_lut_addr,
                    tensor_base,
                )
            else:
                b0, b1 = self._decode_k32_bits_at_base(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(0),
                    n_high,
                    4,
                    rank_lut_addr,
                    tensor_base,
                )

            word0 = k32 * Int32(8) + c * Int32(2)
            packed_base = block * Int32(self.route_block_rows)
            for group in cutlass.range_constexpr(self.m_groups):
                group_base = packed_base + Int32(group * 16)
                packed_lo = group_base + g
                packed_hi = packed_lo + Int32(8)
                route_lo = routes
                route_hi = routes
                if packed_lo < packed_routes:
                    route_lo = packed_route_indices[packed_lo].to(Int32)
                if packed_hi < packed_routes:
                    route_hi = packed_route_indices[packed_hi].to(Int32)
                valid_lo = (route_lo >= Int32(0)) & (route_lo < routes)
                valid_hi = (route_hi >= Int32(0)) & (route_hi < routes)
                input_lo = route_lo
                input_hi = route_hi
                if cutlass.const_expr(self.shared_input):
                    input_lo = route_lo // Int32(self.topk)
                    input_hi = route_hi // Int32(self.topk)

                a0 = Uint32(0)
                a1 = Uint32(0)
                a2 = Uint32(0)
                a3 = Uint32(0)
                if valid_lo:
                    a0 = Uint32(values[input_lo, word0])
                    a2 = Uint32(values[input_lo, word0 + Int32(1)])
                if valid_hi:
                    a1 = Uint32(values[input_hi, word0])
                    a3 = Uint32(values[input_hi, word0 + Int32(1)])

                scale_route = route_lo
                scale_valid = valid_lo
                if (lane & Int32(1)) != Int32(0):
                    scale_route = route_hi
                    scale_valid = valid_hi
                scale_input = scale_route
                if cutlass.const_expr(self.shared_input):
                    scale_input = scale_route // Int32(self.topk)
                sf = Uint32(127)
                if scale_valid:
                    sf = Uint32(scale_rows[scale_input, k32])
                sfa = sf * Uint32(0x01010101)

                frag = accumulators[group]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    frag[0],
                    frag[1],
                    frag[2],
                    frag[3],
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    sfa,
                    Uint32(0x7F7F7F7F),
                )
                frag[0] = d0
                frag[1] = d1
                frag[2] = d2
                frag[3] = d3
            k32 += Int32(1)

        col = n_base + c * Int32(2)
        packed_base = block * Int32(self.route_block_rows)
        for group in cutlass.range_constexpr(self.m_groups):
            group_base = packed_base + Int32(group * 16)
            packed_lo = group_base + g
            packed_hi = packed_lo + Int32(8)
            route_lo = routes
            route_hi = routes
            if packed_lo < packed_routes:
                route_lo = packed_route_indices[packed_lo].to(Int32)
            if packed_hi < packed_routes:
                route_hi = packed_route_indices[packed_hi].to(Int32)
            frag = accumulators[group]
            if selected != Int32(0):
                if route_lo >= Int32(0) and route_lo < routes:
                    output[route_lo, col] = cutlass.Float16(frag[0])
                    output[route_lo, col + Int32(1)] = cutlass.Float16(frag[1])
                if route_hi >= Int32(0) and route_hi < routes:
                    output[route_hi, col] = cutlass.Float16(frag[2])
                    output[route_hi, col + Int32(1)] = cutlass.Float16(frag[3])


class _GLMRoutePackedW4A8MixedProjectionLaunch(_GLMRoutePackedW4A8ProjectionLaunch):
    """Dispatch K3/K4 once per expert-homogeneous route block.

    One grid reads both native pools and selects the owning decoder from the
    block's expert identifier. This preserves independent tensor rates without
    launching inactive CTAs for the non-owning bitrate.
    """

    def __init__(
        self,
        *,
        size_k: int,
        size_n: int,
        topk: int,
        shared_input: bool,
        route_block_rows: int = _GLM_ROUTE_BLOCK_ROWS,
    ) -> None:
        super().__init__(
            size_k=size_k,
            size_n=size_n,
            trellis_bits=3,
            topk=topk,
            shared_input=shared_input,
            route_block_rows=route_block_rows,
        )
        self.trellis_words_k3 = self.size_k * self.size_n * 3 // 32
        self.trellis_words_k4 = self.size_k * self.size_n * 4 // 32

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        trellis_k3_ptr: cute.Pointer,
        trellis_k4_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        packed_route_indices_ptr: cute.Pointer,
        block_expert_ids_ptr: cute.Pointer,
        expert_slots_k3_ptr: cute.Pointer,
        expert_slots_k4_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        input_rows: Int32,
        routes: Int32,
        packed_routes: Int32,
        route_blocks: Int32,
        num_experts: Int32,
        pool_experts_k3: Int32,
        pool_experts_k4: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout((input_rows, self.size_k // 4), order=(1, 0)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((input_rows, self.size_k // 32), order=(1, 0)),
        )
        trellis_k3 = cute.make_tensor(
            trellis_k3_ptr,
            cute.make_layout((Int64(pool_experts_k3) * Int64(self.trellis_words_k3),)),
        )
        trellis_k4 = cute.make_tensor(
            trellis_k4_ptr,
            cute.make_layout((Int64(pool_experts_k4) * Int64(self.trellis_words_k4),)),
        )
        rank_lut = cute.make_tensor(rank_lut_ptr, cute.make_layout((4096,)))
        packed_route_indices = cute.make_tensor(
            packed_route_indices_ptr, cute.make_layout((packed_routes,))
        )
        block_expert_ids = cute.make_tensor(
            block_expert_ids_ptr, cute.make_layout((route_blocks,))
        )
        expert_slots_k3 = cute.make_tensor(
            expert_slots_k3_ptr, cute.make_layout((num_experts,))
        )
        expert_slots_k4 = cute.make_tensor(
            expert_slots_k4_ptr, cute.make_layout((num_experts,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((routes, self.size_n), order=(1, 0)),
        )
        self.kernel(
            values,
            scale_rows,
            trellis_k3,
            trellis_k4,
            rank_lut,
            packed_route_indices,
            block_expert_ids,
            expert_slots_k3,
            expert_slots_k4,
            output,
            routes,
            packed_routes,
            num_experts,
            pool_experts_k3,
            pool_experts_k4,
        ).launch(
            grid=(route_blocks, self.size_n // 8, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis_k3: cute.Tensor,
        trellis_k4: cute.Tensor,
        rank_lut: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        expert_slots_k3: cute.Tensor,
        expert_slots_k4: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        packed_routes: Int32,
        num_experts: Int32,
        pool_experts_k3: Int32,
        pool_experts_k4: Int32,
    ) -> None:
        lane = cute.arch.lane_idx()
        block_idx, n_idx, _ = cute.arch.block_idx()
        block = Int32(block_idx)
        n_base = Int32(n_idx) * Int32(8)
        c = lane & Int32(3)
        g = lane >> Int32(2)

        expert = block_expert_ids[block].to(Int32)
        rate = Int32(0)
        tensor_base_k3 = Int64(0)
        tensor_base_k4 = Int64(0)
        if expert >= Int32(0) and expert < num_experts:
            slot_k3 = expert_slots_k3[expert].to(Int32)
            slot_k4 = expert_slots_k4[expert].to(Int32)
            if slot_k3 >= Int32(0) and slot_k3 < pool_experts_k3:
                rate = Int32(3)
                tensor_base_k3 = Int64(slot_k3) * Int64(self.trellis_words_k3)
            elif slot_k4 >= Int32(0) and slot_k4 < pool_experts_k4:
                rate = Int32(4)
                tensor_base_k4 = Int64(slot_k4) * Int64(self.trellis_words_k4)
        if rate == Int32(3):
            self._run_rate(
                values,
                scale_rows,
                trellis_k3,
                rank_lut,
                packed_route_indices,
                output,
                lane,
                block,
                n_base,
                c,
                g,
                tensor_base_k3,
                routes,
                packed_routes,
                3,
            )
        elif rate == Int32(4):
            self._run_rate(
                values,
                scale_rows,
                trellis_k4,
                rank_lut,
                packed_route_indices,
                output,
                lane,
                block,
                n_base,
                c,
                g,
                tensor_base_k4,
                routes,
                packed_routes,
                4,
            )


_V2_TILE_N = 64
_V2_N8_PER_WARP = 2
_V2_THREADS = 128
_V2_KTILE_K32 = 4  # k32 steps per staged A tile (K128)
_V2_A_BLOCK_WORDS = 16 * 32  # 16 rows x K128 bytes, u32 words
_V2_STAGES = 2


class _GLMRoutePackedW4A8TileLaunch(_GLMRoutePackedW4A8MixedProjectionLaunch):
    """Four-warp M64xN256 route-packed projection with staged A operands.

    The M64xN8 one-warp kernel re-read the quantized A rows from global
    memory once per N8 column strip (256x amplification at N=2048) and had a
    single warp to cover decode plus MMA latency.  This launch keeps the
    packed-route workspace and decode primitives unchanged and fixes the
    schedule:

    - one CTA covers M64 x N256 with 128 threads; each warp owns eight n8
      strips, so a decoded K32xN8 fragment feeds four M16 MMAs instead of
      being re-decoded per M16 pair;
    - A bytes and UE8M0 row scales are cp.async double-buffered through
      shared memory with the xor-swizzle from the fused W4A8 pipeline, so A
      global traffic per route block falls from size_n/8 reads to size_n/256;
    - B stays register-resident: the existing warp trellis decode already
      produces the exact m16n8k32 fragment layout, so there is no B staging,
      no second decode of a fragment inside a CTA, and the compact payload is
      the only weight representation touched.

    Rate independence is preserved: the expert's K3 or K4 pool is selected
    once per CTA exactly as in the mixed one-warp launch.
    """

    def __init__(
        self,
        *,
        size_k: int,
        size_n: int,
        topk: int,
        shared_input: bool,
        blocks_per_cta: int = 8,
        stages: int = 2,
    ) -> None:
        super().__init__(
            size_k=size_k,
            size_n=size_n,
            topk=topk,
            shared_input=shared_input,
        )
        if int(stages) not in (2, 3, 4):
            raise ValueError("v2 stages must be 2, 3, or 4")
        self.stages = int(stages)
        m16_blocks = self.route_block_rows // 16
        if int(blocks_per_cta) not in (2, 4, 8):
            raise ValueError("v2 blocks_per_cta must be 2, 4, or 8")
        if m16_blocks % int(blocks_per_cta):
            raise ValueError("v2 blocks_per_cta must divide the route block")
        if self.size_n % _V2_TILE_N:
            raise ValueError("v2 requires size_n divisible by 64")
        if self.size_k % (_V2_KTILE_K32 * 32):
            raise ValueError("v2 requires size_k divisible by 128")
        self.blocks_per_cta = int(blocks_per_cta)
        # M128 blocks are split into M16 CTA parts when requested.
        self.m_parts = m16_blocks // self.blocks_per_cta
        self.a_words = self.blocks_per_cta * _V2_A_BLOCK_WORDS
        self.asf_words = self.blocks_per_cta * 16
        # Maximum compact B payload for one K128 x N64 tile is K4:
        # 8 K16 rows x 4 N16 columns x 32 u32 words per tile.
        self.b_words = (_V2_KTILE_K32 * 2) * (_V2_TILE_N // 16) * (8 * 4)
        self.b_offset_words = self.a_words + self.asf_words
        self.stage_words = self.b_offset_words + self.b_words
        self.stage_bytes = self.stage_words * 4
        self.lut_words = 4096 // 4
        self.k_tiles = self.size_k // (_V2_KTILE_K32 * 32)

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        trellis_k3_ptr: cute.Pointer,
        trellis_k4_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        packed_route_indices_ptr: cute.Pointer,
        block_expert_ids_ptr: cute.Pointer,
        expert_slots_k3_ptr: cute.Pointer,
        expert_slots_k4_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        input_rows: Int32,
        routes: Int32,
        packed_routes: Int32,
        route_blocks: Int32,
        num_experts: Int32,
        pool_experts_k3: Int32,
        pool_experts_k4: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr,
            cute.make_layout((Int64(input_rows) * Int64(self.size_k // 4),)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_layout((Int64(input_rows) * Int64(self.size_k // 32),)),
        )
        trellis_k3 = cute.make_tensor(
            trellis_k3_ptr,
            cute.make_layout((Int64(pool_experts_k3) * Int64(self.trellis_words_k3),)),
        )
        trellis_k4 = cute.make_tensor(
            trellis_k4_ptr,
            cute.make_layout((Int64(pool_experts_k4) * Int64(self.trellis_words_k4),)),
        )
        rank_lut = cute.make_tensor(rank_lut_ptr, cute.make_layout((4096,)))
        packed_route_indices = cute.make_tensor(
            packed_route_indices_ptr, cute.make_layout((packed_routes,))
        )
        block_expert_ids = cute.make_tensor(
            block_expert_ids_ptr, cute.make_layout((route_blocks,))
        )
        expert_slots_k3 = cute.make_tensor(
            expert_slots_k3_ptr, cute.make_layout((num_experts,))
        )
        expert_slots_k4 = cute.make_tensor(
            expert_slots_k4_ptr, cute.make_layout((num_experts,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((routes, self.size_n), order=(1, 0)),
        )
        self.kernel(
            values,
            scale_rows,
            trellis_k3,
            trellis_k4,
            rank_lut,
            packed_route_indices,
            block_expert_ids,
            expert_slots_k3,
            expert_slots_k4,
            output,
            routes,
            packed_routes,
            num_experts,
            pool_experts_k3,
            pool_experts_k4,
        ).launch(
            grid=(
                route_blocks * Int32(self.m_parts),
                self.size_n // _V2_TILE_N,
                1,
            ),
            block=[_V2_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis_k3: cute.Tensor,
        trellis_k4: cute.Tensor,
        rank_lut: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        expert_slots_k3: cute.Tensor,
        expert_slots_k4: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        packed_routes: Int32,
        num_experts: Int32,
        pool_experts_k3: Int32,
        pool_experts_k4: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tid = Int32(tidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            sData: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint32,
                    self.stages * self.stage_words + self.lut_words,
                ],
                16,
            ]

        storage = smem.allocate(Storage)
        # Hoisted before any dynamic control flow (flattener trap).
        s_base = shared_ptr_to_u32(storage.sData.data_ptr())

        part = Int32(0)
        block = Int32(bidx)
        if cutlass.const_expr(self.m_parts > 1):
            part = block % Int32(self.m_parts)
            block = block // Int32(self.m_parts)
        n_tile = Int32(bidy)

        expert = block_expert_ids[block].to(Int32)
        rate = Int32(0)
        tensor_base_k3 = Int64(0)
        tensor_base_k4 = Int64(0)
        if expert >= Int32(0) and expert < num_experts:
            slot_k3 = expert_slots_k3[expert].to(Int32)
            slot_k4 = expert_slots_k4[expert].to(Int32)
            if slot_k3 >= Int32(0) and slot_k3 < pool_experts_k3:
                rate = Int32(3)
                tensor_base_k3 = Int64(slot_k3) * Int64(self.trellis_words_k3)
            elif slot_k4 >= Int32(0) and slot_k4 < pool_experts_k4:
                rate = Int32(4)
                tensor_base_k4 = Int64(slot_k4) * Int64(self.trellis_words_k4)
        if rate == Int32(3):
            self._tile_body(
                values,
                scale_rows,
                trellis_k3,
                rank_lut,
                packed_route_indices,
                output,
                s_base,
                tid,
                block,
                part,
                n_tile,
                tensor_base_k3,
                routes,
                packed_routes,
                3,
            )
        elif rate == Int32(4):
            self._tile_body(
                values,
                scale_rows,
                trellis_k4,
                rank_lut,
                packed_route_indices,
                output,
                s_base,
                tid,
                block,
                part,
                n_tile,
                tensor_base_k4,
                routes,
                packed_routes,
                4,
            )

    @cute.jit
    def _stage_ktile(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis: cute.Tensor,
        stage_base: Int32,
        tid: Int32,
        kt: Int32,
        n_tile: Int32,
        tensor_base: Int64,
        a_src_words: cute.Tensor,  # rmem [blocks]: -1 = padded row (skip)
        a_dst_addr: Int32,
        asf_src_byte: Int32,  # -1 = padded row (skip)
        asf_dst_addr: Int32,
        bits: cutlass.Constexpr[int],
    ) -> None:
        """cp.async one compact (A, Asf, B) K128 tile into shared memory.

        A: blocks x 16 rows x 128B, xor-swizzled 16B units exactly like the
        fused W4A8 pipeline so fragment reads are bank-conflict-free.  Rows
        whose packed route slot is padding are skipped; their smem bytes are
        never consumed because the epilogue masks those routes out.
        """
        for blk in cutlass.range_constexpr(self.blocks_per_cta):
            src = Int32(a_src_words[blk])
            if src >= Int32(0):
                cp_async4_shared_global(
                    stage_base + Int32(blk * _V2_A_BLOCK_WORDS * 4) + a_dst_addr,
                    get_ptr_as_int64(values, src + kt * Int32(32)),
                )
        if asf_src_byte >= Int32(0):
            cp_async_u32_shared_global(
                stage_base + Int32(self.a_words * 4) + asf_dst_addr,
                get_ptr_as_int64(scale_rows, asf_src_byte + kt * Int32(4)),
            )

        # Cooperatively stage the native compressed trellis payload. The
        # source layout is [K16, N16, 8*bits u32]; the shared layout packs only
        # this CTA's 8-by-4 tile. K3 uses 192 16-byte units, K4 uses 256.
        units_per_trellis_tile = 2 * int(bits)
        n16_per_cta = _V2_TILE_N // 16
        total_units = _V2_KTILE_K32 * 2 * n16_per_cta * units_per_trellis_tile
        for copy in cutlass.range_constexpr(2):
            unit = tid + Int32(copy * _V2_THREADS)
            if unit < Int32(total_units):
                tile = unit // Int32(units_per_trellis_tile)
                unit_in_tile = unit % Int32(units_per_trellis_tile)
                k16_local = tile // Int32(n16_per_cta)
                n16_local = tile % Int32(n16_per_cta)
                global_k16 = kt * Int32(_V2_KTILE_K32 * 2) + k16_local
                global_n16 = n_tile * Int32(n16_per_cta) + n16_local
                src_word = (
                    tensor_base
                    + (Int64(global_k16) * Int64(self.size_n // 16) + Int64(global_n16))
                    * Int64(8 * int(bits))
                    + Int64(unit_in_tile) * Int64(4)
                )
                cp_async4_shared_global(
                    stage_base + Int32(self.b_offset_words * 4) + unit * Int32(16),
                    get_ptr_as_int64(trellis, src_word),
                )

    @cute.jit
    def _decode_tile_at_base_shared(
        self,
        b_base: Int32,
        lane: Int32,
        k16_local: Int32,
        n16_local: Int32,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int32,
    ) -> Uint32:
        ia, ib, s2 = self._lane_geom(lane, bits)
        tile_word = (k16_local * Int32(_V2_TILE_N // 16) + n16_local) * Int32(
            8 * int(bits)
        )
        a = ld_shared_u32(b_base + (tile_word + ia) * Int32(4))
        b = ld_shared_u32(b_base + (tile_word + ib) * Int32(4))
        merged = (Int64(a) << Int64(32)) | Int64(b)
        win_a = Uint32(merged >> Int64(s2))
        win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
        lo, hi = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            win_a,
            win_b,
            rank_lut_addr,
            int(bits),
            t12_in_shared=True,
        )
        value = lo
        if n_high != Int32(0):
            value = hi
        return value

    @cute.jit
    def _decode_k32_pair_shared(
        self,
        b_base: Int32,
        lane: Int32,
        kb: Int32,
        n16_local: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int32,
    ):
        """Decode BOTH n8 halves of one staged n16 tile for one K32 step.

        ``_decode_tile_at_base_shared`` computes the low and high n8 halves
        and discards one by ``n_high``, so both halves of every staged tile
        were decoded twice.  This performs the same two K16 tile reads once,
        keeps both halves, and reproduces the two-call fragment values and
        lane placement bit-exactly with half the shared-memory reads, SQG
        hashing, and LUT gathers.
        """
        ia, ib, s2 = self._lane_geom(lane, bits)
        n16_words = Int32(_V2_TILE_N // 16) * Int32(8 * int(bits))
        tile_word0 = (kb * Int32(2) * Int32(_V2_TILE_N // 16) + n16_local) * Int32(
            8 * int(bits)
        )
        a0 = ld_shared_u32(b_base + (tile_word0 + ia) * Int32(4))
        b0w = ld_shared_u32(b_base + (tile_word0 + ib) * Int32(4))
        merged0 = (Int64(a0) << Int64(32)) | Int64(b0w)
        lo0, hi0 = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            Uint32(merged0 >> Int64(s2)),
            Uint32(merged0 >> Int64(s2 + Int32(4 * int(bits)))),
            rank_lut_addr,
            int(bits),
            t12_in_shared=True,
        )
        tile_word1 = tile_word0 + n16_words
        a1 = ld_shared_u32(b_base + (tile_word1 + ia) * Int32(4))
        b1w = ld_shared_u32(b_base + (tile_word1 + ib) * Int32(4))
        merged1 = (Int64(a1) << Int64(32)) | Int64(b1w)
        lo1, hi1 = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            Uint32(merged1 >> Int64(s2)),
            Uint32(merged1 >> Int64(s2 + Int32(4 * int(bits)))),
            rank_lut_addr,
            int(bits),
            t12_in_shared=True,
        )
        c = lane & Int32(3)
        own_lo = lo0
        send_lo = lo1
        own_hi = hi0
        send_hi = hi1
        if c >= Int32(2):
            own_lo = lo1
            send_lo = lo0
            own_hi = hi1
            send_hi = hi0
        peer_lo = Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2))
        peer_hi = Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2))
        return own_lo, peer_lo, own_hi, peer_hi

    @cute.jit
    def _decode_k32_bits_at_base_shared(
        self,
        b_base: Int32,
        lane: Int32,
        kb: Int32,
        n16_local: Int32,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int32,
    ):
        e0 = self._decode_tile_at_base_shared(
            b_base,
            lane,
            kb * Int32(2),
            n16_local,
            n_high,
            bits,
            rank_lut_addr,
        )
        e1 = self._decode_tile_at_base_shared(
            b_base,
            lane,
            kb * Int32(2) + Int32(1),
            n16_local,
            n_high,
            bits,
            rank_lut_addr,
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
    def _tile_body(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis: cute.Tensor,
        rank_lut: cute.Tensor,
        packed_route_indices: cute.Tensor,
        output: cute.Tensor,
        s_base: Int32,
        tid: Int32,
        block: Int32,
        part: Int32,
        n_tile: Int32,
        tensor_base: Int64,
        routes: Int32,
        packed_routes: Int32,
        bits: cutlass.Constexpr[int],
    ) -> None:
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        packed_base = block * Int32(self.route_block_rows) + part * Int32(
            self.blocks_per_cta * 16
        )
        a_row_words = Int32(self.size_k // 4)
        asf_row_bytes = Int32(self.size_k // 32)

        # ---- Per-thread staging source addresses, resolved once. ----
        # A copy role: row = tid>>3 in [0,16), 16B unit v = tid&7, swizzled
        # destination unit p = v ^ (row & 7).
        stage_row = tid >> Int32(3)
        stage_v = tid & Int32(7)
        stage_p = stage_v ^ (stage_row & Int32(7))
        a_dst_addr = (stage_row << Int32(7)) + (stage_p << Int32(4))
        a_src_words = cute.make_rmem_tensor((self.blocks_per_cta,), cutlass.Int32)
        for blk in cutlass.range_constexpr(self.blocks_per_cta):
            src_words = Int32(-1)
            slot = packed_base + Int32(blk * 16) + stage_row
            if slot < packed_routes:
                route = packed_route_indices[slot].to(Int32)
                if route >= Int32(0) and route < routes:
                    src_row = route
                    if cutlass.const_expr(self.shared_input):
                        src_row = route // Int32(self.topk)
                    src_words = src_row * a_row_words + (stage_v << Int32(2))
            a_src_words[blk] = src_words
        # Asf copy role: tid < blocks*16 loads one u32 (4 UE8M0 bytes = one
        # K128 tile) for packed row tid.
        asf_src_byte = Int32(-1)
        asf_dst_addr = tid << Int32(2)
        if tid < Int32(self.blocks_per_cta * 16):
            slot = packed_base + tid
            if slot < packed_routes:
                route = packed_route_indices[slot].to(Int32)
                if route >= Int32(0) and route < routes:
                    src_row = route
                    if cutlass.const_expr(self.shared_input):
                        src_row = route // Int32(self.topk)
                    asf_src_byte = src_row * asf_row_bytes
        # The decoder performs many byte lookups per fragment. Stage the
        # immutable 4 KiB T12 table once per CTA to replace long-scoreboard
        # global byte loads with shared-memory lookups.
        lut_base = s_base + Int32(self.stages * self.stage_bytes)
        for copy in cutlass.range_constexpr(2):
            lut_src = tid * Int32(32) + Int32(copy * 16)
            cp_async4_shared_global(
                lut_base + lut_src,
                get_ptr_as_int64(rank_lut, lut_src),
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        rank_lut_addr = Int32(lut_base)

        # f32 accumulators [m-block][n8 strip][fragment], full K sweep.
        facc = cute.make_rmem_tensor((self.blocks_per_cta, _V2_N8_PER_WARP, 4), Float32)
        facc.fill(0.0)

        # ---- cp.async pipeline: prefetch the first stages-1 k-tiles. ----
        for p in cutlass.range_constexpr(self.stages - 1):
            if Int32(p) < Int32(self.k_tiles):
                self._stage_ktile(
                    values,
                    scale_rows,
                    trellis,
                    s_base + Int32(p % self.stages) * Int32(self.stage_bytes),
                    tid,
                    Int32(p),
                    n_tile,
                    tensor_base,
                    a_src_words,
                    a_dst_addr,
                    asf_src_byte,
                    asf_dst_addr,
                    bits,
                )
            cute.arch.cp_async_commit_group()

        kt = Int32(0)
        k_tiles = Int32(self.k_tiles)
        while kt < k_tiles:
            nxt = kt + Int32(self.stages - 1)
            if nxt < k_tiles:
                self._stage_ktile(
                    values,
                    scale_rows,
                    trellis,
                    s_base + (nxt % Int32(self.stages)) * Int32(self.stage_bytes),
                    tid,
                    nxt,
                    n_tile,
                    tensor_base,
                    a_src_words,
                    a_dst_addr,
                    asf_src_byte,
                    asf_dst_addr,
                    bits,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            cur = s_base + (kt % Int32(self.stages)) * Int32(self.stage_bytes)
            sasf_base = cur + Int32(self.a_words * 4)
            sb_base = cur + Int32(self.b_offset_words * 4)

            # Lane's SFA row follows the one-warp kernel's parity rule: even
            # lanes carry row q, odd lanes row q+8.
            asf_row = q + ((lane & Int32(1)) << Int32(3))
            asc = cute.make_rmem_tensor((self.blocks_per_cta,), cutlass.Uint32)
            for blk in cutlass.range_constexpr(self.blocks_per_cta):
                asc[blk] = ld_shared_u32(
                    sasf_base + Int32(blk * 64) + (asf_row << Int32(2))
                )

            for kb in cutlass.range_constexpr(_V2_KTILE_K32):
                # Swizzle-aware A fragment loads (rows q and q+8).
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_lo = (
                    cur
                    + (q << Int32(7))
                    + (u_phys << Int32(4))
                    + ((c & Int32(1)) << Int32(3))
                )
                a_frag = cute.make_rmem_tensor((self.blocks_per_cta, 4), cutlass.Uint32)
                for blk in cutlass.range_constexpr(self.blocks_per_cta):
                    blk_off = Int32(blk * _V2_A_BLOCK_WORDS * 4)
                    f0, f2 = ld_shared_v2_u32(a_lo + blk_off)
                    f1, f3 = ld_shared_v2_u32(a_lo + blk_off + Int32(8 * 128))
                    a_frag[blk, 0] = f0
                    a_frag[blk, 1] = f1
                    a_frag[blk, 2] = f2
                    a_frag[blk, 3] = f3
                sfa = cute.make_rmem_tensor((self.blocks_per_cta,), cutlass.Uint32)
                for blk in cutlass.range_constexpr(self.blocks_per_cta):
                    sf = (Uint32(asc[blk]) >> Uint32(8 * kb)) & Uint32(0xFF)
                    sfa[blk] = sf * Uint32(0x01010101)

                for t in cutlass.range_constexpr(_V2_N8_PER_WARP // 2):
                    n_base_lo = (
                        n_tile * Int32(_V2_TILE_N)
                        + warp * Int32(_V2_N8_PER_WARP * 8)
                        + Int32(t * 16)
                    )
                    n16_local = (n_base_lo >> Int32(4)) - n_tile * Int32(
                        _V2_TILE_N // 16
                    )
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_pair_shared(
                        sb_base,
                        lane,
                        Int32(kb),
                        n16_local,
                        bits,
                        rank_lut_addr,
                    )
                    for h in cutlass.range_constexpr(2):
                        i = t * 2 + h
                        if cutlass.const_expr(h == 0):
                            b0 = b0_lo
                            b1 = b1_lo
                        else:
                            b0 = b0_hi
                            b1 = b1_hi
                        for blk in cutlass.range_constexpr(self.blocks_per_cta):
                            frag = facc
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                frag[blk, i, 0],
                                frag[blk, i, 1],
                                frag[blk, i, 2],
                                frag[blk, i, 3],
                                Uint32(a_frag[blk, 0]),
                                Uint32(a_frag[blk, 1]),
                                Uint32(a_frag[blk, 2]),
                                Uint32(a_frag[blk, 3]),
                                b0,
                                b1,
                                Uint32(sfa[blk]),
                                Uint32(0x7F7F7F7F),
                            )
                            frag[blk, i, 0] = d0
                            frag[blk, i, 1] = d1
                            frag[blk, i, 2] = d2
                            frag[blk, i, 3] = d3
            cute.arch.sync_threads()
            kt += Int32(1)

        # ---- epilogue: scatter valid routes, identical to the one-warp
        # kernel's mapping (rows q/q+8 of each M16 block, columns 2c/2c+1 of
        # each n8 strip).
        for blk in cutlass.range_constexpr(self.blocks_per_cta):
            group_base = packed_base + Int32(blk * 16)
            packed_lo = group_base + q
            packed_hi = packed_lo + Int32(8)
            route_lo = routes
            route_hi = routes
            if packed_lo < packed_routes:
                route_lo = packed_route_indices[packed_lo].to(Int32)
            if packed_hi < packed_routes:
                route_hi = packed_route_indices[packed_hi].to(Int32)
            for i in cutlass.range_constexpr(_V2_N8_PER_WARP):
                col = (
                    n_tile * Int32(_V2_TILE_N)
                    + warp * Int32(_V2_N8_PER_WARP * 8)
                    + Int32(i * 8)
                    + c * Int32(2)
                )
                if route_lo >= Int32(0) and route_lo < routes:
                    output[route_lo, col] = cutlass.Float16(facc[blk, i, 0])
                    output[route_lo, col + Int32(1)] = cutlass.Float16(facc[blk, i, 1])
                if route_hi >= Int32(0) and route_hi < routes:
                    output[route_hi, col] = cutlass.Float16(facc[blk, i, 2])
                    output[route_hi, col + Int32(1)] = cutlass.Float16(facc[blk, i, 3])


def _ptr(dtype, address: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _compile_glm_route_packed_projection(
    size_k: int,
    size_n: int,
    trellis_bits: int,
    topk: int,
    shared_input: bool,
    device_index: int,
):
    launch = _GLMRoutePackedW4A8ProjectionLaunch(
        size_k=size_k,
        size_n=size_n,
        trellis_bits=trellis_bits,
        topk=topk,
        shared_input=shared_input,
    )
    key = (
        int(size_k),
        int(size_n),
        int(trellis_bits),
        int(topk),
        bool(shared_input),
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.glm_route_packed_trellis_w4a8_projection",
            1,
            key,
        ),
    )


@functools.cache
def _compile_glm_route_packed_mixed_projection(
    size_k: int,
    size_n: int,
    topk: int,
    shared_input: bool,
    device_index: int,
):
    launch = _GLMRoutePackedW4A8MixedProjectionLaunch(
        size_k=size_k,
        size_n=size_n,
        topk=topk,
        shared_input=shared_input,
    )
    key = (
        int(size_k),
        int(size_n),
        int(topk),
        bool(shared_input),
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.glm_route_packed_trellis_w4a8_mixed_projection",
            1,
            key,
        ),
    )


@functools.cache
def _compile_glm_route_packed_tile_projection(
    size_k: int,
    size_n: int,
    topk: int,
    shared_input: bool,
    blocks_per_cta: int,
    stages: int,
    device_index: int,
):
    launch = _GLMRoutePackedW4A8TileLaunch(
        size_k=size_k,
        size_n=size_n,
        topk=topk,
        shared_input=shared_input,
        blocks_per_cta=blocks_per_cta,
        stages=stages,
    )
    key = (
        int(size_k),
        int(size_n),
        int(topk),
        bool(shared_input),
        int(blocks_per_cta),
        int(stages),
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.glm_route_packed_trellis_w4a8_tile_projection",
            1,
            key,
        ),
    )


def _selected_w4a8_kernel() -> str:
    kernel = os.environ.get("B12X_GLM_W4A8_KERNEL", "m128n64").strip().lower()
    if kernel not in ("m64n8", "m128n64"):
        raise ValueError("B12X_GLM_W4A8_KERNEL must be 'm64n8' (one-warp) or 'm128n64'")
    return kernel


def _v2_stages() -> int:
    stages = int(os.environ.get("B12X_GLM_W4A8_V2_STAGES", "2"))
    if stages not in (2, 3, 4):
        raise ValueError("B12X_GLM_W4A8_V2_STAGES must be 2, 3, or 4")
    return stages


def _v2_blocks_per_cta() -> int:
    blocks = int(os.environ.get("B12X_GLM_W4A8_V2_BLOCKS", "8"))
    if blocks not in (2, 4, 8):
        raise ValueError("B12X_GLM_W4A8_V2_BLOCKS must be 2, 4, or 8")
    return blocks


def _validate_pool(
    name: str,
    pool: torch.Tensor,
    *,
    bits: int,
    size_k: int,
    size_n: int,
    device: torch.device,
) -> int:
    if pool.dtype != torch.int16 or pool.device != device or not pool.is_contiguous():
        raise TypeError(f"{name} must be contiguous int16 on {device}")
    values_per_expert = int(size_k) * int(size_n)
    encoded_i16_per_expert = values_per_expert * int(bits) // 16
    if encoded_i16_per_expert <= 0 or int(pool.numel()) % encoded_i16_per_expert:
        raise ValueError(f"{name} does not contain complete native K{bits} tensors")
    return int(pool.numel()) // encoded_i16_per_expert


def _validate_slot_map(
    name: str,
    slots: torch.Tensor,
    *,
    num_experts: int,
    device: torch.device,
) -> None:
    if (
        slots.dtype != torch.int32
        or slots.device != device
        or not slots.is_contiguous()
        or tuple(slots.shape) != (int(num_experts),)
    ):
        raise TypeError(
            f"{name} must be contiguous int32 [{int(num_experts)}] on {device}"
        )


def run_glm_route_packed_w4a8_projection(
    quantized: MXFP8Rows,
    prepared: GLMRoutePackedW4A8Projection,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    topk: int,
    shared_input: bool,
    clear_output: bool = True,
) -> torch.Tensor:
    """Execute one mixed-K3/K4 GLM projection over packed expert routes."""

    size_k = int(prepared.size_k)
    size_n = int(prepared.size_n)
    num_experts = int(prepared.num_experts)
    topk = int(topk)
    if topk <= 0:
        raise ValueError("topk must be positive")
    if str(prepared.trellis_codebook).lower() != _GLM_TRELLIS_CODEBOOK:
        raise ValueError("GLM route-packed W4A8 requires SQG-XOR-Cheb-T12")
    if output.ndim != 2 or output.dtype != torch.float16 or not output.is_cuda:
        raise TypeError("output must be contiguous CUDA FP16 [routes, N]")
    if not output.is_contiguous() or int(output.shape[1]) != size_n:
        raise ValueError("output must be contiguous with the prepared N dimension")
    device = output.device
    routes = int(output.shape[0])
    if routes <= 0 or routes % topk:
        raise ValueError("output routes must be positive and divisible by topk")
    if size_k <= 0 or size_k % 32 or size_n <= 0 or size_n % 8:
        raise ValueError("GLM W4A8 projection dimensions must close K32 and N8")
    input_rows = routes // topk if shared_input else routes
    _validate_quantized(quantized, m=input_rows, k=size_k, device=device)

    for name, value in (
        ("packed_route_indices", packed_route_indices),
        ("block_expert_ids", block_expert_ids),
    ):
        if (
            value.dtype != torch.int32
            or value.device != device
            or not value.is_contiguous()
        ):
            raise TypeError(f"{name} must be contiguous int32 on {device}")
    route_blocks = min(
        int(block_expert_ids.numel()),
        int(packed_route_indices.numel()) // _GLM_ROUTE_BLOCK_ROWS,
    )
    if route_blocks <= 0:
        raise ValueError("packed route workspace contains no complete M64 block")

    pool_counts = {
        3: _validate_pool(
            "trellis_k3",
            prepared.trellis_k3,
            bits=3,
            size_k=size_k,
            size_n=size_n,
            device=device,
        ),
        4: _validate_pool(
            "trellis_k4",
            prepared.trellis_k4,
            bits=4,
            size_k=size_k,
            size_n=size_n,
            device=device,
        ),
    }
    _validate_slot_map(
        "expert_slots_k3",
        prepared.expert_slots_k3,
        num_experts=num_experts,
        device=device,
    )
    _validate_slot_map(
        "expert_slots_k4",
        prepared.expert_slots_k4,
        num_experts=num_experts,
        device=device,
    )
    if clear_output:
        output.zero_()

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    rank_lut = sqg_xor_cheb_t12_lut(device)
    # Shapes without N256/K128 closure use the one-warp kernel, whose geometry
    # accepts the remaining aligned dimensions.
    use_tile_kernel = (
        _selected_w4a8_kernel() == "m128n64"
        and size_n % _V2_TILE_N == 0
        and size_k % (_V2_KTILE_K32 * 32) == 0
    )
    if use_tile_kernel:
        compiled = _compile_glm_route_packed_tile_projection(
            size_k,
            size_n,
            topk,
            bool(shared_input),
            _v2_blocks_per_cta(),
            _v2_stages(),
            int(device_index),
        )
    else:
        compiled = _compile_glm_route_packed_mixed_projection(
            size_k,
            size_n,
            topk,
            bool(shared_input),
            int(device_index),
        )
    compiled(
        _ptr(cutlass.Uint32, quantized.values.data_ptr()),
        _ptr(cutlass.Uint8, quantized.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, prepared.trellis_k3.data_ptr()),
        _ptr(cutlass.Uint32, prepared.trellis_k4.data_ptr()),
        _ptr(cutlass.Uint8, rank_lut.data_ptr()),
        _ptr(cutlass.Int32, packed_route_indices.data_ptr()),
        _ptr(cutlass.Int32, block_expert_ids.data_ptr()),
        _ptr(cutlass.Int32, prepared.expert_slots_k3.data_ptr()),
        _ptr(cutlass.Int32, prepared.expert_slots_k4.data_ptr()),
        _ptr(cutlass.Float16, output.data_ptr()),
        input_rows,
        routes,
        int(packed_route_indices.numel()),
        route_blocks,
        num_experts,
        pool_counts[3],
        pool_counts[4],
        current_cuda_stream(),
    )
    return output


__all__ = [
    "GLMRoutePackedW4A8Projection",
    "prepare_glm_route_packed_w4a8_projection",
    "run_glm_route_packed_w4a8_projection",
]
