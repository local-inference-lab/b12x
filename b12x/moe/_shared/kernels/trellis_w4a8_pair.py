"""Route-major E4M3 QSRT W4A8 primitives for the qualified local geometry.

These kernels are the arithmetic core of the routed port.  They consume the
compact P24/P33 runtime view produced from canonical QSRT atoms and
an explicit route-to-local-expert vector.  Native trellis windows are decoded
straight into E4M3 MMA B registers; no dense weight tensor is materialized.
SQG-XOR-Cheb-T12 decodes through the active per-rate direct tables read via the
read-only global cache.

The initial implementation assigns one routed row to each CTA.  Four warps
cover one N128 slab, which is the useful SM120 decode geometry and an honest
correctness/performance anchor for later same-expert M packing.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_lut,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows


_THREADS = 128
_WARPS = 4
from b12x.moe._shared.kernels.trellis_decode import (
    _PAIR_U32_PER_K16,
    _NativeTrellisDecode,
)
_W4A8_CODEBOOKS = {"sqg_xor_cheb_t12"}


class TrellisW4A8FC1RoutesKernel(_NativeTrellisDecode):
    """Compute one FC1 projection for one route x N32 slab per CTA.

    Decode has very few routes at autoregressive M=1, so the grid splits the
    four independent N32 slabs and the gate/up projections into separate
    CTAs (256-way).  Each CTA additionally splits the long H/32 reduction
    across eight K-slice warps that combine their single live output row
    through shared memory, lifting residency from ~1.4 to ~11 warps per SM
    without changing the decoded payload or doing any additional MMA work.
    """

    threads = 256

    def __init__(
        self,
        *,
        hidden_size: int,
        topk: int = 1,
        shared_input: bool = False,
        decode_both: bool = True,
        sqg_direct_lut: bool = False,
        pair_mode_filter: int = -1,
        trellis_codebook: str = "sqg_xor_cheb_t12",
    ) -> None:
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0 or self.hidden_size % 256 != 0:
            raise ValueError(
                "FC1 route kernel hidden_size must be a positive multiple of 256"
            )
        self.topk = int(topk)
        self.shared_input = bool(shared_input)
        self.decode_both = bool(decode_both)
        self.sqg_direct_lut = bool(sqg_direct_lut)
        self.pair_mode_filter = int(pair_mode_filter)
        if self.pair_mode_filter not in (-1, 0, 1):
            raise ValueError("pair_mode_filter must be -1, 0 (P33), or 1 (P24)")
        self.trellis_codebook = str(trellis_codebook).lower()
        if self.trellis_codebook not in _W4A8_CODEBOOKS:
            raise ValueError(
                "W4A8 trellis codebook must be one of "
                f"{sorted(_W4A8_CODEBOOKS)}, got {self.trellis_codebook!r}"
            )
        if not self.sqg_direct_lut:
            raise ValueError(
                "SQG-XOR-Cheb-T12 W4A8 requires T12 table execution"
            )
        self.sqg_rank_lut_entries = 1
        self.sqg_lut_entries = 4096 if self.sqg_direct_lut else 1
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        self.k16_count = self.hidden_size // 16
        self.expert_u32 = self.k16_count * _PAIR_U32_PER_K16

    @cute.jit
    def __call__(
        self,
        gate_values_ptr: cute.Pointer,
        gate_scale_ptr: cute.Pointer,
        up_values_ptr: cute.Pointer,
        up_scale_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        pair_modes_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        gate_out_ptr: cute.Pointer,
        up_out_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        input_rows = (
            routes // Int32(self.topk)
            if self.shared_input
            else routes
        )
        gate_values = cute.make_tensor(
            gate_values_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        gate_scales = cute.make_tensor(
            gate_scale_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 32), order=(1, 0)
            ),
        )
        up_values = cute.make_tensor(
            up_values_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        up_scales = cute.make_tensor(
            up_scale_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 32), order=(1, 0)
            ),
        )
        w13 = cute.make_tensor(
            w13_ptr,
            cute.make_layout(
                (Int64(2 * self.expert_u32) * Int64(num_experts),)
            ),
        )
        rank_lut = cute.make_tensor(
            rank_lut_ptr,
            cute.make_layout((self.sqg_lut_entries,)),
        )
        pair_modes = cute.make_tensor(
            pair_modes_ptr, cute.make_layout((num_experts,))
        )
        route_experts = cute.make_tensor(route_experts_ptr, cute.make_layout((routes,)))
        gate_out = cute.make_tensor(
            gate_out_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        up_out = cute.make_tensor(
            up_out_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        self.kernel(
            gate_values,
            gate_scales,
            up_values,
            up_scales,
            w13,
            rank_lut,
            pair_modes,
            route_experts,
            gate_out,
            up_out,
            routes,
            num_experts,
        ).launch(
            grid=(routes, 4, 4),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _tile_base(
        self,
        projection: Int32,
        expert: Int32,
        k16: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
        num_experts: Int32,
    ) -> Int64:
        high_base = 0
        if cutlass.const_expr(int(bits) == 3):
            high_base = 8 * 8 * 3
        elif cutlass.const_expr(int(bits) == 4):
            high_base = 8 * 8 * 2
        return (
            Int64(projection)
            * Int64(num_experts)
            * Int64(self.expert_u32)
            + Int64(expert) * Int64(self.expert_u32)
            + Int64(k16) * Int64(_PAIR_U32_PER_K16)
            + Int64(record) * Int64(high_base)
            + Int64(n16) * Int64(8 * int(bits))
        )

    @cute.jit
    def _tile_pair_base(
        self,
        projection: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
        num_experts: Int32,
    ) -> Int64:
        high_base = 0
        if cutlass.const_expr(int(bits) == 3):
            high_base = 8 * 8 * 3
        elif cutlass.const_expr(int(bits) == 4):
            high_base = 8 * 8 * 2
        return (
            Int64(projection) * Int64(num_experts) * Int64(self.expert_u32)
            + Int64(expert) * Int64(self.expert_u32)
            + Int64(k32) * Int64(2 * _PAIR_U32_PER_K16)
            + Int64(record) * Int64(high_base)
            + Int64(n16) * Int64(8 * int(bits))
        )

    @cute.jit
    def _decode_projection_rate(
        self,
        w13: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        n_high: Int32,
        record: Int32,
        projection: Int32,
        num_experts: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        base0 = self._tile_base(
            projection,
            expert,
            k32 * Int32(2),
            n16,
            record,
            bits,
            num_experts,
        )
        base1 = self._tile_base(
            projection,
            expert,
            k32 * Int32(2) + Int32(1),
            n16,
            record,
            bits,
            num_experts,
        )
        return self._pair_native_words(
            w13,
            lane,
            base0,
            base1,
            n_high,
            bits,
            rank_lut_smem_addr,
        )

    @cute.jit
    def _decode_projection(
        self,
        w13: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        n_high: Int32,
        record: Int32,
        pair_mode: Int32,
        projection: Int32,
        num_experts: Int32,
        rank_lut_smem_addr: Int64,
    ):
        b0 = Uint32(0)
        b1 = Uint32(0)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            b0, b1 = self._decode_projection_rate(
                w13, lane, expert, k32, n16, n_high, record,
                projection, num_experts, 3, rank_lut_smem_addr
            )
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            if record == Int32(0):
                b0, b1 = self._decode_projection_rate(
                    w13, lane, expert, k32, n16, n_high, record,
                    projection, num_experts, 2, rank_lut_smem_addr
                )
            else:
                b0, b1 = self._decode_projection_rate(
                    w13, lane, expert, k32, n16, n_high, record,
                    projection, num_experts, 4, rank_lut_smem_addr
                )
        else:
            if pair_mode == Int32(0):
                b0, b1 = self._decode_projection_rate(
                    w13, lane, expert, k32, n16, n_high, record,
                    projection, num_experts, 3, rank_lut_smem_addr
                )
            else:
                if record == Int32(0):
                    b0, b1 = self._decode_projection_rate(
                        w13, lane, expert, k32, n16, n_high, record,
                        projection, num_experts, 2, rank_lut_smem_addr
                    )
                else:
                    b0, b1 = self._decode_projection_rate(
                        w13, lane, expert, k32, n16, n_high, record,
                        projection, num_experts, 4, rank_lut_smem_addr
                    )
        return b0, b1

    @cute.jit
    def _decode_projection_both_rate(
        self,
        w13: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        projection: Int32,
        num_experts: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        base0 = self._tile_pair_base(
            projection, expert, k32, n16, record, bits, num_experts
        )
        base1 = base0 + Int64(_PAIR_U32_PER_K16)
        return self._pair_native_words_both(
            w13, lane, base0, base1, bits, rank_lut_smem_addr
        )

    @cute.jit
    def _decode_projection_both(
        self,
        w13: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        pair_mode: Int32,
        projection: Int32,
        num_experts: Int32,
        rank_lut_smem_addr: Int64,
    ):
        b0_lo = Uint32(0)
        b1_lo = Uint32(0)
        b0_hi = Uint32(0)
        b1_hi = Uint32(0)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                w13, lane, expert, k32, n16, record,
                projection, num_experts, 3, rank_lut_smem_addr
            )
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            if record == Int32(0):
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                    w13, lane, expert, k32, n16, record,
                    projection, num_experts, 2, rank_lut_smem_addr
                )
            else:
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                    w13, lane, expert, k32, n16, record,
                    projection, num_experts, 4, rank_lut_smem_addr
                )
        else:
            if pair_mode == Int32(0):
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                    w13, lane, expert, k32, n16, record,
                    projection, num_experts, 3, rank_lut_smem_addr
                )
            else:
                if record == Int32(0):
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                        w13, lane, expert, k32, n16, record,
                        projection, num_experts, 2, rank_lut_smem_addr
                    )
                else:
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both_rate(
                        w13, lane, expert, k32, n16, record,
                        projection, num_experts, 4, rank_lut_smem_addr
                    )
        return b0_lo, b1_lo, b0_hi, b1_hi

    @cute.kernel
    def kernel(
        self,
        gate_values: cute.Tensor,
        gate_scales: cute.Tensor,
        up_values: cute.Tensor,
        up_scales: cute.Tensor,
        w13: cute.Tensor,
        rank_lut: cute.Tensor,
        pair_modes: cute.Tensor,
        route_experts: cute.Tensor,
        gate_out: cute.Tensor,
        up_out: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        route_idx, record_projection_idx, n_warp_idx = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        route = Int32(route_idx)
        raw_expert = route_experts[route].to(Int32)
        valid = Int32(0)
        expert = Int32(0)
        if raw_expert >= Int32(0) and raw_expert < num_experts:
            valid = Int32(1)
            expert = raw_expert
        pair_mode = Int32(0)
        if valid != Int32(0):
            pair_mode = pair_modes[expert].to(Int32)
        selected = valid
        if cutlass.const_expr(self.pair_mode_filter == 0):
            selected = valid & (pair_mode == Int32(0)).to(Int32)
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            selected = valid & (pair_mode != Int32(0)).to(Int32)

        rank_lut_smem_addr = get_ptr_as_int64(rank_lut, Int32(0))
        input_row = route
        if cutlass.const_expr(self.shared_input):
            input_row = route // Int32(self.topk)
        record_projection = Int32(record_projection_idx)
        record = record_projection & Int32(1)
        projection = record_projection >> Int32(1)
        n_warp = Int32(n_warp_idx)

        acc = tuple(cute.make_rmem_tensor((4,), Float32) for _ in range(4))
        for nt in cutlass.range_constexpr(4):
            acc[nt].fill(0.0)

        warp = tid >> Int32(5)
        # Keep every K32 tile live when the hidden dimension provides fewer
        # tiles than the eight reduction warps.  Production Kimi dimensions
        # divide evenly, while the smaller correctness geometries leave the
        # excess warps idle.
        k32_count = Int32(self.hidden_size // 32)
        if cutlass.const_expr(
            (self.hidden_size // 32) % (self.threads // 32) == 0
        ):
            k32_slice = Int32(
                (self.hidden_size // 32) // (self.threads // 32)
            )
            k32 = warp * k32_slice
            k32_end = k32 + k32_slice
        else:
            k32_slice = Int32(
                (self.hidden_size // 32 + self.threads // 32 - 1)
                // (self.threads // 32)
            )
            k32 = warp * k32_slice
            k32_end = k32 + k32_slice
            if k32_end > k32_count:
                k32_end = k32_count
        if selected == Int32(0):
            k32 = k32_end
        while k32 < k32_end:
            a0, a1, a2, a3, sfa = self._a_fragment(
                gate_values, gate_scales, input_row, k32, lane
            )
            if projection != Int32(0):
                a0, a1, a2, a3, sfa = self._a_fragment(
                    up_values, up_scales, input_row, k32, lane
                )
            if cutlass.const_expr(self.decode_both):
                for npair in cutlass.range_constexpr(2):
                    n8 = (
                        record * Int32(128)
                        + n_warp * Int32(32)
                        + Int32(npair * 16)
                    )
                    n_local = n8 - record * Int32(128)
                    n16 = n_local // Int32(16)
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_projection_both(
                        w13,
                        lane,
                        expert,
                        k32,
                        n16,
                        record,
                        pair_mode,
                        projection,
                        num_experts,
                        rank_lut_smem_addr,
                    )
                    for half in cutlass.range_constexpr(2):
                        nt = npair * 2 + half
                        b0 = b0_lo
                        b1 = b1_lo
                        if cutlass.const_expr(half == 1):
                            b0 = b0_hi
                            b1 = b1_hi
                        frag = acc[nt]
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
            else:
                for nt in cutlass.range_constexpr(4):
                    n8 = (
                        record * Int32(128)
                        + n_warp * Int32(32)
                        + Int32(nt * 8)
                    )
                    n_local = n8 - record * Int32(128)
                    n16 = n_local // Int32(16)
                    n_high = ((n8 & Int32(15)) >= Int32(8)).to(Int32)
                    b0, b1 = self._decode_projection(
                        w13,
                        lane,
                        expert,
                        k32,
                        n16,
                        n_high,
                        record,
                        pair_mode,
                        projection,
                        num_experts,
                        rank_lut_smem_addr,
                    )
                    frag = acc[nt]
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

        # Combine the eight K-slice partials.  Only lanes 0-3 of each warp
        # carry the single live output row (accumulator fragments 0 and 1),
        # so the exchange is 4 lanes x 4 tiles x 2 values per warp.
        red_smem_ptr = cute.arch.alloc_smem(
            Float32, (self.threads // 32) * 32
        )
        red_smem = cute.make_tensor(
            red_smem_ptr,
            cute.make_layout(((self.threads // 32) * 32,)),
        )
        for nt in cutlass.range_constexpr(4):
            for half in cutlass.range_constexpr(2):
                cute.arch.sync_threads()
                if lane < Int32(4):
                    red_smem[warp * Int32(4) + lane] = acc[nt][half]
                cute.arch.sync_threads()
                if warp == Int32(0) and lane < Int32(4):
                    total = Float32(0.0)
                    for source in cutlass.range_constexpr(self.threads // 32):
                        total += red_smem[Int32(source * 4) + lane]
                    acc[nt][half] = total

        write_output = Int32(1)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            write_output = selected | (valid == Int32(0)).to(Int32)
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            write_output = selected
        if warp == Int32(0) and lane < Int32(4) and write_output != Int32(0):
            c = lane & Int32(3)
            for nt in cutlass.range_constexpr(4):
                col = (
                    record * Int32(128)
                    + n_warp * Int32(32)
                    + Int32(nt * 8)
                    + c * Int32(2)
                )
                out0 = acc[nt][0]
                out1 = acc[nt][1]
                if valid == Int32(0):
                    out0 = Float32(0.0)
                    out1 = Float32(0.0)
                if projection == Int32(0):
                    gate_out[route, col] = cutlass.Float16(out0)
                    gate_out[route, col + Int32(1)] = cutlass.Float16(out1)
                else:
                    up_out[route, col] = cutlass.Float16(out0)
                    up_out[route, col + Int32(1)] = cutlass.Float16(out1)


class TrellisW4A8FC2RoutesKernel(_NativeTrellisDecode):
    """Compute route-major down projection in N128 slabs."""

    threads = _THREADS

    def __init__(
        self,
        *,
        hidden_size: int,
        sqg_direct_lut: bool = False,
        pair_mode_filter: int = -1,
        trellis_codebook: str = "sqg_xor_cheb_t12",
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.sqg_direct_lut = bool(sqg_direct_lut)
        self.pair_mode_filter = int(pair_mode_filter)
        if self.pair_mode_filter not in (-1, 0, 1):
            raise ValueError("pair_mode_filter must be -1, 0 (P33), or 1 (P24)")
        self.trellis_codebook = str(trellis_codebook).lower()
        if self.trellis_codebook not in _W4A8_CODEBOOKS:
            raise ValueError(
                "W4A8 trellis codebook must be one of "
                f"{sorted(_W4A8_CODEBOOKS)}, got {self.trellis_codebook!r}"
            )
        if not self.sqg_direct_lut:
            raise ValueError(
                "SQG-XOR-Cheb-T12 W4A8 requires T12 table execution"
            )
        self.sqg_rank_lut_entries = 1
        self.sqg_lut_entries = 4096 if self.sqg_direct_lut else 1
        self.n16_count = self.hidden_size // 16
        self.expert_u32 = 256 * self.hidden_size * 3 // 32

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        pair_modes_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr, cute.make_ordered_layout((routes, 256 // 4), order=(1, 0))
        )
        scales = cute.make_tensor(
            scale_ptr, cute.make_ordered_layout((routes, 256 // 32), order=(1, 0))
        )
        w2 = cute.make_tensor(
            w2_ptr,
            cute.make_layout((Int64(num_experts) * Int64(self.expert_u32),)),
        )
        rank_lut = cute.make_tensor(
            rank_lut_ptr,
            cute.make_layout((self.sqg_lut_entries,)),
        )
        pair_modes = cute.make_tensor(
            pair_modes_ptr, cute.make_layout((num_experts,))
        )
        route_experts = cute.make_tensor(route_experts_ptr, cute.make_layout((routes,)))
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((routes, self.hidden_size), order=(1, 0)),
        )
        self.kernel(
            values,
            scales,
            w2,
            rank_lut,
            pair_modes,
            route_experts,
            output,
            routes,
            num_experts,
        ).launch(
            grid=(routes, self.hidden_size // 128, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _tile_base(
        self,
        expert: Int32,
        k16: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
    ) -> Int64:
        high_base = 0
        if cutlass.const_expr(int(bits) == 3):
            high_base = 8 * self.n16_count * 8 * 3
        elif cutlass.const_expr(int(bits) == 4):
            high_base = 8 * self.n16_count * 8 * 2
        local_k16 = k16 - record * Int32(8)
        return (
            Int64(expert) * Int64(self.expert_u32)
            + Int64(record) * Int64(high_base)
            + Int64(local_k16) * Int64(self.n16_count * 8 * int(bits))
            + Int64(n16) * Int64(8 * int(bits))
        )

    @cute.jit
    def _tile_pair_base(
        self,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
    ) -> Int64:
        high_base = 0
        if cutlass.const_expr(int(bits) == 3):
            high_base = 8 * self.n16_count * 8 * 3
        elif cutlass.const_expr(int(bits) == 4):
            high_base = 8 * self.n16_count * 8 * 2
        local_k16 = k32 * Int32(2) - record * Int32(8)
        return (
            Int64(expert) * Int64(self.expert_u32)
            + Int64(record) * Int64(high_base)
            + Int64(local_k16) * Int64(self.n16_count * 8 * int(bits))
            + Int64(n16) * Int64(8 * int(bits))
        )

    @cute.jit
    def _decode_k32_rate(
        self,
        w2: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        n_high: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        base0 = self._tile_base(
            expert, k32 * Int32(2), n16, record, bits
        )
        base1 = self._tile_base(
            expert, k32 * Int32(2) + Int32(1), n16, record, bits
        )
        return self._pair_native_words(
            w2, lane, base0, base1, n_high, bits, rank_lut_smem_addr
        )

    @cute.jit
    def _decode_k32(
        self,
        w2: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        n_high: Int32,
        pair_mode: Int32,
        rank_lut_smem_addr: Int64,
    ):
        record = k32 // Int32(4)
        b0 = Uint32(0)
        b1 = Uint32(0)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            b0, b1 = self._decode_k32_rate(
                w2, lane, expert, k32, n16, n_high, record,
                3, rank_lut_smem_addr
            )
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            if record == Int32(0):
                b0, b1 = self._decode_k32_rate(
                    w2, lane, expert, k32, n16, n_high, record,
                    2, rank_lut_smem_addr
                )
            else:
                b0, b1 = self._decode_k32_rate(
                    w2, lane, expert, k32, n16, n_high, record,
                    4, rank_lut_smem_addr
                )
        else:
            if pair_mode == Int32(0):
                b0, b1 = self._decode_k32_rate(
                    w2, lane, expert, k32, n16, n_high, record,
                    3, rank_lut_smem_addr
                )
            else:
                if record == Int32(0):
                    b0, b1 = self._decode_k32_rate(
                        w2, lane, expert, k32, n16, n_high, record,
                        2, rank_lut_smem_addr
                    )
                else:
                    b0, b1 = self._decode_k32_rate(
                        w2, lane, expert, k32, n16, n_high, record,
                        4, rank_lut_smem_addr
                    )
        return b0, b1

    @cute.jit
    def _decode_k32_both_rate(
        self,
        w2: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_smem_addr: Int64,
    ):
        base0 = self._tile_pair_base(expert, k32, n16, record, bits)
        base1 = base0 + Int64(self.n16_count * 8 * int(bits))
        return self._pair_native_words_both(
            w2, lane, base0, base1, bits, rank_lut_smem_addr
        )

    @cute.jit
    def _decode_k32_both(
        self,
        w2: cute.Tensor,
        lane: Int32,
        expert: Int32,
        k32: Int32,
        n16: Int32,
        pair_mode: Int32,
        rank_lut_smem_addr: Int64,
    ):
        record = k32 // Int32(4)
        b0_lo = Uint32(0)
        b1_lo = Uint32(0)
        b0_hi = Uint32(0)
        b1_hi = Uint32(0)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                w2, lane, expert, k32, n16, record, 3, rank_lut_smem_addr
            )
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            if record == Int32(0):
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                    w2, lane, expert, k32, n16, record,
                    2, rank_lut_smem_addr
                )
            else:
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                    w2, lane, expert, k32, n16, record,
                    4, rank_lut_smem_addr
                )
        else:
            if pair_mode == Int32(0):
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                    w2, lane, expert, k32, n16, record,
                    3, rank_lut_smem_addr
                )
            else:
                if record == Int32(0):
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                        w2, lane, expert, k32, n16, record,
                        2, rank_lut_smem_addr
                    )
                else:
                    b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both_rate(
                        w2, lane, expert, k32, n16, record,
                        4, rank_lut_smem_addr
                    )
        return b0_lo, b1_lo, b0_hi, b1_hi

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scales: cute.Tensor,
        w2: cute.Tensor,
        rank_lut: cute.Tensor,
        pair_modes: cute.Tensor,
        route_experts: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        route_idx, output_tile_idx, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        route = Int32(route_idx)
        output_tile = Int32(output_tile_idx)
        raw_expert = route_experts[route].to(Int32)
        valid = Int32(0)
        expert = Int32(0)
        if raw_expert >= Int32(0) and raw_expert < num_experts:
            valid = Int32(1)
            expert = raw_expert
        pair_mode = Int32(0)
        if valid != Int32(0):
            pair_mode = pair_modes[expert].to(Int32)
        selected = valid
        if cutlass.const_expr(self.pair_mode_filter == 0):
            selected = valid & (pair_mode == Int32(0)).to(Int32)
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            selected = valid & (pair_mode != Int32(0)).to(Int32)

        rank_lut_smem_addr = get_ptr_as_int64(rank_lut, Int32(0))
        acc = tuple(cute.make_rmem_tensor((4,), Float32) for _ in range(4))
        for nt in cutlass.range_constexpr(4):
            acc[nt].fill(0.0)

        k32 = Int32(0)
        if selected == Int32(0):
            k32 = Int32(8)
        while k32 < Int32(8):
            a0, a1, a2, a3, sfa = self._a_fragment(
                values, scales, route, k32, lane
            )
            for npair in cutlass.range_constexpr(2):
                n8 = (
                    output_tile * Int32(128)
                    + warp * Int32(32)
                    + Int32(npair * 16)
                )
                n16 = n8 // Int32(16)
                b0_lo, b1_lo, b0_hi, b1_hi = self._decode_k32_both(
                    w2,
                    lane,
                    expert,
                    k32,
                    n16,
                    pair_mode,
                    rank_lut_smem_addr,
                )
                lo_idx = npair * 2
                hi_idx = lo_idx + 1
                frag = acc[lo_idx]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    frag[0],
                    frag[1],
                    frag[2],
                    frag[3],
                    a0,
                    a1,
                    a2,
                    a3,
                    b0_lo,
                    b1_lo,
                    sfa,
                    Uint32(0x7F7F7F7F),
                )
                frag[0] = d0
                frag[1] = d1
                frag[2] = d2
                frag[3] = d3
                frag = acc[hi_idx]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    frag[0],
                    frag[1],
                    frag[2],
                    frag[3],
                    a0,
                    a1,
                    a2,
                    a3,
                    b0_hi,
                    b1_hi,
                    sfa,
                    Uint32(0x7F7F7F7F),
                )
                frag[0] = d0
                frag[1] = d1
                frag[2] = d2
                frag[3] = d3
            k32 += Int32(1)

        write_output = Int32(1)
        if cutlass.const_expr(self.pair_mode_filter == 0):
            write_output = selected | (valid == Int32(0)).to(Int32)
        elif cutlass.const_expr(self.pair_mode_filter == 1):
            write_output = selected
        if lane < Int32(4) and write_output != Int32(0):
            c = lane & Int32(3)
            for nt in cutlass.range_constexpr(4):
                col = (
                    output_tile * Int32(128)
                    + warp * Int32(32)
                    + Int32(nt * 8)
                    + c * Int32(2)
                )
                out0 = acc[nt][0]
                out1 = acc[nt][1]
                if valid == Int32(0):
                    out0 = Float32(0.0)
                    out1 = Float32(0.0)
                output[route, col] = cutlass.Float16(out0)
                output[route, col + Int32(1)] = cutlass.Float16(out1)


def _ptr(dtype, address: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _compile_fc1(
    hidden_size: int,
    topk: int,
    shared_input: bool,
    decode_both: bool,
    sqg_direct_lut: bool,
    pair_mode_filter: int,
    trellis_codebook: str,
    device_index: int,
):
    launch = TrellisW4A8FC1RoutesKernel(
        hidden_size=hidden_size,
        topk=topk,
        shared_input=shared_input,
        decode_both=decode_both,
        sqg_direct_lut=sqg_direct_lut,
        pair_mode_filter=pair_mode_filter,
        trellis_codebook=trellis_codebook,
    )
    key = (
        int(hidden_size),
        int(topk),
        bool(shared_input),
        bool(decode_both),
        bool(sqg_direct_lut),
        int(pair_mode_filter),
        trellis_codebook,
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(
            cutlass.Uint8,
            16,
        ),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("moe.trellis_w4a8_fc1_routes", 2, key),
    )


@functools.cache
def _compile_fc2(
    hidden_size: int,
    sqg_direct_lut: bool,
    pair_mode_filter: int,
    trellis_codebook: str,
    device_index: int,
):
    launch = TrellisW4A8FC2RoutesKernel(
        hidden_size=hidden_size,
        sqg_direct_lut=sqg_direct_lut,
        pair_mode_filter=pair_mode_filter,
        trellis_codebook=trellis_codebook,
    )
    key = (
        int(hidden_size),
        bool(sqg_direct_lut),
        int(pair_mode_filter),
        trellis_codebook,
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(
            cutlass.Uint8,
            16,
        ),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("moe.trellis_w4a8_fc2_routes", 2, key),
    )


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def run_trellis_w4a8_fc1_routes(
    gate_a: MXFP8Rows,
    up_a: MXFP8Rows,
    prepared,
    route_experts: torch.Tensor,
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    *,
    topk: int = 1,
    shared_input: bool = False,
    decode_both: bool = True,
    sqg_direct_lut: bool = False,
    pair_mode_filter: int = -1,
) -> None:
    """Launch route-major FC1 against prepared paired-rate weights."""
    routes = int(route_experts.numel())
    trellis_codebook = str(getattr(prepared, "trellis_codebook", "")).lower()
    if trellis_codebook not in _W4A8_CODEBOOKS:
        raise NotImplementedError(
            "W4A8 trellis decode requires one of "
            f"{sorted(_W4A8_CODEBOOKS)}, got {trellis_codebook!r}"
        )
    compiled = _compile_fc1(
        int(prepared.hidden_size),
        int(topk),
        bool(shared_input),
        bool(decode_both),
        bool(sqg_direct_lut),
        int(pair_mode_filter),
        trellis_codebook,
        _device_index(route_experts.device),
    )
    if not sqg_direct_lut:
        raise ValueError("SQG-XOR-Cheb-T12 W4A8 requires T12 table execution")
    # The 4 KiB modal T12 staircase serves every rate.
    rank_lut = sqg_xor_cheb_t12_lut(route_experts.device)
    compiled(
        _ptr(cutlass.Uint32, gate_a.values.data_ptr()),
        _ptr(cutlass.Uint8, gate_a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, up_a.values.data_ptr()),
        _ptr(cutlass.Uint8, up_a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, prepared.w13.data_ptr()),
        _ptr(
            cutlass.Uint8,
            rank_lut.data_ptr(),
        ),
        _ptr(cutlass.Int32, prepared.fc1_trellis_pair_modes.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, gate_out.data_ptr()),
        _ptr(cutlass.Float16, up_out.data_ptr()),
        routes,
        int(prepared.num_experts),
        current_cuda_stream(),
    )


def run_trellis_w4a8_fc2_routes(
    a: MXFP8Rows,
    prepared,
    route_experts: torch.Tensor,
    output: torch.Tensor,
    *,
    sqg_direct_lut: bool = False,
    pair_mode_filter: int = -1,
) -> None:
    """Launch route-major FC2 against prepared paired-rate weights."""
    routes = int(route_experts.numel())
    trellis_codebook = str(getattr(prepared, "trellis_codebook", "")).lower()
    if trellis_codebook not in _W4A8_CODEBOOKS:
        raise NotImplementedError(
            "W4A8 trellis decode requires one of "
            f"{sorted(_W4A8_CODEBOOKS)}, got {trellis_codebook!r}"
        )
    compiled = _compile_fc2(
        int(prepared.hidden_size),
        bool(sqg_direct_lut),
        int(pair_mode_filter),
        trellis_codebook,
        _device_index(route_experts.device),
    )
    if not sqg_direct_lut:
        raise ValueError("SQG-XOR-Cheb-T12 W4A8 requires T12 table execution")
    rank_lut = sqg_xor_cheb_t12_lut(route_experts.device)
    compiled(
        _ptr(cutlass.Uint32, a.values.data_ptr()),
        _ptr(cutlass.Uint8, a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, prepared.w2.data_ptr()),
        _ptr(
            cutlass.Uint8,
            rank_lut.data_ptr(),
        ),
        _ptr(cutlass.Int32, prepared.fc2_trellis_pair_modes.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, output.data_ptr()),
        routes,
        int(prepared.num_experts),
        current_cuda_stream(),
    )


__all__ = [
    "TrellisW4A8FC1RoutesKernel",
    "TrellisW4A8FC2RoutesKernel",
    "run_trellis_w4a8_fc1_routes",
    "run_trellis_w4a8_fc2_routes",
]
