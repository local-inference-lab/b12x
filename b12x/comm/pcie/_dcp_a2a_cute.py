"""CuTe DSL kernels for the PCIe DCP attention exchange."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    fmax_f32,
    ld_global_v4_u32,
    st_global_v4_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._dcp_cute_common import block_pair_barrier
from ._cute_intrinsics import (
    ld_generic_f32,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    unpack_bf16x2,
    unpack_f16x2,
)


_MAX_BLOCKS = 64
_MAX_RANKS = 16
_SLOT_ALIGNMENT = 256
_SELF_COUNTER_WORDS = _MAX_BLOCKS * _MAX_RANKS
# The barrier consumes only word zero of each 32-word flag record. Keep the
# graph epoch in its first two padding words so the native signal layout and
# slab capacity remain unchanged.
_GRAPH_EPOCH_INDEX = _SELF_COUNTER_WORDS + 1
_GRAPH_ARRIVED_INDEX = _SELF_COUNTER_WORDS + 2

_PREPARED_LSE_LAUNCHERS: set[tuple[object, ...]] = set()
_PREPARED_GATHER_LAUNCHERS: set[tuple[object, ...]] = set()
_PREPARED_PAIR_LAUNCHERS: set[tuple[object, ...]] = set()


@dsl_user_op
def _a2a_graph_epoch_arrive(
    epoch_addr: Int64,
    arrived_addr: Int64,
    blocks: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Retire one A2A graph generation on its serialized local stream.

    Every CTA calls this once after all of its threads have loaded the current
    epoch.  The final CTA is the only epoch writer.  A subsequent A2A kernel
    using this channel cannot start until this kernel completes, so kernel
    ordering makes the reset and plain epoch store visible without another
    GPU fence or an atomic epoch update.  A one-CTA grid needs no arrival
    accounting at all.
    """

    llvm.inline_asm(
        None,
        [
            Int64(epoch_addr).ir_value(loc=loc, ip=ip),
            Int64(arrived_addr).ir_value(loc=loc, ip=ip),
            Uint32(blocks).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred single_block, not_last;
            .reg .b32 prior, last, generation;
            setp.eq.u32 single_block, $2, 1;
            @single_block bra a2a_epoch_advance;
            sub.u32 last, $2, 1;
            atom.global.add.u32 prior, [$1], 1;
            setp.ne.u32 not_last, prior, last;
            @not_last bra a2a_epoch_arrive_done;
            st.global.u32 [$1], 0;
        a2a_epoch_advance:
            ld.global.u32 generation, [$0];
            add.u32 generation, generation, 1;
            st.global.u32 [$0], generation;
        a2a_epoch_arrive_done:
        }
        """,
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _copy_16b(source: cute.Pointer, destination: cute.Pointer) -> None:
    values = ld_global_v4_u32(source.toint())
    st_global_v4_u32(destination.toint(), *values)


@dsl_user_op
def _fma_rn_f32(
    a: Float32,
    b: Float32,
    c: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_generic_v4_u32(
    address: Int64,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load one 16-byte payload through a generic/UVA address."""

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def _add_u64_opaque(
    base: Int64,
    offset: Int64,
    *,
    loc=None,
    ip=None,
) -> Int64:
    """Add a payload offset without CSE into the earlier LSE address phase."""

    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Int64(base).ir_value(loc=loc, ip=ip),
                Int64(offset).ir_value(loc=loc, ip=ip),
            ],
            "add.u64 $0, $1, $2;",
            "=l,l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _DCPA2ABase:
    def __init__(
        self,
        world_size: int,
        rank: int,
        threads: int,
        device_slot_selection: bool,
    ) -> None:
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._warps_per_block = self._threads // 32
        self._device_slot_selection = bool(device_slot_selection)

    @cute.jit
    def _staging_words(self, pointer: cute.Pointer) -> cute.Pointer:
        return cute.recast_ptr(pointer.align(16), dtype=Uint32)

    @cute.jit
    def _staging_lse(
        self,
        pointer: cute.Pointer,
        lse_offset: Int64,
    ) -> cute.Pointer:
        return cute.recast_ptr((pointer + lse_offset).align(4), dtype=Float32)

class _LseReduceScatterLaunch(_DCPA2ABase):
    def __init__(
        self,
        world_size: int,
        rank: int,
        dtype_name: str,
        threads: int,
        device_slot_selection: bool,
    ) -> None:
        super().__init__(
            world_size,
            rank,
            threads,
            device_slot_selection,
        )
        self._dtype_name = str(dtype_name)

    @cute.jit
    def __call__(
        self,
        local_output: cute.Pointer,
        local_lse: cute.Pointer,
        output: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        lse_offset: Int64,
        batch: Int32,
        total_heads: Int32,
        head_dim: Int32,
        input_stride_batch: Int64,
        input_stride_head: Int64,
        output_stride_batch: Int64,
        output_stride_head: Int64,
        natural_log: Int32,
        slot_delta_256b: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            local_output,
            local_lse,
            output,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            lse_offset,
            batch,
            total_heads,
            head_dim,
            input_stride_batch,
            input_stride_head,
            output_stride_batch,
            output_stride_head,
            natural_log,
            slot_delta_256b,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self._threads, 1, 1),
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _unpack_pair(self, value: Uint32) -> tuple[Float32, Float32]:
        if cutlass.const_expr(self._dtype_name == "fp16"):
            return unpack_f16x2(value)
        return unpack_bf16x2(value)

    @cute.jit
    def _pack_pair(self, lo: Float32, hi: Float32) -> Uint32:
        if cutlass.const_expr(self._dtype_name == "fp16"):
            return pack_f32x2_to_f16x2(lo, hi)
        return pack_f32x2_to_bf16x2(lo, hi)

    @cute.kernel
    def kernel(
        self,
        local_output: cute.Pointer,
        local_lse: cute.Pointer,
        output: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        lse_offset: Int64,
        batch: Int32,
        total_heads: Int32,
        head_dim: Int32,
        input_stride_batch: Int64,
        input_stride_head: Int64,
        output_stride_batch: Int64,
        output_stride_head: Int64,
        natural_log: Int32,
        slot_delta_256b: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot = generation % Uint32(2)
            slot_offset = (
                Int64(slot)
                * Int64(slot_delta_256b)
                * Int64(_SLOT_ALIGNMENT)
            )
        lane = Int32(tidx) % Int32(32)
        warp_first = (
            Int32(bidx) * Int32(self._warps_per_block)
            + Int32(tidx) // Int32(32)
        )
        warp_stride = Int32(gdim) * Int32(self._warps_per_block)
        heads_per_rank = total_heads // Int32(self._world_size)
        packs_per_head = head_dim // Int32(8)
        rows = batch * heads_per_rank
        # Keep the remote staging bases in their scalar launch parameters.
        # Materializing all eight slot-adjusted pointers here keeps them live
        # through the copy and barrier phases, which materially increases the
        # graph kernel's register footprint.  Only the local pointer is needed
        # before the barrier; form each remote address at its eventual use.
        local_staging = staging[self._rank] + slot_offset
        local_stage_words = self._staging_words(local_staging)
        local_stage_lse = self._staging_lse(local_staging, lse_offset)

        row = warp_first
        while row < rows:
            batch_index = row // heads_per_rank
            local_head = row - batch_index * heads_per_rank
            for destination in cutlass.range_constexpr(self._world_size):
                source_row = (
                    Int64(batch_index) * Int64(total_heads)
                    + Int64(destination) * Int64(heads_per_rank)
                    + Int64(local_head)
                )
                input_base = (
                    Int64(batch_index) * input_stride_batch
                    + (
                        Int64(destination) * Int64(heads_per_rank)
                        + Int64(local_head)
                    )
                    * input_stride_head
                )
                staging_base = source_row * Int64(packs_per_head)
                pack = lane
                while pack < packs_per_head:
                    _copy_16b(
                        local_output + input_base * Int64(4) + Int64(pack) * Int64(4),
                        local_stage_words
                        + staging_base * Int64(4)
                        + Int64(pack) * Int64(4),
                    )
                    pack += Int32(32)
                if lane == Int32(0):
                    value = cute.arch.load(local_lse + source_row, Float32)
                    cute.arch.store(local_stage_lse + source_row, value)
            row += warp_stride

        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )

        row = warp_first
        while row < rows:
            batch_index = row // heads_per_rank
            local_head = row - batch_index * heads_per_rank
            global_head = Int32(self._rank) * heads_per_rank + local_head
            source_row = (
                Int64(batch_index) * Int64(total_heads) + Int64(global_head)
            )

            # Match the native lane-selected pointer shape: choose one source
            # address, then issue one runtime-guarded LSE load.  Spelling this
            # as eight predicated loads makes every peer address/sanitize chain
            # part of the hot path and materially lengthens the graph kernel.
            lane_lse_base = Int64(local_lse.toint())
            for source_index in cutlass.range_constexpr(1, self._world_size):
                source = (self._rank + source_index) % self._world_size
                source_lse_base = (
                    Int64(staging[source].toint())
                    + slot_offset
                    + lse_offset
                )
                if lane == Int32(source_index):
                    lane_lse_base = source_lse_base

            lane_lse = Float32(float("-inf"))
            if lane < Int32(self._world_size):
                lane_lse = ld_generic_f32(
                    lane_lse_base + source_row * Int64(4)
                )
                if not cute.math.isfinite(lane_lse):
                    lane_lse = Float32(float("-inf"))

            weights = cute.make_rmem_tensor((self._world_size,), Float32)
            max_lse = Float32(float("-inf"))
            for source_index in cutlass.range_constexpr(self._world_size):
                value = cute.arch.shuffle_sync(lane_lse, offset=source_index)
                weights[source_index] = value
                max_lse = fmax_f32(max_lse, value)
            if not cute.math.isfinite(max_lse):
                max_lse = Float32(0.0)

            weight_sum = Float32(0.0)
            if natural_log != Int32(0):
                for source_index in cutlass.range_constexpr(self._world_size):
                    value = weights[source_index]
                    delta = value - max_lse
                    weight = Float32(0.0)
                    if cute.math.isfinite(value):
                        weight = cute.math.exp(delta, fastmath=False)
                    weights[source_index] = weight
                    weight_sum += weight
            else:
                for source_index in cutlass.range_constexpr(self._world_size):
                    value = weights[source_index]
                    delta = value - max_lse
                    weight = Float32(0.0)
                    if cute.math.isfinite(value):
                        weight = cute.math.exp2(delta, approx=True)
                    weights[source_index] = weight
                    weight_sum += weight
            inv_weight_sum = Float32(1.0) / fmax_f32(
                weight_sum, Float32(1.0e-10)
            )

            staging_base = source_row * Int64(packs_per_head)
            local_base = (
                Int64(batch_index) * input_stride_batch
                + Int64(global_head) * input_stride_head
            )
            output_base = (
                Int64(batch_index) * output_stride_batch
                + Int64(local_head) * output_stride_head
            )
            payload_row_addresses = cute.make_rmem_tensor(
                (self._world_size,), Int64
            )
            for source_index in cutlass.range_constexpr(self._world_size):
                source = (self._rank + source_index) % self._world_size
                if cutlass.const_expr(source == self._rank):
                    source_address = (
                        local_output.toint() + local_base * Int64(16)
                    )
                else:
                    source_address = _add_u64_opaque(
                        staging[source].toint(),
                        slot_offset + staging_base * Int64(16),
                    )
                payload_row_addresses[source_index] = source_address
            for pack in cutlass.range(
                lane,
                packs_per_head,
                Int32(32),
                unroll=1,
            ):
                accum = cute.make_rmem_tensor((8,), Float32)
                source_addresses = cute.make_rmem_tensor(
                    (self._world_size,), Int64
                )
                for element in cutlass.range_constexpr(8):
                    accum[element] = Float32(0.0)
                for source_index in cutlass.range_constexpr(self._world_size):
                    source_addresses[source_index] = (
                        payload_row_addresses[source_index]
                        + Int64(pack) * Int64(16)
                    )
                for source_index in cutlass.range_constexpr(self._world_size):
                    normalized_weight = weights[source_index] * inv_weight_sum
                    if normalized_weight != Float32(0.0):
                        words = _ld_generic_v4_u32(
                            source_addresses[source_index]
                        )
                        for pair in cutlass.range_constexpr(4):
                            lo, hi = self._unpack_pair(words[pair])
                            accum[pair * 2] = _fma_rn_f32(
                                normalized_weight, lo, accum[pair * 2]
                            )
                            accum[pair * 2 + 1] = _fma_rn_f32(
                                normalized_weight, hi, accum[pair * 2 + 1]
                            )
                result0 = self._pack_pair(accum[0], accum[1])
                result1 = self._pack_pair(accum[2], accum[3])
                result2 = self._pack_pair(accum[4], accum[5])
                result3 = self._pack_pair(accum[6], accum[7])
                st_global_v4_u32(
                    (
                        output
                        + output_base * Int64(4)
                        + Int64(pack) * Int64(4)
                    ).toint(),
                    result0,
                    result1,
                    result2,
                    result3,
                )
            row += warp_stride

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                self_signal = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (self_signal + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (self_signal + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(gdim),
                )


class _AllGatherHeadsLaunch(_DCPA2ABase):
    @cute.jit
    def __call__(
        self,
        local_input: cute.Pointer,
        output: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        batch: Int32,
        local_heads: Int32,
        packs_per_head: Int32,
        slot_delta_256b: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            local_input,
            output,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            batch,
            local_heads,
            packs_per_head,
            slot_delta_256b,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        local_input: cute.Pointer,
        output: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        batch: Int32,
        local_heads: Int32,
        packs_per_head: Int32,
        slot_delta_256b: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot = generation % Uint32(2)
            slot_offset = (
                Int64(slot)
                * Int64(slot_delta_256b)
                * Int64(_SLOT_ALIGNMENT)
            )
            staging = (
                staging0 + slot_offset,
                staging1 + slot_offset,
                staging2 + slot_offset,
                staging3 + slot_offset,
                staging4 + slot_offset,
                staging5 + slot_offset,
                staging6 + slot_offset,
                staging7 + slot_offset,
                staging8 + slot_offset,
                staging9 + slot_offset,
                staging10 + slot_offset,
                staging11 + slot_offset,
                staging12 + slot_offset,
                staging13 + slot_offset,
                staging14 + slot_offset,
                staging15 + slot_offset,
            )
        lane = Int32(tidx) % Int32(32)
        warp_first = (
            Int32(bidx) * Int32(self._warps_per_block)
            + Int32(tidx) // Int32(32)
        )
        warp_stride = Int32(gdim) * Int32(self._warps_per_block)
        total_heads = local_heads * Int32(self._world_size)
        rows = batch * total_heads
        local_stage = self._staging_words(staging[self._rank])

        row = warp_first
        while row < rows:
            batch_index = row // total_heads
            global_head = row - batch_index * total_heads
            source_rank = global_head // local_heads
            if source_rank == Int32(self._rank):
                local_head = global_head - source_rank * local_heads
                base = (
                    (Int64(batch_index) * Int64(local_heads) + Int64(local_head))
                    * Int64(packs_per_head)
                )
                pack = lane
                while pack < packs_per_head:
                    _copy_16b(
                        local_input + base * Int64(4) + Int64(pack) * Int64(4),
                        local_stage + base * Int64(4) + Int64(pack) * Int64(4),
                    )
                    pack += Int32(32)
            row += warp_stride

        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )

        row = warp_first
        while row < rows:
            batch_index = row // total_heads
            global_head = row - batch_index * total_heads
            source_rank = global_head // local_heads
            local_head = global_head - source_rank * local_heads
            source_base = (
                (Int64(batch_index) * Int64(local_heads) + Int64(local_head))
                * Int64(packs_per_head)
            )
            output_base = Int64(row) * Int64(packs_per_head)
            for source in cutlass.range_constexpr(self._world_size):
                if source_rank == Int32(source):
                    if cutlass.const_expr(source == self._rank):
                        source_words = local_input
                    else:
                        source_words = self._staging_words(staging[source])
                    pack = lane
                    while pack < packs_per_head:
                        _copy_16b(
                            source_words
                            + source_base * Int64(4)
                            + Int64(pack) * Int64(4),
                            output
                            + output_base * Int64(4)
                            + Int64(pack) * Int64(4),
                        )
                        pack += Int32(32)
            row += warp_stride

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                self_signal = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (self_signal + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (self_signal + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(gdim),
                )


class _AllGatherPairLaunch(_DCPA2ABase):
    def __init__(
        self,
        world_size: int,
        rank: int,
        threads: int,
        device_slot_selection: bool,
        kimi_topk: bool,
    ) -> None:
        super().__init__(world_size, rank, threads, device_slot_selection)
        self._kimi_topk = bool(kimi_topk)

    @cute.jit
    def __call__(
        self,
        local_first: cute.Pointer,
        local_second: cute.Pointer,
        correction_bias: cute.Pointer,
        output_first: cute.Pointer,
        output_second: cute.Pointer,
        output_ids: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        batch: Int32,
        first_packs: Int32,
        second_packs: Int32,
        slot_delta_256b: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            local_first,
            local_second,
            correction_bias,
            output_first,
            output_second,
            output_ids,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            batch,
            first_packs,
            second_packs,
            slot_delta_256b,
        ).launch(
            grid=(1, 1, 1),
            block=(self._threads, 1, 1),
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        local_first: cute.Pointer,
        local_second: cute.Pointer,
        correction_bias: cute.Pointer,
        output_first: cute.Pointer,
        output_second: cute.Pointer,
        output_ids: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        batch: Int32,
        first_packs: Int32,
        second_packs: Int32,
        slot_delta_256b: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot = generation % Uint32(2)
            slot_offset = (
                Int64(slot)
                * Int64(slot_delta_256b)
                * Int64(_SLOT_ALIGNMENT)
            )
        combined_packs = first_packs + second_packs
        local_stage = self._staging_words(
            staging[self._rank] + slot_offset
        )

        linear = Int32(tidx)
        while linear < batch * first_packs:
            batch_index = linear // first_packs
            pack = linear - batch_index * first_packs
            _copy_16b(
                local_first + Int64(linear) * Int64(4),
                local_stage
                + (
                    Int64(batch_index) * Int64(combined_packs)
                    + Int64(pack)
                )
                * Int64(4),
            )
            linear += Int32(self._threads)
        linear = Int32(tidx)
        while linear < batch * second_packs:
            batch_index = linear // second_packs
            pack = linear - batch_index * second_packs
            _copy_16b(
                local_second + Int64(linear) * Int64(4),
                local_stage
                + (
                    Int64(batch_index) * Int64(combined_packs)
                    + Int64(first_packs)
                    + Int64(pack)
                )
                * Int64(4),
            )
            linear += Int32(self._threads)

        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )

        first_output_packs = batch * Int32(self._world_size) * first_packs
        linear = Int32(tidx)
        while linear < first_output_packs:
            batch_index = linear // (Int32(self._world_size) * first_packs)
            row_pack = linear - (
                batch_index * Int32(self._world_size) * first_packs
            )
            source_rank = row_pack // first_packs
            pack = row_pack - source_rank * first_packs
            for source in cutlass.range_constexpr(self._world_size):
                if source_rank == Int32(source):
                    if cutlass.const_expr(source == self._rank):
                        source_words = local_first
                        source_base = Int64(batch_index) * Int64(first_packs)
                    else:
                        source_words = self._staging_words(
                            staging[source] + slot_offset
                        )
                        source_base = (
                            Int64(batch_index) * Int64(combined_packs)
                        )
                    _copy_16b(
                        source_words
                        + (source_base + Int64(pack)) * Int64(4),
                        output_first + Int64(linear) * Int64(4),
                    )
            linear += Int32(self._threads)

        if cutlass.const_expr(not self._kimi_topk):
            second_output_packs = (
                batch * Int32(self._world_size) * second_packs
            )
            linear = Int32(tidx)
            while linear < second_output_packs:
                batch_index = linear // (
                    Int32(self._world_size) * second_packs
                )
                row_pack = linear - (
                    batch_index * Int32(self._world_size) * second_packs
                )
                source_rank = row_pack // second_packs
                pack = row_pack - source_rank * second_packs
                for source in cutlass.range_constexpr(self._world_size):
                    if source_rank == Int32(source):
                        if cutlass.const_expr(source == self._rank):
                            source_words = local_second
                            source_base = (
                                Int64(batch_index) * Int64(second_packs)
                            )
                        else:
                            source_words = self._staging_words(
                                staging[source] + slot_offset
                            )
                            source_base = (
                                Int64(batch_index) * Int64(combined_packs)
                                + Int64(first_packs)
                            )
                        _copy_16b(
                            source_words
                            + (source_base + Int64(pack)) * Int64(4),
                            output_second + Int64(linear) * Int64(4),
                        )
                linear += Int32(self._threads)
        else:
            smem_alloc = cutlass.utils.SmemAllocator()

            @cute.struct
            class SharedStorage:
                selection_scores: cute.struct.Align[
                    cute.struct.MemRange[Float32, 896], 16
                ]
                unbiased_scores: cute.struct.Align[
                    cute.struct.MemRange[Float32, 896], 16
                ]

            storage = smem_alloc.allocate(SharedStorage)
            selection_scores = storage.selection_scores.get_tensor(
                cute.make_layout((896,), stride=(1,))
            )
            unbiased_scores = storage.unbiased_scores.get_tensor(
                cute.make_layout((896,), stride=(1,))
            )

            expert = Int32(tidx)
            while expert < Int32(896):
                source_rank = expert // Int32(56)
                local_expert = expert - source_rank * Int32(56)
                router_value = Float32(0.0)
                for source in cutlass.range_constexpr(self._world_size):
                    if source_rank == Int32(source):
                        if cutlass.const_expr(source == self._rank):
                            source_router = cute.recast_ptr(
                                local_second.align(4), dtype=Float32
                            )
                        else:
                            source_router = cute.recast_ptr(
                                (
                                    staging[source]
                                    + slot_offset
                                    + Int64(first_packs) * Int64(16)
                                ).align(4),
                                dtype=Float32,
                            )
                        router_value = cute.arch.load(
                            source_router + Int64(local_expert), Float32
                        )
                bias = cute.arch.load(
                    correction_bias + Int64(expert), Float32
                )
                unbiased = Float32(0.0)
                selection = Float32(float("-inf"))
                if cute.math.isfinite(router_value) and cute.math.isfinite(bias):
                    unbiased = (
                        Float32(0.5)
                        * cute.math.tanh(
                            Float32(0.5) * router_value, fastmath=False
                        )
                        + Float32(0.5)
                    )
                    selection = unbiased + bias
                    if selection == Float32(0.0):
                        selection = Float32(0.0)
                selection_scores[expert] = selection
                unbiased_scores[expert] = unbiased
                expert += Int32(self._threads)
            cute.arch.sync_threads()

            if Int32(tidx) == Int32(0):
                selected_ids = cute.make_rmem_tensor((16,), Int32)
                selected_weights = cute.make_rmem_tensor((16,), Float32)
                for selected in cutlass.range_constexpr(16):
                    selected_ids[selected] = Int32(-1)
                    selected_weights[selected] = Float32(0.0)
                weight_sum = Float32(0.0)
                for selected in cutlass.range_constexpr(16):
                    best_score = Float32(float("-inf"))
                    best_expert = Int32(-1)
                    candidate = Int32(0)
                    while candidate < Int32(896):
                        already_selected = False
                        for prior in cutlass.range_constexpr(selected):
                            if selected_ids[prior] == candidate:
                                already_selected = True
                        score = selection_scores[candidate]
                        if not already_selected:
                            if (
                                score > best_score
                                or (
                                    score == best_score
                                    and (
                                        best_expert < Int32(0)
                                        or candidate < best_expert
                                    )
                                )
                            ):
                                best_score = score
                                best_expert = candidate
                        candidate += Int32(1)
                    selected_ids[selected] = best_expert
                    weight = Float32(0.0)
                    if best_expert >= Int32(0):
                        weight = unbiased_scores[best_expert]
                    selected_weights[selected] = weight
                    weight_sum += weight
                scale = Float32(1.0) / (weight_sum + Float32(1.0e-20))
                for selected in cutlass.range_constexpr(16):
                    cute.arch.store(
                        output_second + Int64(selected),
                        selected_weights[selected] * scale,
                    )
                    cute.arch.store(
                        output_ids + Int64(selected), selected_ids[selected]
                    )

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                self_signal = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (self_signal + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (self_signal + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(1),
                )


def _pad_ptrs(values: Sequence[int], world_size: int) -> tuple[int, ...]:
    if len(values) != world_size:
        raise ValueError("pointer bundle does not match DCP world size")
    padded = [int(value) for value in values]
    padded.extend([padded[-1]] * (_MAX_RANKS - len(padded)))
    return tuple(padded)


def _slot_delta_256b(slot_delta_bytes: int) -> int:
    """Encode a signed aligned delta without narrowing address arithmetic."""

    value = int(slot_delta_bytes)
    if value == 0 or value % _SLOT_ALIGNMENT:
        raise ValueError("DCP staging slot delta must be a nonzero 256B multiple")
    units = value // _SLOT_ALIGNMENT
    if not -(1 << 31) <= units < (1 << 31):
        raise ValueError("DCP staging slot delta exceeds the 512 GiB graph ABI limit")
    return units


def _u32_ptr(value: int, *, align: int = 16) -> cute.Pointer:
    return make_ptr(Uint32, int(value), cute.AddressSpace.gmem, assumed_align=align)


def _u8_ptr(value: int) -> cute.Pointer:
    return make_ptr(Uint8, int(value), cute.AddressSpace.gmem, assumed_align=16)


def _f32_ptr(value: int) -> cute.Pointer:
    return make_ptr(Float32, int(value), cute.AddressSpace.gmem, assumed_align=4)


def _i32_ptr(value: int) -> cute.Pointer:
    return make_ptr(Int32, int(value), cute.AddressSpace.gmem, assumed_align=4)


def _lse_launcher_key(
    world_size: int,
    rank: int,
    dtype_name: str,
    threads: int,
    device_slot_selection: bool,
) -> tuple[object, ...]:
    return (
        int(world_size),
        int(rank),
        str(dtype_name),
        int(threads),
        bool(device_slot_selection),
    )


def is_lse_reduce_scatter_prepared(
    world_size: int,
    rank: int,
    dtype_name: str,
    threads: int,
    device_slot_selection: bool,
) -> bool:
    return _lse_launcher_key(
        world_size,
        rank,
        dtype_name,
        threads,
        device_slot_selection,
    ) in _PREPARED_LSE_LAUNCHERS


@functools.cache
def _get_compiled_lse_reduce_scatter(
    world_size: int,
    rank: int,
    dtype_name: str,
    threads: int,
    device_slot_selection: bool,
) -> Callable:
    launch = _LseReduceScatterLaunch(
        world_size,
        rank,
        dtype_name,
        threads,
        device_slot_selection,
    )
    key = (
        int(world_size),
        int(rank),
        str(dtype_name),
        int(threads),
        bool(device_slot_selection),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    p_u32 = _u32_ptr(16, align=4)
    p_f32 = _f32_ptr(16)
    p_u8 = _u8_ptr(16)
    raw = b12x_compile(
        launch,
        p_u32,
        p_f32,
        p_u32,
        *(p_u8 for _ in range(_MAX_RANKS)),
        *(p_u32 for _ in range(_MAX_RANKS)),
        0,
        1,
        1,
        8,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp_a2a.lse_reduce_scatter",
            35,
            key,
            labels=(
                "world_size",
                "rank",
                "dtype",
                "threads",
                "device_slot_selection",
            ),
        ),
    )

    def run(
        local_output_ptr: int,
        local_lse_ptr: int,
        output_ptr: int,
        staging_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        lse_offset: int,
        batch: int,
        total_heads: int,
        head_dim: int,
        input_stride_batch: int,
        input_stride_head: int,
        output_stride_batch: int,
        output_stride_head: int,
        natural_log: bool,
        slot_delta_256b: int,
        blocks: int,
    ) -> None:
        stages = _pad_ptrs(staging_ptrs, world_size)
        signals = _pad_ptrs(signal_ptrs, world_size)
        raw(
            _u32_ptr(local_output_ptr),
            _f32_ptr(local_lse_ptr),
            _u32_ptr(output_ptr),
            *(_u8_ptr(ptr) for ptr in stages),
            *(_u32_ptr(ptr, align=4) for ptr in signals),
            int(lse_offset),
            int(batch),
            int(total_heads),
            int(head_dim),
            int(input_stride_batch),
            int(input_stride_head),
            int(output_stride_batch),
            int(output_stride_head),
            int(bool(natural_log)),
            int(slot_delta_256b),
            int(blocks),
            current_cuda_stream(),
        )

    _PREPARED_LSE_LAUNCHERS.add(key)
    return run


def _gather_launcher_key(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
) -> tuple[object, ...]:
    return (
        int(world_size),
        int(rank),
        int(threads),
        bool(device_slot_selection),
    )


def is_all_gather_heads_prepared(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
) -> bool:
    return _gather_launcher_key(
        world_size,
        rank,
        threads,
        device_slot_selection,
    ) in _PREPARED_GATHER_LAUNCHERS


@functools.cache
def _get_compiled_all_gather_heads(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
) -> Callable:
    launch = _AllGatherHeadsLaunch(
        world_size,
        rank,
        threads,
        device_slot_selection,
    )
    key = (
        int(world_size),
        int(rank),
        int(threads),
        bool(device_slot_selection),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    p_u32 = _u32_ptr(16, align=4)
    p_u8 = _u8_ptr(16)
    raw = b12x_compile(
        launch,
        p_u32,
        p_u32,
        *(p_u8 for _ in range(_MAX_RANKS)),
        *(p_u32 for _ in range(_MAX_RANKS)),
        1,
        1,
        8,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp_a2a.all_gather_heads",
            12,
            key,
            labels=(
                "world_size",
                "rank",
                "threads",
                "device_slot_selection",
            ),
        ),
    )

    def run(
        local_input_ptr: int,
        output_ptr: int,
        staging_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        batch: int,
        local_heads: int,
        head_dim: int,
        element_size: int,
        slot_delta_256b: int,
        blocks: int,
    ) -> None:
        row_bytes = int(head_dim) * int(element_size)
        if row_bytes % 16:
            raise ValueError("DCP gather rows must be a multiple of 16 bytes")
        packs_per_head = row_bytes // 16
        stages = _pad_ptrs(staging_ptrs, world_size)
        signals = _pad_ptrs(signal_ptrs, world_size)
        raw(
            _u32_ptr(local_input_ptr),
            _u32_ptr(output_ptr),
            *(_u8_ptr(ptr) for ptr in stages),
            *(_u32_ptr(ptr, align=4) for ptr in signals),
            int(batch),
            int(local_heads),
            packs_per_head,
            int(slot_delta_256b),
            int(blocks),
            current_cuda_stream(),
        )

    _PREPARED_GATHER_LAUNCHERS.add(key)
    return run


def _pair_launcher_key(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
    kimi_topk: bool,
) -> tuple[object, ...]:
    return (
        int(world_size),
        int(rank),
        int(threads),
        bool(device_slot_selection),
        bool(kimi_topk),
    )


def is_all_gather_pair_prepared(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
    kimi_topk: bool = False,
) -> bool:
    return _pair_launcher_key(
        world_size,
        rank,
        threads,
        device_slot_selection,
        kimi_topk,
    ) in _PREPARED_PAIR_LAUNCHERS


@functools.cache
def _get_compiled_all_gather_pair(
    world_size: int,
    rank: int,
    threads: int,
    device_slot_selection: bool,
    kimi_topk: bool = False,
) -> Callable:
    launch = _AllGatherPairLaunch(
        world_size,
        rank,
        threads,
        device_slot_selection,
        kimi_topk,
    )
    key = _pair_launcher_key(
        world_size,
        rank,
        threads,
        device_slot_selection,
        kimi_topk,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    p_u32 = _u32_ptr(16, align=4)
    p_f32 = _f32_ptr(16)
    p_i32 = _i32_ptr(16)
    p_u8 = _u8_ptr(16)
    raw = b12x_compile(
        launch,
        p_u32,
        p_u32,
        p_f32,
        p_u32,
        p_f32 if kimi_topk else p_u32,
        p_i32,
        *(p_u8 for _ in range(_MAX_RANKS)),
        *(p_u32 for _ in range(_MAX_RANKS)),
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            (
                "comm.pcie.dcp_a2a.all_gather_pair_kimi_topk"
                if kimi_topk
                else "comm.pcie.dcp_a2a.all_gather_pair"
            ),
            1,
            key,
            labels=(
                "world_size",
                "rank",
                "threads",
                "device_slot_selection",
                "kimi_topk",
            ),
        ),
    )

    def run(
        local_first_ptr: int,
        local_second_ptr: int,
        correction_bias_ptr: int,
        output_first_ptr: int,
        output_second_ptr: int,
        output_ids_ptr: int,
        staging_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        batch: int,
        first_packs: int,
        second_packs: int,
        slot_delta_256b: int,
    ) -> None:
        stages = _pad_ptrs(staging_ptrs, world_size)
        signals = _pad_ptrs(signal_ptrs, world_size)
        raw(
            _u32_ptr(local_first_ptr),
            _u32_ptr(local_second_ptr),
            _f32_ptr(correction_bias_ptr),
            _u32_ptr(output_first_ptr),
            (
                _f32_ptr(output_second_ptr)
                if kimi_topk
                else _u32_ptr(output_second_ptr)
            ),
            _i32_ptr(output_ids_ptr),
            *(_u8_ptr(ptr) for ptr in stages),
            *(_u32_ptr(ptr, align=4) for ptr in signals),
            int(batch),
            int(first_packs),
            int(second_packs),
            int(slot_delta_256b),
            current_cuda_stream(),
        )

    _PREPARED_PAIR_LAUNCHERS.add(key)
    return run


def lse_reduce_scatter(
    *,
    world_size: int,
    rank: int,
    dtype_name: str,
    threads: int,
    local_output_ptr: int,
    local_lse_ptr: int,
    output_ptr: int,
    staging_ptrs: Sequence[int],
    signal_ptrs: Sequence[int],
    lse_offset: int,
    batch: int,
    total_heads: int,
    head_dim: int,
    input_stride_batch: int,
    input_stride_head: int,
    output_stride_batch: int,
    output_stride_head: int,
    natural_log: bool,
    device_slot_selection: bool,
    slot_delta_bytes: int,
    blocks: int,
) -> None:
    slot_delta_256b = _slot_delta_256b(slot_delta_bytes)
    launcher = _get_compiled_lse_reduce_scatter(
        world_size,
        rank,
        dtype_name,
        threads,
        device_slot_selection,
    )
    if not device_slot_selection:
        _get_compiled_lse_reduce_scatter(
            world_size,
            rank,
            dtype_name,
            threads,
            True,
        )
    launcher(
        local_output_ptr,
        local_lse_ptr,
        output_ptr,
        staging_ptrs,
        signal_ptrs,
        lse_offset,
        batch,
        total_heads,
        head_dim,
        input_stride_batch,
        input_stride_head,
        output_stride_batch,
        output_stride_head,
        natural_log,
        slot_delta_256b,
        blocks,
    )


def all_gather_heads(
    *,
    world_size: int,
    rank: int,
    threads: int,
    local_input_ptr: int,
    output_ptr: int,
    staging_ptrs: Sequence[int],
    signal_ptrs: Sequence[int],
    batch: int,
    local_heads: int,
    head_dim: int,
    element_size: int,
    device_slot_selection: bool,
    slot_delta_bytes: int,
    blocks: int,
) -> None:
    slot_delta_256b = _slot_delta_256b(slot_delta_bytes)
    launcher = _get_compiled_all_gather_heads(
        world_size,
        rank,
        threads,
        device_slot_selection,
    )
    if not device_slot_selection:
        _get_compiled_all_gather_heads(
            world_size,
            rank,
            threads,
            True,
        )
    launcher(
        local_input_ptr,
        output_ptr,
        staging_ptrs,
        signal_ptrs,
        batch,
        local_heads,
        head_dim,
        element_size,
        slot_delta_256b,
        blocks,
    )


def all_gather_pair(
    *,
    world_size: int,
    rank: int,
    threads: int,
    local_first_ptr: int,
    local_second_ptr: int,
    output_first_ptr: int,
    output_second_ptr: int,
    staging_ptrs: Sequence[int],
    signal_ptrs: Sequence[int],
    batch: int,
    first_row_bytes: int,
    second_row_bytes: int,
    device_slot_selection: bool,
    slot_delta_bytes: int,
) -> None:
    if first_row_bytes % 16 or second_row_bytes % 16:
        raise ValueError("paired DCP rows must be multiples of 16 bytes")
    slot_delta_256b = _slot_delta_256b(slot_delta_bytes)
    launcher = _get_compiled_all_gather_pair(
        world_size,
        rank,
        threads,
        device_slot_selection,
        False,
    )
    if not device_slot_selection:
        _get_compiled_all_gather_pair(world_size, rank, threads, True, False)
    launcher(
        local_first_ptr,
        local_second_ptr,
        local_second_ptr,
        output_first_ptr,
        output_second_ptr,
        output_second_ptr,
        staging_ptrs,
        signal_ptrs,
        batch,
        first_row_bytes // 16,
        second_row_bytes // 16,
        slot_delta_256b,
    )


def all_gather_pair_kimi_topk(
    *,
    rank: int,
    local_down_ptr: int,
    local_router_ptr: int,
    correction_bias_ptr: int,
    output_down_ptr: int,
    topk_weights_ptr: int,
    topk_ids_ptr: int,
    staging_ptrs: Sequence[int],
    signal_ptrs: Sequence[int],
    device_slot_selection: bool,
    slot_delta_bytes: int,
) -> None:
    slot_delta_256b = _slot_delta_256b(slot_delta_bytes)
    launcher = _get_compiled_all_gather_pair(
        16,
        rank,
        512,
        device_slot_selection,
        True,
    )
    if not device_slot_selection:
        _get_compiled_all_gather_pair(16, rank, 512, True, True)
    launcher(
        local_down_ptr,
        local_router_ptr,
        correction_bias_ptr,
        output_down_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        staging_ptrs,
        signal_ptrs,
        1,
        28,
        14,
        slot_delta_256b,
    )


__all__ = [
    "all_gather_heads",
    "all_gather_pair",
    "all_gather_pair_kimi_topk",
    "lse_reduce_scatter",
]
