"""Native projection dispatch over coalesced MCG K3/K4/K5 expert records.

Projection descriptors select a rate tier and a record within that projection.
Tier offsets, populated counts and payload length are runtime scalar arguments;
only model geometry, descriptor format and planned capacity specialize code.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float16, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

from .trellis_decode import decode_lane, tile_coordinates
from .trellis_gemm import RoutedTrellisGemm


class RoutedMixedTrellisGemm(RoutedTrellisGemm):
    def __init__(self, n, k, experts, capacity, *, descriptor_local_bits):
        super().__init__(n, k, experts, capacity, bits=3, codebook="mcg")
        if descriptor_local_bits not in (8, 24):
            raise ValueError(
                "mixed Trellis requires eight- or 24-bit local descriptors"
            )
        self.descriptor_local_bits = descriptor_local_bits

    @cute.jit
    def __call__(
        self,
        a: cute.Pointer,
        packed: cute.Pointer,
        lut: cute.Pointer,
        ids: cute.Pointer,
        descriptors: cute.Pointer,
        out: cute.Pointer,
        projection: Int32,
        descriptor_stride: Int64,
        packed_words: Int64,
        offsets: tuple[Int64, Int64, Int64],
        counts: tuple[Int32, Int32, Int32],
        live_routes: Int32,
        a_stride: Int64,
        out_stride: Int64,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                Float16,
                Float32,
                (128, 128, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, Float16
        )
        layout = cute.tile_to_shape(atom, (128, 64), order=(0, 1))
        self.kernel(
            mma,
            layout,
            a,
            packed,
            lut,
            ids,
            descriptors,
            out,
            projection,
            descriptor_stride,
            packed_words,
            offsets,
            counts,
            a_stride,
            out_stride,
        ).launch(
            grid=(live_routes * cute.ceil_div(self.n, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def select_record(
        self,
        descriptors,
        expert: Int64,
        projection: Int32,
        descriptor_stride: Int64,
        packed_words: Int64,
        offsets,
        counts,
    ):
        valid = (
            (expert >= 0)
            & (expert < self.experts)
            & (projection >= 0)
            & (projection < 3)
        )
        valid = valid & (descriptor_stride >= self.experts)
        descriptor = Int32(-1)
        if valid:
            descriptor = descriptors[Int64(projection) * descriptor_stride + expert]
        tier = descriptor >> self.descriptor_local_bits
        local = Int64(descriptor & ((1 << self.descriptor_local_bits) - 1))
        count, start = Int64(0), Int64(0)
        for index in cutlass.range_constexpr(3):
            if tier == index:
                count, start = Int64(counts[index]), offsets[index]
        rate = Int32(tier + 3)
        words = Int64(self.k // 16) * Int64(self.n // 16) * Int64(8) * Int64(rate)
        base = start + local * words
        valid = (
            valid & (tier >= 0) & (tier < 3) & (count >= 0) & (count <= self.experts)
        )
        valid = valid & (local < count) & (start >= 0) & (base >= start)
        valid = valid & (base <= packed_words) & (words <= packed_words - base)
        return base, rate, valid

    @cute.kernel
    def kernel(
        self,
        mma,
        layout,
        a,
        packed,
        lut,
        ids,
        descriptors,
        out,
        projection: Int32,
        descriptor_stride: Int64,
        packed_words: Int64,
        offsets,
        counts,
        a_stride: Int64,
        out_stride: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        route = Int64(block // cute.ceil_div(self.n, 128))
        n_base = Int64(block % cute.ceil_div(self.n, 128)) * 128
        route_ids = cute.make_tensor(ids, cute.make_layout(self.capacity))
        source = cute.make_tensor(a, cute.make_layout(Int64(self.capacity) * a_stride))
        weights = cute.make_tensor(packed, cute.make_layout(packed_words))
        table = cute.make_tensor(
            descriptors, cute.make_layout(Int64(3) * descriptor_stride)
        )
        output = cute.make_tensor(
            out, cute.make_layout(Int64(self.capacity) * out_stride)
        )
        selection = self.select_record(
            table,
            Int64(route_ids[route]),
            projection,
            descriptor_stride,
            packed_words,
            offsets,
            counts,
        )
        # Selection is identical for every thread in this CTA. Invalid routes
        # initialize their output without allocating TMEM or reading a record.
        if selection[2]:
            self.execute_tiles(
                mma,
                layout,
                source,
                weights,
                lut,
                selection,
                route,
                a_stride,
                output,
                n_base,
                out_stride,
            )
        else:
            if n_base + Int64(thread) < self.n:
                output[route * out_stride + n_base + Int64(thread)] = Float16(0)

    @cute.jit
    def stage_operands(
        self,
        source: cute.Tensor,
        weights: cute.Tensor,
        lut: cute.Pointer,
        selection: tuple[Int64, Int32, Boolean],
        route: Int64,
        a_stride: Int64,
        stage: Int32,
        n_base: Int64,
        sA: cute.Tensor,
        sB: cute.Tensor,
    ):
        base, rate, valid = selection
        thread, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = Int32(thread % 32)
        for item in cutlass.range(thread, 128 * 64, 128):
            row, col = item // 64, item % 64
            value = Float16(0)
            kk = Int64(stage) * 64 + Int64(col)
            if valid & (row == 0) & (kk < self.k):
                value = source[route * a_stride + kk]
            sA[row, col] = value
        record_words = (
            Int64(self.k // 16) * Int64(self.n // 16) * Int64(8) * Int64(rate)
        )
        record = cute.make_tensor(
            weights.iterator + base, cute.make_layout(record_words)
        )
        for tile in cutlass.range(warp, 32, 4):
            local_k, local_n = tile // 8, tile % 8
            global_k = Int64(stage) * 4 + Int64(local_k)
            global_n = n_base // 16 + Int64(local_n)
            decoded = cute.make_rmem_tensor(8, Float16)
            decoded.fill(Float16(0))
            if valid & (global_k < self.k // 16) & (global_n < self.n // 16):
                tile_id = global_k * Int64(self.n // 16) + global_n
                if rate == 3:
                    values = decode_lane(record, tile_id, lane, lut, 3, "mcg")
                    decoded.store(values.load())
                elif rate == 4:
                    values = decode_lane(record, tile_id, lane, lut, 4, "mcg")
                    decoded.store(values.load())
                elif rate == 5:
                    values = decode_lane(record, tile_id, lane, lut, 5, "mcg")
                    decoded.store(values.load())
            for j in cutlass.range_constexpr(8):
                nn, kk = tile_coordinates(lane, j)
                sB[local_n * 16 + nn, local_k * 16 + kk] = decoded[j]
