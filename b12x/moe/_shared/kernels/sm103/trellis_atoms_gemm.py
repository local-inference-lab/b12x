"""SM103 projection directly from grouped canonical Trellis atom planes."""

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

from .trellis_decode import decode_lane, tile_coordinates
from .trellis_gemm import RoutedTrellisGemm


class RoutedAtomTrellisGemm(RoutedTrellisGemm):
    def __init__(
        self, n, k, experts, capacity, *, group_size, fc1, codebook, dual_input=False
    ):
        super().__init__(
            n, k, experts, capacity, bits=3, codebook=codebook, dual_input=dual_input
        )
        self.hidden = k if fc1 else n
        self.intermediate = n if fc1 else k
        if (
            type(group_size) is not int
            or group_size <= 0
            or group_size % 32
            or self.intermediate % group_size
        ):
            raise ValueError(
                "Trellis atom group size must be a positive multiple of 32 dividing I"
            )
        if codebook not in {"mcg", "sqg_e4m3"}:
            raise ValueError("Trellis atom projection requires MCG or SQG E4M3")
        self.group_size = group_size
        self.groups = self.intermediate // group_size
        self.fc1 = fc1
        self.max_bits = 6 if codebook == "mcg" else 4

    @cute.jit
    def __call__(
        self,
        a,
        packed,
        lut,
        ids,
        offsets,
        rates,
        out,
        projection: c.Int32,
        row_stride: c.Int64,
        packed_words: c.Int64,
        live_routes: c.Int32,
        a_stride: c.Int64,
        out_stride: c.Int64,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                c.Float16,
                c.Float32,
                (128, 128, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, c.Float16
        )
        layout = cute.tile_to_shape(atom, (128, 64), order=(0, 1))
        self.kernel(
            mma,
            layout,
            a,
            packed,
            lut,
            ids,
            offsets,
            rates,
            out,
            projection,
            row_stride,
            packed_words,
            a_stride,
            out_stride,
        ).launch(
            grid=(live_routes * cute.ceil_div(self.n, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def selection(
        self,
        ids,
        offsets,
        rates,
        route: c.Int64,
        projection: c.Int32,
        row_stride: c.Int64,
        packed_words: c.Int64,
    ):
        ids_tensor = cute.make_tensor(ids, cute.make_layout(self.capacity))
        expert = c.Int64(ids_tensor[route])
        metadata_layout = cute.make_layout(c.Int64(self.groups) * self.experts * 3)
        offset_table = cute.make_tensor(offsets, metadata_layout)
        rate_table = cute.make_tensor(rates, metadata_layout)
        valid = (expert >= 0) & (expert < self.experts)
        if c.const_expr(self.fc1):
            valid = valid & (projection >= 0) & (projection < 2)
        else:
            valid = valid & (projection == 2)
        return (
            expert,
            projection,
            offset_table,
            rate_table,
            row_stride,
            packed_words,
            valid,
        )

    @cute.kernel
    def kernel(
        self,
        mma,
        layout,
        a,
        packed,
        lut,
        ids,
        offsets,
        rates,
        out,
        projection: c.Int32,
        row_stride: c.Int64,
        packed_words: c.Int64,
        a_stride: c.Int64,
        out_stride: c.Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        route = c.Int64(block // cute.ceil_div(self.n, 128))
        n_base = c.Int64(block % cute.ceil_div(self.n, 128)) * 128
        source = self.input_tensors(a, a_stride)
        weights = cute.make_tensor(packed, cute.make_layout(packed_words))
        output = cute.make_tensor(
            out, cute.make_layout(c.Int64(self.capacity) * out_stride)
        )
        selected = self.selection(
            ids, offsets, rates, route, projection, row_stride, packed_words
        )
        if selected[6]:
            self.execute_tiles(
                mma,
                layout,
                source,
                weights,
                lut,
                selected,
                route,
                a_stride,
                output,
                n_base,
                out_stride,
            )
        else:
            if n_base + c.Int64(thread) < self.n:
                output[route * out_stride + n_base + c.Int64(thread)] = c.Float16(0)

    @cute.jit
    def select_tile(self, selection, global_n: c.Int64, global_k: c.Int64):
        expert, projection, offsets, rates, stride, words, valid = selection
        if c.const_expr(self.fc1):
            slot, plane, hidden_tile = global_n // 2, global_n % 2, global_k
        else:
            slot, plane, hidden_tile = global_k // 2, global_k % 2, global_n
        valid = valid & (slot >= 0) & (slot < self.intermediate // 32)
        valid = valid & (hidden_tile >= 0) & (hidden_tile < self.hidden // 16)
        offset, code = c.Int64(-1), c.Int32(0)
        if valid:
            group = slot // (self.group_size // 32)
            index = (group * c.Int64(self.experts) + expert) * 3 + c.Int64(projection)
            offset, code = offsets[index], c.Int32(rates[index])
        low, high = code & 15, code >> 4
        rate = low
        plane_offset = c.Int64(0)
        if plane == 1:
            rate = high
            plane_offset = c.Int64(self.hidden // 16) * 8 * c.Int64(low)
        section = c.Int64(self.hidden // 16) * 8 * c.Int64(low + high)
        valid = (
            valid
            & (low >= 2)
            & (low <= self.max_bits)
            & (high >= 2)
            & (high <= self.max_bits)
        )
        valid = (
            valid & (offset >= 0) & (offset <= stride) & (section <= stride - offset)
        )
        base = slot * stride + offset + plane_offset + hidden_tile * 8 * c.Int64(rate)
        valid = (
            valid & (base >= 0) & (base <= words) & (c.Int64(8) * rate <= words - base)
        )
        return base, rate, valid

    @cute.jit
    def stage_operands(
        self,
        source,
        weights,
        lut,
        selection,
        route: c.Int64,
        a_stride: c.Int64,
        stage: c.Int32,
        n_base: c.Int64,
        sA,
        sB,
    ):
        thread, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = c.Int32(thread % 32)
        self.stage_inputs(source, selection[6], route, a_stride, stage, sA)
        for tile in c.range(warp, 32, 4):
            local_k, local_n = tile // 8, tile % 8
            global_k = c.Int64(stage) * 4 + c.Int64(local_k)
            global_n = n_base // 16 + c.Int64(local_n)
            base, rate, valid = self.select_tile(selection, global_n, global_k)
            decoded = cute.make_rmem_tensor(8, c.Float16)
            decoded.fill(c.Float16(0))
            if valid:
                record = cute.make_tensor(
                    weights.iterator + base, cute.make_layout(c.Int64(8) * rate)
                )
                for bits in c.range_constexpr(2, self.max_bits + 1):
                    if rate == bits:
                        values = decode_lane(
                            record, c.Int64(0), lane, lut, bits, self.codebook
                        )
                        decoded.store(values.load())
            for j in c.range_constexpr(8):
                nn, kk = tile_coordinates(lane, j)
                sB[local_n * 16 + nn, local_k * 16 + kk] = decoded[j]
