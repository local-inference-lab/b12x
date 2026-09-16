"""Reuse representative-key operands across two four-head QSA queries."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import bf16_mma_m16n8k16_f32
from b12x.attention.qsa._score_cute import _RepresentativeScoreKernel


class PairedRepresentativeScoreKernel(_RepresentativeScoreKernel):
    @cute.jit
    def __call__(
        self,
        pointers: tuple,
        strides: tuple,
        rows: Int32,
        group_offset: Int32,
        group_count: Int32,
        stream: cuda.CUstream,
    ):
        if rows >= 128:
            self.paired_kernel(
                pointers, strides, rows, group_offset, group_count
            ).launch(
                grid=((group_count + 63) // 64, (rows + 1) // 2, 1),
                block=(128, 1, 1),
                stream=stream,
            )
        else:
            self.kernel(pointers, strides, group_offset, group_count).launch(
                grid=((group_count + 63) // 64, rows, 1),
                block=(128, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def paired_kernel(
        self,
        pointers: tuple,
        strides: tuple,
        rows: Int32,
        group_offset: Int32,
        group_count: Int32,
    ):
        query, positions, requests, lengths, cache, table, scores, counts, merges = (
            pointers
        )
        page_stride, token_stride, table_stride, score_stride = strides
        block, pair, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane, warp = thread % 32, thread // 32
        row = pair * 2
        eligible = cute.make_rmem_tensor((2,), Int32)
        carry = cute.make_rmem_tensor((2,), Int32)
        request = cute.make_rmem_tensor((2,), Int64)
        for part in cutlass.range_constexpr(2):
            eligible[part], carry[part], request[part] = Int32(0), Int32(0), Int64(-1)
            if row + part < rows:
                request[part] = requests[row + part].to(Int64)
                if request[part] >= Int64(0):
                    p = (positions[row + part].to(Int64) + Int64(1)) // self.ratio
                    s = lengths[request[part]].to(Int64) // self.ratio
                    eligible[part] = cutlass.min(
                        cutlass.min(p, s), Int64(self.max_groups)
                    ).to(Int32)
                carry[part] = cutlass.min(
                    cutlass.min(eligible[part], group_offset), Int32(self.budget)
                )
                if (block == 0) & (thread == 0):
                    counts[row + part] = eligible[part]
                    merges[row + part] = carry[part] + cutlass.min(
                        cutlass.max(eligible[part] - group_offset, Int32(0)),
                        group_count,
                    )

        if (request[0] >= Int64(0)) & (request[0] == request[1]):
            self.score_pair(
                query,
                cache,
                table,
                scores,
                request[0],
                row,
                page_stride,
                token_stride,
                table_stride,
                score_stride,
                group_offset,
                group_count,
                block * 64,
                eligible,
                carry,
                lane,
                warp,
            )
        else:
            # Adjacent packed rows can straddle requests or include padding.
            # The scalar-query MMA path retains each row's own page table.
            for part in cutlass.range_constexpr(2):
                if row + part < rows:
                    self.score_mma(
                        query,
                        cache,
                        table,
                        scores,
                        request[part],
                        row + part,
                        page_stride,
                        token_stride,
                        table_stride,
                        (row + part).to(Int64) * score_stride,
                        group_offset,
                        group_count,
                        block * 64,
                        eligible[part],
                        carry[part],
                        lane,
                        warp,
                    )

    @cute.jit
    def score_pair(
        self,
        query,
        cache,
        table,
        scores,
        request,
        row,
        page_stride,
        token_stride,
        table_stride,
        score_stride,
        group_offset,
        group_count,
        local_start,
        eligible,
        carry,
        lane,
        warp,
    ):
        matrix_row, matrix_pair = lane // 4, lane % 4
        warp_start = local_start + warp * 16
        max_eligible = cutlass.max(eligible[0], eligible[1])
        if group_offset + warp_start < max_eligible:
            base = cute.make_rmem_tensor((2,), Int64)
            valid = cute.make_rmem_tensor((2,), Int32)
            for part in cutlass.range_constexpr(2):
                local_group = warp_start + matrix_row + part * 8
                group = group_offset + local_group
                base[part], valid[part] = Int64(0), Int32(0)
                if (local_group < group_count) & (group < max_eligible):
                    page = table[
                        request * table_stride + (group // self.page_size).to(Int64)
                    ].to(Int64)
                    if page >= Int64(0):
                        base[part] = (
                            page * page_stride
                            + (group % self.page_size).to(Int64) * token_stride
                        )
                        valid[part] = Int32(1)
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for tile in cutlass.range_constexpr(self.dim // 16):
                column = (tile * 16 + matrix_pair * 2).to(Int64)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid[0] != 0:
                    a0 = self.load_pair(cache, base[0] + column)
                    a2 = self.load_pair(cache, base[0] + column + Int64(8))
                if valid[1] != 0:
                    a1 = self.load_pair(cache, base[1] + column)
                    a3 = self.load_pair(cache, base[1] + column + Int64(8))
                # Eight MMA columns hold two independent groups of four heads.
                query_base = (
                    (row + matrix_row // 4).to(Int64) * 4 * self.dim
                    + (matrix_row % 4).to(Int64) * self.dim
                    + column
                )
                b0 = self.load_pair(query, query_base)
                b1 = self.load_pair(query, query_base + Int64(8))
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    d0, d1, d2, d3, a0, a1, a2, a3, b0, b1
                )
            for part in cutlass.range_constexpr(2):
                first = cutlass.max(d0 if part == 0 else d2, Float32(0))
                second = cutlass.max(d1 if part == 0 else d3, Float32(0))
                local_group = warp_start + matrix_row + part * 8
                for q in cutlass.range_constexpr(2):
                    score = Float32(0)
                    for head in cutlass.range_constexpr(4):
                        value = first if head % 2 == 0 else second
                        value = cute.arch.shuffle_sync(
                            value, (lane // 4) * 4 + q * 2 + head // 2
                        )
                        score = score + value
                    score = score * Float32(self.scale)
                    if (valid[part] == 0) | (group_offset + local_group >= eligible[q]):
                        score = Float32(-float("inf"))
                    if (matrix_pair == q * 2) & (local_group < group_count):
                        address = (row + q).to(Int64) * score_stride + (
                            carry[q] + local_group
                        ).to(Int64)
                        scores[address] = score
        else:
            local_group = warp_start + lane
            if (lane < 16) & (local_group < group_count):
                for q in cutlass.range_constexpr(2):
                    address = (row + q).to(Int64) * score_stride + (
                        carry[q] + local_group
                    ).to(Int64)
                    scores[address] = Float32(-float("inf"))
