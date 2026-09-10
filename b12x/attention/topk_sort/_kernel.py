"""Counting sort of top-k selections by KV position (CuTe DSL) and its opaque
custom op.

One CTA of 256 threads per row. The row's positions are marked in a
shared-memory bitmap of ``max_words`` 32-bit words (``max_positions / 32``,
a static compile key; the row's own word count is ``ceil(seq_len / 32)``
capped at ``max_words``). The bitmap is then scanned in rounds of 1024 words:
thread ``t`` owns words ``seg + t``, ``seg + 256 + t``, ``seg + 512 + t`` and
``seg + 768 + t`` (interleaved, so a dense run of selected positions spreads
over many threads); population counts are prefix-summed within each warp by
shuffles, the 32 (sub-block, warp) totals are scanned by warp 0 in word
order, and every thread emits the set bits of its words at the resulting
offsets, converting each position to its physical slot. Positions at or
beyond the bitmap limit and negative entries are dropped; the tail of the
row is filled with ``-1``. Duplicate positions collapse into one.

Slots are int32. A position is written as ``-1`` when its block lies beyond
the table width, its page is negative, or its page exceeds
``INT32_MAX >> log2(block_size)``, the largest page whose slots fit the
signed 32-bit range; the bound is checked per page before the shift, so no
slot is ever wrapped.

The output order of a row is a function of the selected set alone (bitwise
repeatable); the slot values also depend on the row's block table and the
block size.
"""

from __future__ import annotations

from typing import Dict

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64, Uint32
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    ld_global_i32,
    ld_shared_i32,
    ld_shared_u32,
    red_or_shared_u32,
    shared_ptr_to_u32,
    st_global_i32,
    st_shared_i32,
    st_shared_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention._shared.cute.ops import warp_prefix_sum

_THREADS = 256
_WARPS = _THREADS // 32
_ROUND_WORDS = 4 * _THREADS
#: Independent row loads issued back to back per thread while marking (8 x
#: 256 threads = one pass over a 2048-wide selection); a partial tail batch
#: loads one entry at a time.
_MARK_BATCH = 8

#: Largest bitmap (32-bit words) one CTA holds in shared memory: 96 KB.
MAX_BITMAP_WORDS = 24 * 1024

_KERNEL_CACHE: Dict[int, object] = {}


def bitmap_words(max_positions: int) -> int:
    """Bitmap words for positions below ``max_positions``."""
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    words = (int(max_positions) + 31) // 32
    if words > MAX_BITMAP_WORDS:
        raise ValueError(
            f"max_positions={max_positions} needs {words} bitmap words; the "
            f"limit is {MAX_BITMAP_WORDS}"
        )
    return words


class TopkSortConvertKernel:
    """One CTA per row; see the module docstring."""

    def __init__(self, max_words: int):
        self.max_words = int(max_words)
        self.warp_offset = self.max_words * 4  # s_warp[32] int32
        self.total_offset = self.warp_offset + 32 * 4  # s_total int32
        self.smem_bytes = self.total_offset + 16

    @cute.jit
    def __call__(
        self,
        idx_ptr: cute.Pointer,
        seq_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        rows: Int32,
        idx_stride: Int64,
        bt_stride0: Int64,
        bt_stride1: Int64,
        bt_width: Int32,
        topk: Int32,
        bs_log2: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            idx_ptr,
            seq_ptr,
            bt_ptr,
            idx_stride,
            bt_stride0,
            bt_stride1,
            bt_width,
            topk,
            bs_log2,
        ).launch(
            grid=(rows, 1, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        idx_ptr: cute.Pointer,
        seq_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        idx_stride: Int64,
        bt_stride0: Int64,
        bt_stride1: Int64,
        bt_width: Int32,
        topk: Int32,
        bs_log2: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        row = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.smem_bytes // 4],
                16,
            ]

        storage = smem.allocate(Storage)
        bitmap = shared_ptr_to_u32(storage.words.data_ptr())
        s_warp = bitmap + Int32(self.warp_offset)
        s_total = bitmap + Int32(self.total_offset)

        out = Int64(idx_ptr.toint()) + Int64(row) * idx_stride * Int64(4)
        table = Int64(bt_ptr.toint()) + Int64(row) * bt_stride0 * Int64(4)
        seq = ld_global_i32(Int64(seq_ptr.toint()) + Int64(row) * Int64(4))
        if seq < Int32(0):
            seq = Int32(0)
        n_words = (seq + Int32(31)) >> Int32(5)
        if n_words > Int32(self.max_words):
            n_words = Int32(self.max_words)
        limit = n_words << Int32(5)
        low_mask = (Int32(1) << bs_log2) - Int32(1)
        # Largest page whose slots stay within the signed int32 range.
        max_page = Int32(0x7FFFFFFF) >> bs_log2

        # 1. Clear the row's bitmap and mark its positions.
        w = tid
        while w < n_words:
            st_shared_u32(bitmap + w * Int32(4), Uint32(0))
            w += Int32(_THREADS)
        cute.arch.sync_threads()
        # Marking reads the row in batches of _MARK_BATCH independent loads
        # per thread (indices clamped to the last entry: re-marking a
        # position is idempotent), so the load latencies overlap instead of
        # serialising one load per loop iteration.
        i = tid
        while i + Int32((_MARK_BATCH - 1) * _THREADS) < topk:
            p0 = ld_global_i32(out + Int64(i) * Int64(4))
            p1 = ld_global_i32(out + Int64(i + Int32(1 * _THREADS)) * Int64(4))
            p2 = ld_global_i32(out + Int64(i + Int32(2 * _THREADS)) * Int64(4))
            p3 = ld_global_i32(out + Int64(i + Int32(3 * _THREADS)) * Int64(4))
            p4 = ld_global_i32(out + Int64(i + Int32(4 * _THREADS)) * Int64(4))
            p5 = ld_global_i32(out + Int64(i + Int32(5 * _THREADS)) * Int64(4))
            p6 = ld_global_i32(out + Int64(i + Int32(6 * _THREADS)) * Int64(4))
            p7 = ld_global_i32(out + Int64(i + Int32(7 * _THREADS)) * Int64(4))
            self._mark(bitmap, limit, p0)
            self._mark(bitmap, limit, p1)
            self._mark(bitmap, limit, p2)
            self._mark(bitmap, limit, p3)
            self._mark(bitmap, limit, p4)
            self._mark(bitmap, limit, p5)
            self._mark(bitmap, limit, p6)
            self._mark(bitmap, limit, p7)
            i += Int32(_MARK_BATCH * _THREADS)
        while i < topk:
            self._mark(bitmap, limit, ld_global_i32(out + Int64(i) * Int64(4)))
            i += Int32(_THREADS)
        cute.arch.sync_threads()

        # 2. Rounds of 1024 words: count, scan, emit.
        carry = Int32(0)
        seg = Int32(0)
        while seg < n_words:
            w0 = seg + tid
            w1 = seg + Int32(_THREADS) + tid
            w2 = seg + Int32(2 * _THREADS) + tid
            w3 = seg + Int32(3 * _THREADS) + tid
            word0 = Uint32(0)
            word1 = Uint32(0)
            word2 = Uint32(0)
            word3 = Uint32(0)
            if w0 < n_words:
                word0 = ld_shared_u32(bitmap + w0 * Int32(4))
            if w1 < n_words:
                word1 = ld_shared_u32(bitmap + w1 * Int32(4))
            if w2 < n_words:
                word2 = ld_shared_u32(bitmap + w2 * Int32(4))
            if w3 < n_words:
                word3 = ld_shared_u32(bitmap + w3 * Int32(4))
            cnt0 = Int32(cute.arch.popc(word0))
            cnt1 = Int32(cute.arch.popc(word1))
            cnt2 = Int32(cute.arch.popc(word2))
            cnt3 = Int32(cute.arch.popc(word3))
            incl0 = warp_prefix_sum(cnt0, lane)
            incl1 = warp_prefix_sum(cnt1, lane)
            incl2 = warp_prefix_sum(cnt2, lane)
            incl3 = warp_prefix_sum(cnt3, lane)
            if lane == Int32(31):
                st_shared_i32(s_warp + (Int32(0 * _WARPS) + warp) * Int32(4), incl0)
                st_shared_i32(s_warp + (Int32(1 * _WARPS) + warp) * Int32(4), incl1)
                st_shared_i32(s_warp + (Int32(2 * _WARPS) + warp) * Int32(4), incl2)
                st_shared_i32(s_warp + (Int32(3 * _WARPS) + warp) * Int32(4), incl3)
            cute.arch.sync_threads()
            if warp == Int32(0):
                # The 32 (sub-block, warp) totals in word order: exclusive scan.
                total = ld_shared_i32(s_warp + lane * Int32(4))
                scanned = warp_prefix_sum(total, lane)
                st_shared_i32(s_warp + lane * Int32(4), scanned - total)
                if lane == Int32(31):
                    st_shared_i32(s_total, scanned)
            cute.arch.sync_threads()

            base0 = carry + ld_shared_i32(
                s_warp + (Int32(0 * _WARPS) + warp) * Int32(4)
            )
            base1 = carry + ld_shared_i32(
                s_warp + (Int32(1 * _WARPS) + warp) * Int32(4)
            )
            base2 = carry + ld_shared_i32(
                s_warp + (Int32(2 * _WARPS) + warp) * Int32(4)
            )
            base3 = carry + ld_shared_i32(
                s_warp + (Int32(3 * _WARPS) + warp) * Int32(4)
            )
            # One page lookup per non-empty word: a block of >= 32 positions
            # holds whole bitmap words, so every set bit of a word maps to the
            # same page (per-bit lookups stay for smaller blocks). The four
            # loads are independent and overlap.
            page0 = self._page_for_word(table, bt_stride1, bt_width, bs_log2, w0, word0)
            page1 = self._page_for_word(table, bt_stride1, bt_width, bs_log2, w1, word1)
            page2 = self._page_for_word(table, bt_stride1, bt_width, bs_log2, w2, word2)
            page3 = self._page_for_word(table, bt_stride1, bt_width, bs_log2, w3, word3)
            self._emit(
                out,
                table,
                bt_stride1,
                bt_width,
                bs_log2,
                low_mask,
                max_page,
                w0,
                word0,
                page0,
                base0 + incl0 - cnt0,
            )
            self._emit(
                out,
                table,
                bt_stride1,
                bt_width,
                bs_log2,
                low_mask,
                max_page,
                w1,
                word1,
                page1,
                base1 + incl1 - cnt1,
            )
            self._emit(
                out,
                table,
                bt_stride1,
                bt_width,
                bs_log2,
                low_mask,
                max_page,
                w2,
                word2,
                page2,
                base2 + incl2 - cnt2,
            )
            self._emit(
                out,
                table,
                bt_stride1,
                bt_width,
                bs_log2,
                low_mask,
                max_page,
                w3,
                word3,
                page3,
                base3 + incl3 - cnt3,
            )
            carry += ld_shared_i32(s_total)
            cute.arch.sync_threads()
            seg += Int32(_ROUND_WORDS)

        # 3. Unused tail.
        o = carry + tid
        while o < topk:
            st_global_i32(out + Int64(o) * Int64(4), Int32(-1))
            o += Int32(_THREADS)

    @cute.jit
    def _mark(self, bitmap: Int32, limit: Int32, p: Int32):
        """Set the bitmap bit of position ``p`` when it is a valid position
        below ``limit``."""
        if p >= Int32(0):
            if p < limit:
                bit = Uint32(1) << Uint32(p & Int32(31))
                red_or_shared_u32(bitmap + (p >> Int32(5)) * Int32(4), bit)

    @cute.jit
    def _page_for_word(
        self,
        table: Int64,
        bt_stride1: Int64,
        bt_width: Int32,
        bs_log2: Int32,
        w: Int32,
        word: Uint32,
    ) -> Int32:
        """Physical page of the block holding bitmap word ``w`` when blocks
        span whole words (``block_size >= 32``); ``-1`` for an empty word, a
        block beyond the table, or a smaller block size (per-bit lookup)."""
        page = Int32(-1)
        if bs_log2 >= Int32(5):
            if word != Uint32(0):
                blk = (w << Int32(5)) >> bs_log2
                if blk < bt_width:
                    page = ld_global_i32(table + Int64(blk) * bt_stride1 * Int64(4))
        return page

    @cute.jit
    def _emit(
        self,
        out: Int64,
        table: Int64,
        bt_stride1: Int64,
        bt_width: Int32,
        bs_log2: Int32,
        low_mask: Int32,
        max_page: Int32,
        w: Int32,
        word: Uint32,
        page: Int32,
        offset: Int32,
    ):
        """Write the set positions of ``word`` (bitmap word ``w``) ascending
        from output offset ``offset``, converted to physical slots. ``page``
        is the word's page from ``_page_for_word`` (used when
        ``block_size >= 32``). A page outside ``[0, max_page]`` yields
        ``-1``: ``max_page`` is ``INT32_MAX >> bs_log2``, so an admitted
        page's slots never exceed the int32 range."""
        bits = word
        o = offset
        while bits != Uint32(0):
            lowest = bits & (Uint32(0) - bits)
            b = Int32(cute.arch.popc(lowest - Uint32(1)))
            bits = bits & (bits - Uint32(1))
            pos = (w << Int32(5)) | b
            slot = Int32(-1)
            if bs_log2 >= Int32(5):
                if page >= Int32(0):
                    if page <= max_page:
                        slot = (page << bs_log2) | (pos & low_mask)
            else:
                blk = pos >> bs_log2
                if blk < bt_width:
                    page_b = ld_global_i32(table + Int64(blk) * bt_stride1 * Int64(4))
                    if page_b >= Int32(0):
                        if page_b <= max_page:
                            slot = (page_b << bs_log2) | (pos & low_mask)
            st_global_i32(out + Int64(o) * Int64(4), slot)
            o += Int32(1)


def _dummy(dtype, alignment: int):
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def compile_sort_convert(max_words: int):
    """Compile the sort for a bitmap of ``max_words`` words. Returns
    ``launch(indices, seq_lens, block_table, block_size)``."""
    max_words = int(max_words)
    cached = _KERNEL_CACHE.get(max_words)
    if cached is not None:
        return cached
    kernel = TopkSortConvertKernel(max_words)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=(max_words,)
    )
    raw = b12x_compile(
        kernel,
        _dummy(cutlass.Int32, 4),
        _dummy(cutlass.Int32, 4),
        _dummy(cutlass.Int32, 4),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "attention.topk_sort",
            1,
            (max_words,),
        ),
    )

    def launch(
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
    ) -> None:
        rows, topk = indices.shape
        if rows == 0:
            return
        raw(
            make_ptr(
                cutlass.Int32,
                indices.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Int32,
                seq_lens.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Int32,
                block_table.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            int(rows),
            int(indices.stride(0)),
            int(block_table.stride(0)),
            int(block_table.stride(1)),
            int(block_table.shape[1]),
            int(topk),
            int(block_size).bit_length() - 1,
            current_cuda_stream(),
        )

    _KERNEL_CACHE[max_words] = launch
    return launch


def get_cached_sort_convert(max_words: int):
    """Cache-only lookup (no JIT). Returns the launch function or ``None``."""
    return _KERNEL_CACHE.get(int(max_words))


def _check_inputs(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> None:
    if indices.dim() != 2 or indices.dtype != torch.int32 or not indices.is_cuda:
        raise ValueError("indices must be a CUDA int32 [rows, topk] tensor")
    rows, topk = (int(indices.shape[0]), int(indices.shape[1]))
    # The kernel rewrites each row in place from one CTA; rows that share
    # storage (an expanded or as_strided view with a row stride below the
    # row width) would be written by several CTAs at once.
    if indices.stride(1) != 1 or (rows > 1 and topk > 0 and indices.stride(0) < topk):
        raise ValueError("indices rows must be contiguous and non-overlapping")
    if (
        seq_lens.dim() != 1
        or seq_lens.dtype != torch.int32
        or seq_lens.device != indices.device
        or not seq_lens.is_contiguous()
        or int(seq_lens.shape[0]) < rows
    ):
        raise ValueError("seq_lens must be a contiguous int32 [rows] tensor")
    if (
        block_table.dim() != 2
        or block_table.dtype != torch.int32
        or block_table.device != indices.device
        or int(block_table.shape[0]) < rows
    ):
        raise ValueError("block_table must be an int32 [rows, width] tensor")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")


@torch.library.custom_op("b12x::topk_sort_convert", mutates_args=("indices",))
def sort_convert(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> None:
    """Sort each row of ``indices`` ascending by logical position and convert
    the positions to physical slots in place (see the package docstring)."""
    _check_inputs(indices, seq_lens, block_table, block_size)
    if int(indices.shape[0]) == 0:
        return
    max_words = bitmap_words(max_positions)
    # The compiled callable and the launch take the current stream of the
    # current device; select the tensors' device so a launch from another
    # current device does not submit these pointers to a foreign stream.
    with torch.cuda.device(indices.device):
        launch = get_cached_sort_convert(max_words)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "b12x topk_sort was not precompiled for "
                    f"max_positions={max_positions} before CUDA-graph capture"
                )
            launch = compile_sort_convert(max_words)
        launch(indices, seq_lens, block_table, block_size)


@sort_convert.register_fake
def _sort_convert_fake(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> None:
    return None
