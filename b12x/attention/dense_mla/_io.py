"""Dense page-table traversal and asynchronous K3 record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    shared_ptr_to_u32,
)

from ._layout import CANDIDATES_PER_CHUNK


@cute.jit
def issue_dense_page_gather(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    record_stride_bytes_global: Int64,
    page_table_stride: Int64,
    page_table_width: Int32,
    num_cache_pages: Int32,
    *,
    page_size: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
):
    """Stage one 64-record dense window with an mbarrier transaction.

    Every pool-scaled product is promoted before multiplication. Invalid
    tail rows copy a request-local page (``page_table[request, 0]``, clamped
    to ``[0, num_cache_pages-1]``) and are masked by the consumer; issuing a
    fixed byte count keeps the full-barrier protocol independent of live
    sequence length and therefore graph-replay stable.

    The page-table lookup is executed only for tokens within the live
    sequence length whose logical page fits the actual table width
    (``page_table_width``, a capture-static shape constant).  Physical page
    IDs from the table are clamped to ``[0, num_cache_pages-1]`` (also
    capture-static).  Every other entry falls back to the request-local
    page, record zero — preventing out-of-bounds table or cache addresses
    even when ``cache_seqlens`` or ``page_table`` values are mutated after
    CUDA graph capture.

    The kernel additionally caps ``cache_length`` at the table capacity and
    zeroes the request when ``page_table[request, 0]`` is invalid, so the
    request-local fallback here is only reached for requests whose first
    page is valid.  Masked PV MMA therefore multiplies zero probability by
    finite data from a real request-local page, not by NaN/Inf from an
    uninitialized global page.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(CANDIDATES_PER_CHUNK * record_bytes),
        )

    # Request-local safe page: load page_table[request, 0] once and
    # validate against the actual cache extent.  This is the fallback
    # source for tail / out-of-range / corrupted entries.  It is
    # guaranteed to be a real, initialized page belonging to this
    # request (the caller maps the first active page at slot 0).
    # page_table_width >= 1 and num_cache_pages >= 1 are enforced at
    # bind by shape-only checks, so the table load and fallback are
    # always safe.
    request_local_page = Int32(
        page_table[request.to(Int64) * page_table_stride]
    )
    if request_local_page < Int32(0):
        request_local_page = Int32(0)
    elif request_local_page >= num_cache_pages:
        request_local_page = num_cache_pages - Int32(1)

    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry

        # Default to the validated request-local page and record zero. Masked
        # records still issue a fixed byte count for the barrier protocol.
        physical_page = request_local_page
        in_page = Int32(0)
        if logical_token < token_end:
            logical_page = logical_token // Int32(page_size)
            if logical_page < page_table_width:
                in_page = logical_token - logical_page * Int32(page_size)
                physical_page = Int32(
                    page_table[
                        request.to(Int64) * page_table_stride
                        + logical_page.to(Int64)
                    ]
                )
                if physical_page < Int32(0) or physical_page >= num_cache_pages:
                    physical_page = request_local_page

        # Load-bearing Int64 conversions. Neither multiplication may occur in
        # Int32, even when benchmark page ids happen to be small.
        source_offset = (
            physical_page.to(Int64) * page_stride_bytes
            + in_page.to(Int64) * record_stride_bytes_global
        )
        cp_async_bulk_g2s_mbar(
            kv_dst_addr + entry * Int32(record_stride_bytes),
            get_ptr_as_int64(cache_bytes, source_offset),
            Int32(record_bytes),
            full_mbar_addr,
        )
        entry += Int32(32)


__all__ = ["issue_dense_page_gather"]
