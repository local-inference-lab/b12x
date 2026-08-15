"""Dense page-table traversal and asynchronous K3 record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_shared_i32,
    shared_ptr_to_u32,
    st_shared_i32,
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

    The producer validates each page id at the point where it forms the cache
    address. It stages a signed page marker in the record's existing padding:
    nonnegative markers are live page ids and negative markers encode the
    request-local fallback page. The consumer uses the same marker to exclude
    invalid records from softmax.

    Markers are published before the bulk-copy transaction begins. The fixed
    transaction byte count therefore retains graph-stable barrier behavior,
    including for invalid and tail records.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    request_local_page = Int32(
        page_table[request.to(Int64) * page_table_stride]
    )

    # Resolve page-table entries once. The encoded markers occupy four of the
    # sixteen padding bytes already present after every staged record.
    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry
        physical_page = request_local_page
        record_valid = Int32(0)
        if logical_token < token_end:
            logical_page = logical_token // Int32(page_size)
            if logical_page < page_table_width:
                candidate_page = Int32(
                    page_table[
                        request.to(Int64) * page_table_stride
                        + logical_page.to(Int64)
                    ]
                )
                if (
                    candidate_page >= Int32(0)
                    and candidate_page < num_cache_pages
                ):
                    physical_page = candidate_page
                    record_valid = Int32(1)

        marker = physical_page
        if record_valid == Int32(0):
            marker = Int32(-1) - physical_page
        st_shared_i32(
            kv_dst_addr
            + entry * Int32(record_stride_bytes)
            + Int32(record_bytes),
            marker,
        )
        entry += Int32(32)

    # mbarrier.try_wait.parity is not a shared-memory fence. Publish the page
    # markers before the transaction whose completion releases the consumers.
    cute.arch.fence_acq_rel_cta()
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(CANDIDATES_PER_CHUNK * record_bytes),
        )

    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry
        marker = ld_shared_i32(
            kv_dst_addr
            + entry * Int32(record_stride_bytes)
            + Int32(record_bytes)
        )
        physical_page = marker
        if marker < Int32(0):
            physical_page = Int32(-1) - marker
        logical_page = logical_token // Int32(page_size)
        in_page = logical_token - logical_page * Int32(page_size)
        if logical_token >= token_end:
            in_page = Int32(0)

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
