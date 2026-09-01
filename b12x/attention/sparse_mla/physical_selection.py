"""Capture-safe GLM pooled-selection expansion to physical MLA cache slots."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_POOL_TOPK = 512
_OUTPUT_WIDTH = 2051


@triton.jit
def _pooled_selection_count_kernel(
    pool_ids,
    positions,
    active_counts,
    pool_stride,
    POOL_SIZE: tl.constexpr,
    POOL_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    sequence_length = tl.maximum(tl.load(positions + row).to(tl.int64) + 1, 0)
    complete_pools = sequence_length // POOL_SIZE
    pool_id = tl.load(
        pool_ids + row * pool_stride + column,
        mask=column < POOL_TOPK,
        other=-1,
    ).to(tl.int64)
    valid = (column < POOL_TOPK) & (pool_id >= 0) & (pool_id < complete_pools)
    first_invalid = tl.min(tl.where(valid, POOL_TOPK, column))
    tail_count = sequence_length - complete_pools * POOL_SIZE
    tl.store(active_counts + row, first_invalid * POOL_SIZE + tail_count)


@triton.jit
def _expand_pooled_selection_kernel(
    pool_ids,
    positions,
    request_ids,
    block_table,
    output,
    active_counts,
    pool_stride,
    block_table_stride,
    output_stride,
    block_table_rows,
    block_table_width,
    block_size,
    block_stride_rows,
    num_cache_blocks,
    POOL_SIZE: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    column_mask = column < OUTPUT_WIDTH
    sequence_length = tl.maximum(tl.load(positions + row).to(tl.int64) + 1, 0)
    active_count = tl.load(active_counts + row).to(tl.int64)
    tail_count = sequence_length % POOL_SIZE
    history_count = active_count - tail_count
    history = column < history_count
    pool_column = column // POOL_SIZE
    pool_offset = column % POOL_SIZE
    pool_id = tl.load(
        pool_ids + row * pool_stride + pool_column,
        mask=column_mask & history,
        other=-1,
    ).to(tl.int64)
    tail_offset = column - history_count
    tail_start = sequence_length - tail_count
    logical_slot = tl.where(
        history,
        pool_id * POOL_SIZE + pool_offset,
        tail_start + tail_offset,
    )
    selected = column < active_count
    request_id = tl.load(request_ids + row).to(tl.int64)
    request_valid = (request_id >= 0) & (request_id < block_table_rows)
    logical_block = logical_slot // block_size
    block_offset = logical_slot - logical_block * block_size
    table_valid = (
        selected
        & request_valid
        & (logical_slot >= 0)
        & (logical_slot < sequence_length)
        & (logical_block >= 0)
        & (logical_block < block_table_width)
    )
    physical_block = tl.load(
        block_table + request_id * block_table_stride + logical_block,
        mask=column_mask & table_valid,
        other=-1,
    ).to(tl.int64)
    physical_valid = (
        table_valid & (physical_block >= 0) & (physical_block < num_cache_blocks)
    )
    physical_slot = physical_block * block_stride_rows + block_offset
    value = tl.where(physical_valid, physical_slot, -1).to(tl.int32)
    tl.store(output + row * output_stride + column, value, mask=column_mask)


def expand_pooled_topk_to_physical_slots(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    active_counts: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    block_stride_rows: int,
    num_cache_blocks: int,
) -> None:
    """Expand selected four-token pools and map them through a page table.

    Valid selected pools must form a prefix of each 512-wide row, which is the
    stable output contract of the B12X paged top-k indexer. Up to three live
    tail tokens are appended immediately after that prefix. ``output`` receives
    compact physical cache slots followed by ``-1`` and ``active_counts``
    receives the compact prefix length. All outputs are caller-owned and both
    launches are CUDA-graph capture safe.
    """
    if pool_ids.ndim != 2 or tuple(pool_ids.shape[1:]) != (_POOL_TOPK,):
        raise ValueError("pool_ids must be int32 [rows, 512]")
    rows = int(pool_ids.shape[0])
    if pool_ids.dtype != torch.int32 or not pool_ids.is_contiguous():
        raise ValueError("pool_ids must be contiguous int32 [rows, 512]")
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("positions must be int64 [rows]")
    if request_ids.shape != (rows,) or request_ids.dtype != torch.int32:
        raise ValueError("request_ids must be int32 [rows]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be rank-2 int32")
    if output.shape != (rows, _OUTPUT_WIDTH) or output.dtype != torch.int32:
        raise ValueError("output must be int32 [rows, 2051]")
    if active_counts.shape != (rows,) or active_counts.dtype != torch.int32:
        raise ValueError("active_counts must be int32 [rows]")
    tensors = (pool_ids, positions, request_ids, block_table, output, active_counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("pooled physical selection requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("pooled physical-selection tensors must share a device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("pooled physical-selection tensors must be contiguous")
    if int(pool_size) != 4:
        raise ValueError(f"GLM pooled selection requires pool_size=4, got {pool_size}")
    if int(block_size) <= 0 or int(block_size) % int(pool_size):
        raise ValueError("block_size must be positive and divisible by pool_size")
    if int(block_stride_rows) < int(block_size):
        raise ValueError("block_stride_rows must cover one logical cache block")
    if int(num_cache_blocks) <= 0:
        raise ValueError("num_cache_blocks must be positive")
    if rows == 0:
        return

    _pooled_selection_count_kernel[(rows,)](
        pool_ids,
        positions,
        active_counts,
        int(pool_ids.stride(0)),
        POOL_SIZE=int(pool_size),
        POOL_TOPK=_POOL_TOPK,
        BLOCK=512,
        num_warps=4,
    )
    block_cols = 128
    _expand_pooled_selection_kernel[
        (rows, triton.cdiv(_OUTPUT_WIDTH, block_cols))
    ](
        pool_ids,
        positions,
        request_ids,
        block_table,
        output,
        active_counts,
        int(pool_ids.stride(0)),
        int(block_table.stride(0)),
        int(output.stride(0)),
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        int(block_size),
        int(block_stride_rows),
        int(num_cache_blocks),
        POOL_SIZE=int(pool_size),
        OUTPUT_WIDTH=_OUTPUT_WIDTH,
        BLOCK_COLS=block_cols,
        num_warps=4,
    )


__all__ = ["expand_pooled_topk_to_physical_slots"]
