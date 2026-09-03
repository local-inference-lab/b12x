"""Triton sparse paged GQA for large Qwen prefill batches.

The kernel assigns one program to each query-row and KV-head pair. Large
prefill batches therefore provide enough independent work without splitting
one attention row across intermediate FP32 tensors. Logical token positions
are translated through the caller's page table, so the kernel reads the
existing QSA cache without a contiguous-cache gather.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_paged_gqa_prefill(
    query,
    key_cache,
    value_cache,
    k_descale,
    v_descale,
    block_table,
    request_ids,
    selected_positions,
    query_positions,
    output,
    rows,
    num_cache_pages,
    num_requests,
    table_width,
    softmax_scale,
    QUERY_STRIDE_ROW: tl.constexpr,
    QUERY_STRIDE_HEAD: tl.constexpr,
    KEY_STRIDE_PAGE: tl.constexpr,
    KEY_STRIDE_TOKEN: tl.constexpr,
    KEY_STRIDE_HEAD: tl.constexpr,
    VALUE_STRIDE_PAGE: tl.constexpr,
    VALUE_STRIDE_TOKEN: tl.constexpr,
    VALUE_STRIDE_HEAD: tl.constexpr,
    TABLE_STRIDE_ROW: tl.constexpr,
    SELECTED_STRIDE_ROW: tl.constexpr,
    OUTPUT_STRIDE_ROW: tl.constexpr,
    OUTPUT_STRIDE_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    FP8_CACHE: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    if row >= rows:
        return

    request_id = tl.load(request_ids + row).to(tl.int64)
    query_position = tl.load(query_positions + row).to(tl.int64)
    row_is_valid = (request_id >= 0) & (request_id < num_requests)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    first_head = kv_head * HEADS_PER_KV
    query_values = tl.load(
        query
        + row * QUERY_STRIDE_ROW
        + (first_head + head_offsets[:, None]) * QUERY_STRIDE_HEAD
        + dim_offsets[None, :],
        mask=(head_offsets < HEADS_PER_KV)[:, None],
        other=0.0,
    )
    key_scale = tl.load(k_descale) if FP8_CACHE else 1.0
    value_scale = tl.load(v_descale) if FP8_CACHE else 1.0
    query_values = (query_values * softmax_scale * key_scale * 1.4426950408889634).to(
        query_values.dtype
    )

    maximum = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    selection_offsets = tl.arange(0, BLOCK_N)
    visible_selection = tl.minimum(SELECTION_WIDTH, query_position + 1)
    selection_limit = ((visible_selection + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    for start in range(0, selection_limit, BLOCK_N):
        selection_index = start + selection_offsets
        logical_position = tl.load(
            selected_positions + row * SELECTED_STRIDE_ROW + selection_index,
            mask=selection_index < SELECTION_WIDTH,
            other=-1,
        ).to(tl.int64)
        logical_page = logical_position // PAGE_SIZE
        page_offset = logical_position - logical_page * PAGE_SIZE
        position_is_valid = (
            row_is_valid
            & (logical_position >= 0)
            & (logical_position <= query_position)
            & (logical_page >= 0)
            & (logical_page < table_width)
        )
        physical_page = tl.load(
            block_table + request_id * TABLE_STRIDE_ROW + logical_page,
            mask=position_is_valid,
            other=-1,
        ).to(tl.int64)
        position_is_valid &= (physical_page >= 0) & (physical_page < num_cache_pages)

        keys = tl.load(
            key_cache
            + physical_page[None, :] * KEY_STRIDE_PAGE
            + page_offset[None, :] * KEY_STRIDE_TOKEN
            + kv_head * KEY_STRIDE_HEAD
            + dim_offsets[:, None],
            mask=position_is_valid[None, :],
            other=0.0,
        ).to(query_values.dtype)
        values = tl.load(
            value_cache
            + physical_page[:, None] * VALUE_STRIDE_PAGE
            + page_offset[:, None] * VALUE_STRIDE_TOKEN
            + kv_head * VALUE_STRIDE_HEAD
            + dim_offsets[None, :],
            mask=position_is_valid[:, None],
            other=0.0,
        ).to(query_values.dtype)
        if FP8_CACHE:
            values = (values * value_scale).to(query_values.dtype)

        scores = tl.where(
            position_is_valid[None, :],
            tl.dot(query_values, keys),
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - next_maximum)
        probabilities = tl.exp2(scores - next_maximum[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        maximum = next_maximum

    normalized = accumulator / normalizer[:, None]
    normalized = tl.where(normalizer[:, None] > 0.0, normalized, 0.0)
    tl.store(
        output
        + row * OUTPUT_STRIDE_ROW
        + (first_head + head_offsets[:, None]) * OUTPUT_STRIDE_HEAD
        + dim_offsets[None, :],
        normalized,
        mask=(head_offsets < HEADS_PER_KV)[:, None],
    )


def launch_sparse_paged_gqa_prefill(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the large-row sparse GQA kernel into caller-owned output storage."""
    rows, query_heads, head_dim = map(int, query.shape)
    kv_heads = int(key_cache.shape[2])
    heads_per_kv = query_heads // kv_heads
    block_m = max(16, triton.next_power_of_2(heads_per_kv))
    fp8_cache = key_cache.dtype == torch.float8_e4m3fn
    if fp8_cache and (k_descale is None or v_descale is None):
        raise ValueError("FP8 QSA caches require K/V descale tensors")
    scale_pointer = query if k_descale is None else k_descale
    value_scale_pointer = query if v_descale is None else v_descale
    _sparse_paged_gqa_prefill[(rows, kv_heads)](
        query,
        key_cache,
        value_cache,
        scale_pointer,
        value_scale_pointer,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        output,
        rows,
        int(key_cache.shape[0]),
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        float(softmax_scale),
        QUERY_STRIDE_ROW=int(query.stride(0)),
        QUERY_STRIDE_HEAD=int(query.stride(1)),
        KEY_STRIDE_PAGE=int(key_cache.stride(0)),
        KEY_STRIDE_TOKEN=int(key_cache.stride(1)),
        KEY_STRIDE_HEAD=int(key_cache.stride(2)),
        VALUE_STRIDE_PAGE=int(value_cache.stride(0)),
        VALUE_STRIDE_TOKEN=int(value_cache.stride(1)),
        VALUE_STRIDE_HEAD=int(value_cache.stride(2)),
        TABLE_STRIDE_ROW=int(block_table.stride(0)),
        SELECTED_STRIDE_ROW=int(selected_positions.stride(0)),
        OUTPUT_STRIDE_ROW=int(output.stride(0)),
        OUTPUT_STRIDE_HEAD=int(output.stride(1)),
        PAGE_SIZE=int(key_cache.shape[1]),
        HEADS_PER_KV=heads_per_kv,
        BLOCK_M=block_m,
        BLOCK_N=16,
        HEAD_DIM=head_dim,
        SELECTION_WIDTH=int(selected_positions.shape[1]),
        FP8_CACHE=fp8_cache,
        num_warps=2,
        num_stages=1,
    )
    return output[:rows]


__all__ = ["launch_sparse_paged_gqa_prefill"]
