"""Adversarial GPU regression tests for MSA q2k block-index range safety (issue #159).

Tests compile and launch real SM121 BF16 and FP8 MSA decode + union extend
for page sizes 64 and 128.  They exercise graph replay mutation with sentinel
(-1), negative (-2), INT32_MAX, physically-valid-but-logically-dead, and
odd-width partial-final blocks, comparing against the reference implementation
which skips invalid blocks.

The producer-before-attention capture lifecycle is followed: q2k_indices is
allocated with torch.empty, bound, then the indexer producer fills it.
Poison tests capture ONCE, mutate the stable q2k buffer, replay the SAME
graph, and compare output+LSE elementwise to the reference.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.nsa_indexer.reference import pack_index_k_cache_reference
from b12x.attention.paged.reference import msa_attention_reference
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    _msa_max_valid_block_id,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.planner import create_paged_plan

from tests._reference.helpers import require_b12x
from tests._reference.paged_attention_helpers import (
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)


_INDEX_HEAD_DIM = 128
_MSA_BLOCK_TOKENS = 128
_MSA_TOPK = 16


def _pack_index_cache_from_attention_k(k_cache: torch.Tensor) -> torch.Tensor:
    k_idx = k_cache[:, :, 0, :].contiguous().view(-1, k_cache.shape[-1])
    return pack_index_k_cache_reference(k_idx, page_size=int(k_cache.shape[1]))


def _make_valid_q2k(
    kv_heads: int,
    total_q: int,
    cache_lens: list[int],
    page_size: int,
    page_table_width: int,
    device: torch.device,
    cu_seqlens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create valid q2k_indices with blocks in [0, max_block_id].

    For decode, each row i maps to request i (cu_seqlens_q=None implies 1:1).
    For extend, cu_seqlens_q maps each row to its request.
    """
    max_block_id = _msa_max_valid_block_id(page_table_width, page_size)
    q2k_indices = torch.full(
        (kv_heads, total_q, _MSA_TOPK), -1, dtype=torch.int32, device=device
    )
    if cu_seqlens_q is not None:
        offsets = cu_seqlens_q.detach().cpu().tolist()
    else:
        offsets = list(range(len(cache_lens) + 1))
    for req_idx in range(len(cache_lens)):
        q_start = int(offsets[req_idx])
        q_end = int(offsets[req_idx + 1])
        cache_len = cache_lens[req_idx]
        num_blocks = min(
            (cache_len + _MSA_BLOCK_TOKENS - 1) // _MSA_BLOCK_TOKENS,
            _MSA_TOPK,
        )
        for h in range(kv_heads):
            for j in range(num_blocks):
                block_id = min(j, max_block_id)
                for q_row in range(q_start, q_end):
                    q2k_indices[h, q_row, j] = block_id
    return q2k_indices


def _assert_close_to_ref(
    output: torch.Tensor,
    lse: torch.Tensor,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
    *,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> None:
    """Elementwise comparison; handles zero-norm (all-sentinel) case."""
    torch.testing.assert_close(lse, ref_lse, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        output.to(torch.float32), ref_out.to(torch.float32), rtol=rtol, atol=atol
    )


def _make_decode_scratch(
    q, k_cache, v_cache, page_table, page_table_width, num_pages
):
    batch = q.shape[0]
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    return scratch_plan, scratch


# ---------------------------------------------------------------------------
# Decode: BF16 page64 graph replay — capture once, mutate, replay same graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poison_value",
    [-1, -2, 0x7FFFFFFF],
    ids=["sentinel", "negative", "int32max"],
)
def test_msa_decode_bf16_page64_same_graph_mutation(poison_value: int) -> None:
    """Capture once, mutate stable q2k, replay SAME graph, compare to ref."""
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 2
    page_table_width = 80
    num_pages = 160
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 5000],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=4101,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    q2k_indices = _make_valid_q2k(
        kv_heads, batch, [2048, 5000], page_size, page_table_width, q.device
    )

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    # Capture ONCE
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)

    # Baseline replay with valid q2k
    graph.replay()
    torch.cuda.synchronize()
    ref_valid, ref_lse_valid = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse_valid = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse_valid, ref_valid, ref_lse_valid)

    # Mutate stable q2k buffer IN PLACE and replay SAME graph
    q2k_indices[0, 0, 8:] = poison_value
    graph.replay()
    torch.cuda.synchronize()

    ref_mut, ref_lse_mut = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse_mut = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse_mut, ref_mut, ref_lse_mut)


# ---------------------------------------------------------------------------
# Decode: BF16 page128 graph replay
# ---------------------------------------------------------------------------


def test_msa_decode_bf16_page128_graph_replay() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 128
    batch = 2
    page_table_width = 64
    num_pages = 128
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[1024, 4096],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=4201,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    q2k_indices = _make_valid_q2k(
        kv_heads, batch, [1024, 4096], page_size, page_table_width, q.device
    )

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Decode: odd page_table_width=3 page_size=64 with adversarial [1, 0] ordering
# ---------------------------------------------------------------------------


def test_msa_decode_bf16_page64_odd_width_adversarial() -> None:
    """width=3, cache_len=192 (3 pages).  Block 1 subpage 1 = page_idx 3 == width (OOB).
    Adversarial q2k=[1, 0] forces the walk to hit block 1 first, reaching subpage 1.
    """
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 1
    page_table_width = 3
    num_pages = 16
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[192],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=4301,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    # Adversarial ordering: block 1 first, then block 0.
    # Block 1: page_idx 2 (subpage 0, valid) + page_idx 3 (subpage 1, OOB).
    # The kernel must predicate page_idx 3 >= page_table_width=3.
    q2k_indices = torch.full(
        (kv_heads, batch, _MSA_TOPK), -1, dtype=torch.int32, device=q.device
    )
    q2k_indices[:, 0, 0] = 1  # block 1 first (adversarial)
    q2k_indices[:, 0, 1] = 0  # block 0 second

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Decode: physically-valid but logically-dead block
# ---------------------------------------------------------------------------


def test_msa_decode_bf16_physically_valid_logically_dead() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 1
    page_table_width = 80
    num_pages = 160
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[512],  # 4 blocks walked; max_block_id at slot 4 is logically dead
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=4401,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    q2k_indices = torch.full(
        (kv_heads, batch, _MSA_TOPK), -1, dtype=torch.int32, device=q.device
    )
    max_block_id = _msa_max_valid_block_id(page_table_width, page_size)
    # Fill slots 0-3 with valid blocks that are actually walked (cache_len=512)
    for j in range(4):
        q2k_indices[:, 0, j] = j
    # Slot 4: physically valid but logically dead (block_start >> cache_len)
    q2k_indices[:, 0, 4] = max_block_id

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Decode: INT32_MAX arithmetic overflow test
# ---------------------------------------------------------------------------


def test_msa_decode_bf16_page64_int32max_arithmetic() -> None:
    """INT32_MAX in q2k must not overflow tail arithmetic or produce wrong selected_token_count.
    The reference skips INT32_MAX (block_start = INT32_MAX*128 overflows), and the kernel
    must match by treating it as logically dead (block_start >= cache_len in Int64).
    """
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 1
    page_table_width = 80
    num_pages = 160
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[2048],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=4501,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    # Fill slots 0-3 with valid blocks, slot 4 with INT32_MAX
    q2k_indices = torch.full(
        (kv_heads, batch, _MSA_TOPK), -1, dtype=torch.int32, device=q.device
    )
    for j in range(4):
        q2k_indices[:, 0, j] = j
    q2k_indices[:, 0, 4] = 0x7FFFFFFF  # INT32_MAX: would overflow Int32 tail math

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Decode: canary block 0, final cache block, final physical block, padding
# ---------------------------------------------------------------------------


def test_msa_decode_bf16_page64_canary_blocks() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 1
    page_table_width = 32
    num_pages = 64
    q_heads = 64
    kv_heads = 4
    cache_len = 2048

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[cache_len],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=6101,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )

    max_block_id = _msa_max_valid_block_id(page_table_width, page_size)
    q2k_indices = torch.full(
        (kv_heads, batch, _MSA_TOPK), -1, dtype=torch.int32, device=q.device
    )
    q2k_indices[:, 0, 0] = 0
    q2k_indices[:, 0, 1] = 15
    q2k_indices[:, 0, 2] = max_block_id

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Extend: BF16 union-tile page64 with poisoned q2k
# ---------------------------------------------------------------------------


def test_msa_extend_bf16_page64_union_poisoned() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 5],
        cache_seqlens=[384, 512],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=5101,
    )

    q2k_indices = _make_valid_q2k(
        kv_heads,
        q.shape[0],
        [384, 512],
        page_size,
        page_table.shape[1],
        q.device,
        cu_seqlens_q=cu_seqlens_q,
    )

    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    lse = lse_base2 * math.log(2.0)

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close_to_ref(output, lse, ref_out, ref_lse)

    # Poison trailing slots with INT32_MAX
    q2k_indices[:, :, 8:] = 0x7FFFFFFF
    out_mut, lse_base2_mut = paged_attention_forward(binding=binding)
    lse_mut = lse_base2_mut * math.log(2.0)

    ref_mut, ref_lse_mut = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close_to_ref(output, lse_mut, ref_mut, ref_lse_mut)


# ---------------------------------------------------------------------------
# Extend: BF16 union-tile page128
# ---------------------------------------------------------------------------


def test_msa_extend_bf16_page128_union() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 128
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 5],
        cache_seqlens=[512, 768],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=5201,
    )

    q2k_indices = _make_valid_q2k(
        kv_heads,
        q.shape[0],
        [512, 768],
        page_size,
        page_table.shape[1],
        q.device,
        cu_seqlens_q=cu_seqlens_q,
    )

    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    lse = lse_base2 * math.log(2.0)

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Extend: all-sentinel q2k (zero-count union)
# ---------------------------------------------------------------------------


def test_msa_extend_bf16_page64_all_sentinel() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 5],
        cache_seqlens=[384, 512],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=5301,
    )

    q2k_indices = torch.full(
        (kv_heads, q.shape[0], _MSA_TOPK), -1, dtype=torch.int32, device=q.device
    )

    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    lse = lse_base2 * math.log(2.0)

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# FP8 decode: page64 graph replay
# ---------------------------------------------------------------------------


def test_msa_decode_fp8_page64_graph_replay() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 2
    page_table_width = 80
    num_pages = 160
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=8101,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )
    k_quant, v_quant, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )

    q2k_indices = _make_valid_q2k(
        kv_heads, batch, [2048, 4096], page_size, page_table_width, q.device
    )

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_quant.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_quant.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_quant.shape[3],
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_quant,
        v_cache=v_quant,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q,
        k_quant,
        v_quant,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)

    # Mutate stable q2k and replay SAME graph
    q2k_indices[0, 0, 8:] = 0x7FFFFFFF
    graph.replay()
    torch.cuda.synchronize()

    ref_mut, ref_lse_mut = msa_attention_reference(
        q,
        k_quant,
        v_quant,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    lse_mut = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse_mut, ref_mut, ref_lse_mut)


# ---------------------------------------------------------------------------
# FP8 extend: page128 union-tile
# ---------------------------------------------------------------------------


def test_msa_extend_fp8_page128_union() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 128
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 5],
        cache_seqlens=[512, 768],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=8201,
    )
    k_quant, v_quant, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache, v_cache, page_table, cache_seqlens
    )

    q2k_indices = _make_valid_q2k(
        kv_heads,
        q.shape[0],
        [512, 768],
        page_size,
        page_table.shape[1],
        q.device,
        cu_seqlens_q=cu_seqlens_q,
    )

    plan = create_paged_plan(
        q,
        k_quant,
        v_quant,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_quant.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_quant.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_quant.shape[3],
            page_size=page_size,
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_quant.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_quant,
        v_cache=v_quant,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    lse = lse_base2 * math.log(2.0)

    ref_out, ref_lse = msa_attention_reference(
        q,
        k_quant,
        v_quant,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


# ---------------------------------------------------------------------------
# Producer-before-attention capture lifecycle
# ---------------------------------------------------------------------------


def test_msa_decode_producer_before_attention_lifecycle() -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 64
    batch = 2
    page_table_width = 80
    num_pages = 160
    q_heads = 64
    kv_heads = 4

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 5000],
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=7101,
        page_table_width=page_table_width,
        num_pages=num_pages,
    )
    index_k_cache = _pack_index_cache_from_attention_k(k_cache)

    q2k_indices = torch.empty(
        (kv_heads, batch, _MSA_TOPK), dtype=torch.int32, device=q.device
    )

    scratch_plan, scratch = _make_decode_scratch(
        q, k_cache, v_cache, page_table, page_table_width, num_pages
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    runtime_page_table = binding.scratch.page_table
    runtime_cache_seqlens = binding.scratch.cache_seqlens
    assert runtime_page_table is not None
    assert runtime_cache_seqlens is not None

    from tests.attention.test_attention_msa_e2e import (
        _decode_msa_q2k_from_index_cache,
    )

    _decode_msa_q2k_from_index_cache(
        q=q,
        index_k_cache=index_k_cache,
        page_table=runtime_page_table,
        cache_seqlens=runtime_cache_seqlens,
        out=q2k_indices,
    )
    paged_attention_forward(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode_msa_q2k_from_index_cache(
            q=q,
            index_k_cache=index_k_cache,
            page_table=runtime_page_table,
            cache_seqlens=runtime_cache_seqlens,
            out=q2k_indices,
        )
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    _assert_close_to_ref(output, lse, ref_out, ref_lse)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
