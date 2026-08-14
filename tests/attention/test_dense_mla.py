"""Native dense MLA correctness, serving, and addressing gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from b12x.attention import dense_mla

from ..conftest import require_b12x

FP8 = torch.float8_e4m3fn
HEADS = 8
QK_DIM = 576
VALUE_DIM = 512


def _scratch(plan: dense_mla.Plan) -> torch.Tensor:
    (spec,) = plan.scratch_specs()
    return torch.empty(
        spec.shape,
        dtype=spec.dtype,
        device=spec.device,
    )


def test_is_supported_accepts_implicit_current_device() -> None:
    require_b12x()
    assert dense_mla.is_supported()


def _guarded_scratch(
    plan: dense_mla.Plan,
    *,
    guard_bytes: int = 16 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact planned scratch surrounded by initialized canaries."""
    (spec,) = plan.scratch_specs()
    assert spec.dtype == torch.uint8
    storage = torch.full(
        (spec.nbytes + 2 * guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=spec.device,
    )
    scratch = storage.narrow(0, guard_bytes, spec.nbytes)
    return storage, scratch


def _assert_matches(
    output: torch.Tensor,
    lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(output).all().item())
    assert bool(torch.isfinite(lse).all().item())
    assert int(torch.count_nonzero(output).item()) == output.numel()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(output.shape[0], -1),
        reference_output.float().reshape(output.shape[0], -1),
        dim=1,
    )
    assert float(cosine.min().item()) > 0.999
    torch.testing.assert_close(
        output.float(),
        reference_output.float(),
        rtol=2e-2,
        atol=5e-4,
    )
    torch.testing.assert_close(lse, reference_lse, rtol=2e-5, atol=2e-5)


def test_source_is_standalone_cute() -> None:
    root = Path(dense_mla.__file__).resolve().parent
    forbidden = (
        "triton",
        "b12x.attention.paged",
        "b12x.attention.sparse_mla",
        "b12x.attention.nsa_indexer",
        "b12x.attention._shared.mla",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        for name in imports:
            assert not name.startswith(forbidden), (path.name, name)


def test_public_types_are_module_scoped_names() -> None:
    assert dense_mla.Caps.__name__ == "Caps"
    assert dense_mla.Plan.__name__ == "Plan"
    assert dense_mla.Binding.__name__ == "Binding"
    assert dense_mla.Scratch.__name__ == "Scratch"
    assert dense_mla.Budget.__name__ == "Budget"


@torch.inference_mode()
def test_fp8_physical_record_stride_ignores_padding() -> None:
    device = require_b12x()
    torch.manual_seed(20260813)
    rows = 2
    heads = 16
    pages = 3
    page_size = 64
    physical_width = 1088
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=rows,
            max_batch=rows,
            max_cache_tokens=65,
            max_page_table_width=2,
            num_cache_pages=pages,
            physical_record_width=physical_width,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, heads, QK_DIM, device=device) * 10).to(FP8)
    cache = torch.empty(
        pages,
        page_size,
        physical_width,
        dtype=FP8,
        device=device,
    )
    cache[..., :QK_DIM] = (
        torch.randn(pages, page_size, QK_DIM, device=device) * 10
    ).to(FP8)
    cache[..., QK_DIM:] = (
        (torch.randn(pages, page_size, physical_width - QK_DIM, device=device) * 100)
        .clamp(-448, 448)
        .to(FP8)
    )
    page_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([64, 65], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    output = torch.empty(rows, heads, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    actual, actual_lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    _assert_matches(actual, actual_lse, expected, expected_lse)


def test_partial_row_budget_changes_native_split_policy() -> None:
    device = require_b12x()
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=16,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=8,
            budget=dense_mla.Budget(max_partial_rows=0),
        )
    )
    assert plan.num_splits == 1
    assert plan.chunks_per_split == 2


@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_bf16_multi_request_decode_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260730)
    batch = 4
    page_size = 16
    pages = 24
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_cache_tokens=96,
            max_page_table_width=6,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(batch, heads, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [
            [4, 7, 1, 5, 0, 3],
            [18, 2, 19, 6, 8, 21],
            [10, 9, 12, 16, 15, 11],
            [23, 17, 20, 14, 13, 22],
        ],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [1, 17, 63, 91],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.arange(
        batch + 1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (batch, heads, VALUE_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    scratch = _scratch(plan)

    # Compile first through a smaller live batch. The same capacity-planned
    # specialization must then accept the full batch without recompilation or
    # stale tensor-layout assumptions.
    small_output = torch.empty(
        1,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    small_binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q[:1],
        kv_cache=cache,
        output=small_output,
        page_table=page_table[:1],
        cache_seqlens=cache_seqlens[:1],
        cu_seqlens_q=torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=device,
        ),
    )
    small_actual, small_lse = dense_mla.run(binding=small_binding)
    small_expected, small_expected_lse = dense_mla.reference(
        small_binding.q,
        small_binding.kv_cache,
        small_binding.page_table,
        small_binding.cache_seqlens,
        small_binding.cu_seqlens_q,
    )
    _assert_matches(
        small_actual,
        small_lse,
        small_expected,
        small_expected_lse,
    )

    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )

    # Compilation must not launch or mutate graph-visible destinations.
    dense_mla.compile(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isnan(output).all().item())
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_fp8_query_tiled_causal_extend_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260731)
    query_rows = 5
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    assert plan.query_tile == 4
    q_float = torch.randn(query_rows, heads, QK_DIM, device=device) * 0.14
    cache_float = torch.randn(pages, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([77], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_bf16_query_tiled_causal_extend_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260732)
    query_rows = 7
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    assert plan.query_tile == 2
    q = (torch.randn(query_rows, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([101], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_padded_page_stride_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260801)
    page_size = 16
    pages = 5
    page_payload = page_size * QK_DIM
    page_stride = page_payload + 128
    storage = torch.empty(
        pages * page_stride,
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.as_strided(
        storage,
        size=(pages, page_size, QK_DIM),
        stride=(page_stride, QK_DIM, 1),
    )
    cache.normal_().mul_(0.1)
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=64,
            max_page_table_width=4,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor(
        [[4, 1, 3, 0]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([61], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    q_before = q.clone()
    cache_before = cache.clone()
    page_table_before = page_table.clone()
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(cache, cache_before, rtol=0, atol=0)
    torch.testing.assert_close(page_table, page_table_before, rtol=0, atol=0)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_cuda_graph_replay_is_allocation_stable_and_reads_live_inputs() -> None:
    device = require_b12x()
    torch.manual_seed(20260802)
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits > 1
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.arange(
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, pages)
    cache_seqlens = torch.tensor([97], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    q.mul_(0.75)
    cache_seqlens.fill_(65)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after == allocated_before

    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_production_split_plan_handles_short_live_sequence() -> None:
    """The 1M K3 plan must not touch inactive split storage or cache pages."""
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    page_width = (131_072 + page_size - 1) // page_size
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=48,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=131_072,
            max_page_table_width=page_width,
            num_cache_pages=1,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits == 94

    q_float = torch.randn(1, 48, QK_DIM, device=device) * 0.1
    cache_float = torch.randn(1, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([257], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        48,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    guarded_storage, scratch = _guarded_scratch(plan)
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
        active_splits=1,
    )
    assert binding.active_splits == 1
    dense_mla.compile(binding=binding)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    graph.replay()
    torch.cuda.synchronize()
    guard_bytes = scratch.storage_offset()
    expected_guard = torch.full(
        (guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    torch.testing.assert_close(guarded_storage[:guard_bytes], expected_guard)
    torch.testing.assert_close(guarded_storage[-guard_bytes:], expected_guard)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )

    # The active prefix preserves the maximum-context split boundaries and
    # only omits mathematically empty tail splits. It must therefore match a
    # full-plan launch exactly, including the BF16 output bits.
    full_output = torch.empty_like(output)
    full_binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=full_output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    assert full_binding.active_splits == plan.num_splits
    full_actual, full_lse = dense_mla.run(binding=full_binding)
    torch.cuda.synchronize()
    torch.testing.assert_close(full_actual, captured_output, rtol=0, atol=0)
    torch.testing.assert_close(full_lse, captured_lse, rtol=0, atol=0)


@torch.inference_mode()
def test_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 16
    record_bytes = (
        QK_DIM
        * torch.empty(
            (),
            dtype=torch.bfloat16,
        ).element_size()
    )
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    # Roughly 2 GiB is intentionally mostly uninitialized. Only the live tail
    # pages are touched; this reproduces high recycled pool ids.
    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cache[high_page:].normal_().mul_(0.1)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    record_bytes = QK_DIM
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=FP8,
        device=device,
    )
    cache_float = (
        torch.randn(
            live_pages,
            page_size,
            QK_DIM,
            device=device,
        )
        * 0.1
    )
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    cache[high_page:] = (cache_float / kv_scale).to(FP8)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )

# ---------------------------------------------------------------------------
# Issue #157: adversarial metadata-safety gates for dense MLA.
#
# These tests prove that no live dense-MLA metadata can produce an OOB
# table or cache address. Bind performs shape checks only (no host
# synchronisation, no allocation). Device-side predicates cap visible
# cache_length at the first invalid page and use the validated request-local
# page_table[request, 0] fallback for masked gather records on every invocation,
# protecting CUDA graph replay after metadata mutation.
#
# Discrimination strategy: sentinel pages filled with NaN (BF16) or a
# maximally distinct value (FP8) are placed where the old unguarded kernel
# would read. The guarded kernel excludes the invalid suffix and the output
# stays finite / matches the capped safe reference.
# ---------------------------------------------------------------------------


def _tiny_plan(
    device,
    *,
    pages: int = 4,
    page_size: int = 16,
    width: int = 4,
    kv_dtype=torch.bfloat16,
    num_q_heads: int = HEADS,
    use_cuda_graph: bool = False,
) -> dense_mla.Plan:
    return dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * width,
            max_page_table_width=width,
            num_cache_pages=pages,
            use_cuda_graph=use_cuda_graph,
        )
    )


@torch.inference_mode()
def test_bind_rejects_zero_cache_pages() -> None:
    """A kv_cache with zero physical pages must raise (no safe fallback)."""
    device = require_b12x()
    plan = _tiny_plan(device, pages=4, page_size=16, width=4)
    q = torch.zeros(1, HEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.empty(0, 16, QK_DIM, dtype=torch.bfloat16, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    page_table = torch.zeros(1, 4, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([16], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="at least one physical page"):
        dense_mla.bind(
            plan,
            scratch=_scratch(plan),
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )


@torch.inference_mode()
def test_bind_rejects_zero_page_table_width() -> None:
    """A page_table with zero columns must raise (no slot for fallback)."""
    device = require_b12x()
    plan = _tiny_plan(device, pages=4, page_size=16, width=4)
    q = torch.zeros(1, HEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(4, 16, QK_DIM, dtype=torch.bfloat16, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    page_table = torch.empty(1, 0, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([16], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="at least one column"):
        dense_mla.bind(
            plan,
            scratch=_scratch(plan),
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )


@torch.inference_mode()
def test_inactive_padding_negative_ids_binds_and_runs() -> None:
    """Inactive page-table slots with -1 must bind and run; only the active
    prefix (determined by cache_seqlens) is accessed by the device."""
    device = require_b12x()
    torch.manual_seed(20260813)
    page_size = 16
    pages = 8
    width = 8
    plan = _tiny_plan(device, pages=pages, page_size=page_size, width=width)
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table[0, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([32], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_inactive_padding_high_ids_binds_and_runs() -> None:
    """Inactive page-table slots with arbitrary high IDs must bind and run."""
    device = require_b12x()
    torch.manual_seed(20260814)
    page_size = 16
    pages = 8
    width = 8
    plan = _tiny_plan(device, pages=pages, page_size=page_size, width=width)
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.full((1, width), 999, dtype=torch.int32, device=device)
    page_table[0, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([32], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_post_capture_logical_bound_mutation_does_not_oob() -> None:
    """After CUDA graph capture, mutating cache_seqlens so a second logical
    page is live must not cause an OOB table read.

    Plans a wide capacity (8 pages, width 8) but binds a 1-column page-table
    view backed by a [1, 8] tensor.  Sentinel page ID 7 in columns 1-7
    points at a page filled with NaN.  After capture, cache_seqlens is
    mutated to 32, but the kernel caps cache_length at page_table_width(1) *
    page_size(16) = 16.  The old kernel reads backing column 1 (page ID 7)
    and produces NaN output.  The guarded kernel never reads the OOB column
    and the output matches the capped reference (cache_seqlens=16).
    """
    device = require_b12x()
    torch.manual_seed(20260815)
    page_size = 16
    backing_pages = 8
    plan = _tiny_plan(
        device,
        pages=backing_pages,
        page_size=page_size,
        width=backing_pages,
        use_cuda_graph=True,
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(backing_pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    cache[7] = float("nan")  # sentinel
    page_table_backing = torch.full(
        (1, backing_pages), 7, dtype=torch.int32, device=device
    )
    page_table_backing[0, 0] = 0
    page_table = page_table_backing[:, :1]  # 1-column view, stride(0)=8
    assert page_table.is_contiguous()
    cache_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    # Mutate cache_seqlens so logical page 1 *would* be live.  The kernel
    # caps cache_length at page_table_width(1) * page_size(16) = 16, so
    # only page 0 is processed.  Old kernel reads page_table_backing[0, 1]
    # = 7 -> cache page 7 (NaN) -> NaN output.
    cache_seqlens.fill_(2 * page_size)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "output must be finite — sentinel NaN page was read (logical bound failed)"
    )
    # The capped behavior matches a reference with cache_seqlens = page_size.
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_post_capture_physical_bound_mutation_does_not_oob() -> None:
    """After CUDA graph capture, mutating a page-table entry to point past
    the actual cache view must not cause an OOB cache read.

    Plans 8 pages but binds a 2-page cache view of an 8-page backing.
    Sentinel NaN data fills backing pages 2-7.  After capture, slot 1
    is mutated to page ID 5 (below planned capacity but above the
    actual 2-page view).  The kernel detects the invalid page, caps
    cache_length to 1*page_size=16 (only slot 0 is valid), and masks
    tokens 16..31.  The old kernel reads backing page 5 (NaN).  The
    guarded kernel matches the capped reference.
    """
    device = require_b12x()
    torch.manual_seed(20260816)
    page_size = 16
    backing_pages = 8
    plan = _tiny_plan(
        device,
        pages=backing_pages,
        page_size=page_size,
        width=2,
        use_cuda_graph=True,
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache_backing = (
        torch.randn(backing_pages, page_size, QK_DIM, device=device) * 0.1
    ).to(torch.bfloat16)
    cache_backing[2:] = float("nan")  # sentinel pages
    cache = cache_backing[:2]  # 2-page view
    assert cache.is_contiguous()
    page_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    # Mutate slot 1 to page ID 5 (OOB for 2-page view).  The kernel
    # detects the invalid page and caps cache_length to page_size (only
    # slot 0 is valid).  Tokens 16..31 are masked.
    page_table[0, 1] = 5
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "output must be finite — sentinel NaN page was read (physical bound failed)"
    )
    # The capped behavior matches a reference with cache_seqlens=page_size
    # and the original page_table (only page 0's tokens are visible).
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    safe_pt = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, safe_pt, capped_seqlens, cu_seqlens_q
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_post_capture_negative_page_id_does_not_oob() -> None:
    """After CUDA graph capture, mutating a page-table entry to -1 must not
    cause an OOB cache read.  Uses nonzero data to discriminate.

    Slot 1 is mutated to -1.  The kernel detects the invalid page and
    caps cache_length to 1*page_size=16 (only slot 0 is valid).
    Tokens 16..31 are masked.  The reference uses the capped seqlens."""
    device = require_b12x()
    torch.manual_seed(20260817)
    page_size = 16
    pages = 4
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=4, use_cuda_graph=True
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, pages)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    # Mutate slot 1 to -1.  Kernel caps cache_length to page_size.
    page_table[0, 1] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item())
    # Capped to cache_seqlens=page_size — only page 0's tokens visible.
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_fp8_post_capture_physical_bound_mutation_does_not_oob() -> None:
    """FP8 path: after CUDA graph capture, mutating an active page-table
    entry past the actual cache view must not OOB.  Sentinel pages use
    max FP8 value (448) to make OOB reads produce visibly different output."""
    device = require_b12x()
    torch.manual_seed(20260818)
    page_size = 16
    backing_pages = 8
    plan = _tiny_plan(
        device,
        pages=backing_pages,
        page_size=page_size,
        width=2,
        kv_dtype=FP8,
        use_cuda_graph=True,
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    cache_float = torch.randn(backing_pages, page_size, QK_DIM, device=device) * 0.1
    cache_float[2:] = 448.0  # sentinel: max E4M3 value
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[:2].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache_backing = (cache_float / kv_scale).to(FP8)
    cache = cache_backing[:2]  # 2-page view
    assert cache.is_contiguous()
    page_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 1] = 5  # OOB for 2-page view
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "FP8 output must be finite — sentinel page was read (physical bound failed)"
    )
    # Kernel caps cache_length to page_size (only slot 0 valid).
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    safe_pt = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, safe_pt, capped_seqlens, cu_seqlens_q,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_bind_does_not_allocate_or_synchronize() -> None:
    """bind must honor the views-only / no-allocation / no-sync contract."""
    device = require_b12x()
    page_size = 16
    pages = 4
    plan = _tiny_plan(device, pages=pages, page_size=page_size, width=4)
    q = torch.randn(1, HEADS, QK_DIM, device=device).to(torch.bfloat16)
    cache = torch.randn(pages, page_size, QK_DIM, device=device).to(torch.bfloat16)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    page_table = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, pages)
    cache_seqlens = torch.tensor([32], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    scratch = _scratch(plan)

    torch.cuda.synchronize()
    alloc_before = torch.cuda.memory_allocated(device)

    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )

    alloc_after = torch.cuda.memory_allocated(device)
    assert alloc_after == alloc_before, (
        f"bind allocated {alloc_after - alloc_before} bytes; "
        "bind must be allocation-free"
    )
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.page_table.data_ptr() == page_table.data_ptr()
    assert binding.cache_seqlens.data_ptr() == cache_seqlens.data_ptr()
    assert binding.kv_cache.data_ptr() == cache.data_ptr()


@torch.inference_mode()
def test_bf16_poisoned_page0_with_padded_table_matches_reference() -> None:
    """BF16: global page 0 is NaN poison, live prefix maps to page 1+,
    inactive padding is -1.  Output must be finite and match the
    active-prefix reference."""
    device = require_b12x()
    torch.manual_seed(20260820)
    page_size = 16
    pages = 4
    width = 8
    plan = _tiny_plan(device, pages=pages, page_size=page_size, width=width)
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache[2] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table[0, 0] = 1
    page_table[0, 1] = 2
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual_output).all().item()), (
        "output must be finite — poisoned page 0 was used as fallback"
    )
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_fp8_poisoned_page0_with_padded_table_matches_reference() -> None:
    """FP8: global page 0 is 448 (max E4M3) poison, live prefix maps to
    page 1+, inactive padding is -1."""
    device = require_b12x()
    torch.manual_seed(20260821)
    page_size = 16
    pages = 4
    width = 8
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, kv_dtype=FP8
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    cache_float = torch.zeros(pages, page_size, QK_DIM, device=device)
    cache_float[0] = 448.0  # poison global page 0
    cache_float[1] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    cache_float[2] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[1:3].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table[0, 0] = 1
    page_table[0, 1] = 2
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual_output).all().item()), (
        "FP8 output must be finite — poisoned page 0 was used as fallback"
    )
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_bf16_poisoned_page0_high_pid_live_prefix_matches_reference() -> None:
    """BF16: global page 0 is NaN, live prefix maps to high page IDs near
    the tail, inactive padding is -1."""
    device = require_b12x()
    torch.manual_seed(20260822)
    page_size = 16
    pages = 4
    width = 8
    plan = _tiny_plan(device, pages=pages, page_size=page_size, width=width)
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[2] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache[3] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table[0, 0] = 2
    page_table[0, 1] = 3
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual_output).all().item()), (
        "output must be finite — poisoned page 0 was used as fallback"
    )
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_fp8_poisoned_page0_high_pid_live_prefix_matches_reference() -> None:
    """FP8: global page 0 is 448, live prefix maps to high page IDs,
    inactive padding is -1."""
    device = require_b12x()
    torch.manual_seed(20260823)
    page_size = 16
    pages = 4
    width = 8
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, kv_dtype=FP8
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    cache_float = torch.zeros(pages, page_size, QK_DIM, device=device)
    cache_float[0] = 448.0  # poison global page 0
    cache_float[2] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    cache_float[3] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[2:4].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table[0, 0] = 2
    page_table[0, 1] = 3
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual_output).all().item()), (
        "FP8 output must be finite — poisoned page 0 was used as fallback"
    )
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, cache_seqlens, cu_seqlens_q,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    _assert_matches(actual_output, actual_lse, expected_output, expected_lse)


@torch.inference_mode()
def test_fp8_post_capture_logical_bound_mutation_does_not_oob() -> None:
    """FP8 path: after CUDA graph capture, mutating cache_seqlens so a
    second logical page is live must not cause an OOB table read.

    The kernel caps cache_length at page_table_width(1) * page_size(16) =
    16, so only page 0 is processed.  Old kernel reads backing col 1
    (page 7, sentinel 448) and produces divergent output.  Guarded
    kernel matches the capped reference (cache_seqlens=16)."""
    device = require_b12x()
    torch.manual_seed(20260824)
    page_size = 16
    backing_pages = 8
    plan = _tiny_plan(
        device,
        pages=backing_pages,
        page_size=page_size,
        width=backing_pages,
        kv_dtype=FP8,
        use_cuda_graph=True,
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    cache_float = torch.randn(backing_pages, page_size, QK_DIM, device=device) * 0.1
    cache_float[7] = 448.0  # sentinel
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[:2].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table_backing = torch.full(
        (1, backing_pages), 7, dtype=torch.int32, device=device
    )
    page_table_backing[0, 0] = 0
    page_table = page_table_backing[:, :1]  # 1-column view
    assert page_table.is_contiguous()
    cache_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    cache_seqlens.fill_(2 * page_size)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "FP8 output must be finite — sentinel page was read (logical bound failed)"
    )
    # Capped to cache_seqlens=page_size — matches capped reference.
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_bf16_post_capture_dead_request_zeroes_output() -> None:
    """BF16: capture with valid slot 0, then mutate slot 0 to -1 and
    cache_seqlens to positive.  The kernel detects the invalid first
    page, sets cache_length=0, takes the early exit, and zeroes the
    output.  Global page 0 is poisoned with NaN to prove the zeroing
    is real, not a stale read."""
    device = require_b12x()
    torch.manual_seed(20260825)
    page_size = 16
    pages = 4
    width = 4
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, use_cuda_graph=True
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, _ = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    # Mutate slot 0 to -1 (invalid) and cache_seqlens to positive.
    page_table[0, 0] = -1
    cache_seqlens.fill_(page_size)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "output must be finite — dead request was not zeroed"
    )
    assert bool((captured_output == 0).all().item()), (
        "dead request output must be zero, not stale data"
    )


@torch.inference_mode()
def test_fp8_post_capture_dead_request_zeroes_output() -> None:
    """Wide FP8: capture with valid slot 0, then mutate slot 0 high.

    The physical 1088-wide record yields a 1024-wide value output, covering
    the full dead-request zeroing loop rather than only its first 512 lanes.
    """
    device = require_b12x()
    torch.manual_seed(20260826)
    page_size = 16
    pages = 4
    width = 4
    physical_width = 1088
    value_dim = 1024
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * width,
            max_page_table_width=width,
            num_cache_pages=pages,
            head_dim=physical_width,
            v_head_dim=value_dim,
            physical_record_width=physical_width,
            use_cuda_graph=True,
        )
    )
    q_float = torch.randn(1, HEADS, physical_width, device=device) * 0.1
    cache_float = torch.zeros(pages, page_size, physical_width, device=device)
    cache_float[0] = 448.0  # poison global page 0
    cache_float[1] = torch.randn(page_size, physical_width, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[1:2].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, value_dim, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, _ = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 0] = 999999
    cache_seqlens.fill_(page_size)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "FP8 output must be finite — dead request was not zeroed"
    )
    assert bool((captured_output == 0).all().item()), (
        "dead request output must be zero, not stale data"
    )


@torch.inference_mode()
def test_bf16_slot1_invalid_with_poisoned_page0_matches_capped_reference() -> None:
    """BF16: slot 0 valid (page 1), slot 1 mutated to -1 after capture.
    Global page 0 is NaN poison.  Kernel caps cache_length to page_size
    (only slot 0 valid), masks tokens 16..31.  Output matches the
    capped reference and is finite despite poisoned page 0."""
    device = require_b12x()
    torch.manual_seed(20260827)
    page_size = 16
    pages = 4
    width = 4
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, use_cuda_graph=True
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache[2] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor([[1, 2, 1, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 1] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "output must be finite — poisoned page 0 was used"
    )
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_fp8_slot1_invalid_with_poisoned_page0_matches_capped_reference() -> None:
    """FP8: slot 0 valid (page 1), slot 1 mutated to 999999 after capture.
    Global page 0 is 448 poison.  Kernel caps cache_length to page_size,
    masks tokens 16..31.  Output matches the capped reference."""
    device = require_b12x()
    torch.manual_seed(20260828)
    page_size = 16
    pages = 4
    width = 4
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width,
        kv_dtype=FP8, use_cuda_graph=True,
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    cache_float = torch.zeros(pages, page_size, QK_DIM, device=device)
    cache_float[0] = 448.0  # poison global page 0
    cache_float[1] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    cache_float[2] = torch.randn(page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float[1:3].abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.tensor([[1, 2, 1, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan, scratch=_scratch(plan),
        q=q, kv_cache=cache, output=output,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, q_scale=q_scale, kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 1] = 999999
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "FP8 output must be finite — poisoned page 0 was used"
    )
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_bf16_high_valid_pid_slot1_invalid_matches_capped_reference() -> None:
    """BF16: slot 0 maps to a high valid PID (last page), slot 1 mutated
    to -1.  Global page 0 is NaN.  Kernel caps to page_size; output
    matches capped reference with the high PID page."""
    device = require_b12x()
    torch.manual_seed(20260829)
    page_size = 16
    pages = 4
    width = 4
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, use_cuda_graph=True
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[3] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor([[3, 0, 0, 0]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2 * page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan, scratch=_scratch(plan),
        q=q, kv_cache=cache, output=output,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 1] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item()), (
        "output must be finite — poisoned page 0 was used"
    )
    capped_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    expected_output, expected_lse = dense_mla.reference(
        q, cache, page_table, capped_seqlens, cu_seqlens_q
    )
    _assert_matches(
        captured_output, captured_lse, expected_output, expected_lse
    )


@torch.inference_mode()
def test_bf16_dead_request_lse_neg_inf_direct() -> None:
    """BF16 direct path (num_splits==1): slot 0 mutated to -1 after
    capture.  Output must be zero and LSE must be -inf."""
    device = require_b12x()
    torch.manual_seed(20260830)
    page_size = 16
    pages = 4
    width = 4
    # num_splits == 1 for a small plan → direct path
    plan = _tiny_plan(
        device, pages=pages, page_size=page_size, width=width, use_cuda_graph=True
    )
    assert plan.num_splits == 1
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan, scratch=_scratch(plan),
        q=q, kv_cache=cache, output=output,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 0] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item())
    assert bool((captured_output == 0).all().item())
    assert bool((captured_lse == float("-inf")).all().item())


@torch.inference_mode()
def test_bf16_dead_request_lse_neg_inf_merged() -> None:
    """BF16 merged path (num_splits>1, active_splits>1): slot 0 mutated
    to -1 after capture.  Output must be zero and LSE must be -inf."""
    device = require_b12x()
    torch.manual_seed(20260831)
    page_size = 16
    pages = 4
    width = 8
    # Wide enough to force multiple splits
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * width,
            max_page_table_width=width,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits > 1
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = dense_mla.bind(
        plan, scratch=_scratch(plan),
        q=q, kv_cache=cache, output=output,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    dense_mla.compile(binding=binding)
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    page_table[0, 0] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item())
    assert bool((captured_output == 0).all().item())
    assert bool((captured_lse == float("-inf")).all().item())


@torch.inference_mode()
def test_bf16_dead_request_lse_neg_inf_active_splits1() -> None:
    """BF16 active_splits=1 path with plan.num_splits>1: slot 0 mutated
    to -1 after capture.  This exercises the distinct early-exit branch
    at _forward.py where num_splits>1 but active_splits==1 writes
    final_lse (not partial_lse).  Output must be zero and LSE=-inf."""
    device = require_b12x()
    torch.manual_seed(20260832)
    page_size = 16
    pages = 4
    width = 8
    # Wide enough to force multiple splits
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * width,
            max_page_table_width=width,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    assert plan.num_splits > 1
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = torch.zeros(pages, page_size, QK_DIM, dtype=torch.bfloat16, device=device)
    cache[0] = float("nan")  # poison global page 0
    cache[1] = (torch.randn(page_size, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.tensor([page_size], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(1, HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device)
    # Bind with active_splits=1 — distinct from both direct (num_splits=1)
    # and merged (active_splits>1) paths.
    binding = dense_mla.bind(
        plan,
        scratch=_scratch(plan),
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_splits=1,
    )
    assert binding.active_splits == 1
    dense_mla.compile(binding=binding)
    # Run once to produce finite output/LSE before capture.
    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    # Poison output and mutate slot 0 to invalid.
    page_table[0, 0] = -1
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_output).all().item())
    assert bool((captured_output == 0).all().item())
    assert bool((captured_lse == float("-inf")).all().item())
