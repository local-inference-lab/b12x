from __future__ import annotations

import threading
import weakref
from dataclasses import replace

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm import tensor_fp8_linear
from b12x.gemm.tensor_fp8_linear import _kernel as _tfp8_kernel
from b12x.gemm.tensor_fp8_linear import api as tensor_fp8_api
from b12x.gemm.tensor_fp8_linear._kernel import (
    _parse_cache_limit,
    _reset_unit_scale_cache,
    _unit_scale_packed_bytes,
    _UNIT_SCALE_HARD_MAX_BYTES,
    _UNIT_SCALE_HARD_MAX_ENTRIES,
)

from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    output_scale = torch.tensor([0.0002], dtype=torch.float32, device="cuda")
    packed = tensor_fp8_linear.pack_weight(weight, output_scale)
    return source, weight, output_scale, packed


def test_mm_matches_static_tensor_fp8_reference() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260729)

    source, weight, output_scale, packed = _make_inputs(7, 128, 64)
    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()

    assert actual.shape == (7, 64)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260730)

    source, weight, output_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.values.shape == (40, 256)
    assert torch.count_nonzero(packed.values[:, 160:]) == 0

    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_uses_plain_fp8_mma_not_scale_storage() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260731)

    source, weight, output_scale, packed = _make_inputs(4, 128, 64)
    packed = replace(packed, scale_mma=torch.zeros_like(packed.scale_mma))

    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_is_supported_honors_kernel_probe(monkeypatch) -> None:
    monkeypatch.setattr(tensor_fp8_api, "default_is_supported", lambda *args, **kw: True)
    monkeypatch.setattr(
        tensor_fp8_api,
        "_kernel_is_supported",
        lambda: (False, "plain FP8 MMA unavailable"),
    )

    assert not tensor_fp8_api.is_supported()


def test_mm_default_path_captures() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260801)

    source, _, _, packed = _make_inputs(4, 128, 64)
    eager = tensor_fp8_linear.mm(source, packed).clone()
    torch.cuda.synchronize()

    tensor_fp8_linear.prewarm(packed, [4])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tensor_fp8_linear.mm(source, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Issue #166: graph-safe bounded admission cache for unit-scale MMA tensors
# ---------------------------------------------------------------------------



@pytest.fixture()
def _clean_cache():
    """Reset the admission cache before and after each cache test."""
    _reset_unit_scale_cache()
    yield
    _reset_unit_scale_cache()


# --- precompute / canonicalization tests (no GPU required for math) ---


def test_unit_scale_packed_bytes_small() -> None:
    """rows=7, width=128 -> 1 tile * 1 tile * 512 = 512 bytes."""
    assert _unit_scale_packed_bytes(7, 128) == 512


def test_unit_scale_packed_bytes_canonical() -> None:
    """rows=128 and rows=1 produce the same packed bytes (same tile)."""
    assert _unit_scale_packed_bytes(1, 128) == _unit_scale_packed_bytes(128, 128)
    assert _unit_scale_packed_bytes(129, 128) == 2 * _unit_scale_packed_bytes(1, 128)


def test_unit_scale_packed_bytes_large_width() -> None:
    """width=512 -> sf_k=16, k_tiles=4, m_tiles=1 -> 1*4*512 = 2048."""
    assert _unit_scale_packed_bytes(128, 512) == 2048


# --- stable allocation / graph safety tests ---


def test_admission_cache_stable_allocation_same_key(_clean_cache) -> None:
    """Same key returns the identical tensor object (graph replay safety)."""
    require_b12x()
    require_mxf8_mma()
    a = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 7, 128)
    b = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 7, 128)
    assert a is b


def test_admission_cache_canonical_rows_share_entry(_clean_cache) -> None:
    """Token counts 1-128 share one cache entry (128-row tile)."""
    require_b12x()
    require_mxf8_mma()
    a = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 1, 128)
    b = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 50, 128)
    c = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 128, 128)
    assert a is b
    assert b is c
    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    assert len(dc.entries) == 1


def test_admission_cache_rejects_before_allocation(_clean_cache) -> None:
    """Oversize geometry is rejected before _unit_scale_mma is called."""
    require_b12x()
    require_mxf8_mma()
    original_bytes = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES
    original_fn = _tfp8_kernel._unit_scale_mma
    called = threading.Event()
    _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES = 1
    try:
        def _spy(*args, **kwargs):
            called.set()
            return original_fn(*args, **kwargs)

        _tfp8_kernel._unit_scale_mma = _spy
        with pytest.raises(RuntimeError, match="exceeding.*byte.*budget"):
            _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 128, 4096)
        assert not called.is_set(), "_unit_scale_mma must not be called for oversize"
    finally:
        _tfp8_kernel._unit_scale_mma = original_fn
        _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES = original_bytes


def test_admission_cache_rejects_aggregate_before_allocation(_clean_cache) -> None:
    """Aggregate-over-budget is rejected before allocating the new entry.

    The candidate must individually fit the budget but its sum with the
    already-used bytes must exceed it, so the 'would exceed' branch is taken
    rather than the single-entry 'exceeding' branch."""
    require_b12x()
    require_mxf8_mma()
    # Admit one entry: (128, 256) = 1024 bytes.
    _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 128, 256)
    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    used = dc.total_bytes  # 1024

    # Candidate (128, 128) = 512 bytes, individually fits.
    candidate_bytes = _unit_scale_packed_bytes(128, 128)  # 512
    # Set budget so used + candidate exceeds but candidate alone fits.
    original_bytes = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES
    original_fn = _tfp8_kernel._unit_scale_mma
    called = threading.Event()
    _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES = used + candidate_bytes - 1  # 1535
    try:
        def _spy(*args, **kwargs):
            called.set()
            return original_fn(*args, **kwargs)

        _tfp8_kernel._unit_scale_mma = _spy
        with pytest.raises(RuntimeError, match="would exceed"):
            _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 128, 128)
        assert not called.is_set()
    finally:
        _tfp8_kernel._unit_scale_mma = original_fn
        _tfp8_kernel._UNIT_SCALE_CACHE_MAX_BYTES = original_bytes


def test_admission_cache_rejects_at_entry_capacity(_clean_cache) -> None:
    """New geometries are rejected once the entry-count ceiling is reached."""
    require_b12x()
    require_mxf8_mma()
    max_entries = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES
    for i in range(max_entries):
        _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 1 + i * 128, 128)
    with pytest.raises(RuntimeError, match="cache.*full"):
        _tfp8_kernel._cached_unit_scale_mma(
            "cuda", 0, 1 + max_entries * 128, 128
        )
    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    assert len(dc.entries) == max_entries


# --- graph capture + pressure + churn + replay test ---


def test_admission_cache_graph_capture_pressure_replay(_clean_cache) -> None:
    """Capture a geometry with a low cap, fill to capacity, reject +1, churn
    allocator, then replay and verify pointer ownership and exact output.

    Only an integer pointer and weak reference are retained — no strong
    tensor or cache-object owner is held by the test after capture, so an
    eviction or cache replacement would drop the weak referent and cause
    replay to fail."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814)

    # Set a small entry cap so we can fill to capacity quickly.
    original_max = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES
    _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES = 4
    try:
        source, _, _, packed = _make_inputs(4, 128, 64)
        eager = tensor_fp8_linear.mm(source, packed).clone()
        torch.cuda.synchronize()

        # The mm call admitted the (128, padded_in_features) geometry.
        # Read the cache transiently to get ptr + weakref, then drop locals.
        dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
        cached_key = (128, packed.padded_in_features)
        assert cached_key in dc.entries
        cached_tensor = dc.entries[cached_key]
        cached_ptr = cached_tensor.data_ptr()
        cached_weak = weakref.ref(cached_tensor)
        # Drop strong references to the tensor and cache object.
        del cached_tensor, dc

        # Capture a CUDA graph using that geometry.
        tensor_fp8_linear.prewarm(packed, [4])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = tensor_fp8_linear.mm(source, packed)

        # Fill the cache to capacity (3 more entries besides the captured one).
        for i in range(3):
            _tfp8_kernel._cached_unit_scale_mma(
                "cuda", 0, 128 + (i + 1) * 128, packed.padded_in_features
            )
        # Re-fetch cache transiently for count check.
        dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
        assert len(dc.entries) == 4
        del dc

        # The +1 must be rejected, not silently evict the graph-owned entry.
        with pytest.raises(RuntimeError, match="cache.*full"):
            _tfp8_kernel._cached_unit_scale_mma(
                "cuda", 0, 128 + 5 * 128, packed.padded_in_features
            )

        # Churn the CUDA allocator with large allocations.
        for _ in range(32):
            torch.empty(1024 * 1024, dtype=torch.float32, device="cuda")

        # The graph-owned entry must still be present at the same address
        # and the weak reference must still be alive.
        dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
        assert cached_key in dc.entries
        assert dc.entries[cached_key].data_ptr() == cached_ptr
        assert cached_weak() is not None
        del dc

        # Replay must produce the exact same output.
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    finally:
        _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES = original_max


def test_capture_rejects_cold_launch_sharing_prewarmed_scale_key(
    monkeypatch,
    _clean_cache,
) -> None:
    """A canonical scale hit must not admit a distinct GEMM specialization."""
    require_b12x()
    require_mxf8_mma()
    _, _, _, packed = _make_inputs(1, 128, 2048)
    tensor_fp8_linear.prewarm(packed, [1])
    source_64, _, _, _ = _make_inputs(64, 128, 2048)

    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    assert (128, packed.padded_in_features) in dc.entries
    assert len(dc.entries) == 1
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    with pytest.raises(RuntimeError, match="launch geometry.*not prewarmed"):
        tensor_fp8_linear.mm(source_64, packed, expected_m=64)


# --- byte accounting tests ---


def test_admission_cache_bytes_match_unique_storage(_clean_cache) -> None:
    """Byte accounting equals sum of unique untyped_storage bytes, deduped by
    data_ptr."""
    require_b12x()
    require_mxf8_mma()
    _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 7, 128)
    _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 200, 256)
    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]

    # Compute unique bytes by data_ptr to catch aliasing/double-accounting.
    seen_ptrs: set[int] = set()
    unique_bytes = 0
    for t in dc.entries.values():
        ptr = t.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            unique_bytes += int(t.untyped_storage().nbytes())
    assert dc.total_bytes == unique_bytes


def test_admission_cache_bytes_match_cuda_allocated_delta(_clean_cache) -> None:
    """Retained cache bytes correspond to a controlled CUDA memory delta."""
    require_b12x()
    require_mxf8_mma()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated("cuda")
    _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 128, 128)
    _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 256, 128)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated("cuda")
    delta = after - before
    dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    # The CUDA allocated delta should be at least the retained cache bytes
    # (may be more due to allocator rounding, but not less).
    assert delta >= dc.total_bytes


# --- per-device independence test ---


def test_admission_cache_per_device_independence(_clean_cache) -> None:
    """Pressure on device 0 does not affect device 1."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    max_entries = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES
    for i in range(max_entries):
        _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 1 + i * 128, 128)
    b = _tfp8_kernel._cached_unit_scale_mma("cuda", 1, 7, 128)
    assert b.device == torch.device("cuda", 1)
    dc0 = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
    dc1 = _tfp8_kernel._unit_scale_caches[("cuda", 1)]
    assert len(dc0.entries) == max_entries
    assert len(dc1.entries) == 1
    assert dc0 is not dc1


# --- concurrency tests ---


def test_admission_cache_concurrent_same_key_single_construction(
    monkeypatch,
    _clean_cache,
) -> None:
    """Concurrent requests for the same key call _unit_scale_mma exactly once."""
    require_b12x()
    require_mxf8_mma()
    original_fn = _tfp8_kernel._unit_scale_mma
    call_count = 0
    count_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def _counting_fn(*args, **kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(_tfp8_kernel, "_unit_scale_mma", _counting_fn)
    results: list[object] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait()
            t = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 99, 128)
            results.append(t)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors
    assert len(results) == 8
    first = results[0]
    for r in results:
        assert r is first
    assert call_count == 1


def test_admission_cache_concurrent_distinct_streams(_clean_cache) -> None:
    """The published tensor is safe to consume on distinct streams."""
    require_b12x()
    require_mxf8_mma()
    t = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 64, 128)
    s1 = torch.cuda.Stream(device="cuda")
    s2 = torch.cuda.Stream(device="cuda")
    with torch.cuda.stream(s1):
        v1 = t.view(torch.uint8).sum().item()
    with torch.cuda.stream(s2):
        v2 = t.view(torch.uint8).sum().item()
    torch.cuda.synchronize()
    # All values are 127 (UE8M0 1.0).
    expected = 127 * t.numel()
    assert v1 == expected
    assert v2 == expected


# --- stream safety tests ---


def test_mm_explicit_noncurrent_stream_early_drop_churn(_clean_cache) -> None:
    """mm on an explicit non-current stream with padded K and bias: drop
    caller-owned inputs immediately after the async call, churn the
    allocator, then sync the target and verify exact output.

    This catches record_stream omissions: without recording, the caching
    allocator can recycle source/weight/bias storage while the GEMM on the
    target stream is still reading it."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260818)

    source, weight, output_scale, packed = _make_inputs(5, 160, 40)
    bias = torch.randn(40, device="cuda", dtype=torch.bfloat16) * 0.01

    # Compute reference on the default stream before any side-stream launch.
    expected = (source.float() @ weight.float().T) * output_scale + bias
    torch.cuda.synchronize()

    # Prewarm to populate the cache (warm-cache path).
    tensor_fp8_linear.prewarm(packed, [5])
    torch.cuda.synchronize()

    target_stream = torch.cuda.Stream(device="cuda")
    # Call mm while target_stream is NOT the current stream — the default
    # stream remains current so any materialization that is not properly
    # stream-guarded would land on the wrong stream.
    actual = tensor_fp8_linear.mm(
        source, packed, bias=bias, stream=target_stream
    )

    # Immediately drop all caller-owned inputs to force record_stream to
    # protect their storage from allocator recycling.
    del source, weight, output_scale, packed, bias

    # Churn the allocator with many allocations to increase the chance of
    # recycling freed storage while the target stream is still in flight.
    for _ in range(64):
        torch.empty(256 * 1024, dtype=torch.float32, device="cuda")

    # Only now synchronize the target stream and verify output.
    target_stream.synchronize()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_rejects_raw_int_stream(_clean_cache) -> None:
    """Raw integer stream handles are rejected."""
    require_b12x()
    require_mxf8_mma()
    source, _, _, packed = _make_inputs(4, 128, 64)
    with pytest.raises(TypeError, match="stream must be a torch.cuda.Stream"):
        tensor_fp8_linear.mm(source, packed, stream=42)


# --- end-to-end multi-device mm test ---


def test_mm_multi_device_current_device_mismatch(_clean_cache) -> None:
    """mm works correctly when the thread's current device differs from the
    operand device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    require_mxf8_mma()
    torch.manual_seed(20260816)

    # Create operands on device 1 but set current device to 0.
    with torch.cuda.device(1):
        source = (
            torch.randn((7, 128), device="cuda", dtype=torch.bfloat16)
            .clamp(-4, 4)
            .to(torch.float8_e4m3fn)
        )
        weight = (
            torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
            .clamp(-4, 4)
            .to(torch.float8_e4m3fn)
        )
        output_scale = torch.tensor([0.0002], dtype=torch.float32, device="cuda")
        packed = tensor_fp8_linear.pack_weight(weight, output_scale)

    with torch.cuda.device(0):
        # Current device is 0, but operands are on device 1.
        actual = tensor_fp8_linear.mm(source, packed)
        torch.cuda.synchronize()

    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()
    assert actual.device == torch.device("cuda", 1)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_wrong_device_stream_rejected(_clean_cache) -> None:
    """An explicit stream on the wrong device is rejected."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    require_mxf8_mma()
    torch.manual_seed(20260817)

    with torch.cuda.device(0):
        source = (
            torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
            .clamp(-4, 4)
            .to(torch.float8_e4m3fn)
        )
        weight = (
            torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
            .clamp(-4, 4)
            .to(torch.float8_e4m3fn)
        )
        output_scale = torch.tensor([0.0002], dtype=torch.float32, device="cuda")
        packed = tensor_fp8_linear.pack_weight(weight, output_scale)

    wrong_stream = torch.cuda.Stream(device=1)
    with pytest.raises(ValueError, match="stream is on device.*but source"):
        tensor_fp8_linear.mm(source, packed, stream=wrong_stream)


# --- numerical correctness after cache operations ---


def test_admission_cache_numerical_unchanged_after_pressure(_clean_cache) -> None:
    """mm output is numerically correct after cache operations."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260815)

    source, weight, output_scale, packed = _make_inputs(7, 128, 64)
    expected = (source.float() @ weight.float().T) * output_scale

    actual = tensor_fp8_linear.mm(source, packed)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )

    actual2 = tensor_fp8_linear.mm(source, packed)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual2.float(),
        expected.to(actual2.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


# --- prewarm non-atomic prefix test ---


def test_prewarm_rejects_batch_before_partial_admission(_clean_cache) -> None:
    """An over-capacity prewarm leaves the permanent cache unchanged."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260819)

    _, _, _, packed = _make_inputs(4, 128, 64)
    original_max = _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES
    _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES = 2
    try:
        with pytest.raises(RuntimeError, match=r"cache.*full"):
            tensor_fp8_linear.prewarm(packed, [4, 132, 260, 388])

        dc = _tfp8_kernel._unit_scale_caches[("cuda", 0)]
        assert dc.entries == {}
        assert dc.total_bytes == 0
    finally:
        _tfp8_kernel._UNIT_SCALE_CACHE_MAX_ENTRIES = original_max


# --- config parsing tests (no GPU required) ---


def test_parse_cache_limit_defaults() -> None:
    assert _parse_cache_limit(None, 32, 256) == 32


def test_parse_cache_limit_valid() -> None:
    assert _parse_cache_limit("64", 32, 256) == 64


def test_parse_cache_limit_clamps_to_hard_max() -> None:
    assert _parse_cache_limit("99999", 32, 256) == 256


def test_parse_cache_limit_zero_falls_back() -> None:
    assert _parse_cache_limit("0", 32, 256) == 32


def test_parse_cache_limit_negative_falls_back() -> None:
    assert _parse_cache_limit("-5", 32, 256) == 32


def test_parse_cache_limit_non_integer_falls_back() -> None:
    assert _parse_cache_limit("abc", 32, 256) == 32


def test_hard_maxima_are_positive() -> None:
    assert _UNIT_SCALE_HARD_MAX_ENTRIES > 0
    assert _UNIT_SCALE_HARD_MAX_BYTES > 0
