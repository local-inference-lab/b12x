from __future__ import annotations

import threading
from dataclasses import replace

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm import tensor_fp8_linear
from b12x.gemm.tensor_fp8_linear import _kernel as _tfp8_kernel
from b12x.gemm.tensor_fp8_linear import api as tensor_fp8_api

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


@pytest.fixture()
def clean_unit_scale_cache():
    _tfp8_kernel._reset_unit_scale_cache()
    yield
    _tfp8_kernel._reset_unit_scale_cache()


def _install_fake_unit_scale_builder(monkeypatch):
    built: list[tuple[int, int, torch.device]] = []

    def build(rows: int, width: int, device: torch.device):
        built.append((rows, width, device))
        return object()

    monkeypatch.setattr(_tfp8_kernel, "_unit_scale_mma", build)
    return built


def test_unit_scale_storage_size_uses_physical_tiles() -> None:
    size = _tfp8_kernel._unit_scale_packed_bytes
    assert size(1, 128) == 512
    assert size(128, 128) == 512
    assert size(129, 128) == 1024
    assert size(128, 512) == 2048


def test_unit_scale_cache_canonicalizes_rows_and_has_lock_free_hits(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    built = _install_fake_unit_scale_builder(monkeypatch)
    first = _tfp8_kernel._cached_unit_scale_mma("cpu", None, 1, 128)

    class FailLock:
        def __enter__(self):
            raise AssertionError("cache hit acquired the cold-path lock")

        def __exit__(self, *args):
            return False

    original_lock = _tfp8_kernel._unit_scale_cache_lock
    _tfp8_kernel._unit_scale_cache_lock = FailLock()
    try:
        assert _tfp8_kernel._cached_unit_scale_mma("cpu", None, 127, 128) is first
    finally:
        _tfp8_kernel._unit_scale_cache_lock = original_lock

    assert built == [(128, 128, torch.device("cpu"))]


def test_unit_scale_cache_rejects_entry_pressure_before_allocation(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    built = _install_fake_unit_scale_builder(monkeypatch)
    monkeypatch.setattr(_tfp8_kernel, "_UNIT_SCALE_CACHE_MAX_ENTRIES", 2)

    _tfp8_kernel._cached_unit_scale_mma("cpu", None, 1, 128)
    _tfp8_kernel._cached_unit_scale_mma("cpu", None, 129, 128)
    with pytest.raises(RuntimeError, match="cache.*full"):
        _tfp8_kernel._cached_unit_scale_mma("cpu", None, 257, 128)

    assert len(built) == 2


def test_unit_scale_cache_rejects_byte_pressure_before_allocation(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    built = _install_fake_unit_scale_builder(monkeypatch)
    monkeypatch.setattr(_tfp8_kernel, "_UNIT_SCALE_CACHE_MAX_BYTES", 511)

    with pytest.raises(RuntimeError, match="would exceed.*byte budget"):
        _tfp8_kernel._cached_unit_scale_mma("cpu", None, 1, 128)

    assert built == []


def test_unit_scale_cache_checks_capture_only_on_miss(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    built = _install_fake_unit_scale_builder(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first = _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 1, 128)

    def fail_capture_query():
        raise AssertionError("cache hit queried capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", fail_capture_query)
    assert _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 64, 128) is first

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not prewarmed"):
        _tfp8_kernel._cached_unit_scale_mma("cuda", 0, 129, 128)
    assert len(built) == 1


def test_unit_scale_cache_constructs_once_under_concurrency(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(8)

    def build(rows: int, width: int, device: torch.device):
        nonlocal calls
        with calls_lock:
            calls += 1
        return object()

    monkeypatch.setattr(_tfp8_kernel, "_unit_scale_mma", build)
    results: list[object] = []

    def worker() -> None:
        start.wait()
        results.append(
            _tfp8_kernel._cached_unit_scale_mma("cpu", None, 64, 128)
        )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 8
    assert all(result is results[0] for result in results)


def test_unit_scale_cache_never_evicts_admitted_storage(
    monkeypatch,
    clean_unit_scale_cache,
) -> None:
    _install_fake_unit_scale_builder(monkeypatch)
    monkeypatch.setattr(_tfp8_kernel, "_UNIT_SCALE_CACHE_MAX_ENTRIES", 2)

    first = _tfp8_kernel._cached_unit_scale_mma("cpu", None, 1, 128)
    _tfp8_kernel._cached_unit_scale_mma("cpu", None, 129, 128)
    with pytest.raises(RuntimeError, match="cache.*full"):
        _tfp8_kernel._cached_unit_scale_mma("cpu", None, 257, 128)

    assert _tfp8_kernel._cached_unit_scale_mma("cpu", None, 1, 128) is first
