"""Dense GEMM explicit-stream ordering and allocator-lifetime contracts."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import b12x._lib.dense_gemm as dense_module
import b12x._lib.utils as utils_module
from b12x._lib.utils import cuda_stream_scope
from tests._reference.helpers import require_b12x


class _FakeStream:
    def __init__(self, handle: int, device: torch.device) -> None:
        self.cuda_stream = handle
        self.device = device
        self.waited_for: list[_FakeStream] = []

    def wait_stream(self, stream: _FakeStream) -> None:
        self.waited_for.append(stream)


class _FakeStorage:
    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


class _FakeTensor:
    def __init__(self, ptr: int, device: torch.device) -> None:
        self._storage = _FakeStorage(ptr)
        self.device = device
        self.recorded_on: list[_FakeStream] = []

    def numel(self) -> int:
        return 1

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def record_stream(self, stream: _FakeStream) -> None:
        self.recorded_on.append(stream)


def test_explicit_side_stream_bridges_once_and_deduplicates_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", 0)
    caller = _FakeStream(11, device)
    launch = _FakeStream(22, device)
    first = _FakeTensor(100, device)
    alias = _FakeTensor(100, device)
    second = _FakeTensor(200, device)
    entered: list[_FakeStream] = []

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: caller)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _null_context())
    monkeypatch.setattr(
        utils_module.cudart,
        "cudaStreamGetDevice",
        lambda _stream: (utils_module.cudart.cudaError_t.cudaSuccess, 0),
    )
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        lambda handle, device: launch if handle == 22 else None,
    )

    @contextmanager
    def use_stream(stream):
        entered.append(stream)
        yield

    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    with cuda_stream_scope(22, device, [first, alias, second]) as resolved:
        assert resolved is launch
        assert caller.waited_for == []

    assert entered == [launch]
    assert launch.waited_for == [caller]
    assert caller.waited_for == [launch]
    assert first.recorded_on == [launch]
    assert alias.recorded_on == []
    assert second.recorded_on == [launch]


def test_current_explicit_stream_avoids_recording_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", 0)
    caller = _FakeStream(11, device)
    tensor = _FakeTensor(100, device)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: caller)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _null_context())
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda _stream: pytest.fail("same-stream path entered a stream context"),
    )

    with cuda_stream_scope(11, device, [tensor]) as resolved:
        assert resolved is caller

    assert tensor.recorded_on == []
    assert caller.waited_for == []


def test_side_stream_restores_consumer_dependency_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", 0)
    caller = _FakeStream(11, device)
    launch = _FakeStream(22, device)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: caller)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _null_context())
    monkeypatch.setattr(
        utils_module.cudart,
        "cudaStreamGetDevice",
        lambda _stream: (utils_module.cudart.cudaError_t.cudaSuccess, 0),
    )
    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda *_args, **_kwargs: launch)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: _null_context())

    with pytest.raises(RuntimeError, match="launch failed"):
        with cuda_stream_scope(22, device, []):
            raise RuntimeError("launch failed")

    assert launch.waited_for == [caller]
    assert caller.waited_for == [launch]


def test_raw_stream_from_another_device_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", 0)
    caller = _FakeStream(11, device)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: caller)
    monkeypatch.setattr(
        utils_module.cudart,
        "cudaStreamGetDevice",
        lambda _stream: (utils_module.cudart.cudaError_t.cudaSuccess, 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        lambda *_args, **_kwargs: pytest.fail("wrong-device stream was wrapped"),
    )

    with pytest.raises(ValueError, match="cuda:1.*cuda:0"):
        with cuda_stream_scope(22, device, []):
            pass


@contextmanager
def _null_context():
    yield


def _dense_kwargs() -> dict[str, object]:
    return {
        "ab_dtype": "float8_e4m3fn",
        "sf_dtype": "float8_e8m0fnu",
        "c_dtype": "bfloat16",
        "sf_vec_size": 32,
    }


def test_none_stream_bypasses_stream_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = torch.empty(1)
    sfa = torch.empty(1)
    b = torch.empty(1)
    sfb = torch.empty(1)

    monkeypatch.setattr(
        dense_module,
        "cuda_stream_scope",
        lambda *_args, **_kwargs: pytest.fail("stream=None resolved CUDA state"),
    )

    with pytest.raises(ValueError, match="load_path"):
        dense_module.dense_gemm(
            (a, sfa),
            (b, sfb),
            load_path="invalid",
            **_dense_kwargs(),
        )


def test_explicit_stream_scopes_complete_dense_operation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = torch.empty(1)
    sfa = torch.empty(1)
    b = torch.empty(1)
    sfb = torch.empty(1)
    out = torch.empty(1)
    alpha = torch.empty(1)
    scopes: list[tuple[object, torch.device, list[torch.Tensor | None]]] = []
    entered: list[bool] = []

    @contextmanager
    def scope(stream, device, tensors):
        scopes.append((stream, device, tensors))
        entered.append(True)
        try:
            yield
        finally:
            entered.append(False)

    monkeypatch.setattr(dense_module, "cuda_stream_scope", scope)

    with pytest.raises(ValueError, match="load_path"):
        dense_module.dense_gemm(
            (a, sfa),
            (b, sfb),
            out=out,
            alpha=alpha,
            stream=123,
            load_path="invalid",
            **_dense_kwargs(),
        )

    assert len(scopes) == 1
    assert scopes[0][0] == 123
    assert scopes[0][1] == a.device
    assert all(
        got is expected
        for got, expected in zip(
            scopes[0][2][:6], [a, sfa, b, sfb, out, alpha], strict=True
        )
    )
    assert entered == [True, False]


def _run_block_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    out: torch.Tensor,
    stream: object = None,
) -> torch.Tensor:
    m, k = map(int, a.shape)
    n = int(b.shape[0])
    return dense_module.dense_gemm(
        (a.view(m, k, 1), a_scale),
        (b.view(n, k, 1), b_scale),
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        expected_m=m,
        block_fp8=True,
        stream=stream,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_side_stream_orders_producer_launch_and_consumer() -> None:
    require_b12x()
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    torch.manual_seed(153)
    device = torch.device("cuda", torch.cuda.current_device())
    m, n, k = 1, 128, 256
    a_source = (torch.randn((m, k), device=device) / 8).to(torch.float8_e4m3fn)
    b_source = (torch.randn((n, k), device=device) / 8).to(torch.float8_e4m3fn)
    a_scale = torch.ones((m, k // 128), device=device)
    b_scale = torch.ones((n // 128, k // 128), device=device)
    expected = torch.empty((m, n, 1), dtype=torch.bfloat16, device=device)
    _run_block_fp8(a_source, b_source, a_scale, b_scale, out=expected)
    torch.cuda.synchronize(device)

    a = torch.zeros_like(a_source)
    b = torch.zeros_like(b_source)
    out = torch.empty_like(expected)
    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        torch.cuda._sleep(5_000_000)
    torch.cuda._sleep(5_000_000)
    a.copy_(a_source)
    b.copy_(b_source)

    _run_block_fp8(a, b, a_scale, b_scale, out=out, stream=side)
    observed = out.clone()
    del a, b, out
    torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device).zero_()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_capture_stream_replays_dense_gemm() -> None:
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    m, n, k = 1, 128, 256
    a = (torch.randn((m, k), device=device) / 8).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device=device) / 8).to(torch.float8_e4m3fn)
    a_scale = torch.ones((m, k // 128), device=device)
    b_scale = torch.ones((n // 128, k // 128), device=device)
    expected = torch.empty((m, n, 1), dtype=torch.bfloat16, device=device)
    actual = torch.empty_like(expected)
    _run_block_fp8(a, b, a_scale, b_scale, out=expected)
    _run_block_fp8(a, b, a_scale, b_scale, out=actual)
    torch.cuda.synchronize(device)

    capture_stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _run_block_fp8(
            a,
            b,
            a_scale,
            b_scale,
            out=actual,
            stream=capture_stream,
        )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
