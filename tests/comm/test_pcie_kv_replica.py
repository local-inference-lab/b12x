from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie.pcie_kv_replica import PCIePagedKvReplica


def _runtime_and_tensors():
    runtime = PCIePagedKvReplica.__new__(PCIePagedKvReplica)
    runtime.device = torch.device("cpu")
    runtime.rank, runtime.world_size = 0, 4
    runtime.max_requests, runtime.max_tokens = 2, 1536
    runtime._closed, runtime._serial_stream = False, None
    state = SimpleNamespace(
        query=SimpleNamespace(
            call={"page_size": 128, "max_tokens": 1536, "stripe": 64}
        ),
        launcher=lambda: "launcher",
    )
    tensors = {
        "cache": torch.empty((4, 128 * 288), dtype=torch.uint8),
        "table": torch.empty((2, 3), dtype=torch.int32),
        "positions": torch.empty(2, dtype=torch.int64),
        "starts": torch.empty(3, dtype=torch.int32),
        "out": torch.empty((2 * 12 + 1, 128 * 288), dtype=torch.uint8),
    }
    return runtime, state, tensors


def test_paged_kv_replica_validates_fixed_abi_when_binding():
    runtime, state, tensors = _runtime_and_tensors()
    binding = runtime._bind_prepared(state, **tensors)

    assert binding.request_capacity == 2
    assert binding.output_pages_per_request == 12
    assert binding.runtime is runtime
    assert binding.state is state
    assert binding.cache is tensors["cache"]

    tensors["positions"] = tensors["positions"].to(torch.int32)
    with pytest.raises(ValueError, match="metadata must describe"):
        runtime._bind_prepared(state, **tensors)


def test_paged_kv_replica_live_launch_uses_retained_binding(monkeypatch):
    runtime, state, tensors = _runtime_and_tensors()
    binding = runtime._bind_prepared(state, **tensors)
    launch = []
    monkeypatch.setattr(runtime, "_bind_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        "b12x.comm.pcie._kv_replica_cute.run_kv_replica",
        lambda *args, **kwargs: launch.append((args, kwargs)),
    )

    runtime._run_prepared(state, binding, requests=2, max_tokens=1024)

    args, kwargs = launch.pop()
    assert args == (
        "launcher",
        runtime,
        tensors["cache"],
        tensors["table"],
        tensors["positions"],
        tensors["starts"],
        tensors["out"],
    )
    assert kwargs == {
        "requests": 2,
        "max_tokens": 1024,
        "output_pages_per_request": 12,
    }

    with pytest.raises(ValueError, match="exceed prepared capacity"):
        runtime._run_prepared(state, binding, requests=3, max_tokens=1024)

    with pytest.raises(ValueError, match="different prepared runtime"):
        runtime._run_prepared(object(), binding, requests=2, max_tokens=1024)

    runtime._closed = True
    with pytest.raises(RuntimeError, match="channel is closed"):
        runtime._run_prepared(state, binding, requests=2, max_tokens=1024)
