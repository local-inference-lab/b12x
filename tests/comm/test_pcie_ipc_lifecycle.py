from __future__ import annotations

import torch

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
from sparkinfer.comm.pcie.pcie_twoshot import PCIeTwoShotSP


class _FakeChannel:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def _close_ipc_imports(self) -> None:
        self.events.append(f"imports:{self.name}")

    def _free_ipc_exports(self) -> None:
        self.events.append(f"exports:{self.name}")


def test_pool_explicit_close_coordinates_imports_before_exports(monkeypatch) -> None:
    events = []
    group = object()
    channel_a = _FakeChannel("a", events)
    channel_b = _FakeChannel("b", events)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        exchange_group=group,
        channel_factory=lambda stream_key: channel_a,
    )
    pool._all_channels = [channel_a, channel_b]
    pool._channels = {1: channel_a, 2: channel_b}
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )

    pool.close()
    pool.close()

    assert events == [
        "synchronize",
        "barrier",
        "imports:a",
        "imports:b",
        "barrier",
        "exports:a",
        "exports:b",
        "barrier",
    ]
    assert pool._closed
    assert pool._all_channels == []
    assert pool._channels == {}


def test_pool_destructor_is_nonblocking_and_does_not_free_exports(
    monkeypatch,
) -> None:
    events = []
    channel = _FakeChannel("a", events)
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        exchange_group=object(),
        channel_factory=lambda stream_key: channel,
    )
    pool._all_channels = [channel]
    pool._channels = {1: channel}
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    pool.__del__()

    assert events == ["imports:a"]


class _FakeExt:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def dispose(self, ptr: int) -> None:
        self.events.append(("dispose", ptr))


class _FakeIPC:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def cudaIpcCloseMemHandle(self, ptr: int) -> None:
        self.events.append(("close", ptr))

    def cudaFree(self, ptr: int) -> None:
        self.events.append(("free", ptr))


class _Shared:
    def __init__(self, local_ptr: int, remote_ptrs: tuple[int, ...]) -> None:
        self.local_ptr = local_ptr
        self.remote_ptrs = remote_ptrs


def _fake_twoshot(events: list[object]) -> PCIeTwoShotSP:
    runtime = object.__new__(PCIeTwoShotSP)
    runtime.rank = 0
    runtime.world_size = 2
    runtime.device = torch.device("cpu")
    runtime.exchange_group = object()
    runtime._ext = _FakeExt(events)
    runtime._fptr = 123
    runtime._owned_buffers = [_Shared(1000, (2000, 3000))]
    runtime._ipc = _FakeIPC(events)
    runtime.max_rows = 8
    runtime.row_elems = 16
    runtime._closed = False
    runtime._ipc_imports_closed = False
    runtime._ipc_exports_freed = False
    return runtime


def test_twoshot_explicit_close_coordinates_unmap_then_free(monkeypatch) -> None:
    events: list[object] = []
    runtime = _fake_twoshot(events)
    runtime.device = torch.device("cuda:0")
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.torch.cuda.synchronize",
        lambda device: events.append("synchronize"),
    )

    runtime.close()
    runtime.close()

    assert events == [
        "synchronize",
        "barrier",
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
        "barrier",
        ("free", 1000),
        "barrier",
    ]
    assert runtime._fptr == 0
    assert runtime._owned_buffers == []
    assert runtime._ipc_imports_closed
    assert runtime._ipc_exports_freed


def test_twoshot_destructor_unmaps_without_barrier_or_export_free(
    monkeypatch,
) -> None:
    events: list[object] = []
    runtime = _fake_twoshot(events)
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    runtime.__del__()

    assert events == [
        ("dispose", 123),
        ("close", 2000),
        ("close", 3000),
    ]
    assert runtime._owned_buffers != []
    assert not runtime._ipc_exports_freed
