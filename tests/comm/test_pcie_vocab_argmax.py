from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from b12x.comm.pcie import pcie_vocab_argmax as vocab_argmax_module
from b12x.comm.pcie.pcie_hierarchical import (
    _selected_peers as _allreduce_peers,
)
from b12x.comm.pcie.pcie_vocab_argmax import (
    PCIeVocabParallelArgmax,
    _exchange_ipc_handles,
    _selected_peers,
    _wait_nanosleep_cycles_from_env,
)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (1, 2, 3, 4, 8, 12)),
        (1, (0, 2, 3, 5, 9, 13)),
        (6, (2, 4, 5, 7, 10, 14)),
        (15, (3, 7, 11, 12, 13, 14)),
    ],
)
def test_vocab_argmax_uses_bounded_lane_topology(
    rank: int,
    expected: tuple[int, ...],
) -> None:
    assert _selected_peers(rank) == expected


def test_vocab_argmax_and_tp16_allreduce_union_stays_within_peer_limit() -> None:
    for rank in range(16):
        peers = set(_selected_peers(rank)) | set(_allreduce_peers(rank, 16))
        assert len(peers) <= 6


def test_vocab_argmax_peer_graph_is_reciprocal_and_connected() -> None:
    peer_sets = {rank: set(_selected_peers(rank)) for rank in range(16)}
    for rank, peers in peer_sets.items():
        for peer in peers:
            assert rank in peer_sets[peer]

    reached = {0}
    frontier = [0]
    while frontier:
        rank = frontier.pop()
        for peer in peer_sets[rank] - reached:
            reached.add(peer)
            frontier.append(peer)
    assert reached == set(range(16))


def test_vocab_argmax_exchanges_metadata_over_gloo(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
    broadcasts = []

    def fake_broadcast(objects, src, group):
        broadcasts.append((objects, src, group))

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    assert _exchange_ipc_handles(b"handle", group) == [b"handle"]
    assert broadcasts == [([b"handle"], 0, group)]


@pytest.mark.parametrize("rank", [-1, 16])
def test_vocab_argmax_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="requires TP16"):
        _selected_peers(rank)


@pytest.mark.parametrize("world_size", [1, 8, 12, 32])
def test_vocab_argmax_rejects_non_tp16_world(world_size: int) -> None:
    with pytest.raises(ValueError, match="requires TP16"):
        _selected_peers(0, world_size)


@pytest.mark.parametrize("value", ["0", "24", "1024"])
def test_vocab_argmax_wait_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", value)
    assert _wait_nanosleep_cycles_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", "1025", "bad"])
def test_vocab_argmax_rejects_invalid_wait_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", value)
    with pytest.raises(ValueError):
        _wait_nanosleep_cycles_from_env()


def _fake_runtime() -> PCIeVocabParallelArgmax:
    runtime = PCIeVocabParallelArgmax.__new__(PCIeVocabParallelArgmax)
    runtime.device = torch.device("cpu")
    runtime.local_vocab_size = 16
    runtime.max_batch_size = 8
    runtime._closed = False
    runtime._runtime = 123
    runtime._ext = MagicMock()
    runtime._ipc = MagicMock()
    runtime._remote_ptrs = [456]
    runtime._local_ptr = 789
    return runtime


def test_vocab_argmax_resolves_implicit_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = MagicMock()
    extension = MagicMock()
    extension.slab_bytes.return_value = 4096
    extension.init_runtime.return_value = 123
    ipc = MagicMock()
    ipc.cudaMalloc.return_value = 1000
    ipc.cudaIpcGetMemHandleBytes.return_value = b"local"
    ipc.cudaIpcOpenMemHandleBytes.side_effect = range(2000, 2006)

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 16)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(vocab_argmax_module, "CudaRTLibrary", lambda: ipc)
    monkeypatch.setattr(
        vocab_argmax_module,
        "_exchange_ipc_handles",
        lambda local_handle, group: [local_handle] * 16,
    )

    runtime = PCIeVocabParallelArgmax(
        exchange_group=group,
        device="cuda",
        local_vocab_size=16,
        ext_module=extension,
    )

    assert runtime.device == torch.device("cuda:7")
    ipc.cudaSetDevice.assert_called_once_with(7)
    runtime._closed = True


def test_vocab_argmax_destructor_never_enters_collective_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_runtime()
    runtime.group = MagicMock()
    barrier = MagicMock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    with pytest.warns(ResourceWarning, match="without close"):
        runtime.__del__()

    barrier.assert_not_called()
    runtime._ext.destroy_runtime.assert_called_once_with(123)
    runtime._ipc.cudaIpcCloseMemHandle.assert_called_once_with(456)
    runtime._ipc.cudaFree.assert_called_once_with(789)
    assert runtime._closed
    assert runtime._runtime == 0
    assert runtime._remote_ptrs == []
    assert runtime._local_ptr == 0


def test_vocab_argmax_allocates_int64_output_and_dispatches() -> None:
    runtime = _fake_runtime()
    base = torch.randn(4, 16, dtype=torch.bfloat16)
    bias = torch.randn_like(base)

    output = runtime.fused_add_argmax(base, bias)

    assert output.shape == (4,)
    assert output.dtype == torch.int64
    runtime._ext.fused_add_argmax.assert_called_once_with(
        123,
        base,
        bias,
        output,
    )


def test_vocab_argmax_accepts_row_strided_inputs() -> None:
    runtime = _fake_runtime()
    base_storage = torch.randn(4, 3, 16, dtype=torch.bfloat16)
    bias_storage = torch.randn(4, 5, 16, dtype=torch.bfloat16)
    base = base_storage[:, 1]
    bias = bias_storage[:, 3]

    assert not base.is_contiguous()
    assert not bias.is_contiguous()
    output = runtime.fused_add_argmax(base, bias)

    runtime._ext.fused_add_argmax.assert_called_once_with(
        123,
        base,
        bias,
        output,
    )


def test_vocab_argmax_rejects_noncontiguous_last_dimension() -> None:
    base = torch.zeros(1, 32, dtype=torch.bfloat16)[:, ::2]
    bias = torch.zeros_like(base)

    with pytest.raises(ValueError, match="last dimensions"):
        _fake_runtime().fused_add_argmax(base, bias)


@pytest.mark.parametrize(
    ("base", "bias", "error"),
    [
        (
            torch.zeros(1, 16, dtype=torch.float32),
            torch.zeros(1, 16, dtype=torch.float32),
            "BF16",
        ),
        (
            torch.zeros(1, 15, dtype=torch.bfloat16),
            torch.zeros(1, 15, dtype=torch.bfloat16),
            "local vocabulary",
        ),
        (
            torch.zeros(9, 16, dtype=torch.bfloat16),
            torch.zeros(9, 16, dtype=torch.bfloat16),
            "capacity",
        ),
        (
            torch.zeros(1, 16, dtype=torch.bfloat16),
            torch.zeros(2, 16, dtype=torch.bfloat16),
            "matching",
        ),
    ],
)
def test_vocab_argmax_validates_inputs(
    base: torch.Tensor,
    bias: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _fake_runtime().fused_add_argmax(base, bias)
