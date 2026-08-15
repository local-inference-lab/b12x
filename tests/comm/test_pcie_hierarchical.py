from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import b12x.comm.pcie.pcie_allreduce as pcie_allreduce
from b12x.comm.pcie import _hierarchical_cute
from b12x.comm.pcie.pcie_allreduce import (
    PCIeAllReduce,
    _algorithm_for_world_size,
)
from b12x.comm.pcie.pcie_hierarchical import (
    _buffer_modes_from_env,
    _make_layout,
    _pick_blocks,
    _selected_peers,
    _threads_from_env,
    _vectorized_bf16x2_from_env,
    _vectorized_bf16x2_max_elements_from_env,
    _wait_nanosleep_cycles_from_env,
)


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_allreduce_uses_direct_path_for_peer_safe_worlds(world_size: int) -> None:
    assert _algorithm_for_world_size(world_size) == "oneshot"


@pytest.mark.parametrize("world_size", [12, 16])
def test_allreduce_uses_bounded_degree_path_for_large_worlds(
    world_size: int,
) -> None:
    assert _algorithm_for_world_size(world_size) == "hierarchical"


def test_allreduce_factory_builds_tp16_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        rank=0,
        world_size=16,
        device=torch.device("cuda:0"),
    )
    hierarchy_factory = MagicMock(return_value=runtime)
    exchange_group = object()
    monkeypatch.setattr(pcie_allreduce.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(
        pcie_allreduce,
        "PCIeHierarchicalAllReduce",
        hierarchy_factory,
    )

    allreduce = PCIeAllReduce.from_exchange_group(
        exchange_group=exchange_group,  # type: ignore[arg-type]
        device="cuda:0",
        max_size=16 * 1024,
    )

    assert allreduce.algorithm == "hierarchical"
    hierarchy_factory.assert_called_once_with(
        exchange_group=exchange_group,
        device="cuda:0",
        max_elements=8 * 1024,
        ext_module=None,
    )


def test_allreduce_factory_forwards_oneshot_channel_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        rank=0,
        world_size=8,
        device=torch.device("cuda:0"),
    )
    oneshot_factory = MagicMock(return_value=runtime)
    exchange_group = object()
    monkeypatch.setattr(pcie_allreduce.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(
        pcie_allreduce.PCIeOneshotAllReducePool,
        "from_exchange_group",
        oneshot_factory,
    )

    allreduce = PCIeAllReduce.from_exchange_group(
        exchange_group=exchange_group,  # type: ignore[arg-type]
        device="cuda:0",
        max_concurrent_channels=3,
    )

    assert allreduce.algorithm == "oneshot"
    oneshot_factory.assert_called_once_with(
        exchange_group=exchange_group,
        device="cuda:0",
        eager_buffer_bytes=8388608,
        max_size=8388608,
        rank_data_bytes=8388608,
        ext_module=None,
        single_channel=False,
        max_concurrent_channels=3,
    )


def test_allreduce_oneshot_forwards_named_channel_contract() -> None:
    runtime = MagicMock()
    runtime.rank = 0
    runtime.world_size = 8
    runtime.device = torch.device("cuda:0")
    allreduce = PCIeAllReduce(runtime, "oneshot")
    stream = object()
    inp = torch.randn(2, 4)
    out = torch.empty_like(inp)

    allreduce.prepare_channels(("target", "draft"))
    allreduce.for_stream(stream, channel_id="target")
    allreduce.all_reduce(inp, out=out, stream=stream, channel_id="target")
    with allreduce.capture(stream, channel_id="target"):
        pass

    runtime.prepare_channels.assert_called_once_with(("target", "draft"))
    runtime.for_stream.assert_called_once_with(stream, channel_id="target")
    runtime.all_reduce.assert_called_once_with(
        inp,
        out=out,
        peer_input_ptrs=None,
        stream=stream,
        channel_id="target",
    )
    runtime.capture.assert_called_once_with(stream=stream, channel_id="target")


def test_allreduce_hierarchy_accepts_named_channel_contract() -> None:
    runtime = MagicMock(
        spec=["rank", "world_size", "device", "for_stream", "all_reduce", "capture"]
    )
    runtime.rank = 0
    runtime.world_size = 16
    runtime.device = torch.device("cuda:0")
    allreduce = PCIeAllReduce(runtime, "hierarchical")
    stream = object()
    inp = torch.randn(2, 4)
    out = torch.empty_like(inp)

    allreduce.prepare_channels(("target", "draft"))
    allreduce.for_stream(stream, channel_id="target")
    allreduce.all_reduce(inp, out=out, stream=stream, channel_id="target")
    with allreduce.capture(stream, channel_id="target"):
        pass

    runtime.for_stream.assert_called_once_with(stream)
    runtime.all_reduce.assert_called_once_with(
        inp,
        out=out,
        blocks=None,
        stream=stream,
    )
    runtime.capture.assert_called_once_with(stream=stream)


@pytest.mark.parametrize("world_size", [1, 3, 10, 14, 20])
def test_allreduce_rejects_unsupported_or_peer_unsafe_worlds(
    world_size: int,
) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _algorithm_for_world_size(world_size)


@pytest.mark.parametrize(
    ("world_size", "rank", "expected"),
    [
        (12, 0, (1, 2, 3, 4, 8)),
        (12, 1, (0,)),
        (12, 3, (0,)),
        (12, 4, (0, 5, 6, 7, 8)),
        (12, 7, (4,)),
        (12, 8, (0, 4, 9, 10, 11)),
        (12, 11, (8,)),
        (16, 0, (1, 2, 3, 4, 8, 12)),
        (16, 5, (4,)),
        (16, 12, (0, 4, 8, 13, 14, 15)),
        (16, 15, (12,)),
    ],
)
def test_selected_peers_respect_four_gpu_islands(
    world_size: int,
    rank: int,
    expected: tuple[int, ...],
) -> None:
    assert _selected_peers(rank, world_size) == expected


def test_selected_peers_never_exceed_cuda_peer_limit() -> None:
    expected_maximums = {12: 5, 16: 6}
    for world_size, expected in expected_maximums.items():
        mapped_peers = [
            len(_selected_peers(rank, world_size)) for rank in range(world_size)
        ]
        assert max(mapped_peers) == expected
        assert max(mapped_peers) <= 8


@pytest.mark.parametrize("world_size", [12, 16])
def test_selected_peer_graph_is_reciprocal_and_connected(world_size: int) -> None:
    peer_sets = {
        rank: set(_selected_peers(rank, world_size)) for rank in range(world_size)
    }
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
    assert reached == set(range(world_size))


@pytest.mark.parametrize("world_size", [12, 16])
def test_hierarchical_protocol_simulation_matches_full_sum(world_size: int) -> None:
    generator = torch.Generator().manual_seed(1234 + world_size)
    inputs = torch.randn(world_size, 257, generator=generator)
    peer_sets = {
        rank: set(_selected_peers(rank, world_size)) for rank in range(world_size)
    }
    leaders = tuple(range(0, world_size, 4))

    partials = {}
    for leader in leaders:
        local_ranks = set(range(leader, leader + 4))
        assert local_ranks - {leader} <= peer_sets[leader]
        partials[leader] = inputs[leader : leader + 4].sum(dim=0)

    outputs = []
    for rank in range(world_size):
        leader = rank // 4 * 4
        if rank != leader:
            assert leader in peer_sets[rank]
        assert set(leaders) - {leader} <= peer_sets[leader]
        outputs.append(torch.stack(tuple(partials.values())).sum(dim=0))

    expected = inputs.sum(dim=0).expand(world_size, -1)
    torch.testing.assert_close(torch.stack(outputs), expected)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 12), (12, 12), (0, 8), (0, 20)],
)
def test_selected_peers_rejects_invalid_topologies(
    rank: int,
    world_size: int,
) -> None:
    with pytest.raises(ValueError):
        _selected_peers(rank, world_size)


@pytest.mark.parametrize(
    ("elements", "expected"),
    [
        (1, 16),
        (3584, 16),
        (4096, 16),
        (4097, 32),
        (7168, 32),
    ],
)
def test_pick_blocks_for_k3_decode(elements: int, expected: int) -> None:
    assert _pick_blocks(elements) == expected


def test_pick_blocks_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        _pick_blocks(0)


@pytest.mark.parametrize(
    ("double_buffered", "deferred_consumption", "expected"),
    [
        ("0", "0", (False, False)),
        ("1", "0", (True, False)),
        ("0", "1", (False, True)),
    ],
)
def test_hierarchical_buffer_modes_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    double_buffered: str,
    deferred_consumption: str,
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER",
        double_buffered,
    )
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        deferred_consumption,
    )
    assert _buffer_modes_from_env() == expected


def test_hierarchical_buffer_modes_reject_ambiguous_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "1")
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        "1",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _buffer_modes_from_env()


@pytest.mark.parametrize("value", ["0", "16", "32", "64", "1024"])
def test_hierarchical_wait_nanosleep_cycles(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
        value,
    )
    assert _wait_nanosleep_cycles_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", "1025", "not-an-int"])
def test_hierarchical_wait_nanosleep_cycles_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
        value,
    )
    with pytest.raises(ValueError):
        _wait_nanosleep_cycles_from_env()


@pytest.mark.parametrize("value", ["32", "128", "224", "256", "1024"])
def test_hierarchical_threads(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_THREADS", value)
    assert _threads_from_env() == int(value)


@pytest.mark.parametrize("value", ["0", "31", "225", "1056", "bad"])
def test_hierarchical_threads_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_THREADS", value)
    with pytest.raises(ValueError):
        _threads_from_env()


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_hierarchical_vectorized_bf16x2(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2", value)
    assert _vectorized_bf16x2_from_env() is expected


def test_hierarchical_vectorized_bf16x2_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_PCIE_HIERARCHICAL_BF16X2", "true")
    with pytest.raises(ValueError):
        _vectorized_bf16x2_from_env()


@pytest.mark.parametrize("value", ["0", "7168", "57344", str(1 << 30)])
def test_hierarchical_vectorized_bf16x2_max_elements(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
        value,
    )
    assert _vectorized_bf16x2_max_elements_from_env() == int(value)


@pytest.mark.parametrize("value", ["-1", str((1 << 30) + 1), "bad"])
def test_hierarchical_vectorized_bf16x2_max_elements_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
        value,
    )
    with pytest.raises(ValueError):
        _vectorized_bf16x2_max_elements_from_env()


@pytest.mark.parametrize("max_elements", [1, 3584, 4096, 7168])
def test_hierarchical_slab_layout_matches_native_alignment(
    max_elements: int,
) -> None:
    layout = _make_layout(max_elements)
    assert layout.stage[0] == 69_888
    for slot in range(2):
        assert (
            layout.partial[slot]
            == ((layout.stage[slot] + max_elements * 2 + 255) // 256) * 256
        )
        assert (
            layout.final[slot]
            == ((layout.partial[slot] + max_elements * 4 + 255) // 256) * 256
        )
        if slot == 0:
            assert (
                layout.stage[1]
                == ((layout.final[0] + max_elements * 2 + 255) // 256) * 256
            )
    assert layout.bytes == ((layout.final[1] + max_elements * 2 + 255) // 256) * 256
    assert all(
        value % 256 == 0
        for offsets in (layout.stage, layout.partial, layout.final)
        for value in offsets
    )
    assert layout.bytes % 256 == 0


def test_hierarchical_protocol_intrinsics_pin_native_ptx_modifiers() -> None:
    assert "ld.acquire.sys.global.u32" in inspect.getsource(
        _hierarchical_cute._load_acquire_sys_u32
    )
    assert "st.release.sys.global.u32" in inspect.getsource(
        _hierarchical_cute._store_release_sys_u32
    )
    assert "fence.sc.sys" in inspect.getsource(_hierarchical_cute._fence_sc_sys)
    assert "nanosleep.u32 $0" in inspect.getsource(_hierarchical_cute._nanosleep)


@pytest.mark.parametrize(
    ("world_size", "rank", "island", "local_rank", "leader_rank"),
    [
        (12, 0, 0, 0, 0),
        (12, 5, 1, 1, 4),
        (12, 11, 2, 3, 8),
        (16, 15, 3, 3, 12),
    ],
)
def test_hierarchical_launch_specializes_rank_topology(
    world_size: int,
    rank: int,
    island: int,
    local_rank: int,
    leader_rank: int,
) -> None:
    launch = _hierarchical_cute._HierarchicalLaunch(world_size, rank)

    assert launch._rank == rank
    assert launch._island == island
    assert launch._local_rank == local_rank
    assert launch._leader_rank == leader_rank


def test_hierarchical_launch_uses_direct_slab_pointer_parameters() -> None:
    parameters = inspect.signature(
        _hierarchical_cute._HierarchicalLaunch.kernel
    ).parameters

    assert tuple(parameters)[1:17] == tuple(f"slab{rank}" for rank in range(16))
    assert "rank" not in parameters
    assert "slabs" not in parameters


@pytest.mark.parametrize(("world_size", "rank"), [(8, 0), (12, -1), (12, 12)])
def test_hierarchical_launch_rejects_invalid_specialization(
    world_size: int, rank: int
) -> None:
    with pytest.raises(ValueError):
        _hierarchical_cute._HierarchicalLaunch(world_size, rank)
