from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import sparkinfer.comm.pcie.pcie_allreduce as pcie_allreduce
from sparkinfer.comm.pcie.pcie_allreduce import (
    PCIeAllReduce,
    _algorithm_for_world_size,
)
from sparkinfer.comm.pcie.pcie_hierarchical import (
    _pick_blocks,
    _selected_peers,
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
