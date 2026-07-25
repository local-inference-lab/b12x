from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _load_extension,
    owner_stage_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_DCP_TOPK_TEST") != "1",
    reason="set SPARKINFER_RUN_PCIE_DCP_TOPK_TEST=1 to run GPU tests",
)

MAX_ROWS = 64
TOPK = 2048


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    step: int, global_rank: int, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
    indices = (base + global_rank * 1_000_000 + step * 10_000).to(device)
    score_bits = (
        torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
        + 0x3E800000
        + global_rank * 4096
        + step * 64
    )
    return indices, score_bits.to(device).view(torch.float32)


def _dcp_groups(tp_world_size: int, dcp_world_size: int):
    return [
        dist.new_group(
            list(range(start, start + dcp_world_size)),
            backend="nccl",
        )
        for start in range(0, tp_world_size, dcp_world_size)
    ]


def _worker(
    rank: int,
    tp_world_size: int,
    dcp_world_size: int,
    port: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=tp_world_size,
    )
    groups = _dcp_groups(tp_world_size, dcp_world_size)
    dcp_partition = rank // dcp_world_size
    dcp_rank = rank % dcp_world_size
    dcp_group = groups[dcp_partition]
    dcp_global_ranks = list(
        range(
            dcp_partition * dcp_world_size,
            (dcp_partition + 1) * dcp_world_size,
        )
    )

    max_rows = (MAX_ROWS // dcp_world_size) * dcp_world_size
    test_rows = sorted(
        {
            dcp_world_size,
            2 * dcp_world_size,
            4 * dcp_world_size,
            max_rows,
        }
    )
    owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )
    graph_owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )
    try:
        for step, rows in enumerate(test_rows):
            local_indices, local_scores = _inputs(step, rank, rows, device)
            candidate_indices, candidate_scores = owner.stage_candidates(
                local_indices, local_scores
            )
            torch.cuda.synchronize(device)

            rank_inputs = [
                _inputs(step, source, rows, device) for source in dcp_global_ranks
            ]
            expected_indices, expected_scores = owner_stage_reference(
                torch.stack([item[0] for item in rank_inputs]),
                torch.stack([item[1] for item in rank_inputs]),
                dcp_rank,
            )
            torch.testing.assert_close(
                candidate_indices, expected_indices, rtol=0, atol=0
            )
            assert torch.equal(
                candidate_scores.view(torch.int32),
                expected_scores.view(torch.int32),
            )
        graph_rows = max_rows
        graph_indices, graph_scores = _inputs(10, rank, graph_rows, device)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            graph_candidate_indices, graph_candidate_scores = (
                graph_owner.stage_candidates(graph_indices, graph_scores)
            )

        for replay_step in (20, 21):
            next_indices, next_scores = _inputs(replay_step, rank, graph_rows, device)
            graph_indices.copy_(next_indices)
            graph_scores.copy_(next_scores)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)

            expected_rank_inputs = [
                _inputs(replay_step, source, graph_rows, device)
                for source in dcp_global_ranks
            ]
            expected_indices, expected_scores = owner_stage_reference(
                torch.stack([item[0] for item in expected_rank_inputs]),
                torch.stack([item[1] for item in expected_rank_inputs]),
                dcp_rank,
            )
            torch.testing.assert_close(
                graph_candidate_indices, expected_indices, rtol=0, atol=0
            )
            assert torch.equal(
                graph_candidate_scores.view(torch.int32),
                expected_scores.view(torch.int32),
            )
        dist.barrier()
        torch.cuda.synchronize(device)
    finally:
        graph_owner.close_coordinated()
        owner.close_coordinated()
        dist.destroy_process_group()


def test_pcie_dcp_topk_exact_owner_exchange():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp_world_size = int(os.getenv("SPARKINFER_PCIE_DCP_TOPK_TP", "8"))
    dcp_world_size = int(os.getenv("SPARKINFER_PCIE_DCP_TOPK_DCP", "4"))
    if (
        tp_world_size not in (2, 3, 4, 6, 8)
        or dcp_world_size not in (2, 3, 4, 6, 8)
        or tp_world_size % dcp_world_size != 0
    ):
        pytest.skip("unsupported TP/DCP top-k geometry")
    if torch.cuda.device_count() < tp_world_size:
        pytest.skip(f"need {tp_world_size} CUDA devices")
    _load_extension()
    mp.spawn(
        _worker,
        args=(tp_world_size, dcp_world_size, _free_port()),
        nprocs=tp_world_size,
        join=True,
    )
