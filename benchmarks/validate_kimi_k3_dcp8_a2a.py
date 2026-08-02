#!/usr/bin/env python3
"""Validate SparkInfer A2A on the exact Kimi K3 TP16/DCP8 MLA geometry."""

from __future__ import annotations

import argparse
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2APool,
    _load_extension,
    lse_reduce_scatter_reference,
)

TOTAL_HEADS = 48  # K3: 96 heads / TP16 * DCP8.
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    step: int,
    rank: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1000 * step + rank)
    output = torch.randn(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    lse = torch.randn(
        batch,
        TOTAL_HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    query = torch.randn(
        batch,
        TOTAL_HEADS // dist.get_world_size(),
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    return output, lse, query


def _reference(
    output: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    outputs = [torch.empty_like(output) for _ in range(world_size)]
    lses = [torch.empty_like(lse) for _ in range(world_size)]
    queries = [torch.empty_like(query) for _ in range(world_size)]
    dist.all_gather(outputs, output)
    dist.all_gather(lses, lse)
    dist.all_gather(queries, query)
    reduced = lse_reduce_scatter_reference(
        torch.stack(outputs),
        torch.stack(lses),
        rank,
    )
    return reduced, torch.cat(queries, dim=1)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
    )
    try:
        for step, batch in enumerate((1, 2, 8, 64), start=1):
            output, lse, query = _inputs(step, rank, batch, device)
            expected_output, expected_query = _reference(output, lse, query, rank)
            actual_query = pool.all_gather_heads(query)
            actual_output = pool.lse_reduce_scatter(output, lse)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
            torch.testing.assert_close(
                actual_output.float(),
                expected_output.float(),
                rtol=3e-2,
                atol=3e-2,
            )

        stream = torch.cuda.Stream(device=device)
        channel = pool.for_stream(stream)
        static_output, static_lse, static_query = _inputs(100, rank, 1, device)
        graph_output = torch.empty(
            1,
            TOTAL_HEADS // world_size,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        graph_query = torch.empty(
            1,
            TOTAL_HEADS,
            QUERY_HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.cuda.stream(stream):
            channel.all_gather_heads(static_query, graph_query)
            channel.lse_reduce_scatter(static_output, static_lse, graph_output)
        stream.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            channel.all_gather_heads(static_query, graph_query)
            channel.lse_reduce_scatter(static_output, static_lse, graph_output)
        stream.synchronize()

        for replay in range(3):
            output, lse, query = _inputs(200 + replay, rank, 1, device)
            expected_output, expected_query = _reference(output, lse, query, rank)
            static_output.copy_(output)
            static_lse.copy_(lse)
            static_query.copy_(query)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(graph_query, expected_query, rtol=0, atol=0)
            torch.testing.assert_close(
                graph_output.float(),
                expected_output.float(),
                rtol=3e-2,
                atol=3e-2,
            )
        dist.barrier()
        if rank == 0:
            print(
                "PASS Kimi K3 TP16/DCP8 A2A: world=8, local_heads=6, "
                "gathered_heads=48, head_dim=512, query_head_dim=576, "
                "eager_batches=1/2/8/64, graph_batch=1",
                flush=True,
            )
    finally:
        pool.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    if args.world_size != 8:
        raise SystemExit("This exact K3 validator requires --world-size 8")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_extension()
    mp.spawn(
        _worker,
        args=(args.world_size, _free_port()),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
