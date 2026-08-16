"""Validate and benchmark vocabulary-parallel greedy sampling."""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from contextlib import suppress

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_vocab_argmax import PCIeVocabParallelArgmax


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    rank: int,
    world_size: int,
    local_vocab_size: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(12000 + rank)
    base = torch.randn(
        (batch, local_vocab_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    bias = torch.randn(
        (batch, local_vocab_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.02)

    if batch > 1:
        base[1:].fill_(-1.0)
        bias[1:].zero_()
    if rank == 0 and batch > 1:
        base[1, 0] = 0.0
    if rank == 0 and batch > 2:
        base[2, 3] = float("nan")
    if rank == 0 and batch > 3:
        base[3, 0] = -0.0
    elif rank != 0 and batch > 1:
        base[1, 0] = 0.0
    if rank == world_size - 1 and batch > 2:
        base[2, local_vocab_size - 1] = float("nan")
    if rank == world_size - 1 and batch > 3:
        base[3, local_vocab_size - 1] = 0.0
    if rank == 0 and batch > 4:
        base[4, 5] = float("inf")
    if rank == world_size - 1 and batch > 4:
        base[4, local_vocab_size - 1] = float("inf")
    if batch > 5:
        base[5].fill_(float("-inf"))

    local_scores = (base.float() + bias.float()).bfloat16()
    gathered = [torch.empty_like(local_scores) for _ in range(world_size)]
    dist.all_gather(gathered, local_scores)
    expected = torch.cat(gathered, dim=1).argmax(dim=1).to(torch.int64)
    if batch > 1:
        expected[1] = 0
    if batch > 2:
        expected[2] = 3
    if batch > 3:
        expected[3] = 0
    if batch > 4:
        expected[4] = 5
    if batch > 5:
        expected[5] = 0
    return base, bias, expected


def _measure(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> float:
    timings: list[float] = []
    for _ in range(samples):
        dist.barrier()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize(device)
        elapsed_us = (time.perf_counter() - start) * 1e6 / iterations
        maximum = torch.tensor(elapsed_us, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        timings.append(float(maximum.item()))
    return statistics.median(timings)


def _validate_random_inputs(
    runtime: PCIeVocabParallelArgmax,
    *,
    rank: int,
    world_size: int,
    local_vocab_size: int,
    batch: int,
    cases: int,
    device: torch.device,
) -> None:
    output = torch.empty(batch, dtype=torch.int64, device=device)
    for case in range(cases):
        generator = torch.Generator(device=device).manual_seed(
            20000 + case * world_size + rank
        )
        base = torch.randn(
            (batch, local_vocab_size),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        bias = torch.randn(
            (batch, local_vocab_size),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        local_scores = (base.float() + bias.float()).bfloat16()
        gathered = [torch.empty_like(local_scores) for _ in range(world_size)]
        dist.all_gather(gathered, local_scores)
        expected = torch.cat(gathered, dim=1).argmax(dim=1).to(torch.int64)
        runtime.fused_add_argmax(base, bias, output)
        torch.cuda.synchronize(device)
        if not torch.equal(output, expected):
            raise AssertionError(
                f"rank {rank}, random case {case}: expected "
                f"{expected.tolist()}, got {output.tolist()}"
            )


def _worker(
    rank: int,
    world_size: int,
    local_vocab_size: int,
    batch: int,
    port: int,
    warmup: int,
    iterations: int,
    samples: int,
    random_cases: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    runtime: PCIeVocabParallelArgmax | None = None
    try:
        runtime = PCIeVocabParallelArgmax.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            local_vocab_size=local_vocab_size,
            max_batch_size=batch,
        )
        _validate_random_inputs(
            runtime,
            rank=rank,
            world_size=world_size,
            local_vocab_size=local_vocab_size,
            batch=batch,
            cases=random_cases,
            device=device,
        )
        base, bias, expected = _inputs(
            rank,
            world_size,
            local_vocab_size,
            batch,
            device,
        )
        output = torch.empty(batch, dtype=torch.int64, device=device)
        for _ in range(3):
            runtime.fused_add_argmax(base, bias, output)
        torch.cuda.synchronize(device)
        if not torch.equal(output, expected):
            raise AssertionError(
                f"rank {rank}: expected {expected.tolist()}, got {output.tolist()}"
            )

        graph = torch.cuda.CUDAGraph()
        with runtime.capture(), torch.cuda.graph(graph):
            runtime.fused_add_argmax(base, bias, output)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(output, expected):
            raise AssertionError(
                "CUDA graph replay changed vocabulary argmax output"
            )

        if batch > 1:
            one_output = torch.empty(1, dtype=torch.int64, device=device)
            one_graph = torch.cuda.CUDAGraph()
            with runtime.capture(), torch.cuda.graph(one_graph):
                runtime.fused_add_argmax(base[:1], bias[:1], one_output)
            for _ in range(100):
                one_graph.replay()
                graph.replay()
            torch.cuda.synchronize(device)
            if not torch.equal(one_output, expected[:1]):
                raise AssertionError(
                    "alternating batch-one graph replay changed argmax output"
                )
            if not torch.equal(output, expected):
                raise AssertionError(
                    "alternating full-batch graph replay changed argmax output"
                )

        latency_us = _measure(
            graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        if rank == 0:
            print(
                "world_size,batch,local_vocab_size,latency_us\n"
                f"{world_size},{batch},{local_vocab_size},{latency_us:.6f}",
                flush=True,
            )
    finally:
        if runtime is not None:
            with suppress(Exception):
                runtime.close()
        if dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--local-vocab-size", type=int)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--random-cases", type=int, default=32)
    args = parser.parse_args()
    if args.world_size not in (8, 12, 16):
        raise SystemExit("world size must be 8, 12, or 16")
    if not 1 <= args.batch <= 8:
        raise SystemExit("batch must be in [1, 8]")
    local_vocab_size = args.local_vocab_size
    if local_vocab_size is None:
        global_vocab_size = 163840 if args.world_size in (8, 16) else 163968
        local_vocab_size = global_vocab_size // args.world_size
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            local_vocab_size,
            args.batch,
            _free_port(),
            args.warmup,
            args.iterations,
            args.samples,
            args.random_cases,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
