from __future__ import annotations

import argparse
import socket
import statistics
import time
from collections.abc import Callable
from contextlib import suppress

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm._custom_ops as ops
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capture(
    fn: Callable[[], None],
    pool: PCIeDCPA2APool,
    channel_id: str,
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pool.capture(channel_id=channel_id), torch.cuda.graph(graph):
        fn()
    return graph


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


def _case(
    name: str,
    *,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "random":
        generator = torch.Generator(device=device).manual_seed(12004)
        logits = torch.randn((1, 896), generator=generator, device=device)
        bias = torch.randn((896,), generator=generator, device=device).mul_(0.02)
    elif name == "ties":
        logits = torch.zeros((1, 896), device=device)
        bias = torch.zeros((896,), device=device)
    elif name == "near_ties":
        logits = torch.linspace(-1e-5, 1e-5, 896, device=device).view(1, 896)
        bias = torch.linspace(1e-5, -1e-5, 896, device=device)
    elif name == "wide":
        logits = torch.linspace(-80.0, 80.0, 896, device=device).view(1, 896)
        bias = torch.sin(torch.arange(896, device=device, dtype=torch.float32))
        logits[0, 17] = float("nan")
        bias[33] = float("inf")
    else:
        raise ValueError(name)
    local = logits[:, rank * 56 : (rank + 1) * 56].contiguous()
    return local, bias.contiguous()


def _equal_with_matching_nan(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(
        torch.all(
            torch.eq(actual, expected) | (torch.isnan(actual) & torch.isnan(expected))
        )
    )


def _worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    iterations: int,
    samples: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    reference_pool: PCIeDCPA2APool | None = None
    fused_pool: PCIeDCPA2APool | None = None
    try:
        reference_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=1,
            total_heads=world_size,
            head_dim=672,
            query_head_dim=672,
        )
        fused_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=1,
            total_heads=world_size,
            head_dim=672,
            query_head_dim=672,
        )
        reference_channel = "k3-pair-topk:reference"
        fused_channel = "k3-pair-topk:fused"
        reference_pool.prepare_channels((reference_channel,))
        fused_pool.prepare_channels((fused_channel,))

        down_local = (
            torch.arange(224, device=device, dtype=torch.float32)
            .add_(rank * 224)
            .to(torch.bfloat16)
            .view(1, 224)
        )
        reference_down = torch.empty((1, 3584), device=device, dtype=torch.bfloat16)
        reference_router = torch.empty((1, 896), device=device, dtype=torch.float32)
        fused_down = torch.empty_like(reference_down)
        fused_weights = torch.empty((1, 16), device=device, dtype=torch.float32)
        fused_ids = torch.empty((1, 16), device=device, dtype=torch.int32)
    except Exception:
        if fused_pool is not None:
            with suppress(Exception):
                fused_pool.close()
        if reference_pool is not None:
            with suppress(Exception):
                reference_pool.close()
        if dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()
        raise

    try:
        if rank == 0:
            print("case,ids_exact,weights_exact,max_abs,down_exact", flush=True)
        benchmark_router: torch.Tensor | None = None
        benchmark_bias: torch.Tensor | None = None
        for name in ("random", "ties", "near_ties", "wide"):
            local_router, bias = _case(name, rank=rank, device=device)
            reference_pool.all_gather_pair(
                down_local,
                local_router,
                reference_down,
                reference_router,
                channel_id=reference_channel,
            )
            reference_weights, reference_ids = ops.grouped_topk(
                reference_router,
                1,
                1,
                16,
                True,
                1.0,
                bias,
                1,
            )
            fused_pool.all_gather_pair_kimi_topk(
                down_local,
                local_router,
                bias,
                fused_down,
                fused_weights,
                fused_ids,
                channel_id=fused_channel,
            )
            torch.cuda.synchronize(device)
            ids_exact = torch.equal(fused_ids, reference_ids)
            weights_exact = _equal_with_matching_nan(fused_weights, reference_weights)
            max_abs = float((fused_weights - reference_weights).abs().max().item())
            down_exact = torch.equal(fused_down, reference_down)
            if rank == 0:
                print(
                    f"{name},{int(ids_exact)},{int(weights_exact)},"
                    f"{max_abs:.9g},{int(down_exact)}",
                    flush=True,
                )
            failures = torch.tensor(
                int(not ids_exact) + int(not weights_exact) + int(not down_exact),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(failures)
            if failures.item() != 0:
                raise AssertionError(f"{name} differs from HH native reference")
            if name == "random":
                benchmark_router = local_router
                benchmark_bias = bias

        assert benchmark_router is not None and benchmark_bias is not None
        reference_outputs: tuple[torch.Tensor, torch.Tensor] | None = None

        def reference_fn() -> None:
            nonlocal reference_outputs
            reference_pool.all_gather_pair(
                down_local,
                benchmark_router,
                reference_down,
                reference_router,
                channel_id=reference_channel,
            )
            reference_outputs = ops.grouped_topk(
                reference_router,
                1,
                1,
                16,
                True,
                1.0,
                benchmark_bias,
                1,
            )

        def fused_fn() -> None:
            fused_pool.all_gather_pair_kimi_topk(
                down_local,
                benchmark_router,
                benchmark_bias,
                fused_down,
                fused_weights,
                fused_ids,
                channel_id=fused_channel,
            )

        reference_graph = _capture(reference_fn, reference_pool, reference_channel)
        fused_graph = _capture(fused_fn, fused_pool, fused_channel)
        reference_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize(device)
        assert reference_outputs is not None
        if not _equal_with_matching_nan(fused_weights, reference_outputs[0]):
            raise AssertionError("captured weights differ")
        if not torch.equal(fused_ids, reference_outputs[1]):
            raise AssertionError("captured ids differ")
        if not torch.equal(fused_down, reference_down):
            raise AssertionError("captured down projection differs")

        reference_us = _measure(
            reference_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        fused_us = _measure(
            fused_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        if rank == 0:
            print(
                "world_size,reference_us,fused_us,speedup,saved_us\n"
                f"{world_size},{reference_us:.6f},{fused_us:.6f},"
                f"{reference_us / fused_us:.6f},{reference_us - fused_us:.6f}",
                flush=True,
            )
    finally:
        if fused_pool is not None:
            with suppress(Exception):
                fused_pool.close()
        if reference_pool is not None:
            with suppress(Exception):
                reference_pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.world_size != 16:
        raise SystemExit("Kimi paired gather+top-k benchmark requires TP16")
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.warmup,
            args.iterations,
            args.samples,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
