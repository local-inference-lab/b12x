"""Benchmark Kimi paired projection gathering with batched expert selection.

The benchmark covers tensor-parallel worlds of two, four, eight, and sixteen
ranks (TP2, TP4, TP8, and TP16) and token batches from one through eight. It
compares paired gathering followed by vLLM ``topk_sigmoid`` against the same
gather followed by B12X Kimi top-16 selection. Expert IDs match the reference
for defined oracle cases; a complete negative-infinity tie selects experts
zero through fifteen in order. Router weights and projection data match exactly.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from collections.abc import Callable
from contextlib import suppress

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    vllm_topk_sigmoid,
)


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
) -> list[float]:
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
    return timings


def _case(
    name: str,
    *,
    batch: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "random":
        generator = torch.Generator(device=device).manual_seed(12004)
        logits = torch.randn((batch, 896), generator=generator, device=device)
        bias = torch.randn((896,), generator=generator, device=device).mul_(0.02)
    elif name == "ties":
        logits = torch.zeros((batch, 896), device=device)
        bias = torch.zeros((896,), device=device)
    elif name == "near_ties":
        logits = (
            torch.linspace(-1e-5, 1e-5, 896, device=device)
            .view(1, 896)
            .repeat(batch, 1)
        )
        bias = torch.linspace(1e-5, -1e-5, 896, device=device)
    elif name == "wide":
        logits = (
            torch.linspace(-80.0, 80.0, 896, device=device)
            .view(1, 896)
            .repeat(batch, 1)
        )
        bias = torch.sin(torch.arange(896, device=device, dtype=torch.float32))
        logits[0, 17] = float("nan")
        bias[33] = float("inf")
    elif name == "all_neg_inf":
        logits = torch.zeros((batch, 896), device=device)
        bias = torch.full((896,), float("-inf"), device=device)
    else:
        raise ValueError(name)
    local_width = 896 // world_size
    local = logits[
        :, rank * local_width : (rank + 1) * local_width
    ].contiguous()
    return local, bias.contiguous()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    iterations: int,
    samples: int,
    batch: int,
    topk_threads: int,
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
    b12x_pool: PCIeDCPA2APool | None = None
    try:
        query_head_dim = 10752 // world_size
        reference_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=batch,
            total_heads=world_size,
            head_dim=query_head_dim,
            query_head_dim=query_head_dim,
        )
        b12x_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=batch,
            total_heads=world_size,
            head_dim=query_head_dim,
            query_head_dim=query_head_dim,
        )
        reference_channel = "k3-pair-topk:reference"
        b12x_channel = "k3-pair-topk:b12x"
        reference_pool.prepare_channels((reference_channel,))
        b12x_pool.prepare_channels((b12x_channel,))
        b12x_pool.prepare_graph_kimi_topk16(
            threads=topk_threads, channel_id=b12x_channel
        )

        local_down_width = 3584 // world_size
        down_local = (
            torch.arange(
                batch * local_down_width,
                device=device,
                dtype=torch.float32,
            )
            .view(batch, local_down_width)
            .add_(rank * batch * local_down_width)
            .to(torch.bfloat16)
        )
        reference_down = torch.empty(
            (batch, 3584), device=device, dtype=torch.bfloat16
        )
        reference_router = torch.empty(
            (batch, 896), device=device, dtype=torch.float32
        )
        reference_weights = torch.empty(
            (batch, 16), device=device, dtype=torch.float32
        )
        reference_ids = torch.empty(
            (batch, 16), device=device, dtype=torch.int32
        )
        reference_token_expert = torch.empty_like(reference_ids)
        b12x_down = torch.empty_like(reference_down)
        b12x_router = torch.empty_like(reference_router)
        b12x_weights = torch.empty(
            (batch, 16), device=device, dtype=torch.float32
        )
        b12x_ids = torch.empty(
            (batch, 16), device=device, dtype=torch.int32
        )
        all_neg_inf_ids = (
            torch.arange(16, device=device, dtype=torch.int32)
            .view(1, 16)
            .expand(batch, -1)
        )
    except Exception:
        if b12x_pool is not None:
            with suppress(Exception):
                b12x_pool.close()
        if reference_pool is not None:
            with suppress(Exception):
                reference_pool.close()
        if dist.is_initialized():
            with suppress(Exception):
                dist.destroy_process_group()
        raise

    try:
        if rank == 0:
            print(
                "case,ids_valid,ids_reference_exact,ids_contract_exact,"
                "ids_unique,ids_in_range,weights_exact,"
                "weights_finite,weights_positive,weights_normalized,"
                "finite_mask_exact,nonzero_mask_exact,max_abs,down_exact",
                flush=True,
            )
        benchmark_router: torch.Tensor | None = None
        benchmark_bias: torch.Tensor | None = None
        for name in ("random", "ties", "near_ties", "wide", "all_neg_inf"):
            local_router, bias = _case(
                name,
                batch=batch,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            reference_pool.all_gather_pair(
                down_local,
                local_router,
                reference_down,
                reference_router,
                channel_id=reference_channel,
            )
            vllm_topk_sigmoid(
                reference_weights,
                reference_ids,
                reference_token_expert,
                reference_router,
                True,
                bias,
                1.0,
            )
            b12x_pool.all_gather_pair(
                down_local,
                local_router,
                b12x_down,
                b12x_router,
                channel_id=b12x_channel,
            )
            b12x_pool.kimi_topk16(
                b12x_router,
                bias,
                b12x_weights,
                b12x_ids,
                threads=topk_threads,
                channel_id=b12x_channel,
            )
            torch.cuda.synchronize(device)
            ids_exact = torch.equal(b12x_ids, reference_ids)
            sorted_ids = torch.sort(b12x_ids, dim=1).values
            ids_unique = bool(
                ((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()).item()
            )
            ids_in_range = bool(
                ((b12x_ids >= 0) & (b12x_ids < 896)).all().item()
            )
            # vLLM emits duplicate expert zero for the complete tie. B12X
            # deterministically selects the lowest sixteen expert IDs.
            ids_contract_exact = torch.equal(
                b12x_ids,
                all_neg_inf_ids if name == "all_neg_inf" else reference_ids,
            )
            ids_valid = ids_contract_exact
            weights_exact = torch.equal(b12x_weights, reference_weights)
            weights_finite = bool(torch.isfinite(b12x_weights).all().item())
            weights_positive = bool((b12x_weights > 0).all().item())
            weights_normalized = bool(
                torch.allclose(
                    b12x_weights.sum(dim=1),
                    torch.ones(batch, device=device, dtype=torch.float32),
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            finite_mask_exact = torch.equal(
                torch.isfinite(b12x_weights),
                torch.isfinite(reference_weights),
            )
            nonzero_mask_exact = torch.equal(
                b12x_weights != 0,
                reference_weights != 0,
            )
            max_abs = float((b12x_weights - reference_weights).abs().max().item())
            down_exact = torch.equal(b12x_down, reference_down)
            if rank == 0:
                print(
                    f"{name},{int(ids_valid)},{int(ids_exact)},"
                    f"{int(ids_contract_exact)},{int(ids_unique)},"
                    f"{int(ids_in_range)},{int(weights_exact)},"
                    f"{int(weights_finite)},{int(weights_positive)},"
                    f"{int(weights_normalized)},"
                    f"{int(finite_mask_exact)},{int(nonzero_mask_exact)},"
                    f"{max_abs:.9g},{int(down_exact)}",
                    flush=True,
                )
            failures = torch.tensor(
                int(not ids_valid)
                + int(not weights_exact)
                + int(not weights_finite)
                + int(not weights_positive)
                + int(not weights_normalized)
                + int(not finite_mask_exact)
                + int(not nonzero_mask_exact)
                + int(not down_exact),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(failures)
            if failures.item() != 0:
                raise AssertionError(
                    f"{name} violates the expert-selection contract"
                )
            if name == "random":
                benchmark_router = local_router
                benchmark_bias = bias

        assert benchmark_router is not None and benchmark_bias is not None
        def reference_fn() -> None:
            reference_pool.all_gather_pair(
                down_local,
                benchmark_router,
                reference_down,
                reference_router,
                channel_id=reference_channel,
            )
            vllm_topk_sigmoid(
                reference_weights,
                reference_ids,
                reference_token_expert,
                reference_router,
                True,
                benchmark_bias,
                1.0,
            )

        def b12x_fn() -> None:
            b12x_pool.all_gather_pair(
                down_local,
                benchmark_router,
                b12x_down,
                b12x_router,
                channel_id=b12x_channel,
            )
            b12x_pool.kimi_topk16(
                b12x_router,
                benchmark_bias,
                b12x_weights,
                b12x_ids,
                threads=topk_threads,
                channel_id=b12x_channel,
            )

        reference_graph = _capture(reference_fn, reference_pool, reference_channel)
        b12x_graph = _capture(b12x_fn, b12x_pool, b12x_channel)
        graph_tensors = (
            benchmark_router,
            benchmark_bias,
            reference_down,
            reference_router,
            reference_weights,
            reference_ids,
            b12x_down,
            b12x_router,
            b12x_weights,
            b12x_ids,
        )
        graph_ptrs = tuple(tensor.data_ptr() for tensor in graph_tensors)
        if rank == 0:
            print(
                "graph_case,ids_valid,ids_reference_exact,"
                "ids_contract_exact,ids_unique,ids_in_range,weights_exact,"
                "weights_finite,"
                "weights_positive,weights_normalized,down_exact,"
                "addresses_stable,replay_allocation_stable",
                flush=True,
            )
        for name in ("random", "ties", "near_ties", "wide", "all_neg_inf"):
            replay_router, replay_bias = _case(
                name,
                batch=batch,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            benchmark_router.copy_(replay_router)
            benchmark_bias.copy_(replay_bias)
            torch.cuda.synchronize(device)
            allocated_before = torch.cuda.memory_allocated(device)
            reference_graph.replay()
            b12x_graph.replay()
            torch.cuda.synchronize(device)
            allocated_after = torch.cuda.memory_allocated(device)

            replay_ids_exact = torch.equal(b12x_ids, reference_ids)
            replay_sorted_ids = torch.sort(b12x_ids, dim=1).values
            replay_ids_unique = bool(
                (
                    replay_sorted_ids[:, 1:]
                    != replay_sorted_ids[:, :-1]
                ).all().item()
            )
            replay_ids_in_range = bool(
                ((b12x_ids >= 0) & (b12x_ids < 896)).all().item()
            )
            replay_ids_contract_exact = torch.equal(
                b12x_ids,
                all_neg_inf_ids if name == "all_neg_inf" else reference_ids,
            )
            replay_ids_valid = replay_ids_contract_exact
            replay_weights_exact = torch.equal(
                b12x_weights, reference_weights
            )
            replay_weights_finite = bool(
                torch.isfinite(b12x_weights).all().item()
            )
            replay_weights_positive = bool((b12x_weights > 0).all().item())
            replay_weights_normalized = bool(
                torch.allclose(
                    b12x_weights.sum(dim=1),
                    torch.ones(batch, device=device, dtype=torch.float32),
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            replay_down_exact = torch.equal(b12x_down, reference_down)
            addresses_stable = graph_ptrs == tuple(
                tensor.data_ptr() for tensor in graph_tensors
            )
            replay_allocation_stable = allocated_before == allocated_after
            if rank == 0:
                print(
                    f"{name},{int(replay_ids_valid)},"
                    f"{int(replay_ids_exact)},"
                    f"{int(replay_ids_contract_exact)},"
                    f"{int(replay_ids_unique)},"
                    f"{int(replay_ids_in_range)},"
                    f"{int(replay_weights_exact)},"
                    f"{int(replay_weights_finite)},"
                    f"{int(replay_weights_positive)},"
                    f"{int(replay_weights_normalized)},"
                    f"{int(replay_down_exact)},{int(addresses_stable)},"
                    f"{int(replay_allocation_stable)}",
                    flush=True,
                )
            replay_failures = torch.tensor(
                int(not replay_ids_valid)
                + int(not replay_weights_exact)
                + int(not replay_weights_finite)
                + int(not replay_weights_positive)
                + int(not replay_weights_normalized)
                + int(not replay_down_exact)
                + int(not addresses_stable)
                + int(not replay_allocation_stable),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(replay_failures)
            if replay_failures.item() != 0:
                raise AssertionError(
                    f"{name} CUDA Graph replay violates the routing contract"
                )

        timing_router, timing_bias = _case(
            "random",
            batch=batch,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        benchmark_router.copy_(timing_router)
        benchmark_bias.copy_(timing_bias)
        torch.cuda.synchronize(device)

        for graph in (reference_graph, b12x_graph):
            for _ in range(warmup):
                graph.replay()
        torch.cuda.synchronize(device)
        reference_samples_us: list[float] = []
        b12x_samples_us: list[float] = []
        for sample in range(samples):
            ordered_graphs = (
                (
                    (reference_graph, reference_samples_us),
                    (b12x_graph, b12x_samples_us),
                )
                if sample % 2 == 0
                else (
                    (b12x_graph, b12x_samples_us),
                    (reference_graph, reference_samples_us),
                )
            )
            for graph, sample_sink in ordered_graphs:
                sample_sink.extend(
                    _measure(
                        graph,
                        device=device,
                        warmup=0,
                        iterations=iterations,
                        samples=1,
                    )
                )
        if rank == 0:
            reference_us = statistics.median(reference_samples_us)
            b12x_us = statistics.median(b12x_samples_us)
            print(
                "reference_samples_us," + json.dumps(reference_samples_us),
                flush=True,
            )
            print(
                "b12x_samples_us," + json.dumps(b12x_samples_us),
                flush=True,
            )
            print(
                "speedup_definition,reference_median_us/b12x_median_us; "
                "values greater than one mean the B12X path is faster",
                flush=True,
            )
            print(
                "world_size,batch,topk_threads,reference_us,b12x_us,"
                "speedup,saved_us\n"
                f"{world_size},{batch},{topk_threads},{reference_us:.6f},"
                f"{b12x_us:.6f},"
                f"{reference_us / b12x_us:.6f},{reference_us - b12x_us:.6f}",
                flush=True,
            )
    finally:
        if b12x_pool is not None:
            with suppress(Exception):
                b12x_pool.close()
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--topk-threads", type=int, default=256)
    args = parser.parse_args()
    if args.world_size not in (2, 4, 8, 16):
        raise SystemExit(
            "Kimi paired gather+top-k benchmark requires TP2, TP4, TP8, or TP16"
        )
    if args.batch < 1 or args.batch > 8:
        raise SystemExit("Kimi paired gather+top-k batch must be between 1 and 8")
    if args.topk_threads not in (128, 256, 512):
        raise SystemExit("Kimi top-k threads must be 128, 256, or 512")
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.warmup,
            args.iterations,
            args.samples,
            args.batch,
            args.topk_threads,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
