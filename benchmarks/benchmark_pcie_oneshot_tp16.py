#!/usr/bin/env python3
"""Model-free TP16 correctness and latency gate for PCIe oneshot all-reduce."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_oneshot import (
    PCIeOneshotAllReducePool,
    _load_extension,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _max_rank(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _graph_us(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    iterations: int,
) -> float:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    stream.synchronize()
    dist.barrier()
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            graph.replay()
    stream.synchronize()
    return (time.perf_counter() - started) * 1.0e6 / iterations


def _worker(
    rank: int,
    world_size: int,
    port: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        eager_buffer_bytes=84 * 1024,
        max_size=84 * 1024,
    )
    try:
        stream = torch.cuda.Stream(device=device)
        pool.for_stream(stream)
        shape = (1, hidden_size)

        custom_inp = torch.full(
            shape, rank + 1, dtype=torch.bfloat16, device=device
        )
        custom_out = torch.empty_like(custom_inp)
        custom_graph = torch.cuda.CUDAGraph()
        with pool.capture(stream), torch.cuda.graph(custom_graph, stream=stream):
            pool.all_reduce(custom_inp, out=custom_out)
        custom_graph.replay()
        stream.synchronize()
        expected_sum = world_size * (world_size + 1) // 2
        torch.testing.assert_close(
            custom_out.float(),
            torch.full_like(custom_out.float(), expected_sum),
            rtol=0,
            atol=0,
        )

        nccl_inp = custom_inp.clone()
        nccl_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(nccl_graph, stream=stream):
            dist.all_reduce(nccl_inp)
        nccl_graph.replay()
        stream.synchronize()
        torch.testing.assert_close(
            nccl_inp.float(),
            torch.full_like(nccl_inp.float(), expected_sum),
            rtol=0,
            atol=0,
        )

        # Also gate the single-launch operation enabled by vLLM's
        # fuse_allreduce_rms pass for the exact K3 decode row.
        fused_inp = custom_inp.clone()
        fused_residual = torch.full_like(fused_inp, 0.5)
        fused_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        fused_out = torch.empty_like(fused_inp)
        fused_residual_out = torch.empty_like(fused_inp)
        fused_graph = torch.cuda.CUDAGraph()
        with pool.capture(stream), torch.cuda.graph(fused_graph, stream=stream):
            pool.all_reduce_fused_add_rms_norm(
                fused_inp,
                fused_residual,
                fused_weight,
                1.0e-6,
                out=fused_out,
                residual_out=fused_residual_out,
            )
        fused_graph.replay()
        stream.synchronize()
        expected_residual = torch.full_like(
            fused_residual_out.float(), expected_sum + 0.5
        )
        torch.testing.assert_close(
            fused_residual_out.float(), expected_residual, rtol=0, atol=0.51
        )
        torch.testing.assert_close(
            fused_out.float(), torch.ones_like(fused_out.float()), rtol=0, atol=0.01
        )

        custom_us = _max_rank(
            _graph_us(custom_graph, stream, warmup=warmup, iterations=iterations),
            device,
        )
        nccl_us = _max_rank(
            _graph_us(nccl_graph, stream, warmup=warmup, iterations=iterations),
            device,
        )
        fused_us = _max_rank(
            _graph_us(fused_graph, stream, warmup=warmup, iterations=iterations),
            device,
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "world_size": world_size,
                        "shape": list(shape),
                        "dtype": "bfloat16",
                        "bytes": hidden_size * 2,
                        "custom_oneshot_us": custom_us,
                        "nccl_us": nccl_us,
                        "custom_over_nccl_speedup": nccl_us / custom_us,
                        "fused_allreduce_add_rmsnorm_us": fused_us,
                        "correctness": "pass",
                    },
                    indent=2,
                ),
                flush=True,
            )
        # At TP16, CUDA IPC teardown in the experimental runtime can block in
        # cudaIpcCloseMemHandle after all measured collectives have completed.
        # This process is the isolation boundary for the model-free probe, so
        # let process exit release its CUDA context instead of hiding a valid
        # benchmark result behind an unrelated teardown stall.
        dist.barrier()
        os._exit(0)
    except BaseException:
        # Preserve the actual validation failure, then bypass the same known
        # TP16 IPC teardown stall as the successful path.
        traceback.print_exc()
        os._exit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    _load_extension()
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.hidden_size,
            args.warmup,
            args.iterations,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
