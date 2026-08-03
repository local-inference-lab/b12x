"""Exact Kimi-K3 TP12 decode all-reduce benchmark.

Measures CUDA-graph replay for the two BF16 collective shapes used by each K3
MoE layer:
  - latent width 3584: 7,168 bytes
  - hidden width 7168: 14,336 bytes

The custom arm maps at most five peers per rank and validates both FP32 error
and bit-identical replicated output before timing.
"""

from __future__ import annotations

import os
import socket
from statistics import median

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "LL")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "8")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie import AllReduce
from sparkinfer.comm.pcie.pcie_hierarchical import SUPPORTED_BLOCKS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_max_us(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    warmup: int = 20,
    iters: int = 200,
    samples: int = 7,
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)

    timings: list[float] = []
    rank_max = torch.empty((), dtype=torch.float64, device=device)
    for _ in range(samples):
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        rank_max.fill_(start.elapsed_time(end) * 1e3 / iters)
        dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
        timings.append(float(rank_max.item()))
    return timings


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    # Establish NCCL's production transport mappings before importing the
    # bounded-degree IPC peers.  This catches exhaustion of the per-context
    # peer-connection budget rather than benchmarking an unrealistically
    # isolated custom runtime.
    bootstrap = torch.zeros(7168, dtype=torch.bfloat16, device=device)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize(device)

    runtime = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=7168 * 2,
    )
    try:
        if runtime.algorithm != "hierarchical":
            raise AssertionError(f"unexpected algorithm {runtime.algorithm}")
        peer_count = torch.tensor(
            runtime.mapped_peer_count,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(peer_count, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                "elements,bytes,blocks,nccl_median_us,custom_median_us,"
                "speedup,nccl_min_us,nccl_max_us,custom_min_us,custom_max_us,"
                "nccl_max_abs_error,custom_max_abs_error,max_rank_delta,"
                "max_mapped_peers",
                flush=True,
            )

        for elements in (3584, 7168):
            generator = torch.Generator(device=device).manual_seed(1234 + rank)
            inp = torch.randn(
                elements,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            fp32_ref = inp.float()
            dist.all_reduce(fp32_ref)
            nccl_actual = inp.clone()
            dist.all_reduce(nccl_actual)
            nccl_max_abs_error = (nccl_actual.float() - fp32_ref).abs().max()
            dist.all_reduce(nccl_max_abs_error, op=dist.ReduceOp.MAX)

            for blocks in SUPPORTED_BLOCKS:
                actual = runtime.all_reduce(inp, blocks=blocks)
                torch.cuda.synchronize(device)
                custom_max_abs_error = (actual.float() - fp32_ref).abs().max()
                dist.all_reduce(custom_max_abs_error, op=dist.ReduceOp.MAX)

                rank0_output = actual.clone()
                dist.broadcast(rank0_output, src=0)
                rank_delta = (actual.float() - rank0_output.float()).abs().max()
                dist.all_reduce(rank_delta, op=dist.ReduceOp.MAX)
                if float(rank_delta.item()) != 0.0:
                    raise AssertionError(
                        f"custom output differs across ranks: {rank_delta.item()}"
                    )

                nccl_in = torch.zeros_like(inp)
                nccl_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(nccl_graph):
                    dist.all_reduce(nccl_in)

                custom_in = inp.clone()
                custom_in_before = custom_in.clone()
                custom_out = torch.empty_like(inp)
                custom_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(custom_graph):
                    runtime.all_reduce(
                        custom_in,
                        out=custom_out,
                        blocks=blocks,
                    )

                nccl_us = _rank_max_us(nccl_graph, device)
                custom_us = _rank_max_us(custom_graph, device)

                # Timing exercised thousands of graph replays.  Recheck the
                # nonzero result, replicated-output contract, and input
                # immutability after that exact replay sequence.
                custom_graph.replay()
                torch.cuda.synchronize(device)
                if not torch.equal(custom_in, custom_in_before):
                    raise AssertionError("custom graph mutated its input")
                graph_error = (custom_out.float() - fp32_ref).abs().max()
                dist.all_reduce(graph_error, op=dist.ReduceOp.MAX)
                graph_rank0 = custom_out.clone()
                dist.broadcast(graph_rank0, src=0)
                graph_rank_delta = (
                    (custom_out.float() - graph_rank0.float()).abs().max()
                )
                dist.all_reduce(graph_rank_delta, op=dist.ReduceOp.MAX)
                if float(graph_rank_delta.item()) != 0.0:
                    raise AssertionError(
                        "custom graph output differs across ranks: "
                        f"{graph_rank_delta.item()}"
                    )
                if float(graph_error.item()) != float(custom_max_abs_error.item()):
                    raise AssertionError(
                        "custom graph error changed after replay: "
                        f"{graph_error.item()} != {custom_max_abs_error.item()}"
                    )

                if rank == 0:
                    nccl_med = float(median(nccl_us))
                    custom_med = float(median(custom_us))
                    print(
                        f"{elements},{elements * 2},{blocks},"
                        f"{nccl_med:.3f},{custom_med:.3f},"
                        f"{nccl_med / custom_med:.4f},"
                        f"{min(nccl_us):.3f},{max(nccl_us):.3f},"
                        f"{min(custom_us):.3f},{max(custom_us):.3f},"
                        f"{float(nccl_max_abs_error.item()):.6f},"
                        f"{float(custom_max_abs_error.item()):.6f},"
                        f"{float(graph_rank_delta.item()):.1f},"
                        f"{int(peer_count.item())}",
                        flush=True,
                    )
                del nccl_graph, custom_graph
                torch.cuda.synchronize(device)
    finally:
        runtime.close()
        dist.destroy_process_group()


def main() -> None:
    world_size = 12
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"TP12 benchmark needs 12 CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
