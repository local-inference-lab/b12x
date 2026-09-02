"""Latency of the RoCE one-shot all-reduce versus torch.distributed (NCCL).

Launch with torchrun on every node (one GPU per node)::

    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29651 \\
        benchmarks/benchmark_roce_oneshot.py --output roce.json

Rank 0 prints a table and writes one JSON document with per-size medians of
the slowest rank (CUDA-event timing around one call, graph replay timing for
the captured variant), plus a correctness check against NCCL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def _median_of_max(values: list[float]) -> float:
    t = torch.tensor([statistics.median(values)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _time_eager(fn, warmups: int, samples: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def _time_graph(fn, warmups: int, samples: int, per_graph: int) -> list[float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(per_graph):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / per_graph)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8192,32768,49152,262144,786432,1048576")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--graph-ops", type=int, default=20)
    ap.add_argument("--max-size", type=int, default=1 << 20)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)

    from b12x.comm import roce

    runtime = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD, device=device, max_size=args.max_size
    )
    runtime.prepare((dtype,))
    sizes = [int(s) for s in args.sizes.split(",") if int(s) <= args.max_size]
    def progress(msg: str) -> None:
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    progress(f"runtime ready: {runtime.stats()}")
    rows = []
    for nbytes in sizes:
        progress(f"size {nbytes}: correctness")
        numel = nbytes // torch.tensor([], dtype=dtype).element_size()
        torch.manual_seed(rank + 17)
        inp = torch.randn(numel, dtype=dtype, device=device)
        out = torch.empty_like(inp)
        expected = inp.clone()
        dist.all_reduce(expected)
        runtime.all_reduce(inp, out=out)
        torch.cuda.synchronize()
        err = (out.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item() or 1.0
        nccl_in = inp.clone()
        progress(f"size {nbytes}: nccl timing")
        nccl = _time_eager(lambda: dist.all_reduce(nccl_in), args.warmups, args.samples)
        progress(f"size {nbytes}: roce eager timing")
        eager = _time_eager(lambda: runtime.all_reduce(inp, out=out), args.warmups, args.samples)
        runtime.check_health()
        progress(f"size {nbytes}: roce graph timing")
        graph = _time_graph(
            lambda: runtime.all_reduce(inp, out=out), args.warmups, args.samples, args.graph_ops
        )
        runtime.check_health()
        row = {
            "bytes": nbytes,
            "nccl_us": round(_median_of_max(nccl), 1),
            "roce_eager_us": round(_median_of_max(eager), 1),
            "roce_graph_us": round(_median_of_max(graph), 1),
            "max_abs_err": err,
            "rel_err": err / scale,
        }
        rows.append(row)
        if rank == 0:
            print(
                f"{nbytes:>9} B  nccl {row['nccl_us']:>8.1f} us  roce eager {row['roce_eager_us']:>8.1f} us"
                f"  roce graph {row['roce_graph_us']:>8.1f} us  rel_err {row['rel_err']:.2e}",
                flush=True,
            )
    stats = runtime.stats()
    if rank == 0:
        doc = {
            "schema": "b12x.comm.roce.oneshot.benchmark",
            "version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "world_size": world,
            "dtype": args.dtype,
            "hostname": os.uname().nodename,
            "runtime": stats,
            "rows": rows,
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(doc, f, indent=2)
            print(f"wrote {args.output}")
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
