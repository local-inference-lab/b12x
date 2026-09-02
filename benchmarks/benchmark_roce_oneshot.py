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
    ap.add_argument("--max-size", type=int, default=2 << 20)
    ap.add_argument("--gather-rows", default="6,16,96", help="rows of a [rows, 38720] bf16 logits shard to all-gather")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)

    from b12x.comm import roce

    runtime = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD, device=device, max_size=args.max_size, max_gather_bytes=16 << 20
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
    # all-gather of a logits shard [rows, 38720] along the last dim vs NCCL all_gather_into_tensor + copy
    gather_rows = []
    for rows in (int(r) for r in args.gather_rows.split(",")):
        shard = torch.randn(rows, 38720, dtype=dtype, device=device)
        if not runtime.should_all_gather(shard, -1):
            continue
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard)
        expected = torch.cat(parts, dim=-1)
        got = runtime.all_gather(shard, dim=-1)
        torch.cuda.synchronize()
        exact = bool(torch.equal(got, expected))
        stacked = torch.empty((world * rows, 38720), dtype=dtype, device=device)
        def nccl_gather():
            dist.all_gather_into_tensor(stacked, shard)
            return stacked.reshape(world, rows, 38720).movedim(0, 1).reshape(rows, world * 38720)
        nccl = _time_eager(nccl_gather, args.warmups, args.samples)
        eager = _time_eager(lambda: runtime.all_gather(shard, dim=-1), args.warmups, args.samples)
        graph = _time_graph(lambda: runtime.all_gather(shard, dim=-1, out=got), args.warmups, args.samples, args.graph_ops)
        row = {"rows": rows, "shard_bytes": shard.numel() * shard.element_size(), "nccl_us": round(_median_of_max(nccl), 1),
               "roce_eager_us": round(_median_of_max(eager), 1), "roce_graph_us": round(_median_of_max(graph), 1), "exact": exact}
        gather_rows.append(row)
        if rank == 0:
            print(f"all-gather [{rows}, 38720]  nccl+copy {row['nccl_us']:>8.1f} us  roce eager {row['roce_eager_us']:>8.1f} us  roce graph {row['roce_graph_us']:>8.1f} us  exact={exact}", flush=True)
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
            "all_gather": gather_rows,
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
