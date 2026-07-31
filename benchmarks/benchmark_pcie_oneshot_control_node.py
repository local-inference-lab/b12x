"""Measure staged PCIe-oneshot latency and its captured control-node contract.

Run the same command at the folded-selector baseline and corrected head:

    torchrun --standalone --nproc-per-node=4 \
      benchmarks/benchmark_pcie_oneshot_control_node.py \
      --label corrected --output corrected.json

Then pass ``--baseline-json baseline.json`` on the corrected run to report
matched deltas. The corrected staged path should expose two ordered graph
kernels (one 1x1 control node plus one worker); the registered path remains
one worker launch and is not graph-capturable through the public runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from cuda.bindings import runtime as cudart

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="unnamed")
    parser.add_argument("--numel", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    return parser.parse_args()


def _graph_kernel_count(graph: torch.cuda.CUDAGraph) -> int:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes(size) failed: {result}")
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    if result != cudart.cudaError_t.cudaSuccess or returned_nodes != num_nodes:
        raise RuntimeError(
            f"cudaGraphGetNodes(data) failed: {result}, {returned_nodes=}, {num_nodes=}"
        )
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    count = 0
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphNodeGetType failed: {result}")
        count += node_type == kernel_type
    return count


def _measure(
    operation,
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iters: int,
) -> tuple[float, list[float]]:
    cold_start = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    cold_start.record(stream)
    operation()
    cold_end.record(stream)
    stream.synchronize()
    cold_us = float(cold_start.elapsed_time(cold_end) * 1000)

    for _ in range(warmup):
        operation()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        operation()
        end.record(stream)
    stream.synchronize()
    samples = [
        float(start.elapsed_time(end) * 1000)
        for start, end in zip(starts, ends, strict=True)
    ]
    return cold_us, samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "mean_us": sum(ordered) / len(ordered),
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _gather_rank_metrics(
    eager_cold_us: float,
    eager_samples: list[float],
    graph_cold_us: float,
    graph_samples: list[float],
    graph_kernel_nodes: int,
    device: torch.device,
) -> list[dict[str, object]]:
    eager = _summary(eager_samples)
    graph = _summary(graph_samples)
    names = ("mean_us", "p50_us", "p95_us", "min_us", "max_us")
    local = torch.tensor(
        [
            eager_cold_us,
            *(eager[name] for name in names),
            graph_cold_us,
            *(graph[name] for name in names),
            float(graph_kernel_nodes),
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)

    result = []
    for rank, values_tensor in enumerate(gathered):
        values = values_tensor.cpu().tolist()
        result.append(
            {
                "rank": rank,
                "eager": {
                    "cold_us": values[0],
                    **dict(zip(names, values[1:6], strict=True)),
                },
                "graph": {
                    "cold_us": values[6],
                    **dict(zip(names, values[7:12], strict=True)),
                    "kernel_nodes": int(values[12]),
                },
            }
        )
    return result


def _slowest_rank_summary(per_rank: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode in ("eager", "graph"):
        mode_rows = [row[mode] for row in per_rank]
        result[mode] = {
            key: max(float(row[key]) for row in mode_rows)
            for key in ("cold_us", "mean_us", "p50_us", "p95_us", "min_us", "max_us")
        }
    result["graph"]["kernel_nodes"] = max(
        int(row["graph"]["kernel_nodes"]) for row in per_rank
    )
    return result


def _comparison(
    current: dict[str, object],
    baseline_path: Path,
) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = {"baseline_label": baseline["label"], "delta": {}}
    for mode in ("eager", "graph"):
        result["delta"][mode] = {}
        for metric in ("mean_us", "p50_us", "p95_us"):
            before = float(baseline["slowest_rank"][mode][metric])
            after = float(current["slowest_rank"][mode][metric])
            result["delta"][mode][metric] = {
                "us": after - before,
                "percent": (after / before - 1) * 100,
            }
    return result


def main() -> None:
    args = _parse_args()
    if args.iters <= 0 or args.warmup < 0 or args.numel <= 0:
        raise ValueError("numel/iters must be positive and warmup non-negative")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    nbytes = args.numel * torch.bfloat16.itemsize
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=nbytes,
        max_size=nbytes,
    )
    stream = torch.cuda.Stream(device=device)
    try:
        channel = pool.for_stream(stream)
        eager_inp = torch.full(
            (args.numel,),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        eager_out = torch.empty_like(eager_inp)

        def eager_operation() -> None:
            with torch.cuda.stream(stream):
                channel.all_reduce(eager_inp, out=eager_out)

        dist.barrier()
        eager_cold_us, eager_samples = _measure(
            eager_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
        torch.testing.assert_close(
            eager_out,
            torch.full_like(eager_out, expected),
            rtol=0,
            atol=0,
        )

        graph_inp = torch.full_like(eager_inp, rank + 2)
        graph_out = torch.empty_like(graph_inp)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with pool.capture(stream) as graph_channel, torch.cuda.graph(
            graph,
            stream=stream,
        ):
            graph_channel.all_reduce(graph_inp, out=graph_out)
        graph_kernel_nodes = _graph_kernel_count(graph)

        def graph_operation() -> None:
            with torch.cuda.stream(stream):
                graph.replay()

        dist.barrier()
        graph_cold_us, graph_samples = _measure(
            graph_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        expected += dist.get_world_size()
        torch.testing.assert_close(
            graph_out,
            torch.full_like(graph_out, expected),
            rtol=0,
            atol=0,
        )

        per_rank = _gather_rank_metrics(
            eager_cold_us,
            eager_samples,
            graph_cold_us,
            graph_samples,
            graph_kernel_nodes,
            device,
        )
        if rank == 0:
            result = {
                "label": args.label,
                "world_size": dist.get_world_size(),
                "numel": args.numel,
                "dtype": "bfloat16",
                "warmup": args.warmup,
                "iters": args.iters,
                "staged_graph_expected_kernel_nodes": 2,
                "registered_expected_kernel_nodes": 1,
                "per_rank": per_rank,
                "slowest_rank": _slowest_rank_summary(per_rank),
            }
            if args.baseline_json is not None:
                result["comparison"] = _comparison(result, args.baseline_json)
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output is not None:
                args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        pool.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
