#!/usr/bin/env python3
"""Graph-first benchmark for the one-launch Kimi-K3 TP12 X4T scale decoder."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from b12x._lib.quant.x4t_scales import (
    X4TScaleBatch,
    decode_x4t_tp12_w4a16_scales,
    make_x4t_scale_batch,
)


def _make_batch(
    *,
    experts: int,
    rows: int,
    columns: int,
    exceptions_per_expert: int,
    device: torch.device,
    exception_task_rows: int,
    exception_row_rotation: int = 0,
) -> X4TScaleBatch:
    tiles = rows // 16
    tile_bytes = 16 * (1 + (columns + 7) // 8)
    fixed = torch.randint(
        0,
        256,
        (experts, tiles, tile_bytes),
        dtype=torch.uint8,
        device="cpu",
    )
    count = experts * exceptions_per_expert
    linear = torch.arange(count, dtype=torch.int64)
    local = linear % exceptions_per_expert
    expert = linear // exceptions_per_expert
    positions = (expert * 104729 + local * 8191 + 17) % (rows * columns)
    values = 80 + ((expert * 13 + local * 7) % 160)
    positions, order = positions.view(experts, -1).sort(dim=1)
    values = values.view(experts, -1).gather(1, order)
    exceptions = (positions | (values << 24)).to(torch.uint32).contiguous()
    return make_x4t_scale_batch(
        [plane.contiguous() for plane in fixed],
        [plane.contiguous() for plane in exceptions],
        rows=rows,
        columns=columns,
        device=device,
        exception_task_rows=exception_task_rows,
        exception_row_rotation=exception_row_rotation,
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=76)
    parser.add_argument("--route-experts", type=int, default=896)
    parser.add_argument("--active-capacity", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.active_capacity > args.experts:
        raise ValueError("active capacity cannot exceed local experts")
    if args.experts > args.route_experts:
        raise ValueError("local experts cannot exceed route experts")

    device = torch.device("cuda")
    torch.manual_seed(20260803)
    # Layer-24 measured means, rounded up: ~0.512% of 57,344 FC1 scales and
    # ~0.310% of 28,672 FC2 scales per expert.
    w13 = _make_batch(
        experts=args.experts,
        rows=512,
        columns=112,
        exceptions_per_expert=294,
        device=device,
        exception_task_rows=128,
        exception_row_rotation=256,
    )
    w2 = _make_batch(
        experts=args.experts,
        rows=3584,
        columns=8,
        exceptions_per_expert=89,
        device=device,
        exception_task_rows=896,
    )
    output_w13 = torch.empty(
        (args.experts, 112, 512), dtype=torch.uint8, device=device
    )
    output_w2 = torch.empty(
        (args.experts, 8, 3584), dtype=torch.uint8, device=device
    )

    expert_map = torch.full(
        (args.route_experts,), -1, dtype=torch.int32, device=device
    )
    global_ids = (torch.arange(args.experts, device=device) * 11) % args.route_experts
    expert_map[global_ids] = torch.arange(
        args.experts, dtype=torch.int32, device=device
    )
    expert_ids = global_ids[: args.active_capacity].to(torch.int32).contiguous()

    def launch() -> None:
        decode_x4t_tp12_w4a16_scales(
            w13,
            w2,
            expert_ids,
            output_w13,
            output_w2,
            expert_map=expert_map,
            expert_ids_unique=True,
        )

    # Resolve/compile before capture, matching serving warmup.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(float(start.elapsed_time(end)) * 1000.0)

    result = {
        "codec": "x4t",
        "target": "kimi_k3_tp12_w4a16_scales",
        "graph_replay": True,
        "launches": 1,
        "experts": args.experts,
        "route_experts": args.route_experts,
        "active_capacity": args.active_capacity,
        "w13_exceptions_per_expert": 294,
        "w2_exceptions_per_expert": 89,
        "scratch_bytes": output_w13.numel() + output_w2.numel(),
        "latency_us": {
            "min": min(samples_us),
            "p10": _percentile(samples_us, 0.10),
            "median": statistics.median(samples_us),
            "p90": _percentile(samples_us, 0.90),
            "max": max(samples_us),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
