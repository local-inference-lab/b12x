#!/usr/bin/env python3
"""Benchmark X4T scale predecode inside the real Kimi-K3 TP12 W4A16 MoE.

The dense and X4T arms use bit-identical FP4 weights and E8M0 scale planes.
Only the X4T arm reconstructs the routed experts' packed scale bytes before
the ordinary BF16-activation W4A16 kernels.  Both arms are CUDA-graph captured
and timed in alternating order.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

try:
    from benchmarks.common import nvidia_smi_gpu_mode_snapshot
except ModuleNotFoundError:
    from common import nvidia_smi_gpu_mode_snapshot
from b12x._lib.quant.x4t_scales import (
    X4TScaleBatch,
    decode_x4t_scales,
    make_x4t_scale_batch,
)
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_fp4_e8m0_k32_weights,
    prepare_w4a16_x4t_weights,
)


_HIDDEN_SIZE = 3_584
_INTERMEDIATE_SIZE = 256
_W13_ROWS = 2 * _INTERMEDIATE_SIZE
_TOPK = 16
_SENTINEL = 0xD6


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


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
    """Build deterministic X4T data at the measured layer-24 exception rate."""

    tiles = rows // 16
    tile_bytes = 16 * (1 + (columns + 7) // 8)
    # Keep the palette bases in a realistic, BF16-safe exponent range.  The
    # sparse stream still exercises values outside each adjacent pair.
    fixed = torch.randint(
        108,
        132,
        (experts, tiles, tile_bytes),
        dtype=torch.uint8,
        device="cpu",
    )
    count = experts * exceptions_per_expert
    linear = torch.arange(count, dtype=torch.int64)
    local = linear % exceptions_per_expert
    expert = linear // exceptions_per_expert
    positions = (expert * 104_729 + local * 8_191 + 17) % (rows * columns)
    values = 104 + ((expert * 13 + local * 7) % 32)
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


def _decode_all_logical(batch: X4TScaleBatch) -> torch.Tensor:
    expert_ids = torch.arange(
        batch.num_experts, dtype=torch.int32, device=batch.fixed.device
    )
    output = torch.empty(
        (batch.num_experts, batch.rows, batch.columns),
        dtype=torch.uint8,
        device=batch.fixed.device,
    )
    decode_x4t_scales(
        batch,
        expert_ids,
        output,
        layout="logical",
        clamp_e8m0_bf16=True,
    )
    return output


def _capture(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _interleaved_times_us(
    graphs: dict[str, torch.cuda.CUDAGraph], *, replays: int
) -> dict[str, list[float]]:
    samples = {name: [] for name in graphs}
    names = tuple(graphs)
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for replay in range(replays):
        order = names if replay % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            events.append((name, start, end))
    torch.cuda.synchronize()
    for name, start, end in events:
        samples[name].append(float(start.elapsed_time(end)) * 1_000.0)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    difference = actual_f - reference_f
    reference_norm = float(torch.linalg.vector_norm(reference_f))
    relative_l2 = float(torch.linalg.vector_norm(difference)) / max(
        reference_norm, torch.finfo(torch.float32).tiny
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.flatten(), reference_f.flatten(), dim=0
        )
    )
    return {
        "bit_exact": bool(torch.equal(actual, reference)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def _require_close(name: str, metrics: dict[str, float | bool]) -> None:
    if bool(metrics["bit_exact"]):
        return
    if (
        not bool(
            torch.isfinite(
                torch.tensor(
                    [
                        float(metrics["max_abs"]),
                        float(metrics["relative_l2"]),
                        float(metrics["cosine"]),
                    ]
                )
            ).all()
        )
        or float(metrics["relative_l2"]) > 5.0e-3
        or float(metrics["cosine"]) < 0.99999
    ):
        raise RuntimeError(f"{name} output mismatch: {metrics}")


def _git_revision() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return {"worktree": str(root), "revision": revision, "dirty": dirty}


def _make_routes(
    *,
    tokens: int,
    active_per_token: int,
    local_experts: int,
    route_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if active_per_token > _TOPK:
        raise ValueError("active experts per token cannot exceed top-k")
    required_local = tokens * active_per_token
    if required_local > local_experts:
        raise ValueError(
            f"case requires {required_local} distinct local experts, "
            f"but only {local_experts} were requested"
        )
    required_routes = tokens * _TOPK
    if required_routes > route_experts:
        raise ValueError("route expert space is too small for unique synthetic routes")

    topk_ids = torch.arange(
        required_routes, dtype=torch.int32, device=device
    ).view(tokens, _TOPK)
    expert_map = torch.full(
        (route_experts,), -1, dtype=torch.int32, device=device
    )
    local = 0
    for token in range(tokens):
        begin = token * _TOPK
        expert_map[begin : begin + active_per_token] = torch.arange(
            local,
            local + active_per_token,
            dtype=torch.int32,
            device=device,
        )
        local += active_per_token
    topk_weights = torch.rand(
        (tokens, _TOPK), dtype=torch.float32, device=device
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return topk_ids, topk_weights, expert_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=76)
    parser.add_argument("--route-experts", type=int, default=896)
    parser.add_argument("--tokens", type=_parse_csv_ints, default=(1,))
    parser.add_argument(
        "--active-per-token",
        type=_parse_csv_ints,
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.experts <= 0 or args.route_experts < args.experts:
        raise SystemExit("expert counts must satisfy 0 < experts <= route-experts")
    if args.warmup < 1 or args.replays < 1:
        raise SystemExit("warmup and replays must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")
    torch.manual_seed(args.seed)

    mode_before = nvidia_smi_gpu_mode_snapshot()
    w13_x4t = _make_batch(
        experts=args.experts,
        rows=_W13_ROWS,
        columns=_HIDDEN_SIZE // 32,
        exceptions_per_expert=294,
        device=device,
        exception_task_rows=128,
        exception_row_rotation=_INTERMEDIATE_SIZE,
    )
    w2_x4t = _make_batch(
        experts=args.experts,
        rows=_HIDDEN_SIZE,
        columns=_INTERMEDIATE_SIZE // 32,
        exceptions_per_expert=89,
        device=device,
        exception_task_rows=896,
    )
    logical_w13 = _decode_all_logical(w13_x4t)
    logical_w2 = _decode_all_logical(w2_x4t)

    w13 = torch.randint(
        0,
        256,
        (args.experts, _W13_ROWS, _HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (args.experts, _HIDDEN_SIZE, _INTERMEDIATE_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_global = torch.ones(args.experts, dtype=torch.float32, device=device)
    w2_global = torch.ones(args.experts, dtype=torch.float32, device=device)
    dense = prepare_w4a16_fp4_e8m0_k32_weights(
        w13,
        logical_w13,
        w13_global,
        w2,
        logical_w2,
        w2_global,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    w13_scratch = torch.full(
        (args.experts, _HIDDEN_SIZE // 32, _W13_ROWS),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    w2_scratch = torch.full(
        (args.experts, _INTERMEDIATE_SIZE // 32, _HIDDEN_SIZE),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    x4t = prepare_w4a16_x4t_weights(
        w13,
        w13_x4t,
        w13_global,
        w2,
        w2_x4t,
        w2_global,
        w13_scratch,
        w2_scratch,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    if not torch.equal(dense.w13, x4t.w13) or not torch.equal(dense.w2, x4t.w2):
        raise RuntimeError("dense and X4T arms do not share identical packed weights")

    cases: list[dict[str, object]] = []
    keepalive: list[object] = [
        w13_x4t,
        w2_x4t,
        logical_w13,
        logical_w2,
        w13,
        w2,
        dense,
        x4t,
    ]
    for tokens in args.tokens:
        for active_per_token in args.active_per_token:
            topk_ids, topk_weights, expert_map = _make_routes(
                tokens=tokens,
                active_per_token=active_per_token,
                local_experts=args.experts,
                route_experts=args.route_experts,
                device=device,
            )
            x = (torch.randn(tokens, _HIDDEN_SIZE, device=device) * 0.125).to(
                torch.bfloat16
            )
            dense_buffers = make_w4a16_packed_buffers(
                dense,
                m=tokens,
                topk=_TOPK,
                dtype=torch.bfloat16,
                device=device,
                route_num_experts=args.route_experts,
            )
            x4t_buffers = make_w4a16_packed_buffers(
                x4t,
                m=tokens,
                topk=_TOPK,
                dtype=torch.bfloat16,
                device=device,
                route_num_experts=args.route_experts,
            )

            def launch(prepared, buffers) -> torch.Tensor:
                return run_w4a16_moe(
                    x,
                    prepared,
                    topk_weights,
                    topk_ids,
                    activation="situ",
                    expert_map=expert_map,
                    intermediate_cache13=buffers.intermediate_cache13,
                    intermediate_cache2=buffers.intermediate_cache2,
                    output=buffers.output,
                    fc1_c_tmp=buffers.fc1_c_tmp,
                    fc2_c_tmp=buffers.fc2_c_tmp,
                    packed_route_indices=buffers.packed_route_indices,
                    block_expert_ids=buffers.block_expert_ids,
                    packed_route_count=buffers.packed_route_count,
                    expert_offsets=buffers.expert_offsets,
                )

            def launch_dense() -> torch.Tensor:
                return launch(dense, dense_buffers)

            def launch_x4t() -> torch.Tensor:
                return launch(x4t, x4t_buffers)

            dense_eager = launch_dense().clone()
            dense_repeat = launch_dense().clone()
            x4t_eager = launch_x4t().clone()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(dense_eager).all().item()):
                raise RuntimeError("dense W4A16 output is nonfinite")
            if not bool((dense_eager != 0).any().item()):
                raise RuntimeError("dense W4A16 output is identically zero")
            active_local = expert_map[topk_ids.long()]
            active_local = torch.unique(active_local[active_local >= 0]).long()
            scale_eager_exact = bool(
                torch.equal(
                    x4t.w13_scale[active_local],
                    dense.w13_scale.view(torch.uint8)[active_local],
                )
                and torch.equal(
                    x4t.w2_scale[active_local],
                    dense.w2_scale.view(torch.uint8)[active_local],
                )
            )
            if not scale_eager_exact:
                raise RuntimeError("X4T eager decode did not reproduce packed scales")
            dense_repeat_metrics = _error_metrics(dense_repeat, dense_eager)
            x4t_eager_metrics = _error_metrics(x4t_eager, dense_eager)
            _require_close("dense repeat", dense_repeat_metrics)
            _require_close("X4T eager", x4t_eager_metrics)

            dense_graph = _capture(launch_dense, warmup=args.warmup)
            x4t_graph = _capture(launch_x4t, warmup=args.warmup)
            x4t.w13_scale.fill_(_SENTINEL)
            x4t.w2_scale.fill_(_SENTINEL)
            x4t_buffers.output.fill_(float("nan"))
            x4t_graph.replay()
            torch.cuda.synchronize()
            scale_graph_exact = bool(
                torch.equal(
                    x4t.w13_scale[active_local],
                    dense.w13_scale.view(torch.uint8)[active_local],
                )
                and torch.equal(
                    x4t.w2_scale[active_local],
                    dense.w2_scale.view(torch.uint8)[active_local],
                )
            )
            if not scale_graph_exact:
                raise RuntimeError(
                    "X4T graph replay did not reproduce packed scales after poison"
                )
            x4t_graph_metrics = _error_metrics(x4t_buffers.output, dense_eager)
            _require_close("X4T graph replay", x4t_graph_metrics)

            samples = _interleaved_times_us(
                {"dense": dense_graph, "x4t": x4t_graph},
                replays=args.replays,
            )
            dense_summary = _summary(samples["dense"])
            x4t_summary = _summary(samples["x4t"])
            dense_median = dense_summary["median_us"]
            x4t_median = x4t_summary["median_us"]
            cases.append(
                {
                    "tokens": tokens,
                    "topk": _TOPK,
                    "active_x4t_per_token": active_per_token,
                    "correctness": {
                        "passed": True,
                        "finite": True,
                        "nonzero": True,
                        "eager_scale_bytes_exact": scale_eager_exact,
                        "graph_replay_scale_bytes_exact_after_poison": (
                            scale_graph_exact
                        ),
                        "dense_repeat_output": dense_repeat_metrics,
                        "x4t_eager_output": x4t_eager_metrics,
                        "x4t_graph_replay_output": x4t_graph_metrics,
                    },
                    "timings": {
                        "dense": dense_summary,
                        "x4t": x4t_summary,
                        "x4t_minus_dense_us": x4t_median - dense_median,
                        "x4t_over_dense": x4t_median / dense_median,
                    },
                    "raw_samples_us": samples,
                }
            )
            keepalive.extend(
                [
                    topk_ids,
                    topk_weights,
                    expert_map,
                    x,
                    dense_buffers,
                    x4t_buffers,
                    dense_eager,
                    dense_repeat,
                    x4t_eager,
                    dense_graph,
                    x4t_graph,
                ]
            )

    result = {
        "kind": "x4t_tp12_w4a16_moe_graph_benchmark_v1",
        "command": [sys.executable, *sys.argv],
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": [major, minor],
            "gpu_mode_before": mode_before,
            "gpu_mode_after": nvidia_smi_gpu_mode_snapshot(),
            **_git_revision(),
        },
        "shape": {
            "local_experts": args.experts,
            "route_experts": args.route_experts,
            "hidden_size": _HIDDEN_SIZE,
            "local_intermediate_size": _INTERMEDIATE_SIZE,
            "w13_exceptions_per_expert": 294,
            "w2_exceptions_per_expert": 89,
        },
        "contract": {
            "graph_replay": True,
            "balanced_order": True,
            "bf16_activations": True,
            "exact_fp4_nibbles": True,
            "x4t_launches_before_w4a16": 1,
            "reusable_scale_scratch_bytes": int(
                x4t.w13_scale.numel() + x4t.w2_scale.numel()
            ),
        },
        "cases": cases,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
