"""Compare native W4A8 and W4A16 SQG/MUL1 trellis execution at TP12.

Both execution paths consume the same compact P24/P33 expert payload, route
table, activations, transforms, and expert-static mode tables.  The W4A8 arm
includes input rotation, MXFP8 activation quantization, direct E4M3 decode
into MMA registers, SiTU, FC2, inverse rotation, and top-k reduction.  The
W4A16 arm is the production BF16-activation numerical control.  Every timed
launch is CUDA-graph replayed.

This benchmark deliberately excludes MCG because MCG does not reconstruct an
E4M3 operand and therefore has no native W4A8 execution contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import sys

import torch
from cuda.bindings import runtime as cudart

try:
    from benchmarks.benchmark_trellis_pair_moe_tp12 import (
        _HIDDEN,
        _LOCAL_INTERMEDIATE,
        _ROUTE_EXPERTS,
        _TOPK,
        _RouteFixture,
        _load_route_fixture,
        _paired_ratio_bootstrap,
        _parse_positive_list,
        _prepare_weights,
        _routing,
        _write_new_result,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from benchmark_trellis_pair_moe_tp12 import (
        _HIDDEN,
        _LOCAL_INTERMEDIATE,
        _ROUTE_EXPERTS,
        _TOPK,
        _RouteFixture,
        _load_route_fixture,
        _paired_ratio_bootstrap,
        _parse_positive_list,
        _prepare_weights,
        _routing,
        _write_new_result,
    )

from b12x.moe import fused_moe
from b12x.moe._shared.kernels.trellis_w4a8 import (
    make_trellis_w4a8_moe_scratch,
    run_trellis_w4a8_moe,
)


_SCENARIOS = ("P33", "P24", "sparse", "mixed")
_RESULT_KIND = "b12x_trellis_w4a8_vs_w4a16_moe_tp12_benchmark"
_RESULT_SCHEMA_VERSION = 2


def _parse_scenarios(value: str) -> list[str]:
    aliases = {item.lower(): item for item in _SCENARIOS}
    result: list[str] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(
                f"scenario must be one of {', '.join(_SCENARIOS)}, got {raw!r}"
            )
        canonical = aliases[key]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return result


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _graph_structure(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    graph_handle = graph.raw_cuda_graph()
    result, _, node_count = cudart.cudaGraphGetNodes(graph_handle)
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes(size) failed: {result}")
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        node_count,
    )
    if result != cudart.cudaError_t.cudaSuccess or returned_nodes != node_count:
        raise RuntimeError(
            f"cudaGraphGetNodes(data) failed: {result}, "
            f"{returned_nodes=}, {node_count=}"
        )
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_count = 0
    node_types: dict[str, int] = {}
    for node in nodes[:node_count]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphNodeGetType failed: {result}")
        kernel_count += node_type == kernel_type
        node_type_name = str(node_type)
        node_types[node_type_name] = node_types.get(node_type_name, 0) + 1
    return {
        "node_count": int(node_count),
        "kernel_node_count": int(kernel_count),
        "node_types": node_types,
    }


def _time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph], *, replays: int
) -> dict[str, list[float]]:
    samples = {name: [] for name in graphs}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    names = tuple(graphs)
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
        samples[name].append(start.elapsed_time(end) * 1000.0)
    return samples


def _numerical_comparison(w4a8: torch.Tensor, w4a16: torch.Tensor) -> dict[str, float]:
    reference = w4a16.float()
    candidate = w4a8.float()
    difference = candidate - reference
    reference_energy = float(reference.square().sum())
    difference_energy = float(difference.square().sum())
    denominator = max(reference_energy, torch.finfo(torch.float32).tiny)
    cosine_denominator = max(
        float(torch.linalg.vector_norm(reference))
        * float(torch.linalg.vector_norm(candidate)),
        torch.finfo(torch.float32).tiny,
    )
    return {
        "nmse_vs_w4a16": difference_energy / denominator,
        "rmse": float(difference.square().mean().sqrt()),
        "mean_abs": float(difference.abs().mean()),
        "max_abs": float(difference.abs().max()),
        "cosine_similarity": float((reference * candidate).sum()) / cosine_denominator,
    }


def _capture_case(
    scenario: str,
    tokens: int,
    experts: int,
    *,
    codebook: str,
    device: torch.device,
    seed: int,
    warmup: int,
    sparse_p24_experts: int,
    route_fixture: _RouteFixture | None,
) -> tuple[dict[str, torch.cuda.CUDAGraph], list[object], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = _prepare_weights(
        scenario,
        experts,
        codebook=codebook,
        device=device,
        generator=generator,
        sparse_p24_experts=sparse_p24_experts,
        route_fixture=route_fixture,
    )
    topk_ids, topk_weights, expert_map = _routing(
        tokens, experts, device=device, route_fixture=route_fixture
    )
    source = (
        torch.randn(
            (tokens, _HIDDEN),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 1.0e-3
    ).to(torch.bfloat16)

    w4a16_plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=tokens,
            num_topk=_TOPK,
            route_num_experts=_ROUTE_EXPERTS,
            device=device,
            weight_plan=weights.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
        )
    )
    scratch_spec = w4a16_plan.scratch_specs()[0]
    w4a16_scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = fused_moe.bind(
        w4a16_plan,
        scratch=w4a16_scratch,
        a=source,
        experts=weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        route_expert_map=expert_map,
        output_expert_map=expert_map,
    )
    prepared = weights.representation_for("w4a16")
    w4a8_scratch = make_trellis_w4a8_moe_scratch(
        m=tokens,
        topk=_TOPK,
        hidden_size=_HIDDEN,
        intermediate_size=_LOCAL_INTERMEDIATE,
        device=device,
    )

    w4a16_eager = binding.run().clone()
    w4a8_eager = run_trellis_w4a8_moe(
        source,
        prepared,
        topk_weights,
        topk_ids,
        w4a8_scratch,
        expert_map=expert_map,
        fast_math=True,
    ).clone()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(w4a16_eager).all()) or not bool(
        torch.isfinite(w4a8_eager).all()
    ):
        raise RuntimeError(f"{scenario} produced non-finite output")

    graph_w4a16 = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph_w4a16):
        w4a16_captured = binding.run()
    graph_w4a8 = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph_w4a8):
        w4a8_captured = run_trellis_w4a8_moe(
            source,
            prepared,
            topk_weights,
            topk_ids,
            w4a8_scratch,
            expert_map=expert_map,
            fast_math=True,
        )
    for _ in range(warmup):
        graph_w4a16.replay()
        graph_w4a8.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(w4a16_captured, w4a16_eager):
        raise RuntimeError(f"{scenario} W4A16 graph replay differs from eager")
    if not torch.equal(w4a8_captured, w4a8_eager):
        raise RuntimeError(f"{scenario} W4A8 graph replay differs from eager")

    comparison = _numerical_comparison(w4a8_eager, w4a16_eager)
    comparison.update(
        {
            "w4a8_finite": True,
            "w4a16_finite": True,
            "w4a8_eager_graph_bit_exact": True,
            "w4a16_eager_graph_bit_exact": True,
            "reference_scope": (
                "same trellis payload and transforms; W4A16 retains BF16 "
                "activations while W4A8 adds MXFP8 activation quantization"
            ),
        }
    )
    keepalive: list[object] = [
        weights,
        topk_ids,
        topk_weights,
        expert_map,
        source,
        w4a16_plan,
        w4a16_scratch,
        binding,
        prepared,
        w4a8_scratch,
        w4a16_eager,
        w4a8_eager,
        w4a16_captured,
        w4a8_captured,
    ]
    return (
        {"w4a16": graph_w4a16, "w4a8": graph_w4a8},
        keepalive,
        comparison,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        choices=("sqg_xor_cheb_t12",),
        default="sqg_xor_cheb_t12",
        help="trellis reconstruction codebook",
    )
    parser.add_argument(
        "--tokens", type=_parse_positive_list, default=_parse_positive_list("1,2,4")
    )
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--route-fixture", type=Path)
    parser.add_argument("--sparse-p24-experts", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=_parse_scenarios("P33,P24,sparse"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    route_fixture = (
        _load_route_fixture(args.route_fixture)
        if args.route_fixture is not None
        else None
    )
    if route_fixture is not None:
        if args.experts is not None and args.experts != route_fixture.num_experts:
            raise SystemExit(
                f"experts={args.experts} disagrees with fixture "
                f"experts={route_fixture.num_experts}"
            )
        experts = route_fixture.num_experts
    else:
        experts = 32 if args.experts is None else args.experts
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")
    if not _TOPK <= experts <= _ROUTE_EXPERTS:
        raise SystemExit(
            f"experts must be in [{_TOPK}, {_ROUTE_EXPERTS}], got {experts}"
        )
    if not 1 <= args.sparse_p24_experts <= experts:
        raise SystemExit("sparse-p24-experts must be in [1, experts]")
    if min(args.tokens) < 1 or max(args.tokens) > 4:
        raise SystemExit("the current native W4A8 decode benchmark supports M=1..4")
    if args.warmup < 1 or args.replays < 1 or args.bootstrap_replicates < 1:
        raise SystemExit("warmup, replays, and bootstrap-replicates must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    result: dict[str, object] = {
        "kind": _RESULT_KIND,
        "schema_version": _RESULT_SCHEMA_VERSION,
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "worktree": str(Path(__file__).resolve().parents[1]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "route_fixture": (
                None if route_fixture is None else str(route_fixture.path)
            ),
        },
        "contract": {
            "tp": 12,
            "hidden_size": _HIDDEN,
            "local_intermediate_size": _LOCAL_INTERMEDIATE,
            "route_experts": _ROUTE_EXPERTS,
            "topk": _TOPK,
            "materialized_experts": experts,
            "codebook": args.codebook,
            "scenarios": args.scenarios,
            "tokens": args.tokens,
            "cuda_graph_replay": True,
            "timed_scope": "complete routed MoE path",
            "w4a16_activation": "BF16",
            "w4a8_activation": "MXFP8 E4M3 with block scales",
        },
        "device": {
            "name": properties.name,
            "capability": [major, minor],
            "torch": torch.__version__,
        },
        "cases": [],
    }
    cases = result["cases"]
    assert isinstance(cases, list)
    keepalive: list[object] = []
    for tokens in args.tokens:
        for scenario_index, scenario in enumerate(args.scenarios):
            graphs, owned, comparison = _capture_case(
                scenario,
                tokens,
                experts,
                codebook=args.codebook,
                device=device,
                seed=args.seed + tokens * 100 + scenario_index,
                warmup=args.warmup,
                sparse_p24_experts=args.sparse_p24_experts,
                route_fixture=route_fixture,
            )
            keepalive.extend(owned)
            raw = _time_graphs(graphs, replays=args.replays)
            summaries = {name: _summary(samples) for name, samples in raw.items()}
            ratio = _paired_ratio_bootstrap(
                raw["w4a8"],
                raw["w4a16"],
                replicates=args.bootstrap_replicates,
                seed=args.seed + 10_000 * tokens + scenario_index,
            )
            graph_structure = {
                name: _graph_structure(graph) for name, graph in graphs.items()
            }
            cases.append(
                {
                    "tokens": tokens,
                    "scenario": scenario,
                    "correctness": comparison,
                    "timings": summaries,
                    "raw_us": raw,
                    "w4a8_over_w4a16": ratio,
                    "graph_structure": graph_structure,
                }
            )

    del keepalive
    result["device"]["memory_allocated_after"] = int(
        torch.cuda.memory_allocated(device)
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        _write_new_result(args.output, payload)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
