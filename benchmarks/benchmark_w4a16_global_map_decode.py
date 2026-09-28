#!/usr/bin/env python3
"""Benchmark in-kernel global expert mapping for Kimi K3 MXFP4 decode.

The default shape matches one K3 TP12 kept-expert launch:

* M=1, H=3584, I=3072/TP12=256
* top-k=16 over 896 global experts
* packed native MXFP4 (E8M0 K/32) weights

The legacy total includes vLLM's global-to-local indexing, inactive-route
masking, clamp/cast/contiguous chain, and the ordinary W4A16 TC-decode launch.
The mapped path passes the original global routes and map directly to the same
W4A16 kernel family, where invalid routes are skipped before weight access.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from b12x.moe import fused_moe
from b12x.preparation import PreparationSession
from benchmarks.moe_preparation import prepared_call, request_for_capacity, scratch_for


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _capture(fn, *, warmup: int, session: PreparationSession) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with session.capture():
        with torch.cuda.graph(graph):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0 / iterations)
    return samples


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    relative_l2 = float(diff.norm() / reference_f.norm().clamp_min(1.0e-12))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.flatten(),
            reference_f.flatten(),
            dim=0,
        )
    )
    return float(diff.abs().max()), relative_l2, cosine


def main() -> None:
    args = _parse_args()
    if args.experts < 8:
        raise ValueError("--experts must be at least 8")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        raise ValueError("iterations, rounds, and warmup must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        raise RuntimeError(f"benchmark requires SM12x, got SM{major}{minor}")

    local_experts = args.experts
    global_experts = 896
    hidden_size = 3_584
    intermediate_size = 256
    topk = 16
    activation = "situ"
    e8m0_dtype = torch.float8_e8m0fnu

    w1 = torch.randint(
        0,
        256,
        (
            local_experts,
            2 * intermediate_size,
            hidden_size // 2,
        ),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (
            local_experts,
            hidden_size,
            intermediate_size // 2,
        ),
        dtype=torch.uint8,
        device=device,
    )
    w1_scale = torch.full(
        (
            local_experts,
            2 * intermediate_size,
            hidden_size // 32,
        ),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(e8m0_dtype)
    w2_scale = torch.full(
        (
            local_experts,
            hidden_size,
            intermediate_size // 32,
        ),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(e8m0_dtype)
    unit_scales = torch.ones(local_experts, dtype=torch.float32, device=device)
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W13,
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity=activation,
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=w1,
            w2=w2,
            w13_block_scales=w1_scale,
            w2_block_scales=w2_scale,
            w13_global_scales=unit_scales,
            w2_global_scales=unit_scales,
            input_scale=unit_scales,
            intermediate_scale=unit_scales,
        ),
    )

    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8,
            top_k=topk,
            warmup_token_counts=(1, 8),
            route_num_experts=global_experts,
        ),
    )
    x = (torch.randn((1, hidden_size), device=device) * 0.01).to(torch.bfloat16)
    topk_ids = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk)
    topk_weights = torch.rand((1, topk), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[0:topk:2] = torch.arange(topk // 2, dtype=torch.int32, device=device)
    remaining = local_experts - topk // 2
    if remaining:
        expert_map[128 : 128 + remaining] = torch.arange(
            topk // 2,
            local_experts,
            dtype=torch.int32,
            device=device,
        )
    prepared_inputs = {
        1: (x, topk_ids, topk_weights, torch.empty_like(x)),
        8: (
            (torch.randn((8, hidden_size), device=device) * 0.01).to(torch.bfloat16),
            topk_ids.expand(8, -1).contiguous(),
            topk_weights.expand(8, -1).contiguous(),
            torch.empty((8, hidden_size), device=device, dtype=torch.bfloat16),
        ),
    }
    request = request_for_capacity(
        declaration,
        name="w4a16-global-map",
        calls={
            count: prepared_call(
                output=prepared_inputs[count][3],
                bind=lambda state, scratch, count=count: state.bind(
                    scratch=scratch,
                    a=prepared_inputs[count][0],
                    experts=experts,
                    topk_weights=prepared_inputs[count][2],
                    topk_ids=prepared_inputs[count][1],
                    output=prepared_inputs[count][3],
                    input_scales_static=True,
                    route_expert_map=expert_map,
                ),
            )
            for count in getattr(declaration, "token_counts", (1,))
        },
    )
    session = PreparationSession(device=device)
    result = session.prepare((request,))
    variants = getattr(declaration, "variants", None)
    plan = declaration if variants is None else variants.get(1, declaration)
    mapped_output = prepared_inputs[1][3]

    mapped_binding = fused_moe.bind(
        plan,
        scratch=scratch_for(plan),
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=mapped_output,
        input_scales_static=True,
        route_expert_map=expert_map,
    )
    local_ids = expert_map[topk_ids.long()]
    active = local_ids >= 0
    local_weights = topk_weights.masked_fill(~active, 0.0).contiguous()
    local_ids = local_ids.clamp_min(0).to(torch.int32).contiguous()
    legacy_kernel_output = torch.empty_like(x)
    legacy_kernel_binding = fused_moe.bind(
        plan,
        scratch=scratch_for(plan),
        a=x,
        experts=experts,
        topk_weights=local_weights,
        topk_ids=local_ids,
        output=legacy_kernel_output,
        input_scales_static=True,
    )
    legacy_total_output = torch.empty_like(x)

    def mapped_run() -> torch.Tensor:
        return fused_moe.run(binding=mapped_binding)

    def legacy_kernel_run() -> torch.Tensor:
        return fused_moe.run(binding=legacy_kernel_binding)

    def legacy_total_run() -> torch.Tensor:
        kept_ids = expert_map[topk_ids.long()]
        kept_active = kept_ids >= 0
        kept_weights = topk_weights.masked_fill(~kept_active, 0.0).contiguous()
        kept_ids = kept_ids.clamp_min(0).to(torch.int32).contiguous()
        binding = fused_moe.bind(
            plan,
            scratch=scratch_for(plan),
            a=x,
            experts=experts,
            topk_weights=kept_weights,
            topk_ids=kept_ids,
            output=legacy_total_output,
            input_scales_static=True,
        )
        return fused_moe.run(binding=binding)

    legacy_reference = legacy_kernel_run().clone()
    mapped = mapped_run().clone()
    legacy_total = legacy_total_run().clone()
    torch.cuda.synchronize()
    mapped_error = _error_metrics(mapped, legacy_reference)
    legacy_total_error = _error_metrics(legacy_total, legacy_reference)
    for name, (max_abs, relative_l2, cosine) in (
        ("mapped", mapped_error),
        ("legacy_total", legacy_total_error),
    ):
        if (
            not torch.isfinite(torch.tensor((max_abs, relative_l2, cosine))).all()
            or relative_l2 > 5.0e-3
            or cosine < 0.99999
        ):
            raise RuntimeError(
                f"{name} output mismatch: max_abs={max_abs}, "
                f"relative_l2={relative_l2}, cosine={cosine}"
            )

    graphs = {
        "legacy_kernel": _capture(
            legacy_kernel_run, warmup=args.warmup, session=session
        ),
        "legacy_total": _capture(
            legacy_total_run, warmup=args.warmup, session=session
        ),
        "mapped_kernel": _capture(
            mapped_run, warmup=args.warmup, session=session
        ),
    }
    timings = {
        name: _time_graph(
            graph,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        for name, graph in graphs.items()
    }
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    preprocess_tax = medians["legacy_total"] - medians["legacy_kernel"]
    mapped_kernel_delta = medians["mapped_kernel"] - medians["legacy_kernel"]
    net_win = medians["legacy_total"] - medians["mapped_kernel"]

    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"shape=M1/H{hidden_size}/I{intermediate_size}/"
        f"E{global_experts}/localE{local_experts}/K{topk}"
    )
    print("path                 median_us      min_us")
    for name in ("legacy_kernel", "legacy_total", "mapped_kernel"):
        print(f"{name:<20} {medians[name]:9.3f} {min(timings[name]):11.3f}")
    print(f"legacy_preprocess_tax_us={preprocess_tax:.3f}")
    print(f"mapped_kernel_delta_us={mapped_kernel_delta:+.3f}")
    print(f"net_win_us_per_layer={net_win:.3f}")
    print(f"projected_92_layer_win_ms_per_token={net_win * 92.0 / 1_000.0:.3f}")
    print(
        "mapped_vs_legacy="
        f"max_abs:{mapped_error[0]:.6g},"
        f"relative_l2:{mapped_error[1]:.6g},"
        f"cosine:{mapped_error[2]:.9f}"
    )
    print(
        "legacy_repeat="
        f"max_abs:{legacy_total_error[0]:.6g},"
        f"relative_l2:{legacy_total_error[1]:.6g},"
        f"cosine:{legacy_total_error[2]:.9f}"
    )
    for graph in graphs.values():
        graph.reset()
    result.close()
    session.close()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
