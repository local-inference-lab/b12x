"""Qualify exact scale/nibble loading and real routed MoE graph execution."""

import argparse
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from b12x._lib.quant.x4t_packed_scales import decode_x4t_packed_scales
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe
from b12x.moe.checkpoints.exact_mxfp4 import checkpoint_contract, read_exact_mxfp4_layer
from tests._reference.helpers import make_tp_moe_fp4_binding


def error(a, b):
    assert torch.isfinite(a).all() and a.abs().any()
    return float((a.float() - b.float()).norm() / b.float().norm())


def graph_microseconds(function, replays):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(3):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(replays):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / replays)
    graph.reset()
    return {"samples_us": samples, "median_us": statistics.median(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--tokens", default="1,8,129,4096")
    parser.add_argument("--native-default-packing", action="store_true")
    parser.add_argument("--x4t-default-packing", action="store_true")
    parser.add_argument("--graph-mutations", type=int, default=4)
    parser.add_argument("--routing-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--timing-replays", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2047)
    family = checkpoint_contract(str(args.checkpoint.resolve()))["family"]
    kimi = family == "kimi_k3"
    e, h, intermediate, topk = (896, 3584, 3072, 16) if kimi else (384, 5120, 2304, 6)
    n = intermediate // args.tp
    device = torch.device("cuda:0")
    scratch = (
        torch.empty((e, h // 32, 2 * n), dtype=torch.uint8, device=device),
        torch.empty((e, n // 32, h), dtype=torch.uint8, device=device),
    )
    weights = read_exact_mxfp4_layer(
        args.checkpoint,
        args.layer,
        num_experts=e,
        hidden_size=h,
        intermediate_size=intermediate,
        tp_rank=args.rank,
        tp_size=args.tp,
        device=device,
        w13_scale_scratch=scratch[0],
        w2_scale_scratch=scratch[1],
    )
    print("compressed layer loaded", flush=True)
    original = args.original
    index = json.loads((original / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    scales13 = torch.empty((e, 2 * n, h // 32), dtype=torch.uint8, device="cpu")
    scales2 = torch.empty((e, h, n // 32), dtype=torch.uint8, device="cpu")
    first, last = args.rank * n, (args.rank + 1) * n
    with ExitStack() as stack:
        handles = {}
        for expert in range(e):
            for i, projection in enumerate(("w1", "w3", "w2")):
                stem = (
                    f"language_model.model.layers.{args.layer}.block_sparse_moe."
                    f"experts.{expert}.{projection}"
                    if kimi else f"layers.{args.layer}.ffn.experts.{expert}.{projection}"
                )
                scale_suffix = ".weight_scale" if kimi else ".scale"
                weight_suffix = ".weight_packed" if kimi else ".weight"
                for suffix in (scale_suffix, weight_suffix):
                    name = stem + suffix
                    filename = index[name]
                    if filename not in handles:
                        handles[filename] = stack.enter_context(
                            safe_open(original / filename, framework="pt", device="cpu")
                        )
                    value = handles[filename].get_tensor(name).view(torch.uint8)
                    if i < 2:
                        value = value[first:last]
                        if suffix == scale_suffix:
                            scales13[expert, i * n : (i + 1) * n].copy_(value)
                        else:
                            assert torch.equal(
                                value, weights.w13[expert, i * n : (i + 1) * n].cpu()
                            )
                    else:
                        divisor = 32 if suffix == scale_suffix else 2
                        value = value[:, first // divisor : last // divisor]
                        if suffix == scale_suffix:
                            scales2[expert].copy_(value)
                        else:
                            assert torch.equal(value, weights.w2[expert].cpu())
    plan_args = dict(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="situ" if kimi else "silu",
            io_dtype=torch.bfloat16, swiglu_limit=None if kimi else 10.0
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=h, intermediate_size=n
        ),
    )
    packed = fused_moe.WeightPlanConstraints(required_packing="mma_packed")
    native_plan = fused_moe.plan_weights(
        **plan_args, constraints=None if args.native_default_packing else packed
    )
    candidate_plan = fused_moe.plan_weights(
        **plan_args, constraints=None if args.x4t_default_packing else packed
    )
    unit = torch.ones(e, dtype=torch.float32, device=device)
    native = fused_moe.prepare_weights(
        plan=native_plan,
        weights=fused_moe.PackedWeights(
            w13=weights.w13.clone(),
            w2=weights.w2.clone(),
            w13_block_scales=scales13.to(device),
            w2_block_scales=scales2.to(device),
            w13_global_scales=unit,
            w2_global_scales=unit,
        ),
    )
    candidate = fused_moe.prepare_weights(plan=candidate_plan, weights=weights)
    raw_native = native._impl.representation.value
    raw_candidate = candidate._impl.representation.value
    same_packing = (
        native_plan.prepared_format.packing == candidate_plan.prepared_format.packing
    )
    if same_packing:
        assert torch.equal(raw_native.w13, raw_candidate.w13)
        assert torch.equal(raw_native.w2, raw_candidate.w2)
    all_ids = torch.arange(e, dtype=torch.int32, device=device)
    for batch, target, source in (
        (weights.w13_scales, scratch[0], raw_native.w13_scale),
        (weights.w2_scales, scratch[1], raw_native.w2_scale),
    ):
        decode_x4t_packed_scales(batch, all_ids, target, expert_ids_unique=True)
        assert torch.equal(target, source.view(torch.uint8))
    print("all expert nibbles and scales exact", flush=True)
    del weights
    cases = []
    for tokens in map(int, args.tokens.split(",")):
        x = torch.randn((tokens, h), dtype=torch.bfloat16, device=device) * 0.5
        ids = torch.randint(
            e, (tokens, topk), dtype=getattr(torch, args.routing_dtype), device=device
        )
        route_weights = torch.softmax(torch.randn(tokens, topk, device=device), -1)
        with ExitStack() as stack:
            bindings = [
                stack.enter_context(
                    make_tp_moe_fp4_binding(
                        a=x,
                        experts=p,
                        topk_weights=route_weights,
                        topk_ids=ids,
                        output=torch.empty_like(x),
                        quant_mode="w4a16",
                    )
                )
                for p in (native, candidate)
            ]
            b, c = bindings
            expected = b.run().clone()
            repeat = b.run().clone()
            actual = c.run().clone()
            baseline_error, actual_error = (
                error(repeat, expected),
                error(actual, expected),
            )
            assert actual_error < 0.005, actual_error
            if same_packing and tokens <= 129:
                assert torch.equal(actual, expected)
            elif same_packing:
                assert actual_error < 1e-4
            timings = None
            if args.timing_replays:
                with kernel_resolution_guard("X4T serving layer timing"):
                    timings = {
                        name: graph_microseconds(binding.run, args.timing_replays)
                        for name, binding in (("native", b), ("x4t", c))
                    }
            with kernel_resolution_guard("X4T serving layer graph"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = c.run()
                discrepancy = None
                for _ in range(args.graph_mutations):
                    ids.random_(0, e)
                    ids[0, 0] = -1
                    ids[0, 1] = ids[0, 2]
                    x.mul_(-0.9)
                    expected = b.run().clone()
                    for t in scratch:
                        t.fill_(0xD6)
                    graph.replay()
                    torch.cuda.synchronize()
                    discrepancy = error(result, expected)
                    assert discrepancy < 0.005, discrepancy
                    if same_packing and tokens <= 129:
                        assert torch.equal(result, expected)
                    elif same_packing:
                        assert discrepancy < 1e-4
            graph.reset()
        record = {
            "tokens": tokens,
            "native_repeat_relative_l2": baseline_error,
            "x4t_relative_l2": actual_error,
            "poisoned_graph_last_relative_l2": discrepancy,
        }
        if timings is not None:
            record["graph_timing"] = timings
        cases.append(record)
        print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "qualified" if args.graph_mutations >= 4 else "research-only",
                "graph_mutations": args.graph_mutations,
                "routing_dtype": args.routing_dtype,
                "layer": args.layer,
                "family": family,
                "tp": args.tp,
                "rank": args.rank,
                "native_packing": native_plan.prepared_format.packing.value,
                "x4t_packing": candidate_plan.prepared_format.packing.value,
                "nibbles_and_scales_exact": True,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
