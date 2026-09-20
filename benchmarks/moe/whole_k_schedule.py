"""Physical SM120 native split-K, whole-K and all-resident cache diagnostics.

Uses real checkpoint bytes, randomized activations/routes and actual native
prepared operations. This is operator evidence, not serving or quality evidence.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.compile_plan import observe_programs
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from benchmarks.moe.sm120_residency_poc import load_layer, make_tier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--layer", default="model.layers.12.mlp.experts")
    p.add_argument("--top-k", type=int, nargs="+", default=[2, 8])
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--check-shape-invariance",
        action="store_true",
        help="Require identical logical rows under changed batch packing",
    )
    p.add_argument(
        "--logical-route-trace",
        type=Path,
        help="Use a recorded serving activation, IDs and actual gate weights as row zero",
    )
    p.add_argument("--trace-step", type=int, default=130)
    p.add_argument("--trace-row", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh evidence path")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("physical SM120 required")
    torch.manual_seed(42)
    config = json.loads((args.checkpoint / "config.json").read_text())
    e = config["num_experts"]
    fields, identity = load_layer(args.checkpoint, args.layer, e)
    h, i = fields["w2"].shape[1], fields["w13"].shape[1] // 2
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"),
    )

    def logical(value):
        _, n, k = value.shape
        return (
            value.reshape(e, n // 128, k // 4, 32, 4, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .reshape(e, n, k)
            .view(torch.float8_e4m3fn)
        )

    source = moe.ExpertWeightSource(
        plan=weight_plan,
        weights=moe.PackedWeights(
            w13=fields["w13"],
            w2=fields["w2"],
            w13_block_scales=logical(fields["s13"]),
            w2_block_scales=logical(fields["s2"]),
            w13_global_scales=fields["g13"].view(torch.float32).reshape(e),
            w2_global_scales=fields["g2"].view(torch.float32).reshape(e),
            checkpoint_fingerprint=identity,
            layer_name=args.layer,
        ),
    )
    for expert in range(e):
        assert all(
            torch.equal(value, fields[name][expert])
            for name, value in source.row(expert).items()
        )
    tier = make_tier(fields, range(e), torch.device("cuda", 0), mapped=False)
    f = tier.fields
    experts = moe.prepare_weights(
        plan=weight_plan,
        weights=moe.PackedWeights(
            w13=f["w13"],
            w2=f["w2"],
            w13_block_scales=f["s13"],
            w2_block_scales=f["s2"],
            w13_global_scales=f["g13"].view(torch.float32).reshape(e),
            w2_global_scales=f["g2"].view(torch.float32).reshape(e),
        ),
    )
    result = {
        "source_receipt": os.environ.get("B12X_SERVING_SOURCE_RECEIPT"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_layer_sha256": identity,
        "geometry": [e, h, i],
        "gpu": subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True),
        "rows": [],
    }
    for top_k in args.top_k:
        a = torch.randn((128, h), device="cuda", dtype=torch.bfloat16) * 0.125
        ids = torch.randint(e, (128, top_k), device="cuda", dtype=torch.int32)
        weights = torch.softmax(torch.randn((128, top_k), device="cuda"), dim=1)
        serving_output = None
        if args.logical_route_trace:
            trace = next(
                r
                for r in torch.load(args.logical_route_trace, weights_only=True)
                if r["step"] == args.trace_step
            )["modules"][args.layer + "/routes"]
            for destination, key in ((a, "input"), (ids, "ids"), (weights, "weights")):
                destination[0].copy_(trace[key][args.trace_row])
            serving_output = trace["output"][args.trace_row].cuda()
        bindings, plans, requests = {}, {}, []
        for m in (1, 2, 4, 8, 16, 32, 64, 128):
            for name in ("preferred", "whole_k", "cache"):
                cap = moe.ExecutionCapacity(max_tokens=m, top_k=top_k)
                if name == "cache":
                    plan = moe.plan_execution(
                        experts=source,
                        capacity=cap,
                        placement=moe.ExpertResidencyPlan(
                            total_experts=e,
                            hbm_expert_ids=tuple(range(e)),
                            grace_expert_ids=(),
                            layer=args.layer,
                            model_fingerprint=identity,
                            workload="schedule",
                            provenance="all-resident compute control",
                        ),
                        memory_budget=moe.ExpertMemoryBudget(
                            hbm_bytes=2 << 30, grace_bytes=2 << 30
                        ),
                    )
                else:
                    plan = moe.plan_execution(
                        experts=experts,
                        capacity=cap,
                        routing=moe.RoutingSpec(deterministic_output=name == "whole_k"),
                    )
                plans[m, name] = plan

                def call(state, m=m, name=name):
                    kwargs = dict(a=a[:m], topk_ids=ids[:m], topk_weights=weights[:m])
                    if name != "cache":
                        kwargs.update(
                            scratch=tuple(
                                torch.empty(s.shape, dtype=s.dtype, device=s.device)
                                for s in state.scratch.scratch_specs()
                            ),
                            output=torch.empty_like(a[:m]),
                        )
                    binding = state.bind(**kwargs)
                    bindings[m, name] = binding
                    return PreparedCall(
                        run=binding.run,
                        output=binding.output,
                        owners=(binding,),
                        close=state.close if name == "cache" else None,
                    )

                requests.append(plan.request(name=f"{m}:{name}", prepare_call=call))
        with PreparationSession(autotune=False, compile_workers=0) as session:
            session.prepare(tuple(requests))
            graphs, captured_programs = {}, {}
            for key, binding in bindings.items():
                graph = torch.cuda.CUDAGraph()
                with (
                    observe_programs() as observed,
                    session.capture(),
                    torch.cuda.graph(graph),
                ):
                    binding.run()
                captured_programs[key] = tuple(
                    sorted(observed, key=lambda p: (p.dialect, p.key))
                )
                graphs[key] = graph
                graph.replay()
            torch.cuda.synchronize()
            session.freeze()
            if serving_output is not None:
                torch.testing.assert_close(
                    bindings[1, "whole_k"].output[0], serving_output, atol=0, rtol=0
                )
                result["serving_route_exact"] = True
            if args.check_shape_invariance:
                # Keep top-k rank and weights fixed for each logical row while
                # changing unrelated routes and expert-block occupancy.
                originals = (a.clone(), ids.clone(), weights.clone())
                oracle = bindings[1, "whole_k"].output[0].clone()
                checks = []
                for layout in ("unrelated", "duplicates", "reversed"):
                    if layout == "duplicates":
                        for tensor, original in zip(
                            (a, ids, weights), originals, strict=True
                        ):
                            tensor.copy_(original[:1].expand_as(tensor))
                    elif layout == "reversed":
                        for tensor, original in zip(
                            (a, ids, weights), originals, strict=True
                        ):
                            tensor.copy_(original.flip(0))
                    for m in (1, 2, 4, 8, 16, 32, 64, 128):
                        if layout == "reversed":
                            for tensor, original in zip(
                                (a, ids, weights), originals, strict=True
                            ):
                                tensor[:m].copy_(original[:m].flip(0))
                        row = m - 1 if layout == "reversed" else 0
                        for name in ("whole_k", "cache"):
                            with kernel_resolution_guard("shape invariance"):
                                graphs[m, name].replay()
                            observed = bindings[m, name].output[row]
                            check = dict(
                                layout=layout,
                                m=m,
                                backend=name,
                                exact=torch.equal(observed, oracle),
                                max_abs=(observed.float() - oracle.float())
                                .abs()
                                .max()
                                .item(),
                            )
                            checks.append(check)
                result.setdefault("shape_invariance", []).append(
                    dict(top_k=top_k, checks=checks)
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if not all(check["exact"] for check in checks):
                    raise AssertionError(
                        "whole-K logical row changes under batch packing"
                    )
                for tensor, original in zip((a, ids, weights), originals, strict=True):
                    tensor.copy_(original)
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
            for m in (1, 2, 4, 8, 16, 32, 64, 128):
                whole = bindings[m, "whole_k"].output
                split = bindings[m, "preferred"].output
                torch.testing.assert_close(
                    bindings[m, "cache"].output, whole, atol=0, rtol=0
                )
                assert torch.isfinite(split).all() and torch.count_nonzero(split)
                assert torch.isfinite(whole).all() and torch.count_nonzero(whole)
                cosine = torch.nn.functional.cosine_similarity(
                    whole.float().flatten(), split.float().flatten(), dim=0
                ).item()
                if cosine < 0.999:
                    raise AssertionError(
                        f"native scheduling cosine gate failed: {cosine}"
                    )
                samples = {n: [] for n in ("preferred", "whole_k", "cache")}
                for sample in range(args.samples):
                    names = list(samples)
                    names = names[sample % 3 :] + names[: sample % 3]
                    if sample % 2:
                        names.reverse()
                    for name in names:
                        graph = graphs[m, name]
                        for _ in range(10):
                            graph.replay()
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        before = torch.cuda.memory_stats()["allocation.all.allocated"]
                        with kernel_resolution_guard("whole-K comparison"):
                            start.record()
                            for _ in range(args.replays):
                                graph.replay()
                            end.record()
                        end.synchronize()
                        assert (
                            torch.cuda.memory_stats()["allocation.all.allocated"]
                            == before
                        )
                        samples[name].append(
                            start.elapsed_time(end) * 1000 / args.replays
                        )
                result["rows"].append(
                    {
                        "m": m,
                        "top_k": top_k,
                        "samples_us": samples,
                        "max_abs_difference": (whole.float() - split.float())
                        .abs()
                        .max()
                        .item(),
                        "different_elements": torch.count_nonzero(
                            whole != split
                        ).item(),
                        "cosine": cosine,
                        "configs": {
                            n: asdict(plans[m, n].prepared.state.config)
                            for n in ("preferred", "whole_k")
                        },
                        "programs": {
                            n: [asdict(p) for p in captured_programs[m, n]]
                            for n in samples
                        },
                        "prepared_programs": {
                            n: [
                                asdict(program)
                                for program in sorted(
                                    plans[m, n].prepared.programs,
                                    key=lambda x: (x.dialect, x.key),
                                )
                            ]
                            for n in samples
                        },
                        "cache_exact_whole_k": True,
                    }
                )
            for graph in graphs.values():
                graph.reset()
        # Preserve completed top-k groups if a later qualification fails.
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
