"""Qualify and time prepared SM103 expert residency with stage-level receipts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe.residency import read_profiles
from b12x.preparation import PreparationSession, PreparedCall
from scripts._sm103_source import package_source_sha256, source_identity


def capture(run):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def time_graph(graph, iterations, samples):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000/iterations)
    return {"raw_us": values, "median_us": statistics.median(values)}


def run(args, receipt):
    import cuda.bindings.driver as cuda
    from b12x._lib.platform import probe_platform
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this benchmark requires physical SM103; compilation is not a timing substitute")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    def gpu_snapshot():
        return subprocess.check_output(["nvidia-smi", "-q", "-i", "GPU-" + str(props.uuid).removeprefix("GPU-")], text=True)
    receipt["gpu_before"] = gpu_snapshot()
    receipt.update(device_uuid=str(props.uuid), device_name=props.name, capability=list(torch.cuda.get_device_capability()),
                   driver=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version,pstate,power.limit", "--format=csv,noheader"], text=True),
                   platform=asdict(probe_platform(device)))
    e, h, i = args.experts, args.hidden, args.intermediate
    source = moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31" if args.gate_first else "w13")
    activation = moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=args.swiglu_limit)
    weight_plan = moe.plan_weights(source=source, activation=activation,
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"))
    torch.manual_seed(args.seed)
    if args.weights:
        with args.weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if args.checkpoint_sha256 != digest:
            raise ValueError("checkpoint bundle hash differs from --checkpoint-sha256")
        tensors = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
        weights = moe.PackedWeights(**tensors)
        fingerprint = digest
    else:
        weights = moe.PackedWeights(torch.randint(0, 256, (e, 2*i, h//2), dtype=torch.uint8),
            torch.randint(0, 256, (e, h, i//2), dtype=torch.uint8),
            torch.randint(121, 125, (e, 2*i, h//32), dtype=torch.uint8),
            torch.randint(121, 125, (e, h, i//32), dtype=torch.uint8), torch.ones(e), torch.ones(e))
        fingerprint = f"synthetic-seed:{args.seed}"
    weights = replace(weights, checkpoint_fingerprint=fingerprint, layer_name=args.layer)
    receipt["checkpoint_fingerprint"] = fingerprint
    if args.profile:
        payload = json.loads(args.profile.read_text())
        if isinstance(payload, dict) and payload.get("schema_version") == 2:
            from b12x.moe.fused_moe.automatic import ResidencyProfile
            artifact = ResidencyProfile.from_dict(payload)
            spec = next(s for s in artifact.model.layers if s.layer == args.layer)
            if (spec.experts != e or spec.hidden != h or spec.intermediate != i
                    or spec.gate_first != args.gate_first or spec.swiglu_limit != args.swiglu_limit
                    or spec.max_tokens < max(args.tokens) or spec.max_top_k < args.top_k):
                raise ValueError("profile recipe or prepared capacity differs from benchmark geometry")
        profiles = [p for p in read_profiles(args.profile) if p.layer == args.layer]
        if len(profiles) != 1 or profiles[0].model_fingerprint != fingerprint:
            raise ValueError("profile layer or model fingerprint differs from checkpoint bundle")
        profile = profiles[0]
    else:
        profile = moe.ExpertResidencyPlan(total_experts=e,
            hbm_expert_ids=tuple(range(args.hot_experts)), grace_expert_ids=tuple(range(args.hot_experts, e)),
            layer=args.layer, model_fingerprint=fingerprint, workload="synthetic-controlled-routing",
            provenance=f"seed:{args.seed}")
    control_profile = moe.ExpertResidencyPlan(total_experts=e, hbm_expert_ids=tuple(range(e)), grace_expert_ids=(),
        layer=args.layer, model_fingerprint=fingerprint, workload="all-HBM-control", provenance="same source bytes")
    budget = moe.ExpertMemoryBudget(hbm_bytes=args.hbm_bytes, grace_bytes=args.grace_bytes,
        hbm_safety_bytes=args.hbm_reserve, grace_safety_bytes=args.grace_reserve, kv_reserved_bytes=args.kv_reserve)
    capacity = moe.ExecutionCapacity(max_tokens=max(args.tokens), top_k=args.top_k)
    def declare(placement):
        return moe.plan_execution(experts=weight_plan, weights=weights, capacity=capacity, placement=placement, memory_budget=budget)
    plan, control = declare(profile), declare(control_profile)
    a = torch.randn(capacity.max_tokens, h, dtype=torch.bfloat16, device=device) * .1
    ids = torch.zeros(capacity.max_tokens, args.top_k, dtype=torch.int64, device=device)
    route_weights = torch.full(ids.shape, 1/args.top_k, device=device)
    def call(state):
        binding = state.bind(a=a, topk_ids=ids, topk_weights=route_weights)
        return PreparedCall(run=binding.run, output=binding.output, owners=(state, binding))
    with PreparationSession(device=device, autotune=False, compile_workers=args.compile_workers) as session:
        session.prepare((plan.request(name="residency", prepare_call=call), control.request(name="hbm_control", prepare_call=call)))
        session.freeze()
        state = plan.prepared.state
        receipt["memory"] = asdict(state.memory)
        receipt["profile"] = profile.to_dict()
        receipt["measurements"] = []
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        for m in args.tokens:
            x = a[:m]
            route_ids = ids[:m]
            w = route_weights[:m]
            chosen = []
            generator = torch.Generator().manual_seed(args.seed+m)
            for _ in range(m*args.top_k):
                cold = float(torch.rand((), generator=generator)) < args.cold_fraction
                tier = profile.grace_expert_ids if cold else profile.hbm_expert_ids
                if not tier:
                    tier = profile.hbm_expert_ids or profile.grace_expert_ids
                chosen.append(tier[int(torch.randint(len(tier), (), generator=generator))])
            route_ids.copy_(torch.tensor(chosen).reshape(m, args.top_k))
            binding = moe.bind(plan, a=x, topk_ids=route_ids, topk_weights=w)
            baseline = moe.bind(control, a=x, topk_ids=route_ids, topk_weights=w)
            actual, expected = moe.run(binding=binding), moe.run(binding=baseline)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if not torch.isfinite(actual).all() or not torch.count_nonzero(actual):
                raise AssertionError("nonfinite or zero output cannot qualify a timing")
            graph = capture(lambda: moe.run(binding=binding))
            control_graph = capture(lambda: moe.run(binding=baseline))
            stages = []
            try:
                # Mutation tests prevent a captured constant output from qualifying.
                x.mul_(.75)
                route_ids.copy_((route_ids+1) % e)
                expected = moe.run(binding=baseline).clone()
                actual.fill_(float("nan"))
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocated_bytes.all.current", "allocation.all.allocated", "allocation.all.freed"):
                    if after[key] != before[key]:
                        raise AssertionError(f"replay changed {key}")
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                # Restore the declared workload before collecting any sample.
                route_ids.copy_(torch.tensor(chosen).reshape(m, args.top_k))
                graph.replay()
                control_graph.replay()
                by_program = {id(program): name for name, program in state.programs.items()}
                stage_results = {}
                for program, params in binding.calls:
                    stage = capture(lambda program=program, params=params: program(*params, stream))
                    stages.append(stage)
                    stage_results[by_program[id(program)]] = time_graph(stage, args.iterations, args.samples)
                # Balanced order retains each arm's raw samples and ratio direction.
                gpu_before = gpu_snapshot()
                arm = {"tiered": [], "all_hbm": []}
                for order in (("all_hbm", "tiered"), ("tiered", "all_hbm")):
                    for name in order:
                        arm[name].extend(time_graph(graph if name == "tiered" else control_graph, args.iterations, args.samples)["raw_us"])
                record = {"tokens": m, "top_k": args.top_k,
                    "cold_fraction": sum(eid in profile.grace_expert_ids for eid in chosen)/len(chosen),
                    "correctness": "bitwise all-HBM parity; mutation replay; no allocator events",
                    "raw_us": arm, "stages": stage_results,
                    "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
                    "stage_timing_scope": "isolated prepared CUDA graph per launch; sums include separate graph launch costs",
                    "tiered_over_all_hbm_latency": statistics.median(arm["tiered"])/statistics.median(arm["all_hbm"])}
                receipt["measurements"].append(record)
                print(json.dumps(record), flush=True)
            finally:
                for stage in stages: stage.reset()
                graph.reset()
                control_graph.reset()
    receipt["gpu_after"] = gpu_snapshot()
    receipt["status"] = "measured"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--intermediate", type=int, default=256)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hot-experts", type=int, default=6)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--cold-fraction", type=float, default=.0155)
    p.add_argument("--profile", type=Path)
    p.add_argument("--layer", default="layer.0")
    p.add_argument("--weights", type=Path, help="torch.save CPU tensor bundle matching PackedWeights")
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--gate-first", action="store_true")
    p.add_argument("--swiglu-limit", type=float)
    p.add_argument("--hbm-bytes", type=int, default=64*2**30)
    p.add_argument("--grace-bytes", type=int, default=64*2**30)
    p.add_argument("--hbm-reserve", type=int, default=4*2**30)
    p.add_argument("--grace-reserve", type=int, default=4*2**30)
    p.add_argument("--kv-reserve", type=int, default=0)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--compile-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=741)
    args = p.parse_args()
    if args.output.exists(): p.error("output must not overwrite an existing receipt")
    if not 0 <= args.cold_fraction <= 1 or args.samples < 1 or args.iterations < 1:
        p.error("invalid sampling parameters")
    receipt = {"schema": "b12x.expert_residency.benchmark.v1", "status": "started",
        "source": source_identity(ROOT), "source_sha256": package_source_sha256(ROOT),
        "command": sys.argv, "arguments": vars(args),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "toolchain": {name: importlib.metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl", "triton", "cuda-bindings")},
        "cuda": torch.version.cuda}
    try:
        run(args, receipt)
        if receipt["benchmark_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError("benchmark source changed during qualification")
        if receipt["source_sha256"] != package_source_sha256(ROOT):
            raise RuntimeError("package source changed during qualification")
    except BaseException:
        receipt["status"] = "failed"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, default=str, indent=2)+"\n")


if __name__ == "__main__":
    main()
