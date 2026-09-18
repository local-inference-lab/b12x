"""Time native residency route partition with optional prepared routing counters."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from b12x.moe.fused_moe.routing_profile import plan_routing_profile, RoutingProfileQuery
from b12x.preparation import PreparationSession, PreparedCall
from scripts._sm103_source import package_source_sha256, source_identity


def run(args, receipt):
    import cutlass
    import cuda.bindings.driver as cuda
    from b12x._lib.architecture import architecture_for
    from b12x.moe.fused_moe._residency_tuning import ResidencyQuery
    from b12x.moe.fused_moe._residency_preparation import _compile_programs
    from b12x.moe._shared.kernels.sm103.launch import pointer
    cc = torch.cuda.get_device_capability()
    if cc != (10, 3) and not (args.portable and cc in ((12, 0), (12, 1))):
        raise RuntimeError("physical SM103 required; --portable permits explicitly labelled SM120/SM121 diagnostics")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    receipt.update(qualification="SM103 profiler microbenchmark" if cc == (10, 3) else "portable profiler diagnostic; not B300 performance",
        device=properties.name, uuid=str(properties.uuid), capability=cc, cuda=torch.version.cuda,
        toolchain={name: importlib.metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl", "triton", "cuda-bindings")})
    def snapshot():
        return subprocess.check_output(["nvidia-smi", "-q", "-i", "GPU-"+str(properties.uuid).removeprefix("GPU-")], text=True)
    receipt["gpu_before"] = snapshot()
    cap, top = max(args.tokens), max(args.top_k)
    q = ResidencyQuery(hidden=256, intermediate=256, experts=args.experts, hot_experts=args.experts//2,
        max_tokens=cap, max_top_k=top, profile_hash="0"*64, model_fingerprint="synthetic-routing", gate_first=False)
    programs = _compile_programs(q, metadata_only=True, target=architecture_for(cc).compilation_target)
    plans = {sample: plan_routing_profile(RoutingProfileQuery(layers=(("layer", args.experts),),
        max_tokens=cap, max_top_k=top, sample_every=sample)) for sample in args.sample_every}
    ids = torch.zeros((cap*top,), dtype=torch.int64, device=device)
    def prime(state):
        binding = state.bind(layer="layer", phase="decode", topk_ids=ids.view(cap, top))
        return PreparedCall(run=binding.run, owners=(state, binding))
    mapping = torch.tensor([(e%2, e//2) for e in range(args.experts)], dtype=torch.int32, device=device)
    stride = (cap*top+3)//4*4
    local = torch.empty((2, stride), device=device, dtype=torch.int32)
    indices = torch.empty_like(local)
    counts = torch.empty(8, device=device, dtype=torch.int32)
    partition = programs["partition_i64"]
    rows = []
    receipt["measurements"] = rows
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(tuple(p.request(name=f"profile.{sample}", prepare_call=prime) for sample, p in plans.items()))
        session.freeze()
        for m in args.tokens:
            for k in args.top_k:
                live = ids[:m*k].view(m, k)
                live.copy_((torch.arange(m*k, device=device) % args.experts).view(m, k))
                if args.contention: live.zero_()
                params = (pointer(cutlass.Int64, live), pointer(cutlass.Int32, mapping), pointer(cutlass.Int32, local),
                    pointer(cutlass.Int32, indices), pointer(cutlass.Int32, counts), cutlass.Int32(m*k))
                def route():
                    partition(*params, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                calls = {"off": route}
                states = {}
                for sample, plan in plans.items():
                    state = plan.prepared.state
                    binding = state.bind(layer="layer", phase="decode", topk_ids=live)
                    def profiled(binding=binding):
                        route()
                        binding.run()
                    name = f"counter_every_{sample}"
                    calls[name] = profiled
                    states[name] = state
                graphs, timing_graphs = {}, {}
                try:
                    for name, call in calls.items():
                        call()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph): call()
                        graphs[name] = graph
                    # Counter and compaction correctness precede timing.
                    for state in states.values(): state.reset(quiescent=True)
                    for name, graph in graphs.items():
                        graph.replay()
                        if name in states:
                            actual = states[name].snapshot(quiescent=True).layers[0].counts
                            expected = tuple(int((live.cpu() == e).sum()) for e in range(args.experts))
                            if actual != expected: raise AssertionError("counter oracle mismatch")
                    flat = live.cpu().flatten().tolist()
                    for tier in (0, 1):
                        expected = [r for r, e in enumerate(flat) if e%2 == tier]
                        if indices[tier, :len(expected)].cpu().tolist() != expected:
                            raise AssertionError("route compaction oracle mismatch")
                    before = torch.cuda.memory_stats()
                    for graph in graphs.values(): graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    if any(before[key] != after[key] for key in ("allocation.all.allocated", "allocation.all.freed")):
                        raise AssertionError("graph replay allocated storage")
                    # Batch the device work inside each graph. A Python replay
                    # loop can starve tiny kernels and measure enqueue latency.
                    for name, call in calls.items():
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(args.iterations): call()
                        timing_graphs[name] = graph
                    for graph in timing_graphs.values():
                        graph.replay()
                        graph.replay()
                    torch.cuda.synchronize()
                    gpu_start = snapshot()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    timings = {name: [] for name in graphs}
                    names = list(graphs)
                    for n in range(args.samples):
                        order = names if n%2 == 0 else list(reversed(names))
                        for name in order:
                            timing_graphs[name].replay()
                            start.record()
                            timing_graphs[name].replay()
                            end.record()
                            end.synchronize()
                            timings[name].append(start.elapsed_time(end)*1000/args.iterations)
                    baseline = statistics.median(timings["off"])
                    rows.append(dict(tokens=m, top_k=k, gpu_before=gpu_start, gpu_after=snapshot(), correctness="exact counters and compact routes; allocation-free replay",
                        raw_us=timings, median_us={name: statistics.median(v) for name, v in timings.items()},
                        ratio_to_off={name: statistics.median(v)/baseline for name, v in timings.items()}))
                finally:
                    for graph in (*graphs.values(), *timing_graphs.values()): graph.reset()
    receipt["gpu_after"] = snapshot()
    receipt["ratio_direction"] = "profiled / profiling-off latency; values above one are slower"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--top-k", type=int, nargs="+", default=[6])
    p.add_argument("--sample-every", type=int, nargs="+", default=[1, 128])
    p.add_argument("--iterations", type=int, default=1024)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--portable", action="store_true")
    p.add_argument("--contention", action="store_true", help="route every selection to expert zero")
    args = p.parse_args()
    if min(args.experts, args.iterations, args.samples, *args.tokens, *args.top_k, *args.sample_every) <= 0:
        p.error("geometry, sample intervals and iteration counts must be positive")
    receipt = dict(command=sys.argv, source=source_identity(ROOT), source_sha256=package_source_sha256(ROOT),
        worktree=str(ROOT), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status="running", arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args, receipt)
        receipt["status"] = "passed"
    except BaseException:
        receipt["status"] = "failed"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
