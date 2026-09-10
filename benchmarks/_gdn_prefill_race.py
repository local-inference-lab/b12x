"""Balanced end-to-end races for GDN prefill production plans."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict

import torch

from b12x.policy.generation.delta_prefill_cases import (
    GDN_PREFILL_CASES, assert_close, check_binding, make_binding, make_inputs, oracle, run_binding,
)
from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot, require_sm120


def select_cases(selection):
    if selection == "all":
        return GDN_PREFILL_CASES
    names = selection.split(",")
    by_name = {case.name: case for case in GDN_PREFILL_CASES}
    missing = set(names)-set(by_name)
    if missing:
        raise ValueError(f"unknown prefill cases: {sorted(missing)}; available: {list(by_name)}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate prefill cases")
    return tuple(by_name[name] for name in names)


def balanced_order(arms, iteration):
    """Alternate the complete order and its reverse for paired measurements."""
    return arms if iteration % 2 == 0 else tuple(reversed(arms))


def policy_context_from_file(path, identity):
    """Require an explicit profile to cover the measured device and each plan query."""
    from b12x.policy import PolicyContext, PolicyMode, ProfileRegistry
    from b12x.policy.serialization import profile_from_dict

    path = pathlib.Path(path)
    raw = path.read_bytes()
    payload = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    profile = profile_from_dict(payload.get("profile", payload))
    registry = ProfileRegistry()
    registry.register(profile)
    registry.freeze()
    if registry.find(identity) is None:
        raise ValueError(f"profile {profile.profile_id!r} does not match device {identity!r}")
    return PolicyContext.for_identity(identity, mode=PolicyMode.PREPLANNED_ONLY, registry=registry)


def _summary(samples):
    return {"median_us": statistics.median(samples), "minimum_us": min(samples),
            "samples_us": samples}


def _source(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _versions():
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda}
    for name in ("nvidia-cutlass-dsl", "flashinfer-python", "cuda-python"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _trace_flashinfer(fn):
    calls = set()
    previous = sys.getprofile()

    def trace(frame, event, arg):
        if event == "call" and "flashinfer/gdn" in frame.f_code.co_filename:
            calls.add((frame.f_code.co_filename, frame.f_code.co_name))

    sys.setprofile(trace)
    try:
        fn()
    finally:
        sys.setprofile(previous)
    return [{"function": name, **_source(path)} for path, name in sorted(calls)]


def benchmark_case(case, *, device, seed, warmup, iterations, mode, race, flush,
                   max_tokens=None, max_seqs=None, policy=None, profile_replays=0):
    tensors = make_inputs(case, device=device, seed=seed, max_tokens=max_tokens, max_seqs=max_seqs)
    initial = tensors["recurrent_state"].clone()
    expected, expected_pool = oracle(case, tensors)
    immutable = {k: v.clone() for k, v in tensors.items() if k not in ("recurrent_state", "output")}
    arms, reports = [], {}
    factories = [("b12x", None)]
    if race == "flashinfer":
        factories += [("flashinfer_auto", "auto"), ("flashinfer_no_cp", False)]
    for name, cp in factories:
        report = {"status": "failed", "timings": {}}
        reports[name] = report
        try:
            if name == "b12x":
                from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
                binding = make_binding(case, tensors, policy=policy)
                fn = lambda: run_binding("gdn", binding)
                buffers = (binding.scratch,)
                poison = lambda: binding.scratch.fill_(0xFF)
                report["policy"] = repr(binding.plan.policy_resolution)
                report["config"] = asdict(binding.plan.policy_resolution.config)
                report["metadata_validation"] = binding.plan.caps.metadata_validation
            else:
                from benchmarks._gdn_prefill_flashinfer import FlashInferArm
                adapter = FlashInferArm(case, tensors, use_cp=cp)
                fn, buffers, poison = adapter, adapter.buffers, adapter.poison
                report["use_cp"] = cp
                report["alpha_semantics"] = "exp(log_decay); installed non-CP preparation uses log2(alpha + 1e-10)"
                tensors["recurrent_state"].copy_(initial)
                report["loaded_functions"] = _trace_flashinfer(fn)
                report["selected_cp"] = any("cp_delta_rule" in x["function"] for x in report["loaded_functions"])
            for _ in range(warmup):
                tensors["recurrent_state"].copy_(initial)
                fn()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            tensors["recurrent_state"].copy_(initial)
            if name == "b12x":
                freeze_kernel_resolution("GDN prefill race capture")
            try:
                with torch.cuda.graph(graph):
                    fn()
            finally:
                if name == "b12x":
                    unfreeze_kernel_resolution()
            tensors["recurrent_state"].copy_(initial)
            tensors["output"].fill_(float("nan"))
            poison()
            bound = (*tensors.values(), *buffers)
            addresses = tuple(t.data_ptr() for t in bound)
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            allocation_delta = torch.cuda.memory_allocated(device)-allocated
            if allocation_delta != 0 or addresses != tuple(t.data_ptr() for t in bound):
                raise AssertionError(f"unstable graph replay: allocation_delta={allocation_delta}")
            if name == "b12x":
                correctness = check_binding(case, binding, expected, expected_pool, initial)
            else:
                correctness = {"output": assert_close("output", tensors["output"][:case.tokens],
                                                       expected[:case.tokens], ratio=1e-2)}
                slots = tensors["final_state_indices"][:len(case.lengths)].tolist()
                correctness["states"] = {str(s): assert_close(f"state[{s}]", tensors["recurrent_state"][s],
                                                             expected_pool[s], ratio=5e-3) for s in slots}
                untouched = [s for s in range(initial.shape[0]) if s not in slots]
                torch.testing.assert_close(tensors["recurrent_state"][untouched], initial[untouched], rtol=0, atol=0)
            for key, saved in immutable.items():
                torch.testing.assert_close(tensors[key], saved, rtol=0, atol=0)
            torch.testing.assert_close(tensors["output"][case.tokens:], expected[case.tokens:],
                                       rtol=0, atol=0, equal_nan=True)
            report.update(status="qualified", correctness=correctness, stable_addresses=True,
                          replay_allocation_bytes=allocation_delta, input_immutability=True,
                          graph_replay_after_output_poison=True, graph_replay_after_scratch_poison=True)
            arms.append((name, fn, graph, buffers))
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{case.name} {name}: {report['error']}", flush=True)
            if "illegal memory access" in str(exc).lower() or "device-side assert" in str(exc).lower():
                return {"case": asdict(case), "arms": reports, "fatal_cuda_error": True}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    modes = ("eager", "graph") if mode == "both" else (mode,)
    for execution in modes:
        for temperature in ("warm", "l2_flushed_before_restore"):
            label = f"{execution}_{temperature}"
            for name, *_ in arms:
                reports[name]["timings"][label] = {"samples_us": [], "restore_samples_us": []}
            before = nvidia_smi_gpu_mode_snapshot()
            for iteration in range(iterations):
                for name, fn, graph, _buffers in balanced_order(arms, iteration):
                    if temperature != "warm":
                        flush()
                    start.record()
                    tensors["recurrent_state"].copy_(initial)
                    end.record()
                    end.synchronize()
                    restore_us = start.elapsed_time(end)*1000
                    start.record()
                    (graph.replay if execution == "graph" else fn)()
                    end.record()
                    end.synchronize()
                    timing = reports[name]["timings"][label]
                    timing["samples_us"].append(start.elapsed_time(end)*1000)
                    timing["restore_samples_us"].append(restore_us)
            after = nvidia_smi_gpu_mode_snapshot()
            for name, *_ in arms:
                timing = reports[name]["timings"][label]
                timing.update(_summary(timing["samples_us"]))
                timing["restore_median_us"] = statistics.median(timing["restore_samples_us"])
                timing.update(gpu_mode_before=before, gpu_mode_after=after)
    if profile_replays and arms:
        torch.cuda.synchronize(device)
        torch.cuda.profiler.start()
        try:
            for iteration in range(profile_replays):
                for name, _fn, graph, _buffers in balanced_order(arms, iteration):
                    tensors["recurrent_state"].copy_(initial)
                    torch.cuda.synchronize(device)
                    with torch.cuda.nvtx.range(f"{case.name}/{name}/replay-{iteration}"):
                        graph.replay()
                        torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()
    ratios = {}
    if reports["b12x"]["status"] == "qualified":
        for name, report in reports.items():
            if name != "b12x" and report["status"] == "qualified":
                ratios[name] = {label: timing["median_us"] / reports["b12x"]["timings"][label]["median_us"]
                                for label, timing in report["timings"].items()}
    return {"case": asdict(case), "arms": reports, "flashinfer_over_b12x": ratios}


def main(args, argv, parser):
    from benchmarks.benchmark_gdn_decode import _device_provenance, _git_provenance

    if args.capacity_columns is not None:
        parser.error("--capacity-columns applies only to decode; prefill uses --capacity-tokens")
    if args.capacity_seqs is not None and args.capacity_seqs > 4096:
        parser.error("prefill --capacity-seqs must be at most 4096")
    try:
        cases = select_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    for case in cases:
        if ((args.capacity_tokens is not None and case.tokens > args.capacity_tokens)
                or (args.capacity_seqs is not None and len(case.lengths) > args.capacity_seqs)):
            parser.error(f"{case.name} exceeds the requested prefill capacity")
    device = require_sm120()
    if torch.cuda.get_device_capability(device)[0] != 12:
        parser.error("the FlashInfer SM120 race requires a compute-capability 12.x GPU")
    policy = None
    if args.policy_profile is not None:
        from b12x.policy.device import detect_device
        policy = policy_context_from_file(args.policy_profile, detect_device(device).identity)
    flush = make_l2_flush_fn(True, args.l2_flush_bytes)
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [p for directory in (root/"b12x/sequence/_shared/delta_prefill", root/"b12x/sequence/gdn_prefill")
               for p in directory.glob("*.py")]
    sources += [pathlib.Path(__file__), root/"benchmarks/_gdn_prefill_flashinfer.py",
                root/"b12x/policy/generation/delta_prefill_cases.py"]
    provenance = {
        "command": [sys.executable, str(root/"benchmarks/benchmark_gdn_decode.py"), *argv],
        "cwd": os.getcwd(), "git": _git_provenance(), "device": _device_provenance(device),
        "toolchain": _versions(), "source_files": [_source(p) for p in sorted(sources)],
        "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(), "timestamp_unix": time.time(),
        "seed": args.seed, "warmup": args.warmup, "iterations": args.iterations,
        "timed_path": "raw Q/K/V, a/b, pooled initial state to recurrence output and pooled final state",
        "checkpoint_export": False, "b12x_metadata_validation": "transactional",
        "restoration": "identical full pool before every invocation; measured separately; L2 flush precedes restore",
        "metric_direction": "FlashInfer_us / b12x_us; larger than one favors b12x",
        "sampling": "alternating complete arm order and reverse; CUDA events per invocation",
        "reference_timed": False,
        "profile_replays": args.profile_replays,
        "profile_capture": "qualified graph replays after timing; state restored outside each NVTX range",
        "policy_profile": None if args.policy_profile is None else _source(args.policy_profile),
    }
    reports = []
    print(json.dumps(provenance, sort_keys=True), flush=True)
    try:
        for case in cases:
            report = benchmark_case(case, device=device, seed=args.seed+GDN_PREFILL_CASES.index(case), warmup=args.warmup,
                                    iterations=args.iterations, mode=args.mode, race=args.race, flush=flush,
                                    max_tokens=args.capacity_tokens, max_seqs=args.capacity_seqs, policy=policy,
                                    profile_replays=args.profile_replays)
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
            if report.get("fatal_cuda_error"):
                break
    finally:
        provenance["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("x", encoding="utf-8") as output:
                json.dump({"provenance": provenance, "reports": reports}, output, indent=2, sort_keys=True)
                output.write("\n")
    return int(len(reports) != len(cases) or any(
        arm["status"] != "qualified" for report in reports for arm in report["arms"].values()))
