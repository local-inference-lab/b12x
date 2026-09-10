#!/usr/bin/env python3
"""Qualify and time KDA production prefill from an explicitly selected source tree.

Run this same harness against both trees with the same device and toolchain.
The harness refuses mismatched input contracts and reports every per-case
median slowdown above 5 percent. No alternate kernel or reference is timed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys


def compare_reports(baseline, candidate):
    for key in ("input_contract_sha256", "seed", "device_uuid", "torch", "cutlass"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise ValueError(f"regression arms have different {key}")
    before = {r["case"]: r for r in baseline["reports"]}
    after = {r["case"]: r for r in candidate["reports"]}
    if set(before) != set(after):
        raise ValueError("regression arms cover different cases")
    return {name: {"candidate_over_baseline": after[name]["median_us"]/before[name]["median_us"],
                   "exceeds_5_percent": after[name]["median_us"] > 1.05*before[name]["median_us"]}
            for name in before}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-revision", help="required for an archived source tree without Git metadata")
    parser.add_argument("--json", type=pathlib.Path, required=True)
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error(f"refusing to overwrite {args.json}")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    source = args.source_root.resolve(strict=True)
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    if args.source_revision is not None:
        if revision and revision != args.source_revision:
            parser.error("--source-revision differs from the selected tree's Git revision")
        revision = args.source_revision
    if not revision:
        parser.error("--source-revision is required for a source tree without Git metadata")
    sys.path.insert(0, str(source))
    import importlib.metadata
    import torch
    import b12x
    if not pathlib.Path(b12x.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"loaded b12x from {b12x.__file__}, expected {source}")
    contract = pathlib.Path(__file__).resolve().parents[1]/"b12x/policy/generation/delta_prefill_cases.py"
    spec = importlib.util.spec_from_file_location("delta_prefill_regression_cases", contract)
    cases_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cases_module
    spec.loader.exec_module(cases_module)
    from benchmarks.common import nvidia_smi_gpu_mode_snapshot
    all_cases = cases_module.KDA_PREFILL_CASES
    by_name = {c.name: c for c in all_cases}
    names = list(by_name) if args.cases == "all" else args.cases.split(",")
    if set(names)-set(by_name) or len(set(names)) != len(names):
        parser.error(f"unknown or duplicate cases; available: {list(by_name)}")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    production = source/"b12x/sequence/kda_prefill"
    files = list(production.glob("*.py"))
    files += list((source/"b12x/sequence/_shared/delta_prefill").glob("*.py"))
    payload = {"provenance": {
        "command": [sys.executable, str(pathlib.Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "source_root": str(source), "seed": args.seed,
        "git_revision": revision,
        "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "source_files": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        "device_uuid": str(properties.uuid), "device_name": properties.name,
        "torch": str(torch.__version__), "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
        "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(), "iterations": args.iterations,
        "warmup": args.warmup, "metric_direction": "lower latency is better; candidate / baseline > 1.05 requires rerun",
        "timed_path": "public KDA prefill; CUDA graph replay; state restore outside timing",
    }, "reports": []}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        for name in names:
            case = by_name[name]
            data = cases_module.make_inputs(case, device=device, seed=args.seed+all_cases.index(case))
            initial = data["recurrent_state"].clone()
            expected, state = cases_module.oracle(case, data)
            binding = cases_module.make_binding(case, data)
            for _ in range(args.warmup):
                binding.recurrent_state.copy_(initial)
                cases_module.run_binding("kda", binding)
            graph = torch.cuda.CUDAGraph()
            binding.recurrent_state.copy_(initial)
            with torch.cuda.graph(graph):
                cases_module.run_binding("kda", binding)
            binding.recurrent_state.copy_(initial)
            binding.output.fill_(float("nan"))
            binding.scratch.fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize(device)
            correctness = cases_module.check_binding(case, binding, expected, state, initial)
            samples = []
            for _ in range(args.iterations):
                binding.recurrent_state.copy_(initial)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end)*1000)
            row = {"case": name, "correctness": correctness, "samples_us": samples,
                   "median_us": statistics.median(samples)}
            payload["reports"].append(row)
            print(json.dumps(row), flush=True)
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["provenance"]["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
        if args.compare is not None and "error" not in payload:
            try:
                payload["comparison"] = compare_reports(json.loads(args.compare.read_text()), payload)
            except ValueError as exc:
                payload["comparison_error"] = str(exc)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
    return int("comparison_error" in payload or any(
        r["exceeds_5_percent"] for r in payload.get("comparison", {}).values()))


if __name__ == "__main__":
    raise SystemExit(main())
