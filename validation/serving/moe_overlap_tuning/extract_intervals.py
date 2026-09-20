#!/usr/bin/env python3
"""Compare complete routed-MoE intervals, including separate phase kernels.

An interval starts at the routed kernel and ends after the following PCIe
all-reduce. This includes joining a concurrent shared expert and any split
FC1/FC2 phases. It is not an unprofiled throughput estimate. All captured
steps remain visible so profiler-start rank skew cannot be hidden.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path, target_layers, draft_layers):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"
         and "dur" in event), key=lambda event: event["ts"],
    )
    routed = [(index, event) for index, event in enumerate(kernels)
              if "MoEDynamicKernel" in event["name"]]
    layer_count = target_layers + draft_layers
    if not routed or len(routed) % layer_count:
        raise ValueError("Routed kernel count does not match the declared layers")
    intervals = []
    for ordinal, (index, event) in enumerate(routed):
        start = event["ts"]
        end = start + event["dur"]
        later = kernels[index + 1:]
        collective = next((item for item in later
                           if "OneshotLaunch" in item["name"]
                           and item["ts"] >= end), None)
        if collective is None:
            raise ValueError("Routed MoE interval lacks its collective endpoint")
        if ordinal + 1 < len(routed) and collective["ts"] >= routed[ordinal + 1][1]["ts"]:
            raise ValueError("Collective attribution crossed another routed layer")
        phases = [item for item in later if start < item["ts"] < collective["ts"]
                  and "MaterializedPhase" in item["name"]]
        concurrent = []
        for item in kernels:
            if item["ts"] >= end:
                break
            overlap = min(end, item["ts"] + item["dur"]) - max(start, item["ts"])
            if overlap > 0 and item["args"]["stream"] != event["args"]["stream"]:
                concurrent.append({
                    "name": item["name"], "duration_us": item["dur"],
                    "overlap_us": overlap, "grid": item["args"]["grid"],
                    "start_relative_us": item["ts"] - start,
                })
        intervals.append({
            "step": ordinal // layer_count,
            "layer": ordinal % layer_count,
            "role": "target" if ordinal % layer_count < target_layers else "draft",
            "routed_kernel_us": event["dur"],
            "routed_grid": event["args"]["grid"],
            "routed_block": event["args"]["block"],
            "separate_phase_count": len(phases),
            "separate_phase_us": sum(item["dur"] for item in phases),
            "through_collective_us": collective["ts"] + collective["dur"] - start,
            "concurrent_kernels": concurrent,
        })
    steps = []
    for step in range(len(routed) // layer_count):
        by_role = {}
        for role in ("target", "draft"):
            selected = [item for item in intervals if item["step"] == step
                        and item["role"] == role]
            if not selected:
                continue
            by_role[role] = {
                "layers": len(selected),
                "mean_routed_kernel_us": statistics.mean(
                    item["routed_kernel_us"] for item in selected),
                "mean_through_collective_us": statistics.mean(
                    item["through_collective_us"] for item in selected),
                "sum_through_collective_ms": sum(
                    item["through_collective_us"] for item in selected) / 1000,
                "separate_phase_count": sum(item["separate_phase_count"]
                                            for item in selected),
            }
        steps.append({"step": step, **by_role})
    return {"trace": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "steps": steps, "intervals": intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="label=trace.json[.gz]")
    parser.add_argument("--target-layers", type=int, required=True)
    parser.add_argument("--draft-layers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path,
                        help="Attach image/source identities from a serving comparison")
    parser.add_argument("--compact", action="store_true",
                        help="Store kernel names once and retain all numeric intervals")
    args = parser.parse_args()
    if args.target_layers < 1 or args.draft_layers < 0:
        parser.error("Target layers must be positive and draft layers nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = [item.split("=", 1) for item in args.trace]
    if len({label for label, _ in pairs}) != len(pairs):
        parser.error("Trace labels must be unique")
    arms = {label: summarize(Path(path), args.target_layers, args.draft_layers)
            for label, path in pairs}
    result = {"method": __doc__, "target_layers": args.target_layers,
              "draft_layers": args.draft_layers, "arms": arms}
    if args.results:
        serving = json.loads(args.results.read_text())
        expected = {f"{side}-rank{rank}" for side in ("baseline", "candidate")
                    for rank in (0, 1)}
        if set(arms) != expected:
            parser.error("Serving evidence requires baseline/candidate-rank0/rank1")
        result["serving_results_sha256"] = hashlib.sha256(
            args.results.read_bytes()).hexdigest()
        for label, arm in arms.items():
            source = serving["arms"][label.split("-rank")[0]]
            arm["source"] = {key: source[key] for key in ("image_id", "vllm", "b12x")}
    if args.compact:
        names = sorted({kernel["name"] for arm in arms.values()
                        for interval in arm["intervals"]
                        for kernel in interval["concurrent_kernels"]})
        fields = ("step", "layer", "role", "routed_kernel_us", "routed_grid",
                  "routed_block", "separate_phase_count", "separate_phase_us",
                  "through_collective_us", "concurrent_kernels")
        kernel_fields = ("duration_us", "overlap_us", "grid", "start_relative_us")
        result["interval_fields"] = fields
        result["concurrent_kernel_fields"] = ("name_index", *kernel_fields)
        result["kernel_names"] = names
        for arm in arms.values():
            for interval in arm["intervals"]:
                interval["concurrent_kernels"] = [
                    [names.index(kernel["name"]), *(kernel[key] for key in kernel_fields)]
                    for kernel in interval["concurrent_kernels"]
                ]
            arm["intervals"] = [[interval[key] for key in fields]
                                for interval in arm["intervals"]]
    args.output.write_text(json.dumps(result, indent=None if args.compact else 2) + "\n")
    print(json.dumps({label: arm["steps"] for label, arm in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
