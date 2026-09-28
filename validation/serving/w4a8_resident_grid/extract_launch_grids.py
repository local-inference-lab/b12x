"""Extract routed-MoE launch geometry from the two qualified serving traces.

Trace arguments use ``baseline=path`` and ``candidate=path`` (one per rank).
Image and source identities are read from the adjacent serving results. Each
trace must contain four complete steps with 43 target and three draft layers.
No profiled duration is used as an unprofiled throughput measurement.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def extract(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = sorted((event for event in events
                      if event.get("cat") == "kernel"
                      and "MoEDynamicKernel" in event["name"]),
                     key=lambda event: event["ts"])
    if len(kernels) != 4 * 46:
        raise ValueError("Expected four complete target/draft steps")
    records = []
    for index, event in enumerate(kernels):
        args = event["args"]
        records.append({
            "step": index // 46, "layer": index % 46,
            "role": "target" if index % 46 < 43 else "draft",
            "grid": args["grid"], "block": args["block"],
            "registers_per_thread": args["registers per thread"],
            "shared_memory_bytes": args["shared memory"],
        })
    return {
        "trace_file": path.name,
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kernel_names": sorted({event["name"] for event in kernels}),
        "counts": {role: [
            {"grid": list(grid), "calls": count}
            for grid, count in sorted(Counter(
                tuple(record["grid"]) for record in records
                if record["role"] == role).items())
        ] for role in ("target", "draft")},
        "launches": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    serving = json.loads(args.results.read_text())
    arms = {}
    for value in args.trace:
        label, filename = value.split("=", 1)
        if label not in {"baseline", "candidate"}:
            parser.error("Trace labels must be baseline or candidate")
        arm = arms.setdefault(label, {
            name: serving["arms"][label][name]
            for name in ("image_id", "vllm", "b12x")
        })
        arm.setdefault("traces", []).append(extract(Path(filename)))
    if set(arms) != {"baseline", "candidate"} or any(
        len(arm["traces"]) != 2 for arm in arms.values()
    ):
        parser.error("Supply exactly two rank traces for each arm")
    result = {"purpose": __doc__, "serving_results_sha256":
              hashlib.sha256(args.results.read_bytes()).hexdigest(), "arms": arms}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
