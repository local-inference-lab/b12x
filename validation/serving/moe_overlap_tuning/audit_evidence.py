"""Check the recorded MoE tuning comparison and print every serving median.

This audits exported evidence only; it does not run a model or qualify a GPU.
Kernel intervals overlap and are deliberately not converted to throughput.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def require(condition, message):
    """Reject invalid evidence even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def main():
    """Check serving/trace identities and recompute the recorded medians."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=root / "results.json")
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    summaries = {}
    for name, arm in results["arms"].items():
        require(arm["correctness"]["all_smoke_passed"], f"{name}: smoke failure")
        require(not arm["correctness"]["invalid_decode_cells"], f"{name}: invalid cell")
        require(not arm["correctness"]["missing_decode_cells"], f"{name}: missing cell")
        require(len(arm["prefill"]) == len(arm["decode"]) == 5,
                f"{name}: expected five windows per cell")
        require(len(arm["busy_telemetry"]) == 2, f"{name}: expected two GPUs")
        require(not any("MAX_ACTIVE_CLUSTERS=" in arg for arg in arm["docker_argv"]),
                f"{name}: explicit grid override is not automatic selection")
        prefill = []
        for cell in arm["prefill"]:
            require(not cell.get("skipped") and cell["tok_per_sec"] > 0,
                    f"{name}: invalid prefill")
            require(cell["server_validation"]["cached_tokens"] == 0,
                    f"{name}: prefill reused cached tokens")
            prefill.append(cell["tok_per_sec"])
        summary = {"prefill32k": statistics.median(prefill)}
        for concurrency in (1, 8):
            cells = [cell for window in arm["decode"] for cell in window["cells"]
                     if cell["concurrency"] == concurrency]
            require(len(cells) == 5, f"{name}: missing C{concurrency} window")
            for cell in cells:
                require(not any(cell.get(key) for key in (
                    "failure_reason", "num_errors", "loop_detected", "underfilled",
                    "warmup_timed_out", "capacity_limited",
                )), f"{name}: failed C{concurrency} window")
                require(cell["aggregate_tps"] > 0 and cell["server_steps_per_s"] > 0,
                        f"{name}: missing throughput")
            for field, label in (("aggregate_tps", "output"),
                                 ("server_steps_per_s", "verifier"),
                                 ("server_spec_accept_length", "accepted")):
                summary[f"C{concurrency}_{label}"] = statistics.median(
                    cell[field] for cell in cells)
        summaries[name] = summary
    require(set(summaries) == {"baseline", "candidate"}, "Expected two identified arms")
    summaries["change_percent"] = {
        key: (summaries["candidate"][key] / value - 1) * 100
        for key, value in summaries["baseline"].items()
    }
    trace_path = args.results.with_name("intervals.json")
    if trace_path.is_file():
        traces = json.loads(trace_path.read_text())
        require(traces["serving_results_sha256"] == hashlib.sha256(
            args.results.read_bytes()).hexdigest(), "Trace/result identity mismatch")
        fields = {name: index for index, name in enumerate(traces["interval_fields"])}
        trace_summary = {}
        for name, arm in traces["arms"].items():
            serving = results["arms"][name.split("-rank")[0]]
            require(all(arm["source"][key] == serving[key]
                        for key in ("image_id", "vllm", "b12x")),
                    f"{name}: serving source mismatch")
            require(len(arm["steps"]) == 4 and len(arm["intervals"]) == 184,
                    f"{name}: incomplete four-step trace")
            grids = {}
            shared_down = []
            for row in arm["intervals"]:
                key = row[fields["role"]] + ":" + str(row[fields["routed_grid"]])
                grids[key] = grids.get(key, 0) + 1
                if row[fields["role"]] == "target":
                    gemms = [kernel for kernel in row[fields["concurrent_kernels"]]
                             if "deep_gemm::" in traces["kernel_names"][kernel[0]]]
                    require(bool(gemms), f"{name}: missing concurrent shared GEMM")
                    shared_down.append(max(gemms, key=lambda kernel: kernel[4])[1])
            require(len(shared_down) == 172, f"{name}: incomplete target coverage")
            trace_summary[name] = {
                "launch_counts": grids,
                "shared_down_mean_us": statistics.mean(shared_down),
            }
        summaries["profiled_intervals_not_throughput"] = trace_summary
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
