"""Recompute the recorded QSA performance tables without running inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import prompt_helpers


def close(actual, expected):
    """Reject a receipt whose reported value differs from its raw samples."""
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
        actual, expected
    )


def restore_prompts(root, destination=None):
    """Reconstruct and hash every measured prompt from its compact recipe."""
    records = json.loads((root / "prompts.json").read_text())
    for record in records:
        nominal = record["nominal_tokens"]
        prefix = f"[QWEN_MATCHED_{record['name']}_20260916] "
        padding = prompt_helpers.generate_padding_text(max(nominal * 2, 1))
        messages = prompt_helpers.build_messages(
            nominal, prefix + padding[:record["padding_chars"]] if nominal else ""
        )
        if not nominal:
            messages[-1]["content"] += "\nRequest identifier: " + prefix
        original = {
            "name": record["name"],
            "nominal_tokens": nominal,
            "actual_tokens": record["actual_tokens"],
            "messages": messages,
            "prompt_ids_sha256": record["prompt_ids_sha256"],
        }
        serialized = (json.dumps(original) + "\n").encode()
        assert hashlib.sha256(serialized).hexdigest() == record["serialized_sha256"]
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / (record["name"] + ".json")
            if path.exists():
                assert path.read_bytes() == serialized, path
            else:
                path.write_bytes(serialized)
    return len(records)


def verify_serving(root, arm):
    """Recompute each batch's throughput from server duration or stream events."""
    records = [json.loads(line) for line in (root / arm["samples"]).read_text().splitlines()]
    assert len(records) == arm["performance_cells"]
    grouped = {}
    for record in records:
        stats = record["statistics"]
        requests = record["requests"]
        assert stats["cached_tokens"] == 0
        assert record["preemptions_before_after"] == [0.0, 0.0]
        for request in requests:
            assert sum(n for _, n in request["events"]) == request["usage"]["completion_tokens"]
        if "prefill_tps" in stats:
            assert len(requests) == 1
            rate = requests[0]["usage"]["prompt_tokens"] / stats["prefill_seconds"]
            close(rate, stats["prefill_tps"])
        else:
            begin = max(r["events"][0][0] for r in requests)
            end = min(r["events"][-1][0] for r in requests)
            tokens = sum(n for r in requests for t, n in r["events"] if begin < t <= end)
            assert tokens == stats["common_tokens"]
            close(end - begin, stats["common_seconds"])
            rate = tokens / (end - begin)
            close(rate, stats["common_decode_tps"])
            close(stats["accepted_tokens"] / stats["proposed_tokens"], stats["acceptance_fraction"])
        cell = record["name"].split("-", 1)[1]
        grouped.setdefault(cell, []).append(rate)
    if arm["complete"]:
        assert len(grouped) == 7
        assert all(len(values) == 3 for values in grouped.values())
    return {cell: statistics.mean(values) for cell, values in grouped.items()}


def main():
    """Validate artifact hashes, sample arithmetic, and deterministic prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-prompts", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    receipt = json.loads((root / "receipt.json").read_text())
    for topology, arm in receipt["component_cases"].items():
        assert hashlib.sha256((root / arm["benchmark"]).read_bytes()).hexdigest() == arm["benchmark_sha256"]
        result = json.loads((root / arm["samples"]).read_text())
        assert result["complete"]
        for case in result["cases"]:
            assert "exact nonzero output" in case["correctness"]
            medians = {}
            for name, samples in case["samples_us"].items():
                assert len(samples) == 30
                assert all(math.isfinite(value) and value > 0 for value in samples)
                medians[name] = statistics.median(samples)
                close(medians[name], case["median_us"][name])
            ratio = medians["fused_cleanup"] / medians["baseline"]
            close(ratio, case["candidate_over_baseline"])
            print(f"{topology} component: {case['rows']} rows, {case['context']} context: {100 * (1 - ratio):.4f}% lower latency")
    means = {name: verify_serving(root, arm) for name, arm in receipt["serving_arms"].items()}
    for cell, baseline in means["tp4_control"].items():
        candidate = means["tp4_candidate"][cell]
        print(f"{cell}: TP4 {baseline:.4f} -> {candidate:.4f} ({100 * (candidate / baseline - 1):+.4f}%); TP2 {means['tp2_candidate'][cell]:.4f}")
    count = restore_prompts(root, args.write_prompts)
    print(f"Verified {len(manifest)} artifact hashes and {count} byte-identical prompts. No inference executed.")


if __name__ == "__main__":
    main()
