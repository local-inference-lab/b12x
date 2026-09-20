"""Paired serving economics at matched delivered-token counts.

These are retrospective delivery curves, not a predictor or a counterfactual GPU
cost model. Async probe response latency is not charged as scheduler pause time.
"""

import argparse
import json
from pathlib import Path

from benchmarks.moe.summarize_expert_cache_serving import summarize, distribution
from b12x.moe.residency.health import RoutingHealthThresholds


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def deliveries(rows, label):
    selected = [r for r in rows if r["kind"] == "request" and r["workload"] == label]
    origin = min(r["start_wall_ns"] for r in selected)
    times = sorted(
        r["start_wall_ns"] + event["ns"]
        for r in selected
        for event in r["events"]
        for _ in range(event["new_tokens"])
    )
    return origin, [(t - origin) / 1e6 for t in times]


def economics(reference, candidate, *, prior_penalty_ms=0):
    if not reference or len(reference) != len(candidate):
        raise ValueError("paired delivery curves must cover identical token counts")
    gain = [a - b - prior_penalty_ms for a, b in zip(reference, candidate, strict=True)]
    # The last negative sample prevents a transient early lead being reported as
    # a durable payback. Finite-corpus survival is not a future guarantee.
    last_negative = max((i for i, v in enumerate(gain) if v < 0), default=-1)
    index = last_negative + 1
    return dict(
        tokens=len(gain),
        final_gain_ms=gain[-1],
        sustained_break_even_tokens=index + 1 if index < len(gain) else None,
        sustained_break_even_ms=candidate[index] if index < len(gain) else None,
        candidate_regime_ms=candidate[-1],
        reference_regime_ms=reference[-1],
        curve=[
            dict(
                tokens=i + 1,
                candidate_ms=candidate[i],
                reference_ms=reference[i],
                gain_ms=gain[i],
            )
            for i in range(len(gain))
            if i % 32 == 0 or i == len(gain) - 1
        ],
    )


def maintenance_costs(rows):
    """Keep nested stage measurements separate; they overlap by construction."""
    groups = {}
    for row in rows:
        receipt = row["receipt"]
        worker = receipt["worker"]
        for category, stages in (
            ("engine", receipt["engine_stages_ns"]),
            ("worker", worker["stages_ns"]),
            ("snapshot", worker.get("snapshot_stages_ns", {})),
            ("policy", worker.get("policy_stages_ns", {})),
        ):
            for name, ns in stages.items():
                groups.setdefault(f"{category}.{name}", []).append(ns / 1e6)
    return {name: distribution(values) for name, values in groups.items()}


def compare(baseline_path, candidate_path):
    baseline, candidate = summarize(baseline_path), summarize(candidate_path)
    keys = (
        "model",
        "profile",
        "prompts",
        "tokens",
        "concurrency",
        "capacity",
        "cache_gib",
        "kv_gib",
        "context",
        "eager",
        "inductor",
        "admission",
    )
    if any(baseline["arguments"].get(k) != candidate["arguments"].get(k) for k in keys):
        raise ValueError("paired engine/admission/fixture configurations differ")
    if candidate["arguments"]["admission"] != "together":
        raise ValueError("numerical qualification requires controlled admission")
    if baseline["output_token_sha256"] != candidate["output_token_sha256"]:
        raise ValueError(
            "paired output token equality failed; retain unqualified receipts"
        )
    if any(
        baseline["complete"][k] != candidate["complete"][k]
        for k in ("checkpoint", "profile")
    ):
        raise ValueError("paired checkpoint/profile identity differs")
    rows, other = records(candidate_path), records(baseline_path)
    labels = list(candidate["workloads"])
    candidate_start = min(r["start_wall_ns"] for r in rows if r["kind"] == "request")
    baseline_start = min(r["start_wall_ns"] for r in other if r["kind"] == "request")
    origins = [deliveries(rows, label)[0] for label in labels]
    phases = {}
    for index, label in enumerate(labels):
        begin, ct = deliveries(rows, label)
        base_begin, bt = deliveries(other, label)
        end = origins[index + 1] if index + 1 < len(origins) else float("inf")
        maintenance = [
            r
            for r in rows
            if r["kind"] == "maintenance"
            and not r["receipt"]["worker"]["baseline"]
            and begin <= r["time_ns"] < end
        ]
        probes = [
            r
            for r in rows
            if r["kind"] == "health_probe" and begin <= r["time_ns"] < end
        ]
        pressure = [r for r in probes if r["assessment"]["health"] == "pressure"]
        if not probes:
            pressure = [
                r
                for r in maintenance
                if r["receipt"]["worker"].get("health") == "pressure"
            ]
        promotion = [r for r in maintenance if r["receipt"]["worker"]["selected_pairs"]]
        comparisons = {}
        for name, gate in [
            ("global", RoutingHealthThresholds(cold_fraction=0.15)),
            (
                "half_layers",
                RoutingHealthThresholds(cold_fraction=0.15, minimum_layer_fraction=0.5),
            ),
            (
                "most_layers",
                RoutingHealthThresholds(cold_fraction=0.15, minimum_layer_fraction=0.8),
            ),
        ]:
            comparisons[name] = sum(
                gate.assess(r["receipt"]["summary"])["health"] == "pressure"
                for r in probes
            )
        phases[label] = dict(
            economics=economics(bt, ct),
            economics_including_prior_intervals=economics(
                bt,
                ct,
                prior_penalty_ms=(
                    (begin - candidate_start) - (base_begin - baseline_start)
                )
                / 1e6,
            ),
            health_probes=len(probes),
            maintenance=len(maintenance),
            no_op_maintenance=sum(
                not r["receipt"]["worker"]["selected_pairs"] for r in maintenance
            ),
            promotions=sum(
                r["receipt"]["worker"]["selected_pairs"] for r in maintenance
            ),
            maintenance_triggers={
                name: sum(r.get("trigger") == name for r in maintenance)
                for name in ("pressure", "maximum_interval")
            },
            first_pressure_ms=(pressure[0]["time_ns"] - begin) / 1e6
            if pressure
            else None,
            first_maintenance_ms=(maintenance[0]["time_ns"] - begin) / 1e6
            if maintenance
            else None,
            first_maintenance_request_ms=(
                maintenance[0]["time_ns"]
                - maintenance[0]["receipt"]["total_wall_ns"]
                - begin
            )
            / 1e6
            if maintenance
            else None,
            first_promotion_ms=(promotion[0]["time_ns"] - begin) / 1e6
            if promotion
            else None,
            blocked_scheduling_ms=sum(
                r["receipt"]["engine_wall_ns"] for r in maintenance
            )
            / 1e6,
            worker_apply_ms=sum(
                r["receipt"]["worker"]["stages_ns"].get("apply", 0) for r in maintenance
            )
            / 1e6,
            maintenance_stages_ms=maintenance_costs(maintenance),
            probe_response_ms=distribution(
                [r["receipt"]["total_wall_ns"] / 1e6 for r in probes]
            ),
            probe_worker_submit_ms=distribution(
                [r["receipt"]["worker_submit_ns"] / 1e6 for r in probes]
            ),
            probe_worker_poll_ms=distribution(
                [r["receipt"]["summary"]["worker_poll_ns"] / 1e6 for r in probes]
            ),
            reduction_us=distribution(
                [r["receipt"]["summary"]["reduction_us"] for r in probes]
            ),
            summary_copy_us=distribution(
                [r["receipt"]["summary"]["copy_us"] for r in probes]
            ),
            signal_pressure_probes=comparisons,
            maintenance_series=[
                dict(
                    completion_ms=(r["time_ns"] - begin) / 1e6,
                    delivered_tokens=r.get("output_tokens_so_far"),
                    trigger=r.get("trigger"),
                    cold_fraction=r["receipt"]["worker"].get("cold_fraction"),
                    promotions=r["receipt"]["worker"]["selected_pairs"],
                    blocked_scheduling_ms=r["receipt"]["engine_wall_ns"] / 1e6,
                )
                for r in maintenance
            ],
            health_series=[
                dict(
                    ms=(r["time_ns"] - begin) / 1e6,
                    delivered_tokens=r["output_tokens_so_far"],
                    cold_fraction=r["receipt"]["summary"]["cold_fraction"],
                    assessment=r["assessment"],
                    repeated_cold_fraction=sum(
                        v["repeated_cold_selections"]
                        for v in r["receipt"]["summary"]["layers"].values()
                    )
                    / r["receipt"]["summary"]["selections"]
                    if r["receipt"]["summary"]["selections"]
                    else None,
                )
                for r in probes
            ],
        )
    return dict(
        baseline=baseline,
        candidate=candidate,
        phases=phases,
        final_control_tail_ms=(
            candidate["serving"]["elapsed_ns"]
            - candidate["serving"]["request_elapsed_ns"]
        )
        / 1e6,
        complete_run_gain_ms=(
            baseline["serving"]["elapsed_ns"] - candidate["serving"]["elapsed_ns"]
        )
        / 1e6,
        qualification="exact paired tokens; controlled group admission; retrospective delivered work",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidates", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = [compare(args.baseline, path) for path in args.candidates]
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
