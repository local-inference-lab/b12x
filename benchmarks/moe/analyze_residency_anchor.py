"""Score recorded canonical routing against the immutable learned placement.

Counterfactual cold selections are routing evidence, not predicted execution
latency. Windows crossing request-class boundaries are labeled explicitly.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from b12x.integration.vllm.expert_cache import digest
from b12x.moe.fused_moe.residency import ExpertResidencyPlan
from b12x.moe.residency import compare_anchor
from benchmarks.moe.summarize_expert_health import records


def score_placements(counts, current, anchor):
    """Compare two resident sets against the same nonnegative route counts."""
    compared = compare_anchor(counts, current, anchor)
    layers = {
        n: dict(
            selections=v["selections"],
            current_cold=v["cold_selections"],
            anchor_cold=v["anchor_cold_selections"],
            anchor_advantage=v["cold_selections"] - v["anchor_cold_selections"],
            missing_anchor=v["missing_anchor"],
            non_anchor_resident=v["missing_anchor"],
            resident=v["resident"],
        )
        for n, v in compared["layers"].items()
    }
    total = sum(v["selections"] for v in layers.values())
    cold = sum(v["current_cold"] for v in layers.values())
    prior = sum(v["anchor_cold"] for v in layers.values())
    resident = sum(v["resident"] for v in layers.values())
    missing = sum(v["missing_anchor"] for v in layers.values())
    return dict(
        layers=layers,
        selections=total,
        current_cold=cold,
        anchor_cold=prior,
        current_cold_fraction=cold / total if total else None,
        anchor_cold_fraction=prior / total if total else None,
        anchor_advantage=cold - prior,
        normalized_anchor_advantage=(cold - prior) / total if total else None,
        layers_favoring_anchor=sum(v["anchor_advantage"] > 0 for v in layers.values()),
        layers_favoring_current=sum(v["anchor_advantage"] < 0 for v in layers.values()),
        missing_anchor=missing,
        anchor_overlap=1 - missing / resident if resident else 1,
    )


def load_anchor(profile, prepared):
    artifact = json.loads(profile.read_text())
    identity = artifact.pop("hash")
    if digest(artifact) != identity or prepared["profile"] != identity:
        raise ValueError("anchor artifact hash differs from prepared profile")
    if artifact["identity"]["checkpoint"] != prepared["checkpoint"]:
        raise ValueError("anchor checkpoint differs from prepared model")
    plans = {
        n: ExpertResidencyPlan.from_dict(v) for n, v in artifact["placements"].items()
    }
    if set(plans) != set(prepared["layers"]):
        raise ValueError("anchor does not cover prepared layers")
    for name, plan in plans.items():
        actual = prepared["layers"][name]
        if (
            plan.total_experts != actual["experts"]
            or len(plan.hbm_expert_ids) != actual["resident"]
            or plan.model_fingerprint != prepared["checkpoint"]
        ):
            raise ValueError("anchor layer geometry or resident budget differs")
    return plans


def analyze(path, profile):
    rows = records(path)
    prepared = next(r["status"] for r in rows if r["kind"] == "prepared")
    plans = load_anchor(profile, prepared)
    anchor = {n: set(p.hbm_expert_ids) for n, p in plans.items()}
    current = {n: set(ids) for n, ids in anchor.items()}
    phases = []
    for r in sorted(
        (r for r in rows if r["kind"] == "request"), key=lambda r: r["index"]
    ):
        begin = phases[-1][2] if phases else 0
        if phases and phases[-1][0] == r["workload"]:
            phases[-1][2] += r["output_tokens"]
        else:
            phases.append([r["workload"], begin, begin + r["output_tokens"]])
    windows, previous = [], 0
    totals = defaultdict(lambda: defaultdict(int))
    for record in rows:
        if record["kind"] != "maintenance" or record["receipt"]["worker"]["baseline"]:
            continue
        worker = record["receipt"]["worker"]
        observations = worker.get("policy_observations")
        if not observations:
            raise ValueError("counterfactual analysis requires full policy diagnostics")
        counts = {n: [0] * p.total_experts for n, p in plans.items()}
        for observation in observations:
            if set(observation["layers"]) != set(counts):
                raise ValueError("diagnostic layer set differs from anchor")
            for name, value in observation["layers"].items():
                counts[name] = [
                    a + b for a, b in zip(counts[name], value["counts"], strict=True)
                ]
        score = score_placements(counts, current, anchor)
        if (
            score["selections"] != worker["selections"]
            or score["current_cold"] != worker["cold_selections"]
        ):
            raise ValueError("reconstructed map/counts differ from maintenance receipt")
        end = record["output_tokens_so_far"]
        labels = [
            name for name, begin, stop in phases if previous >= begin and end <= stop
        ]
        label = labels[0] if len(labels) == 1 else "boundary"
        score.update(
            start_delivered_tokens=previous,
            end_delivered_tokens=end,
            phase=label,
            promotions=worker["selected_pairs"],
            proposed_pairs=worker["proposed_pairs"],
            trigger=record.get("trigger"),
            time_ns=record["time_ns"],
        )
        windows.append(score)
        for key in ("selections", "current_cold", "anchor_cold", "promotions"):
            totals[label][key] += score[key]
        totals[label]["windows"] += 1
        for name, value in worker["layers"].items():
            for candidate, victim in value["pairs"]:
                if victim not in current[name] or candidate in current[name]:
                    raise ValueError(
                        "promotion receipt disagrees with reconstructed map"
                    )
                current[name].remove(victim)
                current[name].add(candidate)
        previous = end
    return dict(
        receipt=str(path),
        profile=prepared["profile"],
        checkpoint=prepared["checkpoint"],
        windows=windows,
        phases=dict(totals),
        unobserved_tail_tokens=phases[-1][2] - previous,
        interpretation="routing miss difference against learned anchor, not latency regret",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump([analyze(p, args.profile) for p in args.receipts], stream, indent=2)


if __name__ == "__main__":
    main()
