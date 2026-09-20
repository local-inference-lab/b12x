"""Explain bounded restoration using recorded counts and unchanged policy rules.

Eligibility beyond a layer's proposal cap is counterfactual, not a rejection
executed by the serving controller. Missing experts receive one primary reason.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.moe.analyze_residency_anchor import load_anchor, score_placements
from benchmarks.moe.summarize_expert_health import records


def classify_layer(*, counts, scores, hot, anchor, protected, minimum_count,
                   margin, layer_pairs, proposed, selected, allow=True):
    hot, anchor, protected = set(hot), set(anchor), set(protected)
    missing = anchor - hot
    reasons = {e: "unobserved" if counts[e] == 0 else "minimum_count"
               for e in missing if counts[e] < minimum_count}
    candidates = sorted(missing - reasons.keys(), key=lambda e: (-scores[e], e))
    victims = sorted(hot - anchor - protected, key=lambda e: (scores[e], e))
    eligible = []
    for i, candidate in enumerate(candidates):
        if i >= len(victims):
            reasons[candidate] = "protected_victim" if protected & (hot - anchor) else "no_victim"
        elif scores[candidate] - scores[victims[i]] < margin:
            reasons[candidate] = "score_margin"
        else:
            eligible.append((candidate, victims[i]))
    expected = eligible[:layer_pairs] if allow else []
    if proposed is not None and tuple(map(tuple, proposed)) != tuple(expected):
        raise ValueError("recorded anchor proposals differ from reconstructed policy")
    selected = set(map(tuple, selected))
    if not selected <= set(expected):
        raise ValueError("selected restoration is not a proposed pair")
    for i, pair in enumerate(eligible):
        reasons[pair[0]] = ("anchor_gate" if not allow else
                            "per_layer_cap" if i >= layer_pairs else
                            "selected" if pair in selected else
                            "eligible_unselected" if proposed is None else "global_budget")
    denominator = sum(counts[e] for e in anchor)
    uncovered = sum(counts[e] for e in missing)
    return dict(
        missing=len(missing), non_anchor_residents=len(hot - anchor),
        policy_eligible=len(eligible), layer_proposed=len(expected),
        selected=len(selected), protected_victims=sorted(protected & (hot - anchor)),
        reasons=dict(Counter(reasons.values())),
        experts=[dict(expert=e, count=counts[e], score=scores[e], reason=reasons[e])
                 for e in sorted(missing, key=lambda e: (-counts[e], e))],
        anchor_selections=denominator, missing_anchor_selections=uncovered,
        traffic_weighted_coverage=(denominator - uncovered) / denominator if denominator else None,
        offline_pairs=dict(
            unchanged=eligible[:layer_pairs],
            no_margin=list(zip(candidates, victims, strict=False))[:layer_pairs],
            recent_count=list(zip(sorted(candidates, key=lambda e: (-counts[e], e)),
                                  sorted(victims, key=lambda e: (counts[e], e)), strict=False))[:layer_pairs],
            direct_restore=list(zip(sorted(missing, key=lambda e: (-counts[e], e)),
                                    sorted(hot - anchor, key=lambda e: (counts[e], e)), strict=False))[:layer_pairs],
        ),
    )


def analyze(path, profile, *, minimum_count=2, margin=1, layer_pairs=2):
    rows = records(path)
    if rows[-1]["kind"] != "complete":
        raise ValueError("recovery analysis requires a complete serving receipt")
    prepared = next(r["status"] for r in rows if r["kind"] == "prepared")
    baseline = next(r["receipt"]["worker"] for r in rows
                    if r["kind"] == "maintenance" and r["receipt"]["worker"]["baseline"])
    configs = baseline.get("policy_configs", {})
    plans = load_anchor(profile, prepared)
    anchor = {n: set(p.hbm_expert_ids) for n, p in plans.items()}
    hot = {n: set(ids) for n, ids in anchor.items()}
    requests = [r for r in rows if r["kind"] == "request"]
    starts = sorted((min(r["start_wall_ns"] for r in requests if r["workload"] == name), name)
                    for name in {r["workload"] for r in requests})
    windows = []
    for row in rows:
        if row["kind"] != "maintenance" or row["receipt"]["worker"]["baseline"]:
            continue
        worker = row["receipt"]["worker"]
        observations = worker.get("policy_observations")
        if not observations or len(observations) != 1:
            raise ValueError("recovery analysis requires full diagnostics with history disabled")
        values = observations[0]["layers"]
        counts = {n: v["counts"] for n, v in values.items()}
        score = score_placements(counts, hot, anchor)
        if score["current_cold"] != worker["cold_selections"]:
            raise ValueError("reconstructed cache differs from the recorded cold count")
        phase = max((v for v in starts if v[0] <= row["time_ns"]), default=(0, "warmup"))[1]
        layers = {}
        actual = worker.get("movement_mode") == "recenter"
        for n, v in values.items():
            config = configs.get(n, {})
            layers[n] = classify_layer(
                counts=v["counts"], scores=v["scores"], hot=hot[n], anchor=anchor[n],
                protected=(*v["protected"], *v.get("recenter_protected", ())),
                minimum_count=config.get("minimum_cold_selections", minimum_count),
                margin=config.get("minimum_score_gain", margin),
                layer_pairs=config.get("max_pairs", layer_pairs),
                proposed=v["candidates"] if actual else None,
                selected=v["selected"] if actual else (),
                allow=worker["anchor"]["anchor_better"] if actual else True,
            )
        denominator = sum(sum(counts[n][e] for e in ids) for n, ids in anchor.items())
        missing_demand = sum(sum(counts[n][e] for e in ids - hot[n]) for n, ids in anchor.items())
        summary = Counter()
        for value in layers.values():
            summary.update(value["reasons"])
        windows.append(dict(
            time_ns=row["time_ns"], phase=phase, delivered_tokens=row["output_tokens_so_far"],
            mode=worker.get("movement_mode", "adapt"), trigger=row.get("trigger"),
            restoration_diagnostic="actual" if actual else "hypothetical",
            score=score, anchor_selections=denominator, missing_anchor_selections=missing_demand,
            traffic_weighted_coverage=(denominator-missing_demand)/denominator if denominator else None,
            blockers=dict(summary), layers=layers, proposed=worker["proposed_pairs"],
            selected=worker["selected_pairs"], backlog=worker.get("proposal_backlog"),
            gate=worker.get("anchor"), copy_bytes=worker["copy_bytes"],
        ))
        for n, value in worker["layers"].items():
            for candidate, victim in value["pairs"]:
                if candidate in hot[n] or victim not in hot[n]:
                    raise ValueError("recorded exchange disagrees with resident set")
                hot[n].remove(victim)
                hot[n].add(candidate)
    return dict(receipt=str(path), windows=windows, policy_configs=configs,
                fallback_config=dict(minimum_count=minimum_count, margin=margin, layer_pairs=layer_pairs),
                interpretation="routing coverage and one primary blocker per missing expert; no latency prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=2)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--layer-pairs", type=int, default=2)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(analyze(args.receipt, args.profile, minimum_count=args.minimum_count,
                          margin=args.margin, layer_pairs=args.layer_pairs), stream, indent=2)


if __name__ == "__main__":
    main()
