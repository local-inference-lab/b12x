"""Paired movement, anchor coverage and recovery diagnostics from serving receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.moe.analyze_residency_anchor import load_anchor
from benchmarks.moe.summarize_expert_health import compare, records
from benchmarks.moe.summarize_expert_history import churn


def analyze(baseline, candidate, profile):
    paired = compare(baseline, candidate)
    rows = records(candidate)
    prepared = next(r["status"] for r in rows if r["kind"] == "prepared")
    plans = load_anchor(profile, prepared)
    anchor = {n: set(p.hbm_expert_ids) for n, p in plans.items()}
    hot = {n: set(ids) for n, ids in anchor.items()}
    resident = sum(map(len, hot.values()))
    requests = [r for r in rows if r["kind"] == "request"]
    starts = sorted(
        (min(r["start_wall_ns"] for r in requests if r["workload"] == name), name)
        for name in paired["candidate"]["workloads"]
    )

    def phase(time):
        return max((p for p in starts if p[0] <= time), default=(0, "warmup"))[1]

    def overlap():
        missing = sum(len(anchor[n] - ids) for n, ids in hot.items())
        return dict(
            missing_anchor=missing,
            anchor_overlap=1 - missing / resident if resident else 1.0,
        )

    movement, probes = [], []
    for row in rows:
        if row["kind"] == "maintenance":
            worker = row["receipt"]["worker"]
            if worker["baseline"]:
                continue
            before = overlap()
            for name, layer in worker["layers"].items():
                for candidate_id, victim in layer["pairs"]:
                    if victim not in hot[name] or candidate_id in hot[name]:
                        raise ValueError("promotion disagrees with reconstructed map")
                    hot[name].remove(victim)
                    hot[name].add(candidate_id)
            movement.append(
                dict(
                    time_ns=row["time_ns"],
                    phase=phase(row["time_ns"]),
                    delivered_tokens=row["output_tokens_so_far"],
                    trigger=row.get("trigger"),
                    mode=worker.get("movement_mode", "adapt"),
                    selected=worker["selected_pairs"],
                    proposed=worker["proposed_pairs"],
                    copy_bytes=worker["copy_bytes"],
                    backlog=worker.get("proposal_backlog"),
                    anchor=worker.get("anchor"),
                    before=before,
                    after=overlap(),
                    engine_ms=row["receipt"]["engine_wall_ns"] / 1e6,
                    apply_ms=worker["stages_ns"].get("apply", 0) / 1e6,
                )
            )
        elif row["kind"] == "health_probe":
            summary = row["receipt"]["summary"]
            if "anchor_cold_selections" not in summary:
                continue
            total = summary["selections"]
            cold, prior = summary["cold_selections"], summary["anchor_cold_selections"]
            layers = summary["layers"]
            probes.append(
                dict(
                    time_ns=row["time_ns"],
                    phase=phase(row["time_ns"]),
                    delivered_tokens=row["output_tokens_so_far"],
                    selections=total,
                    current_cold=cold,
                    anchor_cold=prior,
                    advantage=(cold - prior) / total if total else None,
                    layers_favoring_anchor=sum(
                        v["cold_selections"] > v["anchor_cold_selections"]
                        for v in layers.values()
                    ),
                    layers=layers,
                    assessment=row["assessment"],
                    generation=summary["generation"],
                    **overlap(),
                )
            )

    phases = {}
    for start, name in starts:
        updates = [r for r in movement if r["phase"] == name]
        samples = [r for r in probes if r["phase"] == name]
        preceding = [r for r in movement if r["time_ns"] < start]
        initial_overlap = preceding[-1]["after"]["anchor_overlap"] if preceding else 1.0
        indications = [r for r in samples if r["assessment"].get("anchor_better")]
        restores = [r for r in updates if r["mode"] == "recenter" and r["selected"]]
        # A probe is an interval ending at its completion. Boundary probes may
        # include the preceding phase; raw series retain that ambiguity.
        phases[name] = dict(
            initial_anchor_overlap=initial_overlap,
            sampled_selections=sum(r["selections"] for r in samples),
            sampled_current_cold=sum(r["current_cold"] for r in samples),
            sampled_anchor_cold=sum(r["anchor_cold"] for r in samples),
            first_anchor_indication_ms=(indications[0]["time_ns"] - start) / 1e6
            if indications
            else None,
            first_restore_ms=(restores[0]["time_ns"] - start) / 1e6
            if restores
            else None,
            restored_pairs=sum(r["selected"] for r in restores),
            declined_recenter=sum(
                r["mode"] == "recenter" and not r["selected"] for r in updates
            ),
            triggers=dict(Counter(r["trigger"] for r in updates)),
            final_overlap=updates[-1]["after"] if updates else None,
            overlap_recovery_ms={
                str(threshold): 0.0
                if initial_overlap >= threshold
                else next(
                    (
                        (r["time_ns"] - start) / 1e6
                        for r in updates
                        if r["before"]["anchor_overlap"]
                        < threshold
                        <= r["after"]["anchor_overlap"]
                    ),
                    None,
                )
                for threshold in (0.90, 0.95)
            },
        )
    return dict(
        paired=paired,
        phases=phases,
        maintenance=movement,
        anchor_probes=probes,
        churn=churn(
            rows,
            tokens=paired["candidate"]["serving"]["output_tokens"],
            initial_hot=anchor,
        ),
        probe_scope="intervals ending in phase; boundary intervals can contain preceding traffic",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(
            {str(p): analyze(args.baseline, p, args.profile) for p in args.candidates},
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
