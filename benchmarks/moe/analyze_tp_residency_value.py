"""Measure observed routing benefit of committed TP exchanges, without replaying policy.

Only complete cumulative-counter windows are used. A saved map is the map before
that epoch's transaction. The final transaction has no inferred future demand.
Routing differences are selection counts, never latency or throughput estimates.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def snapshots(records):
    epochs = []
    for record in records:
        if record["kind"] != "maintenance":
            continue
        receipt = record["receipt"]
        worker = receipt["worker"]
        if worker["status"] != "complete":
            raise ValueError("maintenance did not complete")
        ranks = worker["workers"]
        if not ranks or set(map(int, ranks)) != set(range(len(ranks))):
            raise ValueError("rank inventory is incomplete")
        owner = ranks["0"]
        if epochs and (
            worker["baseline"]
            or owner["checkpoint_id"] != epochs[0]["checkpoint_id"]
            or owner["initial_profile_id"] != epochs[0]["profile_id"]
        ):
            raise ValueError("identity or baseline changed within the run")

        def counts(report):
            rows = report["snapshot"]["layers"]
            result = {r["layer"]: r["counts"] for r in rows if r["phase"] == "decode"}
            if len(result) != len(rows):
                raise ValueError("requires one decode counter row per layer")
            return result

        def slots(report):
            return {
                name: {k: row["slots"][k] for k in ("generation", "expert_map")}
                for name, row in report["layers"].items()
            }

        before, observed = slots(owner), counts(owner)
        for other in ranks.values():
            if (
                other["checkpoint_id"] != owner["checkpoint_id"]
                or other["initial_profile_id"] != owner["initial_profile_id"]
                or counts(other) != observed
                or slots(other) != before
            ):
                raise ValueError("TP ranks disagree on identity, routing or placement")
        if set(before) != set(observed):
            raise ValueError("counter and placement layers differ")
        after = deepcopy(before)
        pairs = []
        for name, row in worker["layers"].items():
            used = set()
            for candidate, victim in row["pairs"]:
                mapping = after[name]["expert_map"]
                if (
                    candidate in used
                    or victim in used
                    or candidate == victim
                    or not 0 <= candidate < len(mapping)
                    or not 0 <= victim < len(mapping)
                    or mapping[candidate][0] != 1
                    or mapping[victim][0] != 0
                ):
                    raise ValueError("invalid or overlapping committed exchange")
                used.update((candidate, victim))
                mapping[candidate] = [0, mapping[victim][1]]
                mapping[victim] = [1, victim]
                pairs.append((name, candidate, victim))
            after[name]["generation"] += bool(row["pairs"])
            if after[name]["generation"] != row["generation"]:
                raise ValueError("committed generation differs from exchanges")
        if len(pairs) != worker["selected_pairs"]:
            raise ValueError("selected pair total differs from layer receipts")
        if epochs and before != epochs[-1]["after"]:
            raise ValueError("placement changed between recorded transactions")
        epochs.append(
            dict(
                before=before,
                after=after,
                counts=observed,
                pairs=pairs,
                worker=worker,
                receipt=receipt,
                time_ns=record["time_ns"],
                checkpoint_id=owner["checkpoint_id"],
                profile_id=owner["initial_profile_id"],
            )
        )
    if not epochs or not epochs[0]["worker"]["baseline"]:
        raise ValueError("initial baseline snapshot is required")
    return epochs


def analyze(records):
    epochs = snapshots(records)
    initial = epochs[0]["before"]
    intervals = []
    for index, (previous, epoch) in enumerate(
        zip(epochs[:-1], epochs[1:], strict=True), 1
    ):
        delta = {}
        current_cold = initial_cold = total = 0
        for name, values in epoch["counts"].items():
            change = [
                b - a for a, b in zip(previous["counts"][name], values, strict=True)
            ]
            if any(v < 0 for v in change):
                raise ValueError("counter reset inside retained observation history")
            delta[name] = change
            total += sum(change)
            current_cold += sum(
                v
                for v, (tier, _) in zip(
                    change, epoch["before"][name]["expert_map"], strict=True
                )
                if tier
            )
            initial_cold += sum(
                v
                for v, (tier, _) in zip(
                    change, initial[name]["expert_map"], strict=True
                )
                if tier
            )
        if (total, current_cold) != (
            epoch["worker"]["selections"],
            epoch["worker"]["cold_selections"],
        ):
            raise ValueError("receipt totals differ from canonical counter deltas")
        intervals.append(
            dict(
                ending_epoch=index,
                counts=delta,
                selections=total,
                current_cold=current_cold,
                initial_map_cold=initial_cold,
                avoided_vs_initial=initial_cold - current_cold,
            )
        )
    transactions = []
    for index, epoch in enumerate(epochs[1:], 1):
        lifetimes = []
        for name, candidate, victim in epoch["pairs"]:
            hits = eviction_demand = windows = positive = 0
            generations = set()
            ended = False
            for interval in intervals[index:]:
                end = interval["ending_epoch"]
                mapping = epochs[end]["before"][name]
                if (
                    mapping["expert_map"][candidate][0] != 0
                    or mapping["expert_map"][victim][0] != 1
                ):
                    ended = True
                    break
                counts = interval["counts"][name]
                hits += counts[candidate]
                eviction_demand += counts[victim]
                windows += 1
                positive += counts[candidate] > counts[victim]
                if counts[candidate]:
                    generations.add(mapping["generation"])
                after = epochs[end]["after"][name]["expert_map"]
                if after[candidate][0] != 0 or after[victim][0] != 1:
                    ended = True
                    break
            lifetimes.append(
                dict(
                    layer=name,
                    promoted=candidate,
                    evicted=victim,
                    observed_windows=windows,
                    useful_windows=positive,
                    useful_generations=len(generations),
                    promoted_hits=hits,
                    evicted_expert_demand=eviction_demand,
                    net_avoided_cold=hits - eviction_demand,
                    right_censored=not ended,
                )
            )
        worker, receipt = epoch["worker"], epoch["receipt"]

        def observed_sum(field):
            if not any(v["observed_windows"] for v in lifetimes):
                return None
            return sum(v[field] for v in lifetimes)

        transactions.append(
            dict(
                epoch=index,
                proposed=worker["proposed_pairs"],
                selected=len(lifetimes),
                backlog=worker["proposal_backlog"],
                aggregate_copy_bytes=worker["copy_bytes"],
                per_rank_copy_bytes=worker["per_rank_copy_bytes"],
                engine_wall_ns=receipt["engine_wall_ns"],
                engine_stages_ns=receipt["engine_stages_ns"],
                promoted_hits=observed_sum("promoted_hits"),
                evicted_expert_demand=observed_sum("evicted_expert_demand"),
                net_avoided_cold=observed_sum("net_avoided_cold"),
                unobserved_pairs=sum(not v["observed_windows"] for v in lifetimes),
                zero_hit_observed_pairs=sum(
                    v["observed_windows"] > 0 and not v["promoted_hits"]
                    for v in lifetimes
                ),
                pairs=lifetimes,
            )
        )
    serving = [r for r in records if r["kind"] == "serving"]
    return dict(
        schema="b12x-tp-residency-marginal-value/v1",
        checkpoint_id=epochs[0]["checkpoint_id"],
        profile_id=epochs[0]["profile_id"],
        ranks=len(epochs[0]["worker"]["workers"]),
        scope="complete decode-counter windows; routing counterfactual, not time saved",
        final_demand_tail="unobserved; final transaction remains right-censored",
        intervals=[
            {k: v for k, v in row.items() if k != "counts"} for row in intervals
        ],
        transactions=transactions,
        serving=serving,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for path in args.receipts:
        data = path.read_bytes()
        results.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                analysis_script_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                analysis=analyze([json.loads(line) for line in data.splitlines()]),
            )
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
