"""Deferred policy windows and completed promotion lifetimes in serving receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.moe.summarize_expert_cache_serving import distribution, summarize
from benchmarks.moe.summarize_expert_health import records


def churn(rows, *, tokens, initial_hot=None):
    """Count real movements; unfinished lifetimes remain right-censored."""
    active, seen, evictions, ended = {}, set(), Counter(), []
    hot = {n: set(ids) for n, ids in (initial_hot or {}).items()}
    repeated, promotions, copied = 0, 0, 0
    demand = Counter()
    previously_evicted = set()
    for record in rows:
        if record["kind"] != "maintenance":
            continue
        worker = record["receipt"]["worker"]
        if worker["baseline"]:
            continue
        position = record["output_tokens_so_far"]
        copied += worker["copy_bytes"]
        for observation in worker.get("policy_observations", ()):
            for layer, value in observation["layers"].items():
                for expert, count in enumerate(value["counts"]):
                    if (layer, expert) in previously_evicted:
                        demand[layer, expert] += count
        for layer, value in worker["layers"].items():
            hits = dict(value["evicted_hits"])
            for candidate, victim in value["pairs"]:
                key = layer, victim
                if key in active:
                    ended.append(
                        dict(
                            layer=layer,
                            expert=victim,
                            hits=hits[victim],
                            delivered_tokens=position - active.pop(key),
                        )
                    )
                evictions[key] += 1
                previously_evicted.add(key)
                key = layer, candidate
                repeated += key in seen
                seen.add(key)
                active[key] = position
                promotions += 1
                if layer in hot:
                    if victim not in hot[layer] or candidate in hot[layer]:
                        raise ValueError(
                            "movement receipt disagrees with initial placement"
                        )
                    hot[layer].remove(victim)
                    hot[layer].add(candidate)
    return dict(
        promotions=promotions,
        promotions_per_1000_tokens=promotions * 1000 / tokens if tokens else None,
        copy_bytes=copied,
        re_promotions=repeated,
        completed_promoted_lifetimes=len(ended),
        zero_hit_completed_lifetimes=sum(x["hits"] == 0 for x in ended),
        completed_lifetimes=ended,
        active_right_censored=len(active),
        evictions=[
            dict(layer=n, expert=e, count=c) for (n, e), c in sorted(evictions.items())
        ],
        observed_demand_after_first_eviction=[
            dict(layer=n, expert=e, selections=c)
            for (n, e), c in sorted(demand.items())
        ],
        demand_observation_available=any(
            r["kind"] == "maintenance"
            and r["receipt"]["worker"].get("policy_observations")
            for r in rows
        ),
        final_membership_change={
            n: len(set(initial_hot[n]) - ids) / len(ids) if ids else 0
            for n, ids in hot.items()
        },
    )


def analyze(path, *, profile=None):
    base = summarize(path)
    rows = records(path)
    initial = None
    if profile is not None:
        artifact = json.loads(profile.read_text())
        if artifact["hash"] != base["complete"]["profile"]:
            raise ValueError("initial profile differs from serving receipt")
        initial = {n: p["hbm_expert_ids"] for n, p in artifact["placements"].items()}
    checkpoints, replays, coalesced, copies, readbacks, history_ms = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    windows = {}
    for row in rows:
        if row["kind"] != "maintenance":
            continue
        worker = row["receipt"]["worker"]
        if h := worker.get("history"):
            checkpoints.append(h["checkpoints"])
            replays.append(h["replayed_windows"])
            coalesced.append(h["coalesced_checkpoints"])
            copies.extend(x["copy_us"] for x in h.get("observations", ()))
            if "readback_us" in h:
                readbacks.append(h["readback_us"])
            history_ms.append(worker["stages_ns"]["history"] / 1e6)
        for observation in worker.get("policy_observations", ()):
            for name, value in observation["layers"].items():
                windows[name] = value["window"]
    return dict(
        serving=base,
        history=dict(
            recorded_health_checkpoints=sum(
                r["kind"] == "health_probe" and r["receipt"].get("history") is not None
                for r in rows
            ),
            # A healthy tail may never reach maintenance. Event samples cover
            # only retained slots, not overwritten or still-unconsumed cuts.
            checkpoints=sum(checkpoints),
            checkpoint_scope="cuts consumed at completed maintenance",
            replayed_windows=sum(replays),
            coalesced_checkpoints=sum(coalesced),
            checkpoint_event_us=distribution(copies),
            checkpoint_event_scope="retained slots read at completed maintenance",
            readback_event_us=distribution(readbacks),
            decode_and_policy_replay_ms=distribution(history_ms),
            final_recorded_policy_windows=windows,
        ),
        churn=churn(rows, tokens=base["serving"]["output_tokens"], initial_hot=initial),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", type=Path, nargs="+")
    p.add_argument("--profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    results = {str(path): analyze(path, profile=args.profile) for path in args.receipts}
    with args.output.open("x") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
