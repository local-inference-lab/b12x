"""Summarize independent engine trials by admitted expert capacity.

Serving wall time already includes control tails. Nested control timers and
scheduler drains remain diagnostics and are never added to elapsed wall time.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.moe.summarize_expert_cache_serving import summarize


def samples(values):
    return dict(
        values=values,
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else None,
    )


def replay_static_cold(profile, records):
    """Score immutable placement against a separate, nonmutating route receipt."""
    config = next(r for r in records if r["kind"] == "configuration")
    status = next(r["status"] for r in records if r["kind"] == "prepared")
    if (
        records[-1]["kind"] != "complete"
        or config["arguments"]["control"] != "observe"
        or config["arguments"]["admission"] != "together"
        or config["numerical_recipe"] != "cache_w4a16_bf16_whole_k"
        or profile["identity"]["recipe"]
        != "nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum"
        or status["checkpoint"] != profile["identity"]["checkpoint"]
        or any(r["kind"] in ("epoch", "maintenance") for r in records)
    ):
        raise ValueError(
            "static replay requires completed same-checkpoint observation-only traffic"
        )
    boundaries = [r for r in records if r["kind"] == "routing_boundary"]
    requests = {r["index"]: r for r in records if r["kind"] == "request"}
    if (
        len(boundaries) < 2
        or boundaries[0]["next_request"] != 0
        or boundaries[-1]["next_request"] != len(requests)
    ):
        raise ValueError("route replay does not cover the complete request sequence")
    windows = []
    previous, previous_maps = None, None
    for boundary in boundaries:
        result = boundary["result"]
        if len(result) != 1:
            raise ValueError("route replay requires one worker")
        value = result[0]
        maps = value["slots"]
        if any(s["generation"] != 0 for s in maps.values()) or (
            previous_maps is not None and maps != previous_maps
        ):
            raise ValueError("route replay placement changed")
        previous_maps = maps
        counts = {r["layer"]: r["counts"] for r in value["snapshot"]["layers"]}
        if set(counts) != set(profile["placements"]):
            raise ValueError("route replay layer geometry mismatch")
        if previous is not None:
            cold = total = 0
            for name, now in counts.items():
                placement = profile["placements"][name]
                if len(now) != placement["total_experts"]:
                    raise ValueError("route replay expert geometry mismatch")
                delta = [a - b for a, b in zip(now, previous[1][name], strict=True)]
                if any(n < 0 for n in delta):
                    raise ValueError("route replay counter reset")
                resident = set(placement["hbm_expert_ids"])
                total += sum(delta)
                cold += sum(n for e, n in enumerate(delta) if e not in resident)
            end = boundary["next_request"]
            if end <= previous[0]:
                raise ValueError("route replay request boundary did not advance")
            labels = sorted({requests[i]["workload"] for i in range(previous[0], end)})
            windows.append(
                dict(
                    first_request=previous[0],
                    next_request=end,
                    workloads=labels,
                    selections=total,
                    cold=cold,
                    cold_fraction=cold / total if total else None,
                )
            )
        previous = boundary["next_request"], counts
    total, cold = sum(w["selections"] for w in windows), sum(w["cold"] for w in windows)
    return dict(
        selections=total,
        cold=cold,
        cold_fraction=cold / total if total else None,
        windows=windows,
        kind="separate observation-only canonical-count replay",
    )


def trial(path):
    summary = summarize(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    status = next(r["status"] for r in rows if r["kind"] == "prepared")
    resources = [
        json.loads(line)
        for line in path.with_name(path.stem + "-resources.jsonl")
        .read_text()
        .splitlines()
    ]
    resource_stages = {
        r["stage"]: {
            k: r.get(k)
            for k in (
                "allocated",
                "reserved",
                "peak_allocated",
                "device_free",
                "device_total",
                "mapped_bytes",
                "cpu_source_bytes",
                "graph_owners",
                "pending_health",
                "status",
            )
        }
        for r in resources
    }
    epochs = [
        r["receipt"]["worker"]
        for r in rows
        if r["kind"] == "maintenance" and not r["receipt"]["worker"]["baseline"]
    ]
    backlog = [
        dict(
            proposed_pairs=e["proposed_pairs"],
            selected_pairs=e["selected_pairs"],
            **e.get("proposal_backlog", {}),
        )
        for e in epochs
    ]
    cap = summary["arguments"]["epoch_pairs"]
    summary.update(
        resident_counts={n: r["resident"] for n, r in status["layers"].items()},
        memory=status["memory"],
        load_device_peak_bytes=status["load_device_peak_bytes"],
        resource_stages=resource_stages,
        resource_checkpoints=[
            {
                k: r.get(k)
                for k in (
                    "stage",
                    "time_ns",
                    "allocated",
                    "reserved",
                    "peak_allocated",
                    "device_free",
                    "device_total",
                    "mapped_bytes",
                    "cpu_source_bytes",
                    "graph_owners",
                    "pending_health",
                    "status",
                )
            }
            for r in resources
        ],
        backlog=backlog,
        pair_cap_saturation=sum(e["selected_pairs"] == cap for e in epochs)
        / len(epochs)
        if epochs and cap
        else None,
        shutdown_ns=next(r for r in rows if r["kind"] == "shutdown"),
        health_probe_count=sum(r["kind"] == "health_probe" for r in rows),
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("acceptance", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--routing-receipt", type=Path)
    p.add_argument("--profiles", type=Path, nargs="+", default=[])
    args = p.parse_args()
    profiles = {
        hashlib.sha256(p.read_bytes()).hexdigest(): json.loads(p.read_text())
        for p in args.profiles
    }
    routing = None
    if args.routing_receipt:
        routing = [json.loads(r) for r in args.routing_receipt.read_text().splitlines()]
        routing_output = summarize(args.routing_receipt)["output_token_sha256"]
    reports = []
    for path in args.acceptance:
        acceptance = json.loads(path.read_text())
        if acceptance["status"] != "passed":
            raise ValueError(f"acceptance failed: {path}")
        runs = {
            p.stem: trial(p)
            for p in sorted(path.parent.glob("pair-*.jsonl"))
            if not p.stem.endswith("-resources")
        }
        modes = {}
        for mode in ("static", "adaptive"):
            values = [r for n, r in runs.items() if n.endswith("-" + mode)]
            if not values:
                continue
            modes[mode] = dict(
                overall=samples(
                    [r["serving"]["aggregate_tokens_per_s"] for r in values]
                ),
                workloads={
                    w: samples(
                        [r["workloads"][w]["aggregate_tokens_per_s"] for r in values]
                    )
                    for w in values[0]["workloads"]
                },
            )
        paired = []
        static_cold = None
        if routing is not None:
            if any(r["output_token_sha256"] != routing_output for r in runs.values()):
                raise ValueError("route replay output IDs differ from timed traffic")
            static_cold = replay_static_cold(
                profiles[acceptance["profile_sha256"]], routing
            )
        for name, static in runs.items():
            adaptive = runs.get(name.replace("-static", "-adaptive"))
            if not name.endswith("-static") or adaptive is None:
                continue
            if (
                static["output_token_sha256"] != adaptive["output_token_sha256"]
                or static["resident_counts"] != adaptive["resident_counts"]
            ):
                raise ValueError("paired outputs or initial capacity differ")
            a = static["serving"]["aggregate_tokens_per_s"]
            b = adaptive["serving"]["aggregate_tokens_per_s"]
            paired.append(dict(pair=name, absolute_gain=b - a, relative_gain=b / a - 1))
        reports.append(
            dict(
                acceptance=str(path),
                source=acceptance["source"],
                modes=modes,
                paired=paired,
                static_cold_replay=static_cold,
                runs=runs,
            )
        )
    with args.output.open("x") as stream:
        json.dump(reports, stream, indent=2)


if __name__ == "__main__":
    main()
