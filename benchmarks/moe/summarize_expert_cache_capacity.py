"""Summarize independent engine trials by admitted expert capacity.

Serving wall time already includes control tails. Nested control timers and
scheduler drains remain diagnostics and are never added to elapsed wall time.
"""

import argparse
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
    args = p.parse_args()
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
                runs=runs,
            )
        )
    with args.output.open("x") as stream:
        json.dump(reports, stream, indent=2)


if __name__ == "__main__":
    main()
