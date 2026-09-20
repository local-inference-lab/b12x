"""Summarize complete serving receipts without inferring unobserved routing.

Delivery gaps are client-visible intervals, not CUDA iteration timings. Promotion
hits are cumulative; only evicted promotions have a completed useful-hit lifetime.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(p):
        index = (len(values) - 1) * p
        lower = int(index)
        return values[lower] + (
            values[min(lower + 1, len(values) - 1)] - values[lower]
        ) * (index - lower)

    return dict(
        count=len(values),
        mean=statistics.mean(values),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        maximum=values[-1],
    )


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        not records
        or records[-1]["kind"] != "complete"
        or any(r["kind"] == "failure" for r in records)
    ):
        raise ValueError(f"{path}: serving receipt did not complete successfully")
    configuration = next(r for r in records if r["kind"] == "configuration")
    serving = next(r for r in records if r["kind"] == "serving")
    requests = sorted(
        (r for r in records if r["kind"] == "request"), key=lambda r: r["index"]
    )
    # Normalize compact worker-local receipts without inventing route histories.
    # Engine wall time is the blocked scheduling interval. Client roundtrip and
    # nested device-drain/worker stages remain distinct diagnostics.
    for record in records:
        if record["kind"] != "maintenance":
            continue
        value, worker = record["receipt"], record["receipt"]["worker"]
        value.update(
            baseline=worker["baseline"],
            status="resumed",
            pause_to_resume_wall_ns=value["engine_wall_ns"],
            stages_ns={
                **{"engine_" + n: t for n, t in value["engine_stages_ns"].items()},
                **{"worker_" + n: t for n, t in worker["stages_ns"].items()},
            },
        )
    epochs = [
        r
        for r in records
        if r["kind"] in ("epoch", "maintenance") and not r["receipt"]["baseline"]
    ]
    pauses = [
        (r["time_ns"] - r["receipt"]["total_wall_ns"], r["time_ns"]) for r in epochs
    ]
    stages, cold, selections, promotions, copy_bytes, skipped = (
        defaultdict(list),
        0,
        0,
        0,
        0,
        0,
    )
    last_outcomes, evictions, series = {}, [], []
    for r in epochs:
        value = r["receipt"]
        if value["status"] != "resumed":
            raise ValueError("epoch did not acknowledge and resume")
        for name, ns in value["stages_ns"].items():
            stages[name].append(ns / 1e6)
        if r["kind"] == "maintenance":
            d = value["worker"]
            n, misses = d["selections"], d["cold_selections"]
            outcomes = {
                name: {
                    "observed_hits_after_promotion": row["hits"],
                    "evicted_promotion_hits": row["evicted_hits"],
                }
                for name, row in d["layers"].items()
            }
        else:
            d = value["decision"]
            n = sum(sum(v["decision"]["counts"]) for v in d["layers"])
            misses = sum(v["decision"]["cold_selections"] for v in d["layers"])
            outcomes = value["outcomes"]
        promotions += d["selected_pairs"]
        copy_bytes += d["copy_bytes"]
        skipped += d["proposed_pairs"] - d["selected_pairs"]
        cold += misses
        selections += n
        last_outcomes.update(outcomes)
        evictions.extend(
            hits
            for layer in outcomes.values()
            for _, hits in layer["evicted_promotion_hits"]
        )
        series.append(
            dict(
                output_tokens=r.get("output_tokens_so_far"),
                next_check_tokens=r.get("next_check_tokens"),
                cold_fraction=misses / n if n else None,
                selections=n,
                promotions=d["selected_pairs"],
                copy_bytes=d["copy_bytes"],
                health=d.get("health"),
                pause_ms=value["pause_to_resume_wall_ns"] / 1e6,
            )
        )

    def request_metrics(rows):
        gaps, ordinary, adjacent, coalesced = [], [], [], 0
        for r in rows:
            coalesced += sum(e["new_tokens"] != 1 for e in r["events"])
            for a, b in zip(r["events"], r["events"][1:], strict=False):
                gap = (b["ns"] - a["ns"]) / 1e6
                gaps.append(gap)
                start = r.get("start_wall_ns")
                if start is not None:
                    overlaps = any(
                        start + a["ns"] < end and start + b["ns"] > begin
                        for begin, end in pauses
                    )
                    (adjacent if overlaps else ordinary).append(gap)
        starts = [r["start_wall_ns"] for r in rows if "start_wall_ns" in r]
        span = (
            max(r["start_wall_ns"] + r["elapsed_ns"] for r in rows) - min(starts)
            if len(starts) == len(rows)
            else None
        )
        return dict(
            requests=len(rows),
            output_tokens=sum(r["output_tokens"] for r in rows),
            aggregate_tokens_per_s=sum(r["output_tokens"] for r in rows) * 1e9 / span
            if span
            else None,
            ttft_ms=distribution([r["ttft_ns"] / 1e6 for r in rows]),
            request_decode_tokens_per_s=distribution(
                [r["decode_tokens_per_s"] for r in rows if r["decode_tokens_per_s"]]
            ),
            delivery_gap_ms=distribution(gaps),
            ordinary_delivery_gap_ms=distribution(ordinary),
            epoch_overlapping_delivery_gap_ms=distribution(adjacent),
            coalesced_delivery_events=coalesced,
        )

    return dict(
        receipt=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arguments=configuration["arguments"],
        source=configuration.get("source_receipt"),
        serving=serving,
        requests=request_metrics(requests),
        workloads={
            label: request_metrics([r for r in requests if r["workload"] == label])
            for label in dict.fromkeys(r["workload"] for r in requests)
        },
        epochs=dict(
            count=len(epochs),
            pause_ms=distribution(
                [r["receipt"]["pause_to_resume_wall_ns"] / 1e6 for r in epochs]
            ),
            total_pause_ms=sum(r["receipt"]["pause_to_resume_wall_ns"] for r in epochs)
            / 1e6,
            stages_ms={name: distribution(values) for name, values in stages.items()},
            promotions=promotions,
            copy_bytes=copy_bytes,
            skipped_pairs=skipped,
            observed_selections=selections,
            observed_cold_fraction=cold / selections if selections else None,
            useful_hits_after_promotion=sum(
                v["observed_hits_after_promotion"] for v in last_outcomes.values()
            ),
            completed_promotion_evictions=len(evictions),
            zero_hit_completed_evictions=sum(hits == 0 for hits in evictions),
            series=series,
            no_op_pause_ms=distribution(
                [v["pause_ms"] for v in series if v["promotions"] == 0]
            ),
            promotion_pause_ms=distribution(
                [v["pause_ms"] for v in series if v["promotions"] > 0]
            ),
        ),
        output_token_sha256=hashlib.sha256(
            json.dumps([r["token_ids"] for r in requests]).encode()
        ).hexdigest(),
        complete=records[-1]["status"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.output.open("x") as stream:
        json.dump([summarize(path) for path in args.receipts], stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
