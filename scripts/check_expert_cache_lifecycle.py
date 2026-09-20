#!/usr/bin/env python3
"""Check explicit worker release and post-warmup bounds from repeated serving."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b12x.testing.artifacts import sha256
from benchmarks.moe.summarize_expert_cache_serving import summarize


def check(cycles, resources, *, tolerance_bytes=64 << 20):
    receipts = sorted(cycles.glob("cycle-*.jsonl"))
    if len(receipts) < 3:
        raise ValueError("at least three complete cycles are required")
    summaries = [summarize(p) for p in receipts]
    rows = [json.loads(s) for s in resources.read_text().splitlines()]
    closed = [r for r in rows if r["stage"] == "after_worker_shutdown"]
    if len(closed) != len(receipts):
        raise ValueError("each cycle requires acknowledged worker release")
    for r in closed:
        if any(
            r[n]
            for n in (
                "mapped_bytes",
                "cpu_source_bytes",
                "graph_owners",
                "pending_health",
            )
        ):
            raise ValueError("cache owners remain after shutdown")
    client_path = cycles / "client-resources.jsonl"
    clients = [json.loads(s) for s in client_path.read_text().splitlines()]
    if len(clients) != len(receipts):
        raise ValueError("missing client checkpoints")

    def rss(row):
        value, unit = row["status"]["VmRSS"].split()
        if unit != "kB":
            raise ValueError("unexpected RSS units")
        return int(value) * 1024

    samples = {
        "worker_cuda_live": [r["allocated"] for r in closed],
        "worker_cuda_reserved": [r["reserved"] for r in closed],
        "worker_device_free": [r["device_free"] for r in closed],
        "worker_rss": [rss(r) for r in closed],
        "client_rss": [rss(r) for r in clients],
    }
    for name, values in samples.items():
        if max(values[1:]) - min(values[1:]) > tolerance_bytes:
            raise ValueError(f"{name} exceeds the declared post-warmup bound")
    for name, values in (
        ("descriptors", [r["descriptors"] for r in clients]),
        ("child count", [len(r["children"]) for r in clients]),
    ):
        if max(values[1:]) != min(values[1:]):
            raise ValueError(f"client {name} grows after warmup")
    return dict(
        status="passed",
        cycles=len(receipts),
        samples=samples,
        tolerance_bytes=tolerance_bytes,
        resources_sha256=sha256(resources),
        receipts={str(p): sha256(p) for p in receipts},
        promotions=[s["epochs"]["promotions"] for s in summaries],
        note="Worker processes are reconstructed; only the client repeats in-process. Cached pools and CUDA contexts are recorded separately from released cache owners.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycles", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance-mib", type=int, default=64)
    a = p.parse_args()
    if a.tolerance_mib < 0:
        p.error("tolerance must be nonnegative")
    result = check(a.cycles, a.resources, tolerance_bytes=a.tolerance_mib << 20)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
