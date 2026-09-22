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
    receipts = sorted(cycles.glob("cycle-*.jsonl"), key=lambda p: int(p.stem.split("-")[-1]))
    if len(receipts) < 3:
        raise ValueError("at least three complete cycles are required")
    summaries = [summarize(p) for p in receipts]
    worlds = {s["arguments"].get("tp_size", 1) for s in summaries}
    if len(worlds) != 1:
        raise ValueError("lifecycle cycles changed TP participant count")
    world = worlds.pop()
    if type(world) is not int or world < 1:
        raise ValueError("invalid lifecycle TP participant count")
    rows = [json.loads(s) for s in resources.read_text().splitlines()]
    closed = [r for r in rows if r["stage"] == "after_worker_shutdown"]
    if len(closed) != len(receipts) * world:
        raise ValueError("each cycle requires every TP worker release")
    per_rank = {rank: [] for rank in range(world)}
    assigned = set()
    for path in receipts:
        records = [json.loads(s) for s in path.read_text().splitlines()]
        begin = next(r["time_ns"] for r in records if r["kind"] == "configuration")
        end = next(r["time_ns"] for r in records if r["kind"] == "shutdown")
        releases = [(i, r) for i, r in enumerate(closed) if begin <= r["time_ns"] <= end]
        ranks = []
        for i, r in releases:
            # Historical TP1 receipts predate rank labels. Multi-rank evidence
            # must carry explicit identity; arrival order is not rank order.
            rank = r.get("rank", 0 if world == 1 else None)
            if (type(rank) is not int or rank not in per_rank
                    or r.get("tp_size", world) != world or i in assigned):
                raise ValueError("worker release has invalid or ambiguous TP identity")
            ranks.append(rank)
            per_rank[rank].append(r)
            assigned.add(i)
        if sorted(ranks) != list(range(world)):
            raise ValueError("cycle lacks exactly one release from every TP rank")
    if len(assigned) != len(closed):
        raise ValueError("worker release falls outside its lifecycle cycle")
    for r in closed:
        if any(
            r[n]
            for n in (
                "mapped_bytes",
                "cpu_source_bytes",
                "graph_owners",
                "pending_health",
            )
        ) or r.get("health_host_bytes", 0):
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

    samples = {"client_rss": [rss(r) for r in clients]}
    for rank, releases in per_rank.items():
        suffix = "" if world == 1 else f"/rank{rank}"
        for name, field in (("cuda_live", "allocated"), ("cuda_reserved", "reserved"),
                            ("device_free", "device_free"), ("rss", None)):
            samples[f"worker_{name}{suffix}"] = [
                rss(r) if field is None else r[field] for r in releases]
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
        tp_size=world,
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
