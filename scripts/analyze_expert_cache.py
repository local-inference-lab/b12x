#!/usr/bin/env python3
"""Analyze invocation traces without importing a device backend.

All policy parameters are experimental inputs. Learned population uses only
segments labeled train. Test segments retain their declared serial ordering.
"""

import argparse
from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path

from b12x.moe.residency import ResidencyCacheConfig
from b12x.testing.residency_replay import (
    locality,
    read_trace,
    replay,
    trace_digest,
    reuse_diagnostics,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budgets", type=int, nargs="+", required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[4, 16, 128])
    p.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 128, 512]
    )
    p.add_argument(
        "--policies",
        nargs="+",
        choices=["static", "lru", "lfu", "decayed_lfu", "b12x"],
        default=["static", "lru", "lfu", "decayed_lfu", "b12x"],
    )
    p.add_argument(
        "--initial",
        nargs="+",
        choices=["learned", "positional"],
        default=["learned", "positional"],
    )
    p.add_argument("--workloads", nargs="+")
    p.add_argument("--train-workloads", nargs="+")
    p.add_argument("--layers", nargs="+")
    p.add_argument("--max-pairs", type=int, default=1)
    p.add_argument("--minimum-cold-selections", type=int, default=2)
    p.add_argument("--minimum-score-gain", type=int, default=2)
    p.add_argument("--minimum-residency-windows", type=int, default=1)
    p.add_argument(
        "--details", action="store_true", help="retain per-miss and per-window records"
    )
    a = p.parse_args()
    if min(a.horizons) < 1 or min(a.windows) < 1 or min(a.budgets) < 0:
        p.error("horizons/windows must be positive; budgets nonnegative")
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "b12x-cache-replay-v1",
        "trace_sha256": trace_digest(a.trace),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        },
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parent.parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "b12x/testing/residency_replay.py",
                "b12x/moe/residency/policy.py",
                "b12x/moe/residency/contracts.py",
                "scripts/analyze_expert_cache.py",
            )
        },
        "timings": None,
        "status": "running",
    }
    path = a.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    layers = read_trace(a.trace)
    if a.layers and set(a.layers) - {t.layer for t in layers}:
        p.error("requested layer is absent from the trace")
    workloads = {c.workload for t in layers for c in t.calls}
    if set((a.workloads or []) + (a.train_workloads or [])) - workloads:
        p.error("requested workload is absent from the trace")
    with (a.output / "results.jsonl").open("w") as out:
        for trace in layers:
            if a.layers and trace.layer not in a.layers:
                continue
            if max(a.budgets) > trace.experts:
                raise ValueError("budget exceeds layer expert count")
            trace = replace(
                trace,
                calls=tuple(
                    c
                    for c in trace.calls
                    if (not a.workloads or c.workload in a.workloads)
                    if c.split == "test"
                )
                + tuple(
                    c
                    for c in trace.calls
                    if c.split == "train"
                    and (
                        not (a.train_workloads or a.workloads)
                        or c.workload in (a.train_workloads or a.workloads)
                    )
                ),
            )
            config = ResidencyCacheConfig(
                max_pairs=a.max_pairs,
                minimum_cold_selections=a.minimum_cold_selections,
                minimum_score_gain=a.minimum_score_gain,
                minimum_residency_windows=a.minimum_residency_windows,
                phase=trace.phase,
            )
            test = [c for c in trace.calls if c.split == "test"]
            out.write(
                json.dumps(
                    dict(
                        kind="locality",
                        layer=trace.layer,
                        result=locality(
                            test, trace.experts, horizons=a.horizons, budgets=a.budgets
                        ),
                    )
                )
                + "\n"
            )
            future = reuse_diagnostics(test, a.horizons)
            for initial in a.initial:
                for budget in a.budgets:
                    for policy in a.policies:
                        for window in (
                            [a.windows[0]] if policy == "static" else a.windows
                        ):
                            result = replay(
                                trace,
                                budget=budget,
                                window=window,
                                policy=policy,
                                initial=initial,
                                config=config,
                                horizons=a.horizons,
                                future_use=future,
                            )
                            if not a.details:
                                miss_rows = result.pop("misses")
                                result["miss_reuse"] = {
                                    h: dict(
                                        eligible=sum(
                                            m["horizons"][h] is not None
                                            for m in miss_rows
                                        ),
                                        reused=sum(
                                            m["horizons"][h] is not None
                                            and m["horizons"][h]["touches"] > 0
                                            for m in miss_rows
                                        ),
                                        later_selections=sum(
                                            m["horizons"][h]["selections"]
                                            for m in miss_rows
                                            if m["horizons"][h] is not None
                                        ),
                                    )
                                    for h in a.horizons
                                }
                                result.pop("windows")
                                for row in result["promotions"]:
                                    row.pop("hit_steps")
                            out.write(
                                json.dumps(
                                    dict(
                                        kind="replay",
                                        layer=trace.layer,
                                        config=asdict(config),
                                        result=result,
                                    )
                                )
                                + "\n"
                            )
                            out.flush()
            print(trace.layer, flush=True)
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
