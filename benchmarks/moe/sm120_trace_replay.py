"""Replay real canonical routes through one native SM120 operator.

Activations are synthetic. Decisions come from causal offline b12x replay, with
one static learned initial population. Timing adds graph event intervals and
complete transaction wall time; it excludes trace loading, route H2D, policy
calculation and engine scheduling. It is not full-model or serving latency.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from b12x.testing.residency_replay import read_trace, replay, trace_digest
from .sm120_residency_poc import Experiment, load_layer
from .sm120_residency_spectrum import Graphs, gpu_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--trace-layer", default="layer.0")
    p.add_argument("--prefix", default="model.language_model.layers.0.mlp.experts")
    p.add_argument("--hot", type=int, default=256)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    trace = next(t for t in read_trace(a.trace) if t.layer == a.trace_layer)
    calls = [c for c in trace.calls if c.split == "test"]
    if any(len(c.ids) != 1 or len(c.ids[0]) != len(calls[0].ids[0]) for c in calls):
        raise ValueError("this paired physical replay requires C1 and fixed top-k")
    policy = replay(trace, budget=a.hot, window=a.window)
    changes = {w["end"]: w["pairs"] for w in policy["windows"] if w["pairs"]}
    train = Counter()
    for c in trace.calls:
        if c.split == "train":
            train.update(c.counts)
    initial = tuple(sorted(range(trace.experts), key=lambda e: (-train[e], e))[: a.hot])
    source, digest = load_layer(a.checkpoint, a.prefix, trace.experts)
    if (source["w2"].shape[1], source["w13"].shape[1] // 2) != (
        trace.hidden,
        trace.intermediate,
    ):
        raise ValueError("trace and checkpoint geometry differ")
    record = dict(
        command=sys.argv,
        activation_seed=0,
        status="running",
        trace_sha256=trace_digest(a.trace),
        checkpoint_fields_sha256=digest,
        scope=__doc__,
        gpu_before=gpu_snapshot(),
        initial_hot=initial,
        policy=policy,
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(
                [*Path("b12x").rglob("*.py"), *Path("benchmarks/moe").glob("sm120*.py")]
            )
        },
    )
    record["policy"].pop("misses")
    es = {}
    gs = {}
    rows = []
    path = a.output / "results.json"
    path.write_text(json.dumps(record, indent=2))
    try:
        for name in ("static", "exchange", "canonical"):
            e = Experiment(
                source,
                hot=a.hot,
                capacity=128,
                topk=len(calls[0].ids[0]),
                initial_hot=initial,
                canonical_backing=name == "canonical",
                backing_write_combined=False,
                journal_write_combined=False,
                cache_dir=a.output / "cache",
            )
            es[name] = e
            gs[name] = Graphs(e, 1)
        for e in es.values():
            e.a.copy_(es["static"].a)
        pairs = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in es
        }
        for i, call in enumerate(calls):
            ids = torch.tensor(call.ids, dtype=torch.int64)
            for name in list(es) if i % 2 == 0 else list(es)[::-1]:
                e, g = es[name], gs[name]
                g.inputs(ids)
                cold = sum(
                    n
                    for expert, n in call.counts.items()
                    if e.updates.snapshot().expert_map[expert][0]
                )
                start, end = pairs[name]
                start.record()
                g.graphs["static"].replay()
                end.record()
                end.synchronize()
                graph_us = start.elapsed_time(end) * 1000
                transaction_us = 0.0
                moved = changes.get(i + 1, ()) if name != "static" else ()
                if moved:
                    assert len(moved) == 1
                    before = e.updates.snapshot()
                    begin = time.perf_counter()
                    if name == "exchange":
                        e.updates.exchange(moved, expected=before, quiescent=True)
                    else:
                        e.updates.promote(*moved[0], expected=before, quiescent=True)
                    transaction_us = (time.perf_counter() - begin) * 1e6
                validation = (
                    g.validate(allocator=True)
                    if i in (0, len(calls) - 1)
                    else g.validate()
                    if changes.get(i + 1)
                    else None
                )
                row = dict(
                    arm=name,
                    invocation=i,
                    request=call.request,
                    workload=call.workload,
                    cold_selections=cold,
                    graph_us=graph_us,
                    transaction_us=transaction_us,
                    pairs=moved,
                    validation=validation,
                )
                rows.append(row)
                with (a.output / "replays.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
            if i % 128 == 0:
                print("invocations", i, flush=True)
        totals = {}
        for name in es:
            arm = [r for r in rows if r["arm"] == name]
            totals[name] = {
                key: sum(r[key] for r in arm)
                for key in ("cold_selections", "graph_us", "transaction_us")
            }
            totals[name]["operator_plus_transaction_us"] = (
                totals[name]["graph_us"] + totals[name]["transaction_us"]
            )
        record.update(status="passed", totals=totals, gpu_after=gpu_snapshot())
    except BaseException:
        record.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        for g in gs.values():
            g.close()
        for e in es.values():
            e.close()
        path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
