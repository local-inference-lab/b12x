"""Replay real canonical routes through one native SM120 operator.

Activations are synthetic. Decisions come from causal offline policy replay, with
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


def prepare_policy(trace, *, hot, window, policy):
    """Select causal decisions and a training-only initial population.

    Comparison policies remain offline experiments. The physical backend receives
    canonical candidate/victim IDs, independent of its backing-row representation.
    """
    calls = [c for c in trace.calls if c.split == "test"]
    if not calls or any(
        len(c.ids) != 1 or len(c.ids[0]) != len(calls[0].ids[0]) for c in calls
    ):
        raise ValueError("this paired physical replay requires C1 and fixed top-k")
    result = replay(trace, budget=hot, window=window, policy=policy)
    train = Counter()
    for c in trace.calls:
        if c.split == "train":
            train.update(c.counts)
    initial = tuple(sorted(range(trace.experts), key=lambda e: (-train[e], e))[:hot])
    return calls, initial, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--trace-layer", default="layer.0")
    p.add_argument("--prefix", default="model.language_model.layers.0.mlp.experts")
    p.add_argument("--hot", type=int, default=256)
    p.add_argument("--window", type=int, default=16)
    p.add_argument(
        "--policy", choices=["b12x", "lru", "lfu", "decayed_lfu"], default="b12x"
    )
    p.add_argument(
        "--transports",
        nargs="+",
        choices=["exchange", "canonical"],
        default=["exchange", "canonical"],
        help="adaptive transports; a matched static arm is always included",
    )
    p.add_argument(
        "--expected-fields-sha256",
        help="independently verified source-layer fingerprint, when available",
    )
    p.add_argument(
        "--static-backing",
        choices=["exclusive", "canonical"],
        default="exclusive",
        help="match canonical backing to isolate policy from backing-row geometry",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    trace = next(t for t in read_trace(a.trace) if t.layer == a.trace_layer)
    calls, initial, policy = prepare_policy(
        trace, hot=a.hot, window=a.window, policy=a.policy
    )
    changes = {w["end"]: w["pairs"] for w in policy["windows"] if w["pairs"]}
    source, digest = load_layer(a.checkpoint, a.prefix, trace.experts)
    if a.expected_fields_sha256 and digest != a.expected_fields_sha256:
        raise ValueError("checkpoint fields differ from independent source fingerprint")
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
        static_backing=a.static_backing,
        scope=__doc__,
        gpu_before=gpu_snapshot(),
        gpu_samples=[],
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
        for name in ("static", *dict.fromkeys(a.transports)):
            e = Experiment(
                source,
                hot=a.hot,
                capacity=128,
                topk=len(calls[0].ids[0]),
                initial_hot=initial,
                canonical_backing=name == "canonical"
                or (name == "static" and a.static_backing == "canonical"),
                backing_write_combined=False,
                journal_write_combined=False,
                cache_dir=a.output / "cache",
            )
            es[name] = e
            gs[name] = Graphs(e, 1)
        for e in es.values():
            e.a.copy_(es["static"].a)
        captured_pointers = {name: e.pointers() for name, e in es.items()}
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
                if e.pointers() != captured_pointers[name]:
                    raise AssertionError("captured addresses changed across promotions")
                row = dict(
                    arm=name,
                    invocation=i,
                    request=call.request,
                    workload=call.workload,
                    cold_selections=cold,
                    graph_us=graph_us,
                    transaction_us=transaction_us,
                    pairs=moved,
                    generation=e.updates.snapshot().generation,
                    validation=validation,
                )
                rows.append(row)
                with (a.output / "replays.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
            if i % 128 == 0:
                print("invocations", i, flush=True)
            if i % 512 == 0:
                record["gpu_samples"].append(dict(invocation=i, gpu=gpu_snapshot()))
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
            expected_cold = policy["totals"][
                "static_cold_selections" if name == "static" else "cold_selections"
            ]
            if totals[name]["cold_selections"] != expected_cold:
                raise AssertionError(
                    "physical residency differs from causal offline replay"
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
