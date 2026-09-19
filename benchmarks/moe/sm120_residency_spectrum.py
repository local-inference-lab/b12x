"""Record native SM120 residency correctness, latency and bounded policy trials.

Real checkpoint weights and synthetic routing exercise a single serialized
layer. CUDA-event timings describe graph replay; policy wall times additionally
include counter snapshots, decisions and journaled exchanges. These are operator
diagnostics, not end-to-end serving throughput or B300 qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.moe.residency import (
    ResidencyCacheConfig,
    ResidencyCacheController,
    ResidencyExchangeSpec,
    RoutingObservationSpec,
)
from .sm120_residency_poc import Experiment, load_layer


def routes(experts, hot, live, topk, cold_count, *, window=0, concentrated=False):
    """Deterministic unique top-k IDs with exactly cold_count cold selections."""
    if min(hot, experts - hot) < topk or not 0 <= cold_count <= live * topk:
        raise ValueError("route fixture needs at least top-k experts in each tier")
    result = []
    for token in range(live):
        cold = (token + 1) * cold_count // live - token * cold_count // live
        start = window * topk + (0 if concentrated else token * topk)
        row = [(start + r) % hot for r in range(topk - cold)]
        row += [hot + (start + r) % (experts - hot) for r in range(cold)]
        # Spread cold routes through original rank order without renaming IDs.
        shift = token % topk
        result.append(row[shift:] + row[:shift])
    return torch.tensor(result, dtype=torch.int64)


def policy_routes(experts, hot, live, topk, pattern, epoch, epochs):
    cold = pattern != "steady_hot" and (
        pattern != "phase_shift" or epoch >= epochs // 2
    )
    return routes(
        experts,
        hot,
        live,
        topk,
        live * topk if cold else 0,
        window=epoch if pattern == "rotating_cold" else 0,
        concentrated=True,
    )


def gpu_snapshot():
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    fields = "uuid,name,driver_version,compute_mode,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active,pcie.link.gen.current,pcie.link.width.current"
    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uuid,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    return dict(zip(fields.split(","), (v.strip() for v in values), strict=True))


def capture(experiment, call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with experiment.session.capture(), torch.cuda.graph(graph):
        call()
    experiment.graphs.append(graph)
    return graph


class Graphs:
    def __init__(self, experiment, live):
        self.e, self.live = experiment, live
        self.bindings = experiment.bind(live)
        observer = moe.bind_routing_profile(
            experiment.counter,
            layer="experiment",
            phase="decode",
            topk_ids=experiment.ids[:live],
        )

        def remap():
            experiment.refresh(live)

        def hot():
            moe.run(binding=self.bindings[0])

        def cold():
            moe.run(binding=self.bindings[1])

        def reduce():
            experiment.invoke(
                experiment.reduce,
                (
                    self.bindings[0].intermediate_cache13,
                    self.bindings[1].intermediate_cache13,
                    experiment.ids,
                    experiment.mapping,
                    experiment.output,
                ),
                (live,),
            )

        def static():
            remap()
            hot()
            cold()
            reduce()

        def profiled():
            static()
            observer.run()

        self.graphs = {
            name: capture(experiment, call)
            for name, call in (
                ("all_vram", lambda: moe.run(binding=self.bindings[2])),
                ("static", static),
                ("profiled", profiled),
                ("remap", remap),
                ("hot_operator", hot),
                ("cold_operator", cold),
                ("ordered_sum", reduce),
                ("counter", observer.run),
            )
        }

    def inputs(self, ids):
        self.e.ids[: self.live].copy_(ids)
        # The all-VRAM control has no extra residency map/sanitizing node.
        # This benchmark generates only in-range canonical IDs.
        self.e.safe_ids[: self.live].copy_(ids)
        self.e.refresh(self.live)
        torch.cuda.synchronize()

    def validate(self, *, allocator=False):
        pointers = self.e.pointers()
        gc.collect()
        before = torch.cuda.memory_stats() if allocator else None
        with kernel_resolution_guard("residency spectrum replay"):
            self.graphs["static"].replay()
            self.graphs["profiled"].replay()
        torch.cuda.synchronize()
        if allocator:
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "allocated_bytes.all.current",
            ):
                if before[key] != after[key]:
                    raise AssertionError(f"replay allocator event: {key}")
        self.graphs["all_vram"].replay()
        torch.cuda.synchronize()
        actual, reference = (
            self.e.output[: self.live].float(),
            self.bindings[2].output.float(),
        )
        if (
            not torch.isfinite(actual).all()
            or not torch.isfinite(reference).all()
            or not torch.count_nonzero(reference)
        ):
            raise AssertionError("native outputs must be finite and nonzero")
        relative = float((actual - reference).norm() / reference.norm())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            )
        )
        if relative > 0.005 or cosine < 0.9999:
            raise AssertionError(
                f"native parity failed: relative_l2={relative}, cosine={cosine}"
            )
        # This gate separates native GEMM rounding from the experimental reducer.
        ids = self.e.ids[: self.live].cpu().tolist()
        mapping = self.e.updates.snapshot().expert_map
        rows = [
            b.intermediate_cache13[: self.live * self.e.ids.shape[1] * self.e.h]
            .view(self.live, self.e.ids.shape[1], self.e.h)
            .cpu()
            .float()
            for b in self.bindings[:2]
        ]
        ordered = torch.zeros_like(actual, device="cpu")
        for token, experts in enumerate(ids):
            for rank, expert in enumerate(experts):
                ordered[token] += rows[mapping[expert][0]][token, rank]
        torch.testing.assert_close(
            actual.cpu().bfloat16(), ordered.bfloat16(), atol=0, rtol=0
        )
        if pointers != self.e.pointers():
            raise AssertionError("captured addresses changed")
        return dict(
            relative_l2=relative,
            cosine=cosine,
            max_abs=float((actual - reference).abs().max()),
            bitwise_equal=torch.equal(actual, reference),
            ordered_sum_exact=True,
            replay_allocator_events=0 if allocator else None,
            stable_pointers=True,
        )

    def close(self):
        torch.cuda.synchronize()
        for graph in self.graphs.values():
            graph.reset()
            self.e.graphs.remove(graph)


class Timer:
    def __init__(self, experiment):
        from cuda.bindings import runtime as cuda

        error, l2 = cuda.cudaDeviceGetAttribute(
            cuda.cudaDeviceAttr.cudaDevAttrL2CacheSize, experiment.device.index
        )
        if error != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"L2 query failed: {error}")
        self.l2_bytes = l2
        self.flush_buffer = torch.empty(
            max(64 * 1024**2, 2 * l2), dtype=torch.uint8, device=experiment.device
        )
        self.flush = capture(experiment, lambda: self.flush_buffer.zero_())

    def sample(self, graph, repeats, *, evict=False):
        # Cache-scrub time is excluded. Paired events are preallocated before
        # submission; they measure each target replay after the scrub completes.
        pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats if evict else 1)
        ]
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        if evict:
            for start, end in pairs:
                self.flush.replay()
                start.record()
                graph.replay()
                end.record()
        else:
            pairs[0][0].record()
            for _ in range(repeats):
                graph.replay()
            pairs[0][1].record()
        pairs[-1][1].synchronize()
        wall_us = (time.perf_counter() - start_wall) * 1e6 / repeats
        return dict(
            gpu_us=sum(a.elapsed_time(b) * 1000 for a, b in pairs) / repeats,
            wall_us=wall_us,
            wall_includes_scrub=evict,
        )


def latency_case(graphs, timer, ids, *, rounds, repeats):
    graphs.inputs(ids)
    correctness = graphs.validate(allocator=True)
    raw = {
        condition: {name: [] for name in graphs.graphs}
        for condition in ("warm", "scrubbed")
    }
    snapshots = []
    names = list(graphs.graphs)
    for round_index in range(rounds):
        # Alternate arm ordering; warm each arm before its repeated-replay timing.
        for condition in (
            ("warm", "scrubbed") if round_index % 2 == 0 else ("scrubbed", "warm")
        ):
            for name in names if round_index % 2 == 0 else names[::-1]:
                graph = graphs.graphs[name]
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                raw[condition][name].append(
                    timer.sample(graph, repeats, evict=condition == "scrubbed")
                )
        snapshots.append(gpu_snapshot())
    medians = {
        condition: {
            name: statistics.median(row["gpu_us"] for row in samples)
            for name, samples in arms.items()
        }
        for condition, arms in raw.items()
    }
    return dict(
        correctness=correctness,
        raw=raw,
        median_gpu_us=medians,
        gpu_modes=snapshots,
        ratios={
            c: dict(
                static_over_all_vram=m["static"] / m["all_vram"],
                profiled_over_static=m["profiled"] / m["static"],
            )
            for c, m in medians.items()
        },
    )


def make_policy(experiment):
    return ResidencyCacheController(
        config=ResidencyCacheConfig(
            max_pairs=1,
            minimum_cold_selections=2,
            minimum_score_gain=2,
            minimum_residency_windows=1,
        ),
        observations=RoutingObservationSpec(
            layer="experiment",
            experts=experiment.e,
            phase="decode",
            max_top_k=experiment.ids.shape[1],
        ),
        exchange=ResidencyExchangeSpec(
            backend="sm120-nvfp4-pcie-poc",
            direct_backing_execution=True,
            fixed_address_quiescent_exchange=True,
            payload_copy_bytes_per_pair=4 * experiment.row_bytes,
            map_copy_bytes_per_transaction=16 * experiment.e,
        ),
        slots=experiment.updates.snapshot(),
        baseline=experiment.controls.snapshot(quiescent=True),
    )


def adaptive_case(graphs, timer, *, pattern, period, epochs, rounds, hot):
    e, records, initial = graphs.e, [], graphs.e.updates.snapshot().expert_map
    for round_index in range(rounds):
        for arm in (
            ("static", "adaptive") if round_index % 2 == 0 else ("adaptive", "static")
        ):
            e.controls.reset(quiescent=True)
            policy = make_policy(e)
            history, windows = [], []
            try:
                for epoch in range(epochs):
                    ids = policy_routes(
                        e.e, hot, graphs.live, e.ids.shape[1], pattern, epoch, epochs
                    )
                    graphs.inputs(ids)
                    # Warmup and correctness probes must not enter this window's
                    # cumulative counter delta. Snapshot a fresh baseline after
                    # the probe only before the first epoch; later probes use
                    # the static graph and do not execute observer nodes.
                    before_slots = e.updates.snapshot()
                    if epoch == 0:
                        correctness = graphs.validate(allocator=True)
                        e.controls.reset(quiescent=True)
                        policy = make_policy(e)
                    graph = graphs.graphs["profiled" if arm == "adaptive" else "static"]
                    sample = timer.sample(graph, period)
                    # Correctness of the actual measured invocation, before any
                    # promotion, is checked against all-VRAM outside the timer.
                    graphs.graphs["all_vram"].replay()
                    torch.cuda.synchronize()
                    actual, reference = (
                        e.output[: graphs.live].float(),
                        graphs.bindings[2].output.float(),
                    )
                    relative = float((actual - reference).norm() / reference.norm())
                    if not torch.isfinite(actual).all() or relative > 0.005:
                        raise AssertionError(
                            f"policy replay parity failed: {relative=}"
                        )
                    cold = (
                        sum(
                            before_slots.expert_map[i][0]
                            for i in ids.flatten().tolist()
                        )
                        * period
                    )
                    start = time.perf_counter()
                    decision, outcome, exchange_us = None, None, 0.0
                    if arm == "adaptive":
                        decision = policy.observe(
                            e.controls.snapshot(quiescent=True), slots=before_slots
                        )
                        if (
                            decision.cold_selections != cold
                            or sum(decision.counts) != ids.numel() * period
                        ):
                            raise AssertionError(
                                "counter delta includes work outside the measured epoch"
                            )
                        after_slots = before_slots
                        if decision.pairs:
                            swap_start = time.perf_counter()
                            after_slots = e.updates.exchange(
                                decision.pairs, expected=before_slots, quiescent=True
                            )
                            exchange_us = (time.perf_counter() - swap_start) * 1e6
                            history.append(decision.pairs)
                        outcome = policy.finish(decision, slots=after_slots)
                    control_us = (
                        (time.perf_counter() - start) * 1e6
                        if arm == "adaptive"
                        else 0.0
                    )
                    windows.append(
                        dict(
                            epoch=epoch,
                            gpu_us_per_replay=sample["gpu_us"],
                            replay_wall_us=sample["wall_us"] * period,
                            control_us=control_us,
                            exchange_us=exchange_us,
                            cold_selections=cold,
                            selections=ids.numel() * period,
                            unique_cold=len(
                                {
                                    i
                                    for i in ids.flatten().tolist()
                                    if before_slots.expert_map[i][0]
                                }
                            ),
                            generation=before_slots.generation,
                            relative_l2=relative,
                            decision=asdict(decision) if decision else None,
                            outcome=asdict(outcome) if outcome else None,
                        )
                    )
                total_replays = period * epochs
                records.append(
                    dict(
                        round=round_index,
                        arm=arm,
                        windows=windows,
                        initial_correctness=correctness,
                        total_gpu_us=sum(
                            w["gpu_us_per_replay"] * period for w in windows
                        ),
                        amortized_wall_us=(
                            sum(w["replay_wall_us"] + w["control_us"] for w in windows)
                            / total_replays
                        ),
                        cold_fraction=sum(w["cold_selections"] for w in windows)
                        / sum(w["selections"] for w in windows),
                        promotions=sum(
                            len(w["decision"]["pairs"])
                            for w in windows
                            if w["decision"]
                        ),
                        hits_after_promotion=windows[-1]["outcome"][
                            "observed_hits_after_promotion"
                        ]
                        if arm == "adaptive"
                        else 0,
                        gpu_mode=gpu_snapshot(),
                    )
                )
            finally:
                # Restore via the same transaction, not direct map edits. Setup
                # restoration is excluded from the measured serving epoch cost.
                for pairs in reversed(history):
                    e.updates.exchange(
                        pairs, expected=e.updates.snapshot(), quiescent=True
                    )
                if e.updates.snapshot().expert_map != initial:
                    raise AssertionError("static baseline was not restored")
    medians = {
        arm: statistics.median(
            r["amortized_wall_us"] for r in records if r["arm"] == arm
        )
        for arm in ("static", "adaptive")
    }
    return dict(
        pattern=pattern,
        period=period,
        epochs=epochs,
        records=records,
        median_wall_us=medians,
        adaptive_over_static_wall=medians["adaptive"] / medians["static"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", default="model.language_model.layers.0.mlp.experts")
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--hot-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--live", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--cold-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.015625, 0.05, 0.25, 1.0],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--policy-live", type=int, nargs="*", default=[])
    parser.add_argument("--periods", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if min(
        *args.live, args.rounds, args.repeats, args.epochs, *args.periods
    ) < 1 or any(not 0 <= x <= 1 for x in args.cold_fractions):
        parser.error("counts must be positive and fractions must be in [0,1]")
    if not set(args.policy_live) <= set(args.live):
        parser.error("policy-live must be contained in live")
    if args.output.exists():
        parser.error("output must not exist; preserve prior receipts")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        [*root.glob("b12x/**/*.py"), *root.glob("benchmarks/moe/sm120_residency*.py")]
    )
    manifest = dict(
        source_revision=args.source_revision,
        command=sys.argv,
        source_files={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        toolchain={
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "triton", "cuda-bindings")
        },
        torch_cuda=torch.version.cuda,
        gpu_before=gpu_snapshot(),
        status="running",
        method="single-layer real weights; synthetic routes; diagnostic default GPU clocks",
        ratio_direction="values above one mean the numerator is slower",
    )
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    failures = 0
    experiment = None
    try:
        torch.manual_seed(731)
        source, checksum = load_layer(args.checkpoint, args.prefix, args.experts)
        manifest["checkpoint_fields_sha256"] = checksum
        experiment = Experiment(
            source, hot=args.hot_experts, capacity=max(args.live), topk=args.top_k
        )
        timer = Timer(experiment)
        manifest.update(
            l2_bytes=timer.l2_bytes,
            scrub_bytes=timer.flush_buffer.numel(),
            resident_slab_bytes=experiment.tiers[0].slab.numel(),
            backing_slab_bytes=experiment.tiers[1].slab.numel(),
            journal_bytes=experiment.journal_owner.host_view.numel(),
            geometry=dict(
                experts=args.experts,
                hot=args.hot_experts,
                hidden=experiment.h,
                intermediate=source["w13"].shape[1] // 2,
                top_k=args.top_k,
                capacity=max(args.live),
            ),
        )
        with (args.output / "cases.jsonl").open("w") as stream:

            def record(row):
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            in (
                                "kind",
                                "live",
                                "cold_count",
                                "pattern",
                                "period",
                                "status",
                                "error",
                            )
                        }
                    ),
                    flush=True,
                )

            for live in args.live:
                graphs = Graphs(experiment, live)
                try:
                    counts = sorted(
                        {
                            0 if f == 0 else max(1, round(live * args.top_k * f))
                            for f in args.cold_fractions
                        }
                    )
                    for count in counts:
                        row = dict(
                            kind="latency",
                            live=live,
                            cold_count=count,
                            actual_cold_fraction=count / (live * args.top_k),
                        )
                        try:
                            ids = routes(
                                args.experts, args.hot_experts, live, args.top_k, count
                            )
                            row.update(
                                latency_case(
                                    graphs,
                                    timer,
                                    ids,
                                    rounds=args.rounds,
                                    repeats=args.repeats,
                                ),
                                status="passed",
                            )
                        except AssertionError:
                            failures += 1
                            row.update(status="failed", error=traceback.format_exc())
                        record(row)
                    if live in args.policy_live:
                        for pattern in (
                            "steady_hot",
                            "steady_cold",
                            "phase_shift",
                            "rotating_cold",
                        ):
                            for period in args.periods:
                                row = dict(
                                    kind="policy",
                                    live=live,
                                    pattern=pattern,
                                    period=period,
                                )
                                try:
                                    row.update(
                                        adaptive_case(
                                            graphs,
                                            timer,
                                            pattern=pattern,
                                            period=period,
                                            epochs=args.epochs,
                                            rounds=args.rounds,
                                            hot=args.hot_experts,
                                        ),
                                        status="passed",
                                    )
                                except AssertionError:
                                    failures += 1
                                    row.update(
                                        status="failed", error=traceback.format_exc()
                                    )
                                record(row)
                finally:
                    graphs.close()
        manifest.update(
            status="failed" if failures else "passed", failed_cases=failures
        )
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if experiment is not None:
            experiment.close()
        manifest["gpu_after"] = gpu_snapshot()
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
