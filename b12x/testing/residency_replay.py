"""Offline locality and cache replay; no device execution or serving policy changes.

Invocations preserve row/top-k grouping. Future-use statistics stop at request
boundaries and exclude right-censored horizons. Replay concatenates the declared
evaluation requests in order; training calls never enter evaluation statistics.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import json
from pathlib import Path

from b12x.moe.residency import (
    ExpertPlacement,
    LayerRoutingCounts,
    ResidencyCacheConfig,
    ResidencyCacheController,
    ResidencyExchangeSpec,
    ResidencySlotSnapshot,
    RoutingObservationSpec,
    RoutingSnapshot,
)


@dataclass(frozen=True)
class RouteCall:
    request: str
    workload: str
    split: str
    ids: tuple[tuple[int, ...], ...]

    @cached_property
    def counts(self):
        return Counter(e for row in self.ids for e in row if e >= 0)


@dataclass(frozen=True)
class LayerTrace:
    layer: str
    experts: int
    hidden: int
    intermediate: int
    checkpoint: str
    phase: str
    calls: tuple[RouteCall, ...]


def read_trace(path):
    """Read the explicit invocation format, rejecting ambiguity and lost records."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "b12x-routing-invocations-v1":
        raise ValueError("expected b12x-routing-invocations-v1")
    if not data.get("checkpoint") or data.get("truncated") is not False:
        raise ValueError("trace requires checkpoint identity and untruncated records")
    result = []
    names = set()
    for layer in data["layers"]:
        name = layer["layer"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("layer names must be unique nonempty strings")
        names.add(name)
        for key in ("experts", "hidden", "intermediate"):
            if type(layer[key]) is not int or layer[key] < 1:
                raise ValueError("trace geometry must be positive integers")
        phase = layer["phase"]
        if phase not in ("decode", "prefill", "verify", "draft"):
            raise ValueError("one explicit routing phase is required")
        calls, seen = [], set()
        for segment in layer["segments"]:
            request = segment["request"]
            if not request or request in seen:
                raise ValueError("request segments must be unique within a layer")
            seen.add(request)
            if segment["split"] not in ("train", "test") or not segment["workload"]:
                raise ValueError("each segment requires a split and workload")
            for call in segment["calls"]:
                ids = tuple(tuple(row) for row in call["ids"])
                if not ids or not ids[0] or any(len(row) != len(ids[0]) for row in ids):
                    raise ValueError(
                        "routing invocation must be a nonempty rectangular matrix"
                    )
                if any(
                    type(e) is not int or not -1 <= e < layer["experts"]
                    for row in ids
                    for e in row
                ):
                    raise ValueError("canonical expert ID outside declared geometry")
                calls.append(
                    RouteCall(request, segment["workload"], segment["split"], ids)
                )
        if not calls:
            raise ValueError("empty layer trace")
        result.append(
            LayerTrace(
                name,
                layer["experts"],
                layer["hidden"],
                layer["intermediate"],
                data["checkpoint"],
                phase,
                tuple(calls),
            )
        )
    if not result:
        raise ValueError("empty trace")
    return tuple(result)


def _summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, p50=None, p90=None, max=None)
    return dict(
        n=len(values),
        mean=sum(values) / len(values),
        p50=values[(len(values) - 1) // 2],
        p90=values[int((len(values) - 1) * 0.9)],
        max=values[-1],
    )


def locality(
    calls, experts, *, horizons=(1, 2, 4, 8, 16, 32, 128, 512), budgets=(128, 256)
):
    """Per-layer route/touch reuse with explicit horizon censoring.

    Stack distance counts distinct canonical experts referenced since the last
    route selection, in original row/top-k order. Invocation distances instead
    count later calls, excluding repeated selections within the same call.
    """
    frequency = Counter()
    unique, interarrival, stack_distance = [], [], []
    segments = {}
    for call in calls:
        segments.setdefault(call.request, []).append(call)
        frequency.update(call.counts)
        unique.append(len(call.counts))
    horizon_rows = {
        h: dict(
            eligible_touches=0,
            censored_touches=0,
            reused=0,
            later_route_selections=0,
            later_invocations=0,
            working_sets=[],
            hot_jaccard=[],
        )
        for h in horizons
    }
    for segment in segments.values():
        positions = [[] for _ in range(experts)]
        prefix = [[0] for _ in range(experts)]
        recent, step = {}, 0
        for i, call in enumerate(segment):
            count = call.counts
            for e, n in count.items():
                if positions[e]:
                    interarrival.append(i - positions[e][-1])
                positions[e].append(i)
                prefix[e].append(prefix[e][-1] + n)
            for row in call.ids:
                for e in row:
                    if e < 0:
                        continue
                    if e in recent:
                        stack_distance.append(
                            sum(t > recent[e] for t in recent.values())
                        )
                    recent[e] = step
                    step += 1
        for h, stats in horizon_rows.items():
            prior_hot = None
            for i, call in enumerate(segment):
                if i + h >= len(segment):
                    stats["censored_touches"] += len(call.counts)
                    continue
                for e in call.counts:
                    lo, hi = (
                        bisect_right(positions[e], i),
                        bisect_right(positions[e], i + h),
                    )
                    stats["eligible_touches"] += 1
                    stats["reused"] += hi > lo
                    stats["later_invocations"] += hi - lo
                    stats["later_route_selections"] += prefix[e][hi] - prefix[e][lo]
            # Disjoint windows avoid presenting overlapping windows as independent evidence.
            for start in range(0, len(segment) - h + 1, h):
                window = Counter()
                for c in segment[start : start + h]:
                    window.update(c.counts)
                stats["working_sets"].append(len(window))
                hot = set(
                    sorted(range(experts), key=lambda e: (-window[e], e))[
                        : min(budgets)
                    ]
                )
                if prior_hot is not None:
                    stats["hot_jaccard"].append(
                        len(hot & prior_hot) / len(hot | prior_hot)
                        if hot | prior_hot
                        else 1.0
                    )
                prior_hot = hot
    for stats in horizon_rows.values():
        denominator = stats["eligible_touches"]
        stats["p_reused"] = stats["reused"] / denominator if denominator else None
        stats["expected_later_selections"] = (
            stats["later_route_selections"] / denominator if denominator else None
        )
        stats["working_set"] = _summary(stats.pop("working_sets"))
        stats["hot_set_jaccard"] = _summary(stats.pop("hot_jaccard"))
    total = sum(frequency.values())
    rank = sorted(range(experts), key=lambda e: (-frequency[e], e))
    return dict(
        selections=total,
        calls=len(calls),
        unique_experts=len(frequency),
        selection_counts=[frequency[e] for e in range(experts)],
        unique_per_invocation=_summary(unique),
        interarrival_invocations=_summary(interarrival),
        stack_distance_selections=_summary(stack_distance),
        top_n_coverage={
            n: sum(frequency[e] for e in rank[:n]) / total if total else None
            for n in budgets
        },
        hot_set_size_for_stability=min(budgets),
        horizons=horizon_rows,
    )


def reuse_diagnostics(calls, horizons):
    """Precompute future-use diagnostics once for a policy/budget sweep."""
    future = {}
    for i, call in enumerate(calls):
        for e, n in call.counts.items():
            future.setdefault((call.request, e), []).append((i, n))
    ends = {c.request: i for i, c in enumerate(calls)}
    result = []
    for i, call in enumerate(calls):
        row = {}
        for e, n in call.counts.items():
            later = [(j, k) for j, k in future[(call.request, e)] if j > i]
            row[e] = dict(
                invocation=i,
                request=call.request,
                expert=e,
                selections=n,
                next_reuse=later[0][0] - i if later else None,
                right_censored=not later,
                horizons={
                    h: None
                    if i + h > ends[call.request]
                    else dict(
                        touches=sum(j <= i + h for j, k in later),
                        selections=sum(k for j, k in later if j <= i + h),
                    )
                    for h in horizons
                },
            )
        result.append(row)
    return result


def replay(
    trace,
    *,
    budget,
    window,
    policy="b12x",
    initial="learned",
    config=None,
    horizons=(1, 2, 4, 8, 16, 32, 128, 512),
    expert_bytes=None,
    future_use=None,
):
    """Replay complete invocations, deciding only after a fixed-map window.

    LRU/LFU are bounded window-end comparisons, not per-selection hardware cache
    simulations. They select only observed cold candidates, use the same movement
    budget and never turn the current miss into a hit. b12x uses the real host
    controller, including its residence guard and acknowledgement semantics.
    """
    if (
        type(budget) is not int
        or not 0 <= budget <= trace.experts
        or type(window) is not int
        or window < 1
    ):
        raise ValueError("invalid resident budget or decision window")
    if policy not in ("static", "lru", "lfu", "decayed_lfu", "b12x") or initial not in (
        "learned",
        "positional",
    ):
        raise ValueError("unknown replay policy or initial population")
    train = [c for c in trace.calls if c.split == "train"]
    calls = [c for c in trace.calls if c.split == "test"]
    if not calls or (initial == "learned" and not train):
        raise ValueError(
            "evaluation and learned training must be nonempty and disjoint"
        )
    config = config or ResidencyCacheConfig(
        max_pairs=1,
        minimum_cold_selections=2,
        minimum_score_gain=2,
        minimum_residency_windows=1,
        phase=trace.phase,
    )
    if config.phase != trace.phase:
        raise ValueError("trace and policy phase differ")
    counts = Counter()
    for c in train:
        counts.update(c.counts)
    ranking = sorted(range(trace.experts), key=lambda e: (-counts[e], e))
    hot = set((ranking if initial == "learned" else range(trace.experts))[:budget])
    initial_hot = set(hot)
    placement = ExpertPlacement(
        total_experts=trace.experts,
        resident_expert_ids=tuple(sorted(hot)),
        backing_expert_ids=tuple(e for e in range(trace.experts) if e not in hot),
    )
    slots = ResidencySlotSnapshot(
        preparation_id="offline-replay",
        generation=0,
        expert_map=placement.expert_map,
        healthy=True,
    )
    cumulative, total_tokens = Counter(), 0

    def snapshot(n):
        return RoutingSnapshot(
            epoch=0,
            rank=0,
            layers=(
                LayerRoutingCounts(
                    layer=trace.layer,
                    phase=trace.phase,
                    counts=tuple(cumulative[e] for e in range(trace.experts)),
                    calls=n,
                    sampled_calls=n,
                    tokens=total_tokens,
                    sampled_tokens=total_tokens,
                ),
            ),
        )

    controller = ResidencyCacheController(
        config=config,
        observations=RoutingObservationSpec(
            layer=trace.layer,
            experts=trace.experts,
            phase=trace.phase,
            max_top_k=max(len(c.ids[0]) for c in calls),
        ),
        exchange=ResidencyExchangeSpec(
            backend="offline-exclusive",
            direct_backing_execution=True,
            fixed_address_quiescent_exchange=True,
            payload_copy_bytes_per_pair=4 * (expert_bytes or 0),
            map_copy_bytes_per_transaction=16 * trace.experts,
        ),
        slots=slots,
        baseline=snapshot(0),
    )
    promotions, active, windows, misses = [], {}, [], []
    totals = Counter()
    decayed, last, entered = Counter(), {}, {e: 0 for e in hot}
    recent = Counter()
    comparison_window = 0
    if future_use is None:
        future_use = reuse_diagnostics(calls, horizons)
    for i, call in enumerate(calls):
        row = call.counts
        cold = {e for e in row if e not in hot}
        totals["selections"] += sum(row.values())
        totals["cold_selections"] += sum(row[e] for e in cold)
        totals["static_cold_selections"] += sum(
            n for e, n in row.items() if e not in initial_hot
        )
        totals["cold_unique_touches"] += len(cold)
        misses.extend(future_use[i][e] for e in cold)
        for e, n in row.items():
            if e in active:
                p = promotions[active[e]]
                p["useful_hits"] += n
                p["useful_invocations"] += 1
                p["hit_steps"].append((i, n))
            last[e] = i
        cumulative.update(row)
        recent.update(row)
        total_tokens += len(call.ids)
        if (i + 1) % window and i + 1 != len(calls):
            continue
        decision = controller.observe(snapshot(i + 1), slots=slots)
        comparison_window += bool(sum(recent.values()))
        pairs = decision.pairs if policy == "b12x" else ()
        if policy in ("lru", "lfu", "decayed_lfu"):
            decayed = Counter(
                {e: decayed[e] * 0.5 + recent[e] for e in range(trace.experts)}
            )
            score = (
                last if policy == "lru" else cumulative if policy == "lfu" else decayed
            )
            unseen = -1 if policy == "lru" else 0
            candidates = sorted(
                (
                    e
                    for e in recent
                    if e not in hot and recent[e] >= config.minimum_cold_selections
                ),
                key=lambda e: (-score.get(e, unseen), e),
            )
            victims = sorted(
                (
                    e
                    for e in hot
                    if comparison_window - entered[e]
                    >= config.minimum_residency_windows
                ),
                key=lambda e: (score.get(e, unseen), e),
            )
            # Frequency score margin applies to frequency policies only; LRU requires newer use.
            margin = 1 if policy == "lru" else config.minimum_score_gain
            pairs = tuple(
                (c, v)
                for c, v in zip(candidates, victims, strict=False)
                if score.get(c, unseen) - score.get(v, unseen) >= margin
            )[: config.max_pairs]
        if i + 1 == len(calls):
            pairs = ()  # No claimed value for a fill after the trace ends.
        old_hot = set(hot)
        if pairs:
            mapping = list(slots.expert_map)
            for candidate, victim in pairs:
                mapping[candidate], mapping[victim] = (
                    mapping[victim],
                    mapping[candidate],
                )
                hot.remove(victim)
                hot.add(candidate)
                entered.pop(victim)
                entered[candidate] = comparison_window
                if victim in active:
                    p = promotions[active.pop(victim)]
                    p["evicted_at"] = i + 1
                active[candidate] = len(promotions)
                promotions.append(
                    dict(
                        expert=candidate,
                        victim=victim,
                        promoted_at=i + 1,
                        evicted_at=None,
                        useful_hits=0,
                        useful_invocations=0,
                        hit_steps=[],
                    )
                )
            slots = replace(
                slots, generation=slots.generation + 1, expert_map=tuple(mapping)
            )
        if policy == "b12x":
            outcome = controller.finish(decision, slots=slots)
            assert outcome.observed_hits_after_promotion == sum(
                p["useful_hits"] for p in promotions
            )
        else:
            controller.finish(decision, slots=decision.expected)
            # The offline comparison owns a different policy; reset only its diagnostics adapter.
            controller = ResidencyCacheController(
                config=config,
                observations=controller.observations,
                exchange=controller.exchange,
                slots=slots,
                baseline=snapshot(i + 1),
            )
        windows.append(
            dict(
                end=i + 1,
                workload=call.workload,
                generation=slots.generation,
                cold_selections=decision.cold_selections,
                selections=sum(recent.values()),
                pairs=pairs,
                hot_experts=sorted(hot),
                jaccard_previous=len(hot & old_hot) / len(hot | old_hot)
                if hot | old_hot
                else 1.0,
                jaccard_initial=len(hot & initial_hot) / len(hot | initial_hot)
                if hot | initial_hot
                else 1.0,
            )
        )
        recent.clear()
    for p in promotions:
        p["residence_invocations"] = (p["evicted_at"] or len(calls)) - p["promoted_at"]
        p["censored_at_end"] = p["evicted_at"] is None
        p["wasted_completed"] = p["evicted_at"] is not None and p["useful_hits"] == 0
    return dict(
        policy=policy,
        initial=initial,
        budget=budget,
        window=window,
        totals=dict(totals),
        selections_avoided_vs_static=totals["static_cold_selections"]
        - totals["cold_selections"],
        promotions=promotions,
        movement_pairs=len(promotions),
        wasted_completed=sum(p["wasted_completed"] for p in promotions),
        zero_hit_censored=sum(
            p["censored_at_end"] and not p["useful_hits"] for p in promotions
        ),
        payload_api_bytes=4 * expert_bytes * len(promotions) if expert_bytes else None,
        windows=windows,
        misses=misses,
    )


def retrospective_cost(result, *, promotion_us, cold_saving_us_per_selection):
    """Explicit linear sensitivity model, not a measured execution prediction.

    Caller supplies geometry-specific costs. Useful-hit attribution excludes the
    opportunity cost of evicting a victim; net-vs-static includes that cost through
    the difference in cold selections. No constants are imported from GPU receipts.
    """
    import math

    if any(
        not math.isfinite(x) or x < 0
        for x in (promotion_us, cold_saving_us_per_selection)
    ):
        raise ValueError("costs must be finite nonnegative values")
    rows = []
    for p in result["promotions"]:
        hits = 0
        breakeven = None
        for step, n in p["hit_steps"]:
            hits += n
            if (
                breakeven is None
                and hits * cold_saving_us_per_selection >= promotion_us
            ):
                breakeven = step - p["promoted_at"] + 1
        rows.append(
            dict(
                expert=p["expert"],
                promotion_cost_us=promotion_us,
                gross_avoided_us=p["useful_hits"] * cold_saving_us_per_selection,
                gross_net_us=p["useful_hits"] * cold_saving_us_per_selection
                - promotion_us,
                invocations_to_break_even=breakeven,
            )
        )
    return dict(
        scope="linear sensitivity; excludes profiling/control and nonlinear cold execution",
        promotions=rows,
        net_vs_static_us=result["selections_avoided_vs_static"]
        * cold_saving_us_per_selection
        - result["movement_pairs"] * promotion_us,
    )


def trace_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
