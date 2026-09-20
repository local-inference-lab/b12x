"""Worker-local policy at an engine-owned quiescent maintenance boundary.

Single-rank only. The distributed epoch adapter remains the all-rank protocol.
No callback runs inside graph execution and no new routing kernel is installed.
"""

import asyncio
from dataclasses import asdict
import math
from time import perf_counter_ns
from uuid import uuid4

from b12x.moe.residency import (
    ResidencyCacheConfig,
    ResidencyCacheController,
    ResidencyEpochBudget,
    ResidencyEpochCoordinator,
    ResidencySlotSnapshot,
    updated_slot_map,
)


class LocalResidencyMaintenance:
    """Advance recency at each observation; permit movement only above pressure.

    Pressure is the recent cold-selection fraction, not predicted throughput.
    The explicit threshold is experimental. Skipped movement still advances
    decayed scores and hit accounting, with unchanged placement generations.
    """

    def __init__(self, runtime, config):
        if runtime.rank != 0 or runtime.owner_rank != 0:
            raise ValueError("local maintenance requires a single authoritative rank")
        if runtime._controller_kind not in (None, "local"):
            raise RuntimeError("residency controller changed; reload the lane")
        runtime._controller_kind = "local"
        self.runtime = runtime
        self.config = config
        self.configs = {
            n: ResidencyCacheConfig(**v) for n, v in config["layers"].items()
        }
        self.budget = ResidencyEpochBudget(**config["budget"])
        self.threshold = config["cold_fraction_threshold"]
        if self.threshold is not None and (
            type(self.threshold) not in (float, int)
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("cold fraction threshold must be explicit and in [0, 1]")
        if set(self.configs) != set(runtime.bindings):
            raise ValueError("maintenance policy must cover every prepared layer")
        for name, binding in runtime.bindings.items():
            if self.configs[name].max_pairs > binding.max_pairs:
                raise ValueError("maintenance exceeds prepared fill capacity")
        self.coordinator = None

    def run(self):
        r = self.runtime
        start = perf_counter_ns()
        stages = {}
        try:
            r._validate()
            if r._stage != "idle":
                raise RuntimeError("another residency transaction is pending")
            slots = {n: b.snapshot() for n, b in r.bindings.items()}
            observed = perf_counter_ns()
            snapshot = r.snapshot_counters(quiescent=True)
            stages["snapshot"] = perf_counter_ns() - observed
            state = getattr(r.snapshot_counters, "__self__", None)
            snapshot_stages = getattr(state, "last_snapshot_timings_ns", {})
            if self.coordinator is None:
                controllers = {
                    n: ResidencyCacheController(
                        config=self.configs[n],
                        observations=b.observations,
                        exchange=b.exchange,
                        slots=slots[n],
                        baseline=snapshot,
                    )
                    for n, b in r.bindings.items()
                }
                self.coordinator = ResidencyEpochCoordinator(
                    controllers, budget=self.budget
                )
                return {
                    "status": "complete",
                    "baseline": True,
                    "stages_ns": stages,
                    "snapshot_stages_ns": snapshot_stages,
                    "worker_wall_ns": perf_counter_ns() - start,
                }

            measured = perf_counter_ns()
            selections, cold = 0, 0
            for name, controller in self.coordinator.controllers.items():
                _, counts, _ = controller.validate_observation(
                    snapshot, slots=slots[name]
                )
                selections += sum(counts)
                cold += sum(
                    n
                    for n, (tier, _) in zip(counts, slots[name].expert_map, strict=True)
                    if tier == 1
                )
            fraction = cold / selections if selections else None
            allow = bool(selections) and (
                self.threshold is None or fraction >= self.threshold
            )
            stages["pressure"] = perf_counter_ns() - measured
            measured = perf_counter_ns()
            decision = self.coordinator.observe(
                snapshot, slots=slots, allow_movement=allow
            )
            stages["policy"] = perf_counter_ns() - measured
            measured = perf_counter_ns()
            # Preflight every layer before any slot can be overwritten.
            for layer in decision.layers:
                if r.bindings[layer.layer].snapshot() != slots[layer.layer]:
                    raise RuntimeError("layer changed before maintenance preflight")
                if layer.pairs:
                    updated_slot_map(
                        slots[layer.layer],
                        layer.pairs,
                        backing_mode=r.bindings[layer.layer].exchange.backing_mode,
                    )
            stages["preflight"] = perf_counter_ns() - measured
            measured = perf_counter_ns()
            layer_times = {}
            for layer in decision.layers:
                if layer.pairs:
                    before = perf_counter_ns()
                    r.bindings[layer.layer].apply(
                        layer.pairs, expected=slots[layer.layer], quiescent=True
                    )
                    layer_times[layer.layer] = perf_counter_ns() - before
            stages["apply"] = perf_counter_ns() - measured
            measured = perf_counter_ns()
            after = {n: b.snapshot() for n, b in r.bindings.items()}
            for layer in decision.layers:
                before = slots[layer.layer]
                if not layer.pairs:
                    if after[layer.layer] != before:
                        raise RuntimeError(
                            "unchanged maintenance layer changed generation"
                        )
                    continue
                expected = ResidencySlotSnapshot(
                    preparation_id=before.preparation_id,
                    generation=before.generation + bool(layer.pairs),
                    healthy=True,
                    expert_map=updated_slot_map(
                        before,
                        layer.pairs,
                        backing_mode=r.bindings[layer.layer].exchange.backing_mode,
                    ),
                )
                if after[layer.layer] != expected:
                    raise RuntimeError(
                        "maintenance generation did not commit completely"
                    )
            r._validate()
            outcomes = self.coordinator.finish(decision, slots=after)
            stages["acknowledge"] = perf_counter_ns() - measured
            return {
                "status": "complete",
                "baseline": False,
                "health": "unobserved"
                if fraction is None
                else "pressure"
                if allow
                else "healthy",
                "selections": selections,
                "cold_selections": cold,
                "cold_fraction": fraction,
                "proposed_pairs": decision.proposed_pairs,
                "selected_pairs": decision.selected_pairs,
                "copy_bytes": decision.copy_bytes,
                "stages_ns": stages,
                "snapshot_stages_ns": snapshot_stages,
                "policy_stages_ns": self.coordinator.last_timings_ns,
                "transaction_ns": layer_times,
                "transaction_stages_ns": {
                    n: r.bindings[n].timings()
                    for n in layer_times
                    if r.bindings[n].timings is not None
                },
                "layers": {
                    n: {
                        "generation": o.generation,
                        "pairs": o.pairs,
                        "promotions": o.promotions,
                        "hits": o.observed_hits_after_promotion,
                        "evicted_hits": o.evicted_promotion_hits,
                    }
                    for n, o in outcomes.items()
                },
                "worker_wall_ns": perf_counter_ns() - start,
            }
        except BaseException:
            r._failed = True
            if self.coordinator is not None:
                self.coordinator.fail()
            raise


class VllmResidencyMaintenance:
    """Explicit engine control; no task or cadence is installed automatically."""

    def __init__(self, engine, *, configs, budget, cold_fraction_threshold=None):
        self.engine = engine
        self.config = {
            "session": uuid4().hex,
            "layers": {n: asdict(c) for n, c in configs.items()},
            "budget": asdict(budget),
            "cold_fraction_threshold": cold_fraction_threshold,
        }
        self._lock = asyncio.Lock()
        self.failed = False
        self.last_receipt = None

    async def run(self):
        requested = perf_counter_ns()
        async with self._lock:
            if self.failed:
                raise RuntimeError("maintenance outcome is uncertain; reload the lane")
            acquired = perf_counter_ns()
            try:
                result = await self.engine.residency_maintenance(self.config)
            except BaseException:
                self.failed = True
                raise
            result.update(
                lock_wait_ns=acquired - requested,
                total_wall_ns=perf_counter_ns() - requested,
            )
            self.last_receipt = result
            return result
