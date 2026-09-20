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
    updated_slot_map, RoutingAnchorThresholds, compare_anchor,
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
        self.recenter_budget = (
            ResidencyEpochBudget(**config["recenter_budget"])
            if config.get("recenter_budget") is not None else self.budget
        )
        self.threshold = config["cold_fraction_threshold"]
        self.anchor_thresholds = (RoutingAnchorThresholds(**config["anchor_thresholds"])
                                  if config.get("anchor_thresholds") is not None else None)
        if self.anchor_thresholds is not None and runtime.anchor is None:
            raise ValueError("anchor recovery requires a prepared learned anchor")
        self.diagnostics = config.get("policy_diagnostics", False)
        if type(self.diagnostics) is not bool:
            raise TypeError("policy_diagnostics must be boolean")
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

    def run(self, *, movement_mode="adapt"):
        r = self.runtime
        start = perf_counter_ns()
        stages = {}
        try:
            if movement_mode not in ("adapt", "recenter"):
                raise ValueError("unknown residency movement mode")
            if movement_mode == "recenter" and self.anchor_thresholds is None:
                raise ValueError("re-centering requires configured anchor thresholds")
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
                    **({"policy_configs": {n: asdict(c) for n, c in self.configs.items()}}
                       if self.diagnostics else {}),
                }

            measured = perf_counter_ns()
            selections, cold = 0, 0
            anchor_counts = {}
            for name, controller in self.coordinator.controllers.items():
                _, counts, _ = controller.validate_observation(
                    snapshot, slots=slots[name]
                )
                if movement_mode == "recenter":
                    anchor_counts[name] = counts
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
            anchor_result, recenter = None, None
            if movement_mode == "recenter":
                anchor_result = compare_anchor(
                    anchor_counts,
                    {n: tuple(e for e, (t, _) in enumerate(s.expert_map) if t == 0)
                     for n, s in slots.items()}, r.anchor.resident_ids,
                )
                assessment = self.anchor_thresholds.assess(anchor_result)
                anchor_result.update(assessment, profile=r.anchor.profile_id)
                recenter, allow = r.anchor, assessment["anchor_better"]
            stages["pressure"] = perf_counter_ns() - measured
            measured = perf_counter_ns()
            history = getattr(state, "history", None)
            history_receipt, policy_observations = None, []
            if history is not None:
                generation = tuple(
                    (n, s.preparation_id, s.generation)
                    for n, s in sorted(slots.items())
                )
                cuts, history_receipt = history.read(generation, quiescent=True)
                from b12x.moe.residency.contracts import validate_routing_progress

                validate_routing_progress((*cuts, snapshot))
                if cuts:
                    for name, controller in self.coordinator.controllers.items():
                        controller.validate_observation(cuts[0], slots=slots[name])
                # The final checkpoint and the maintenance tail form one window.
                # Deferred cuts age history/guards but never propose movement.
                for cut in cuts[:-1]:
                    deferred = self.coordinator.observe(
                        cut, slots=slots, allow_movement=False
                    )
                    self.coordinator.finish(deferred, slots=slots)
                    if self.diagnostics:
                        policy_observations.append(
                            policy_observation(deferred, deferred=True)
                        )
                history_receipt["replayed_windows"] = max(0, len(cuts) - 1)
                stages["history"] = perf_counter_ns() - measured
                measured = perf_counter_ns()
            epoch_budget = self.recenter_budget if movement_mode == "recenter" else self.budget
            decision = self.coordinator.observe(
                snapshot, slots=slots, allow_movement=allow, recenter=recenter,
                budget=epoch_budget,
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
                "movement_mode": "recenter" if recenter is not None else "adapt",
                **({"anchor": anchor_result} if anchor_result is not None else {}),
                "proposed_pairs": decision.proposed_pairs,
                "proposal_backlog": {
                    "pair_cap_skips": decision.pair_cap_skips,
                    "byte_cap_skips": decision.byte_cap_skips,
                    "skipped_incremental_copy_bytes": decision.skipped_incremental_copy_bytes,
                    "proposing_layers": decision.proposing_layers,
                    "budget": asdict(epoch_budget),
                },
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
                **({"history": history_receipt} if history_receipt is not None else {}),
                **(
                    {
                        "policy_observations": policy_observations
                        + [policy_observation(decision)]
                    }
                    if self.diagnostics
                    else {}
                ),
                "worker_wall_ns": perf_counter_ns() - start,
            }
        except BaseException:
            r._failed = True
            if self.coordinator is not None:
                self.coordinator.fail()
            raise


class VllmResidencyMaintenance:
    """Explicit engine control; no task or cadence is installed automatically."""

    def __init__(
        self,
        engine,
        *,
        configs,
        budget,
        cold_fraction_threshold=None,
        policy_diagnostics=False,
        anchor_thresholds=None,
        recenter_budget=None,
    ):
        self.engine = engine
        self.config = {
            "session": uuid4().hex,
            "layers": {n: asdict(c) for n, c in configs.items()},
            "budget": asdict(budget),
            "cold_fraction_threshold": cold_fraction_threshold,
            "policy_diagnostics": policy_diagnostics,
            "anchor_thresholds": None if anchor_thresholds is None else asdict(anchor_thresholds),
            "recenter_budget": None if recenter_budget is None else asdict(recenter_budget),
        }
        self._lock = asyncio.Lock()
        self.failed = False
        self.last_receipt = None

    async def run(self, *, movement_mode="adapt"):
        if movement_mode not in ("adapt", "recenter"):
            raise ValueError("unknown residency movement mode")
        requested = perf_counter_ns()
        async with self._lock:
            if self.failed:
                raise RuntimeError("maintenance outcome is uncertain; reload the lane")
            acquired = perf_counter_ns()
            try:
                result = await self.engine.residency_maintenance(
                    self.config if movement_mode == "adapt" else
                    {**self.config, "movement_mode": movement_mode}
                )
            except BaseException:
                self.failed = True
                raise
            result.update(
                lock_wait_ns=acquired - requested,
                total_wall_ns=perf_counter_ns() - requested,
            )
            self.last_receipt = result
            return result


def policy_observation(decision, *, deferred=False):
    """Opt-in control-plane evidence; never recorded by graph execution."""
    return {
        "deferred": deferred,
        "layers": {
            layer.layer: {
                "window": layer.decision.window,
                "movement_mode": layer.decision.movement_mode,
                "counts": layer.decision.counts,
                "scores": layer.decision.scores,
                "calls": layer.decision.calls,
                "cold_selections": layer.decision.cold_selections,
                "candidates": layer.decision.pairs,
                "selected": layer.pairs,
                "protected": layer.decision.protected_hot_experts,
                "hits": layer.decision.observed_hits_since_promotion,
            }
            for layer in decision.layers
        },
    }
