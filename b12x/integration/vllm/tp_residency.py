"""Engine-owned all-rank maintenance over the shared residency policy.

The engine is the single policy authority. Workers retain only local payloads;
RPC commands carry logical IDs and expected generations, never weight bytes.
"""

from dataclasses import asdict
from time import perf_counter_ns
from uuid import uuid4

from b12x.moe.residency import (
    ResidencyCacheConfig, ResidencyCacheController, ResidencyEpochBudget,
    ResidencyEpochCoordinator, ResidencyExchangeSpec, RoutingObservationSpec,
    ResidencySlotSnapshot, updated_slot_map,
)
from .residency_epoch import VllmResidencyEpochs, _slots, _snapshot


class TensorParallelResidencyMaintenance:
    """Uniform local shard costs, per-rank byte limits and one logical pair cap.

    Every RPC must complete for all N ranks. Before publication, reversible
    payloads are restored on every reachable rank. Any error still leaves the
    engine paused; missing participants or uncertain publication require reload.
    """

    _ranks = VllmResidencyEpochs._ranks
    _validate_replicas = VllmResidencyEpochs._validate_replicas
    owner_rank = 0
    verify_rank_counters = True

    def __init__(self, config, world_size):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("TP participant count must be positive")
        if config.get("anchor_thresholds") or config.get("movement_mode", "adapt") != "adapt":
            raise ValueError("TP maintenance requires ordinary adaptation without anchor recovery")
        self.config, self.tp_size = dict(config), world_size
        self.configs = {n: ResidencyCacheConfig(**v) for n, v in config["layers"].items()}
        self.budget = ResidencyEpochBudget(**config["budget"])
        self.threshold = config["cold_fraction_threshold"]
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("cold fraction threshold must be in [0, 1]")
        self.coordinator, self.failed = None, False
        self.last_receipt = None

    def run(self, collective_rpc):
        if self.failed:
            raise RuntimeError("distributed residency requires worker reload")
        token, started = uuid4().hex, perf_counter_ns()
        stages, publication = {}, False
        self.last_receipt = {"token": token, "status": "preparing", "stages_ns": stages}

        def rpc(stage, *args):
            mark = perf_counter_ns()
            try:
                return self._ranks(collective_rpc("b12x_residency_" + stage, args=args, timeout=60))
            finally:
                stages[stage] = perf_counter_ns() - mark

        try:
            ranks = rpc("begin", token)
            self._validate_replicas(ranks)
            owner = ranks[self.owner_rank]
            snapshot = _snapshot(owner["snapshot"])
            slots = {n: _slots(v["slots"]) for n, v in owner["layers"].items()}
            baseline = self.coordinator is None
            if baseline:
                self.coordinator = ResidencyEpochCoordinator({
                    n: ResidencyCacheController(config=self.configs[n],
                        observations=RoutingObservationSpec(**v["observations"]),
                        exchange=ResidencyExchangeSpec(**v["exchange"]),
                        slots=slots[n], baseline=snapshot)
                    for n, v in owner["layers"].items()
                }, budget=self.budget)
            selections = cold = 0
            decision = None
            if not baseline:
                for name, controller in self.coordinator.controllers.items():
                    _, counts, _ = controller.validate_observation(snapshot, slots=slots[name])
                    selections += sum(counts)
                    cold += sum(c for c, (tier, _) in zip(counts, slots[name].expert_map, strict=True) if tier)
                decision = self.coordinator.observe(snapshot, slots=slots,
                    allow_movement=bool(selections) and (self.threshold is None or cold / selections >= self.threshold))
            pairs = {n: () for n in slots} if baseline else {v.layer: v.pairs for v in decision.layers}
            commands = {str(rank): {n: {"expected": v["slots"], "pairs": pairs[n]}
                for n, v in report["layers"].items()} for rank, report in ranks.items()}
            for stage, args, key in (("prepare", (token, commands), "ready"), ("stage", (token,), "staged")):
                if not all(v.get(key) is True for v in rpc(stage, *args).values()):
                    raise RuntimeError("TP participant did not complete " + stage)
            publication = True
            applied = rpc("publish", token)
            for rank, report in applied.items():
                if set(report["layers"]) != set(slots):
                    raise RuntimeError("TP completion omitted a layer")
                for name, value in report["layers"].items():
                    before = _slots(ranks[rank]["layers"][name]["slots"])
                    expected = ResidencySlotSnapshot(preparation_id=before.preparation_id,
                        generation=before.generation + bool(pairs[name]), healthy=True,
                        expert_map=updated_slot_map(before, pairs[name],
                            backing_mode=owner["layers"][name]["exchange"]["backing_mode"]))
                    if _slots(value) != expected:
                        raise RuntimeError("TP published map differs from logical decision")
            if not all(v.get("acknowledged") is True for v in rpc("acknowledge_staged", token).values()):
                raise RuntimeError("TP publication acknowledgement missing")
            outcomes = {} if baseline else self.coordinator.finish(decision,
                slots={n: _slots(v) for n, v in applied[self.owner_rank]["layers"].items()})
            if not all(v.get("finished") is True for v in rpc("finish", token).values()):
                raise RuntimeError("TP completion acknowledgement missing")
            per_rank_bytes = 0 if baseline else decision.copy_bytes
            self.last_receipt["status"] = "complete"
            return {
                "status": "complete", "baseline": baseline,
                "health": "pressure" if selections and (self.threshold is None or cold / selections >= self.threshold) else "healthy",
                "selections": selections, "cold_selections": cold,
                "cold_fraction": cold / selections if selections else None,
                "proposed_pairs": 0 if baseline else decision.proposed_pairs,
                "selected_pairs": 0 if baseline else decision.selected_pairs,
                "copy_bytes": per_rank_bytes * self.tp_size,
                "per_rank_copy_bytes": [per_rank_bytes] * self.tp_size,
                "max_rank_copy_bytes": per_rank_bytes,
                "proposal_backlog": {} if baseline else {
                    "pair_cap_skips": decision.pair_cap_skips, "byte_cap_skips": decision.byte_cap_skips,
                    "proposing_layers": decision.proposing_layers,
                    "skipped_incremental_copy_bytes": decision.skipped_incremental_copy_bytes,
                    "budget": asdict(self.budget), "byte_budget_scope": "per_rank"},
                "layers": {n: asdict(o) for n, o in outcomes.items()},
                "stages_ns": stages, "workers": ranks,
                "worker_wall_ns": perf_counter_ns() - started,
            }
        except BaseException as error:
            self.failed = True
            self.last_receipt.update(status="reload_required", publication_started=publication,
                                     error=type(error).__name__ + ": " + str(error))
            if self.coordinator is not None:
                self.coordinator.fail()
            if not publication:
                try:
                    reverted = rpc("rollback", token)
                    self.last_receipt["rollback"] = reverted
                except BaseException as rollback_error:
                    self.last_receipt["rollback_error"] = type(rollback_error).__name__ + ": " + str(rollback_error)
            raise
