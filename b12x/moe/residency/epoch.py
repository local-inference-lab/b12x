"""Bounded model-wide decisions; engines own quiescence and backend execution."""
from dataclasses import dataclass
from types import MappingProxyType
from time import perf_counter_ns

from .contracts import _integer
from .anchor import ResidencyAnchor
from .policy import ResidencyCacheController, ResidencyCacheDecision


@dataclass(frozen=True, kw_only=True)
class ResidencyEpochBudget:
    """Hard limits on logical pairs and aggregate successful API-copy bytes.

    Byte admission includes payload copies and one map transaction per changed
    layer on every replica. It is not a pause-duration or transport guarantee.
    """
    max_pairs: int
    max_copy_bytes: int

    def __post_init__(self):
        _integer("max_pairs", self.max_pairs)
        _integer("max_copy_bytes", self.max_copy_bytes)


@dataclass(frozen=True, kw_only=True)
class ResidencyLayerDecision:
    layer: str
    decision: ResidencyCacheDecision
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, kw_only=True)
class ResidencyEpochDecision:
    epoch: int
    layers: tuple[ResidencyLayerDecision, ...]
    proposed_pairs: int
    selected_pairs: int
    copy_bytes: int
    pair_cap_skips: int = 0
    byte_cap_skips: int = 0
    skipped_incremental_copy_bytes: int = 0
    proposing_layers: int = 0

    @property
    def skipped_pairs(self):
        return self.proposed_pairs - self.selected_pairs


class ResidencyEpochCoordinator:
    """One observation owner, one phase and one pending model-wide decision.

    Layer policies retain their own scores and lifetimes. Opportunities are
    ranked by score gain per payload-copy byte, then layer and canonical IDs.
    This is a deterministic experimental ordering, not a latency prediction.
    Replicas must have equal declared copy costs; heterogeneous shards require
    a separate accounting contract. Never combine independently routed DP lanes.
    """
    def __init__(self, controllers, *, budget: ResidencyEpochBudget, replicas=1):
        if not isinstance(budget, ResidencyEpochBudget):
            raise TypeError("epoch requires a typed promotion budget")
        _integer("replicas", replicas, 1)
        controllers = dict(controllers)
        if not controllers or any(not isinstance(c, ResidencyCacheController)
                                  or c.observations.layer != name for name, c in controllers.items()):
            raise ValueError("epoch requires uniquely named layer controllers")
        if len({(c.observations.owner_rank, c.config.phase, c.observations.sample_every)
                for c in controllers.values()}) != 1:
            raise ValueError("model epoch requires one observation owner, phase and sampling rate")
        self.controllers = MappingProxyType(dict(sorted(controllers.items())))
        self.budget, self.replicas = budget, replicas
        self._epoch, self._pending, self._failed = 0, None, False

    def _require_healthy(self):
        if self._failed:
            raise RuntimeError("residency epoch failed; reload the lane and establish fresh baselines")

    def observe(self, snapshot, *, slots, allow_movement=True, recenter=None):
        """Advance observation history even when an external health gate declines movement."""
        self._require_healthy()
        if type(allow_movement) is not bool:
            raise TypeError("allow_movement must be bool")
        if self._pending is not None:
            raise RuntimeError("finish the pending model epoch before observing another")
        if set(slots) != set(self.controllers):
            raise ValueError("epoch slot set differs from declared layers")
        references = None
        if recenter is not None:
            if not isinstance(recenter, ResidencyAnchor):
                raise TypeError("re-centering requires a validated learned anchor")
            references = recenter.resident_ids
            if set(references) != set(self.controllers):
                raise ValueError("anchor layer set differs from epoch")
            for name, placement in recenter.placements:
                if (placement.total_experts != len(slots[name].expert_map)
                        or len(placement.resident_expert_ids)
                        != sum(t == 0 for t, _ in slots[name].expert_map)):
                    raise ValueError("anchor geometry differs from prepared slots")
        # Invalid later layers must not consume earlier layers' observations.
        for name, controller in self.controllers.items():
            controller.validate_observation(snapshot, slots=slots[name])
        started = perf_counter_ns()
        decisions = {name: c.observe(snapshot, slots=slots[name], propose=allow_movement,
                     recenter_to=None if references is None else references[name])
                     for name, c in self.controllers.items()}
        ranked = perf_counter_ns()
        opportunities = []
        for name, decision in decisions.items():
            cost = self.controllers[name].exchange.payload_copy_bytes_per_pair * self.replicas
            for cold, hot in decision.pairs:
                gain = decision.scores[cold] - decision.scores[hot]
                opportunities.append((-gain/max(1, cost), name, cold, hot))
        selected = {name: [] for name in decisions}
        used, pairs = 0, 0
        pair_skips = byte_skips = skipped_bytes = 0
        for _, name, cold, hot in sorted(opportunities):
            spec = self.controllers[name].exchange
            cost = self.replicas * (spec.payload_copy_bytes_per_pair
                                   + (0 if selected[name] else spec.map_copy_bytes_per_transaction))
            if pairs == self.budget.max_pairs:
                pair_skips += 1
                skipped_bytes += cost
                continue
            if used + cost > self.budget.max_copy_bytes:
                byte_skips += 1
                skipped_bytes += cost
                continue
            selected[name].append((cold, hot))
            used += cost
            pairs += 1
        self._epoch += 1
        self._pending = ResidencyEpochDecision(epoch=self._epoch,
            layers=tuple(ResidencyLayerDecision(layer=name, decision=d, pairs=tuple(selected[name]))
                         for name, d in decisions.items()),
            proposed_pairs=len(opportunities), selected_pairs=pairs, copy_bytes=used,
            pair_cap_skips=pair_skips, byte_cap_skips=byte_skips,
            skipped_incremental_copy_bytes=skipped_bytes,
            proposing_layers=sum(bool(d.pairs) for d in decisions.values()))
        self.last_timings_ns = {"layer_policy": ranked-started,
                               "opportunity_ranking": perf_counter_ns()-ranked}
        return self._pending

    def finish(self, decision, *, slots):
        self._require_healthy()
        if self._pending is None or decision is not self._pending:
            raise ValueError("model epoch decision is stale or foreign")
        if set(slots) != set(self.controllers):
            raise ValueError("epoch completion is missing a layer")
        for layer in decision.layers:
            self.controllers[layer.layer].validate_completion(layer.decision,
                slots=slots[layer.layer], accepted_pairs=layer.pairs)
        outcomes = {layer.layer: self.controllers[layer.layer].finish(layer.decision,
            slots=slots[layer.layer], accepted_pairs=layer.pairs) for layer in decision.layers}
        self._pending = None
        return outcomes

    def fail(self):
        """Poison the control loop after partial/unknown backend or rank failure.

        Individual backend rollback does not prove model-wide rollback. The
        engine must keep the lane paused, reload all ranks, and construct fresh
        controllers. This method never resumes requests or mutates storage.
        """
        self._failed = True
