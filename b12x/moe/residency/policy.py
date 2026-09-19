"""Out-of-band recent-frequency decisions for quiescent expert slot exchanges.

One observation window belongs to one immutable placement generation. The engine
owns the pause, counter snapshots, exchange and acknowledgement; this module
allocates no device state and performs no execution or storage mutation.
"""
from dataclasses import dataclass

from .contracts import (
    PHASES, ResidencySlotSnapshot, RoutingSnapshot, RoutingObservationSpec,
    ResidencyExchangeSpec, _integer,
)


@dataclass(frozen=True, kw_only=True)
class ResidencyCacheConfig:
    """Explicit experimental thresholds; none is a measured production default."""
    max_pairs: int
    minimum_cold_selections: int
    minimum_score_gain: int
    minimum_residency_windows: int
    phase: str = "decode"

    def __post_init__(self):
        for name in ("max_pairs", "minimum_cold_selections", "minimum_score_gain"):
            _integer(name, getattr(self, name), 1)
        _integer("minimum_residency_windows", self.minimum_residency_windows)
        if self.phase not in PHASES:
            raise ValueError("cache policy requires one explicit routing phase")


@dataclass(frozen=True, kw_only=True)
class ResidencyCacheDecision:
    window: int
    expected: ResidencySlotSnapshot
    pairs: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    calls: int
    sampled_calls: int
    sample_every: int
    cold_selections: int
    unique_cold_experts: tuple[int, ...]
    cold_fraction: float | None
    counterfactual_cold_fraction: float | None
    below_threshold: int
    below_hysteresis: int
    protected_hot_experts: tuple[int, ...]
    unpaired_candidates: int
    observed_hits_since_promotion: tuple[tuple[int, int], ...]


@dataclass(frozen=True, kw_only=True)
class ResidencyCacheOutcome:
    window: int
    generation: int
    pairs: tuple[tuple[int, int], ...]
    hot_experts: tuple[int, ...]
    promotions: int
    observed_hits_after_promotion: int
    evicted_promotion_hits: tuple[tuple[int, int], ...]
    committed_payload_copy_bytes: int
    committed_map_copy_bytes: int


def _validate_slots(slots, experts):
    if not isinstance(slots, ResidencySlotSnapshot) or not slots.healthy:
        raise ValueError("cache decisions require a healthy residency snapshot")
    if len(slots.expert_map) != experts or not slots.preparation_id:
        raise ValueError("slot snapshot differs from layer geometry or preparation")
    _integer("generation", slots.generation)
    for tier in (0, 1):
        rows = sorted(row for t, row in slots.expert_map if t == tier)
        if rows != list(range(len(rows))):
            raise ValueError("slot snapshot must contain every physical row exactly once")
    if any(type(t) is not int or t not in (0, 1) or type(row) is not int for t, row in slots.expert_map):
        raise ValueError("slot snapshot requires integer tier and row IDs")


class ResidencyCacheController:
    """One layer's policy, serialized by the engine outside graph execution.

    Initialize after warmup with an authoritative counter baseline and slot
    snapshot. Finish every decision, including a declined or rolled-back one,
    before observing another window. Counter reset/repreparation or an external
    placement change requires a fresh controller and baseline.
    """
    def __init__(self, *, config: ResidencyCacheConfig,
                 observations: RoutingObservationSpec, exchange: ResidencyExchangeSpec,
                 slots: ResidencySlotSnapshot, baseline: RoutingSnapshot):
        if not isinstance(config, ResidencyCacheConfig) or not isinstance(observations, RoutingObservationSpec):
            raise TypeError("cache policy requires typed configuration and observation contracts")
        if not isinstance(exchange, ResidencyExchangeSpec):
            raise TypeError("cache policy requires a backend exchange contract")
        if not exchange.direct_backing_execution or not exchange.fixed_address_quiescent_exchange:
            raise ValueError("cache policy requires direct backing execution and fixed-address quiescent exchange")
        if observations.rank != observations.owner_rank or observations.phase != config.phase:
            raise ValueError("cache policy requires the authoritative rank and declared phase")
        _validate_slots(slots, observations.experts)
        if not isinstance(baseline, RoutingSnapshot):
            raise TypeError("cache baseline requires RoutingSnapshot")
        self.config, self.observations, self.exchange = config, observations, exchange
        self._slots = slots
        self._epoch = baseline.epoch
        self._baseline = self._row(baseline)
        self._window = 0
        self._pending = None
        self._entered = {e: 0 for e, (tier, _) in enumerate(slots.expert_map) if tier == 0}
        self._hits = {}
        self._promotions = 0
        self._total_hits = 0

    def _row(self, snapshot):
        if not isinstance(snapshot, RoutingSnapshot):
            raise TypeError("cache observation requires RoutingSnapshot")
        if snapshot.rank != self.observations.owner_rank or snapshot.epoch != self._epoch:
            raise ValueError("counter owner/reset epoch changed; establish a fresh policy baseline")
        matches = [row for row in snapshot.layers if row.layer == self.observations.layer and row.phase == self.config.phase]
        if len(matches) != 1 or len(matches[0].counts) != self.observations.experts:
            raise ValueError("counter snapshot is missing the declared layer/phase geometry")
        return matches[0]

    def observe(self, snapshot: RoutingSnapshot, *, slots: ResidencySlotSnapshot):
        """Propose pairs from completed requests; never perform an exchange."""
        if self._pending is not None:
            raise RuntimeError("finish the pending cache decision before observing another window")
        if slots != self._slots:
            raise ValueError("placement changed inside a counter window; establish a fresh baseline")
        row = self._row(snapshot)
        previous = self._baseline
        counts = tuple(b-a for a, b in zip(previous.counts, row.counts, strict=True))
        metadata = tuple(getattr(row, name)-getattr(previous, name)
                         for name in ("calls", "sampled_calls", "tokens", "sampled_tokens"))
        if any(value < 0 for value in (*counts, *metadata)):
            raise ValueError("cumulative routing counters decreased")
        calls, sampled_calls, tokens, sampled_tokens = metadata
        if (sampled_calls > calls or sampled_tokens > tokens
                or sum(counts) > sampled_tokens*self.observations.max_top_k
                or (sum(counts) and not sampled_calls)):
            raise ValueError("counter window has inconsistent selection/sampling totals")
        # Empty/unsampled windows and repeated polls do not age eviction guards.
        window = self._window + bool(sum(counts))
        hot = tuple(e for e, (tier, _) in enumerate(slots.expert_map) if tier == 0)
        cold = tuple(e for e, (tier, _) in enumerate(slots.expert_map) if tier == 1 and counts[e])
        protected = tuple(e for e in hot if window-self._entered[e] < self.config.minimum_residency_windows)
        candidates = sorted((e for e in cold if counts[e] >= self.config.minimum_cold_selections),
                            key=lambda e: (-counts[e], e))
        victims = sorted((e for e in hot if e not in protected), key=lambda e: (counts[e], e))
        pairs, hysteresis = [], 0
        for candidate, victim in zip(candidates, victims):
            if len(pairs) == self.config.max_pairs:
                break
            if counts[candidate]-counts[victim] < self.config.minimum_score_gain:
                # Remaining candidates score no higher and victims no lower.
                hysteresis = min(len(candidates), len(victims))-len(pairs)
                break
            pairs.append((candidate, victim))
        cold_count = sum(counts[e] for e in cold)
        remaining = cold_count-sum(counts[c]-counts[v] for c, v in pairs)
        hits = {e: n+counts[e] for e, n in self._hits.items()}
        decision = ResidencyCacheDecision(window=window, expected=slots, pairs=tuple(pairs), counts=counts,
            calls=calls, sampled_calls=sampled_calls, sample_every=self.observations.sample_every,
            cold_selections=cold_count, unique_cold_experts=cold,
            cold_fraction=cold_count/sum(counts) if sum(counts) else None,
            counterfactual_cold_fraction=remaining/sum(counts) if sum(counts) else None,
            below_threshold=len(cold)-len(candidates), below_hysteresis=hysteresis,
            protected_hot_experts=protected, unpaired_candidates=len(candidates)-len(pairs)-hysteresis,
            observed_hits_since_promotion=tuple(sorted(hits.items())))
        self._baseline, self._window, self._hits = row, window, hits
        self._total_hits += sum(counts[e] for e in hits)
        self._pending = decision
        return decision

    def finish(self, decision: ResidencyCacheDecision, *, slots: ResidencySlotSnapshot):
        """Acknowledge a committed exchange, or the unchanged placement on decline.

        After a resumable exchange error, acknowledge the restored snapshot.
        An unhealthy state or partial/unrelated transaction cannot be accepted.
        """
        if self._pending is None or decision is not self._pending:
            raise ValueError("cache decision is stale or belongs to another controller")
        _validate_slots(slots, self.observations.experts)
        pairs = ()
        if slots != self._slots:
            expected_map = list(self._slots.expert_map)
            for cold, hot in decision.pairs:
                expected_map[cold], expected_map[hot] = expected_map[hot], expected_map[cold]
            if (not decision.pairs or slots.preparation_id != self._slots.preparation_id
                    or slots.generation != self._slots.generation+1 or slots.expert_map != tuple(expected_map)):
                raise ValueError("completed exchange differs from the pending cache decision")
            pairs = decision.pairs
        evicted_hits = tuple((hot, self._hits[hot]) for _, hot in pairs if hot in self._hits)
        for cold, hot in pairs:
            self._entered.pop(hot)
            self._hits.pop(hot, None)
            self._entered[cold] = self._window
            self._hits[cold] = 0
        self._promotions += len(pairs)
        self._slots, self._pending = slots, None
        return ResidencyCacheOutcome(window=self._window, generation=slots.generation, pairs=pairs,
            hot_experts=tuple(e for e, (tier, _) in enumerate(slots.expert_map) if tier == 0),
            promotions=self._promotions, observed_hits_after_promotion=self._total_hits,
            evicted_promotion_hits=evicted_hits, committed_payload_copy_bytes=self.exchange.payload_copy_bytes_per_pair*len(pairs),
            committed_map_copy_bytes=self.exchange.map_copy_bytes_per_transaction if pairs else 0)
