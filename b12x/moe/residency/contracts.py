"""Host contracts for canonical expert identity and exclusive two-tier storage.

Tier 0 is resident execution storage; tier 1 is backing storage. These roles
do not imply separate memory capacity, coherent access or a particular device.
Backends retain allocation, admission, synchronization and execution ownership.
"""
from dataclasses import dataclass


PHASES = ("decode", "prefill", "verify", "draft")


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, kw_only=True)
class ExpertPlacement:
    """Canonical IDs in physical row order; every expert occupies one row."""
    total_experts: int
    resident_expert_ids: tuple[int, ...]
    backing_expert_ids: tuple[int, ...]

    def __post_init__(self):
        _integer("total_experts", self.total_experts, 1)
        for name in ("resident_expert_ids", "backing_expert_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = self.resident_expert_ids + self.backing_expert_ids
        if any(type(x) is not int for x in ids) or sorted(ids) != list(range(self.total_experts)):
            raise ValueError("placement must partition every expert exactly once")

    @property
    def expert_map(self) -> tuple[tuple[int, int], ...]:
        """Canonical expert -> (tier, row), resident=0 and backing=1."""
        rows = [None] * self.total_experts
        for tier, ids in enumerate((self.resident_expert_ids, self.backing_expert_ids)):
            for row, expert in enumerate(ids):
                rows[expert] = (tier, row)
        return tuple(rows)


@dataclass(frozen=True, kw_only=True)
class RoutingObservationSpec:
    """One layer/phase's cumulative counter semantics, independent of kernels.

    Replicated routing counts belong to one authoritative rank. Integrations
    supply canonical IDs and establish a fresh baseline after counter reset or
    repreparation. This descriptor neither prepares counters nor resolves TP/EP.
    """
    layer: str
    experts: int
    phase: str
    max_top_k: int
    sample_every: int = 1
    rank: int = 0
    owner_rank: int = 0

    def __post_init__(self):
        _text("layer", self.layer)
        for name in ("experts", "max_top_k", "sample_every"):
            _integer(name, getattr(self, name), 1)
        for name in ("rank", "owner_rank"):
            _integer(name, getattr(self, name))
        if self.phase not in PHASES:
            raise ValueError("routing observations require one explicit phase")


@dataclass(frozen=True, kw_only=True)
class ResidencyExchangeSpec:
    """Backend guarantees and successful transaction API-copy accounting.

    The backend must validate these guarantees during preparation. Declaring
    them is not hardware qualification. Copy bytes include journaling and map
    publication when required; they are not measured interconnect traffic.
    Rows within one layer must be interchangeable without reallocating storage.
    """
    backend: str
    direct_backing_execution: bool
    fixed_address_quiescent_exchange: bool
    payload_copy_bytes_per_pair: int
    map_copy_bytes_per_transaction: int
    backing_mode: str = "exclusive"

    def __post_init__(self):
        _text("backend", self.backend)
        if self.backing_mode not in ("exclusive", "canonical"):
            raise ValueError("backing_mode must be exclusive or canonical")
        for name in ("direct_backing_execution", "fixed_address_quiescent_exchange"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("payload_copy_bytes_per_pair", "map_copy_bytes_per_transaction"):
            _integer(name, getattr(self, name))


@dataclass(frozen=True, kw_only=True)
class ResidencyUpdateCapacity:
    """Opt-in quiescent exchanges with preparation-owned rollback storage."""
    max_pairs: int

    def __post_init__(self):
        _integer("max_pairs", self.max_pairs, 1)


@dataclass(frozen=True, kw_only=True)
class LayerRoutingCounts:
    layer: str
    phase: str
    counts: tuple[int, ...]
    calls: int = 0
    sampled_calls: int = 0
    tokens: int = 0
    sampled_tokens: int = 0

    def __post_init__(self):
        _text("layer", self.layer)
        if self.phase not in PHASES:
            raise ValueError("routing statistics require an explicit engine phase")
        object.__setattr__(self, "counts", tuple(self.counts))
        for n in (*self.counts, self.calls, self.sampled_calls, self.tokens, self.sampled_tokens):
            _integer("routing observation", n)
            if n >= 2**64:
                raise ValueError("routing counter overflow")
        if self.sampled_calls > self.calls or self.sampled_tokens > self.tokens:
            raise ValueError("sampled observations exceed total observations")


@dataclass(frozen=True, kw_only=True)
class RoutingSnapshot:
    """Cumulative counters in one reset epoch, copied at a quiescent boundary."""
    epoch: int
    rank: int
    layers: tuple[LayerRoutingCounts, ...]

    def __post_init__(self):
        _integer("epoch", self.epoch)
        _integer("rank", self.rank)
        object.__setattr__(self, "layers", tuple(self.layers))
        if len({(x.layer, x.phase) for x in self.layers}) != len(self.layers):
            raise ValueError("snapshot contains duplicate layer/phase records")


@dataclass(frozen=True, kw_only=True)
class ResidencySlotSnapshot:
    preparation_id: str
    generation: int
    expert_map: tuple[tuple[int, int], ...]
    healthy: bool


class ResidencyUpdateError(RuntimeError):
    """An exchange failed; resumable is true only after successful rollback."""
    def __init__(self, message, *, resumable):
        super().__init__(message)
        self.resumable = resumable
