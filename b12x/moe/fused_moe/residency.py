"""Immutable, workload-specific expert placement and memory admission contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, kw_only=True)
class ExpertResidencyPlan:
    """Storage rows for one layer; expert identity remains the checkpoint index.

    Tuple order determines physical rows within each tier. Every expert occurs
    exactly once. A profile describes a workload, not a checkpoint invariant.
    """
    total_experts: int
    hbm_expert_ids: tuple[int, ...]
    grace_expert_ids: tuple[int, ...]
    layer: str
    model_fingerprint: str
    workload: str
    provenance: str
    selection_counts: tuple[int, ...] = ()
    phase: str = "all"
    version: int = 1

    def __post_init__(self):
        _integer("total_experts", self.total_experts, 1)
        if self.version != 1:
            raise ValueError("unsupported expert placement version")
        for name in ("hbm_expert_ids", "grace_expert_ids", "selection_counts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = self.hbm_expert_ids + self.grace_expert_ids
        if any(type(x) is not int for x in ids) or sorted(ids) != list(range(self.total_experts)):
            raise ValueError("placement must partition every expert exactly once")
        if self.selection_counts:
            if len(self.selection_counts) != self.total_experts:
                raise ValueError("selection counts must cover every expert")
            for count in self.selection_counts:
                _integer("selection count", count)
        for name in ("layer", "model_fingerprint", "workload", "provenance"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.phase not in ("all", "decode", "prefill"):
            raise ValueError("profile phase must be all, decode, or prefill")

    @property
    def expert_map(self) -> tuple[tuple[int, int], ...]:
        """Original expert -> (tier, row), with HBM=0 and Grace=1."""
        rows = [None] * self.total_experts
        for tier, ids in enumerate((self.hbm_expert_ids, self.grace_expert_ids)):
            for row, expert in enumerate(ids):
                rows[expert] = (tier, row)
        return tuple(rows)

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def expected_cold_fraction(self) -> float | None:
        total = sum(self.selection_counts)
        return None if not total else sum(self.selection_counts[e] for e in self.grace_expert_ids) / total

    def to_dict(self):
        return {**asdict(self), "profile_hash": self.profile_hash}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        digest = values.pop("profile_hash")
        result = cls(**values)
        if digest != result.profile_hash:
            raise ValueError("expert placement hash mismatch")
        return result


@dataclass(frozen=True, kw_only=True)
class ExpertMemoryBudget:
    """Per-plan admission limits, including reservations outside expert storage.

    The integration apportions these limits across layers and other operators.
    KV and safety reservations reduce the usable HBM limit before admission.
    """
    hbm_bytes: int
    grace_bytes: int
    hbm_safety_bytes: int = 0
    grace_safety_bytes: int = 0
    kv_reserved_bytes: int = 0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _integer(name, getattr(self, name))
        if self.hbm_safety_bytes + self.kv_reserved_bytes > self.hbm_bytes or self.grace_safety_bytes > self.grace_bytes:
            raise ValueError("memory reservations exceed the declared budget")

    def admit(self, memory: ExpertMemoryAccounting):
        if memory.hbm_total_bytes + self.hbm_safety_bytes + self.kv_reserved_bytes > self.hbm_bytes:
            raise ValueError("expert placement, scratch, and reservations exceed the HBM budget")
        if memory.grace_expert_bytes + self.grace_safety_bytes > self.grace_bytes:
            raise ValueError("expert placement and safety reserve exceed the Grace budget")


@dataclass(frozen=True, kw_only=True)
class ExpertMemoryAccounting:
    """Owned storage bytes; allocator alignment is included in each slab."""
    hbm_expert_bytes: int
    grace_expert_bytes: int
    scratch_bytes: int
    route_map_bytes: int

    @property
    def hbm_total_bytes(self):
        return self.hbm_expert_bytes + self.scratch_bytes + self.route_map_bytes


def profile_from_counts(*, counts: Iterable[int], hot_count: int | None = None,
                        hot_bytes: int | None = None, expert_bytes: int | None = None,
                        **metadata) -> ExpertResidencyPlan:
    """Rank one layer independently; ties use checkpoint expert order."""
    counts = tuple(counts)
    if not counts:
        raise ValueError("routing counts cannot be empty")
    for value in counts:
        _integer("selection count", value)
    if (hot_count is None) == (hot_bytes is None):
        raise ValueError("specify exactly one of hot_count or hot_bytes")
    if hot_bytes is not None:
        _integer("hot_bytes", hot_bytes)
        _integer("expert_bytes", expert_bytes, 1)
        hot_count = min(len(counts), hot_bytes // expert_bytes)
    _integer("hot_count", hot_count)
    if hot_count > len(counts):
        raise ValueError("hot_count exceeds the expert count")
    ranked = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
    # Canonical row order simplifies checkpoint copies; rank determines membership.
    return ExpertResidencyPlan(
        total_experts=len(counts), hbm_expert_ids=tuple(sorted(ranked[:hot_count])),
        grace_expert_ids=tuple(sorted(ranked[hot_count:])), selection_counts=counts, **metadata,
    )


def profiles_from_trace(records: Iterable[Mapping], *, experts_per_layer: Mapping[str, int],
                        hot_count: int | Mapping[str, int] | None = None,
                        hot_bytes: int | Mapping[str, int] | None = None,
                        expert_bytes: int | Mapping[str, int] | None = None,
                        phase="all", **metadata):
    """Count JSONL records {layer, phase, expert_ids}; -1 is an unused route.

    Each record contains a flat sequence of selected IDs, including duplicates.
    Out-of-range IDs are rejected so malformed traces cannot bias residency.
    """
    if phase not in ("all", "decode", "prefill"):
        raise ValueError("trace phase must be all, decode, or prefill")
    counts = {}
    for layer, size in experts_per_layer.items():
        _integer("expert count", size, 1)
        counts[str(layer)] = [0] * size
    for record in records:
        layer = str(record["layer"])
        if layer not in counts or record["phase"] not in ("decode", "prefill"):
            raise ValueError("trace layer or phase is not declared")
        for expert in record["expert_ids"]:
            if type(expert) is not int or expert < -1 or expert >= len(counts[layer]):
                raise ValueError("trace contains an invalid expert ID")
            if expert != -1 and (phase == "all" or record["phase"] == phase):
                counts[layer][expert] += 1
    def layer_value(value, layer):
        return value[layer] if isinstance(value, Mapping) else value
    return tuple(profile_from_counts(
        counts=values, hot_count=layer_value(hot_count, layer),
        hot_bytes=layer_value(hot_bytes, layer), expert_bytes=layer_value(expert_bytes, layer),
        layer=layer, phase=phase, **metadata,
    ) for layer, values in sorted(counts.items()))


def write_profiles(path: str | Path, profiles: Iterable[ExpertResidencyPlan]):
    """Write a versioned profile artifact with independently hashed layers."""
    profiles = tuple(profiles)
    if len({p.layer for p in profiles}) != len(profiles):
        raise ValueError("placement artifact contains duplicate layers")
    Path(path).write_text(json.dumps({"version": 1, "layers": [p.to_dict() for p in profiles]}, indent=2) + "\n")


def read_profiles(path: str | Path) -> tuple[ExpertResidencyPlan, ...]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        # The model artifact verifies its complete hash before exposing static
        # layer plans. Automatic reuse additionally requires profile.validate().
        from .automatic import ResidencyProfile
        return ResidencyProfile.from_dict(payload).placements
    if set(payload) != {"version", "layers"} or payload["version"] != 1:
        raise ValueError("unsupported placement artifact")
    profiles = tuple(ExpertResidencyPlan.from_dict(p) for p in payload["layers"])
    if len({p.layer for p in profiles}) != len(profiles):
        raise ValueError("placement artifact contains duplicate layers")
    return profiles
