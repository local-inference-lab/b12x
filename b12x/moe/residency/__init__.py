"""Shared expert identity, observations and quiescent cache policy.

This host-only namespace imports no tensor library or device backend. Plans and
PreparationSession retain storage/program ownership; engines retain scheduling.
"""
from .contracts import (
    ExpertPlacement, LayerRoutingCounts, ResidencyExchangeSpec,
    ResidencySlotSnapshot, ResidencyUpdateCapacity, ResidencyUpdateError,
    RoutingObservationSpec, RoutingSnapshot,
)
from .policy import (
    ResidencyCacheConfig, ResidencyCacheController, ResidencyCacheDecision,
    ResidencyCacheOutcome,
)

__all__ = [
    "ExpertPlacement", "LayerRoutingCounts", "ResidencyExchangeSpec",
    "ResidencySlotSnapshot", "ResidencyUpdateCapacity", "ResidencyUpdateError",
    "RoutingObservationSpec", "RoutingSnapshot", "ResidencyCacheConfig",
    "ResidencyCacheController", "ResidencyCacheDecision", "ResidencyCacheOutcome",
]
