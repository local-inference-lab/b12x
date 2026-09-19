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
    updated_slot_map,
)
from .epoch import (
    ResidencyEpochBudget, ResidencyEpochCoordinator, ResidencyEpochDecision,
    ResidencyLayerDecision,
)

__all__ = [
    "ExpertPlacement", "LayerRoutingCounts", "ResidencyExchangeSpec",
    "ResidencySlotSnapshot", "ResidencyUpdateCapacity", "ResidencyUpdateError",
    "RoutingObservationSpec", "RoutingSnapshot", "ResidencyCacheConfig",
    "ResidencyCacheController", "ResidencyCacheDecision", "ResidencyCacheOutcome",
    "updated_slot_map", "ResidencyEpochBudget", "ResidencyEpochCoordinator",
    "ResidencyEpochDecision", "ResidencyLayerDecision",
]
