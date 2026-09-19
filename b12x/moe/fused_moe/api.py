"""Public surface for :mod:`b12x.moe.fused_moe`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ..._lib.gating import default_is_supported
from ...preparation.types import FrozenMapping, Plan, require_prepared
from . import META
from ._impl import (
    TPMoEFP4Binding as Binding,
    TPMoERouteBinding as RouteBinding,
    TPMoESparseFP4Binding as SparseBinding,
    b12x_moe_fp4 as _run,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    clear_tp_moe_caches as clear_caches,
)
from .config import TrellisConfig
from ._tuning import MoeDecodeConfig, MoeDecodeQuery
from ._preparation import (
    FC2Invocation,
    RouteTopKInvocation,
    plan_fc2 as _plan_fc2,
    plan_route_topk as _plan_route_topk,
)
from .execution import ExecutionCapacity, RoutingSpec, plan_execution as _plan_execution
from .planning import (
    ActivationMode,
    ActivationSpec,
    MoEGeometry,
    WeightPlan,
    WeightPlanConstraints,
    plan_weights as _plan_weights,
    prepare_weights as _prepare_weights,
)
from .source import BtxSource, PackedSource, PackedSourceFormat, W13Layout, WeightSource
from .residency import ExpertResidencyPlan, ExpertMemoryBudget, ExpertMemoryAccounting, ResidencyUpdateCapacity
from ._residency_updates import (
    ResidencySlotSnapshot, ResidencyUpdateError, residency_slot_snapshot, exchange_expert_slots,
)
from .residency_cache import (
    ResidencyCacheConfig, ResidencyCacheController, ResidencyCacheDecision, ResidencyCacheOutcome,
)
from .automatic import (
    AutomaticResidencyConfig,
    ResidencyCalibrationConfig,
    ResidencyMonitorConfig,
    ResidencyModelSpec,
    ResidencyLayerSpec,
    ModelExpertMemoryBudget,
    ResidencyHardware,
    ResidencyProfileStore,
    ResidencyController,
    ResidencyProfile,
    RoutingSnapshot,
    LayerRoutingCounts,
)
from .routing_profile import (
    RoutingProfileQuery, RoutingProfileConfig, plan_routing_profile,
    bind_routing_profile, routing_profile_state,
)
from ._residency_tuning import ResidencyConfig, ResidencyQuery
from .weights import (
    BtxWeights,
    PackedWeights,
    PreparedExperts,
    PreparedWeightFormat,
    ScaleEncoding,
    ScaleFactors,
    TrellisWeights,
    WeightEncoding,
    WeightPacking,
)



def plan_weights(
    *,
    source: WeightSource,
    activation: ActivationSpec,
    geometry: MoEGeometry,
    constraints: WeightPlanConstraints | None = None,
) -> WeightPlan:
    """Declare canonical checkpoint representation and preparation only."""
    return _plan_weights(
        source=source,
        activation=activation,
        geometry=geometry,
        constraints=constraints,
    )


def prepare_weights(
    *, plan: WeightPlan, weights: PackedWeights | TrellisWeights | BtxWeights
) -> PreparedExperts:
    """Prepare the canonical weight representation owned by this layer."""
    return _prepare_weights(plan=plan, weights=weights)


def plan_execution(
    *,
    experts: PreparedExperts | WeightPlan,
    capacity: ExecutionCapacity,
    weights: PackedWeights | None = None,
    placement: ExpertResidencyPlan | None = None,
    memory_budget: ExpertMemoryBudget | None = None,
    updates: ResidencyUpdateCapacity | None = None,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override: MoeDecodeConfig | ResidencyConfig | None = None,
):
    """Declare capacity variants; preparation publishes executable states."""
    if placement is not None:
        from ._residency_preparation import plan as residency_plan
        return residency_plan(weight_plan=experts, weights=weights, capacity=capacity,
            placement=placement, memory_budget=memory_budget, routing=routing,
            invocation=invocation, override=override, updates=updates)
    if weights is not None or memory_budget is not None or updates is not None:
        raise ValueError("source weights and residency budgets require an expert placement")
    return _plan_execution(
        experts=experts,
        capacity=capacity,
        routing=routing,
        invocation=invocation,
        override=override,
    )


def plan_route_topk(
    invocation: RouteTopKInvocation, *, invocation_metadata: FrozenMapping = FrozenMapping(),
    override=None,
):
    """Declare a standalone native top-k route operation."""
    return _plan_route_topk(
        invocation, declaration_invocation=invocation_metadata, override=override
    )


def plan_fc2(
    *, experts: PreparedExperts, invocation: FC2Invocation,
    invocation_metadata: FrozenMapping = FrozenMapping(), override=None,
):
    """Declare standalone route-major W4A16 FC2 without a full MoE plan."""
    return _plan_fc2(
        experts, invocation, declaration_invocation=invocation_metadata, override=override
    )


def bind(plan: Plan, **kwargs: Any) -> Binding:
    """Bind live tensors within a session-prepared token capacity."""
    if plan.component_id == "moe.expert_residency" and plan.prepared is None:
        raise RuntimeError("expert residency plan is not prepared; use PreparationSession before binding")
    component = "moe.expert_residency" if plan.component_id == "moe.expert_residency" else "moe.decode"
    state = require_prepared(plan, component)
    return replace(state.bind(**kwargs), plan=plan)


def run(*, binding: Binding):
    """Run only a binding created from a prepared plan."""
    plan = binding.plan
    if plan.component_id == "moe.expert_residency":
        from ._residency_preparation import ResidencyBinding
        if plan.prepared is None:
            raise RuntimeError("expert residency plan is not prepared or has been released")
        state = require_prepared(plan, "moe.expert_residency", binding.a.device)
        if not isinstance(binding, ResidencyBinding):
            raise TypeError("hierarchical execution requires its prepared binding")
        if not binding.owners or binding.owners[0] is not state:
            raise ValueError("binding belongs to another preparation of this plan")
        return binding.run()
    require_prepared(plan, "moe.decode", binding.a.device)
    return _run(binding=binding)

def _state_for(plan: Plan, hidden_states: torch.Tensor):
    root = require_prepared(plan, "moe.decode", hidden_states.device)
    if hasattr(root, "variants"):
        from ._preparation import variant_for
        return variant_for(root.variants, hidden_states.shape[0])
    return root


def route_topk(
    plan: Plan, router_logits: torch.Tensor, topk_logits: torch.Tensor,
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, **kwargs: Any,
) -> None:
    """Run caller-owned top-k buffers through their retained route launcher."""
    state = require_prepared(plan, "moe.route_topk", router_logits.device)
    state.run(router_logits, topk_logits, topk_ids, topk_weights, **kwargs)


def bind_route(plan: Plan, *, hidden_states: torch.Tensor, **kwargs: Any) -> RouteBinding:
    """Bind native routing to an exact prepared MoE plan."""
    _state_for(plan, hidden_states)
    scratch = kwargs.pop("scratch")
    return replace(build_tp_moe_route_binding(
        scratch=scratch, hidden_states=hidden_states, **kwargs,
    ), plan=plan)


def route(plan: Plan, *, binding: RouteBinding) -> object:
    """Run a route binding through the selected prepared route launcher."""
    if binding.plan is not plan:
        raise ValueError("route binding belongs to another prepared plan")
    return _state_for(plan, binding.hidden_states).route(binding)


def bind_sparse(plan: Plan, *, hidden_states: torch.Tensor, **kwargs: Any) -> SparseBinding:
    """Bind sparse MoE math to an exact prepared native plan."""
    state = _state_for(plan, hidden_states)
    scratch = kwargs.pop("scratch")
    return replace(build_tp_moe_sparse_fp4_binding(
        scratch=scratch, hidden_states=hidden_states, experts=state.experts._impl, **kwargs,
    ), plan=plan)


def run_sparse(plan: Plan, *, binding: SparseBinding):
    """Run sparse MoE through its prepared route and expert launchers."""
    if binding.plan is not plan:
        raise ValueError("sparse binding belongs to another prepared plan")
    return _state_for(plan, binding.hidden_states).run_sparse(binding)


def run_fc2(
    plan: Plan, intermediate: torch.Tensor,
    route_expert_ids: torch.Tensor, route_weights: torch.Tensor, *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run standalone FC2 through its retained prepared native launcher."""
    state = require_prepared(plan, "moe.fc2_w4a16", intermediate.device)
    return state.run(intermediate, route_expert_ids, route_weights, output=output)


def is_supported(device=None) -> bool:
    """Return whether the active device satisfies the fused-MoE requirements."""
    return default_is_supported(device, requires=META.requires, archs=META.archs)


__all__ = [
    "ResidencyUpdateCapacity",
    "ResidencySlotSnapshot",
    "ResidencyUpdateError",
    "residency_slot_snapshot",
    "exchange_expert_slots",
    "ResidencyCacheConfig",
    "ResidencyCacheController",
    "ResidencyCacheDecision",
    "ResidencyCacheOutcome",
    "AutomaticResidencyConfig",
    "ResidencyCalibrationConfig",
    "ResidencyMonitorConfig",
    "ResidencyModelSpec",
    "ResidencyLayerSpec",
    "ModelExpertMemoryBudget",
    "ResidencyHardware",
    "ResidencyProfileStore",
    "ResidencyController",
    "ResidencyProfile",
    "RoutingSnapshot",
    "LayerRoutingCounts",
    "RoutingProfileQuery",
    "RoutingProfileConfig",
    "plan_routing_profile",
    "bind_routing_profile",
    "routing_profile_state",

    "BtxSource",
    "BtxWeights",
    "ActivationMode",
    "ActivationSpec",
    "ExecutionCapacity",
    "ExpertResidencyPlan",
    "ExpertMemoryBudget",
    "ExpertMemoryAccounting",
    "ResidencyConfig",
    "ResidencyQuery",
    "Binding",
    "RouteBinding",
    "RouteTopKInvocation",
    "FC2Invocation",
    "SparseBinding",
    "MoEGeometry",
    "MoeDecodeConfig",
    "MoeDecodeQuery",
    "PackedSource",
    "PackedSourceFormat",
    "PackedWeights",
    "PreparedExperts",
    "PreparedWeightFormat",
    "RoutingSpec",
    "ScaleEncoding",
    "ScaleFactors",
    "TrellisConfig",
    "TrellisWeights",
    "W13Layout",
    "WeightEncoding",
    "WeightPacking",
    "WeightPlan",
    "WeightPlanConstraints",
    "WeightSource",
    "bind",
    "bind_route",
    "bind_sparse",
    "clear_caches",
    "is_supported",
    "plan_execution",
    "plan_route_topk",
    "plan_fc2",
    "plan_weights",
    "prepare_weights",
    "route_topk",
    "route",
    "run_fc2",
    "run_sparse",
    "run",
]
