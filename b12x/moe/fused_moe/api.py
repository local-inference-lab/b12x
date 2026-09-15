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
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override: MoeDecodeConfig | None = None,
):
    """Declare capacity variants; preparation publishes executable states."""
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
    state = require_prepared(plan, "moe.decode")
    return replace(state.bind(**kwargs), plan=plan)


def run(*, binding: Binding):
    """Run only a binding created from a prepared plan."""
    plan = binding.plan
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
    "BtxSource",
    "BtxWeights",
    "ActivationMode",
    "ActivationSpec",
    "ExecutionCapacity",
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
