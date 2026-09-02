"""Public surface for gemm.dense_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..._lib.gating import default_is_supported
from ...policy import PolicyContext, get_auto_policy
from ..blockscaled._linear import (
    MXFP8LinearWeight,
    TensorFP8LinearWeight,
    mxfp8_linear,
    tensor_fp8_linear,
)
from ..blockscaled.api import mm as _blockscaled_mm
from ._policy import (
    DENSE_LINEAR_POLICY,
    DenseLinearConfig,
    DenseLinearQuery,
)
from . import META

_LOAD_PATH_CODES = {"tma": 1, "cpasync": 2}


def _dtype_name(dtype: torch.dtype | str) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Immutable planning capacity for one packed dense linear weight."""

    device: torch.device | str
    recipe: str
    in_features: int
    out_features: int
    max_tokens: int
    output_dtype: torch.dtype | str = torch.bfloat16

    def query(self) -> DenseLinearQuery:
        return DenseLinearQuery(
            recipe=str(self.recipe),
            in_features=int(self.in_features),
            out_features=int(self.out_features),
            max_tokens=int(self.max_tokens),
            output_dtype=_dtype_name(self.output_dtype),
        )


@dataclass(frozen=True)
class Plan:
    """Resolved launch options for one capacity of one packed weight."""

    caps: Caps
    config: DenseLinearConfig
    policy_resolution: object | None = None

    def launch_codes(self) -> tuple[int, int, int, int, int, int]:
        """Integer encoding consumed by the blockscaled custom ops."""
        return (
            int(self.config.tile_m),
            int(self.config.tile_n),
            int(self.config.tile_k),
            int(self.config.split_k),
            _LOAD_PATH_CODES[self.config.load_path],
            2 if self.config.swap_ab else 1,
        )

    def launch_kwargs(self) -> dict[str, Any]:
        """Explicit ``dense_gemm`` overrides equivalent to this plan."""
        kwargs: dict[str, Any] = {
            "mma_tiler_mn": self.config.mma_tiler_mn,
            "load_path": self.config.load_path,
            "swap_ab": self.config.swap_ab,
        }
        if self.config.tile_k:
            kwargs["_tile_k_override"] = int(self.config.tile_k)
        if self.config.split_k:
            kwargs["_split_k_slices_override"] = int(self.config.split_k)
        return kwargs


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve the launch plan for ``caps`` exactly once."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be dense_linear.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(DENSE_LINEAR_POLICY, caps.query())
    return Plan(caps=caps, config=resolution.config, policy_resolution=resolution)


class PlanTable:
    """Plans for a capacity ladder, selected by the live ``expected_m``.

    Integrations plan every serving capacity once (before graph capture) and
    pick the smallest planned capacity that covers a launch; no policy lookup
    happens at run time.
    """

    def __init__(self, plans: dict[int, Plan]) -> None:
        if not plans:
            raise ValueError("PlanTable requires at least one plan")
        self._plans = dict(sorted((int(k), v) for k, v in plans.items()))
        self._capacities = tuple(self._plans)

    @property
    def capacities(self) -> tuple[int, ...]:
        return self._capacities

    def select(self, expected_m: int) -> Plan:
        expected_m = int(expected_m)
        for capacity in self._capacities:
            if expected_m <= capacity:
                return self._plans[capacity]
        return self._plans[self._capacities[-1]]

    def __iter__(self):
        return iter(self._plans.values())


def plan_table(
    caps: Caps,
    token_counts: tuple[int, ...],
    *,
    policy: PolicyContext | None = None,
) -> PlanTable:
    """Plan every capacity in ``token_counts`` for the geometry of ``caps``."""
    policy = policy or get_auto_policy(caps.device)
    plans: dict[int, Plan] = {}
    for tokens in sorted({int(t) for t in token_counts if int(t) > 0}):
        capacity_caps = Caps(
            device=caps.device,
            recipe=caps.recipe,
            in_features=caps.in_features,
            out_features=caps.out_features,
            max_tokens=tokens,
            output_dtype=caps.output_dtype,
        )
        plans[tokens] = plan(capacity_caps, policy=policy)
    return PlanTable(plans)


def mm(
    source: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    packed_weight: MXFP8LinearWeight | TensorFP8LinearWeight,
    *,
    plan: Plan | PlanTable | None,
    expected_m: int | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run a packed dense linear with a resolved plan (no policy lookup)."""
    if isinstance(plan, PlanTable):
        live = int(expected_m) if expected_m is not None else int(
            (source[0] if isinstance(source, tuple) else source).shape[0]
        )
        plan = plan.select(max(1, live))
    if isinstance(packed_weight, MXFP8LinearWeight):
        return mxfp8_linear(
            source,
            packed_weight,
            bias=bias,
            out_dtype=out_dtype,
            expected_m=expected_m,
            stream=stream,
            plan=plan,
        )
    if isinstance(packed_weight, TensorFP8LinearWeight):
        if isinstance(source, tuple):
            raise TypeError("tensor-FP8 dense linear takes the FP8 values tensor")
        return tensor_fp8_linear(
            source,
            packed_weight,
            bias=bias,
            out_dtype=torch.bfloat16 if out_dtype is None else out_dtype,
            expected_m=expected_m,
            stream=stream,
            plan=plan,
        )
    raise TypeError("packed_weight must be an MXFP8 or tensor-FP8 packed weight")


def mm_serialized(*args: Any, plan: Plan | PlanTable | None, expected_m: int | None = None, **kwargs: Any) -> torch.Tensor:
    """``blockscaled.mm`` for serialized (values, scale) pairs with a plan."""
    if isinstance(plan, PlanTable):
        live = int(expected_m) if expected_m is not None else int(args[0][0].shape[0])
        plan = plan.select(max(1, live))
    return _blockscaled_mm(*args, plan=plan, expected_m=expected_m, **kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "PlanTable",
    "DenseLinearConfig",
    "DenseLinearQuery",
    "plan",
    "plan_table",
    "mm",
    "mm_serialized",
    "is_supported",
]
