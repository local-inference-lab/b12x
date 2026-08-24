"""Public surface for moe.fused_moe (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import replace

import torch

from ..._lib.gating import default_is_supported
from .._shared.execution import (
    MoEWeightPreparationPlan as WeightsPlan,
)
from .._shared.routing import (
    route_topk,
)
from ._impl import (
    B12XFP4ExpertWeights as ExpertWeights,
)
from ._impl import (
    B12XTopKRouting as Routing,
)
from ._impl import (
    TPMoEFP4Binding as Binding,
)
from ._impl import (
    TPMoERouteBinding as RouteBinding,
)
from ._impl import (
    TPMoEScratchCaps as Caps,
)
from ._impl import (
    TPMoEScratchPlan as Plan,
)
from ._impl import (
    TPMoESparseFP4Binding as SparseBinding,
)
from ._impl import (
    build_tp_moe_route_binding as bind_route,
)
from ._impl import (
    build_tp_moe_sparse_fp4_binding as bind_sparse,
)
from ._impl import (
    clear_tp_moe_caches as clear_caches,
)
from ._impl import plan_b12x_fp4_moe_weights as _plan_weights
from ._impl import (
    plan_tp_moe_scratch as plan,
)
from ._impl import (
    tp_moe_required_nbytes as required_nbytes,
)
from ._impl import prepare_b12x_fp4_moe_weights as _prepare_weights
from ._impl import prepare_b12x_trellis_v2_weights as _prepare_trellis_weights
from ._impl import (
    prepare_w4a16_fc2_e8m0 as prepare_fc2_weights,
)
from ._impl import (
    prewarm_w4a16_fc2_e8m0 as prewarm_fc2,
)
from ._impl import (
    b12x_moe_fp4 as run,
)
from ._impl import (
    run_w4a16_fc2_e8m0 as run_fc2,
)
from ._impl import (
    b12x_route_experts_fast as route,
)
from ._impl import (
    b12x_sparse_moe_fp4 as run_sparse,
)
from . import META
from .config import PackedConfig, TrellisConfig
from .weights import PackedWeights, ScaleFactors, TrellisWeights


def plan_weights(
    *,
    config: PackedConfig | TrellisConfig,
    activation: str,
    dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> WeightsPlan:
    """Plan one checkpoint encoding; runtime recipes remain private policy."""

    if isinstance(config, TrellisConfig):
        if config.codebook.value == "sqg_fp16":
            raise NotImplementedError(
                "sqg_fp16 is defined by the checkpoint schema but is not "
                "implemented by the routed fused MoE runtime"
            )
        if config.rate.group_size is not None:
            raise NotImplementedError(
                "grouped trellis rates are defined by the checkpoint schema but "
                "are not implemented by the fused MoE runtime"
            )
        if config.transform.projection.kind != "scaled_hadamard" or (
            config.transform.projection.block_size != 128
        ):
            raise NotImplementedError(
                "fused MoE trellis execution requires scaled_hadamard(128)"
            )
        expert = config.transform.expert
        if expert.kind == "coupled_hadamard" and (
            config.codebook.value != "sqg_e4m3"
            or config.rate.granularity.value != "uniform"
        ):
            raise NotImplementedError(
                "fused MoE coupled_hadamard execution currently requires "
                "the sqg_e4m3 codebook with uniform rates"
            )
        if expert.kind == "coupled_hadamard" and (
            expert.pre_block_size,
            expert.post_block_size,
        ) != (512, 128):
            raise NotImplementedError(
                "fused MoE coupled_hadamard requires block sizes (512, 128)"
            )
        plan = _plan_weights(
            quant_modes="w4a16",
            source_format="b12x_trellis",
            activation=activation,
            params_dtype=dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout="w31",
            w4a16_layout="trellis_native",
            trellis_bits=3,
            trellis_tile_config=(
                (128, 256, 64, 256)
                if config.codebook.value == "mcg"
                else (64, 256, 64, 256)
            ),
            coupled_hadamard=expert.kind == "coupled_hadamard",
            trellis_codebook=config.codebook.value,
            trellis_rate_granularity=config.rate.granularity.value,
            coupled_hadamard_blocks=(
                None
                if expert.kind == "none"
                else (expert.pre_block_size, expert.post_block_size)
            ),
        )
        return replace(plan, checkpoint_config=config)
    if not isinstance(config, PackedConfig):
        raise TypeError("config must be a PackedConfig or TrellisConfig")
    private_recipe = {
        "modelopt_nvfp4": "nvfp4",
        "fp4_e8m0_k32": "w4a16",
        "compressed_tensors": "w4a16",
        "mxfp6_e2m3": "w6a8_mx",
    }[config.source_format]
    plan = _plan_weights(
        quant_modes=private_recipe,
        source_format=config.source_format,
        activation=activation,
        params_dtype=dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout=config.w13_layout,
    )
    return replace(plan, checkpoint_config=config)


def prepare_weights(
    *,
    plan: WeightsPlan,
    weights: PackedWeights | TrellisWeights,
) -> ExpertWeights:
    """Prepare the typed checkpoint tensor bundle selected by ``plan_weights``."""

    if plan.checkpoint_config is None:
        raise ValueError("weight plan has no checkpoint config")
    if isinstance(plan.checkpoint_config, TrellisConfig):
        return _prepare_trellis_weights(plan=plan, weights=weights)
    if not isinstance(plan.checkpoint_config, PackedConfig):
        raise TypeError("weight plan contains an unsupported checkpoint config")
    if not isinstance(weights, PackedWeights):
        raise TypeError("packed weight preparation requires PackedWeights")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(plan.io_dtype)
    if dtype is None:
        raise TypeError(f"unsupported MoE activation dtype {plan.io_dtype!r}")
    return _prepare_weights(
        plan=plan,
        params_dtype=dtype,
        w1_fp4=weights.w13,
        w2_fp4=weights.w2,
        w1_global_scale=weights.w13_global_scales,
        w2_global_scale=weights.w2_global_scales,
        w1_blockscale=weights.w13_block_scales,
        w2_blockscale=weights.w2_block_scales,
        a1_gscale=weights.input_scale,
        a2_gscale=weights.intermediate_scale,
    )


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "PackedConfig",
    "PackedWeights",
    "Plan",
    "Binding",
    "SparseBinding",
    "RouteBinding",
    "ExpertWeights",
    "Routing",
    "ScaleFactors",
    "TrellisConfig",
    "TrellisWeights",
    "WeightsPlan",
    "plan",
    "required_nbytes",
    "plan_weights",
    "prepare_weights",
    "prepare_fc2_weights",
    "prewarm_fc2",
    "bind",
    "bind_sparse",
    "bind_route",
    "run",
    "run_fc2",
    "run_sparse",
    "route",
    "route_topk",
    "is_supported",
    "clear_caches",
]
