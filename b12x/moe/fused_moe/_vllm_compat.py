"""Compatibility contract consumed by vLLM's B12X modular MoE backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .._shared.execution import MoEWeightPreparationPlan
from ._impl import (
    B12XFP4ExpertWeights,
    TPMoEPlan,
    TPMoEScratchCaps,
    TPMoEScratchPlan,
    plan_b12x_fp4_moe_weights,
    plan_tp_moe_execution,
    plan_tp_moe_scratch,
    prepare_b12x_fp4_moe_weights,
    tp_moe_required_nbytes,
)

Caps = TPMoEScratchCaps
Plan = TPMoEScratchPlan
ExecutionPlan = TPMoEPlan
WeightsPlan = MoEWeightPreparationPlan
ExpertWeights = B12XFP4ExpertWeights


def plan_weights(
    *,
    quant_modes: str | Sequence[str],
    source_format: str,
    activation: str,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    w13_layout: str = "w13",
    w4a16_layout: str | None = None,
    trellis_bits: int | None = None,
    trellis_tile_config: tuple[int, int, int, int] | None = None,
    coupled_hadamard: bool | None = None,
    trellis_codebook: str | None = None,
    trellis_rate_granularity: str | None = None,
    trellis_pair_kinds: Sequence[str] | frozenset[str] | None = None,
    coupled_hadamard_blocks: tuple[int, int] | None = None,
    numerical_recipe: str = "default",
) -> WeightsPlan:
    return plan_b12x_fp4_moe_weights(
        quant_modes=quant_modes,
        source_format=source_format,
        activation=activation,
        params_dtype=params_dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout=w13_layout,
        w4a16_layout=w4a16_layout,
        trellis_bits=trellis_bits,
        trellis_tile_config=trellis_tile_config,
        coupled_hadamard=coupled_hadamard,
        trellis_codebook=trellis_codebook,
        trellis_rate_granularity=trellis_rate_granularity,
        trellis_pair_kinds=trellis_pair_kinds,
        coupled_hadamard_blocks=coupled_hadamard_blocks,
        numerical_recipe=numerical_recipe,
    )


def prepare_weights(
    *,
    plan: WeightsPlan,
    params_dtype: torch.dtype,
    w1_fp4: torch.Tensor | None = None,
    w2_fp4: torch.Tensor | None = None,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
    w1_blockscale: torch.Tensor | None = None,
    w2_blockscale: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    btx_layer: object | None = None,
    btx_device: torch.device | str | None = None,
    dummy_scale: torch.Tensor | None = None,
    immutable_input_scales: bool = False,
) -> ExpertWeights:
    return prepare_b12x_fp4_moe_weights(
        plan=plan,
        params_dtype=params_dtype,
        w1_fp4=w1_fp4,
        w2_fp4=w2_fp4,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_blockscale=w1_blockscale,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        btx_layer=btx_layer,
        btx_device=btx_device,
        dummy_scale=dummy_scale,
        immutable_input_scales=immutable_input_scales,
    )


def plan_execution(
    *,
    num_tokens: int,
    num_topk: int,
    device: torch.device | str,
    weight_plan: WeightsPlan,
    quant_mode: str,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    apply_router_weight_on_input: bool = False,
    deterministic_output: bool | None = None,
) -> ExecutionPlan:
    return plan_tp_moe_execution(
        num_tokens=num_tokens,
        num_topk=num_topk,
        device=device,
        weight_plan=weight_plan,
        quant_mode=quant_mode,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        apply_router_weight_on_input=apply_router_weight_on_input,
        deterministic_output=deterministic_output,
    )


def plan(caps: Caps) -> Plan:
    return plan_tp_moe_scratch(caps)


def required_nbytes(caps: Caps) -> int:
    return tp_moe_required_nbytes(caps)


__all__ = [
    "Caps",
    "ExecutionPlan",
    "ExpertWeights",
    "Plan",
    "WeightsPlan",
    "plan",
    "plan_execution",
    "plan_weights",
    "prepare_weights",
    "required_nbytes",
]
