from __future__ import annotations

import pytest
import torch

from sparkinfer.moe import fused_moe
import sparkinfer.moe.fused_moe._impl as fused_moe_impl


def _weight_plan() -> fused_moe.WeightsPlan:
    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
    )


def _caps(*, block_size_m: int | None) -> fused_moe.Caps:
    return fused_moe.Caps(
        max_tokens=64,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=_weight_plan(),
        quant_mode="w4a16",
        w4a16_block_size_m=block_size_m,
    )


def _trellis_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
    )
    return fused_moe.Caps(
        max_tokens=3072,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        w4a16_block_size_m=64,
    )


def _small_packed_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=16,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )
    return fused_moe.Caps(
        max_tokens=4,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def _subset_router_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def test_required_nbytes_avoids_launch_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    def fail_launch_prewarm(**_kwargs) -> None:
        raise AssertionError("launch prewarm called")

    monkeypatch.setattr(
        fused_moe_impl,
        "_plan_full_rotation_w4a16_launches",
        fail_launch_prewarm,
    )
    caps = _trellis_caps()

    required = fused_moe.required_nbytes(caps)

    assert required > 1024 * 1024 * 1024
    assert "required_nbytes" in fused_moe.META.entry_points
    with pytest.raises(TypeError, match="TPMoEScratchCaps"):
        fused_moe.required_nbytes(object())


def test_required_nbytes_matches_scratch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = _caps(block_size_m=8)

    plan = fused_moe.plan(caps)

    assert fused_moe.required_nbytes(caps) == plan.scratch_specs()[0].shape[0]


def test_small_packed_plan_covers_direct_topk_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_small_packed_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert specs["fc1_c_tmp"].shape == (131072,)
    assert specs["fc2_c_tmp"].shape == (65536,)


def test_non_trellis_core_sizes_routes_for_weight_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_subset_router_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.route_E == 160
    assert specs["packed_route_indices"].shape == (512,)
    assert specs["block_expert_ids"].shape == (64,)
    assert specs["expert_offsets"].shape == (161,)


def test_unpinned_small_capacity_matches_reachable_block_8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    automatic = fused_moe.required_nbytes(_caps(block_size_m=None))
    exact = fused_moe.required_nbytes(_caps(block_size_m=8))
    oversized = fused_moe.required_nbytes(_caps(block_size_m=64))

    assert automatic == exact
    assert oversized - automatic > 64 * 1024 * 1024
