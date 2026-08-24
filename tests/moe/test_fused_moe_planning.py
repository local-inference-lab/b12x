from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import b12x.moe.fused_moe._impl as fused_moe_impl
from b12x.moe import fused_moe


def _weight_plan() -> fused_moe.WeightsPlan:
    config = fused_moe.PackedConfig(
        source_format="fp4_e8m0_k32",
        w13_layout="w13",
    )
    return fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
    )


def _caps(*, block_size_m: int | None) -> fused_moe.Caps:
    weight_plan = _weight_plan()
    return fused_moe.Caps(
        max_tokens=64,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        config=weight_plan.checkpoint_config,
        weight_plan=weight_plan,
        w4a16_block_size_m=block_size_m,
    )


def _trellis_caps() -> fused_moe.Caps:
    config = fused_moe.TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "mcg",
            "rate": {"granularity": "per_expert_projection"},
            "scale": {
                "input_scales": {"vectors": "per_layer", "gains": "none"},
                "intermediate_scales": {
                    "vectors": "per_expert",
                    "gains": "none",
                },
                "output_scales": {"vectors": "per_layer", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        }
    )
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
    )
    return fused_moe.Caps(
        max_tokens=3072,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        config=config,
        weight_plan=weight_plan,
        w4a16_block_size_m=64,
    )


def _small_packed_caps() -> fused_moe.Caps:
    config = fused_moe.PackedConfig(source_format="compressed_tensors")
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=16,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=4,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        config=config,
        weight_plan=weight_plan,
    )


def _subset_router_caps() -> fused_moe.Caps:
    config = fused_moe.PackedConfig(source_format="compressed_tensors")
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        config=config,
        weight_plan=weight_plan,
    )


def _mapped_packed_caps() -> fused_moe.Caps:
    config = fused_moe.PackedConfig(source_format="compressed_tensors")
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=2,
        route_num_experts=12,
        device="cpu",
        config=config,
        weight_plan=weight_plan,
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

    assert required > 800 * 1024 * 1024
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


def test_mapped_packed_plan_covers_global_route_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_mapped_packed_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.weight_E == 8
    assert plan._core_workspace_plan.route_E == 12
    assert specs["expert_offsets"].shape == (13,)
    assert specs["expert_counts"].shape == (12,)


def test_unpinned_small_capacity_matches_reachable_block_8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    automatic = fused_moe.required_nbytes(_caps(block_size_m=None))
    exact = fused_moe.required_nbytes(_caps(block_size_m=8))
    oversized = fused_moe.required_nbytes(_caps(block_size_m=64))

    assert automatic == exact
    assert oversized - automatic > 64 * 1024 * 1024


def _k3_config() -> fused_moe.TrellisConfig:
    return fused_moe.TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "sqg_e4m3",
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
                "intermediate_scales": {
                    "vectors": "per_layer",
                    "gains": "per_expert",
                },
                "output_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {
                    "kind": "coupled_hadamard",
                    "pre_block_size": 512,
                    "post_block_size": 128,
                    "draw_granularity": "per_expert",
                },
            },
        }
    )


def test_trellis_plan_derives_private_recipe_from_config() -> None:
    config = _k3_config()
    plan = fused_moe.plan_weights(
        config=config,
        activation="situ",
        dtype=torch.float16,
        num_experts=8,
        hidden_size=256,
        intermediate_size=256,
    )

    assert plan.checkpoint_config is config
    assert plan.source_format == "b12x_trellis"
    assert plan.quant_modes == frozenset({"w4a16"})
    assert plan.trellis_codebook == "sqg_e4m3"
    assert plan.trellis_rate_granularity == "uniform"
    assert plan.coupled_hadamard
    assert plan.coupled_hadamard_blocks == (512, 128)


def test_runtime_rejects_unimplemented_orthogonal_trellis_combination() -> None:
    value = _k3_config().to_dict()
    value["codebook"] = "mcg"
    config = fused_moe.TrellisConfig.from_dict(value)

    with pytest.raises(NotImplementedError, match="sqg_e4m3.*uniform"):
        fused_moe.plan_weights(
            config=config,
            activation="situ",
            dtype=torch.float16,
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
        )


def test_runtime_rejects_schema_only_sqg_fp16_codebook() -> None:
    value = _k3_config().to_dict()
    value["codebook"] = "sqg_fp16"
    value["transform"]["expert"] = {"kind": "none"}
    config = fused_moe.TrellisConfig.from_dict(value)

    with pytest.raises(NotImplementedError, match="sqg_fp16"):
        fused_moe.plan_weights(
            config=config,
            activation="silu",
            dtype=torch.float16,
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
        )


def test_trellis_output_finalization_casts_into_caller_buffer() -> None:
    accumulated = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
    target = torch.empty_like(accumulated, dtype=torch.bfloat16)

    result = fused_moe_impl._finalize_trellis_output(
        SimpleNamespace(output_cast_target=target),
        accumulated,
    )

    assert result.data_ptr() == target.data_ptr()
    torch.testing.assert_close(result, accumulated.to(torch.bfloat16))


def test_projection_mixed_config_selects_fixed_mixed_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = _trellis_caps()
    plan = fused_moe.plan(caps)
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.implementation == "trellis_mixed3"
    assert plan._core_workspace_plan.projection_mixed_trellis
    assert specs["intermediate_cache13"].shape == (3072 * 8 * 1024,)
    assert specs["intermediate_cache2"].shape == (3072 * 8 * 512,)
    assert specs["rotation_a_gate"].shape == (3072 * 8, 6144)
    assert specs["rotation_a_up"].shape == (3072 * 8, 6144)
    assert specs["full_rotation_output"].shape == (3072, 6144)


def test_public_planning_api_has_no_quant_mode() -> None:
    config = fused_moe.PackedConfig(source_format="modelopt_nvfp4")
    with pytest.raises(TypeError, match="quant_modes"):
        fused_moe.plan_weights(
            config=config,
            quant_modes="w4a16",
            activation="silu",
            dtype=torch.bfloat16,
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
        )

    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=256,
        intermediate_size=256,
    )
    with pytest.raises(TypeError, match="quant_mode"):
        fused_moe.Caps(
            config=config,
            weight_plan=weight_plan,
            quant_mode="w4a16",
            max_tokens=1,
            num_topk=1,
            device="cpu",
        )


def test_caps_rejects_a_different_checkpoint_config() -> None:
    weight_plan = _weight_plan()
    with pytest.raises(ValueError, match="does not match"):
        fused_moe.Caps(
            config=fused_moe.PackedConfig(source_format="modelopt_nvfp4"),
            weight_plan=weight_plan,
            max_tokens=1,
            num_topk=1,
            device="cpu",
        )
