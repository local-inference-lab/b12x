from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
import b12x.moe.fused_moe._impl as fused_moe_impl


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
        source_format="btx",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
        trellis_bits=3,
        trellis_codebook="mcg",
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


def _mapped_packed_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=2,
        route_num_experts=12,
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


def _btx_plan(**overrides) -> fused_moe.WeightsPlan:
    kwargs = dict(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=8,
        hidden_size=256,
        intermediate_size=256,
        trellis_bits=2,
        trellis_codebook="sqg_e4m3",
        coupled_hadamard=True,
        trellis_tile_config=(128, 128, 128, 128),
    )
    kwargs.update(overrides)
    return fused_moe.plan_weights(**kwargs)


def test_btx_plan_round_trips_declarations() -> None:
    plan = _btx_plan()
    assert plan.source_format == "btx"
    assert plan.trellis_codebook == "sqg_e4m3"
    assert plan.trellis_rate_structure == "uniform"
    assert plan.trellis_pair_kinds is None
    assert plan.coupled_hadamard
    assert plan.coupled_hadamard_blocks == (512, 128)

    pair_plan = _btx_plan(
        trellis_bits=3,
        coupled_hadamard=False,
        trellis_tile_config=(64, 256, 64, 256),
        trellis_rate_structure="per_expert_pair",
        trellis_pair_kinds=("P33", "P43"),
    )
    assert pair_plan.trellis_pair_kinds == frozenset({"P33", "P43"})
    assert pair_plan.coupled_hadamard_blocks is None


def test_btx_plan_supports_w4a8_mx_uniform() -> None:
    plan = _btx_plan(quant_modes="w4a8_mx")
    assert plan.quant_modes == frozenset({"w4a8_mx"})
    with pytest.raises(ValueError, match="uniform-rate"):
        _btx_plan(
            quant_modes="w4a8_mx",
            trellis_bits=3,
            coupled_hadamard=False,
            trellis_tile_config=(64, 256, 64, 256),
            trellis_rate_structure="per_expert_pair",
            trellis_pair_kinds=("P33", "P43"),
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"trellis_codebook": None}, "trellis_codebook"),
        ({"trellis_codebook": "sqg_xor_cheb_t12"}, "trellis_codebook"),
        ({"trellis_codebook": "sqg_fp16", "trellis_bits": 3}, "sqg_fp16"),
        (
            {"trellis_codebook": "mcg", "coupled_hadamard": True},
            "qualified only",
        ),
        (
            {"coupled_hadamard_blocks": (256, 128)},
            r"blocks \(512, 128\)",
        ),
        (
            {
                "trellis_bits": 3,
                "coupled_hadamard": False,
                "trellis_tile_config": (64, 256, 64, 256),
                "trellis_rate_structure": "per_expert_pair",
                "trellis_pair_kinds": ("P33", "P44"),
            },
            "no fused execution arm",
        ),
        (
            {
                "trellis_bits": 3,
                "coupled_hadamard": False,
                "trellis_tile_config": (64, 256, 64, 256),
                "trellis_rate_structure": "per_expert_pair",
                "trellis_pair_kinds": ("P24", "P43"),
            },
            "pair-kind sets must be",
        ),
        (
            {
                "trellis_bits": 2,
                "coupled_hadamard": False,
                "trellis_rate_structure": "per_expert_pair",
                "trellis_pair_kinds": ("P33", "P43"),
            },
            "trellis_bits=3",
        ),
        (
            {
                "trellis_bits": 3,
                "trellis_rate_structure": "per_expert_pair",
                "trellis_pair_kinds": ("P33", "P43"),
            },
            "only for uniform rate structures",
        ),
        (
            {"trellis_pair_kinds": ("P33", "P43")},
            "uniform btx rates declare no trellis_pair_kinds",
        ),
    ],
)
def test_btx_plan_fails_closed(overrides, match) -> None:
    if overrides.get("trellis_codebook") == "mcg":
        overrides.setdefault("trellis_bits", 3)
    with pytest.raises(ValueError, match=match):
        _btx_plan(**overrides)


def test_btx_declarations_require_btx_source() -> None:
    with pytest.raises(ValueError, match="trellis source format"):
        fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="modelopt_nvfp4",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
            trellis_codebook="sqg_e4m3",
        )


def test_btx_pair_kind_derivation_from_plan() -> None:
    derive = fused_moe_impl._fc_trellis_pair_kind

    class _Stub:
        def __init__(self, kinds):
            self.trellis_pair_kinds = kinds

    assert derive(_Stub(frozenset({"P33", "P43"}))) == "P33_P43"
    assert derive(_Stub(frozenset({"P33", "P24"}))) == "PDYNAMIC"
    assert derive(_Stub(frozenset({"P33"}))) == "PDYNAMIC"
    assert derive(_Stub(None)) is None
