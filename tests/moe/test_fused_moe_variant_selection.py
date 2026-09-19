"""Host-only selection rules for prepared fused-MoE variants and W4A16 launches."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe._preparation import (
    _FusedMoeCapacityState,
    _W4A16PrimaryLaunches,
    variant_for,
)


class _Variant:
    def __init__(self, count):
        self.count = count

    def bind(self, **kwargs):
        return (self.count, kwargs)


@pytest.mark.parametrize(
    "tokens,topk,experts,expected_unique",
    ((1, 6, 384, 6), (4, 6, 384, 14), (6, 6, 384, 22), (8, 6, 384, 29),
     (7, 3, 128, 13), (8, 6, 6, 6)),
)
def test_shared_40_workload_reuses_experts_across_distinct_topk_rows(
    tokens, topk, experts, expected_unique,
):
    """Count reuse across the batch, never duplicate experts within a token."""
    from b12x.moe.fused_moe.workloads import make_routing_ids

    for seed in (42, 43, 44, 45):
        ids = make_routing_ids(tokens, topk, experts, seed=seed)
        assert ids.shape == (tokens, topk)
        assert ids.dtype == torch.int32
        assert ids.unique().numel() == expected_unique
        assert all(row.unique().numel() == topk for row in ids)
        assert ids.min() >= 0 and ids.max() < experts
        torch.testing.assert_close(
            ids, make_routing_ids(tokens, topk, experts, seed=seed),
        )


def test_variant_for_preserves_exact_counts_and_reuses_prefill_capacity():
    variants = {count: _Variant(count) for count in (1, 2, 4, 8, 125, 128)}
    assert variant_for(variants, 4) is variants[4]
    assert variant_for(variants, 125) is variants[125]
    for count in (3, 11, 31, 126, 128):
        assert variant_for(variants, count) is variants[128]
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        variant_for(variants, 129)


def test_capacity_state_binds_the_planned_variant_with_the_live_activations():
    variants = {count: _Variant(count) for count in (4, 128)}
    state = _FusedMoeCapacityState(variants)
    activations = torch.empty(11, 16)
    count, kwargs = state.bind(a=activations, topk_ids=None)
    assert count == 128
    assert kwargs["a"] is activations
    with pytest.raises(TypeError):
        state.bind(a=torch.empty(11))
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        state.bind(a=torch.empty(129, 16))


def _launches(*, direct, route_pack, route_mode="auto"):
    return _W4A16PrimaryLaunches(
        tokens=8, route_mode=route_mode, packed="packed", packed_mapped="packed_mapped",
        direct="direct" if direct else None,
        direct_mapped="direct_mapped" if direct else None,
        topk_sum="topk_sum", mapped_topk_sum="mapped_topk_sum",
        route_pack=route_pack,
    )


@pytest.mark.parametrize("route_mode", ("auto", "direct"))
def test_w4a16_select_preserves_exact_direct_and_dynamic_packed_routes(route_mode):
    launches = _launches(direct=True, route_pack="route_pack", route_mode=route_mode)
    assert launches.select(
        tokens=8, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
    ) == ("direct", "topk_sum", None)
    assert launches.select(
        tokens=8, route_ids_dtype=torch.int64, has_route_map=False, activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3, route_ids_dtype=torch.int64, has_route_map=True, activation_amax=None,
    ) == ("packed_mapped", "topk_sum", "route_pack")
    with pytest.raises(RuntimeError, match="requested=9, prepared=8"):
        launches.select(
            tokens=9, route_ids_dtype=torch.int32, has_route_map=False, activation_amax=None,
        )


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_native_w4a16_selects_retained_micro_and_keeps_packed_boundary(dtype):
    packed = SimpleNamespace(small_m_direct_launches=tuple(
        SimpleNamespace(topk_ids_dtype=value) for value in (torch.int32, torch.int64)
    ))
    launches = replace(_launches(direct=False, route_pack="route_pack", route_mode="direct"), packed=packed)
    exact = dict(tokens=8, route_ids_dtype=dtype, has_route_map=False, activation_amax=None)
    assert launches.select(**exact) == (packed, "topk_sum", None)
    assert launches.select(**{**exact, "tokens": 3}) == (packed, "topk_sum", "route_pack")
    assert launches.select(**{**exact, "has_route_map": True}) == ("packed_mapped", "topk_sum", "route_pack")
    assert launches.select(**{**exact, "activation_amax": object()}) == (packed, "topk_sum", "route_pack")
    assert replace(launches, route_mode="packed").select(**exact) == (packed, "topk_sum", "route_pack")


def test_native_launch_metadata_tracks_capacity_when_packed_program_is_cached(monkeypatch):
    from b12x._lib.compile_plan import ProgramKey, program_keys
    from b12x.moe._shared.kernels.w4a16 import kernel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(kernel, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(kernel, "_FUSED_CACHE", {})
    compiled = []

    def compile_fused(*args, **kwargs):
        program = SimpleNamespace(__b12x_programs__=(ProgramKey("cute", f"fused-{len(compiled)}", "fused"),))
        compiled.append(program)
        return program

    def compile_direct(*, m, topk_ids_dtype, **kwargs):
        return SimpleNamespace(
            m=m, topk_ids_dtype=topk_ids_dtype,
            compiled=SimpleNamespace(__b12x_programs__=(
                ProgramKey("cute", f"direct-{m}-{topk_ids_dtype}", "direct"),
            )),
        )

    monkeypatch.setattr(kernel, "b12x_compile", compile_fused)
    monkeypatch.setattr(kernel, "_compile_w4a16_small_m_direct", compile_direct)
    packed_specializations = {}
    for m in (1, 2, 4, 8, 9, 8, 1):
        launch = kernel.compile_w4a16_fused_moe(
            size_m=m, hidden_size=256, intermediate_size=256, num_experts=8, top_k=2,
            activation="silu", apply_router_weight_on_input=False, zero_fc2_output=False,
            moe_block_size=8, max_m_blocks=32, sms=188,
            max_shared_mem=kernel._DEFAULT_MAX_SHARED_MEM, weight_layout="modelopt",
            force_tile_config=(128, 128, 128, 128),
        )
        assert launch.size_m == m
        assert launch.compiled is packed_specializations.setdefault(m == 1, launch.compiled)
        assert {(direct.m, direct.topk_ids_dtype) for direct in launch.small_m_direct_launches} == (
            {(m, torch.int32), (m, torch.int64)} if m <= 8 else set()
        )
        expected_programs = program_keys(launch.compiled) + tuple(
            key for direct in launch.small_m_direct_launches for key in program_keys(direct.compiled)
        )
        assert program_keys(launch) == expected_programs
    assert len(compiled) == 2
