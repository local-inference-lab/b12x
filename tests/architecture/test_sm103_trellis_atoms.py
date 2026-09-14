"""Rate-group metadata and retained public SM103 atom-plane bindings."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl, _sm103_trellis as backend
from b12x.moe.fused_moe.trellis_atoms import normalize_rates, prepare_atom_weights
from tests.moe.test_trellis_config import _glm_config


@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("per_expert", [False, True])
def test_input_scale_axes_follow_declarations(experts, per_expert):
    from b12x.moe.fused_moe.trellis import _effective_input_scales
    from b12x.moe.fused_moe.config import TrellisScaleFactorsConfig

    declaration = TrellisScaleFactorsConfig.from_dict(
        {
            "vectors": "per_expert" if per_expert else "per_layer",
            "gains": "per_expert" if per_expert else "per_layer",
        },
        name="input_scales",
    )
    shape = (experts, 2, 512) if per_expert else (2, 512)
    vectors = torch.arange(torch.tensor(shape).prod()).reshape(shape).half() / 4096
    gains = torch.tensor([0.5, 1.5], dtype=torch.float16)
    if per_expert:
        gains = gains.expand(experts, -1).contiguous()
    gate, up = _effective_input_scales(
        fused_moe.ScaleFactors(vectors, gains),
        declaration,
        num_experts=experts,
        hidden_size=512,
        device=torch.device("cpu"),
    )
    expected = (vectors * gains.unsqueeze(-1)).reshape(-1, 2, 512)
    torch.testing.assert_close(gate, expected[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(up, expected[:, 1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "granularity", ["uniform", "per_layer", "per_expert", "per_expert_projection"]
)
@pytest.mark.parametrize("group_size", [None, 32, 256])
@pytest.mark.parametrize("codebook", ["mcg", "sqg_fp16"])
def test_rate_axes_preserve_both_plane_nibbles(granularity, group_size, codebook):
    config = _glm_config()
    config["codebook"] = codebook
    config["rate"] = {"granularity": granularity}
    if group_size is not None:
        config["rate"]["group_size"] = group_size
    config = fused_moe.TrellisConfig.from_dict(config)
    groups = 256 // (group_size or 256)
    shape = {
        "uniform": (1,),
        "per_layer": (1,),
        "per_expert": (3,),
        "per_expert_projection": (3, 3),
    }[granularity]
    if group_size is not None:
        shape += (groups,)
    first, last = (0x65, 0x56) if codebook == "sqg_fp16" else (0x42, 0x64)
    raw = torch.full(shape, first, dtype=torch.uint8)
    if group_size is not None:
        raw[..., -1] = last
    value, atomic = normalize_rates(
        config, raw, experts=3, intermediate_size=256, device=torch.device("cpu")
    )
    assert atomic and value.shape == (groups, 3, 3) and value.is_contiguous()
    assert torch.all(value[-1] == (last if group_size is not None else first))
    invalid_rates = (0x13, 0x37, 0xFF)
    if codebook == "sqg_fp16":
        invalid_rates += (0x45, 0x54, 0x75, 0x57)
    for invalid in invalid_rates:
        raw.fill_(invalid)
        with pytest.raises(ValueError, match="atom planes"):
            normalize_rates(
                config,
                raw,
                experts=3,
                intermediate_size=256,
                device=torch.device("cpu"),
            )


def cpu_prepare(plan, bundle):
    from b12x.moe.fused_moe.trellis import (
        _effective_input_scales,
        _effective_intermediate_scales,
        _effective_output_scales,
        _coupled_input_scales,
    )

    config = plan.source
    e, h, i = (
        plan.geometry.num_experts,
        plan.geometry.hidden_size,
        plan.geometry.intermediate_size,
    )
    device = bundle.atoms.device
    rates, _ = normalize_rates(
        config, bundle.rate, experts=e, intermediate_size=i, device=device
    )
    gate, up = _effective_input_scales(
        bundle.input_scales,
        config.scale.input_scales,
        num_experts=e,
        hidden_size=h,
        device=device,
    )
    if config.transform.expert.kind == "coupled_hadamard":
        gate, up = _coupled_input_scales(bundle, gate, up, i)
    middle = _effective_intermediate_scales(
        bundle.intermediate_scales,
        config.scale.intermediate_scales,
        num_experts=e,
        intermediate_size=i,
        device=device,
    )
    down = _effective_output_scales(
        bundle.output_scales,
        config.scale.output_scales,
        num_experts=e,
        hidden_size=h,
        device=device,
    )
    return prepare_atom_weights(
        config,
        bundle,
        rates,
        num_experts=e,
        hidden_size=h,
        intermediate_size=i,
        gate_suh=gate,
        up_suh=up,
        intermediate=middle,
        down_svh=down,
        params_dtype=plan.activation.io_dtype,
    )


@pytest.mark.parametrize("codebook", ["mcg", "sqg_e4m3", "sqg_fp16"])
@pytest.mark.parametrize("coupled,group_size", [(False, 32), (True, 64), (True, None)])
def test_atom_binding_reuses_capacity_kernels(
    monkeypatch, codebook, coupled, group_size
):
    import cutlass.cute as cute
    from b12x.moe._shared.execution import PreparedWeightLayout
    from tests._reference.trellis_atoms import atom_fixture
    from tests.architecture.test_sm103_trellis_moe import caps

    public, bundle, rates, _ = atom_fixture(
        codebook=codebook, group_size=group_size, coupled=coupled
    )
    payload = cpu_prepare(public, bundle)
    torch.testing.assert_close(payload.rates, rates, atol=0, rtol=0)
    assert payload.w13.data_ptr() == bundle.atoms.data_ptr()
    raw = replace(public._impl, trellis_group_size=payload.group_size)
    capacity = replace(
        caps(monkeypatch, coupled=coupled),
        weight_plan=raw,
        num_topk=2,
        route_num_experts=6,
    )
    plan = _impl.plan_tp_moe_scratch(capacity, prewarm_launches=False)
    with patch.object(cute, "compile", side_effect=lambda *args, **kwargs: object()):
        launches = backend.compile_launches(capacity, offline=True)
    assert len(launches) == (15 if coupled else 14)
    assert {name for name in launches if name.startswith("fc")} == (
        {"fc1_atoms", "fc2_atoms", "fc1_atoms_dual"}
        if coupled
        else {"fc1_atoms", "fc2_atoms"}
    )
    plan = replace(
        plan,
        _backend_plan=replace(
            plan._backend_plan,
            launches=launches,
            lut=torch.zeros(16, dtype=torch.uint8),
        ),
    )
    owner = _impl.B12XFP4ExpertWeights(
        plan=raw,
        a1_gscale=payload.w13_global_scale,
        a2_gscale=payload.w2_global_scale,
        w1_fp4=payload.w13,
        w2_fp4=payload.w2,
        w1_blockscale=payload.w13_scale,
        w2_blockscale=payload.w2_scale,
        w1_alphas=payload.w13_global_scale,
        w2_alphas=payload.w2_global_scale,
        representation=_impl._PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=payload,
        ),
    )
    prepared = fused_moe.PreparedExperts(plan=replace(public, _impl=raw), _impl=owner)
    execution = fused_moe.plan_execution(
        experts=prepared,
        policy=capacity.policy_context,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8, top_k=2, warmup_token_counts=(1, 4), route_num_experts=6
        ),
    )
    execution._impl = plan
    execution._prewarmed = True
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype) for s in plan.scratch_specs()
    }
    source, ids, router = (
        torch.empty(8, 512, dtype=torch.bfloat16),
        torch.zeros(8, 2, dtype=torch.int64),
        torch.ones(8, 2),
    )
    with patch.object(cute, "compile", side_effect=AssertionError("resolution frozen")):
        for live in (8, 1, 4, 3):
            bound = fused_moe.bind(
                execution,
                scratch=scratch,
                experts=prepared,
                a=source[:live],
                topk_ids=ids[:live],
                topk_weights=router[:live],
            )
            calls = bound._backend_binding.calls
            assert len(calls) == 8
            fc1 = launches["fc1_atoms_dual" if coupled else "fc1_atoms"]
            projections = [
                args for fn, args in calls if fn in (fc1, launches["fc2_atoms"])
            ]
            assert [int(args[7]) for args in projections] == [0, 1, 2]
            assert all(
                int(args[8]) == payload.row_stride_words and int(args[10]) == 2 * live
                for args in projections
            )
    for corrupted in (
        replace(payload, rates=payload.rates.flatten()),
        replace(payload, row_stride_words=payload.row_stride_words - 1),
        replace(payload, offsets=payload.offsets.int()),
    ):
        with pytest.raises(ValueError):
            backend._atom_contract(capacity, corrupted)
