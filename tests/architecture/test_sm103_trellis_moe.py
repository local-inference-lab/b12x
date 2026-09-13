"""Host-only planning and prewarm coverage for uniform SM103 Trellis MoE."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl, _sm103_trellis as backend
from b12x.policy import PolicyContext, PolicySource
from b12x.policy.generation.providers.moe import _config_covers_query
from tests.architecture.test_sm103 import B300
from tests.moe.test_trellis_config import _k3_config, _glm_config


def weight_plan(coupled=True, mixed=False):
    config = _glm_config() if mixed else _k3_config()
    if not coupled:
        config["transform"]["expert"] = {"kind": "none"}
    return fused_moe.plan_weights(
        source=fused_moe.TrellisConfig.from_dict(config),
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ" if coupled else "silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=384, hidden_size=5120, intermediate_size=2304
        ),
    )._impl


def caps(monkeypatch, *, coupled=True, **kwargs):
    import b12x.policy.context as context

    monkeypatch.setattr(
        context,
        "detect_device",
        lambda device: SimpleNamespace(identity=B300, ordinal=None),
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=8,
        device="cpu",
        weight_plan=weight_plan(coupled),
        quant_mode="w4a16",
        core_token_counts=(1, 4, 8),
        route_num_experts=768,
        policy_context=PolicyContext.for_identity(B300),
        **kwargs,
    )


@pytest.mark.parametrize("coupled", [False, True])
def test_public_scratch_plan_and_policy(coupled, monkeypatch):
    capacity = caps(monkeypatch, coupled=coupled)
    plan = _impl.plan_tp_moe_scratch(capacity, prewarm_launches=False)
    assert plan.full_rotation
    assert plan.launch_plan.implementation == backend.BACKEND
    assert plan.launch_plan.execution.gemm_engine.value == "trellis_tcgen05"
    assert plan.launch_plan.policy_resolution.source is PolicySource.HEURISTIC
    buffers = plan._backend_plan.buffers
    assert all(b.offset % 1024 == 0 for b in buffers)
    assert all(
        a.offset + a.nbytes <= b.offset
        for a, b in zip(buffers, buffers[1:], strict=False)
    )
    assert ("input_up" in {b.name for b in buffers}) is not coupled
    assert fused_moe.required_nbytes(capacity) == plan.scratch_specs()[0].nbytes
    query = fused_moe.MoeDecodeQuery(
        "w4a16", "b12x_trellis", capacity.activation, 384, 5120, 2304, 8, 8, 64
    )
    assert _config_covers_query(
        asdict(query), asdict(plan.launch_plan.policy_resolution.config)
    )
    with pytest.raises(RuntimeError, match="prewarmed"):
        plan._backend_plan.bind(
            plan,
            scratch=None,
            a=None,
            experts=None,
            topk_weights=None,
            topk_ids=None,
            output=None,
            activation_amax=None,
            route_expert_map=None,
            output_expert_map=None,
            unit_scale_contract=False,
        )
    with (
        patch("torch.cuda.device"),
        patch("torch.cuda.is_current_stream_capturing", return_value=True),
        pytest.raises(RuntimeError, match="before graph capture"),
    ):
        plan._backend_plan.prewarm(plan)


def test_unsupported_mixed_and_coupled_geometry_fail_before_compile():
    with pytest.raises(NotImplementedError, match="uniform projection"):
        backend.validate_weight_plan(weight_plan(coupled=False, mixed=True))
    with pytest.raises(NotImplementedError, match="divisible by 512"):
        backend.validate_weight_plan(replace(weight_plan(), hidden_size=4992))


def test_canonical_rates_are_all_precompiled(monkeypatch):
    import cutlass.cute as cute

    capacity = caps(monkeypatch)
    recorded = []

    def compile_fake(kernel, *args, **kwargs):
        recorded.append((kernel, args, kwargs))
        return object()

    with patch.object(cute, "compile", side_effect=compile_fake):
        launches = backend.compile_launches(capacity, offline=True)
    assert len(launches) == 18
    assert {name for name in launches if name.startswith("fc")} == {
        f"{stage}_k{rate}" for stage in ("fc1", "fc2") for rate in (2, 3, 4)
    }
    assert all(entry[2]["no_jit_engine"] for entry in recorded)
    assert all(entry[2]["options"] == "--gpu-arch=sm_103a" for entry in recorded)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_native_binding_retains_capacity_launches_and_checks_aliases(
    monkeypatch, coupled, bits
):
    import cutlass.cute as cute
    from b12x.moe._shared.execution import PreparedWeightLayout
    from b12x.moe._shared.kernels.w4a16.prepare import (
        PreparedW4A16MoeWeights,
        TrellisWeightState,
    )

    capacity = caps(monkeypatch, coupled=coupled)
    wp = replace(
        capacity.weight_plan,
        num_experts=3,
        hidden_size=512,
        intermediate_size=256,
        trellis_bits=bits,
    )
    capacity = replace(capacity, weight_plan=wp, num_topk=2, route_num_experts=6)
    plan = _impl.plan_tp_moe_scratch(capacity, prewarm_launches=False)
    with patch.object(cute, "compile", side_effect=lambda *a, **kw: object()):
        launches = backend.compile_launches(capacity, offline=True)
    plan = replace(
        plan,
        _backend_plan=replace(
            plan._backend_plan,
            launches=launches,
            lut=torch.zeros(16, dtype=torch.uint8),
        ),
    )
    words = 3 * 32 * 16 * 8 * bits
    scales = torch.ones(3, 512, dtype=torch.float16)
    unit = torch.ones(3)
    dummy = torch.zeros(4, dtype=torch.uint8)
    payload = PreparedW4A16MoeWeights(
        w13=torch.zeros(2 * words, dtype=torch.int32),
        w2=torch.zeros(words, dtype=torch.int32),
        w13_scale=dummy,
        w2_scale=dummy,
        w13_global_scale=unit,
        w2_global_scale=unit,
        workspace=torch.empty(0, dtype=torch.int32),
        hidden_size=512,
        intermediate_size=256,
        num_experts=3,
        is_gated=True,
        params_dtype=torch.bfloat16,
        fc1_tile_n=256,
        fc2_tile_n=256,
        w13_layout="trellis_t256_proj",
        trellis=TrellisWeightState(
            codebook="sqg_e4m3",
            bits=bits,
            gate_suh=scales,
            up_suh=scales if coupled else scales.clone(),
            down_svh=scales.clone(),
            intermediate_rotations=torch.ones(
                3, 256 * (6 if coupled else 3), dtype=torch.float16
            ),
            coupled_hadamard=coupled,
        ),
    )
    experts = _impl.B12XFP4ExpertWeights(
        plan=wp,
        a1_gscale=unit,
        a2_gscale=unit,
        w1_fp4=payload.w13,
        w2_fp4=payload.w2,
        w1_blockscale=dummy,
        w2_blockscale=dummy,
        w1_alphas=unit,
        w2_alphas=unit,
        representation=_impl._PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=payload,
        ),
    )
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    }
    a = torch.empty(8, 512, dtype=torch.bfloat16)
    weights = torch.empty(8, 2)
    mapping = torch.tensor([0, 1, 2, -1, 0, 1], dtype=torch.int32)
    for live in (8, 1, 4, 3):
        for dtype in (torch.int32, torch.int64):
            ids = torch.empty(live, 2, dtype=dtype)
            for mapped in (False, True):
                kwargs = dict(
                    scratch=scratch,
                    a=a[:live],
                    experts=experts,
                    topk_weights=weights[:live],
                    topk_ids=ids,
                    route_expert_map=mapping if mapped else None,
                    output_expert_map=mapping if mapped else None,
                )
                bound = fused_moe.bind(plan, **kwargs)
                assert bound.output.dtype == torch.float32 and bound.output.shape == (
                    live,
                    512,
                )
                assert all(
                    fn in launches.values() for fn, _ in bound._backend_binding.calls
                )
                assert len(bound._backend_binding.calls) == (7 if coupled else 8)
                external = torch.empty_like(a)
                rebound = fused_moe.bind(plan, output=external, **kwargs)
                assert rebound.output.data_ptr() == external.data_ptr()
                with pytest.raises(ValueError, match="alias inputs"):
                    fused_moe.bind(plan, output=a, **kwargs)
                with pytest.raises(ValueError, match="planned output"):
                    forged = bound.output.view(torch.bfloat16).reshape(-1, 512)[:live]
                    fused_moe.bind(plan, output=forged, **kwargs)
