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


def weight_plan(coupled=True, mixed=False, canonical=False, codebook="sqg_e4m3"):
    config = _glm_config() if mixed else _k3_config()
    if not mixed:
        config["codebook"] = codebook
    config["transform"]["expert"] = (
        _k3_config()["transform"]["expert"] if coupled else {"kind": "none"}
    )
    result = fused_moe.plan_weights(
        source=fused_moe.TrellisConfig.from_dict(config),
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ" if coupled else "silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=384, hidden_size=5120, intermediate_size=2304
        ),
    )
    return result if canonical else result._impl


def canonical_execution(capacity, plan, experts, *, coupled, mixed=False):
    public = weight_plan(
        coupled, mixed=mixed, canonical=True,
        codebook=capacity.weight_plan.trellis_codebook,
    )
    public = replace(
        public,
        _impl=capacity.weight_plan,
        geometry=fused_moe.MoEGeometry(
            num_experts=capacity.weight_E,
            hidden_size=capacity.k,
            intermediate_size=capacity.n,
        ),
    )
    experts = fused_moe.PreparedExperts(plan=public, _impl=experts)
    execution = fused_moe.plan_execution(
        experts=experts,
        policy=capacity.policy_context,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=capacity.max_tokens,
            top_k=capacity.num_topk,
            warmup_token_counts=(1, 4),
            route_num_experts=capacity.route_num_experts,
        ),
    )
    # Compiler stubs let the public host binding exercise the exact prepared
    # native launch ABI without creating a CUDA context.
    execution._impl = plan
    execution._prewarmed = True
    return execution, experts


def caps(monkeypatch, *, coupled=True, mixed=False, codebook="sqg_e4m3", **kwargs):
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
        weight_plan=weight_plan(coupled, mixed=mixed, codebook=codebook),
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
    assert "input_up" in {b.name for b in buffers}
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


def test_mixed_plan_is_supported_and_invalid_coupled_geometry_fails():
    backend.validate_weight_plan(weight_plan(coupled=False, mixed=True))
    mixed = weight_plan(coupled=True, mixed=True)
    assert mixed.coupled_hadamard
    backend.validate_weight_plan(mixed)
    with pytest.raises(NotImplementedError, match="divisible by 512"):
        backend.validate_weight_plan(replace(weight_plan(), hidden_size=4992))
    with pytest.raises(NotImplementedError, match="SiTU"):
        backend.validate_weight_plan(
            replace(mixed, specs=(replace(mixed.specs[0], activation="silu"),))
        )


def test_canonical_rates_are_all_precompiled(monkeypatch):
    import cutlass.cute as cute

    capacity = caps(monkeypatch)
    recorded = []

    def compile_fake(kernel, *args, **kwargs):
        recorded.append((kernel, args, kwargs))
        return object()

    with patch.object(cute, "compile", side_effect=compile_fake):
        launches = backend.compile_launches(capacity, offline=True)
    assert len(launches) == 21
    assert {name for name in launches if name.startswith("fc")} == {
        f"{stage}_k{rate}" for stage in ("fc1", "fc2") for rate in (2, 3, 4)
    } | {f"fc1_k{rate}_dual" for rate in (2, 3, 4)}
    assert all(entry[2]["no_jit_engine"] for entry in recorded)
    assert all(entry[2]["options"] == "--gpu-arch=sm_103a" for entry in recorded)


@pytest.mark.parametrize(
    "coupled,split", [(False, None), (True, None), (True, 64), (True, 192)]
)
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_native_binding_retains_capacity_launches_and_checks_aliases(
    monkeypatch, coupled, split, bits
):
    import cutlass.cute as cute
    from b12x.moe._shared.execution import PreparedWeightLayout
    from b12x.moe._shared.kernels.w4a16.prepare import (
        PreparedW4A16MoeWeights,
        TrellisWeightState,
    )

    codebook = "sqg_fp16" if bits >= 5 else "sqg_e4m3"
    capacity = caps(monkeypatch, coupled=coupled, codebook=codebook)
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
            codebook=codebook,
            bits=bits,
            gate_suh=scales,
            up_suh=scales if coupled and split is None else scales.clone(),
            down_svh=scales.clone(),
            intermediate_rotations=torch.ones(
                3, 256 * (6 if coupled else 3), dtype=torch.float16
            ),
            coupled_hadamard=coupled,
            input_scale_split=split,
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
    execution, public_experts = canonical_execution(
        capacity, plan, experts, coupled=coupled
    )
    public_bound = fused_moe.bind(
        execution,
        scratch=scratch,
        a=a[:1],
        experts=public_experts,
        topk_ids=torch.zeros(1, 2, dtype=torch.int64),
        topk_weights=weights[:1],
    )
    assert len(public_bound._backend_binding.calls) == (
        7 if coupled and split is None else 8
    )
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
                assert len(bound._backend_binding.calls) == (
                    7 if coupled and split is None else 8
                )
                fc1_name = f"fc1_k{bits}" + ("_dual" if split is not None else "")
                selected = [
                    args
                    for fn, args in bound._backend_binding.calls
                    if fn is launches[fc1_name]
                ]
                assert len(selected) == 2
                if split is not None:
                    assert all(int(args[0][2]) == split for args in selected)
                    assert all(int(args[-3]) == 2 * live for args in selected)
                external = torch.empty_like(a)
                rebound = fused_moe.bind(plan, output=external, **kwargs)
                assert rebound.output.data_ptr() == external.data_ptr()
                with pytest.raises(ValueError, match="alias inputs"):
                    fused_moe.bind(plan, output=a, **kwargs)
                with pytest.raises(ValueError, match="planned output"):
                    forged = bound.output.view(torch.bfloat16).reshape(-1, 512)[:live]
                    fused_moe.bind(plan, output=forged, **kwargs)

    for invalid in (0, 256, True, 64.0):
        malformed = replace(
            payload, trellis=replace(payload.trellis, input_scale_split=invalid)
        )
        bad_owner = replace(
            experts, representation=replace(experts.representation, value=malformed)
        )
        with pytest.raises(ValueError, match="input-scale split"):
            fused_moe.bind(plan, **{**kwargs, "experts": bad_owner})
