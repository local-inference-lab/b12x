"""Descriptor capacity contracts for native mixed-rate expert dispatch."""

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    build_projection_tiered_maps,
    _check_descriptor_projection_counts,
)


def test_wide_projection_descriptors_cover_384_experts():
    tiers = ([0] * 384, [1] * 384, [2] * 384)
    route, descriptors = build_projection_tiered_maps(
        *tiers, tier_slots=(384,) * 3, device=torch.device("cpu"), local_index_bits=24
    )
    torch.testing.assert_close(route, torch.arange(384, dtype=torch.int32))
    rows = descriptors.view(3, 1152)
    for projection in range(3):
        torch.testing.assert_close(rows[projection, :384] & ((1 << 24) - 1), route)
        assert torch.all(rows[projection, :384] >> 24 == projection)
        assert torch.all(rows[projection, 384:] == -1)
    with pytest.raises(ValueError, match="eight-bit"):
        _check_descriptor_projection_counts(
            descriptors, 1152, gate_counts=(384, 0, 0), up_counts=(0, 384, 0)
        )
    with pytest.raises(ValueError, match="256"):
        build_projection_tiered_maps(
            *tiers, tier_slots=(384,) * 3, device=torch.device("cpu")
        )


@pytest.mark.parametrize("bits", [0, 16, 30, 32])
def test_projection_descriptor_format_is_explicit(bits):
    with pytest.raises(ValueError, match="8 or 24"):
        build_projection_tiered_maps(
            [0],
            [1],
            [2],
            tier_slots=(1, 1, 1),
            device=torch.device("cpu"),
            local_index_bits=bits,
        )


def host_prepared(experts, uniform, coupled=False):
    from b12x.moe.fused_moe.trellis import (
        PreparedProjectionTrellisWeights,
        _coalesce_payloads,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import MixedTrellisRotations
    from b12x.moe._shared.kernels.w4a16.prepare import (
        PreparedW4A16MoeWeights,
        TrellisWeightState,
    )
    from dataclasses import replace

    hidden, width = 512 if coupled else 256, 128
    bits = 24 if experts > 256 else 8
    membership = [
        [0 if uniform else (e + p) % 3 for e in range(experts)] for p in range(3)
    ]
    mapping, descriptors = build_projection_tiered_maps(
        *membership,
        tier_slots=(experts,) * 3,
        device=torch.device("cpu"),
        local_index_bits=bits,
    )
    counts = tuple(tuple(row.count(t) for t in range(3)) for row in membership)
    dummy, unit = torch.zeros(16, dtype=torch.uint8), torch.ones(experts)
    tiers = []
    for tier, rate in enumerate((3, 4, 5)):
        words = hidden * width * rate // 32
        tiers.append(
            PreparedW4A16MoeWeights(
                w13=torch.zeros(
                    max(counts[0][tier] + counts[1][tier], 1) * words, dtype=torch.int32
                ),
                w2=torch.zeros(max(counts[2][tier], 1) * words, dtype=torch.int32),
                w13_scale=dummy,
                w2_scale=dummy,
                w13_global_scale=unit,
                w2_global_scale=unit,
                workspace=torch.empty(0, dtype=torch.int32),
                hidden_size=hidden,
                intermediate_size=width,
                num_experts=experts,
                is_gated=True,
                params_dtype=torch.float16,
                fc1_tile_n=128,
                fc2_tile_n=128,
                trellis=TrellisWeightState(
                    codebook="mcg", bits=rate, coupled_hadamard=coupled
                ),
            )
        )
    w13, views13 = _coalesce_payloads(tuple(t.w13 for t in tiers))
    w2, views2 = _coalesce_payloads(tuple(t.w2 for t in tiers))
    tiers = tuple(
        replace(t, w13=a, w2=b) for t, a, b in zip(tiers, views13, views2, strict=True)
    )
    gate = torch.ones(1, hidden, dtype=torch.float16)
    return PreparedProjectionTrellisWeights(
        tiers=tiers,
        global_to_combined=mapping,
        descriptor_map=descriptors,
        rotations=MixedTrellisRotations(
            intermediate=torch.ones(
                3 * experts, (6 if coupled else 3) * width, dtype=torch.float16
            ),
            gate_suh=gate,
            up_suh=gate if coupled else gate.clone(),
            down_svh=torch.ones(1, hidden, dtype=torch.float16),
        ),
        gate_counts=counts[0],
        up_counts=counts[1],
        down_counts=counts[2],
        w13=w13,
        w2=w2,
        w13_scale=dummy,
        w2_scale=dummy,
        w13_global_scale=unit,
        w2_global_scale=unit,
        workspace=torch.empty(0, dtype=torch.int32),
        hidden_size=hidden,
        intermediate_size=width,
        num_experts=experts,
        params_dtype=torch.bfloat16,
        descriptor_local_bits=bits,
        coupled_hadamard=coupled,
    )


@pytest.mark.parametrize("experts", [5, 384])
@pytest.mark.parametrize("coupled", [False, True])
def test_mixed_bind_reuses_callables_across_rates_and_live_counts(
    monkeypatch, experts, coupled
):
    from dataclasses import replace
    from unittest.mock import patch
    import cutlass.cute as cute
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import _impl, _sm103_trellis as backend
    from b12x.moe._shared.execution import PreparedWeightLayout
    from tests.architecture.test_sm103_trellis_moe import caps, canonical_execution

    hidden = 512 if coupled else 256
    capacity = caps(monkeypatch, coupled=coupled, mixed=True)
    capacity = replace(
        capacity,
        num_topk=2,
        route_num_experts=2 * experts,
        weight_plan=replace(
            capacity.weight_plan,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=128,
        ),
    )
    plan = _impl.plan_tp_moe_scratch(capacity, prewarm_launches=False)
    with patch.object(cute, "compile", side_effect=lambda *a, **kw: object()):
        launches = backend.compile_launches(capacity, offline=True)
    assert len(launches) == 15
    assert {k for k in launches if k.startswith("fc")} == {"fc1_mixed", "fc2_mixed"}
    plan = replace(
        plan,
        _backend_plan=replace(
            plan._backend_plan,
            launches=launches,
            lut=torch.zeros(16, dtype=torch.uint8),
        ),
    )
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype) for s in plan.scratch_specs()
    }
    source, router = torch.empty(8, hidden, dtype=torch.bfloat16), torch.empty(8, 2)
    observed_counts = []
    for uniform in (False, True):
        payload = host_prepared(experts, uniform, coupled)
        state, offsets, counts = backend._mixed_contract(capacity, payload)
        assert state.coupled_hadamard is coupled
        assert (
            state.intermediate_rotations.data_ptr()
            == payload.rotations.intermediate.data_ptr()
        )
        weights = _impl.B12XFP4ExpertWeights(
            plan=capacity.weight_plan,
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
        with patch.object(
            cute, "compile", side_effect=AssertionError("resolution is frozen")
        ):
            for live in (8, 1, 4, 3):
                for dtype in (torch.int32, torch.int64):
                    bound = fused_moe.bind(
                        plan,
                        scratch=scratch,
                        a=source[:live],
                        experts=weights,
                        topk_ids=torch.empty(live, 2, dtype=dtype),
                        topk_weights=router[:live],
                    )
                    calls = bound._backend_binding.calls
                    assert len(calls) == (8 if coupled else 9) and all(
                        fn in launches.values() for fn, _ in calls
                    )
                    projection_calls = [
                        (fn, args)
                        for fn, args in calls
                        if fn in (launches["fc1_mixed"], launches["fc2_mixed"])
                    ]
                    for phase, (_, args) in enumerate(projection_calls):
                        assert int(args[6]) == phase and int(args[7]) == 3 * experts
                        assert tuple(int(v) for v in args[9]) == offsets[phase]
                        assert tuple(int(v) for v in args[10]) == counts[phase]
                        assert int(args[11]) == 2 * live
                    assert (
                        bound.output.shape == (live, hidden)
                        and bound.output.dtype == torch.float32
                    )
        observed_counts.append(counts)
        execution, public_experts = canonical_execution(
            capacity, plan, weights, coupled=coupled, mixed=True
        )
        public_bound = fused_moe.bind(
            execution,
            scratch=scratch,
            a=source[:1],
            experts=public_experts,
            topk_ids=torch.zeros(1, 2, dtype=torch.int64),
            topk_weights=router[:1],
        )
        assert len(public_bound._backend_binding.calls) == (8 if coupled else 9)
        if coupled:
            for corrupted, error in (
                (replace(payload, coupled_hadamard=False), "tier transforms"),
                (
                    replace(
                        payload,
                        rotations=replace(
                            payload.rotations, up_suh=payload.rotations.up_suh.clone()
                        ),
                    ),
                    "shared input",
                ),
            ):
                with pytest.raises(ValueError, match=error):
                    fused_moe.bind(
                        execution,
                        scratch=scratch,
                        a=source[:1],
                        experts=replace(
                            public_experts,
                            _impl=replace(
                                weights,
                                representation=replace(
                                    weights.representation, value=corrupted
                                ),
                            ),
                        ),
                        topk_ids=torch.zeros(1, 2, dtype=torch.int64),
                        topk_weights=router[:1],
                    )
    assert observed_counts[0] != observed_counts[1]


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "counts",
        "detached_tier",
        "payload_length",
        "rotations",
        "descriptor_length",
    ],
)
def test_mixed_binding_rejects_malformed_metadata(fault):
    from dataclasses import replace
    from types import SimpleNamespace
    from b12x.moe.fused_moe._sm103_trellis import _mixed_contract

    payload = host_prepared(5, False)
    if fault == "format":
        payload = replace(payload, descriptor_local_bits=24)
    elif fault == "counts":
        payload = replace(payload, down_counts=(0, 0, 0))
    elif fault == "detached_tier":
        payload = replace(
            payload,
            tiers=(
                replace(payload.tiers[0], w13=payload.tiers[0].w13.clone()),
                *payload.tiers[1:],
            ),
        )
    elif fault == "payload_length":
        payload = replace(payload, w13=payload.w13[:-1])
    elif fault == "rotations":
        payload = replace(
            payload,
            rotations=replace(
                payload.rotations, intermediate=payload.rotations.intermediate[:5]
            ),
        )
    else:
        payload = replace(payload, descriptor_map=payload.descriptor_map[:-1])
    with pytest.raises(ValueError):
        _mixed_contract(
            SimpleNamespace(weight_E=5, k=256, n=128, device=torch.device("cpu")),
            payload,
        )


def test_generator_identifies_native_mixed_materialized_path():
    from dataclasses import asdict
    from types import SimpleNamespace
    from b12x.moe.fused_moe._policy import MoeDecodeConfig
    from b12x.policy.generation.providers.moe_gpu_worker import (
        _concrete_candidate_path,
        _CandidateContractError,
    )

    config = MoeDecodeConfig("tcgen05_trellis", "internal", None)
    variant = SimpleNamespace(
        _impl=SimpleNamespace(policy_resolution=SimpleNamespace(config=config)),
        implementation=config.backend,
        execution=SimpleNamespace(
            gemm_engine=SimpleNamespace(value="trellis_tcgen05"),
            graph_partition=SimpleNamespace(value="materialized"),
        ),
    )
    kwargs = dict(
        geometry=None,
        case=SimpleNamespace(num_tokens=1),
        candidate=SimpleNamespace(config=asdict(config)),
        plan=SimpleNamespace(variant_for=lambda _: variant),
        prepared_payload=SimpleNamespace(weight_layout="trellis_mixed3"),
    )
    assert (
        _concrete_candidate_path(
            **kwargs,
            binding=SimpleNamespace(
                implementation=config.backend, _backend_binding=object()
            ),
        )
        == "w4a16.trellis_mixed3.tcgen05.materialized"
    )
    with pytest.raises(_CandidateContractError, match="compressed native"):
        _concrete_candidate_path(
            **kwargs, binding=SimpleNamespace(implementation=config.backend)
        )
