"""Uniform Trellis preparation and deferred native SM103 MoE qualification."""

import pytest
import torch

from b12x.moe import fused_moe
from b12x.policy.generation.providers.trellis_reference import moe_reference
from tests.moe.test_trellis_config import _k3_config


def prepare_experts(*, coupled, bits, dtype, device, geometry=(3, 512, 256)):
    experts, hidden, width = geometry
    config = _k3_config()
    if not coupled:
        config["transform"]["expert"] = {"kind": "none"}
    for field in config["scale"]:
        config["scale"][field] = {"vectors": "per_expert", "gains": "none"}
    plan = fused_moe.plan_weights(
        source=fused_moe.TrellisConfig.from_dict(config),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="situ" if coupled else "silu", io_dtype=dtype
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=width
        ),
    )

    def scales(shape):
        return fused_moe.ScaleFactors(
            (0.75 + 0.5 * torch.rand(shape, device=device)).half()
        )

    bundle = fused_moe.TrellisWeights(
        atoms=torch.randint(
            0,
            256,
            (width // 32, experts * 3 * (hidden // 16) * 64 * bits),
            dtype=torch.uint8,
            device=device,
        ),
        rate=torch.tensor([bits * 17], dtype=torch.uint8, device=device),
        input_scales=scales((experts, hidden)),
        intermediate_scales=scales((experts, 3, width)),
        output_scales=scales((experts, hidden)),
        expert_transform_draws=torch.zeros(experts, dtype=torch.uint8, device=device)
        if coupled
        else None,
    )
    return fused_moe.prepare_weights(plan=plan, weights=bundle)


@pytest.mark.parametrize(
    "coupled,bits",
    [(False, 2), (False, 3), (False, 4), (True, 2), (True, 3), (True, 4)],
)
def test_canonical_preparation_and_independent_oracle(coupled, bits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA preparation requires a GPU")
    torch.manual_seed(123)
    dtype = torch.float16 if coupled else torch.bfloat16
    experts = prepare_experts(coupled=coupled, bits=bits, dtype=dtype, device="cuda")
    payload = experts._impl.representation_for("w4a16")
    assert payload.trellis.bits == bits
    assert experts.plan._impl.trellis_bits == bits == experts._impl.plan.trellis_bits
    assert payload.w13.dtype == torch.int32 and payload.w2.dtype == torch.int32
    source = torch.randn(4, 512, device="cuda", dtype=dtype) * 0.02
    ids = torch.tensor([[0, 1], [2, -1], [1, 0], [-1, 2]], device="cuda")
    weights = torch.rand(4, 2, device="cuda")
    reference = moe_reference(
        source,
        payload,
        ids,
        weights,
        activation_kind=experts.plan.activation.nonlinearity,
    )
    assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
    # Identical routes cancel after FC2; this checks routing and reduction in
    # the independent oracle without using the implementation's transforms.
    duplicated = ids[:, :1].expand(-1, 2).contiguous()
    cancelling = torch.tensor([[1.0, -1.0]], device="cuda").expand(4, 2).contiguous()
    cancelled = moe_reference(
        source,
        payload,
        duplicated,
        cancelling,
        activation_kind=experts.plan.activation.nonlinearity,
    )
    torch.testing.assert_close(cancelled, torch.zeros_like(cancelled), atol=0, rtol=0)

    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        return
    import b12x

    # Qualify preparation changes against the existing SM12x implementation.
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8, top_k=2, warmup_token_counts=(1, 4)
        ),
    )
    fused_moe.prewarm(plan)
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    }
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=source,
        experts=experts,
        topk_ids=ids,
        topk_weights=weights,
    )

    def check(value, expected):
        assert torch.isfinite(value).all() and torch.count_nonzero(value)
        relative = torch.linalg.vector_norm(
            value - expected
        ) / torch.linalg.vector_norm(expected)
        cosine = torch.nn.functional.cosine_similarity(
            value.flatten(), expected.flatten(), dim=0
        )
        assert relative < 0.01 and cosine >= 0.9999

    check(fused_moe.run(binding=binding), reference)
    graph = torch.cuda.CUDAGraph()
    b12x.freeze_kernel_resolution("canonical Trellis preparation regression")
    try:
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        source.mul_(0.75)
        changed = moe_reference(
            source,
            payload,
            ids,
            weights,
            activation_kind=experts.plan.activation.nonlinearity,
        )
        addresses = tuple(
            t.data_ptr() for t in (source, ids, binding.output, *scratch.values())
        )
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert (
            tuple(
                t.data_ptr() for t in (source, ids, binding.output, *scratch.values())
            )
            == addresses
        )
        check(binding.output, changed)
    finally:
        b12x.unfreeze_kernel_resolution()


def _run_native_moe(coupled, bits, id_dtype, *, geometry=(3, 512, 256), top_k=2):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("physical SM103 required for native Trellis MoE")
    import b12x

    expert_count, hidden, _ = geometry
    route_capacity = max(6, expert_count)
    torch.manual_seed(813)
    dtype = torch.float16 if coupled else torch.bfloat16
    experts = prepare_experts(
        coupled=coupled, bits=bits, dtype=dtype, device="cuda", geometry=geometry
    )
    payload = experts._impl.representation_for("w4a16")
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8,
            top_k=top_k,
            warmup_token_counts=(1, 4),
            route_num_experts=route_capacity,
        ),
    )
    assert all(v.implementation == "tcgen05_trellis" for v in plan.variants)
    assert len({id(v._impl) for v in plan.variants}) == 1
    fused_moe.prewarm(plan)
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    }
    source = torch.randn(8, hidden, device="cuda", dtype=dtype) * 0.02
    ids = (torch.arange(8 * top_k, device="cuda", dtype=id_dtype) % 6).view(8, top_k)
    ids[0, 1] = -1
    if id_dtype == torch.int64:
        ids[2, 1] = 2**32 + 1
    weights = torch.rand(8, top_k, device="cuda")
    route_map = torch.full((route_capacity,), -1, device="cuda", dtype=torch.int32)
    output_map = torch.full_like(route_map, -1)
    route_map[:6] = torch.tensor([0, -1, 1, 2, 0, 2], device="cuda", dtype=torch.int32)
    output_map[:6] = torch.tensor(
        [2, -1, 0, 1, -1, 2], device="cuda", dtype=torch.int32
    )
    output = torch.empty(8, hidden, device="cuda", dtype=dtype)

    def expected(live):
        return moe_reference(
            source[:live],
            payload,
            ids[:live],
            weights[:live],
            activation_kind=experts.plan.activation.nonlinearity,
            route_expert_map=route_map,
            output_expert_map=output_map,
        )

    def check(actual, reference):
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        if hidden == 512:
            torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.01)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.flatten(), dim=0
        )
        assert cosine >= 0.999
        assert (
            torch.linalg.vector_norm(actual.float() - reference)
            / torch.linalg.vector_norm(reference)
            < 0.01
        )

    refs = {live: expected(live) for live in (1, 4, 8, 3)}
    callables = tuple(id(fn) for fn in plan._impl._backend_plan.launches.values())
    b12x.freeze_kernel_resolution("Trellis serving contract")
    try:
        for live in (8, 1, 4, 3):
            for target in (None, output):
                output.fill_(float("nan"))
                binding = fused_moe.bind(
                    plan,
                    scratch=scratch,
                    a=source[:live],
                    experts=experts,
                    topk_ids=ids[:live],
                    topk_weights=weights[:live],
                    output=target,
                    route_expert_map=route_map,
                    output_expert_map=output_map,
                )
                result = fused_moe.run(binding=binding)
                check(result, refs[live])
                if target is not None:
                    assert torch.isnan(output[live:]).all()
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=source,
            experts=experts,
            topk_ids=ids,
            topk_weights=weights,
            route_expert_map=route_map,
            output_expert_map=output_map,
        )
        fused_moe.run(binding=binding)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        before = binding.output.clone()
        source.mul_(0.75)
        payload.w13.bitwise_xor_(0x124)
        payload.trellis.down_svh.mul_(0.8)
        ids.fill_(4)
        route_map[4] = expert_count - 1
        output_map[4] = expert_count - 1
        changed = expected(8)
        owners = (
            source,
            ids,
            weights,
            payload.w13,
            payload.w2,
            binding.output,
            *scratch.values(),
        )
        addresses = tuple(t.data_ptr() for t in owners)
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert tuple(t.data_ptr() for t in owners) == addresses
        assert (
            tuple(id(fn) for fn in plan._impl._backend_plan.launches.values())
            == callables
        )
        assert not torch.equal(binding.output, before)
        check(binding.output, changed)
    finally:
        b12x.unfreeze_kernel_resolution()


@pytest.mark.parametrize(
    "coupled,bits",
    [(False, 2), (False, 3), (False, 4), (True, 2), (True, 3), (True, 4)],
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_native_moe_oracle_capacity_binding_and_graph(coupled, bits, id_dtype):
    _run_native_moe(coupled, bits, id_dtype)


def test_v41_geometry_native_moe():
    _run_native_moe(True, 3, torch.int64, geometry=(384, 5120, 2304), top_k=6)
