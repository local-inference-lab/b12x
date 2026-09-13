"""Canonical mixed-rate preparation and native SM103 expert qualification."""

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe._sm103_trellis import _mixed_contract
from b12x.policy.generation.providers.trellis_reference import moe_reference
from tests.moe.test_trellis_config import _glm_config


def prepare_mixed(
    *,
    experts=5,
    hidden=256,
    width=128,
    uniform=False,
    device="cuda",
    dtype=torch.bfloat16,
    activation="silu",
):
    config = _glm_config()
    rates = [
        [3 if uniform else 3 + (e + p) % 3 for p in range(3)] for e in range(experts)
    ]
    sections, native = [], {}
    for expert in range(experts):
        for projection, bits in enumerate(rates[expert]):
            n, k = (width, hidden) if projection < 2 else (hidden, width)
            record = torch.randint(
                -32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int16
            )
            native[projection, expert] = record
            # Atom rows hold two adjacent intermediate tiles. FC1's K axis
            # and FC2's N axis are both the hidden dimension.
            if projection < 2:
                row = record.view(hidden // 16, width // 32, 2, 16 * bits).permute(
                    1, 2, 0, 3
                )
            else:
                row = record.view(width // 32, 2, hidden // 16, 16 * bits)
            sections.append(row.contiguous().view(width // 32, -1))
    atoms = torch.cat(sections, dim=1).contiguous().view(torch.uint8).to(device)
    plan = fused_moe.plan_weights(
        source=fused_moe.TrellisConfig.from_dict(config),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity=activation, io_dtype=dtype
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=width
        ),
    )
    weights = fused_moe.TrellisWeights(
        atoms=atoms,
        rate=torch.tensor(rates, device=device, dtype=torch.uint8) * 17,
        input_scales=fused_moe.ScaleFactors(
            (0.875 + 0.25 * torch.rand(hidden, device=device)).half()
        ),
        intermediate_scales=fused_moe.ScaleFactors(
            (0.875 + 0.25 * torch.rand(experts, 3, width, device=device)).half()
        ),
        output_scales=fused_moe.ScaleFactors(
            (0.875 + 0.25 * torch.rand(hidden, device=device)).half()
        ),
    )
    return fused_moe.prepare_weights(plan=plan, weights=weights), native, rates


@pytest.mark.parametrize(
    "experts,uniform", [(5, False), (5, True), (384, False), (384, True)]
)
def test_mixed_preparation_preserves_records_and_large_namespace(experts, uniform):
    if not torch.cuda.is_available():
        pytest.skip("canonical preparation requires a GPU")
    from types import SimpleNamespace

    torch.manual_seed(73)
    weights, native, rates = prepare_mixed(experts=experts, uniform=uniform)
    prepared = weights._impl.representation_for("w4a16")
    assert prepared.descriptor_local_bits == (24 if experts > 256 else 8)
    state, offsets, counts = _mixed_contract(
        SimpleNamespace(weight_E=experts, k=256, n=128, device=weights.device), prepared
    )
    assert state.intermediate_rotations.shape == (experts, 384)
    rows = prepared.descriptor_map.cpu().view(3, -1)
    for projection in range(3):
        for expert in sorted({0, 1, experts // 2, experts - 1}):
            value = int(rows[projection, expert])
            tier, local = (
                value >> prepared.descriptor_local_bits,
                value & ((1 << prepared.descriptor_local_bits) - 1),
            )
            bits = rates[expert][projection]
            assert tier == bits - 3 and local < counts[projection][tier]
            word_count = 256 * 128 * bits // 32
            start = offsets[projection][tier] + local * word_count
            payload = prepared.w13 if projection < 2 else prepared.w2
            restored = (
                payload[start : start + word_count]
                .view(torch.int16)
                .cpu()
                .view_as(native[projection, expert])
            )
            torch.testing.assert_close(
                restored, native[projection, expert], atol=0, rtol=0
            )
    source = torch.randn(4, 256, device=weights.device, dtype=torch.bfloat16) * 0.01
    ids = torch.tensor(
        [[0, experts - 1], [1, -1], [experts - 1, 0], [0, 1]], device=weights.device
    )
    router = torch.rand(4, 2, device=weights.device)
    expected = moe_reference(source, prepared, ids, router, activation_kind="silu")
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    if experts <= 256 and torch.cuda.get_device_capability() in ((12, 0), (12, 1)):
        execution = fused_moe.plan_execution(
            experts=weights,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=8, top_k=2, warmup_token_counts=(1, 4)
            ),
        )
        fused_moe.prewarm(execution)
        scratch = {
            s.name: torch.empty(s.shape, device=s.device, dtype=s.dtype)
            for s in execution.scratch_specs()
        }
        bound = fused_moe.bind(
            execution,
            scratch=scratch,
            experts=weights,
            a=source,
            topk_ids=ids,
            topk_weights=router,
        )
        actual = fused_moe.run(binding=bound).float()
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
            expected
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
        assert error < 0.02 and cosine >= 0.999


def _run_native_mixed(
    *, experts, uniform, dtype, activation, hidden=256, width=128, top_k=2
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("physical SM103 required for native mixed Trellis MoE")
    import b12x

    torch.manual_seed(912)
    weights, _, _ = prepare_mixed(
        experts=experts,
        uniform=uniform,
        dtype=dtype,
        activation=activation,
        hidden=hidden,
        width=width,
    )
    payload = weights._impl.representation_for("w4a16")
    plan = fused_moe.plan_execution(
        experts=weights,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8,
            top_k=top_k,
            warmup_token_counts=(1, 4),
            route_num_experts=2 * experts,
        ),
    )
    fused_moe.prewarm(plan)
    assert all(v.implementation == "tcgen05_trellis" for v in plan.variants)
    assert len({id(v._impl) for v in plan.variants}) == 1
    launches = tuple(id(fn) for fn in plan._impl._backend_plan.launches.values())
    assert len(launches) == 15
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    }
    source = torch.randn(8, hidden, device="cuda", dtype=dtype) * 0.01
    id_dtype = torch.int32 if dtype == torch.float16 else torch.int64
    ids = (torch.arange(8 * top_k, device="cuda", dtype=id_dtype) % 3).view(8, top_k)
    ids[1, 0] = -1
    if id_dtype == torch.int64:
        ids[2, 0] = 2**32 + 1
    router = torch.rand(8, top_k, device="cuda")
    route_map = torch.full((2 * experts,), -1, device="cuda", dtype=torch.int32)
    output_map = torch.full_like(route_map, -1)
    route_map[:3] = torch.tensor([0, experts - 1, 1], device="cuda", dtype=torch.int32)
    output_map[:3] = torch.tensor([1, 0, experts - 1], device="cuda", dtype=torch.int32)
    payload.global_to_combined.copy_(payload.global_to_combined.flip(0))
    external = torch.empty_like(source)

    def expected(live):
        return moe_reference(
            source[:live],
            payload,
            ids[:live],
            router[:live],
            activation_kind=activation,
            route_expert_map=route_map,
            output_expert_map=output_map,
        )

    def check(actual, reference):
        value = actual.float()
        assert torch.isfinite(value).all() and torch.count_nonzero(value)
        relative = torch.linalg.vector_norm(
            value - reference
        ) / torch.linalg.vector_norm(reference)
        cosine = torch.nn.functional.cosine_similarity(
            value.flatten(), reference.flatten(), dim=0
        )
        assert relative < 0.02 and cosine >= 0.999

    def bind(live, target=None):
        return fused_moe.bind(
            plan,
            scratch=scratch,
            experts=weights,
            a=source[:live],
            topk_ids=ids[:live],
            topk_weights=router[:live],
            output=target,
            route_expert_map=route_map,
            output_expert_map=output_map,
        )

    references = {live: expected(live) for live in (1, 3, 4, 8)}
    b12x.freeze_kernel_resolution("mixed Trellis serving contract")
    try:
        for live in (8, 1, 4, 3):
            for target in (None, external):
                external.fill_(float("nan"))
                bound = bind(live, target)
                assert len(bound._backend_binding.calls) == 9
                check(fused_moe.run(binding=bound), references[live])
                if target is not None:
                    assert torch.isnan(external[live:]).all()
        bound = bind(8)
        fused_moe.run(binding=bound)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=bound)
        before = bound.output.clone()
        source.mul_(0.8)
        payload.w13.bitwise_xor_(0x124)
        payload.rotations.down_svh.mul_(0.75)
        # Changing valid descriptors must affect the captured compressed path.
        rows = payload.descriptor_map.view(3, -1)
        rows[:, :experts].copy_(rows[:, :experts].flip(1))
        ids.fill_(0)
        route_map[0] = experts - 1
        reference = expected(8)
        owners = (
            source,
            ids,
            router,
            payload.w13,
            payload.w2,
            payload.descriptor_map,
            payload.global_to_combined,
            bound.output,
            *scratch.values(),
        )
        addresses = tuple(t.data_ptr() for t in owners)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert tuple(t.data_ptr() for t in owners) == addresses
        assert (
            tuple(id(fn) for fn in plan._impl._backend_plan.launches.values())
            == launches
        )
        check(bound.output, reference)
        assert not torch.equal(before, bound.output)
        ids.fill_(-1)
        source.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(bound.output) == 0
    finally:
        b12x.unfreeze_kernel_resolution()


@pytest.mark.parametrize("experts,uniform", [(5, False), (384, False), (384, True)])
@pytest.mark.parametrize(
    "dtype,activation", [(torch.float16, "situ"), (torch.bfloat16, "silu")]
)
def test_native_mixed_moe_graph_and_capacity(experts, uniform, dtype, activation):
    _run_native_mixed(
        experts=experts, uniform=uniform, dtype=dtype, activation=activation
    )


def test_native_mixed_moe_v41_geometry():
    _run_native_mixed(
        experts=384,
        uniform=False,
        dtype=torch.bfloat16,
        activation="silu",
        hidden=5120,
        width=2304,
        top_k=6,
    )
