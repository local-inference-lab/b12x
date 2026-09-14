"""Canonical atom preparation and deferred complete SM103 expert execution."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x.moe import fused_moe
from b12x.policy.generation.providers.trellis_reference import moe_reference
from tests._reference.trellis_atoms import atom_fixture


@pytest.mark.parametrize("codebook", ["mcg", "sqg_e4m3", "sqg_fp16"])
@pytest.mark.parametrize("coupled", [False, True])
def test_atom_preparation_rejects_invalid_storage_and_sm12x(codebook, coupled):
    if not torch.cuda.is_available():
        pytest.skip("CUDA preparation requires a GPU")
    public, bundle, _, _ = atom_fixture(
        codebook=codebook, group_size=64, coupled=coupled, device="cuda"
    )
    prepared = fused_moe.prepare_weights(plan=public, weights=bundle)
    payload = prepared._impl.representation_for("w4a16")
    assert prepared.plan._impl.trellis_group_size == payload.group_size == 64
    assert payload.w13.data_ptr() == bundle.atoms.data_ptr()
    assert payload.trellis.coupled_hadamard is coupled
    assert payload.trellis.input_scale_split == (128 if coupled else None)
    if torch.cuda.get_device_capability() in ((12, 0), (12, 1)):
        with pytest.raises(
            (UnsupportedArchitectureError, NotImplementedError), match="SM103 backend"
        ):
            fused_moe.plan_execution(
                experts=prepared,
                capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=2),
            )
    source = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    ids = torch.tensor([[0, 1], [2, -1]], device="cuda")
    router = torch.ones(2, 2, device="cuda")
    reference = moe_reference(
        source,
        payload,
        ids,
        router,
        activation_kind=public.activation.nonlinearity,
    )
    assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
    poisoned = bundle.atoms.clone()
    poisoned[0, -1] = 1
    for atoms, message in (
        (poisoned, "padding"),
        (bundle.atoms[:, :16].contiguous(), "shorter"),
        (
            torch.empty(bundle.atoms.numel() + 1, dtype=torch.uint8, device="cuda")[
                1:
            ].view_as(bundle.atoms),
            "aligned",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            fused_moe.prepare_weights(plan=public, weights=replace(bundle, atoms=atoms))
    with pytest.raises(ValueError, match="group boundary|post-transform blocks"):
        fused_moe.prepare_weights(
            plan=public,
            weights=replace(
                bundle, global_intermediate_size=512, intermediate_offset=32
            ),
        )


@pytest.mark.parametrize("codebook", ["mcg", "sqg_e4m3", "sqg_fp16"])
@pytest.mark.parametrize(
    "coupled,group_size,width",
    [
        (False, 32, 256),
        (True, 64, 256),
        (True, None, 384),
    ],
)
def test_native_atom_moe_graph_and_capacity(codebook, coupled, group_size, width):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("physical SM103 required for complete grouped Trellis MoE")
    import b12x

    public, bundle, _, _ = atom_fixture(
        codebook=codebook,
        group_size=group_size,
        coupled=coupled,
        width=width,
        device="cuda",
    )
    experts = fused_moe.prepare_weights(plan=public, weights=bundle)
    payload = experts._impl.representation_for("w4a16")
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=8,
            top_k=2,
            warmup_token_counts=(1, 4),
            route_num_experts=6,
        ),
    )
    fused_moe.prewarm(plan)
    launches = tuple(id(fn) for fn in plan._impl._backend_plan.launches.values())
    assert len(launches) == (15 if coupled else 14)
    assert all(v.implementation == "tcgen05_trellis" for v in plan.variants)
    assert len({id(v._impl) for v in plan.variants}) == 1
    scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    }
    source = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16) * 0.01
    ids = (torch.arange(16, device="cuda") % 3).reshape(8, 2)
    ids[1, 0], ids[2, 0] = -1, 2**32 + 1
    router = torch.rand(8, 2, device="cuda")
    route_map = torch.tensor([0, 2, 1, -1, -1, -1], dtype=torch.int32, device="cuda")
    output_map = torch.tensor([1, 0, 2, -1, -1, -1], dtype=torch.int32, device="cuda")
    external = torch.empty_like(source)

    def expected(live):
        return moe_reference(
            source[:live],
            payload,
            ids[:live],
            router[:live],
            activation_kind=public.activation.nonlinearity,
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
            experts=experts,
            a=source[:live],
            topk_ids=ids[:live],
            topk_weights=router[:live],
            output=target,
            route_expert_map=route_map,
            output_expert_map=output_map,
        )

    references = {live: expected(live) for live in (1, 3, 4, 8)}
    b12x.freeze_kernel_resolution("grouped atom serving contract")
    try:
        for live in (8, 1, 4, 3):
            for target in (None, external):
                external.fill_(float("nan"))
                bound = bind(live, target)
                assert len(bound._backend_binding.calls) == 8
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
        payload.rates.copy_(payload.rates.flip(1))
        payload.offsets.copy_(payload.offsets.flip(1))
        payload.trellis.up_suh.mul_(0.7)
        payload.trellis.down_svh.mul_(0.75)
        ids.fill_(0)
        route_map[0] = 2
        reference = expected(8)
        owners = (
            source,
            ids,
            router,
            payload.w13,
            payload.rates,
            payload.offsets,
            route_map,
            output_map,
            bound.output,
            *scratch.values(),
        )
        addresses = tuple(t.data_ptr() for t in owners)
        allocated = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocated
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
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
