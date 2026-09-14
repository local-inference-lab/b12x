"""Deferred B300 qualification through the canonical b12x MoE API."""

import pytest
import torch
import torch.nn.functional as F

import b12x
from b12x.moe import fused_moe
from b12x._lib.intrinsics import swizzle_block_scale
from tests.architecture.test_sm103 import make_experts
from b12x.moe._shared.kernels.materialized_nvfp4_reference import reference


def require_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip(
            "physical SM103/B300 required; offline compilation is not runtime qualification"
        )
    return torch.device("cuda", torch.cuda.current_device())


def case(
    device,
    *,
    hidden=256,
    intermediate=256,
    experts=4,
    top_k=2,
    capacity=8,
    w13_layout="w13",
    mode="a4",
):
    prepared = make_experts(
        device=device, k=hidden, n=intermediate, e=experts, w13_layout=w13_layout, mode=mode
    )
    raw = prepared._impl
    for w in (raw.w1_fp4, raw.w2_fp4):
        w.random_(0, 256)
    for sf in (raw.w1_blockscale, raw.w2_blockscale):
        conventional = (torch.rand(sf.shape, device=device) * 0.1 + 0.01).to(
            torch.float8_e4m3fn
        )
        sf.copy_(swizzle_block_scale(conventional))
    raw.a1_gscale.copy_(torch.linspace(0.5, 1.5, experts, device=device))
    raw.a2_gscale.copy_(torch.linspace(1.5, 0.5, experts, device=device))
    # Prepared alpha is weight_global / activation_reciprocal_global.
    raw.w1_alphas.copy_(1 / raw.a1_gscale)
    raw.w2_alphas.copy_(1 / raw.a2_gscale)
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=top_k),
    )
    fused_moe.prewarm(plan)
    scratch = torch.empty(plan.scratch.nbytes, dtype=torch.uint8, device=device)
    return prepared, plan, scratch


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mode,w13_layout", [("a4", "w13"), ("a4", "w31"), ("auto", "w13")])
def test_native_moe_correctness_and_graph_capacity_reuse(id_dtype, mode, w13_layout):
    device = require_sm103()
    prepared, plan, scratch = case(device, w13_layout=w13_layout, mode=mode)
    initial_ptrs = (
        scratch.data_ptr(),
        prepared._impl.w1_fp4.data_ptr(),
        prepared._impl.w2_fp4.data_ptr(),
    )
    b12x.freeze_kernel_resolution("SM103 qualification")
    try:
        for m in (1, 4, 8, 2):
            a = torch.randn((m, 256), device=device, dtype=torch.bfloat16) * 0.1
            ids = torch.arange(m * 2, device=device, dtype=id_dtype).reshape(m, 2) % 4
            ids[0, 0] = -1
            if m == 4:
                ids[0, 0] = 2**32 + 1 if id_dtype == torch.int64 else 4
            weights = torch.rand((m, 2), device=device)
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                experts=prepared,
                a=a,
                topk_ids=ids,
                topk_weights=weights,
            )
            expected = reference(a, prepared._impl, ids, weights)
            actual = fused_moe.run(binding=binding)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            assert (
                F.cosine_similarity(
                    actual.float().flatten(), expected.float().flatten(), dim=0
                )
                > 0.999
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            for _ in range(3):
                a.mul_(0.75)
                ids.copy_((ids + 1) % 4)
                expected = reference(a, prepared._impl, ids, weights)
                actual.fill_(float("nan"))
                before = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            assert initial_ptrs == (
                scratch.data_ptr(),
                prepared._impl.w1_fp4.data_ptr(),
                prepared._impl.w2_fp4.data_ptr(),
            )
    finally:
        b12x.unfreeze_kernel_resolution()


def test_native_moe_prefill_crosses_route_grid_boundary_with_frozen_plan():
    """An eight-expert prefill reuses decode callables past 65,535 routes."""
    device = require_sm103()
    capacity, top_k, period = 8193, 8, 8
    prepared, plan, scratch = case(
        device, experts=8, top_k=top_k, capacity=capacity
    )
    rows = torch.arange(capacity, device=device) % period
    source = torch.randn((period, 256), device=device, dtype=torch.bfloat16) * 0.1
    source_ids = torch.arange(period * top_k, device=device).reshape(period, top_k)
    source_ids = (source_ids + torch.arange(period, device=device)[:, None]) % 8
    source_weights = torch.rand((period, top_k), device=device)
    a, ids, weights = source[rows], source_ids[rows], source_weights[rows]
    scratch_ptr = scratch.data_ptr()
    b12x.freeze_kernel_resolution("SM103 prefill route boundary")
    try:
        for live in (1, 8192, capacity):
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                experts=prepared,
                a=a[:live],
                topk_ids=ids[:live],
                topk_weights=weights[:live],
            )
            actual = fused_moe.run(binding=binding)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            source.mul_(0.75)
            source_weights.mul_(0.5)
            a.copy_(source[rows])
            weights.copy_(source_weights[rows])
            expected = reference(
                source, prepared._impl, source_ids, source_weights
            )[rows[:live]]
            actual.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert scratch.data_ptr() == scratch_ptr
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            assert F.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            ) > 0.999
    finally:
        b12x.unfreeze_kernel_resolution()
