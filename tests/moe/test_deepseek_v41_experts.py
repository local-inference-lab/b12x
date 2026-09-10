"""Published V4.1 expert boundaries, including inactive local routes and replay."""

import pytest
import torch

from b12x.moe import fused_moe
from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)


def _mxfp8(x):
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1.0e-4) / 448.0)))
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def _decode(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return lut[codes] * scale


def _gemm_k32(activations, weights):
    """Published kernel.py:534-554 uses FP32 accumulation of K32 partials.

    MX scales are exact powers of two, so decoding each K32 operand group
    commutes with its contraction; summing all K at once does not preserve
    the reference's intermediate FP32 rounding.
    """
    result = torch.zeros(
        (activations.shape[0], weights.shape[0]),
        dtype=torch.float32,
        device=activations.device,
    )
    for k in range(0, activations.shape[1], 32):
        result.add_(activations[:, k:k + 32] @ weights[:, k:k + 32].T)
    return result


def _oracle(x, ids, weights, checkpoint, *, round_fc1=True, weight_before_fc2=True):
    w13, sf13, w2, sf2 = checkpoint
    result = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    x8 = _mxfp8(x)
    for expert in range(w13.shape[0]):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        gate, up = _gemm_k32(x8[token], _decode(w13[expert], sf13[expert])).chunk(2, dim=-1)
        if round_fc1:
            gate, up = gate.bfloat16().float(), up.bfloat16().float()
        mid = torch.nn.functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)
        route = weights[token, slot, None]
        if weight_before_fc2:
            mid = mid * route
        down = _gemm_k32(_mxfp8(mid.bfloat16()), _decode(w2[expert], sf2[expert])).bfloat16().float()
        if not weight_before_fc2:
            down = down * route
        result.index_add_(0, token, down)
    return result


def _setup(experts, hidden, intermediate, topk, capacity):
    generator = torch.Generator(device="cuda").manual_seed(41083)
    w13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device="cuda", generator=generator)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda", generator=generator)
    sf13 = torch.full((experts, 2 * intermediate, hidden // 32), 122, dtype=torch.uint8, device="cuda")
    sf2 = torch.full((experts, hidden, intermediate // 32), 122, dtype=torch.uint8, device="cuda")
    checkpoint = tuple(t.clone() for t in (w13, sf13, w2, sf2))
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, numerical_recipe="deepseek_v41"),
        geometry=fused_moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate),
    )
    unit = torch.ones(experts, device="cuda")
    prepared = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=sf13, w2_block_scales=sf2,
        w13_global_scales=unit, w2_global_scales=unit,
    ))
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
    )
    fused_moe.prewarm(plan)
    scratch = {spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in plan.scratch_specs()}
    x = torch.randn((capacity, hidden), generator=generator, device="cuda").bfloat16()
    # Only three experts receive work, and -1 denotes routes owned by other ranks.
    ids = torch.arange(capacity * topk, device="cuda", dtype=torch.int32).reshape(capacity, topk) % min(experts, 3)
    ids[:, -1] = -1
    weights = torch.rand((capacity, topk), generator=generator, device="cuda") * .7 + .07
    output = torch.empty((capacity, hidden), device="cuda", dtype=torch.float32)
    return plan, prepared, scratch, checkpoint, x, ids, weights, output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepseek_v41_rounding_and_router_placement():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        plan, experts, scratch, checkpoint, x, ids, weights, output = _setup(4, 256, 128, 3, 7)
        expected = _oracle(x, ids, weights, checkpoint)
        unrounded = _oracle(x, ids, weights, checkpoint, round_fc1=False)
        postweighted = _oracle(x, ids, weights, checkpoint, weight_before_fc2=False)
        # The adversarial oracle must distinguish both historical semantics.
        assert (expected - unrounded).abs().max().item() > .01
        assert (expected - postweighted).abs().max().item() > .01
        binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=output, input_scales_static=True)
        fused_moe.run(binding=binding)
        torch.testing.assert_close(output, expected, rtol=0, atol=.002)
        assert (output - expected).abs().sum() < (output - unrounded).abs().sum()
        assert (output - expected).abs().sum() < (output - postweighted).abs().sum()
        bf16_output = torch.empty_like(x)
        bf16_binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=bf16_output, input_scales_static=True)
        fused_moe.run(binding=bf16_binding)
        torch.testing.assert_close(bf16_output, expected.bfloat16(), rtol=0, atol=.002)
        x.mul_(1.0e-6)
        fused_moe.run(binding=binding)
        tiny_expected = _oracle(x, ids, weights, checkpoint)
        torch.testing.assert_close(output, tiny_expected, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("local_experts,topk", [(48, 6), (16, 3)])
def test_deepseek_v41_local_experts_live_counts_and_graph(local_experts, topk):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        plan, experts, scratch, checkpoint, x, ids, weights, output = _setup(local_experts, 5120, 2304, topk, 65)
        freeze_kernel_resolution("V4.1 expert capacity reuse")
        try:
            for count in (1, 7, 65):
                binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x[:count], topk_ids=ids[:count], topk_weights=weights[:count], output=output[:count], input_scales_static=True)
                fused_moe.run(binding=binding)
                expected = _oracle(x[:count], ids[:count], weights[:count], checkpoint)
                torch.testing.assert_close(output[:count], expected, rtol=.01, atol=.08)
            binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=output, input_scales_static=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            # Replay transitions from three active experts to one, then none;
            # stale route scratch must never survive inactive local routes.
            ids.fill_(-1)
            ids[::2, 0] = local_experts - 1
            x.mul_(.5)
            weights.mul_(.8)
            graph.replay()
            torch.testing.assert_close(output, _oracle(x, ids, weights, checkpoint), rtol=.01, atol=.08)
            ids.fill_(-1)
            graph.replay()
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        finally:
            unfreeze_kernel_resolution()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
