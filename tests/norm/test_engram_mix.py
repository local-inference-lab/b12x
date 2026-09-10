from __future__ import annotations

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.norm import hyperconnection as hc

from ..conftest import require_b12x


def _reference(state, projected_kv, norm_weights, streams, eps, token_mask=None):
    hidden = state.shape[1] // streams
    residual = state.view(-1, streams, hidden).float()
    key = projected_kv[:, : streams * hidden].reshape(-1, streams, hidden).float()
    value = projected_kv[:, streams * hidden :].float()
    rstd = torch.rsqrt(residual.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (residual * norm_weights.view(streams, hidden) * key).sum(-1)
    dot = dot * rstd * hidden**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask[:, None], 0)
    return (residual + gate[..., None] * value[:, None]).flatten(1).bfloat16()


def test_engram_zero_dot_uses_positive_signed_sqrt_and_masks_images():
    device = require_b12x()
    streams, hidden, tokens = 4, 5120, 3
    plan = hc.plan(hc.Caps(device=device, max_tokens=tokens, hidden_size=hidden))
    state = torch.full((tokens, streams * hidden), -0.5, device=device, dtype=torch.bfloat16)
    projected = torch.zeros((tokens, (streams + 1) * hidden), device=device, dtype=torch.bfloat16)
    projected[:, streams * hidden :] = 1
    norm_weights = torch.ones(streams * hidden, device=device, dtype=torch.float32)
    # The odd byte offset detects an invalid two-byte alignment assumption for bool pointers.
    mask_owner = torch.tensor([False, True, False, True], device=device)
    mask = mask_owner[1:]
    output = torch.empty_like(state)
    actual = hc.run_engram_mix(state, projected, norm_weights, eps=1e-20, plan=plan, out=output, token_mask=mask)
    expected = _reference(state, projected, norm_weights, streams, 1e-20, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert torch.all(actual[0] > 0), "sign(0) would erase the signed-sqrt gate correction"
    torch.testing.assert_close(actual[1], state[1], rtol=0, atol=0)


def test_engram_mix_replays_mutated_inputs_with_frozen_live_counts():
    device = require_b12x()
    streams, hidden, capacity = 4, 5120, 7
    generator = torch.Generator(device=device).manual_seed(4109)
    state = torch.randn((capacity, streams * hidden), generator=generator, device=device, dtype=torch.bfloat16)
    projected = torch.randn((capacity, (streams + 1) * hidden), generator=generator, device=device, dtype=torch.bfloat16)
    weights = torch.randn(streams * hidden, generator=generator, device=device)
    mask = torch.ones(capacity, dtype=torch.bool, device=device)
    output = torch.empty_like(state)
    plan = hc.plan(hc.Caps(device=device, max_tokens=capacity, hidden_size=hidden))

    def launch(rows, with_mask):
        return hc.run_engram_mix(
            state[:rows], projected[:rows], weights, eps=1e-20, plan=plan,
            out=output[:rows], token_mask=mask[:rows] if with_mask else None,
        )

    launch(1, True)
    launch(1, False)
    torch.cuda.synchronize(device)
    freeze_kernel_resolution()
    try:
        for rows in (0, 1, capacity):
            output.fill_(123)
            actual = launch(rows, True)
            expected = _reference(state[:rows], projected[:rows], weights, streams, 1e-20, mask[:rows])
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(output[rows:], torch.full_like(output[rows:], 123), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch(capacity, True)
        address = captured.data_ptr()
        state.mul_(-0.5)
        projected.mul_(0.75)
        weights.neg_()
        mask[::2] = False
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize(device)
        assert captured.data_ptr() == address
        expected = _reference(state, projected, weights, streams, 1e-20, mask)
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(captured[~mask], state[~mask], rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize('hidden', [128, 512, 1280])
def test_ordinary_rmsnorm_does_not_add_one_to_affine_weight(hidden):
    device = require_b12x()
    capacity = 7
    plan = hc.plan(hc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, streams=1, lowrank=1))
    state = torch.randn((capacity, hidden), device=device, dtype=torch.bfloat16)
    weight = torch.linspace(-0.5, 1.5, hidden, device=device).bfloat16()
    normalized = torch.empty_like(state)
    bottleneck = torch.empty((capacity, 1), device=device, dtype=torch.bfloat16)
    block_input = torch.empty_like(state)
    for rows in (1, capacity):
        binding = hc.bind(plan, normalized=normalized, bottleneck=bottleneck, block_input=block_input, tokens=rows)
        actual = hc.run_grouped_rmsnorm(state[:rows], weight, eps=1e-20, binding=binding, zero_centered=False)
        values = state[:rows].float()
        expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-20) * weight.float()).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
