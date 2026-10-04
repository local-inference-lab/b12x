"""Opt-in A4 prefill over W4A16 packed weights.

Gates the NVFP4-activation prefill pipeline (route pack 128 -> quantize ->
A4 FC1 -> A4 FC2 -> FP32-weighted top-k sum) against a float64 torch emulation
of the same quantization contract, for one activation plane (NVFP4) and two
planes (NVFP4 value + NVFP4 residual). Checks the W4A16 binding's selection
(A4 at or above the threshold, W4A16 below it, W4A16 for weights prepared
without calibrated activation scales) and graph replay across live token
counts with the same compiled callables.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests._reference.helpers import require_b12x

E, H, I, TOPK = 8, 512, 256, 2

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_table(device):
    mags = torch.tensor(_E2M1, device=device)
    return torch.cat((mags, -mags))


def _dequant_weight(codes_u8, scale_e4m3, gscale):
    table = _e2m1_table(codes_u8.device)
    lo = table[(codes_u8 & 0xF).long()]
    hi = table[(codes_u8 >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).reshape(codes_u8.shape[0], -1).double()
    return w * scale_e4m3.double().repeat_interleave(16, dim=1) * float(gscale)


def _e2m1_round(v):
    grid = torch.tensor(_E2M1, device=v.device, dtype=v.dtype)
    mids = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=v.device, dtype=v.dtype)
    mag = v.abs()
    idx = torch.bucketize(mag, mids)
    tie = (idx < 7) & (mag == mids[idx.clamp(max=6)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.sign(v) * grid[idx]


def _nvfp4_qdq(x, gs):
    """quantize_block_fp4 semantics per 16 values along the last dim."""
    xb = x.float().reshape(-1, 16)
    gsb = gs.float().reshape(-1, 1)
    if gsb.shape[0] != 1:
        gsb = gsb.repeat_interleave(xb.shape[0] // gsb.shape[0], dim=0)
    amax = xb.abs().amax(dim=1, keepdim=True)
    sf = (amax * gsb / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    vs = sf / gsb
    q = torch.where(vs > 0, _e2m1_round(xb / torch.where(vs > 0, vs, torch.ones_like(vs))),
                    torch.zeros_like(xb))
    return (q * vs).reshape(x.shape)


def _qdq(x, gs, terms):
    q1 = _nvfp4_qdq(x, gs)
    if terms == 1:
        return q1
    return q1 + _nvfp4_qdq(x.float() - q1, gs)


def _make_case(seed):
    import numpy as np

    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.kernels.w4a16.host import unswizzle_expert_scales

    dev = torch.device("cuda")
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=gen)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=gen)

    def scales(rows, cols):
        base = rng.integers(0x28, 0x48, size=(E, rows, 1))
        raw = (base + rng.integers(0, 14, size=(E, rows, cols))).astype(np.uint8)
        return torch.from_numpy(raw).to(dev).view(torch.float8_e4m3fn)

    s13, s2 = scales(2 * I, H // 16), scales(H, I // 16)
    g13 = (torch.rand(E, device=dev, generator=gen) * 0.5 + 0.75) * 0.02
    g2 = (torch.rand(E, device=dev, generator=gen) * 0.5 + 0.75) * 0.02
    return dict(
        w13=w13, w2=w2, s13=s13, s2=s2, g13=g13, g2=g2,
        s13_log=unswizzle_expert_scales(s13, rows=2 * I, cols=H),
        s2_log=unswizzle_expert_scales(s2, rows=H, cols=I),
        plan=moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
        ),
    )


def _env(monkeypatch, threshold, terms):
    # Plan-time options: preparation may run lazily at the first bind, so they
    # stay set for the whole test.
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", str(threshold))
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_TERMS", str(terms))


def _prepare(case, a1g, a2g):
    from b12x.moe import fused_moe as moe

    return moe.prepare_weights(plan=case["plan"], weights=moe.PackedWeights(
        w13=case["w13"].clone(), w2=case["w2"].clone(),
        w13_block_scales=case["s13"].clone(), w2_block_scales=case["s2"].clone(),
        w13_global_scales=case["g13"], w2_global_scales=case["g2"],
        input_scale=a1g, intermediate_scale=a2g, immutable_input_scales=True))


def _plan(experts, tokens):
    from b12x.moe import fused_moe as moe

    return moe.plan_execution(
        experts=experts, capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=TOPK),
        invocation={"fast_math": True})


def _bind(xp, x, ids, wts, out, scratch):
    from b12x.moe import fused_moe as moe

    return moe.bind(xp, a=x, topk_ids=ids, topk_weights=wts, output=out, scratch=scratch,
                    input_scales_static=True)


def _emulate(case, x, ids, wts, a1g, a2g, terms):
    tokens = x.shape[0]
    flat = ids.flatten().long()
    xq = _qdq(x.float(), a1g, terms).double()
    y = torch.zeros(tokens * TOPK, H, device=x.device, dtype=torch.float64)
    for e in torch.unique(flat).tolist():
        r = (flat == e).nonzero().flatten()
        w13 = _dequant_weight(case["w13"][e], case["s13_log"][e], case["g13"][e])
        gate = (xq[r // TOPK] @ w13[I:].T).float().to(torch.bfloat16).float()
        up = (xq[r // TOPK] @ w13[:I].T).float().to(torch.bfloat16).float()
        act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
        act = _qdq(act, a2g[e].reshape(1), terms).double()
        w2 = _dequant_weight(case["w2"][e], case["s2_log"][e], case["g2"][e])
        y[r] = (act @ w2.T).float().to(torch.bfloat16).double()
    return (y.view(tokens, TOPK, H) * wts.double().view(tokens, TOPK, 1)).sum(dim=1)


def _inputs(tokens, seed):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(tokens, H, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    ids = torch.argsort(torch.rand(tokens, E, device=dev, generator=gen), dim=1)[:, :TOPK]
    ids = ids.to(torch.int32).contiguous()
    wts = torch.softmax(torch.randn(tokens, TOPK, device=dev, generator=gen), dim=-1).float()
    return x, ids, wts.contiguous()


def _scales(case):
    """Calibration-like global scales: 448 * 6 / amax with headroom. Odd factors
    keep the synthetic data off exact E2M1 midpoints."""
    dev = torch.device("cuda")
    x, ids, _ = _inputs(64, 999)
    flat = ids.flatten().long()
    amax = 0.0
    for e in torch.unique(flat).tolist():
        r = (flat == e).nonzero().flatten()
        w13 = _dequant_weight(case["w13"][e], case["s13_log"][e], case["g13"][e])
        gate = x[r // TOPK].double() @ w13[I:].T
        up = x[r // TOPK].double() @ w13[:I].T
        amax = max(amax, float((gate * torch.sigmoid(gate) * up).abs().max()))
    a1g = torch.full((E,), 448.0 * 6.0 / 2.75 * 1.0123457, device=dev)
    a2g = torch.full((E,), 448.0 * 6.0 / (1.5 * amax) * 0.9876543, device=dev)
    return a1g, a2g


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("terms", [1, 2])
def test_a4_prefill_matches_float64_emulation(terms, monkeypatch):
    require_b12x()
    _env(monkeypatch, 64, terms)
    case = _make_case(11)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    assert experts._impl.a4_prefill_scales
    tokens = 192
    xp = _plan(experts, tokens)
    x, ids, wts = _inputs(tokens, 5)
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=x.device) for s in xp.scratch_specs())
    out = torch.empty_like(x)
    binding = _bind(xp, x, ids, wts, out, scratch)
    launches = getattr(binding, "_impl", binding).a4_prefill_launches
    assert launches is not None and launches.terms == terms
    from b12x.moe import fused_moe as moe

    moe.run(binding=binding)
    torch.cuda.synchronize()
    ref = _emulate(case, x, ids, wts, a1g, a2g, terms)
    assert torch.isfinite(out.float()).all() and out.float().abs().max() > 0
    rel = ((out.double() - ref).norm() / ref.norm()).item()
    # BF16 output rounding plus the BF16 top-k sum bound the difference.
    assert rel < 4e-3, rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a4_prefill_threshold_and_scale_gating(monkeypatch):
    require_b12x()
    _env(monkeypatch, 128, 1)
    case = _make_case(12)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    xp = _plan(experts, 256)
    dev = torch.device("cuda")
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs())
    for tokens, expect_a4 in ((64, False), (127, False), (128, True), (256, True)):
        x, ids, wts = _inputs(tokens, tokens)
        binding = _bind(xp, x, ids, wts, torch.empty_like(x), scratch)
        assert (getattr(binding, "_impl", binding).a4_prefill_launches is not None) == expect_a4, tokens
    # Without calibrated scales (or with invalid ones) the weights stay W4A16 only.
    bad = a1g.clone()
    bad[3] = float("nan")
    ones = torch.ones_like(a1g)
    for scales in ((None, None), (bad, a2g), (ones, ones)):
        plain = _prepare(case, scales[0], scales[1])
        assert not plain._impl.a4_prefill_scales
        xp_plain = _plan(plain, 256)
        x, ids, wts = _inputs(256, 3)
        scratch_plain = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev)
                              for s in xp_plain.scratch_specs())
        binding = _bind(xp_plain, x, ids, wts, torch.empty_like(x), scratch_plain)
        assert getattr(binding, "_impl", binding).a4_prefill_launches is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("terms", [1, 2])
def test_a4_prefill_graph_replay_reuses_launches(terms, monkeypatch):
    require_b12x()
    from b12x.moe import fused_moe as moe

    _env(monkeypatch, 64, terms)
    case = _make_case(13)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    xp = _plan(experts, 320)
    dev = torch.device("cuda")
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs())
    seen = set()
    for tokens in (96, 320, 200):
        x, ids, wts = _inputs(tokens, 100 + tokens)
        eager_out = torch.empty_like(x)
        binding = _bind(xp, x, ids, wts, eager_out, scratch)
        launches = getattr(binding, "_impl", binding).a4_prefill_launches
        assert launches is not None
        seen.add((id(launches.quant), id(launches.fc1), id(launches.fc2)))
        moe.run(binding=binding)
        torch.cuda.synchronize()
        expected = eager_out.clone()
        eager_out.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.run(binding=binding)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager_out, expected), tokens
        graph.reset()
    # One compiled pipeline serves every live token count of the capacity.
    assert len(seen) == 1
