from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from b12x.norm.mhc._impl import (
    B12XMHCScratchCaps,
    b12x_mhc_post,
    b12x_mhc_post_pre,
    b12x_mhc_pre,
    plan_mhc_scratch,
)
from b12x.norm.mhc._policy import MHC_POLICY, MhcConfig, MhcQuery
from b12x.policy import MHC, PolicyContext, PolicyMode

from tests._reference.helpers import require_b12x


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    y_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = y.to(residual.dtype if y_dtype is None else y_dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 4
    ).to(torch.bfloat16)
    fn = torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(device) / 64
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return residual.contiguous(), x.contiguous(), fn.contiguous(), scale.contiguous(), bias.contiguous()


def _make_mhc_binding(
    *,
    tokens: int,
    hidden_size: int,
    device: torch.device,
    split_k: int = 64,
    expected_m: int | None = None,
    config: MhcConfig | None = None,
):
    max_tokens = max(tokens, expected_m or tokens)
    policy = None
    if config is not None:
        policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        ).with_override(MHC, config)
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        ),
        policy=policy,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    return plan.bind(
        scratch=scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
        out=torch.empty((tokens, 4, hidden_size), dtype=torch.bfloat16, device=device),
    )


def _planned_mhc_config(
    *,
    hidden_size: int,
    split_k: int,
    schedule: str,
) -> MhcConfig:
    anchor_tokens = {
        "native_decode": 1,
        "native_prefill_block_m": 96,
        "tf32_tma": 384,
    }[schedule]
    return MHC_POLICY.heuristic(
        MhcQuery(
            dtype="bfloat16",
            max_tokens=anchor_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        ),
        None,
    )


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_b12x_mhc_pre_broadcast_match_reference(tokens: int) -> None:
    device = require_b12x()
    hidden_size = 4096
    _, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_420 + tokens,
        device=device,
    )
    fn_broadcast = fn.view(24, 4, hidden_size).sum(dim=1).contiguous()
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_421 + tokens)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )

    residual, post, comb, y = b12x_mhc_pre(
        x,
        fn_broadcast,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        binding=binding,
    )
    torch.cuda.synchronize(device)

    residual_ref = x.unsqueeze(1).expand(-1, 4, -1)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale = torch.rsqrt(
        y_raw_ref.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    y_ref = (
        y_raw_ref.to(torch.bfloat16).float()
        * rms_scale
        * norm_weight.float()
    ).to(torch.bfloat16)
    assert residual.untyped_storage().data_ptr() == binding.out.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == binding.post_buffer.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == binding.comb_buffer.untyped_storage().data_ptr()
    assert y.untyped_storage().data_ptr() == binding.y.untyped_storage().data_ptr()
    torch.testing.assert_close(residual, residual_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)


def test_glm53_mhc_rms_eps_match_reference() -> None:
    device = require_b12x()
    tokens = 3
    hidden_size = 4096
    rms_eps = 1e-5
    hc_eps = 1e-6
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=53_001,
        device=device,
    )
    fn_broadcast = fn.view(24, 4, hidden_size).sum(dim=1).contiguous()
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(53_002)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )

    residual_cur, post, comb, y = b12x_mhc_pre(
        x,
        fn_broadcast,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=rms_eps,
        binding=_make_mhc_binding(
            tokens=tokens,
            hidden_size=hidden_size,
            device=device,
        ),
    )
    torch.cuda.synchronize(device)

    residual_ref = x.unsqueeze(1).expand(-1, 4, -1)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    norm_scale = torch.rsqrt(
        y_raw_ref.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    y_ref = (
        y_raw_ref.to(torch.bfloat16).float()
        * norm_scale
        * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)

    next_residual, next_post, next_comb, next_y = b12x_mhc_post_pre(
        x,
        residual_cur,
        post,
        comb,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=rms_eps,
        binding=_make_mhc_binding(
            tokens=tokens,
            hidden_size=hidden_size,
            device=device,
        ),
    )
    torch.cuda.synchronize(device)

    next_residual_ref = _mhc_post_reference(x, residual_cur, post, comb)
    next_y_raw_ref, next_post_ref, next_comb_ref = _mhc_pre_reference(
        next_residual_ref,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    next_norm_scale = torch.rsqrt(
        next_y_raw_ref.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    next_y_ref = (
        next_y_raw_ref.to(torch.bfloat16).float()
        * next_norm_scale
        * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(next_residual, next_residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(next_y, next_y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(next_post, next_post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(next_comb, next_comb_ref, rtol=2e-6, atol=4e-5)


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_b12x_mhc_post_match_reference(tokens: int) -> None:
    device = require_b12x()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_430 + tokens,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    out = torch.empty_like(residual)

    residual_cur = b12x_mhc_post(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        out=out,
    )
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    assert residual_cur.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_b12x_mhc_fused_post_pre_match_reference(tokens: int) -> None:
    device = require_b12x()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_450 + tokens,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    if tokens == 3:
        prev_post_arg = prev_post_arg.unsqueeze(-1).contiguous()

    residual_cur, post, comb, y = b12x_mhc_post_pre(
        x,
        residual,
        prev_post_arg,
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        binding=binding,
    )
    torch.cuda.synchronize(device)

    assert residual_cur.untyped_storage().data_ptr() == binding.out.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == binding.post_buffer.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == binding.comb_buffer.untyped_storage().data_ptr()
    assert y.untyped_storage().data_ptr() == binding.y.untyped_storage().data_ptr()

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=8e-3)
    scalar_atol = 2e-5 if tokens >= 8 else 1e-5
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=scalar_atol)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=scalar_atol)


@pytest.mark.parametrize("tokens", [1, 3])
def test_b12x_mhc_fused_post_pre_with_rmsnorm_match_reference(tokens: int) -> None:
    device = require_b12x()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_470 + tokens,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_471 + tokens)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    residual_cur, post, comb, y = b12x_mhc_post_pre(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        binding=binding,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    # The fused post_pre kernel (like vLLM's TileLang kernel) computes the
    # RMSNorm variance in fp32 from the collapsed activation -- not from the
    # bf16-rounded activation -- so reference the variance from fp32 y too
    # (matching vllm_y_max == fused_y_max in the benchmark). The activation
    # itself is still bf16 (it is stored bf16 before the norm multiply).
    y_raw_ref_fp32, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale = torch.rsqrt(
        y_raw_ref_fp32.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    y_ref = (
        y_raw_ref_fp32.to(torch.bfloat16).float() * rms_scale * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)


@pytest.mark.parametrize(
    ("hidden_size", "split_k", "schedule"),
    [
        (4096, 64, "native_decode"),
        (4096, 64, "native_prefill_block_m"),
        (4096, 64, "tf32_tma"),
        (7168, 112, "native_decode"),
        (7168, 112, "native_prefill_block_m"),
        (7168, 112, "tf32_tma"),
    ],
    ids=lambda value: str(value),
)
def test_b12x_mhc_planned_post_pre_expected_m_matches_reference(
    hidden_size: int,
    split_k: int,
    schedule: str,
) -> None:
    device = require_b12x()
    tokens = 33
    expected_m = 384
    config = _planned_mhc_config(
        hidden_size=hidden_size,
        split_k=split_k,
        schedule=schedule,
    )
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_469 + hidden_size,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        split_k=split_k,
        device=device,
        expected_m=expected_m,
        config=config,
    )
    assert binding.plan.config == config
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_470 + hidden_size)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    prev_post_arg = prev_post.contiguous()
    prev_comb_arg = prev_comb.contiguous()

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return b12x_mhc_post_pre(
            x,
            residual,
            prev_post_arg,
            prev_comb_arg,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            binding=binding,
            norm_weight=norm_weight,
            norm_eps=1e-6,
        )

    outputs = run()
    torch.cuda.synchronize(device)
    residual_cur, post, comb, y = outputs
    expected_ptrs = tuple(output.data_ptr() for output in outputs)
    assert expected_ptrs == (
        binding.out.data_ptr(),
        binding.post_buffer.data_ptr(),
        binding.comb_buffer.data_ptr(),
        binding.y.data_ptr(),
    )

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale = torch.rsqrt(y_raw_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
    y_ref = (
        y_raw_ref.to(torch.bfloat16).float() * rms_scale * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(post, post_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-4, atol=2e-4)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run()
    assert tuple(output.data_ptr() for output in graph_outputs) == expected_ptrs
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    for output in outputs:
        assert bool(torch.isfinite(output).all())
        assert int(torch.count_nonzero(output)) > 0
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(post, post_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-4, atol=2e-4)

    # Replay must consume live serving inputs at the captured addresses; an
    # unchanged-output replay can otherwise pass while accidentally baking a
    # warmup value into the graph.
    residual.mul_(-0.5).add_(0.03125)
    x.mul_(0.75).sub_(0.015625)
    residual_ref_live = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_ref_live, post_ref_live, comb_ref_live = _mhc_pre_reference(
        residual_ref_live,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale_live = torch.rsqrt(
        y_raw_ref_live.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    y_ref_live = (
        y_raw_ref_live.to(torch.bfloat16).float()
        * rms_scale_live
        * norm_weight.float()
    ).to(torch.bfloat16)
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    assert tuple(output.data_ptr() for output in outputs) == expected_ptrs
    torch.testing.assert_close(
        residual_cur, residual_ref_live, rtol=0.0, atol=2e-2
    )
    torch.testing.assert_close(y, y_ref_live, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(post, post_ref_live, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(comb, comb_ref_live, rtol=2e-4, atol=2e-4)


def test_b12x_mhc_pro_hidden_match_reference() -> None:
    device = require_b12x()
    tokens = 1
    hidden_size = 7168
    split_k = 112
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_472,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        split_k=split_k,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_473)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    residual_post = torch.empty_like(residual)
    residual_cur = b12x_mhc_post(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        out=residual_post,
    )
    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)

    residual_cur, post, comb, y = b12x_mhc_post_pre(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        binding=binding,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)

    y_raw_ref_fp32, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale = torch.rsqrt(
        y_raw_ref_fp32.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    y_ref = (
        y_raw_ref_fp32.to(torch.bfloat16).float() * rms_scale * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)

    residual_func, post_func, comb_func, y_func = b12x_mhc_post_pre(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        split_k=split_k,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(residual_func, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y_func, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post_func, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb_func, comb_ref, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize(
    ("hidden_size", "split_k", "decode_mode"),
    [
        (4096, 64, "split"),
        (4096, 64, "fused"),
        (4096, 64, "post"),
        (4096, 64, "pre"),
        (7168, 112, "fused"),
        (7168, 112, "post"),
        (7168, 112, "pre"),
    ],
    ids=[
        "h4096-split",
        "h4096-fused",
        "h4096-post",
        "h4096-pre",
        "h7168-fused",
        "h7168-post",
        "h7168-pre",
    ],
)
def test_b12x_mhc_decode_specialization_live_graph_oracle(
    hidden_size: int,
    split_k: int,
    decode_mode: str,
) -> None:
    device = require_b12x()
    decode_config = _planned_mhc_config(
        hidden_size=hidden_size,
        split_k=split_k,
        schedule="native_decode",
    )
    if decode_mode == "split":
        decode_config = replace(
            decode_config,
            decode_source_splits=4,
            decode_tile_n=6,
        )
    tokens = 1
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_480 + hidden_size,
        device=device,
    )
    norm_weight = torch.ones(
        (hidden_size,), dtype=torch.bfloat16, device=device
    )

    def assert_outputs_match(
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        residual_ref: torch.Tensor,
        y_ref: torch.Tensor,
        post_ref: torch.Tensor,
        comb_ref: torch.Tensor,
    ) -> None:
        residual_cur, post, comb, y = outputs
        for output in outputs:
            assert bool(torch.isfinite(output).all())
            assert int(torch.count_nonzero(output)) > 0
        torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
        # One BF16 ULP near unit magnitude is 0.0078125.
        torch.testing.assert_close(y, y_ref, rtol=0.0, atol=8e-3)
        torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)

    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    y_ref = (
        y_raw_ref.to(torch.bfloat16).float()
        * torch.rsqrt(y_raw_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)
    if decode_mode == "post":
        out = torch.empty_like(residual)

        def run_post() -> torch.Tensor:
            return b12x_mhc_post(
                x, residual, prev_post, prev_comb, out=out
            )

        result = run_post()
        torch.cuda.synchronize(device)
        output_ptr = result.data_ptr()
        assert output_ptr == out.data_ptr()
        torch.testing.assert_close(result, residual_ref, rtol=0.0, atol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_result = run_post()
        assert graph_result.data_ptr() == output_ptr
        x.mul_(0.75).sub_(0.0078125)
        residual.mul_(-0.375).add_(0.015625)
        residual_ref_live = _mhc_post_reference(
            x, residual, prev_post, prev_comb
        )
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        assert result.data_ptr() == output_ptr
        torch.testing.assert_close(
            result, residual_ref_live, rtol=0.0, atol=2e-2
        )
        return

    if decode_mode == "pre":
        residual_ref = x.unsqueeze(1).expand(-1, 4, -1)
        pre_fn = fn.view(24, 4, hidden_size).sum(dim=1).contiguous()
        y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
            residual_ref,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            y_dtype=torch.float32,
        )
        y_ref = (
            y_raw_ref.to(torch.bfloat16).float()
            * torch.rsqrt(y_raw_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
            * norm_weight.float()
        ).to(torch.bfloat16)
        binding = _make_mhc_binding(
            tokens=tokens,
            hidden_size=hidden_size,
            split_k=split_k,
            device=device,
            config=decode_config,
        )
        assert binding.plan.config == decode_config

        def run_pre() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            return b12x_mhc_pre(
                x,
                pre_fn,
                scale,
                bias,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                binding=binding,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )

        run = run_pre
    elif decode_mode in {"split", "fused"}:
        binding = _make_mhc_binding(
            tokens=tokens,
            hidden_size=hidden_size,
            split_k=split_k,
            device=device,
            config=decode_config,
        )
        assert binding.plan.config == decode_config

        def run_post_pre() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            return b12x_mhc_post_pre(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                binding=binding,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )

        run = run_post_pre
    else:
        raise AssertionError(f"unknown decode mode {decode_mode}")

    outputs = run()
    torch.cuda.synchronize(device)
    assert_outputs_match(
        outputs,
        residual_ref=residual_ref,
        y_ref=y_ref,
        post_ref=post_ref,
        comb_ref=comb_ref,
    )
    output_ptrs = tuple(output.data_ptr() for output in outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run()
    assert tuple(output.data_ptr() for output in graph_outputs) == output_ptrs

    x.mul_(-0.5).add_(0.01171875)
    if decode_mode == "pre":
        residual_ref_live = x.unsqueeze(1).expand(-1, 4, -1)
    else:
        residual.mul_(0.625).sub_(0.01953125)
        residual_ref_live = _mhc_post_reference(
            x, residual, prev_post, prev_comb
        )
    y_raw_ref_live, post_ref_live, comb_ref_live = _mhc_pre_reference(
        residual_ref_live,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    y_ref_live = (
        y_raw_ref_live.to(torch.bfloat16).float()
        * torch.rsqrt(
            y_raw_ref_live.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        * norm_weight.float()
    ).to(torch.bfloat16)
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    assert tuple(output.data_ptr() for output in outputs) == output_ptrs
    assert_outputs_match(
        outputs,
        residual_ref=residual_ref_live,
        y_ref=y_ref_live,
        post_ref=post_ref_live,
        comb_ref=comb_ref_live,
    )


def test_b12x_mhc_fused_post_pre_graph_capture() -> None:
    device = require_b12x()
    tokens = 2
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_460,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    prev_comb_arg = prev_comb.contiguous()
    # CUDA graph capture requires caller-owned scratch (the partials buffer).
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    residual_cur = binding.out
    y = binding.y
    post = binding.post_buffer
    comb = binding.comb_buffer

    def run() -> None:
        b12x_mhc_post_pre(
            x,
            residual,
            prev_post_arg,
            prev_comb_arg,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            binding=binding,
        )

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=4e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)
