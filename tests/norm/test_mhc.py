from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.norm import mhc

B12XMHCScratchCaps = mhc.Caps
b12x_mhc_post = mhc.run_post
b12x_mhc_post_pre = mhc.run_post_pre
plan_mhc_scratch = mhc.plan

from ..conftest import require_b12x as require_sm120


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
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 4
    ).to(torch.bfloat16)
    fn = (
        torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 64
    )
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return (
        residual.contiguous(),
        x.contiguous(),
        fn.contiguous(),
        scale.contiguous(),
        bias.contiguous(),
    )


def _make_mhc_binding(
    *,
    tokens: int,
    hidden_size: int,
    device: torch.device,
    split_k: int = 64,
    expected_m: int | None = None,
):
    max_tokens = max(tokens, expected_m or tokens)
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        )
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


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_b12x_mhc_fused_post_pre_match_reference(tokens: int) -> None:
    device = require_sm120()
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

    assert (
        residual_cur.untyped_storage().data_ptr()
        == binding.out.untyped_storage().data_ptr()
    )
    assert (
        post.untyped_storage().data_ptr()
        == binding.post_buffer.untyped_storage().data_ptr()
    )
    assert (
        comb.untyped_storage().data_ptr()
        == binding.comb_buffer.untyped_storage().data_ptr()
    )
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
    device = require_sm120()
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
    rms_scale = torch.rsqrt(y_raw_ref_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
    y_ref = (
        y_raw_ref_fp32.to(torch.bfloat16).float() * rms_scale * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)


def test_b12x_mhc_fused_post_pre_graph_capture() -> None:
    device = require_sm120()
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


def _lagged_reference(residual, fn, scale, bias, incoming, weight):
    flat = residual.flatten(1).float()
    projection = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-20
    )
    predicted = torch.sigmoid(projection[:, :4] * scale[0] + bias[:4]) + 1e-6
    _, post, comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    collapsed = (incoming.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    y = collapsed.float()
    if weight is not None:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * weight.float()
    return post, comb, y.bfloat16(), predicted


@pytest.mark.parametrize("hidden_size", [4096, 5120])
def test_mhc_lagged_multisublayer_propagation(hidden_size):
    device = require_sm120()
    tokens = 3
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens, hidden_size=hidden_size, seed=92150, device=device
    )
    weight = torch.linspace(0.5, 1.5, hidden_size, device=device).bfloat16()
    incoming = torch.zeros((tokens, 4), device=device)
    incoming[:, 0] = 1
    prev_post = prev_comb = None
    for layer in range(3):
        predicted = torch.full_like(incoming, float("nan"))
        if layer == 0:
            actual = mhc.run_pre(
                residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
                rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                norm_weight=weight, norm_eps=1e-20,
            )
            current = residual
        else:
            current = _mhc_post_reference(x, residual, prev_post, prev_comb)
            actual = mhc.run_post_pre(
                x, residual, prev_post, prev_comb, fn, scale, bias,
                pre_mix=incoming, pre_out=predicted, rms_eps=1e-20,
                hc_eps=1e-6, sinkhorn_iters=20, norm_weight=weight, norm_eps=1e-20,
            )
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
        expected = _lagged_reference(actual[0], fn, scale, bias, incoming, weight)
        for got, want in zip((*actual[1:], predicted), expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008 if got.dtype == torch.bfloat16 else 4e-5)
        residual, prev_post, prev_comb, x = actual
        incoming = predicted
        # Distinct sublayer projections prevent accidentally carrying a stale mix.
        fn = -fn
        bias = bias.roll(4)


def test_mhc_lagged_rounded_variance_and_ownership():
    device = require_sm120()
    hidden_size = 5120
    residual, _, fn, scale, bias = _make_inputs(
        tokens=1, hidden_size=hidden_size, seed=92151, device=device
    )
    residual[:, 0] = 1
    residual[:, 1] = 1
    residual[:, 1, 4096:] = 1.0078125
    incoming = torch.tensor([[1.0, -0.999, 0.0, 0.0]], device=device)
    predicted = torch.empty_like(incoming)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    kwargs = dict(rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    actual = mhc.run_pre(
        residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
        norm_weight=weight, norm_eps=1e-20, **kwargs,
    )
    expected = _lagged_reference(residual, fn, scale, bias, incoming, weight)
    torch.testing.assert_close(actual[-1], expected[2], rtol=0, atol=0.001)
    with pytest.raises(ValueError, match="supplied together"):
        mhc.run_pre(residual, fn, scale, bias, pre_mix=incoming, **kwargs)
    with pytest.raises(ValueError, match="must not alias"):
        mhc.run_pre(residual, fn, scale, bias, pre_mix=incoming, pre_out=incoming, **kwargs)
    with pytest.raises(ValueError, match="must not alias"):
        mhc.run_pre(
            residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
            post_out=predicted, **kwargs,
        )
    plan = mhc.plan(mhc.Caps(device=device, max_tokens=1, hidden_size=hidden_size))
    scratch = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in plan.shapes_and_dtypes())
    binding = mhc.bind(plan, scratch=scratch, pre_out=predicted)
    with pytest.raises(ValueError, match="binding owns scratch and output buffers"):
        mhc.run_pre(
            residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
            binding=binding, **kwargs,
        )


@pytest.mark.parametrize(
    ("phase", "capacity", "fuse_norm"),
    [("pre", 17, False), ("pre", 17, True), ("pre", 389, True),
     ("post_pre", 17, True), ("post_pre", 389, True)],
)
def test_mhc_lagged_frozen_multilive_graph(phase, capacity, fuse_norm, request):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    device = require_sm120()
    hidden_size = 5120
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden_size, seed=92152, device=device
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device) if fuse_norm else None
    incoming = torch.zeros((capacity, 4), device=device)
    incoming[:, 0] = 1
    predicted = torch.empty_like(incoming)
    plan = mhc.plan(mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden_size))
    scratch = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in plan.shapes_and_dtypes())
    binding = mhc.bind(
        plan, scratch=scratch, pre_out=predicted,
        y=torch.empty_like(x), out=torch.empty_like(residual),
        post=torch.empty_like(prev_post), comb=torch.empty_like(prev_comb),
    )

    def run(live):
        if phase == "pre":
            return mhc.run_pre(
                residual[:live], fn, scale, bias, binding=binding,
                pre_mix=incoming[:live], norm_weight=weight, norm_eps=1e-20,
                rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
            )
        return mhc.run_post_pre(
            x[:live], residual[:live], prev_post[:live], prev_comb[:live],
            fn, scale, bias, binding=binding, pre_mix=incoming[:live],
            norm_weight=weight, norm_eps=1e-20, rms_eps=1e-20,
            hc_eps=1e-6, sinkhorn_iters=20,
        )

    run(capacity)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("V4.1 lagged mHC fixed-capacity multi-live capture")
    for live in (3, capacity - 1, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(live)
        pointers = tuple(t.data_ptr() for t in (*actual, predicted))
        # Replay must read the incoming mix anew, not the warmup one-hot value.
        incoming.copy_(incoming.roll(1, dims=1))
        for output in (*actual, predicted):
            output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert tuple(t.data_ptr() for t in (*actual, predicted)) == pointers
        current = (
            residual[:live] if phase == "pre" else
            _mhc_post_reference(x[:live], residual[:live], prev_post[:live], prev_comb[:live])
        )
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
        expected = _lagged_reference(actual[0], fn, scale, bias, incoming[:live], weight)
        for got, want in zip((*actual[1:], predicted[:live]), expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008 if got.dtype == torch.bfloat16 else 4e-5)
        assert bool(torch.isnan(predicted[live:]).all())


def test_lagged_prefill_fp32_projection_precision():
    """Tensor-core projection retains FP32 mixing coefficients at long K."""
    device = require_sm120()
    hidden, rows = 5120, 513
    residual, _, fn, scale, bias = _make_inputs(
        tokens=rows, hidden_size=hidden, seed=415120, device=device
    )
    incoming = torch.full((rows, 4), 0.25, device=device)
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    plan = mhc.plan(mhc.Caps(device=device, max_tokens=4096, hidden_size=hidden))
    assert plan.config.backend == "tf32_tma"
    scratch = tuple(torch.empty(shape, dtype=dtype, device=device)
                    for shape, dtype in plan.shapes_and_dtypes())
    predicted = torch.empty_like(incoming)
    binding = mhc.bind(
        plan, scratch=scratch, tokens=rows, pre_out=predicted,
        y=torch.empty((rows, hidden), dtype=torch.bfloat16, device=device),
        out=torch.empty_like(residual), post=torch.empty_like(incoming),
        comb=torch.empty((rows, 4, 4), device=device),
    )
    _, post, _, _ = mhc.run_pre(
        residual, fn, scale, bias, binding=binding,
        pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
        rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
    )
    flat = residual.flatten(1).double()
    mixes = (flat @ fn.double().T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1e-20
    )
    expected = 2 * torch.sigmoid(mixes[:, 4:8] * scale.double()[1] + bias.double()[4:8])
    torch.testing.assert_close(post.double(), expected, rtol=1e-6, atol=1e-6)


def test_lagged_prefill_scalar_parity_graph(request):
    """Projection changes preserve scalar BF16 rounding and fixed-capacity replay.

    The scalar finalizer defines BF16 normalization rounding. Independent Torch
    norm reductions can straddle a BF16 midpoint; strict scalar parity prevents
    a projection optimization from changing that rounding contract.
    """
    from dataclasses import replace
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    device = require_sm120()
    hidden, capacity = 5120, 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden, seed=92152, device=device
    )
    incoming = torch.zeros((capacity, 4), device=device)
    incoming[:, 0] = 1
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    plan = mhc.plan(mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden))
    native = replace(plan, config=replace(plan.config, backend="native"))
    bindings = []
    for selected in (plan, native):
        scratch = tuple(torch.empty(shape, dtype=dtype, device=device)
                        for shape, dtype in selected.shapes_and_dtypes())
        bindings.append(mhc.bind(
            selected, scratch=scratch, out=torch.empty_like(residual),
            y=torch.empty_like(x), pre_out=torch.empty_like(incoming),
            post=torch.empty_like(incoming),
            comb=torch.empty((capacity, 4, 4), device=device),
        ))

    def run(binding, live):
        return mhc.run_pre(
            residual[:live], fn, scale, bias, binding=binding,
            pre_mix=incoming[:live], norm_weight=weight, norm_eps=1e-20,
            rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
        )

    for binding in bindings:
        run(binding, capacity)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("lagged prefill scalar parity")
    for live in (3, capacity - 1, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(bindings[0], live)
        incoming.copy_(incoming.roll(1, dims=1))
        for output in (*actual, bindings[0].pre_out):
            output.fill_(float("nan"))
        graph.replay()
        expected = run(bindings[1], live)
        torch.cuda.synchronize(device)
        for got, want in zip((*actual, bindings[0].pre_out[:live]),
                             (*expected, bindings[1].pre_out[:live]), strict=True):
            if got.dtype == torch.bfloat16:
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            else:
                torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)
        assert bool(torch.isnan(bindings[0].pre_out[live:]).all())


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
def test_standalone_collapse_fp32_accumulation_and_frozen_replay(hidden, request):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    device = require_sm120()
    capacity = 9
    state = torch.empty(capacity, 4, hidden, dtype=torch.bfloat16, device=device)
    # BF16 intermediate accumulation would lose the small positive streams.
    streams = torch.tensor([256.0, 1.0, -256.0, 0.5], device=device)
    state.copy_(streams[None, :, None].expand_as(state))
    mix = torch.tensor([1.0, 0.75, 1.0, 0.25], device=device).repeat(capacity, 1)
    weighted = torch.empty(capacity, hidden, dtype=torch.bfloat16, device=device)
    mean = torch.empty_like(weighted)

    def launch(rows):
        return (
            mhc.run_collapse(state[:rows], mix[:rows], out=weighted[:rows]),
            mhc.run_collapse(state[:rows], None, out=mean[:rows]),
        )

    launch(capacity)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("standalone mHC collapse across live rows")
    for rows in (0, 1, 7, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = launch(rows)
        pointers = tuple(t.data_ptr() for t in outputs)
        state[:, 1].add_(0.5)
        mix[:, 3].add_(0.125)
        weighted.fill_(float("nan"))
        mean.fill_(float("nan"))
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        assert tuple(t.data_ptr() for t in outputs) == pointers
        expected = (state[:rows].float() * mix[:rows, :, None]).sum(1).bfloat16()
        expected_mean = state[:rows].float().mean(1).bfloat16()
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], expected_mean, rtol=0, atol=0)
        assert bool(torch.isnan(weighted[rows:]).all())
        assert bool(torch.isnan(mean[rows:]).all())
