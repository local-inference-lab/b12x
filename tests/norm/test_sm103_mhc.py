"""Serving qualification for planned mHC on SM103 and SM12x."""

from dataclasses import replace

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.norm import mhc
from b12x.policy import MHC, PolicyContext
from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery
from ..conftest import require_sm103_or_sm12x
from .test_mhc import _make_inputs, _mhc_pre_reference, _mhc_post_reference


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("capacity", [17, 389])
@pytest.mark.parametrize("phase", ["pre", "post_pre"])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_current_mix_reuses_capacity_and_mutable_graph_inputs(
    hidden,
    capacity,
    phase,
    fuse_norm,
    monkeypatch,
    request,
):
    device = require_sm103_or_sm12x()
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity,
        hidden_size=hidden,
        seed=80201,
        device=device,
    )
    prev_post = torch.full((capacity, 4), 0.25, device=device)
    prev_comb = torch.eye(4, device=device).expand(capacity, 4, 4).contiguous()
    weight = (
        torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
        if fuse_norm
        else None
    )
    context = PolicyContext.for_device(device)
    query = MhcQuery(
        dtype="bfloat16", max_tokens=capacity, hidden_size=hidden, split_k=hidden // 64
    )
    config = replace(
        MHC_POLICY.heuristic(query, context.device), projection_split_fp32=True
    )
    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden),
        policy=context.with_override(MHC, config),
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in plan.scratch_specs()
    )
    y = torch.empty_like(x)
    output = torch.empty_like(residual)
    post, comb = torch.empty_like(prev_post), torch.empty_like(prev_comb)
    binding = mhc.bind(plan, scratch=scratch, y=y, out=output, post=post, comb=comb)

    def run(live):
        kwargs = dict(
            binding=binding,
            norm_weight=weight,
            norm_eps=1e-6,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
        )
        if phase == "pre":
            return mhc.run_pre(residual[:live], fn, scale, bias, **kwargs)
        return mhc.run_post_pre(
            x[:live],
            residual[:live],
            prev_post[:live],
            prev_comb[:live],
            fn,
            scale,
            bias,
            **kwargs,
        )

    # One warmup count compiles the capacity specialization, including prefill
    # plans whose first invocation contains only one live row.
    run(1)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("mHC current-mix capacity qualification")

    def refuse(*args, **kwargs):
        pytest.fail("mHC replay attempted policy resolution")

    monkeypatch.setattr(PolicyContext, "resolve", refuse)
    from b12x.norm.mhc import _kernels, _policy

    for module in (_kernels, _policy):
        for name in (
            "_selected_post_pre_decode_split_n",
            "_selected_mhc_decode_finalize_threads",
            "_selected_post_pre_partials_per_cta",
        ):
            monkeypatch.setattr(module, name, refuse)
    # A serving plan retains the setup-time overrides after the environment changes.
    monkeypatch.setenv(
        "B12X_MHC_PREFILL_TF32_MMA", "0" if plan.schedule.tf32_enabled else "1"
    )
    monkeypatch.setenv("B12X_MHC_PARTIALS_PER_CTA", "19")
    for live in (0, 1, 3, 8, 16, capacity - 1, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(live)
        pointers = tuple(t.data_ptr() for t in (*actual, *scratch))
        residual.mul_(0.875)
        x.add_(0.03125)
        fn.mul_(0.9375)
        bias.add_(0.0625)
        scale.mul_(1.03125)
        prev_post.add_(0.015625)
        if weight is not None:
            weight.mul_(0.96875)
        for buffer in (*scratch, y, output, post, comb):
            if buffer.dtype.is_floating_point:
                buffer.fill_(float("nan"))
            else:
                buffer.fill_(255)
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert tuple(t.data_ptr() for t in (*actual, *scratch)) == pointers
        if live:
            current = (
                residual[:live]
                if phase == "pre"
                else _mhc_post_reference(
                    x[:live],
                    residual[:live],
                    prev_post[:live],
                    prev_comb[:live],
                )
            )
            torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
            raw, expected_post, expected_comb = _mhc_pre_reference(
                actual[0],
                fn,
                scale,
                bias,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                y_dtype=torch.float32,
            )
            expected_y = raw.bfloat16()
            if weight is not None:
                expected_y = (
                    expected_y.float()
                    * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
                    * weight.float()
                ).bfloat16()
            for got, expected in zip(
                actual[1:], (expected_post, expected_comb, expected_y), strict=True
            ):
                assert bool(torch.isfinite(got).all()) and bool(
                    torch.count_nonzero(got)
                )
                torch.testing.assert_close(
                    got,
                    expected,
                    rtol=2e-5,
                    atol=0.016 if got.dtype == torch.bfloat16 else 4e-5,
                )
        for buffer in (output, y, post, comb):
            assert bool(torch.isnan(buffer[live:]).all())
