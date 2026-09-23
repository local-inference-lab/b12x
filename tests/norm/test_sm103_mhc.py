"""Serving qualification for planned mHC on SM103 and SM12x."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.norm import mhc
from b12x.preparation import FrozenMapping
from b12x.norm.mhc._tuning import TUNING
from ..conftest import require_sm103_or_sm12x
from tests._reference.mhc import _make_inputs, _mhc_pre_reference, _mhc_post_reference


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
    invocation = FrozenMapping({"operation": phase, "has_norm_weight": fuse_norm,
                                "expanded_residual": phase == "pre",
                                "norm_eps": 1e-6, "rms_eps": 1e-6, "hc_eps": 1e-6,
                                "sinkhorn_iters": 20})
    caps = mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden)
    declaration = mhc.plan(caps, invocation=invocation)
    from b12x.preparation.device import detect_device
    config = TUNING.configure(declaration.query, device=detect_device(device).identity,
                              search=False).default
    plan = mhc.plan(caps, invocation=invocation,
                    override=replace(config, projection_split_fp32=True))
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
    guard = kernel_resolution_guard("mHC current-mix capacity qualification")
    guard.__enter__()
    request.addfinalizer(lambda: guard.__exit__(None, None, None))

    def refuse(*args, **kwargs):
        pytest.fail("mHC replay attempted policy resolution")

    from b12x.preparation.tuning import TuningContract
    monkeypatch.setattr(TuningContract, "configure", refuse)
    monkeypatch.setenv("B12X_MHC_PREFILL_TF32_MMA", "0" if config.backend == "tf32_tma" else "1")
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
                if got.dtype == torch.bfloat16:
                    # FP32 reduction order may round across one BF16 boundary;
                    # 0.016 is below one ULP for magnitudes of four and above.
                    reference = expected.float()
                    ulp = torch.ldexp(torch.ones_like(reference), torch.frexp(reference).exponent - 8)
                    bound = torch.maximum(ulp, torch.full_like(reference, 0.016))
                    assert bool(((got.float() - reference).abs() <= bound).all())
                else:
                    torch.testing.assert_close(got, expected, rtol=2e-5, atol=4e-5)
        for buffer in (output, y, post, comb):
            assert bool(torch.isnan(buffer[live:]).all())
