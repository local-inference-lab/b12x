"""Compare direct FP32 projection with split-TF32 lagged MHC prefill.

This probe explicitly invokes the candidate and does not change serving
dispatch. Both paths retain FP32 weights and use caller-owned scratch.
"""

import json
import statistics
import os
from dataclasses import replace

import torch

from b12x.preparation import PreparationSession, PreparedCall, FrozenMapping
from b12x.norm import mhc
from b12x.norm.mhc._impl import _b12x_mhc_pre_impl
from b12x.norm.mhc._tuning import TUNING


def main():
    torch.manual_seed(415120)
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = 5120
    capacity = 4096
    caps = mhc.Caps(device=device, hidden_size=hidden, max_tokens=capacity)
    invocation = FrozenMapping({
        "operation": "pre", "output_mode": "provided", "expanded_residual": True,
        "lagged_mix": True, "has_norm_weight": True, "norm_weight_dtype": "bfloat16",
        "rms_eps": 1e-20, "hc_eps": 1e-6, "sinkhorn_iters": 20, "norm_eps": 1e-20,
    })
    declaration = mhc.plan(caps, invocation=invocation)
    config = replace(TUNING.configure(declaration.query, device=None).default,
                     backend="tf32_tma", lagged_prepare=False,
                     projection_k_splits=int(os.getenv("MHC_PROBE_K_SPLITS", "1")))
    plans = [mhc.plan(caps, invocation=invocation, override=selected)
             for selected in (replace(config, backend="native"), config)]
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    fn = torch.randn((24, 4 * hidden), device=device) / 64
    scale = torch.randn((3,), device=device) / 3
    bias = torch.randn((24,), device=device) / 5
    weight = torch.randn((hidden,), device=device).bfloat16()
    try:
        for rows in (64, 512, 1024, 4096):
            residual = (torch.randn((rows, 4, hidden), device=device) / 3).bfloat16()
            incoming = torch.sigmoid(torch.randn((rows, 4), device=device))
            (spec,) = plans[0].scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty_like(residual)
            y = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
            post = torch.empty((rows, 4), device=device)
            comb = torch.empty((rows, 4, 4), device=device)
            pre = torch.empty((rows, 4), device=device)
            options = dict(rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                           norm_weight=weight, norm_eps=1e-20, pre_mix=incoming)
            if plans[0].prepared is None:
                def prime(state):
                    (trial_spec,) = state.scratch_specs()
                    trial_scratch = torch.empty(trial_spec.shape, dtype=trial_spec.dtype, device=device)
                    trial = state.bind(
                        scratch=trial_scratch, tokens=rows, out=torch.empty_like(out),
                        y=torch.empty_like(y), post=torch.empty_like(post), comb=torch.empty_like(comb),
                        pre_out=torch.empty_like(pre),
                    )
                    return PreparedCall(run=lambda: _b12x_mhc_pre_impl(
                        residual, fn, scale, bias, **options, binding=trial, _state=state,
                    ), owners=(trial_scratch, trial))
                session.prepare(tuple(plan.request(name=f"projection-{i}", prepare_call=prime)
                                      for i, plan in enumerate(plans)))
                session.freeze()
            bindings = [mhc.bind(plan, scratch=scratch, tokens=rows, out=out, y=y,
                                 post=post, comb=comb, pre_out=pre) for plan in plans]
            binding = bindings[0]

            def baseline():
                mhc.run_pre(residual, fn, scale, bias, **options, binding=bindings[0])

            def candidate():
                mhc.run_pre(residual, fn, scale, bias, **options, binding=bindings[1])

            baseline()
            expected = [value.clone() for value in (out, y, post, comb, pre)]
            scalar_projection = binding.partials[:, : hidden // 128].sum(1)[:, 1:].clone()
            scratch.fill_(0xA5)
            candidate()
            tc_projection = binding.partials[:, 0, 1:].clone()
            for part in range(1, config.projection_k_splits):
                tc_projection += binding.partials[:, part + 1, 1:]
            reference_projection = residual.flatten(1).double() @ fn.double().T
            normalized = reference_projection * torch.rsqrt(
                residual.double().flatten(1).square().mean(-1, keepdim=True) + 1e-20
            )
            reference_post = 2 * torch.sigmoid(
                normalized[:, 4:8] * scale.double()[1] + bias.double()[4:8]
            )
            precision = {
                "scalar_projection_rmse": float(
                    (scalar_projection.double() - reference_projection).square().mean().sqrt()
                ),
                "tensorcore_projection_rmse": float(
                    (tc_projection.double() - reference_projection).square().mean().sqrt()
                ),
                "scalar_post_max_abs": float(
                    (expected[2].double() - reference_post).abs().max()
                ),
                "tensorcore_post_max_abs": float((post.double() - reference_post).abs().max()),
            }
            errors = {}
            for name, actual, reference in zip(
                ("residual", "y", "post", "comb", "pre"), (out, y, post, comb, pre), expected,
                strict=True,
            ):
                errors[name] = float((actual.float() - reference.float()).abs().max())
                torch.testing.assert_close(actual, reference, rtol=0.003, atol=0.003)
            graphs = {}
            try:
                for name, run in (("scalar", baseline), ("tensorcore", candidate)):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(5):
                            run()
                    graphs[name] = graph
                samples = {name: [] for name in graphs}
                for repeat in range(7):
                    order = ("scalar", "tensorcore") if repeat % 2 else ("tensorcore", "scalar")
                    for name in order:
                        graphs[name].replay()
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(10):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 20)
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "projection_k_splits": config.projection_k_splits,
                            "max_abs_error": errors,
                            "fp64_comparison": precision,
                            "samples_us": samples,
                            "median_us": {
                                name: statistics.median(values)
                                for name, values in samples.items()
                            },
                        }
                    ),
                    flush=True,
                )
            finally:
                for graph in graphs.values():
                    graph.reset()
    finally:
        session.close()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
