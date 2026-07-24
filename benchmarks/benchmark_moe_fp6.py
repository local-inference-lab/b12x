#!/usr/bin/env python3
"""Small-batch W6A8 MX-FP6 MoE benchmark using synthetic quantized expert weights.

Drives the upstream ``sparkinfer.moe.fused_moe`` plan/bind/run flow with
``quant_mode="w6a8_mx"`` / ``source_format="mxfp6_e2m3"`` and times CUDA-graph
replays of ``fused_moe.run``.  Graph capture is safe here: ``bind`` is
documented capture-safe (views only, never allocates) and the analogous
w4a8_mx dynamic path is graph-capture gated upstream
(tests/moe/test_w4a8_mx_tp_moe.py).
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.fp6_common import (
    bf16_grouped_moe,
    capture_graph_replay,
    check_outputs,
    fmt_us,
    make_l2_flush_fn,
    resolve_l2_flush_bytes,  # noqa: F401  (re-exported CLI helper)
    unswizzled_ue8m0_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    from sparkinfer.moe import fused_moe
    from sparkinfer.quantization.mxfp6 import quantize_moe_weights_to_fp6

    # Fully-qualified device: the scratch binder compares device strings
    # exactly, and tensors allocated on "cuda" report "cuda:0".
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    m, k, n, e, topk = args.m, args.k, args.n, args.experts, args.topk
    if k % 128 != 0 or n % 128 != 0:
        raise SystemExit(
            f"w6a8_mx requires K % 128 == 0 and N % 128 == 0, got K={k} N={n}"
        )

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).to(torch.float32)

    w1_bf = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.15
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_e2m3")
    w1_grid = unswizzled_ue8m0_grid(w1_bf)
    w2_grid = unswizzled_ue8m0_grid(w2_bf)

    fused_moe.clear_caches()
    weight_plan = fused_moe.plan_weights(
        quant_modes="w6a8_mx",
        source_format="mxfp6_e2m3",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=e,
        hidden_size=k,
        intermediate_size=n,
        w13_layout="w13",  # [up; gate] FC1 rows (the only w6a8_mx layout)
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        w1_fp4=w.w1_fp6,  # packed FP6 code bytes ride the fp4-named args
        w1_blockscale=w1_grid,
        w1_global_scale=w.w1_alphas,
        a1_gscale=w.a1_gscale,
        w2_fp4=w.w2_fp6,
        w2_blockscale=w2_grid,
        w2_global_scale=w.w2_alphas,
        a2_gscale=w.a2_gscale,
        params_dtype=torch.bfloat16,
    )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=m,
            num_topk=topk,
            device=device,
            weight_plan=weight_plan,
            core_token_counts=(m,),
            route_num_experts=0,
            quant_mode="w6a8_mx",
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=plan.scratch_specs()[i].device)
        for i, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
    )
    out = torch.empty(m, k, device=device, dtype=torch.bfloat16)
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=out,
        input_scales_static=True,
    )

    def launch() -> None:
        fused_moe.run(binding=binding)

    # Correctness gate before any timing (coding guideline): a silently
    # broken execution path must fail here, not report timings. Also warms
    # the compiled launch. bf16_grouped_moe is the same reference/threshold
    # approach scripts/bench_fp6_moe.py uses.
    launch()
    torch.cuda.synchronize()
    ref = bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n)
    check_outputs(out, ref, label="bf16 grouped MoE", cosine_threshold=0.99)
    del w1_bf, w2_bf

    replay = capture_graph_replay(launch)
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    for _ in range(args.warmup):
        if l2_flush is not None:
            l2_flush()
        replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for i in range(args.iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]
    med = statistics.median(times)
    print(
        f"W6A8 MX-FP6 MoE synthetic m={m} k={k} n={n} E={e} topk={topk}: "
        f"{fmt_us(times)}  (median {med:.3f} ms)"
    )


if __name__ == "__main__":
    main()
