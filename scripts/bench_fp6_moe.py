#!/usr/bin/env python
"""Microbenchmark: b12x W6A8 MX-FP6 fused MoE vs a BF16 grouped-MoE baseline.

Drives the upstream ``b12x.moe.fused_moe`` plan/bind/run flow with a
``PackedConfig(source_format="mxfp6_e2m3")`` across token counts,
and compares against a straightforward BF16 grouped MoE (the same gated-SiLU
math in full precision).  This is the Step 5 acceptance gate for the FP6 MoE
port: each token count prints a correctness line (max_abs / rmse / cosine vs
the BF16 reference) alongside the latency numbers.

Weights are random and quantized once via the offline MX-FP6 path
(``quantize_moe_weights_to_fp6``) for the packed codes; the per-K/32 UE8M0
block-scale grids are recomputed *unswizzled* here because
``prepare_weights`` (``prepare_w6a8_mxfp6_weights``) expects unswizzled
``[E, rows, K//32]`` grids and applies the MMA swizzle itself.

Example:
    python scripts/bench_fp6_moe.py --experts 256 --k 2048 --n 512 --topk 8 \
        --tokens 1,8,128,512,4096
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.fp6_common import bf16_grouped_moe, unswizzled_ue8m0_grid
from b12x.moe import fused_moe
from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--k", type=int, default=2048, help="hidden size")
    p.add_argument("--n", type=int, default=512, help="intermediate size")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tokens", default="1,8,128,512,4096")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true", help="skip the BF16 baseline")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (the FP6 kernel runs on SM120)")
    # Fully-qualified device: the scratch binder compares device strings
    # exactly, and tensors allocated on "cuda" report "cuda:0".
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    e, k, n, topk = args.experts, args.k, args.n, args.topk
    if k % 128 != 0 or n % 128 != 0:
        raise SystemExit(
            f"w6a8_mx requires K % 128 == 0 and N % 128 == 0, got K={k} N={n}"
        )

    det = os.environ.get("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "")
    print(
        "deterministic combine "
        f"(B12X_DYNAMIC_DETERMINISTIC_OUTPUT): {'ON' if det == '1' else 'off'}"
    )

    print(f"quantizing random weights: E={e} K={k} N={n} topk={topk}")
    w1_bf = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.15
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_e2m3")
    # prepare_weights wants UNswizzled UE8M0 grids (it swizzles internally);
    # the offline container's blockscales are already swizzled, so recompute.
    w1_grid = unswizzled_ue8m0_grid(w1_bf)
    w2_grid = unswizzled_ue8m0_grid(w2_bf)

    config = fused_moe.PackedConfig(
        source_format="mxfp6_e2m3",
        w13_layout="w13",
    )
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=e,
        hidden_size=k,
        intermediate_size=n,
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=w.w1_fp6,
            w2=w.w2_fp6,
            w13_block_scales=w1_grid,
            w2_block_scales=w2_grid,
            w13_global_scales=w.w1_alphas,
            w2_global_scales=w.w2_alphas,
            input_scale=w.a1_gscale,
            intermediate_scale=w.a2_gscale,
        ),
    )

    print(f"\n{'tokens':>7} {'backend':>8} {'fp6_ms':>9} {'fp6_tok/s':>11} "
          f"{'bf16_ms':>9} {'speedup':>8}")
    print("-" * 60)
    for m in [int(t) for t in args.tokens.split(",")]:
        tk = min(topk, e)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
        topk_ids = torch.randint(0, e, (m, tk), device=device, dtype=torch.int32)
        topk_weights = torch.softmax(
            torch.randn(m, tk, device=device), dim=-1
        ).to(torch.float32)

        fused_moe.clear_caches()
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=m,
                num_topk=tk,
                device=device,
                config=config,
                weight_plan=weight_plan,
                core_token_counts=(m,),
                route_num_experts=0,
            )
        )
        scratch = tuple(
            torch.empty(shape, dtype=dtype, device=plan.scratch_specs()[i].device)
            for i, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
        )
        out = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
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

        def _fp6():
            return fused_moe.run(binding=binding)

        # Correctness gate first (also warms the compiled launch).
        got = _fp6()
        torch.cuda.synchronize()
        ref = bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n)
        diff = (got.float() - ref).abs()
        max_abs = diff.max().item()
        rmse = diff.square().mean().sqrt().item()
        cos = F.cosine_similarity(
            got.float().flatten(), ref.flatten(), dim=0
        ).item()
        print(f"  m={m}: correctness vs bf16 ref: max_abs={max_abs:.6f} "
              f"rmse={rmse:.6f} cosine={cos:.6f}")

        fp6_ms = _time_ms(_fp6, args.warmup, args.iters)
        tok_s = m / (fp6_ms * 1e-3)
        bf16_ms = float("nan")
        speedup = float("nan")
        if not args.no_bf16:
            bf16_ms = _time_ms(
                lambda: bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n),
                args.warmup, args.iters,
            )
            speedup = bf16_ms / fp6_ms
        print(f"{m:>7} {plan.launch_plan.implementation:>8} "
              f"{fp6_ms:>9.3f} {tok_s:>11.0f} "
              f"{bf16_ms:>9.3f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
