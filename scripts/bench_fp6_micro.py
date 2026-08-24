#!/usr/bin/env python
"""Decode-latency benchmark: MX-FP6 (W6A8) BS1 micro kernel vs dynamic fused path.

Builds a synthetic routed-MoE layer, quantizes it to packed MX-FP6, then times
the ``b12x.moe.fused_moe`` plan/bind/run flow with
``B12X_ENABLE_FP6_MICRO`` off (dynamic fused path) and on (BS1 decode
micro kernel). Reports per-call latency, speedup, and the cosine similarity
between the two outputs so a regression in either is obvious.

The micro path only engages for small ``m`` (decode) and shapes that pass its
dispatch predicate; otherwise both runs hit the dynamic path and the speedup
is ~1.0 (still a useful correctness check). The env var is consulted at
dispatch time, so caches are cleared and the plan is rebuilt between the two
timings.

Examples
--------
Qwen3.6-35B-A3B-ish expert geometry, batch sizes 1/2/4:
    python scripts/bench_fp6_micro.py --k 2048 --n 768 --experts 128 --topk 8 --m 1 2 4

Small smoke test:
    python scripts/bench_fp6_micro.py --k 512 --n 512 --experts 8 --topk 2 --m 1
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.fp6_common import (
    bf16_grouped_moe,
    check_outputs,
    unswizzled_ue8m0_grid,
)


def _bench_once(fn, iters: int, warmup: int) -> float:
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
    return start.elapsed_time(end) / iters  # ms/call


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2048, help="hidden size (in/out features)")
    p.add_argument("--n", type=int, default=768, help="expert intermediate size")
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--m", type=int, nargs="+", default=[1, 2, 4], help="token counts")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    from b12x.moe import fused_moe
    from b12x.quantization.mxfp6 import quantize_moe_weights_to_fp6

    # Fully-qualified device: the scratch binder compares device strings
    # exactly, and tensors allocated on "cuda" report "cuda:0".
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    k, n, experts, topk = args.k, args.n, args.experts, args.topk
    if k % 128 != 0 or n % 128 != 0:
        raise SystemExit(
            f"w6a8_mx requires K % 128 == 0 and N % 128 == 0, got K={k} N={n}"
        )

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"shape        : k={k} n={n} experts={experts} topk={topk}")

    w1_bf = (torch.randn(experts, 2 * n, k, device=device) * 0.1).to(torch.bfloat16)
    w2_bf = (torch.randn(experts, k, n, device=device) * 0.1).to(torch.bfloat16)
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_e2m3")
    # prepare_weights wants UNswizzled UE8M0 grids (it swizzles internally).
    w1_grid = unswizzled_ue8m0_grid(w1_bf)
    w2_grid = unswizzled_ue8m0_grid(w2_bf)
    # w1_bf/w2_bf are kept alive: the correctness gates below need an
    # independent BF16 ground truth, not just micro-vs-dynamic agreement.

    config = fused_moe.PackedConfig(
        source_format="mxfp6_e2m3",
        w13_layout="w13",
    )
    weight_plan = fused_moe.plan_weights(
        config=config,
        activation="silu",
        dtype=torch.bfloat16,
        num_experts=experts,
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

    def make_runner(m: int, x, topk_ids, topk_weights):
        """(Re)plan + bind under the CURRENT env so dispatch re-decides."""
        fused_moe.clear_caches()
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=m,
                num_topk=topk,
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

        def run() -> torch.Tensor:
            fused_moe.run(binding=binding)
            return out

        return run

    print()
    header = f"{'m':>4} {'dynamic(ms)':>12} {'micro(ms)':>12} {'speedup':>9} {'cosine':>9}"
    print(header)
    print("-" * len(header))
    try:
        for m in args.m:
            x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
            topk_ids = torch.randint(
                0, experts, (m, topk), device=device, dtype=torch.int32
            )
            topk_weights = torch.softmax(
                torch.randn(m, topk, device=device), dim=-1
            ).float()

            os.environ["B12X_ENABLE_FP6_MICRO"] = "0"
            run = make_runner(m, x, topk_ids, topk_weights)
            dyn_out = run().clone()

            # Correctness gates before any timing (coding guideline). The
            # dynamic path is checked against an independent BF16 ground
            # truth (so a bug shared by both FP6 paths still fails), then
            # micro against dynamic (same quantized math, different kernel,
            # so the self-consistency bar is tighter). CorrectnessError
            # aborts the run on divergence or non-finite output.
            ref = bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n)
            check_outputs(
                dyn_out, ref, label=f"bf16 ref (m={m})", cosine_threshold=0.99
            )
            t_dyn = _bench_once(run, args.iters, args.warmup)

            os.environ["B12X_ENABLE_FP6_MICRO"] = "1"
            run = make_runner(m, x, topk_ids, topk_weights)
            micro_out = run().clone()
            check_outputs(
                micro_out,
                dyn_out,
                label=f"dynamic path (m={m})",
                cosine_threshold=0.995,
            )
            t_micro = _bench_once(run, args.iters, args.warmup)

            cos = torch.nn.functional.cosine_similarity(
                micro_out.reshape(-1).float(), dyn_out.reshape(-1).float(), dim=0
            ).item()
            speedup = t_dyn / t_micro if t_micro > 0 else float("nan")
            print(f"{m:>4} {t_dyn:>12.4f} {t_micro:>12.4f} {speedup:>9.2f} {cos:>9.4f}")
    finally:
        os.environ.pop("B12X_ENABLE_FP6_MICRO", None)


if __name__ == "__main__":
    main()
