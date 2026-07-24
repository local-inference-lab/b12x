#!/usr/bin/env python
"""Decode-latency benchmark: MX-FP6 (W6A6) BS1 micro kernel vs static fused path.

Builds a synthetic routed-MoE layer, quantizes it to packed MX-FP6, then times
the FP6 MoE launch (historically ``b12x_moe_fp6``; pending a
``sparkinfer.moe.fused_moe`` rewrite) with ``SPARKINFER_ENABLE_FP6_MICRO`` off (static/dynamic fused path)
and on (BS1 decode micro kernel). Reports per-call latency, speedup, and the
cosine similarity between the two outputs so a regression in either is obvious.

The micro path only engages for small ``m`` (decode) and shapes that pass
``is_supported_mxfp6`` + the smem-fit guard; otherwise both runs hit the static
path and the speedup is ~1.0 (still a useful correctness check).

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

import torch


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
    p.add_argument("--source-format", type=str, default="mxfp6_default")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    # TODO(port): this benchmark drove the historical b12x.integration.tp_moe host
    # API (allocate_tp_moe_workspace / b12x_moe_fp6 / clear_tp_moe_caches) and the
    # FP6 BS1 decode micro kernel (b12x.moe.fused.micro_fp6), neither of which has
    # an upstream sparkinfer equivalent yet. Rewrite the setup below against the
    # sparkinfer.moe.fused_moe plan/bind/run flow once the w6a8_mx run path lands.
    raise SystemExit("pending: rewrite against sparkinfer.moe.fused_moe")

    # Import after arg parse so a --help works without CUDA libs loaded.
    from b12x.integration.tp_moe import (  # historical b12x reference (see TODO above)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.micro_fp6 import (  # historical b12x reference (see TODO above)
        fp6_micro_fits_smem,
        fp6_micro_smem_bytes,
    )
    from sparkinfer.quantization.mxfp6 import quantize_moe_weights_to_fp6

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    k, n, experts, topk = args.k, args.n, args.experts, args.topk
    two_n = 2 * n

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"shape        : k={k} n={n} experts={experts} topk={topk}")
    print(f"micro smem   : {fp6_micro_smem_bytes(k, n)} bytes "
          f"(fits={fp6_micro_fits_smem(k, n, device)})")
    print(f"source_format: {args.source_format}")

    w1 = (torch.randn(experts, two_n, k, device=device) * 0.1).to(torch.bfloat16)
    w2 = (torch.randn(experts, k, n, device=device) * 0.1).to(torch.bfloat16)
    weights = quantize_moe_weights_to_fp6(
        w1, w2, source_format=args.source_format, activation="silu", use_gpu=True
    )

    def make_runner(m: int):
        x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
        topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
        topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).float()
        ws = allocate_tp_moe_workspace(
            x, weights.a1_gscale, weights.w1_fp6, weights.a2_gscale, weights.w2_fp6,
            topk_ids, quant_mode="w6a6", input_scales_static=True,
        )

        def run() -> torch.Tensor:
            return b12x_moe_fp6(
                x, weights.a1_gscale, weights.w1_fp6, weights.w1_blockscale,
                weights.w1_alphas, weights.a2_gscale, weights.w2_fp6,
                weights.w2_blockscale, weights.w2_alphas, topk_weights, topk_ids,
                workspace=ws, input_scales_static=True,
                source_format=args.source_format,
            )

        return run

    print()
    header = f"{'m':>4} {'static(ms)':>12} {'micro(ms)':>12} {'speedup':>9} {'cosine':>9}"
    print(header)
    print("-" * len(header))
    for m in args.m:
        run = make_runner(m)

        os.environ["SPARKINFER_ENABLE_FP6_MICRO"] = "0"
        clear_tp_moe_caches()
        static_out = run().clone()
        t_static = _bench_once(run, args.iters, args.warmup)

        os.environ["SPARKINFER_ENABLE_FP6_MICRO"] = "1"
        clear_tp_moe_caches()
        micro_out = run().clone()
        t_micro = _bench_once(run, args.iters, args.warmup)

        cos = torch.nn.functional.cosine_similarity(
            micro_out.reshape(-1).float(), static_out.reshape(-1).float(), dim=0
        ).item()
        speedup = t_static / t_micro if t_micro > 0 else float("nan")
        print(f"{m:>4} {t_static:>12.4f} {t_micro:>12.4f} {speedup:>9.2f} {cos:>9.4f}")

    os.environ.pop("SPARKINFER_ENABLE_FP6_MICRO", None)


if __name__ == "__main__":
    main()
