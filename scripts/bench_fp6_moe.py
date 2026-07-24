#!/usr/bin/env python
"""Microbenchmark: sparkinfer W6A8 MX-FP6 fused MoE vs a BF16 grouped-MoE baseline.

Drives the upstream ``sparkinfer.moe.fused_moe`` plan/bind/run flow with
``quant_mode="w6a8_mx"`` / ``source_format="mxfp6_e2m3"`` across token counts,
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

import torch
import torch.nn.functional as F

from sparkinfer.moe import fused_moe
from sparkinfer.quantization.mxfp6 import quantize_moe_weights_to_fp6


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


def _bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n):
    """Reference BF16 gated-SiLU MoE grouped by expert ([up; gate] row order).

    Row order matches the fused kernel's ``w13_layout="w13"`` contract
    (``_gated_row_slices``): FC1 rows ``[0:N]`` are the up projection and rows
    ``[N:2N]`` are the gate, i.e. ``inter = silu(h[:, n:]) * h[:, :n]``.
    """
    m, k = x.shape
    out = torch.zeros(m, k, device=x.device, dtype=torch.float32)
    xf = x.float()
    for e in range(w1_bf.shape[0]):
        sel = topk_ids == e
        if not bool(sel.any()):
            continue
        rows, cols = sel.nonzero(as_tuple=True)
        xe = xf[rows]
        h = xe @ w1_bf[e].float().T
        up, gate = h[:, :n], h[:, n:]
        inter = F.silu(gate) * up
        down = inter @ w2_bf[e].float().T
        out.index_add_(0, rows, down * topk_weights[rows, cols].float().unsqueeze(1))
    return out


def _unswizzled_ue8m0_grid(w_bf16: torch.Tensor) -> torch.Tensor:
    """Per-K/32 UE8M0 scale bytes, unswizzled ``[E, rows, K//32]`` uint8.

    Mirrors the offline quantizer's block-scale rule
    (``_swizzled_block_scales`` minus the swizzle): the packed codes were
    quantized against exactly these exponents, and ``prepare_weights``
    applies the MMA swizzle itself.
    """
    from sparkinfer._lib.fp6 import (
        FLOAT6_E2M3_MAX,
        SF_VEC_SIZE_FP6,
        _ue8m0_scale_from_block_max,
    )

    e, rows, cols = w_bf16.shape
    blocks = cols // SF_VEC_SIZE_FP6
    block_max = (
        w_bf16.float().abs().view(e, rows, blocks, SF_VEC_SIZE_FP6).amax(dim=-1)
    )
    return _ue8m0_scale_from_block_max(block_max, FLOAT6_E2M3_MAX).contiguous()


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

    det = os.environ.get("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "")
    print(
        "deterministic combine "
        f"(SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT): {'ON' if det == '1' else 'off'}"
    )

    print(f"quantizing random weights: E={e} K={k} N={n} topk={topk}")
    w1_bf = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.15
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_e2m3")
    # prepare_weights wants UNswizzled UE8M0 grids (it swizzles internally);
    # the offline container's blockscales are already swizzled, so recompute.
    w1_grid = _unswizzled_ue8m0_grid(w1_bf)
    w2_grid = _unswizzled_ue8m0_grid(w2_bf)

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
        exec_plan = fused_moe.plan_execution(
            num_tokens=m,
            num_topk=tk,
            device=device,
            weight_plan=weight_plan,
            quant_mode="w6a8_mx",
        )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=m,
                num_topk=tk,
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
        ref = _bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n)
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
                lambda: _bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n),
                args.warmup, args.iters,
            )
            speedup = bf16_ms / fp6_ms
        print(f"{m:>7} {exec_plan.implementation:>8} {fp6_ms:>9.3f} {tok_s:>11.0f} "
              f"{bf16_ms:>9.3f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
