#!/usr/bin/env python
"""Validate a converted MX-FP6 MoE layer artifact against the sparkinfer FP6 kernel.

Loads a ``layer_{L}.moe_fp6.safetensors`` produced by
``scripts/quantize_model_fp6.py`` (or ``scripts/quantize_moe_fp6.py``), checks
its tensors match the kernel's input contract, then actually runs it through
``sparkinfer.moe.fused_moe`` (quant_mode ``w6a8_mx``) on random tokens at
several token counts and asserts the output is the right shape, finite, and
non-zero.

This proves the artifact *loads and executes* in the kernel end-to-end. Format
correctness (vs the torch reference recipe) is separately proven by
``tests/quantization/test_fp6_moe_weights_pipeline.py``.

Example:
    python scripts/validate_fp6_moe_artifact.py --artifact out_moe/layer_0.moe_fp6.safetensors
"""
from __future__ import annotations

import argparse
import time

import torch

from sparkinfer.quantization.mxfp6 import load_fp6_moe_weights


def _act_fmt_for_source(source_format: str) -> str:
    """FP6 sub-format for the *activation* operand (mirrors the kernel's fmt_a)."""
    sf = source_format.lower()
    if sf in ("mxfp6_e2m3", "e2m3"):
        return "e2m3"
    # mxfp6_default / mxfp6_mixed / mxfp6_e3m2: E3M2 activations.
    return "e3m2"


def _fast_dequant(packed_2d, swizzled_scales, num_fp6, fmt, gs=1.0):
    """Vectorized MX-FP6 dequant of ``(rows, 3*num_fp6//4)`` codes -> ``(rows, num_fp6)``.

    Uses the 64-entry FP6 LUT + the unswizzle helper, so it is O(rows*K) on GPU
    rather than the per-element Python loop in ``dequant_mxfp6_torch`` (which is
    intractable at E=256/K=2048).
    """
    from sparkinfer._lib.fp6 import (
        _fp6_e2m3_lut,
        _fp6_e3m2_lut,
        expand_mxfp6_packed_to_bytes,
        unswizzle_mxfp6_scales,
    )

    rows = packed_2d.shape[0]
    num_blocks = num_fp6 // 32
    # Vectorized 3:4 unpack to 6-bit codes (the per-element Python unpack is ~500k
    # .item() syncs/expert -- the second slowdown). This is the kernel's own path.
    codes = expand_mxfp6_packed_to_bytes(packed_2d, num_fp6).long()  # (rows, num_fp6)
    lut_vals = _fp6_e3m2_lut()[0] if fmt == "e3m2" else _fp6_e2m3_lut()[0]
    lut = torch.tensor(lut_vals, dtype=torch.float32, device=packed_2d.device)
    vals = lut[codes]
    sc = unswizzle_mxfp6_scales(swizzled_scales, rows, num_blocks).to(torch.int32)
    block_scale = torch.where(
        sc == 0, torch.zeros_like(sc, dtype=torch.float32),
        torch.pow(torch.tensor(2.0, device=sc.device), (sc.float() - 127.0)),
    )
    inv_gs = (1.0 / gs) if gs != 0.0 else 0.0
    return vals * block_scale.repeat_interleave(32, dim=1) * inv_gs


def _fast_qdq(x, fmt):
    """Vectorized activation FP6 quant->dequant (gs=1), matching the kernel recipe.

    Per-32-block UE8M0 scale, ``f32 -> bf16 -> e3m2`` rounding, nearest-FP6 via the
    64-entry LUT, then dequant. Fully tensorized over ``(rows, K)`` -- no per-element
    Python loops or ``.item()`` syncs (the reason the naive path took ~an hour).
    """
    from sparkinfer._lib.fp6 import (
        FLOAT6_E2M3_MAX,
        FLOAT6_E3M2_MAX,
        _fp6_e2m3_lut,
        _fp6_e3m2_lut,
        _ue8m0_scale_from_block_max,
    )

    rows, kdim = x.shape
    nb = kdim // 32
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    xf = x.float().view(rows, nb, 32)
    bmax = xf.abs().amax(dim=-1)  # (rows, nb)
    ue = _ue8m0_scale_from_block_max(bmax, fmt_max).to(torch.int32)
    bs = torch.where(
        ue == 0, torch.zeros_like(ue, dtype=torch.float32),
        torch.pow(torch.tensor(2.0, device=x.device), ue.float() - 127.0),
    )  # (rows, nb)
    inv = torch.where(bs == 0, torch.zeros_like(bs), 1.0 / bs)
    scaled = (xf * inv.unsqueeze(-1)).to(torch.bfloat16).to(torch.float32)
    scaled = scaled.clamp(-fmt_max, fmt_max)
    lut_vals = _fp6_e3m2_lut()[0] if fmt == "e3m2" else _fp6_e2m3_lut()[0]
    lut = torch.tensor(lut_vals, dtype=torch.float32, device=x.device)
    idx = (scaled.unsqueeze(-1) - lut).abs().argmin(dim=-1)  # (rows, nb, 32)
    q = lut[idx]
    deq = q * bs.unsqueeze(-1)  # gs=1 => inv_gs=1
    return deq.view(rows, kdim)


def _reference_moe(x, weights, topk_ids, topk_weights, act_fmt):
    """Dequant-FP6 reference MoE, fully vectorized (grouped by expert).

    Mirrors the validated numeric recipe (``test_moe_fp6_*``): gated SiLU FC1
    ([up; gate] row order), FP6-quantized intermediate, then FC2 -- using the
    weights actually stored in the artifact. Only the routed experts are
    dequantized, and all math is batched GPU matmuls (no Python per-token loop).
    """
    device = x.device
    m, k = x.shape
    n = weights.n
    fmt_w = weights.weight_fmt

    x_deq = _fast_qdq(x, act_fmt)  # (m, K)
    out = torch.zeros(m, k, dtype=torch.float32, device=device)
    for eid in torch.unique(topk_ids).tolist():
        sel = topk_ids == eid
        rows, cols = sel.nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w1e = _fast_dequant(
            weights.w1_fp6[eid], weights.w1_blockscale[eid], k, fmt_w,
            float(weights.w1_alphas[eid]),
        )  # (2N, K)
        w2e = _fast_dequant(
            weights.w2_fp6[eid], weights.w2_blockscale[eid], n, fmt_w,
            float(weights.w2_alphas[eid]),
        )  # (K, N)
        xe = x_deq[rows]  # (R, K)
        h = xe @ w1e.T  # (R, 2N)
        up, gate = h[:, :n], h[:, n:]
        inter = (torch.sigmoid(gate) * gate * up).to(torch.bfloat16)
        inter_deq = _fast_qdq(inter, act_fmt)  # (R, N)
        down = inter_deq @ w2e.T  # (R, K)
        out.index_add_(0, rows, down * topk_weights[rows, cols].float().unsqueeze(1))
    return out


def _unswizzle_grid(swizzled: torch.Tensor, rows: int, num_blocks: int) -> torch.Tensor:
    """Unswizzle the artifact's per-expert UE8M0 scales to ``[E, rows, blocks]``.

    ``prepare_weights`` validates UNswizzled grids and applies the MMA swizzle
    itself; the artifact stores the already-swizzled bytes, so invert them.
    """
    from sparkinfer._lib.fp6 import unswizzle_mxfp6_scales

    return torch.stack(
        [unswizzle_mxfp6_scales(swizzled[eid], rows, num_blocks)
         for eid in range(swizzled.shape[0])]
    ).contiguous()


def _run(x, weights, topk_ids, topk_weights):
    """Run the artifact through sparkinfer.moe.fused_moe (quant_mode w6a8_mx)."""
    from sparkinfer.moe import fused_moe

    device = x.device
    m, k = x.shape
    e, n, tk = weights.num_experts, weights.n, topk_ids.shape[1]

    weight_plan = fused_moe.plan_weights(
        quant_modes="w6a8_mx",
        source_format="mxfp6_e2m3",
        activation=weights.activation,
        params_dtype=torch.bfloat16,
        num_experts=e,
        hidden_size=k,
        intermediate_size=n,
        w13_layout="w13",  # [up; gate] FC1 rows (the only w6a8_mx layout)
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        w1_fp4=weights.w1_fp6,  # packed FP6 code bytes ride the fp4-named args
        w1_blockscale=_unswizzle_grid(weights.w1_blockscale, 2 * n, k // 32),
        w1_global_scale=weights.w1_alphas,
        a1_gscale=weights.a1_gscale,
        w2_fp4=weights.w2_fp6,
        w2_blockscale=_unswizzle_grid(weights.w2_blockscale, k, n // 32),
        w2_global_scale=weights.w2_alphas,
        a2_gscale=weights.a2_gscale,
        params_dtype=torch.bfloat16,
    )
    fused_moe.clear_caches()
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
    result = fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", required=True, help="path to layer_{L}.moe_fp6.safetensors")
    p.add_argument("--tokens", default="8,128,512", help="comma-separated token counts to test")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference", action="store_true",
                   help="also compute a dequant-FP6 reference MoE and report cosine per token count")
    p.add_argument("--cos-threshold", type=float, default=0.90,
                   help="minimum cosine(kernel, reference) to count as OK in --reference mode")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (the FP6 kernel runs on SM120)")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    w = load_fp6_moe_weights(args.artifact, device=device)
    e, k, n = w.num_experts, w.k, w.n
    print(f"artifact: {args.artifact}")
    print(f"  experts={e} K(hidden)={k} N(inter)={n} fmt={w.weight_fmt} src={w.source_format}")
    print(f"  w1_fp6={tuple(w.w1_fp6.shape)}[{w.w1_fp6.dtype}] "
          f"w1_blockscale={tuple(w.w1_blockscale.shape)}")
    print(f"  w2_fp6={tuple(w.w2_fp6.shape)}[{w.w2_fp6.dtype}] "
          f"w2_blockscale={tuple(w.w2_blockscale.shape)}")

    # Contract checks.
    assert w.w1_fp6.shape == (e, 2 * n, 3 * k // 4), "w1_fp6 shape mismatch"
    assert w.w2_fp6.shape == (e, k, 3 * n // 4), "w2_fp6 shape mismatch"
    assert w.w1_fp6.dtype == torch.uint8 and w.w2_fp6.dtype == torch.uint8

    topk = min(args.topk, e)
    all_ok = True
    for m in [int(t) for t in args.tokens.split(",")]:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
        topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
        topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(torch.float32)
        t0 = time.perf_counter()
        out = _run(x, w, topk_ids, topk_weights)
        dt = (time.perf_counter() - t0) * 1e3
        finite = bool(torch.isfinite(out).all())
        nonzero = float(out.abs().max()) > 0.0
        shape_ok = tuple(out.shape) == (m, k)
        ok = finite and nonzero and shape_ok
        cos_str = ""
        if args.reference:
            ref = _reference_moe(
                x, w, topk_ids, topk_weights, _act_fmt_for_source(w.source_format)
            )
            cos = float(torch.nn.functional.cosine_similarity(
                out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
            ))
            ok = ok and cos > args.cos_threshold
            cos_str = f" cos={cos:.4f}"
        all_ok &= ok
        print(f"  tokens={m:>5}: out={tuple(out.shape)} finite={finite} "
              f"nonzero={nonzero} |out|max={float(out.abs().max()):.4f}{cos_str} "
              f"{dt:.1f}ms  {'OK' if ok else 'FAIL'}")

    print("RESULT:", "READY (loads + runs in kernel)" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
