#!/usr/bin/env python
"""Decode-latency micro-benchmark for the sparkinfer FP6 *dense* linear path.

Localizes the vLLM ~2 tok/s decode bottleneck by isolating a single FP6 dense
linear at decode batch size (M=1). For each representative ``(out_features,
in_features)`` shape it reports:

* ``bf16 M=1``      -- torch GEMV baseline (the speed-of-light for one token).
* ``bf16 M=128``    -- torch GEMM at the padding tile size.
* ``fp6  M=1``      -- full ``dense_fp6_linear`` (activation quant + GEMM); note
                       the path pads M=1 -> 128 internally.
* ``fp6  quant``    -- the 128-row TMA activation-quant kernel alone.
* ``quant m1``      -- the small-M activation quantizer alone (real rows only;
                       this is what ``fp6 M=1`` actually uses now).
* ``gemm exp``      -- measured M=1 GEMM, byte-container weight (b_preexpanded).
* ``gemm pk``       -- measured M=1 GEMM, 3:4-packed weight streamed natively
                       (b_packed: 25% less B HBM traffic, in-smem expansion).

Reading the result:

* If ``fp6 quant`` is a large fraction of ``fp6 M=1``, the activation quantizer
  (not the GEMM) is the hot spot and should be optimized first.
* ``pk/exp < 1.0`` is the native packed-FP6 streaming win (plan Phase 2c gate:
  packed must beat expanded before it goes on the serving path).

Examples
--------
    python scripts/bench_fp6_decode.py
    python scripts/bench_fp6_decode.py --shapes 6144x5120 5120x5120 13824x5120 5120x13824
"""
from __future__ import annotations

import argparse

import torch


def _bench(fn, iters: int, warmup: int) -> float:
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


def _parse_shape(s: str) -> tuple[int, int]:
    n, k = s.lower().split("x")
    return int(n), int(k)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["6144x5120", "5120x5120", "13824x5120", "5120x13824"],
        help="out_features x in_features pairs (representative dense Linears)",
    )
    p.add_argument("--source-format", type=str, default="mxfp6_default")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hbm",
        action="store_true",
        help="Measure gemm exp/pk with weights streamed from HBM: rotate over "
        "enough weight copies (>2x the 128 MB L2) that no copy stays cache-"
        "resident — matches serving, where ~20 GB of weights stream per token. "
        "Without this flag the single reused weight sits in L2 and the columns "
        "measure L2 bandwidth + expansion cost instead.",
    )
    p.add_argument(
        "--ms",
        nargs="+",
        type=int,
        default=None,
        help="Phase 2.1 m-sweep: for each m (<=16) time the full fp6 path plus "
        "the isolated GEMM forced onto the (16,128) and (128,128) tiles, "
        "quantifying the MTP-verify tile cliff (e.g. --ms 1 4 5 8 16).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    from sparkinfer._lib.fp6 import (
        SF_VEC_SIZE_FP6,
        as_grouped_mxfp6_scale_view,
        mx_gs_numerator,
    )
    import sparkinfer._lib.dense_gemm as _dense_mod
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as _wmod
    from sparkinfer._lib.dense_gemm import dense_gemm
    from sparkinfer.quantization.mxfp6 import dense_fp6_linear, quantize_dense_weight_to_fp6
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6_bytes,
        _quantize_matrix_fp6_bytes_small_m,
        _TILE,
    )

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"source_format: {args.source_format}  (M pads to {_TILE})")
    print()
    header = (
        f"{'shape (NxK)':>14} {'bf16 M=1':>10} {'bf16 M=128':>11} "
        f"{'fp6 M=1':>10} {'fp6 fused':>10} {'fp6 quant':>10} {'quant m1':>9} {'gemm exp':>10} "
        f"{'gemm pk':>10} {'pk/exp':>7} {'fp6/bf16(1)':>11}"
    )
    print(header)
    print("-" * len(header))

    total_fp6 = 0.0
    for shape in args.shapes:
        n, k = _parse_shape(shape)
        w = (torch.randn(n, k, device=device) * 0.1).to(torch.bfloat16)
        fp6w = quantize_dense_weight_to_fp6(w, source_format=args.source_format)

        x1 = (torch.randn(1, k, device=device) * 0.1).to(torch.bfloat16)
        x128 = (torch.randn(_TILE, k, device=device) * 0.1).to(torch.bfloat16)

        t_bf16_1 = _bench(lambda: torch.nn.functional.linear(x1, w), args.iters, args.warmup)
        t_bf16_128 = _bench(lambda: torch.nn.functional.linear(x128, w), args.iters, args.warmup)

        _dense_mod._DENSE_FUSED_QUANT = False
        _wmod._DENSE_FUSED_QUANT = False
        t_fp6_1 = _bench(lambda: dense_fp6_linear(x1, fp6w), args.iters, args.warmup)

        _dense_mod._DENSE_FUSED_QUANT = True
        _wmod._DENSE_FUSED_QUANT = True
        t_fp6_fused = _bench(lambda: dense_fp6_linear(x1, fp6w), args.iters, args.warmup)
        _dense_mod._DENSE_FUSED_QUANT = False
        _wmod._DENSE_FUSED_QUANT = False

        # Activation-quant only (M padded to the tile, matching the hot path).
        # Bytes mode: the serving path's emit="bytes" quantizer (no expansion).
        xpad = torch.cat([x1, x1.new_zeros(_TILE - 1, k)], dim=0)
        a_amax = xpad.abs().amax().to(torch.float32).clamp_min(1e-6)
        a_gs = (mx_gs_numerator(fp6w.fmt) / a_amax).reshape(1)
        t_quant = _bench(
            lambda: _quantize_matrix_fp6_bytes(xpad, fp6w.fmt, a_gs),
            args.iters,
            args.warmup,
        )
        # Fully fused: in-kernel amax + quant + alpha (no vector_norm launch).
        t_quant_m1 = _bench(
            lambda: _quantize_matrix_fp6_bytes_small_m(
                x1, fp6w.fmt, fp6w.global_scale, _TILE
            ),
            args.iters,
            args.warmup,
        )

        # Isolated M=1 GEMM, expanded vs natively-packed B (same activation
        # codes/scales/alpha; only the weight streaming format differs).
        a_codes, a_scale = _quantize_matrix_fp6_bytes(xpad, fp6w.fmt, a_gs)
        a_sf = as_grouped_mxfp6_scale_view(a_scale.view(1, -1), _TILE, k)
        b_sf = fp6w.scale_view()
        gemm_common = dict(
            alpha=torch.reciprocal(a_gs * fp6w.global_scale),
            ab_dtype=fp6w.ab_dtype,
            sf_dtype="float8_e8m0fnu",
            sf_vec_size=SF_VEC_SIZE_FP6,
            c_dtype="bfloat16",
            a_preexpanded=True,
        )
        a_operand = (a_codes[:1].unsqueeze(-1), a_sf)
        expanded = fp6w.expanded_weight()
        packed = fp6w.packed.unsqueeze(-1)
        y = torch.empty((1, n, 1), device=device, dtype=torch.bfloat16)

        if args.hbm:
            # Enough copies that the rotation working set is ~2x L2: every
            # iteration streams its weight from HBM, like serving does.
            l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
            copies = max(2, (2 * l2_bytes) // expanded.numel() + 1)
            exp_copies = [expanded.clone() for _ in range(copies)]
            pk_copies = [packed.clone() for _ in range(copies)]
        else:
            exp_copies = [expanded]
            pk_copies = [packed]
        state = {"i": 0}

        def _run_exp():
            b = exp_copies[state["i"] % len(exp_copies)]
            state["i"] += 1
            dense_gemm(
                a_operand, (b, b_sf), out=y, b_preexpanded=True, **gemm_common
            )

        def _run_pk():
            b = pk_copies[state["i"] % len(pk_copies)]
            state["i"] += 1
            dense_gemm(a_operand, (b, b_sf), out=y, b_packed=True, **gemm_common)

        t_gemm_exp = _bench(_run_exp, args.iters, args.warmup)
        t_gemm_pk = _bench(_run_pk, args.iters, args.warmup)
        del exp_copies, pk_copies

        pk_ratio = t_gemm_pk / t_gemm_exp if t_gemm_exp > 0 else float("nan")
        ratio = t_fp6_1 / t_bf16_1 if t_bf16_1 > 0 else float("nan")
        total_fp6 += t_fp6_1
        print(
            f"{shape:>14} {t_bf16_1:>10.4f} {t_bf16_128:>11.4f} "
            f"{t_fp6_1:>10.4f} {t_fp6_fused:>10.4f} {t_quant:>10.4f} {t_quant_m1:>9.4f} "
            f"{t_gemm_exp:>10.4f} {t_gemm_pk:>10.4f} {pk_ratio:>6.2f}x {ratio:>11.1f}x"
        )

    print("-" * len(header))
    print(f"sum fp6 M=1 over {len(args.shapes)} linears: {total_fp6:.4f} ms")
    print(
        "Per-token estimate scales this by (#FP6 linears/layer x #layers x MTP "
        "passes); compare against the observed end-to-end tok/s."
    )

    if args.ms:
        # Phase 2.1 tile-cliff sweep: t16/t128 < 1.0 is the (16,128) win at
        # each decode/verify m; 'fp6 full' uses the default selector (now
        # (16,128) for every m <= 16).
        print()
        header2 = (
            f"{'shape (NxK)':>14} {'m':>4} {'fp6 full':>10} "
            f"{'gemm t16':>10} {'gemm t128':>10} {'t16/t128':>9}"
        )
        print(header2)
        print("-" * len(header2))
        for shape in args.shapes:
            n, k = _parse_shape(shape)
            w = (torch.randn(n, k, device=device) * 0.1).to(torch.bfloat16)
            fp6w = quantize_dense_weight_to_fp6(
                w, source_format=args.source_format
            )
            expanded = fp6w.expanded_weight()
            b_sf = fp6w.scale_view()
            for m in args.ms:
                xm = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
                t_full = _bench(
                    lambda: dense_fp6_linear(xm, fp6w), args.iters, args.warmup
                )
                row = f"{shape:>14} {m:>4} {t_full:>10.4f}"
                if m <= 16:
                    a_codes, a_scale, alpha = _quantize_matrix_fp6_bytes_small_m(
                        xm, fp6w.fmt, fp6w.global_scale, _TILE
                    )
                    a_sf = as_grouped_mxfp6_scale_view(
                        a_scale.view(1, -1), _TILE, k
                    )
                    a_op = (a_codes[:m].unsqueeze(-1), a_sf)
                    ym = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
                    common = dict(
                        alpha=alpha,
                        ab_dtype=fp6w.ab_dtype,
                        sf_dtype="float8_e8m0fnu",
                        sf_vec_size=SF_VEC_SIZE_FP6,
                        c_dtype="bfloat16",
                        a_preexpanded=True,
                        b_preexpanded=True,
                    )
                    t16 = _bench(
                        lambda: dense_gemm(
                            a_op, (expanded, b_sf), out=ym,
                            mma_tiler_mn=(16, 128), **common,
                        ),
                        args.iters,
                        args.warmup,
                    )
                    t128 = _bench(
                        lambda: dense_gemm(
                            a_op, (expanded, b_sf), out=ym,
                            mma_tiler_mn=(128, 128), **common,
                        ),
                        args.iters,
                        args.warmup,
                    )
                    row += (
                        f" {t16:>10.4f} {t128:>10.4f} "
                        f"{t16 / t128 if t128 > 0 else float('nan'):>8.2f}x"
                    )
                print(row)


if __name__ == "__main__":
    main()
