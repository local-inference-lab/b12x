"""Phase 1 gate for native packed-FP6 streaming (dense_gemm b_packed=True).

Compiles the packed-B kernel for the decode tile (16,128), the MTP/coarse
small-M tiles and a prefill tile, and asserts the output is BIT-IDENTICAL to
the validated b_preexpanded path (same activation codes, same scales, same
alpha — only the B streaming format differs). Finishes with a small M=1 GEMM
timing teaser on a real serving shape.

Run on the CUDA box:

    python scripts/smoke_packed_b.py
"""
from __future__ import annotations

import sys
import traceback

import torch


def _bench_ms(fn, iters: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required")
        return 2

    from sparkinfer._lib.fp6 import SF_VEC_SIZE_FP6, as_grouped_mxfp6_scale_view
    from sparkinfer._lib.dense_gemm import dense_gemm
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        _TILE,
        _quantize_matrix_fp6_bytes,
        quantize_dense_weight_to_fp6,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    f8_max = float(torch.finfo(torch.float8_e4m3fn).max)

    def make_case(m: int, n: int, k: int, source_format: str):
        w_bf16 = (torch.randn(n, k, device=device) * 0.1).to(torch.bfloat16)
        fp6w = quantize_dense_weight_to_fp6(w_bf16, source_format=source_format)
        fmt = fp6w.fmt
        x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
        a_amax = (
            torch.linalg.vector_norm(x, ord=float("inf"))
            .to(torch.float32)
            .clamp_min_(1e-6)
        )
        a_gs = (f8_max * 28.0 / a_amax).reshape(1)
        m_pad = ((m + _TILE - 1) // _TILE) * _TILE
        xpad = torch.empty(m_pad, k, dtype=torch.bfloat16, device=device)
        xpad[:m].copy_(x)
        a_codes, a_scale = _quantize_matrix_fp6_bytes(xpad, fmt, a_gs)
        a_sf = as_grouped_mxfp6_scale_view(a_scale.view(1, -1), m_pad, k)
        b_sf = as_grouped_mxfp6_scale_view(
            fp6w.scale_storage.view(1, -1), n, k
        )
        alpha = torch.reciprocal(a_gs * fp6w.global_scale)
        common = dict(
            alpha=alpha,
            ab_dtype=f"float6_{fmt}fn",
            sf_dtype="float8_e8m0fnu",
            sf_vec_size=SF_VEC_SIZE_FP6,
            c_dtype="bfloat16",
            a_preexpanded=True,
        )
        a_operand = (a_codes[:m].unsqueeze(-1), a_sf)
        return fp6w, a_operand, b_sf, common

    cases = [
        # (m, n, k, source_format) — tile coverage:
        (1, 6144, 512, "mxfp6_default"),  # (16,128) decode tile, 4 K-tiles
        (1, 6144, 512, "e3m2"),           # same tile, other sub-format
        (3, 6144, 512, "mxfp6_default"),  # MTP verify shape, same tile
        (5, 6144, 512, "mxfp6_default"),  # (64,128) small-M coarse tile
        (128, 256, 256, "mxfp6_default"),  # small square, 2 K-tiles
        (128, 6144, 512, "mxfp6_default"),  # prefill-ish wide-N tile
    ]

    failures = 0
    for m, n, k, source_format in cases:
        label = f"m={m:<4d} n={n:<5d} k={k:<5d} fmt={source_format}"
        try:
            fp6w, a_operand, b_sf, common = make_case(m, n, k, source_format)
            y_ref = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
            dense_gemm(
                a_operand,
                (fp6w.expanded_weight(), b_sf),
                out=y_ref,
                b_preexpanded=True,
                **common,
            )
            y_pk = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
            dense_gemm(
                a_operand,
                (fp6w.packed.unsqueeze(-1), b_sf),
                out=y_pk,
                b_packed=True,
                **common,
            )
            torch.cuda.synchronize()
            if torch.equal(y_ref, y_pk):
                print(f"PASS  {label}")
            else:
                diff = (y_ref.float() - y_pk.float()).abs()
                bad = (~torch.isclose(y_ref.float(), y_pk.float())).sum().item()
                print(
                    f"FAIL  {label}  max|diff|={diff.max().item():.4e} "
                    f"mismatched={bad}/{diff.numel()}"
                )
                failures += 1
        except Exception as exc:  # noqa: BLE001 — gate script: report and continue
            print(f"ERROR {label}  {type(exc).__name__}: {exc}")
            if failures == 0:
                # Full traceback once: DSL trace errors hide the failing source
                # line in their summary message.
                traceback.print_exc()
            failures += 1

    if failures:
        print(f"\n{failures} case(s) failed")
        return 1

    # Timing teaser on a real serving decode shape (full perf pass is Phase 3).
    m, n, k = 1, 6144, 5120
    fp6w, a_operand, b_sf, common = make_case(m, n, k, "mxfp6_default")
    expanded = fp6w.expanded_weight()
    packed = fp6w.packed.unsqueeze(-1)
    y = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
    t_ref = _bench_ms(
        lambda: dense_gemm(
            a_operand, (expanded, b_sf), out=y, b_preexpanded=True, **common
        )
    )
    t_pk = _bench_ms(
        lambda: dense_gemm(
            a_operand, (packed, b_sf), out=y, b_packed=True, **common
        )
    )
    print(f"\nM=1 {n}x{k} GEMM: preexpanded {t_ref:.4f} ms | packed {t_pk:.4f} ms "
          f"({t_ref / t_pk:.2f}x)")
    print("\nall packed-B cases bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
