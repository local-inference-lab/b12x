#!/usr/bin/env python
"""Micro-benchmark: small-N bf16 GEMV vs torch/cuBLAS ``F.linear``.

The GDN ``in_proj_ba`` (N ~ 64-128, K = hidden) stays bf16 in the FP6 serving
path, and at decode sizes cuBLAS picks a ~28 us 16x16 WMMA tile kernel for it
(48 calls/step ~= 1.35 ms/step). ``sparkinfer::bf16_gemv_small_n`` replaces that
with a one-CTA-per-column reduction kernel; this script measures both.

READ THE ``graph`` COLUMNS, NOT THE ``eager`` ONES. In eager mode the CUTE
JIT executor pays ~25 us of *Python* per-call dispatch (dlpack conversion +
execution-arg adaptation), which buries the ~2-3 us kernel. vLLM serving
replays decode steps from captured CUDA graphs, where that host cost does not
exist — the graph columns (capture once, time ``g.replay()``) are the
serving-relevant numbers for both cuBLAS and the GEMV.

Examples
--------
    python scripts/bench_bf16_gemv.py
    python scripts/bench_bf16_gemv.py --shapes 96x5120 64x5120 --ms 1 2 4
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


def _bench_graph(fn, iters: int, warmup: int) -> float:
    """Time ``fn`` as CUDA-graph replays (how serving runs decode)."""
    for _ in range(warmup):  # also JIT-compiles before capture
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # A few iterations per replay so replay-dispatch (~us) amortizes and
        # the measurement is the kernel itself.
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * 10)  # ms/call


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["64x5120", "96x5120", "128x5120", "256x5120"],
        help="N x K pairs (weight is (N, K) row-major as F.linear takes it)",
    )
    p.add_argument("--ms", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    from sparkinfer.gemm import bf16_gemv

    # The package API is lazy (install_lazy_api); touching the attribute
    # imports _kernel, which registers the torch custom op.
    bf16_gemv.bf16_gemv_small_n  # noqa: B018

    from sparkinfer.gemm.bf16_gemv._kernel import SMALL_M_MAX

    op = torch.ops.sparkinfer.bf16_gemv_small_n
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(
        f"{'shape':>12} {'m':>3} "
        f"{'cublas eager':>13} {'gemv eager':>11} "
        f"{'cublas graph':>13} {'gemv graph':>11} {'graph spdup':>11}"
    )
    for shape in args.shapes:
        n, k = (int(v) for v in shape.lower().split("x"))
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        for m in args.ms:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            # Correctness spot check before timing.
            if m <= SMALL_M_MAX:
                # Kernel path: f32-accumulated GEMV vs f32 reference.
                torch.testing.assert_close(
                    op(x, w).float(),
                    x.float() @ w.float().t(),
                    rtol=1e-2,
                    atol=1e-2,
                )
            else:
                # m > SMALL_M_MAX returns F.linear from inside the op, so the
                # only contract to verify is the fallback wiring — bitwise.
                # (Checking cuBLAS against an f32 reference here tests cuBLAS,
                # not this repo: torch's default bf16 reduced-precision
                # split-K reductions legitimately exceed 1% on
                # cancellation-heavy elements at skinny shapes.)
                if not torch.equal(op(x, w), torch.nn.functional.linear(x, w)):
                    raise AssertionError(
                        f"m={m} fallback output differs from F.linear"
                    )
            ref_fn = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
            gemv_fn = lambda: op(x, w)  # noqa: E731
            t_ref = _bench(ref_fn, args.iters, args.warmup)
            t_gemv = _bench(gemv_fn, args.iters, args.warmup)
            tg_ref = _bench_graph(ref_fn, args.iters, args.warmup)
            tg_gemv = _bench_graph(gemv_fn, args.iters, args.warmup)
            print(
                f"{shape:>12} {m:>3} "
                f"{t_ref * 1e3:>11.2f}us {t_gemv * 1e3:>9.2f}us "
                f"{tg_ref * 1e3:>11.2f}us {tg_gemv * 1e3:>9.2f}us "
                f"{tg_ref / tg_gemv:>10.2f}x"
            )


if __name__ == "__main__":
    main()
