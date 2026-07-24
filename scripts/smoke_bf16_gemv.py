#!/usr/bin/env python
"""Standalone phase-by-phase smoke test for the small-N bf16 GEMV.

The GEMV segfaults the vLLM engine-core process at weight-load warm-run while
passing the full pytest suite in the dev venv — a venv-specific failure
(different CUTLASS DSL / CUDA toolchain stacks compile the kernel
independently; the disk cache is keyed per toolchain). This script runs each
phase (compile -> first launch -> sync -> correctness) with a flushed print
before and after, so a hard crash (segfault) pinpoints the exact phase and m.

Run it in BOTH venvs and compare:

    # dev venv (expected: OK)
    python scripts/smoke_bf16_gemv.py --n 96 --k 5120

    # vLLM venv (the crashing one)
    ~/kld-nightly-vllm/venv/bin/python scripts/smoke_bf16_gemv.py --n 96 --k 5120

Use the real in_proj_ba (N, K) from the server log line
"sparkinfer FP6: small-N bf16 GEMV for ... (N=..., K=...)". To rule the disk cache
in/out, rerun with SPARKINFER_COMPILE_DISK_CACHE=0.
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--m-max", type=int, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    import cutlass

    print(
        f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} "
        f"cutlass_dsl={getattr(cutlass, '__version__', '?')} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    from sparkinfer._lib.compiler import compile_cache_info
    from sparkinfer.gemm.bf16_gemv import SMALL_M_MAX
    from sparkinfer.gemm.bf16_gemv._kernel import compile_bf16_gemv_small_n

    n, k = args.n, args.k
    m_max = args.m_max or SMALL_M_MAX
    torch.manual_seed(0)
    device = torch.device("cuda")
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    for m in range(1, m_max + 1):
        print(f"[smoke] m={m}: compiling ({n}x{k})...", flush=True)
        launch = compile_bf16_gemv_small_n(m, n, k)
        print(f"[smoke] m={m}: compiled  ({compile_cache_info()})", flush=True)

        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        y = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        print(f"[smoke] m={m}: launching...", flush=True)
        launch(x, w, y)
        print(f"[smoke] m={m}: launched, syncing...", flush=True)
        torch.cuda.synchronize()
        print(f"[smoke] m={m}: synced", flush=True)

        ref = x.float() @ w.float().t()
        ok = torch.allclose(y.float(), ref, rtol=1e-2, atol=1e-2)
        max_err = (y.float() - ref).abs().max().item()
        print(f"[smoke] m={m}: correct={ok} max_err={max_err:.4f}", flush=True)
        if not ok:
            raise SystemExit(f"mismatch at m={m}")

    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main()
