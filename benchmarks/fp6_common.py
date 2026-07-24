"""Shared timing/correctness helpers for the FP6 benchmarks.

Vendored from ``benchmarks/benchmark_dense_gemm.py`` so the FP6 benchmarks do
not import that module: its top level imports ``flashinfer`` (an optional
dependency used only for upstream's FP4/MXFP8 reference baselines), which is
not required for FP6 work.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, List

import torch
import torch.nn.functional as F

from benchmarks.common import (  # noqa: F401  (re-exported for the fp6 benches)
    make_l2_flush_fn,
    resolve_l2_flush_bytes,
)


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    cosine_threshold: float,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected during correctness check vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    cos = cosine_similarity(candidate, reference)
    print(f"    check vs {label}: max_abs={max_abs:.8f} rmse={rmse:.8f} cos={cos:.10f}")
    if not math.isfinite(cos):
        raise CorrectnessError(
            f"cosine similarity vs {label} is non-finite: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )
    if cos < cosine_threshold:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below threshold "
            f"{cosine_threshold:.6f}: got {cos:.10f}"
        )


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    # Warm eager launch state before capture so compile/cache work does not
    # leak into the replay measurement.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return replay
