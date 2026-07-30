#!/usr/bin/env python3
"""Phase-A evidence: MX-FP6 dense GEMM vs packed-B / expanded-B / optional CUTLASS FP8.

Times the production W6A8 MMA path (FP8-E4M3 activations x FP6-E2M3 weights) with
operands pre-quantized outside the captured region, so the measurement is the
GEMM itself. Records the AGENTS.md evidence header (command, commit, GPU UUID /
mode, correctness state, raw timings, ratio direction).

Default shape preset is Behemoth-R1-123B TP=2 shards on 2x RTX PRO 6000:

  qkv:     (N, K) = (7168, 12288)   # column-parallel shard of 14336 x 12288
  o:       (N, K) = (12288, 6144)   # row-parallel shard of 12288 x 12288
  gate_up: (N, K) = (28672, 12288)  # column-parallel shard of 57344 x 12288
  down:    (N, K) = (12288, 14336)  # row-parallel shard of 12288 x 28672

M sweep: 1 2 4 8 16 2048 4096 8192 (decode + prefill).

Examples
--------
Behemoth Phase-A matrix (serving box)::

    python benchmarks/benchmark_dense_gemm_fp6.py --preset behemoth-tp2 \\
        --warmup 10 --iters 50

Single shape, packed vs expanded only::

    python benchmarks/benchmark_dense_gemm_fp6.py --m 8192 --n 28672 --k 12288

Skip the optional vLLM CUTLASS FP8 arm::

    python benchmarks/benchmark_dense_gemm_fp6.py --preset behemoth-tp2 --no-fp8

Item-4 tile sweep (expanded-B arm, candidate MMA tiles, cross-tile
bit-equality check; run once with the default unroll and once with
``SPARKINFER_FP6_LARGE_M_UNROLL=0`` for the unroll A/B)::

    python benchmarks/benchmark_dense_gemm_fp6.py --preset behemoth-tp2 \\
        --tile-sweep --warmup 10 --iters 50 --json-out fp6_tile_sweep.json

Decode-tile sweep (packed-B stream, M<=16 regime)::

    python benchmarks/benchmark_dense_gemm_fp6.py --preset behemoth-tp2 \\
        --tile-sweep --tile-arm packed --tiles 16x128,16x64,32x64,32x128 \\
        --json-out fp6_decode_tile_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from benchmarks.fp6_common import (
    capture_graph_replay,
    check_outputs,
    fmt_us,
)
from sparkinfer._lib.dense_gemm import dense_gemm
from sparkinfer._lib.fp6 import (
    SF_VEC_SIZE_FP6,
    as_grouped_mxfp6_scale_view,
    mx_gs_numerator,
)
from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
    _TILE,
    _quantize_matrix_fp6_bytes,
    quantize_dense_weight_to_fp6,
)

# Behemoth-R1-123B (Mistral-Large-2411 tune) at TP=2, per-GPU shard sizes.
# Full model: hidden=12288, intermediate=28672, heads=96, head_dim=128.
BEHEMOTH_TP2_SHAPES: dict[str, tuple[int, int]] = {
    "qkv": (7168, 12288),  # (96*128 + 2*8*128) / 2
    "o": (12288, 6144),  # row-parallel: K = hidden/tp
    "gate_up": (28672, 12288),  # (2 * intermediate) / 2
    "down": (12288, 14336),  # row-parallel: K = intermediate/tp
}

DEFAULT_M_SWEEP = (1, 2, 4, 8, 16, 2048, 4096, 8192)
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FP8_BLOCK = 128
# The FP8 arm reports under whichever kernel actually ran. Only a genuine
# vLLM blockwise CUTLASS result carries the CUTLASS name.
FP8_CUTLASS_ARM = "fp8_cutlass"
FP8_CUBLASLT_ARM = "fp8_cublaslt_pertensor"


@dataclass(frozen=True)
class ArmResult:
    name: str
    median_us: float
    min_us: float
    raw_ms: list[float]
    cosine: Optional[float]
    skipped: str = ""


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pathlib.Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _med_us(times_ms: list[float]) -> float:
    return statistics.median(times_ms) * 1000.0


def _min_us(times_ms: list[float]) -> float:
    return min(times_ms) * 1000.0


def _tflops(m: int, n: int, k: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return (2.0 * m * n * k) / (us * 1e-6) / 1e12


def _bench_events(
    replay: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush,
) -> list[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        replay()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    denom = af.norm() * bf.norm()
    if float(denom) == 0.0:
        return 1.0 if float(af.norm()) == 0.0 else 0.0
    return float(torch.dot(af, bf) / denom)


def _pad_m(m: int) -> int:
    return ((m + _TILE - 1) // _TILE) * _TILE


def _setup_w6a8_operands(m: int, n: int, k: int, *, seed: int):
    """Pre-quantize W6A8 operands the same way production dense_fp6_linear does.

    Returns activation byte-containers + scales, packed weight, expanded weight,
    weight scale view, alpha, and a BF16 matmul oracle for the correctness gate.
    Quantization is outside the timed region.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.2).to(torch.bfloat16)
    w_bf16 = (torch.randn(n, k, device=device) * 0.2).to(torch.bfloat16)
    fp6w = quantize_dense_weight_to_fp6(w_bf16, source_format="mxfp6_w6a8")
    assert fp6w.fmt == "e2m3" and fp6w.act_fmt == "e4m3"

    # Per-tensor activation GS (GEMM-only microbench; per-row host chain is
    # attributed separately in the serving-profile Phase-A arm).
    a_amax = torch.linalg.vector_norm(x, ord=float("inf")).to(torch.float32)
    a_gs = (mx_gs_numerator("e4m3") / a_amax.clamp_min(1e-6)).reshape(1)
    m_pad = _pad_m(m)
    x_pad = torch.zeros(m_pad, k, dtype=torch.bfloat16, device=device)
    x_pad[:m].copy_(x)
    a_codes, a_scale = _quantize_matrix_fp6_bytes(x_pad, "e4m3", a_gs)
    a_sf = as_grouped_mxfp6_scale_view(a_scale.view(1, -1), m_pad, k)
    b_sf = fp6w.scale_view()
    alpha = torch.reciprocal(a_gs * fp6w.global_scale)
    oracle = x.float() @ w_bf16.float().T
    return {
        "a": (a_codes[:m].unsqueeze(-1).contiguous(), a_sf),
        "b_packed": fp6w.packed.unsqueeze(-1).contiguous(),
        "b_expanded": fp6w.expanded_weight().contiguous(),
        "b_sf": b_sf,
        "alpha": alpha,
        "oracle": oracle,
        "ab_dtype": "float6_e2m3fn",
        "a_fmt": "e4m3",
        "b_fmt": "e2m3",
    }


def _run_fp6_arm(
    *,
    name: str,
    m: int,
    n: int,
    k: int,
    operands: dict,
    b_packed: bool,
    warmup: int,
    iters: int,
    l2_flush,
    check: bool,
    mma_tiler_mn: Optional[tuple[int, int]] = None,
    out_holder: Optional[list] = None,
) -> ArmResult:
    """Time one FP6 GEMM arm.

    ``mma_tiler_mn`` overrides the policy tile (the item-4 tile sweep).
    ``out_holder``, when given, receives the output tensor (holding the last
    replay's result) so the sweep can assert cross-tile bit-equality.
    """
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    if out_holder is not None:
        out_holder.append(out)
    b = operands["b_packed"] if b_packed else operands["b_expanded"]

    def launch() -> None:
        dense_gemm(
            operands["a"],
            (b, operands["b_sf"]),
            alpha=operands["alpha"],
            ab_dtype=operands["ab_dtype"],
            sf_dtype="float8_e8m0fnu",
            sf_vec_size=SF_VEC_SIZE_FP6,
            c_dtype="bfloat16",
            out=out,
            a_preexpanded=True,
            b_preexpanded=not b_packed,
            b_packed=b_packed,
            a_fmt=operands["a_fmt"],
            b_fmt=operands["b_fmt"],
            mma_tiler_mn=mma_tiler_mn,
        )

    try:
        replay = capture_graph_replay(launch)
        times = _bench_events(
            replay, warmup=warmup, iters=iters, l2_flush=l2_flush
        )
    except Exception as exc:  # unsupported tile/plan: report, keep sweeping
        return ArmResult(
            name=name,
            median_us=float("nan"),
            min_us=float("nan"),
            raw_ms=[],
            cosine=None,
            skipped=f"{type(exc).__name__}: {exc}",
        )
    cos: Optional[float] = None
    if check:
        cos = _cosine(out[:, :, 0], operands["oracle"])
        check_outputs(
            out[:, :, 0],
            operands["oracle"],
            label=f"{name} vs bf16 oracle",
            cosine_threshold=0.95,
            rel_rmse_threshold=0.35,
        )
    return ArmResult(
        name=name,
        median_us=_med_us(times),
        min_us=_min_us(times),
        raw_ms=list(times),
        cosine=cos,
    )


def _try_import_cutlass_scaled_mm() -> Optional[Callable]:
    """Resolve vLLM's SM120 blockwise CUTLASS FP8 GEMM entry point, if present."""
    candidates: list[Callable] = []
    try:
        from vllm import _custom_ops as ops  # type: ignore

        if hasattr(ops, "cutlass_scaled_mm"):
            candidates.append(ops.cutlass_scaled_mm)
    except Exception:
        pass
    try:
        fn = getattr(torch.ops._C, "cutlass_scaled_mm", None)
        if fn is not None:
            candidates.append(fn)
    except Exception:
        pass
    return candidates[0] if candidates else None


def _quantize_fp8_blockwise(
    x: torch.Tensor, *, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block FP8-E4M3 + fp32 scale (compressed-tensors / vLLM blockwise)."""
    if x.ndim != 2:
        raise ValueError(f"expected rank-2, got {tuple(x.shape)}")
    rows, cols = x.shape
    if rows % block != 0 or cols % block != 0:
        raise ValueError(
            f"blockwise FP8 needs multiples of {block}, got {(rows, cols)}"
        )
    xf = x.float().view(rows // block, block, cols // block, block)
    amax = xf.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = (amax / FLOAT8_E4M3_MAX).to(torch.float32)
    inv = (1.0 / scale).unsqueeze(1).unsqueeze(3)
    q = (xf * inv).to(torch.float8_e4m3fn).view(rows, cols)
    return q.contiguous(), scale.contiguous()


def _quantize_fp8_per_token_kblock(
    x: torch.Tensor, *, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token, K-block FP8-E4M3 (vLLM dynamic activation scale layout)."""
    m, k = x.shape
    if k % block != 0:
        raise ValueError(f"K={k} not divisible by {block}")
    xf = x.float().view(m, k // block, block)
    amax = xf.abs().amax(dim=-1).clamp_min(1e-12)
    scale = (amax / FLOAT8_E4M3_MAX).to(torch.float32)  # (M, K/block)
    inv = (1.0 / scale).unsqueeze(-1)
    q = (xf * inv).to(torch.float8_e4m3fn).view(m, k)
    return q.contiguous(), scale.contiguous()


def _make_torch_scaled_mm_launch(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    out: torch.Tensor,
) -> Optional[Callable]:
    """cuBLASLt FP8 GEMM on the same shape, as a fallback reference arm.

    vLLM's ``cutlass_scaled_mm`` signature and its blockwise-scale support move
    between nightlies, and when it rejects a call it raises a bare
    ``AssertionError`` with no message - unusable as an arm we depend on. For
    "what can this card do at FP8 on this shape" the vendor path answers the
    question at least as well: per-tensor scales instead of blockwise changes
    the epilogue, not the MMA rate, and the MMA rate is the comparison we need.

    ``mat2`` must be column-major, which ``b_q`` (N,K contiguous) satisfies as
    ``.t()`` without a copy.
    """
    scaled_mm = getattr(torch, "_scaled_mm", None)
    if scaled_mm is None:
        return None
    b_t = b_q.t()
    one = torch.ones((), dtype=torch.float32, device=a_q.device)

    def _kw() -> None:
        out.copy_(
            scaled_mm(
                a_q,
                b_t,
                scale_a=one,
                scale_b=one,
                out_dtype=torch.bfloat16,
            )
        )

    def _pos() -> None:
        out.copy_(scaled_mm(a_q, b_t, one, one, None, None, torch.bfloat16))

    for candidate in (_kw, _pos):
        try:
            candidate()
            torch.cuda.synchronize()
            return candidate
        except Exception:
            continue
    return None


def _run_fp8_cutlass_arm(
    *,
    m: int,
    n: int,
    k: int,
    seed: int,
    warmup: int,
    iters: int,
    l2_flush,
    check: bool,
) -> ArmResult:
    # Renamed on the fallback path below: a cuBLASLt per-tensor result recorded
    # under the CUTLASS label reads as vLLM's blockwise serving kernel when it
    # is neither blockwise nor vLLM's, and one FP8 reference was misquoted that
    # way already.
    arm_name = FP8_CUTLASS_ARM
    cutlass_mm = _try_import_cutlass_scaled_mm()
    if cutlass_mm is None:
        return ArmResult(
            name=arm_name,
            median_us=float("nan"),
            min_us=float("nan"),
            raw_ms=[],
            cosine=None,
            skipped="vLLM cutlass_scaled_mm not importable in this env",
        )
    # Unaligned M is left alone: some vLLM builds want 128-aligned A scales for
    # the blockwise layout, but decode Ms often work anyway, and a failure here
    # is caught by the launch probe below rather than pre-empted.
    if k % FP8_BLOCK != 0 or n % FP8_BLOCK != 0:
        return ArmResult(
            name=arm_name,
            median_us=float("nan"),
            min_us=float("nan"),
            raw_ms=[],
            cosine=None,
            skipped=f"N={n} or K={k} not divisible by {FP8_BLOCK}",
        )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.2).to(torch.bfloat16)
    w = (torch.randn(n, k, device=device) * 0.2).to(torch.bfloat16)
    # Match compressed-tensors W8A8-FP8-BLOCK: 128x128 weight blocks, dynamic
    # per-token activation scales along K/128.
    a_q, a_s = _quantize_fp8_per_token_kblock(x)
    b_q, b_s = _quantize_fp8_blockwise(w)
    # CUTLASS scaled_mm expects B as (K, N) column-major / transposed weight in
    # many vLLM builds. Try (N,K) first; fall back to .T on TypeError/RuntimeError.
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    oracle = x.float() @ w.float().T

    def _launch_nk() -> None:
        cutlass_mm(out, a_q, b_q, a_s, b_s)

    def _launch_kn() -> None:
        cutlass_mm(out, a_q, b_q.T.contiguous(), a_s, b_s.T.contiguous())

    launch = _launch_nk
    try:
        launch()
        torch.cuda.synchronize()
    except Exception:
        launch = _launch_kn
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as exc:
            fallback = _make_torch_scaled_mm_launch(a_q, b_q, out)
            if fallback is None:
                return ArmResult(
                    name=arm_name,
                    median_us=float("nan"),
                    min_us=float("nan"),
                    raw_ms=[],
                    cosine=None,
                    skipped=(
                        f"cutlass_scaled_mm failed: {type(exc).__name__}: {exc}"
                        "; torch._scaled_mm fallback also unavailable"
                    ),
                )
            print(
                "fp8 arm: cutlass_scaled_mm rejected the call "
                f"({type(exc).__name__}); using torch._scaled_mm (cuBLASLt) "
                f"with per-tensor scales instead, recorded as {FP8_CUBLASLT_ARM}"
            )
            arm_name = FP8_CUBLASLT_ARM
            launch = fallback
            check = False

    replay = capture_graph_replay(launch)
    try:
        times = _bench_events(replay, warmup=warmup, iters=iters, l2_flush=l2_flush)
    except Exception as exc:
        return ArmResult(
            name=arm_name,
            median_us=float("nan"),
            min_us=float("nan"),
            raw_ms=[],
            cosine=None,
            skipped=f"{arm_name} replay failed: {type(exc).__name__}: {exc}",
        )

    cos: Optional[float] = None
    if check:
        cos = _cosine(out, oracle)
        if cos < 0.95:
            return ArmResult(
                name=arm_name,
                median_us=_med_us(times),
                min_us=_min_us(times),
                raw_ms=list(times),
                cosine=cos,
                skipped=f"correctness gate failed: cos={cos:.5f} < 0.95",
            )
    return ArmResult(
        name=arm_name,
        median_us=_med_us(times),
        min_us=_min_us(times),
        raw_ms=list(times),
        cosine=cos,
    )


def _print_evidence_header(args: argparse.Namespace) -> dict:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    mode = nvidia_smi_gpu_mode_snapshot()
    header = {
        "command": " ".join(sys.argv),
        "commit": _git_rev(),
        "cwd": str(pathlib.Path.cwd()),
        "captured_unix_ns": time.time_ns(),
        "torch_cuda_uuid": str(getattr(props, "uuid", "")),
        "gpu_name": props.name,
        "sm_count": int(props.multi_processor_count),
        "major_minor": f"{props.major}.{props.minor}",
        "nvidia_smi_mode": mode,
        "env": {
            "SPARKINFER_PACKED_B_MIN_N": os.getenv("SPARKINFER_PACKED_B_MIN_N"),
            "SPARKINFER_DENSE_PER_ROW_GS": os.getenv("SPARKINFER_DENSE_PER_ROW_GS"),
            "SPARKINFER_FP6_LARGE_M_UNROLL": os.getenv(
                "SPARKINFER_FP6_LARGE_M_UNROLL"
            ),
            "SPARKINFER_FP6_LARGE_M_TILE": os.getenv("SPARKINFER_FP6_LARGE_M_TILE"),
            "SPARKINFER_FP6_DECODE_TILE": os.getenv("SPARKINFER_FP6_DECODE_TILE"),
            "SPARKINFER_DENSE_TILE_SWIZZLE": os.getenv(
                "SPARKINFER_DENSE_TILE_SWIZZLE"
            ),
            "SPARKINFER_DENSE_AB_STAGES": os.getenv("SPARKINFER_DENSE_AB_STAGES"),
            "SPARKINFER_DENSE_TARGET_OCCUPANCY": os.getenv(
                "SPARKINFER_DENSE_TARGET_OCCUPANCY"
            ),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "flush_l2": args.flush_l2,
        "check": not args.no_check,
    }
    print("=== Phase A evidence header ===")
    print(json.dumps(header, indent=2, default=str))
    print("=== end header ===")
    return header


def _resolve_shapes(args: argparse.Namespace) -> list[tuple[str, int, int, list[int]]]:
    """Return list of (label, N, K, M_list)."""
    if args.preset == "behemoth-tp2":
        ms = list(args.m) if args.m else list(DEFAULT_M_SWEEP)
        return [(name, n, k, ms) for name, (n, k) in BEHEMOTH_TP2_SHAPES.items()]
    if args.n is None or args.k is None:
        raise SystemExit("provide --preset behemoth-tp2 or both --n and --k")
    ms = list(args.m) if args.m else [128]
    return [("custom", args.n, args.k, ms)]


TILE_SWEEP_DEFAULT_M = (32, 128, 512, 2048, 8192)
TILE_SWEEP_DECODE_DEFAULT_M = (1, 2, 4, 8, 16)


def _parse_tiles(spec: str) -> list[tuple[int, int]]:
    tiles: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        tm, tn = part.split("x")
        tiles.append((int(tm), int(tn)))
    return tiles


def _run_tile_sweep(args: argparse.Namespace, l2_flush, header: dict) -> None:
    """Item-4 evidence: FP6 expanded-B GEMM time per candidate MMA tile.

    Two arms (``--tile-arm``): ``expanded`` gathers prefill-regime (M > 16)
    data for the wide-N ladder in ``_select_default_mma_tiler_mn`` (which
    moved from the (128,128) pin to the measured (128,64) winner);
    ``packed`` times the decode stream (M <= 16, in-smem expansion) for the
    (16,64)-style decode-tile decision. Any winning tile must be
    M-INDEPENDENT across its regime (one kernel per (N,K) under frozen
    resolution) and BIT-IDENTICAL to the default tile — both are checked
    here (bit-equality via ``torch.equal`` vs the first tile). Unsupported
    tiles report SKIP with the raising error and the sweep continues.
    """
    tiles = _parse_tiles(args.tiles)
    default_ms = (
        TILE_SWEEP_DECODE_DEFAULT_M
        if args.tile_arm == "packed"
        else TILE_SWEEP_DEFAULT_M
    )
    ms = list(args.m) if args.m else list(default_ms)
    if args.preset == "behemoth-tp2":
        shapes = [(name, n, k, ms) for name, (n, k) in BEHEMOTH_TP2_SHAPES.items()]
    else:
        if args.n is None or args.k is None:
            raise SystemExit("provide --preset behemoth-tp2 or both --n and --k")
        shapes = [("custom", args.n, args.k, ms)]

    b_packed = args.tile_arm == "packed"
    print(
        f"\nMX-FP6 W6A8 {args.tile_arm}-B tile sweep (CUDA graph replay)"
    )
    print(
        f"{'shape':12s} {'M':>6s} {'tile':>9s} {'med_us':>10s} "
        f"{'tflops':>8s} {'vs_t0':>8s} {'bit':>5s} {'cos':>8s}"
    )

    rows: list[dict] = []
    mismatches: list[str] = []
    for label, n, k, m_list in shapes:
        for m in m_list:
            operands = _setup_w6a8_operands(m, n, k, seed=args.seed)
            ref_out: Optional[torch.Tensor] = None
            ref_us = float("nan")
            for idx, tile in enumerate(tiles):
                holder: list = []
                arm = _run_fp6_arm(
                    name=f"tile_{tile[0]}x{tile[1]}",
                    m=m,
                    n=n,
                    k=k,
                    operands=operands,
                    b_packed=b_packed,
                    warmup=args.warmup,
                    iters=args.iters,
                    l2_flush=l2_flush,
                    check=not args.no_check,
                    mma_tiler_mn=tile,
                    out_holder=holder,
                )
                tile_s = f"{tile[0]}x{tile[1]}"
                if arm.skipped:
                    print(
                        f"{label:12s} {m:6d} {tile_s:>9s} {'SKIP':>10s} "
                        f"{'':>8s} {'':>8s} {'':>5s}  ({arm.skipped})"
                    )
                    rows.append(
                        {
                            "shape": label,
                            "M": m,
                            "N": n,
                            "K": k,
                            "tile": tile_s,
                            "skipped": arm.skipped,
                        }
                    )
                    continue
                out = holder[0][:, :, 0]
                if idx == 0 or ref_out is None:
                    ref_out = out.clone()
                    ref_us = arm.median_us
                    bit = "ref"
                else:
                    bit = "OK" if torch.equal(out, ref_out) else "DIFF"
                    if bit == "DIFF":
                        mismatches.append(f"{label} M={m} tile={tile_s}")
                vs = (
                    f"{arm.median_us / ref_us:.2f}x"
                    if ref_us == ref_us and ref_us > 0
                    else "n/a"
                )
                cos_s = f"{arm.cosine:.5f}" if arm.cosine is not None else "n/a"
                print(
                    f"{label:12s} {m:6d} {tile_s:>9s} {arm.median_us:10.1f} "
                    f"{_tflops(m, n, k, arm.median_us):8.1f} {vs:>8s} "
                    f"{bit:>5s} {cos_s:>8s}"
                )
                rows.append(
                    {
                        "shape": label,
                        "M": m,
                        "N": n,
                        "K": k,
                        "tile": tile_s,
                        "median_us": arm.median_us,
                        "min_us": arm.min_us,
                        "tflops": _tflops(m, n, k, arm.median_us),
                        "vs_first_tile": vs,
                        "ratio_direction": "ratio>1 = slower_than_first_tile",
                        "bit_exact_vs_first_tile": bit,
                        "cosine": arm.cosine,
                        "raw_ms": arm.raw_ms,
                        "fmt_us": fmt_us(arm.raw_ms),
                    }
                )

    payload = {
        "header": header,
        "mode": "tile_sweep",
        "tile_arm": args.tile_arm,
        "rows": rows,
    }
    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")

    # After the JSON so the evidence survives the failure. A tile only changes
    # which CTA owns which output element, never accumulation order, so a DIFF
    # is a kernel bug and the sweep timings below it are not comparable.
    if mismatches:
        raise SystemExit(
            "tile sweep produced non-bit-identical output vs the first tile: "
            + "; ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("none", "behemoth-tp2"),
        default="none",
        help="Shape preset. behemoth-tp2 = Phase-A Behemoth TP=2 shard matrix.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=None,
        help="M values. Default: Phase-A sweep for --preset, else 128.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-fp8", action="store_true", help="Skip CUTLASS FP8 arm")
    parser.add_argument("--no-packed", action="store_true", help="Skip packed-B arm")
    parser.add_argument("--no-expanded", action="store_true", help="Skip expanded-B arm")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Write full results JSON to this path",
    )
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--tile-sweep",
        action="store_true",
        help=(
            "Item-4 evidence mode: sweep candidate MMA tiles for the FP6 "
            "expanded-B arm (the production prefill path) per shape x M, "
            "asserting cross-tile bit-equality against the policy-default "
            "(128,128) tile."
        ),
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="128x128,64x128,32x128,16x128,128x64,64x64",
        help="Comma-separated MxN tile candidates for --tile-sweep.",
    )
    parser.add_argument(
        "--tile-arm",
        choices=("expanded", "packed"),
        default="expanded",
        help=(
            "Weight arm for --tile-sweep: 'expanded' (prefill path) or "
            "'packed' (the decode M<=16 stream; pair with --m 1 2 4 8 16 "
            "and decode-tile candidates like 16x128,16x64,32x64,32x128)."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    header = _print_evidence_header(args)
    if args.tile_sweep:
        _run_tile_sweep(args, l2_flush, header)
        return
    shapes = _resolve_shapes(args)

    print(
        "\nMX-FP6 W6A8 dense GEMM (CUDA graph replay) — "
        "packed-B vs expanded-B"
        + ("" if args.no_fp8 else " vs optional vLLM CUTLASS FP8")
    )
    print(
        f"{'shape':12s} {'M':>6s} {'N':>6s} {'K':>6s} "
        f"{'arm':12s} {'med_us':>10s} {'tflops':>8s} {'vs_exp':>8s} {'cos':>8s}"
    )

    rows: list[dict] = []
    for label, n, k, ms in shapes:
        for m in ms:
            operands = _setup_w6a8_operands(m, n, k, seed=args.seed)
            arms: list[ArmResult] = []
            if not args.no_expanded:
                arms.append(
                    _run_fp6_arm(
                        name="fp6_expanded",
                        m=m,
                        n=n,
                        k=k,
                        operands=operands,
                        b_packed=False,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                        check=not args.no_check,
                    )
                )
            if not args.no_packed:
                arms.append(
                    _run_fp6_arm(
                        name="fp6_packed",
                        m=m,
                        n=n,
                        k=k,
                        operands=operands,
                        b_packed=True,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                        check=not args.no_check,
                    )
                )
            if not args.no_fp8:
                arms.append(
                    _run_fp8_cutlass_arm(
                        m=m,
                        n=n,
                        k=k,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                        check=not args.no_check,
                    )
                )

            expanded_us = next(
                (a.median_us for a in arms if a.name == "fp6_expanded" and not a.skipped),
                float("nan"),
            )
            for arm in arms:
                if arm.skipped:
                    print(
                        f"{label:12s} {m:6d} {n:6d} {k:6d} {arm.name:12s} "
                        f"{'SKIP':>10s} {'':>8s} {'':>8s}  ({arm.skipped})"
                    )
                    rows.append(
                        {
                            "shape": label,
                            "M": m,
                            "N": n,
                            "K": k,
                            "arm": arm.name,
                            "skipped": arm.skipped,
                        }
                    )
                    continue
                tflops = _tflops(m, n, k, arm.median_us)
                if expanded_us == expanded_us and expanded_us > 0:  # not NaN
                    ratio = arm.median_us / expanded_us
                    # ratio > 1 => this arm slower than expanded FP6
                    vs = f"{ratio:.2f}x"
                    direction = (
                        "slower_than_fp6_expanded"
                        if ratio > 1.0
                        else "faster_than_fp6_expanded"
                    )
                else:
                    vs = "n/a"
                    direction = "n/a"
                cos_s = f"{arm.cosine:.5f}" if arm.cosine is not None else "n/a"
                print(
                    f"{label:12s} {m:6d} {n:6d} {k:6d} {arm.name:12s} "
                    f"{arm.median_us:10.1f} {tflops:8.1f} {vs:>8s} {cos_s:>8s}"
                )
                rows.append(
                    {
                        "shape": label,
                        "M": m,
                        "N": n,
                        "K": k,
                        "arm": arm.name,
                        "median_us": arm.median_us,
                        "min_us": arm.min_us,
                        "tflops": tflops,
                        "vs_fp6_expanded": vs,
                        "ratio_direction": direction,
                        "cosine": arm.cosine,
                        "raw_ms": arm.raw_ms,
                        "fmt_us": fmt_us(arm.raw_ms),
                    }
                )

    # Compact ratio summary: packed/expanded and fp8/expanded at prefill Ms.
    print("\n=== Ratio summary (median_us / fp6_expanded; >1 = slower) ===")
    print(
        f"{'shape':12s} {'M':>6s} {'packed/exp':>12s} {'fp8/exp':>12s} "
        f"{'fp8/packed':>12s}"
    )
    by_key: dict[tuple, dict[str, float]] = {}
    for row in rows:
        if "median_us" not in row:
            continue
        key = (row["shape"], row["M"], row["N"], row["K"])
        by_key.setdefault(key, {})[row["arm"]] = row["median_us"]
    for (shape, m, _n, _k), arms_us in by_key.items():
        exp = arms_us.get("fp6_expanded")
        pk = arms_us.get("fp6_packed")
        fp8 = arms_us.get(FP8_CUTLASS_ARM)
        if fp8 is None:
            fp8 = arms_us.get(FP8_CUBLASLT_ARM)
        def _r(a: Optional[float], b: Optional[float]) -> str:
            if a is None or b is None or b <= 0:
                return "n/a"
            return f"{a / b:.2f}x"

        print(
            f"{shape:12s} {m:6d} {_r(pk, exp):>12s} {_r(fp8, exp):>12s} "
            f"{_r(fp8, pk):>12s}"
        )

    payload = {"header": header, "rows": rows}
    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
