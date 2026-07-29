#!/usr/bin/env python3
"""Attribute a vLLM Torch profiler chrome-trace into kernel/host classes.

Used for Phase A of the FP6 performance recovery plan: split a ctx-8192
prefill and a decode-window capture into GEMM / act-quant / host GS /
allreduce / attention / other so each Phase-B lever has a measured budget.

Accepts ``*.pt.trace.json`` or ``*.pt.trace.json.gz`` (vLLM torch profiler
output). Prints a ranked table and optional JSON.

Examples
--------
::

    python scripts/summarize_vllm_trace.py /tmp/vllm_prof/fp6_prefill/*.pt.trace.json.gz
    python scripts/summarize_vllm_trace.py /tmp/vllm_prof --json-out /tmp/attr.json

Traces only exist if the server was launched with ``PROFILE=1
PROFILE_DIR=/tmp/vllm_prof_<tag>``; point this script at that ``PROFILE_DIR``
(the capture tool's ``--out-dir`` holds request logs, not traces).

Decode overhead attribution (D1) additionally needs the GPU idle timeline, not
just what ran::

    python scripts/summarize_vllm_trace.py rank0.pt.trace.json.gz --gaps \\
        --step-anchor 'lm_head|Sampler|argmax' --json-out /tmp/attr.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import gzip
import json
import os
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

try:  # 5-10x faster parse of multi-GB chrome traces when available
    import orjson as _fast_json  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _fast_json = None


@dataclass
class Bucket:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    cuda_us: float = 0.0
    cpu_us: float = 0.0
    count: int = 0
    examples: list[str] = field(default_factory=list)


# Order matters: first match wins. Keep specific kernels ahead of generic aten.
_BUCKET_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        # FIRST on purpose. Inductor names its fused norm kernels after the op
        # they were fused with, e.g.
        # `triton_red_fused_fp6_dense_linear_fused_add_rms_norm_0`, which the
        # `fp6_dense` GEMM pattern below would otherwise claim (0.6 ms/step of
        # norm time booked as GEMM at Behemoth-123B decode). No real GEMM kernel
        # name contains these tokens.
        "norm_act_misc",
        (
            r"rms_norm",
            r"rmsnorm",
            r"silu",
            r"SwiGLU",
            r"rotary",
            r"rope",
        ),
    ),
    (
        "fp6_gemm",
        (
            r"dense_gemm",
            r"sparkinfer.*fp6",
            r"fp6_dense",
            r"mxfp6",
            r"mxf8f6f4",
            r"PackedB",
            r"b_packed",
        ),
    ),
    (
        "fp6_act_quant",
        (
            r"bf16_to_fp6",
            r"SmallMQuant",
            r"quantize.*fp6",
            r"quantize_block_fp8_e4m3",
            r"fp6.*quant",
        ),
    ),
    (
        "fp8_gemm",
        (
            r"cutlass_scaled_mm",
            r"cutlass.*fp8",
            r"blockwise.*fp8",
            r"fp8.*gemm",
            r"scaled_mm",
            r"cublasLt.*fp8",
            r"nvjet",
        ),
    ),
    (
        "fp8_act_quant",
        (
            r"quantize.*fp8",
            r"per_token.*fp8",
            r"dynamic.*fp8",
            r"fp8_quant",
        ),
    ),
    (
        # Zero-fills and memsets. Phase C removed ~704 baked graph fills per
        # decode step; this bucket exists so a regression is visible instead of
        # hiding in "other".
        "fill_memset",
        (
            r"FillFunctor",
            r"fill_kernel",
            r"Memset",
            r"memset",
        ),
    ),
    (
        "memcpy",
        (
            r"Memcpy",
            r"memcpy",
        ),
    ),
    (
        # Graph replay launches. A decode step that is fully captured shows one
        # (or few) of these; a high count means the step fell back to eager.
        "graph_launch",
        (
            r"cudaGraphLaunch",
            r"GraphLaunch",
            r"cudaGraphKernelNode",
        ),
    ),
    (
        "host_gs_chain",
        (
            r"aten::amax",
            r"aten::_aminmax",
            r"aten::linalg_vector_norm",
            r"aten::mul(\.|_)",
            r"aten::div(\.|_)",
            r"aten::reciprocal",
            r"aten::to(\.|$)",
            r"aten::_to_copy",
            r"aten::copy_",
            r"aten::empty",
            r"aten::zeros",
            r"aten::clamp",
        ),
    ),
    (
        "allreduce",
        (
            r"nccl",
            r"all_reduce",
            r"AllReduce",
            r"vllm::all_reduce",
            r"custom_all_reduce",
        ),
    ),
    (
        "attention",
        (
            r"flash",
            r"FlashInfer",
            r"flashinfer",
            r"fmha",
            r"attention",
            r"cutlass_mha",
            r"cudnn.*attn",
            r"reshape_and_cache",
            r"paged_attention",
            r"unified_attention",
        ),
    ),
    (
        "sampler_logit",
        (
            r"sample",
            r"softmax",
            r"topk",
            r"gather",
            r"lm_head",
        ),
    ),
    (
        # Generic pointwise work left over after the specific buckets above
        # (residual adds, casts, inductor pointwise). Kept last so it never
        # steals a more specific classification.
        "elementwise",
        (
            r"vectorized_elementwise_kernel",
            r"unrolled_elementwise_kernel",
            r"elementwise_kernel",
            r"triton_poi_",
        ),
    ),
)


def _compile_buckets() -> list[Bucket]:
    return [
        Bucket(name=name, patterns=tuple(re.compile(p, re.I) for p in pats))
        for name, pats in _BUCKET_SPECS
    ]


def _open_trace(path: pathlib.Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
    else:
        raw = path.read_bytes()
    if _fast_json is not None:
        return _fast_json.loads(raw)
    return json.loads(raw)


def _iter_events(trace: Any) -> Iterable[dict]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents") or trace.get("events") or []
    elif isinstance(trace, list):
        events = trace
    else:
        events = []
    for ev in events:
        if isinstance(ev, dict):
            yield ev


def _duration_us(ev: dict) -> float:
    # Chrome trace uses microseconds in `dur`. Some exporters stash ns in args.
    if "dur" in ev:
        return float(ev["dur"])
    args = ev.get("args") or {}
    for key in ("dur", "duration", "cuda_time_total", "cpu_time_total"):
        if key in args:
            val = float(args[key])
            # Heuristic: values that look like nanoseconds.
            return val / 1000.0 if val > 1e7 else val
    return 0.0


def _classify(name: str, buckets: list[Bucket]) -> Bucket:
    for bucket in buckets:
        if any(p.search(name) for p in bucket.patterns):
            return bucket
    other = buckets[-1]
    return other


# Checked BEFORE _IS_CUDA_CAT_RE. Torch projects host ranges onto the GPU
# timeline as `gpu_user_annotation` and tags runtime/driver calls `cuda_runtime`
# / `cuda_driver`; all three match the loose "cuda|gpu|kernel" rule below and
# would otherwise be summed as device time. A single per-step
# `gpu_user_annotation` spanning the whole step, plus its `cudaEventSynchronize`,
# is enough to push "other" to ~75% of a decode trace and make `cuda_total_us`
# exceed wall-clock several times over. These are host-side rows: count them as
# CPU.
_IS_HOST_CAT_RE = re.compile(
    r"cuda_runtime|cuda_driver|user_annotation|cpu_op|python|ac2g|fwdbwd", re.I
)
_IS_CUDA_CAT_RE = re.compile(r"cuda|gpu|kernel", re.I)
_IS_CUDA_NAME_RE = re.compile(r"cuda|gemm|nccl|cutlass|triton", re.I)

# STRICTER than _IS_CUDA_CAT_RE: real device-side execution only. The bucket
# totals deliberately keep the looser rule (it also sweeps in cuda_runtime
# rows), but a timeline must not count host-side API calls as GPU occupancy or
# every launch would look like it overlapped the kernel it launched.
_IS_GPU_EXEC_CAT_RE = re.compile(r"^kernel$|gpu_memcpy|gpu_memset", re.I)


def _timeline_report(
    spans_by_stream: dict[tuple[str, str], list[tuple[float, float, str]]],
    *,
    gap_min_us: float,
    gap_top: int,
) -> dict[str, Any]:
    """Busy/idle split and the largest idle gaps, per GPU stream.

    Bucket sums answer "what ran"; they cannot answer "what did the GPU do
    nothing during", which is exactly the FP6-vs-FP8 decode question (FP6 spends
    ~5.6 ms/step outside its GEMMs against FP8's ~2 ms). For each stream we
    merge overlapping kernel intervals, then report span, busy, idle, and the
    biggest holes with the kernels on either side — the boundary names are the
    actionable part, since they name the launch that stalled.
    """
    streams: list[dict[str, Any]] = []
    gap_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (pid, tid), spans in spans_by_stream.items():
        if not spans:
            continue
        spans.sort(key=lambda s: s[0])
        span_start = spans[0][0]
        span_end = max(s[1] for s in spans)
        busy = 0.0
        gaps: list[tuple[float, str, str]] = []
        cur_start, cur_end, cur_name = spans[0]
        for start, end, name in spans[1:]:
            if start > cur_end:
                busy += cur_end - cur_start
                hole = start - cur_end
                if hole >= gap_min_us:
                    gaps.append((hole, cur_name, name))
                    gap_pairs[(cur_name, name)].append(hole)
                cur_start, cur_end, cur_name = start, end, name
            else:
                if end > cur_end:
                    cur_end, cur_name = end, name
        busy += cur_end - cur_start
        total = span_end - span_start
        gaps.sort(key=lambda g: g[0], reverse=True)
        streams.append(
            {
                "pid": pid,
                "tid": tid,
                "span_us": total,
                "busy_us": busy,
                "idle_us": total - busy,
                "idle_pct": (100.0 * (total - busy) / total) if total else 0.0,
                "kernel_count": len(spans),
                "gap_count": len(gaps),
                "top_gaps": [
                    {"gap_us": g, "after": a[:90], "before": b[:90]}
                    for g, a, b in gaps[:gap_top]
                ],
            }
        )
    streams.sort(key=lambda s: s["busy_us"], reverse=True)
    aggregated = sorted(
        (
            {
                "after": a[:90],
                "before": b[:90],
                "total_us": sum(v),
                "count": len(v),
                "mean_us": sum(v) / len(v),
            }
            for (a, b), v in gap_pairs.items()
        ),
        key=lambda d: d["total_us"],
        reverse=True,
    )[:gap_top]
    return {"streams": streams, "top_gap_boundaries": aggregated}


def summarize_trace(
    path: pathlib.Path,
    *,
    gaps: bool = False,
    gap_min_us: float = 20.0,
    gap_top: int = 15,
    step_anchor: str = "",
) -> dict[str, Any]:
    buckets = _compile_buckets()
    other = Bucket(name="other", patterns=())
    buckets.append(other)

    trace = _open_trace(path)
    # Timeline collection is opt-in: it retains one tuple per GPU event, and a
    # multi-hundred-MB decode trace holds millions of them.
    spans_by_stream: dict[tuple[str, str], list[tuple[float, float, str]]] = (
        defaultdict(list)
    )
    gpu_cat_memo: dict[str, bool] = {}
    anchor_re = re.compile(step_anchor, re.I) if step_anchor else None
    anchor_count = 0
    cuda_total = 0.0
    cpu_total = 0.0
    by_name_cuda: dict[str, float] = defaultdict(float)
    by_name_count: dict[str, int] = defaultdict(int)
    # Kernel names repeat millions of times per trace; classify each unique
    # (name, cat-class) once instead of running every bucket regex per event.
    bucket_memo: dict[str, Bucket] = {}
    cuda_name_memo: dict[str, bool] = {}
    cuda_cat_memo: dict[str, str] = {}

    for ev in _iter_events(trace):
        # X = complete events; also accept C++ CUDA kernels tagged similarly.
        if ev.get("ph") not in ("X", "x"):
            continue
        name = str(ev.get("name") or "")
        if not name:
            continue
        cat = str(ev.get("cat") or "")
        dur = _duration_us(ev)
        if dur <= 0:
            continue
        kind = cuda_cat_memo.get(cat)
        if kind is None:
            if _IS_HOST_CAT_RE.search(cat):
                kind = "host"
            elif _IS_CUDA_CAT_RE.search(cat):
                kind = "gpu"
            else:
                kind = "unknown"
            cuda_cat_memo[cat] = kind
        if kind == "unknown":
            # Untagged rows: fall back to the kernel-name heuristic.
            name_hit = cuda_name_memo.get(name)
            if name_hit is None:
                name_hit = bool(_IS_CUDA_NAME_RE.search(name))
                cuda_name_memo[name] = name_hit
            is_cuda = name_hit
        else:
            is_cuda = kind == "gpu"
        bucket = bucket_memo.get(name)
        if bucket is None:
            bucket = _classify(name, buckets)
            bucket_memo[name] = bucket
        if is_cuda:
            bucket.cuda_us += dur
            cuda_total += dur
            by_name_cuda[name] += dur
            by_name_count[name] += 1
        else:
            bucket.cpu_us += dur
            cpu_total += dur
        if len(bucket.examples) < 5 and name not in bucket.examples:
            bucket.examples.append(name)
        bucket.count += 1

        if gaps or anchor_re is not None:
            gpu_hit = gpu_cat_memo.get(cat)
            if gpu_hit is None:
                gpu_hit = bool(_IS_GPU_EXEC_CAT_RE.search(cat))
                gpu_cat_memo[cat] = gpu_hit
            if gpu_hit:
                if anchor_re is not None and anchor_re.search(name):
                    anchor_count += 1
                if gaps:
                    ts = float(ev.get("ts") or 0.0)
                    if ts > 0.0:
                        spans_by_stream[
                            (str(ev.get("pid")), str(ev.get("tid")))
                        ].append((ts, ts + dur, name))

    top_kernels = sorted(by_name_cuda.items(), key=lambda kv: kv[1], reverse=True)[:40]
    timeline = (
        _timeline_report(spans_by_stream, gap_min_us=gap_min_us, gap_top=gap_top)
        if gaps
        else None
    )
    return {
        "path": str(path),
        "timeline": timeline,
        "step_anchor": step_anchor,
        "step_anchor_count": anchor_count,
        "cuda_total_us": cuda_total,
        "cpu_total_us": cpu_total,
        "buckets": [
            {
                "name": b.name,
                "cuda_us": b.cuda_us,
                "cuda_pct": (100.0 * b.cuda_us / cuda_total) if cuda_total else 0.0,
                "cpu_us": b.cpu_us,
                "count": b.count,
                "examples": b.examples,
            }
            for b in buckets
            if b.cuda_us > 0 or b.cpu_us > 0 or b.count > 0
        ],
        "top_cuda_kernels": [
            {"name": n, "cuda_us": us, "count": by_name_count[n]}
            for n, us in top_kernels
        ],
    }


# The API-server/frontend process writes its own trace alongside the per-rank
# GPU traces (``<host>_<pid>.async_llm.*.pt.trace.json.gz``). It contains no
# device work, so including it only costs parse time and dilutes the tables.
_FRONTEND_TRACE_RE = re.compile(r"async_llm|frontend|api_server", re.I)


def _find_traces(
    root: pathlib.Path, include_frontend: bool = False
) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    found: list[pathlib.Path] = []
    for pat in ("*.pt.trace.json.gz", "*.pt.trace.json"):
        found.extend(root.rglob(pat))
    if not found:
        # Fallback: any chrome-trace-looking gzip JSON under the tree.
        for pat in ("*.json.gz", "*.json"):
            found.extend(
                p
                for p in root.rglob(pat)
                if "trace" in p.name.lower() or "chrome" in p.name.lower()
            )
    if not include_frontend:
        kept = [p for p in found if not _FRONTEND_TRACE_RE.search(p.name)]
        # Only drop the frontend traces if per-rank traces actually exist; a
        # directory holding nothing else should still summarize rather than
        # report "no trace files found".
        if kept:
            found = kept
    return sorted({p.resolve() for p in found})


def _print_summary(summary: dict[str, Any], steps: int = 0) -> None:
    print(f"\n=== {summary['path']} ===")
    cuda_ms = summary["cuda_total_us"] / 1000.0
    cpu_ms = summary["cpu_total_us"] / 1000.0
    print(f"CUDA total: {cuda_ms:.2f} ms    CPU total: {cpu_ms:.2f} ms")
    if summary.get("step_anchor"):
        print(
            f"step anchor /{summary['step_anchor']}/ matched "
            f"{summary['step_anchor_count']} GPU event(s)"
        )
    n = steps or summary.get("step_anchor_count") or 0
    per_step = f" {'ms/step':>10s}" if n else ""
    print(
        f"{'bucket':18s} {'cuda_ms':>10s} {'cuda_%':>8s} {'cpu_ms':>10s} "
        f"{'count':>8s}{per_step}"
    )
    for b in sorted(summary["buckets"], key=lambda x: x["cuda_us"], reverse=True):
        tail = f" {b['cuda_us']/1000/n:10.3f}" if n else ""
        print(
            f"{b['name']:18s} {b['cuda_us']/1000:10.2f} {b['cuda_pct']:7.1f}% "
            f"{b['cpu_us']/1000:10.2f} {b['count']:8d}{tail}"
        )
    print("\nTop CUDA kernels:")
    for row in summary["top_cuda_kernels"][:20]:
        print(
            f"  {row['cuda_us']/1000:10.2f} ms  x{row['count']:<5d}  {row['name'][:100]}"
        )
    _print_timeline(summary, n)


def _print_timeline(summary: dict[str, Any], steps: int) -> None:
    timeline = summary.get("timeline")
    if not timeline:
        return
    print("\nGPU stream occupancy (device-execution events only):")
    print(
        f"{'pid/tid':>14s} {'span_ms':>10s} {'busy_ms':>10s} {'idle_ms':>10s} "
        f"{'idle_%':>8s} {'kernels':>9s}"
    )
    for s in timeline["streams"][:6]:
        print(
            f"{s['pid'] + '/' + s['tid']:>14s} {s['span_us']/1000:10.2f} "
            f"{s['busy_us']/1000:10.2f} {s['idle_us']/1000:10.2f} "
            f"{s['idle_pct']:7.1f}% {s['kernel_count']:9d}"
        )
    if timeline["streams"]:
        busiest = timeline["streams"][0]
        if steps:
            print(
                f"  busiest stream per step: busy {busiest['busy_us']/1000/steps:.3f} "
                f"ms, idle {busiest['idle_us']/1000/steps:.3f} ms"
            )
    print("\nLargest idle gaps by kernel boundary (after -> before):")
    for g in timeline["top_gap_boundaries"]:
        tail = f" {g['total_us']/1000/steps:8.3f} ms/step" if steps else ""
        print(
            f"  {g['total_us']/1000:9.2f} ms  x{g['count']:<6d} "
            f"mean {g['mean_us']:7.1f} us{tail}\n"
            f"      after: {g['after']}\n     before: {g['before']}"
        )


def _self_test() -> None:
    """Offline smoke: classify a tiny synthetic chrome trace without GPU."""
    import tempfile

    # ts/tid chosen so stream 7 has one 500us hole between the GEMM and the
    # allreduce, and the sampler anchor fires twice (two "steps").
    fake = {
        "traceEvents": [
            {"ph": "X", "name": "sparkinfer::fp6_dense_linear", "cat": "cpu_op", "dur": 100.0},
            {"ph": "X", "name": "dense_gemm_mxfp6", "cat": "kernel", "dur": 5000.0,
             "ts": 1000.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "bf16_to_fp6_tma", "cat": "kernel", "dur": 200.0,
             "ts": 6000.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "aten::amax", "cat": "cpu_op", "dur": 50.0},
            {"ph": "X", "name": "ncclAllReduce", "cat": "kernel", "dur": 800.0,
             "ts": 6700.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "flash_attn_varlen", "cat": "kernel", "dur": 1200.0,
             "ts": 7500.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "cutlass_scaled_mm", "cat": "kernel", "dur": 3000.0,
             "ts": 8700.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "at::native::FillFunctor<unsigned char>", "cat": "kernel",
             "dur": 40.0, "ts": 11700.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "Sampler_argmax", "cat": "kernel", "dur": 60.0,
             "ts": 11740.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "Sampler_argmax", "cat": "kernel", "dur": 60.0,
             "ts": 11800.0, "pid": 0, "tid": 7},
            {"ph": "X", "name": "cudaGraphLaunch", "cat": "cuda_runtime", "dur": 15.0},
            {"ph": "X", "name": "triton_red_fused_fp6_dense_linear_fused_add_rms_norm_0",
             "cat": "kernel", "dur": 300.0, "ts": 11860.0, "pid": 0, "tid": 7},
            # Host range projected onto the GPU timeline: must NOT be counted as
            # device time even though its category matches /gpu/.
            {"ph": "X", "name": "execute_context_0(0)_generation_1(1)",
             "cat": "gpu_user_annotation", "dur": 999999.0, "ts": 1000.0,
             "pid": 0, "tid": 7},
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "fake.pt.trace.json"
        path.write_text(json.dumps(fake), encoding="utf-8")
        summary = summarize_trace(
            path, gaps=True, gap_min_us=100.0, step_anchor="Sampler|argmax"
        )
    by_name = {b["name"]: b for b in summary["buckets"]}
    assert by_name["fp6_gemm"]["cuda_us"] == 5000.0
    assert by_name["fp6_act_quant"]["cuda_us"] == 200.0
    assert by_name["allreduce"]["cuda_us"] == 800.0
    assert by_name["attention"]["cuda_us"] == 1200.0
    assert by_name["fp8_gemm"]["cuda_us"] == 3000.0
    assert by_name["host_gs_chain"]["cpu_us"] == 50.0
    assert by_name["fill_memset"]["cuda_us"] == 40.0
    assert by_name["graph_launch"]["count"] == 1
    assert summary["step_anchor_count"] == 2
    # Fused norm stays out of the GEMM bucket.
    assert by_name["norm_act_misc"]["cuda_us"] == 300.0
    assert by_name["fp6_gemm"]["cuda_us"] == 5000.0
    # The host annotation is booked as CPU and never inflates device totals.
    assert by_name["other"]["cuda_us"] == 0.0
    assert by_name["other"]["cpu_us"] == 999999.0
    assert summary["cuda_total_us"] == 10660.0

    stream = summary["timeline"]["streams"][0]
    # 1000 -> 12160 span, with a single 500us hole after bf16_to_fp6_tma. The
    # 999999us gpu_user_annotation must not appear here either.
    assert stream["span_us"] == 11160.0
    assert stream["idle_us"] == 500.0
    assert stream["gap_count"] == 1
    # Frontend traces are filtered out only when per-rank traces are present.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "rank0.123.pt.trace.json.gz").write_bytes(b"")
        (root / "host_1.async_llm.456.pt.trace.json.gz").write_bytes(b"")
        assert [p.name for p in _find_traces(root)] == ["rank0.123.pt.trace.json.gz"]
        assert len(_find_traces(root, include_frontend=True)) == 2
        (root / "rank0.123.pt.trace.json.gz").unlink()
        assert len(_find_traces(root)) == 1

    boundary = summary["timeline"]["top_gap_boundaries"][0]
    assert boundary["after"] == "bf16_to_fp6_tma"
    assert boundary["before"] == "ncclAllReduce"
    assert boundary["total_us"] == 500.0
    _print_summary(summary)
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Trace file(s) or directories containing vLLM torch profiler output",
    )
    parser.add_argument("--json-out", default="", help="Write combined JSON summary")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel worker processes (one per trace file). "
            "0 = min(number of traces, CPU count). 1 = serial."
        ),
    )
    parser.add_argument(
        "--include-frontend",
        action="store_true",
        help=(
            "Also summarize the async_llm/api_server trace, which is skipped by "
            "default because it holds no device work"
        ),
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help=(
            "Also report GPU busy/idle split and the largest idle gaps per "
            "stream. Costs memory proportional to the GPU event count."
        ),
    )
    parser.add_argument(
        "--gap-min-us",
        type=float,
        default=20.0,
        help="Ignore idle gaps shorter than this (default 20us)",
    )
    parser.add_argument(
        "--gap-top", type=int, default=15, help="How many gaps/boundaries to show"
    )
    parser.add_argument(
        "--step-anchor",
        default="",
        help=(
            "Regex matched against GPU kernel names; the match count is used as "
            "the decode step count for ms/step normalization "
            "(e.g. 'lm_head|Sampler|argmax')."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Explicit step count for ms/step normalization (overrides anchor)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline classification smoke test and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.paths:
        parser.error("paths required unless --self-test")

    traces: list[pathlib.Path] = []
    for raw in args.paths:
        traces.extend(_find_traces(pathlib.Path(raw), args.include_frontend))
    if not traces:
        raise SystemExit("no trace files found")

    workers = args.workers or min(len(traces), os.cpu_count() or 1)
    if _fast_json is None:
        print(
            "note: orjson not installed; JSON parse is the slow step. "
            "`pip install orjson` for a large speedup.",
            flush=True,
        )
    print(f"summarizing {len(traces)} trace(s) with {workers} worker(s)...", flush=True)
    run = functools.partial(
        summarize_trace,
        gaps=args.gaps,
        gap_min_us=args.gap_min_us,
        gap_top=args.gap_top,
        step_anchor=args.step_anchor,
    )
    if workers <= 1 or len(traces) == 1:
        summaries = [run(p) for p in traces]
    else:
        # One process per trace: each parses its own multi-GB JSON on its own
        # core. Order of results matches the input order.
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(run, traces))
    for s in summaries:
        _print_summary(s, args.steps)

    # Cross-trace budget table when exactly two traces (e.g. fp6 vs fp8).
    if len(summaries) == 2:
        print("\n=== Side-by-side CUDA bucket ms ===")
        names = sorted(
            {
                b["name"]
                for s in summaries
                for b in s["buckets"]
            }
        )
        a, b = summaries
        print(f"{'bucket':18s} {pathlib.Path(a['path']).name[:22]:>22s} "
              f"{pathlib.Path(b['path']).name[:22]:>22s} {'ratio_a/b':>10s}")
        a_map = {x["name"]: x["cuda_us"] for x in a["buckets"]}
        b_map = {x["name"]: x["cuda_us"] for x in b["buckets"]}
        for name in names:
            au = a_map.get(name, 0.0) / 1000.0
            bu = b_map.get(name, 0.0) / 1000.0
            ratio = f"{au / bu:.2f}x" if bu > 0 else ("inf" if au > 0 else "n/a")
            print(f"{name:18s} {au:22.2f} {bu:22.2f} {ratio:>10s}")

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
