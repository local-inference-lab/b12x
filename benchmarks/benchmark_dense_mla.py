#!/usr/bin/env python3
"""Graph-replay benchmark for the real native K3 dense MLA path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkinfer.attention import dense_mla

FP8 = torch.float8_e4m3fn


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "sparkinfer/attention/dense_mla").glob("*.py")),
        Path(__file__).resolve(),
    ]
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--mode", choices=("decode", "extend"), default="decode")
    parser.add_argument("--query-rows", type=int, default=1)
    parser.add_argument("--cache-tokens", type=int, default=65536)
    parser.add_argument("--page-size", type=int, default=944)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise SystemExit(f"SM120/SM121 required, got {capability}")
    if args.mode == "decode" and args.query_rows != 1:
        raise SystemExit("decode benchmark requires --query-rows 1")
    if args.query_rows <= 0 or args.cache_tokens < args.query_rows:
        raise SystemExit("cache must include every query row")

    torch.manual_seed(args.seed)
    dtype = FP8 if args.dtype == "fp8" else torch.bfloat16
    pages = (args.cache_tokens + args.page_size - 1) // args.page_size
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode=args.mode,
            kv_dtype=dtype,
            num_q_heads=args.heads,
            page_size=args.page_size,
            max_total_q=args.query_rows,
            max_batch=1,
            max_cache_tokens=args.cache_tokens,
            max_page_table_width=pages,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(
        spec.shape,
        dtype=spec.dtype,
        device=spec.device,
    )
    q_float = (
        torch.randn(
            args.query_rows,
            args.heads,
            576,
            device=device,
        )
        * 0.1
    )
    cache_float = (
        torch.randn(
            pages,
            args.page_size,
            576,
            device=device,
        )
        * 0.1
    )
    q_scale = None
    kv_scale = None
    if dtype == FP8:
        q_scale = (q_float.abs().max() / 400).reshape(1).float()
        kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
        q = (q_float / q_scale).to(FP8)
        cache = (cache_float / kv_scale).to(FP8)
    else:
        q = q_float.to(torch.bfloat16)
        cache = cache_float.to(torch.bfloat16)
    page_table = (
        torch.randperm(
            pages,
            dtype=torch.int64,
            device=device,
        )
        .to(torch.int32)
        .reshape(1, pages)
    )
    cache_seqlens = torch.tensor(
        [args.cache_tokens],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor(
        [0, args.query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        args.query_rows,
        args.heads,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    dense_mla.compile(binding=binding)
    for _ in range(args.warmup):
        actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(
        actual_output.float().flatten(),
        expected_output.float().flatten(),
        dim=0,
    )
    output_max_error = (actual_output.float() - expected_output.float()).abs().max()
    lse_max_error = (actual_lse - expected_lse).abs().max()
    correctness = bool(
        torch.isfinite(actual_output).all()
        and torch.isfinite(actual_lse).all()
        and torch.count_nonzero(actual_output) == actual_output.numel()
        and cosine > 0.999
        and lse_max_error < 2e-5
    )
    if not correctness:
        raise RuntimeError(
            "correctness gate failed: "
            f"cos={float(cosine):.8f}, "
            f"output_max_error={float(output_max_error):.8g}, "
            f"lse_max_error={float(lse_max_error):.8g}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)) / args.iterations)
    allocated_after = torch.cuda.memory_allocated(device)
    props = torch.cuda.get_device_properties(device)
    result = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "dense_mla_source_sha256": _source_sha256(),
        "worktree": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": props.name,
        "device_uuid": f"GPU-{props.uuid}",
        "capability": list(capability),
        "multiprocessor_count": int(props.multi_processor_count),
        "shared_memory_per_block_optin": int(props.shared_memory_per_block_optin),
        "gpu_mode": "CUDA graph replay; clocks/power queried externally",
        "dtype": args.dtype,
        "mode": args.mode,
        "query_rows": args.query_rows,
        "heads": args.heads,
        "cache_tokens": args.cache_tokens,
        "page_size": args.page_size,
        "query_tile": plan.query_tile,
        "num_splits": plan.num_splits,
        "chunks_per_split": plan.chunks_per_split,
        "correctness": {
            "passed": correctness,
            "cosine": float(cosine),
            "output_max_error": float(output_max_error),
            "lse_max_error": float(lse_max_error),
            "finite": bool(torch.isfinite(actual_output).all()),
            "nonzero": int(torch.count_nonzero(actual_output)),
        },
        "allocation_stable": allocated_before == allocated_after,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "raw_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
