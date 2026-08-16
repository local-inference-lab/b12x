#!/usr/bin/env python3
"""Qualification benchmark for the MXFP4 dense GEMM enablement.

Times CUDA-graph replays of ``gemm.blockscaled.mm`` on three tracks and
emits a raw JSON receipt suitable for a validation/performance record:

- ``nvfp4``: Float4E2M1FN values, Float8E4M3FN scales, sf_vec_size=16.
  Regression arm — the MXFP4 enablement must not change this path.
- ``mxfp8``: Float8E4M3FN values, Float8E8M0FNU scales, sf_vec_size=32.
  Regression arm — shares the non-block-FP8 scale-factor mainloop.
- ``mxfp4``: Float4E2M1FN values, Float8E8M0FNU scales, sf_vec_size=32.
  New capability — fails at MMA op construction before the fix.

Each track fixes its torch seed, so operands are identical across source
revisions and the output SHA-256 digests are directly comparable between
the "before" and "after" arms. Correctness for the FP4 tracks is checked
bit-exactly against a pure-torch dequantized einsum reference.

Run from a checkout of the revision under test, selecting the source tree
explicitly:

    python3 benchmarks/benchmark_mxfp4_dense_gemm.py \
        --source /src/b12x --revision <sha> --label after \
        --tracks nvfp4,mxfp8,mxfp4 --json-out receipt-after.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback


def _gpu_identity(device_index: int) -> dict:
    fields = (
        "name,uuid,pci.bus_id,pstate,clocks.sm,clocks.mem,power.limit,"
        "driver_version,compute_mode,persistence_mode"
    )
    out: dict = {}
    try:
        row = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={fields}",
                    "--format=csv,noheader",
                    "-i",
                    str(device_index),
                ],
                text=True,
            )
            .strip()
            .split(", ")
        )
        out = dict(zip(fields.split(","), row, strict=False))
    except Exception as exc:  # tool differences must not kill the receipt
        out["error"] = repr(exc)
    for key in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        try:
            out["throttle_reasons_active"] = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--query-gpu={key}",
                        "--format=csv,noheader",
                        "-i",
                        str(device_index),
                    ],
                    text=True,
                ).strip()
            )
            break
        except Exception:
            continue
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t) -> str:
    import torch

    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _time_graph_replays(graph, warmup: int, samples: int) -> dict:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    first_replay_us = start.elapsed_time(end) * 1000.0

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    raw_us = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        raw_us.append(start.elapsed_time(end) * 1000.0)

    ordered = sorted(raw_us)
    return {
        "first_replay_us": first_replay_us,
        "warmup_replays": warmup,
        "recorded_samples": samples,
        "median_us": statistics.median(raw_us),
        "mean_us": statistics.fmean(raw_us),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p99_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "raw_samples_us": raw_us,
    }


def _make_mxfp4_operand(shape):
    import torch

    from b12x._lib.intrinsics import (
        FLOAT4_E2M1_MAX,
        MX_SF_VEC_SIZE,
        _ue8m0_output_scale_torch,
        as_grouped_scale_view_mx,
        fp4_quantize_values_torch,
        pack_grouped_fp4_values,
        pow2_ceil_ue8m0_torch,
        swizzle_block_scale,
    )

    groups, rows, cols = shape
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 4
    blocked = source.float().view(
        groups, rows, cols // MX_SF_VEC_SIZE, MX_SF_VEC_SIZE
    )
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    rounded, byte = pow2_ceil_ue8m0_torch(block_max / FLOAT4_E2M1_MAX)
    inv = _ue8m0_output_scale_torch(byte)
    values = fp4_quantize_values_torch(
        (blocked * inv).clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    )
    packed = pack_grouped_fp4_values(values.view(groups, rows, cols))
    swizzled = swizzle_block_scale(byte.squeeze(-1).view(torch.float8_e8m0fnu))
    scale_view = as_grouped_scale_view_mx(swizzled.view(torch.uint8), rows, cols)
    dequantized = (values * rounded).view(groups, rows, cols)
    return (packed, scale_view), dequantized


def run_nvfp4(m: int, n: int, k: int, warmup: int, samples: int) -> dict:
    import torch

    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
    from b12x.gemm import blockscaled

    torch.manual_seed(42)

    def make(shape):
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 4
        row_counts = torch.full(
            (shape[0],), shape[1], dtype=torch.int32, device="cuda"
        )
        amax = source.abs().max().to(torch.float32)
        gs = torch.tensor([448.0 * 6.0 / amax], dtype=torch.float32, device="cuda")
        return quantize_grouped_nvfp4_torch(source, row_counts, gs), gs

    lhs, ls = make((1, m, k))
    rhs, rs = make((1, n, k))
    alpha = (1.0 / (ls[0] * rs[0])).view(1)

    def call(out=None):
        return blockscaled.mm(
            lhs,
            rhs,
            out=out,
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
        )

    eager = call()
    torch.cuda.synchronize()
    out_buf = torch.empty_like(eager)
    call(out=out_buf)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(out=out_buf)
    timing = _time_graph_replays(graph, warmup, samples)
    torch.testing.assert_close(out_buf, eager, rtol=0, atol=0)
    return {
        "shape_mnk": [m, n, k],
        "output_sha256": _sha256_tensor(out_buf),
        "graph_matches_eager_bitexact": True,
        **timing,
    }


def run_mxfp8(m: int, n: int, k: int, groups: int, warmup: int, samples: int) -> dict:
    import torch

    from b12x.gemm import blockscaled
    from b12x.gemm._shared.wo_mxfp8 import (
        pack_fp8_block_scaled_weight_mxfp8,
        quantize_mxfp8_rows_torch,
    )

    torch.manual_seed(29)
    a = torch.randn((m, k, groups), device="cuda", dtype=torch.bfloat16) / 4
    a_q = quantize_mxfp8_rows_torch(a)
    b_values = (
        torch.randn((groups * n, k), device="cuda", dtype=torch.bfloat16) / 32
    ).to(torch.float8_e4m3fn)
    b_scales = torch.ones(
        (groups * (n // 128), k // 128), device="cuda", dtype=torch.float32
    )
    b_q = pack_fp8_block_scaled_weight_mxfp8(
        b_values, b_scales, m=n, k=k, num_groups=groups
    )

    def call():
        return blockscaled.mm(
            (a_q.values, a_q.scale_mma),
            (b_q.values, b_q.scale_mma),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            mma_tiler_mn=(128, 128),
            expected_m=2048,
            sfb_k_replicated=True,
        )

    eager = call()
    torch.cuda.synchronize()
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = call()
    timing = _time_graph_replays(graph, warmup, samples)
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)
    return {
        "shape_mnk": [m, n, k],
        "groups": groups,
        "output_sha256": _sha256_tensor(graph_out),
        "graph_matches_eager_bitexact": True,
        **timing,
    }


def run_mxfp4(m: int, n: int, k: int, warmup: int, samples: int) -> dict:
    import torch

    from b12x.gemm import blockscaled

    torch.manual_seed(7)
    lhs, a_deq = _make_mxfp4_operand((1, m, k))
    rhs, b_deq = _make_mxfp4_operand((1, n, k))

    def call(out=None):
        return blockscaled.mm(
            lhs,
            rhs,
            out=out,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
        )

    eager = call()
    torch.cuda.synchronize()
    ref = torch.einsum("gmk,gnk->mng", a_deq, b_deq).to(torch.bfloat16)
    mismatches = int((eager != ref).sum().item())
    out_buf = torch.empty_like(eager)
    call(out=out_buf)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(out=out_buf)
    timing = _time_graph_replays(graph, warmup, samples)
    graph_mismatches = int((out_buf != ref).sum().item())
    return {
        "shape_mnk": [m, n, k],
        "output_sha256": _sha256_tensor(out_buf),
        "dequant_reference_mismatches_eager": mismatches,
        "dequant_reference_mismatches_graph": graph_mismatches,
        **timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="b12x tree to import")
    parser.add_argument("--revision", required=True, help="git SHA of --source")
    parser.add_argument("--label", required=True, help="arm label (before/after)")
    parser.add_argument("--tracks", default="nvfp4,mxfp8,mxfp4")
    parser.add_argument("--fp4-shape", default="2048,4096,4096")
    parser.add_argument("--mxfp8-shape", default="512,1024,512")
    parser.add_argument("--mxfp8-groups", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.source)
    import torch  # noqa: E402

    import b12x  # noqa: E402

    dense_gemm_path = f"{args.source}/b12x/_lib/dense_gemm.py"
    receipt = {
        "benchmark": "benchmarks/benchmark_mxfp4_dense_gemm.py",
        "command": " ".join(sys.argv),
        "label": args.label,
        "source_tree": args.source,
        "source_revision": args.revision,
        "b12x_module_file": b12x.__file__,
        "dense_gemm_sha256": _sha256_file(dense_gemm_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_before_tracks": _gpu_identity(args.device_index),
        "warmup": args.warmup,
        "samples": args.samples,
        "tracks": {},
    }

    fp4_m, fp4_n, fp4_k = (int(v) for v in args.fp4_shape.split(","))
    m8_m, m8_n, m8_k = (int(v) for v in args.mxfp8_shape.split(","))
    runners = {
        "nvfp4": lambda: run_nvfp4(fp4_m, fp4_n, fp4_k, args.warmup, args.samples),
        "mxfp8": lambda: run_mxfp8(
            m8_m, m8_n, m8_k, args.mxfp8_groups, args.warmup, args.samples
        ),
        "mxfp4": lambda: run_mxfp4(fp4_m, fp4_n, fp4_k, args.warmup, args.samples),
    }
    for track in args.tracks.split(","):
        try:
            receipt["tracks"][track] = {"status": "ok", **runners[track]()}
        except Exception as exc:
            receipt["tracks"][track] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-3:],
            }

    receipt["gpu_after_tracks"] = _gpu_identity(args.device_index)
    with open(args.json_out, "w") as f:
        json.dump(receipt, f, indent=1)
    for track, data in receipt["tracks"].items():
        if data["status"] == "ok":
            print(
                f"{args.label}/{track}: median {data['median_us']:.3f} us "
                f"(min {data['min_us']:.3f}, max {data['max_us']:.3f}, "
                f"n={data['recorded_samples']})"
            )
        else:
            print(f"{args.label}/{track}: {data['error_type']}: "
                  f"{data['error_message'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
