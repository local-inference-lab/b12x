"""Measure complete FP6 linear graph replay against a BF16 Torch projection.

The FP6 arm includes activation quantization and output correction. The BF16
arm uses the original BF16 weights. Their numerical contracts differ; every
FP6 result is checked against an independently quantized oracle before timing.
"""

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.benchmark_blockscaled_precision import _capture, _check, _clock_checks, _paired, _snapshot
from benchmarks.common import make_l2_flush_fn
from b12x.quantization.mxfp6 import allocate_fp6_linear_workspace, dense_fp6_linear, quantize_dense_weight_to_fp6
from scripts._sm103_source import package_source_sha256, source_identity
from tests.quantization.test_fp6_workspace import decode, oracle, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 128])
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--act-fmt", choices=("e2m3", "e3m2", "e4m3"), default="e4m3")
    parser.add_argument("--weight-fmt", choices=("e2m3", "e3m2"), default="e2m3")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) <= 0 or max(args.rows) > args.capacity or args.samples < 10 or args.warmup < 1:
        parser.error("positive live rows must fit capacity; require warmup and at least ten samples")
    if torch.cuda.get_device_capability() != (10, 3):
        parser.error("this deferred benchmark requires physical SM103")
    before = _snapshot()
    identity = dict(zip(before["fields"], before["values"], strict=True))
    if identity["uuid"] != args.device_uuid:
        parser.error("selected GPU does not match --device-uuid")
    record = dict(status="running", command=sys.argv, **source_identity(ROOT),
                  package_python_sha256=package_source_sha256(ROOT), initial_snapshot=before,
                  toolchain={name: importlib.metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl")},
                  gpu_mode=subprocess.check_output(["nvidia-smi", f"--id={args.device_uuid}",
                      "--query-gpu=driver_version,compute_mode", "--format=csv,noheader"], text=True).strip(),
                  geometry=dict(n=args.n, k=args.k, capacity=args.capacity, act_fmt=args.act_fmt, weight_fmt=args.weight_fmt),
                  ratio_direction="FP6 quantize+GEMM+correction / BF16 Torch projection; lower favors FP6",
                  correctness=[], measurements=[])
    torch.manual_seed(421)
    source = torch.randn(args.capacity, args.k, device="cuda", dtype=torch.bfloat16)
    original_weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    weight = replace(quantize_dense_weight_to_fp6(original_weight, source_format=args.weight_fmt), act_fmt=args.act_fmt)
    workspace = allocate_fp6_linear_workspace(args.capacity, args.k, act_fmt=args.act_fmt)
    output = torch.empty(args.capacity, args.n, device="cuda", dtype=torch.bfloat16)
    baseline = torch.empty_like(output)
    r = torch.arange(args.n, device="cuda")[:, None]
    g = torch.arange(args.k // 32, device="cuda")[None, :]
    offset = (r // 128) * (args.k // 128) * 512 + (g // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + g % 4
    decoded_weight = decode(unpack(weight.packed.reshape(args.n, -1)), weight.fmt)
    decoded_weight *= torch.exp2(weight.scale_storage.flatten()[offset].float() - 127).repeat_interleave(32, -1)
    flush = make_l2_flush_fn(enabled=True)
    graphs = []
    for m in args.rows:
        def fp6(m=m):
            dense_fp6_linear(source[:m], weight, out=output[:m], workspace=workspace, expected_m=args.capacity)
        def bf16(m=m):
            torch.mm(source[:m], original_weight.T, out=baseline[:m])
        fp6()
        codes, sf, _, inverse, alpha = oracle(source[:m], args.act_fmt, True, weight.global_scale)
        decoded = decode(codes, args.act_fmt) * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)
        expected = ((decoded @ decoded_weight.T * alpha).to(torch.bfloat16).float() * inverse[:, None].float()).to(torch.bfloat16)
        correctness = _check(output[:m], expected, f"FP6 M={m}")
        arms = {"fp6": _capture(fp6), "bf16": _capture(bf16)}
        output.fill_(float("nan"))
        arms["fp6"].replay()
        _check(output[:m], expected, f"FP6 replay M={m}")
        record["correctness"].append(dict(rows=m, **correctness, graph_replay=True))
        graphs.append((m, arms))
    for _, arms in graphs:
        for _ in range(args.warmup):
            for graph in arms.values():
                graph.replay()
    torch.cuda.synchronize()
    record["before"] = _snapshot()
    record["workspace_addresses"] = [t.data_ptr() for t in (
        source, weight.packed, weight.scale_storage, weight.global_scale, output,
        workspace.values, workspace.scale_storage, workspace.global_scales,
        workspace.inverse_scales, workspace.alpha)]
    for m, arms in graphs:
        for mode, flush_fn in (("warm", None), ("cold", flush)):
            samples = _paired(arms, args.warmup, args.samples, flush_fn)
            ratio = statistics.median(row["fp6"] for row in samples) / statistics.median(row["bf16"] for row in samples)
            record["measurements"].append(dict(rows=m, mode=mode, raw_microseconds=samples, median_ratio=ratio))
    record["after"] = _snapshot()
    record["clock_checks"] = _clock_checks(record["before"], record["after"])
    record["status"] = "measured" if record["clock_checks"]["valid"] else "invalid_gpu_mode"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] != "measured":
        raise SystemExit("GPU mode changed or failed qualification; raw samples retained as invalid")


if __name__ == "__main__":
    main()
