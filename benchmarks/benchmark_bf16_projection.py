#!/usr/bin/env python3
"""Race retained native SIMT and 32x64x64 MMA BF16 projections, not dispatch.

Example (run on an explicitly isolated GPU):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_bf16_projection.py \
    --cases wo-a,compressor --rows 8,9,16,24 --output /tmp/projection.json

Use --rows all for 1/6/8/9/16/24/64/256/4096, --output-dtypes bf16,fp32,
--layouts contiguous,row-strided, and --cases vocab --vocab-width 32768 for
expanded diagnosis. Large-M SIMT and vocabulary sweeps are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as compile_native,
    run_compiled,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import current_cuda_stream
from b12x.gemm.bf16_gemv import _kernel as native
from b12x.gemm.bf16_gemv import query_from_call
from b12x.gemm.bf16_gemv._tuning import TUNING
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
    resolve_l2_flush_bytes,
)

CASES = {
    "wo-a": (1024, 4096),
    "compressor": (512, 5120),
    "dspark-aux": (5120, 15360),
    "router": (256, 5120),
}
ALL_ROWS = (1, 6, 8, 9, 16, 24, 64, 256, 4096)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ARMS = ("simt", "mma")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file outside the repository",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help="Comma-separated wo-a,compressor,dspark-aux,router,vocab",
    )
    parser.add_argument(
        "--rows",
        default="1,8,9,16,24,256",
        help="Comma-separated positive M values, or all",
    )
    parser.add_argument(
        "--output-dtypes", default="bf16", help="bf16, fp32, or bf16,fp32"
    )
    parser.add_argument(
        "--layouts",
        default="contiguous",
        help="contiguous,row-strided; strided rows have 16 padding elements",
    )
    parser.add_argument("--vocab-width", type=int, default=32768)
    parser.add_argument("--vocab-hidden", type=int, default=5120)
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Even number of paired blocks; alternate AB/BA",
    )
    parser.add_argument(
        "--replays", type=int, default=8, help="Graph samples per arm per paired block"
    )
    parser.add_argument(
        "--launches",
        type=int,
        default=1,
        help="Native calls per graph replay; samples normalized per call",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Calls per arm before capture"
    )
    parser.add_argument(
        "--cache",
        choices=("warm", "cold"),
        default="warm",
        help="Cold flushes L2 before each replay, outside timing",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="0 selects common helper's automatic L2 flush size",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Benchmark checkpoint projections through prepared APIs",
    )
    args = parser.parse_args()
    for name, choices in (
        ("cases", (*CASES, "vocab")),
        ("output_dtypes", DTYPES),
        ("layouts", ("contiguous", "row-strided")),
    ):
        values = getattr(args, name).split(",")
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in choices for value in values)
        ):
            parser.error(
                f"invalid or duplicate --{name.replace('_', '-')}: choose from {','.join(choices)}"
            )
        setattr(args, name, values)
    try:
        args.rows = (
            list(ALL_ROWS)
            if args.rows == "all"
            else [int(value) for value in args.rows.split(",")]
        )
    except ValueError:
        parser.error("--rows must contain integers or be all")
    if (
        not args.rows
        or any(value < 1 or value >= 2**31 for value in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("--rows must be distinct positive int32 values")
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be positive and even for balanced arm ordering")
    if min(args.replays, args.launches, args.warmup) < 1 or args.l2_flush_bytes < 0:
        parser.error(
            "--replays, --launches, --warmup must be positive; flush bytes nonnegative"
        )
    if min(args.vocab_width, args.vocab_hidden) < 16:
        parser.error("vocabulary N and K must be at least 16 for the native MMA arm")
    if args.cache == "cold" and args.launches != 1:
        parser.error(
            "cold-L2 requires --launches 1 so every measured projection follows eviction"
        )
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def toolchain():
    versions = {}
    for name in (
        "torch",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "cuda-bindings",
        "triton",
        "b12x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "torch_git": torch.version.git_version,
    }


def oracle(x, weight):
    # Full FP32 contraction of the actual BF16 operands, never a timed arm.
    # Column chunking bounds temporary FP32 weights even for vocabulary widths.
    expected = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )
    x32 = x.float()
    for start in range(0, weight.shape[0], 512):
        stop = min(start + 512, weight.shape[0])
        expected[:, start:stop] = x32 @ weight[start:stop].float().T
    torch.cuda.synchronize()
    if (
        not torch.isfinite(expected).all().item()
        or not (expected.abs().amax(dim=1) > 0).all().item()
    ):
        raise AssertionError(
            "FP32 oracle must be finite with nonzero output in every row"
        )
    return expected


def check_output(out, expected, label):
    actual = out.float()
    finite = bool(torch.isfinite(actual).all().item())
    nonzero_rows = bool((actual.abs().amax(dim=1) > 0).all().item())
    if not finite or not nonzero_rows:
        raise AssertionError(
            f"{label}: output is nonfinite, unwritten, or has an all-zero row"
        )
    # BF16 rounding is part of the output contract, not an FP32 accumulation error.
    target = expected.to(out.dtype).float()
    rtol, atol = (0.0, 5e-5) if out.dtype == torch.bfloat16 else (2e-4, 3e-5)
    if out.dtype == torch.bfloat16:
        # Independent FP32 summation orders can straddle a BF16 midpoint.
        # Permit one representable BF16 step, plus a small cancellation floor.
        ulps = (
            out.view(torch.int16).int() - expected.to(out.dtype).view(torch.int16).int()
        ).abs()
        bad = (ulps > 1) & ((actual - target).abs() > atol)
        if bad.any().item():
            raise AssertionError(
                f"{label}: {bad.sum().item()} outputs exceed one BF16 ULP "
                f"and the {atol} cancellation floor"
            )
    else:
        torch.testing.assert_close(
            actual, target, rtol=rtol, atol=atol, msg=lambda msg: f"{label}: {msg}"
        )
    error = actual - expected
    return {
        "finite": finite,
        "every_row_nonzero": nonzero_rows,
        "all_elements_checked": out.numel(),
        "rtol": rtol,
        "atol": atol,
        "bf16_ulp_budget": 1 if out.dtype == torch.bfloat16 else None,
        "oracle": "FP32 x @ weight.T with TF32 disabled; BF16 allows one ULP plus cancellation floor",
        "max_abs_error_vs_fp32": float(error.abs().max().item()),
        "relative_rms_error_vs_fp32": float(
            (error.square().mean() / expected.square().mean()).sqrt().item()
        ),
    }


def run_case(args, name, rows, dtype_name, layout, case_index, flush):
    n, k = (args.vocab_width, args.vocab_hidden) if name == "vocab" else CASES[name]
    padding = 16 if layout == "row-strided" else 0
    # A geometry-specific seed keeps operands stable when selecting a CLI subset.
    seed = args.seed + rows * 1000003 + n * 1009 + k
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((rows, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    weight = torch.empty((n, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    x.normal_(generator=generator)
    weight.normal_(std=1 / math.sqrt(k), generator=generator)
    out = torch.empty((rows, n + padding), device="cuda", dtype=DTYPES[dtype_name])[
        :, :n
    ]
    native._validate(x, weight, out, None)
    key = (n, k, str(x.dtype), str(weight.dtype), str(out.dtype), 8)
    default = TUNING.configure(query_from_call(x, weight, out=out), device=None).default
    pointer_types = (x.dtype, weight.dtype, x.dtype, out.dtype)
    pointers = tuple(
        native.make_ptr(
            native._DTYPES[dtype],
            16,
            native.cute.AddressSpace.gmem,
            assumed_align=dtype.itemsize,
        )
        for dtype in pointer_types
    )
    # Compile the native entrypoints directly to compare both implementations.
    compiled = tuple(
        compile_native(
            kernel,
            *pointers,
            native.Int32(1),
            native.Int64(k),
            native.Int64(k),
            native.Int64(n),
            native.Int64(1),
            native.Int64(1),
            native.Int32(0),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "benchmark.bf16_projection",
                1,
                (*key, arm),
            ),
        )
        for arm, kernel in zip(
            ARMS,
            (
                native.SmallNGemvKernel(n, k, True, False),
                native.Bf16GemmKernel(n, k, False),
            ),
            strict=True,
        )
    )
    vector_loads = int(
        x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and x.stride(0) % 8 == 0
        and weight.stride(0) % 8 == 0
        and x.stride(1) == weight.stride(1) == 1
        and k % 8 == 0
    )
    launch_args = (
        native._pointer(x),
        native._pointer(weight),
        native._pointer(x),
        native._pointer(out),
        rows,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        x.stride(1),
        weight.stride(1),
        vector_loads,
    )

    def launch_kernel(kernel):
        # torch.cuda.graph uses its own capture stream; never bind that stream
        # before entering capture, even though pointers and kernels are fixed.
        run_compiled(kernel, (*launch_args, current_cuda_stream()))

    launches = {
        arm: partial(launch_kernel, kernel)
        for arm, kernel in zip(ARMS, compiled, strict=True)
    }
    expected = oracle(x, weight)
    correctness = {}
    initial_order = ARMS if case_index % 2 == 0 else ARMS[::-1]
    for arm in initial_order:
        out.fill_(float("nan"))
        for _ in range(args.warmup):
            launches[arm]()
        torch.cuda.synchronize()
        correctness[arm] = {"eager": check_output(out, expected, arm)}

    graphs = {}
    blocks = []
    samples = {arm: [] for arm in ARMS}
    with kernel_resolution_guard(
        "BF16 projection native SIMT/MMA graph capture and paired replay"
    ):
        for arm in initial_order:

            def repeated(launch=launches[arm]):
                for _ in range(args.launches):
                    launch()

            graphs[arm] = capture_cuda_graph(repeated, warmup=0)
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_before_timing"] = check_output(out, expected, arm)
        gpu_before = nvidia_smi_gpu_mode_snapshot()
        for repeat in range(args.repeats):
            order = initial_order if repeat % 2 == 0 else initial_order[::-1]
            block = {"repeat": repeat, "order": list(order), "replay_us": {}}
            for arm in order:
                raw = bench_cuda_graph(
                    graphs[arm], replays=args.replays, l2_flush=flush
                )["replay_us"]
                if any(not math.isfinite(value) or value <= 0 for value in raw):
                    raise RuntimeError(f"Invalid CUDA event samples for {arm}: {raw}")
                block["replay_us"][arm] = raw
                samples[arm].extend(value / args.launches for value in raw)
            block["simt_over_mma_ratio"] = statistics.median(
                block["replay_us"]["simt"]
            ) / statistics.median(block["replay_us"]["mma"])
            blocks.append(block)
        gpu_after = nvidia_smi_gpu_mode_snapshot()
        for arm in ARMS:
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_after_timing"] = check_output(out, expected, arm)
    for graph in graphs.values():
        graph.reset()
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    ratio = medians["simt"] / medians["mma"]
    return {
        "case": name,
        "M": rows,
        "N": n,
        "K": k,
        "seed": seed,
        "operand_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": dtype_name,
        "layout": layout,
        "strides": {
            "x": list(x.stride()),
            "weight": list(weight.stride()),
            "out": list(out.stride()),
        },
        "shared_operands_and_output_between_arms": True,
        "simt_vector_loads": bool(vector_loads),
        "cache_key": list(key),
        "implementations": {
            "simt": "SmallNGemvKernel",
            "mma": "Bf16GemmKernel(32,64,64)",
        },
        "production_policy_arm": default.backend,
        "production_policy_regresses_vs_simt": (
            ratio < 1
            if default.backend == "mma"
            else False
            if default.backend == "simt"
            else None
        ),
        "correctness": correctness,
        "median_us_per_projection": medians,
        "simt_over_mma_ratio": ratio,
        "faster_arm": "mma" if ratio > 1 else "simt" if ratio < 1 else "tie",
        "mma_winning_blocks": sum(block["simt_over_mma_ratio"] > 1 for block in blocks),
        "samples_us_per_projection": samples,
        "blocks": blocks,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def main():
    args = parse_args()
    if args.model_path is not None:
        from benchmarks.checkpoint_projection import run

        return run(args)
    require_sm120()
    # These options apply only to the untimed FP32 numerical oracle.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush_bytes = (
        resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache == "cold" else 0
    )
    flush = make_l2_flush_fn(args.cache == "cold", flush_bytes)
    sources = [
        Path(__file__),
        Path(native.__file__),
        ROOT / "benchmarks/common.py",
        ROOT / "b12x/_lib/compiler.py",
        ROOT / "b12x/_lib/intrinsics.py",
        ROOT / "b12x/_lib/runtime_control.py",
        ROOT / "b12x/_lib/utils.py",
    ]
    result = {
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "command": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "B12X_", "CUTE_", "CUTLASS_"))
        },
        "toolchain": toolchain(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "device": {
            "logical_index": torch.cuda.current_device(),
            "uuid": str(getattr(properties, "uuid", "")),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        },
        "timing_scope": "CUDA events around captured native cached callable replay only; oracle, compilation, warmup, snapshots, correctness and optional L2 flush excluded. Paired alternating AB/BA blocks; one shared output; no Torch/CUBLAS measured arm.",
        "ratio_definition": "simt_over_mma_ratio = median SIMT us / median MMA us; >1 means MMA faster, <1 means MMA is slower than SIMT. No significance claim.",
        "l2_flush_bytes": flush_bytes,
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for name in args.cases:
            for rows in args.rows:
                for dtype in args.output_dtypes:
                    for layout in args.layouts:
                        record = run_case(
                            args,
                            name,
                            rows,
                            dtype,
                            layout,
                            len(result["records"]),
                            flush,
                        )
                        result["records"].append(record)
                        save()
                        print(
                            json.dumps(
                                {
                                    key: record[key]
                                    for key in (
                                        "case",
                                        "M",
                                        "N",
                                        "K",
                                        "output_dtype",
                                        "layout",
                                        "median_us_per_projection",
                                        "simt_over_mma_ratio",
                                        "mma_winning_blocks",
                                        "production_policy_regresses_vs_simt",
                                    )
                                }
                            ),
                            flush=True,
                        )
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_unix_ns"] = time.time_ns()
        result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
        save()


if __name__ == "__main__":
    main()
