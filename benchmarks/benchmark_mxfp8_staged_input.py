#!/usr/bin/env python3
"""Measure concatenated and staged MXFP8 linear input construction.

The default geometry is the Kimi-K3 DFlash auxiliary projection: six BF16
feature slices of width 7,168 feed one 43,008-by-7,168 MXFP8 linear. The staged
arm releases each BF16 slice after quantizing it into retained MXFP8 storage and
writes the GEMM result directly into caller-owned output storage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm import mxfp8_linear
from benchmarks.common import nvidia_smi_gpu_mode_snapshot


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--slice-width", type=int, default=7168)
    parser.add_argument("--slices", type=int, default=6)
    parser.add_argument("--output-width", type=int, default=7168)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _git_metadata(repository: pathlib.Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    remote = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repository": remote,
        "worktree": str(repository),
        "revision": revision,
        "worktree_state": "clean" if not status else "modified",
        "worktree_status": status,
    }


def _gpu_metadata() -> dict[str, object]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    driver_versions = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "device": device,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "driver_version": driver_versions[0],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "mode": nvidia_smi_gpu_mode_snapshot(),
    }


def _source(tokens: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        (tokens, width),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).div_(4)


def _weight(rows: int, columns: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = torch.empty((rows, columns), device="cuda", dtype=torch.float8_e4m3fn)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        source = torch.randn(
            (stop - start, columns),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).div_(8)
        values[start:stop].copy_(source)
    scales = torch.full(
        (rows, columns // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    return mxfp8_linear.pack_weight(values, scales)


def _elapsed_ms(operation: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def main() -> None:
    args = _arguments()
    if args.slice_width % 128:
        raise ValueError("slice width must be divisible by 128")
    repository = pathlib.Path(__file__).resolve().parents[1]
    input_width = args.slice_width * args.slices
    packed_weight = _weight(args.output_width, input_width, args.seed + 1000)
    sources = [
        _source(args.tokens, args.slice_width, args.seed + index)
        for index in range(args.slices)
    ]
    torch.cuda.synchronize()
    concatenated_live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def concatenated() -> torch.Tensor:
        joined = torch.cat(sources, dim=-1)
        return mxfp8_linear.mm(joined, packed_weight, expected_m=args.tokens)

    concatenated_result = concatenated()
    torch.cuda.synchronize()
    concatenated_peak = torch.cuda.max_memory_allocated()
    del concatenated_result

    retained = mxfp8_linear.empty_input(args.tokens, input_width, device="cuda")
    output = torch.empty(
        (args.tokens, args.output_width),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def staged() -> torch.Tensor:
        for index, source in enumerate(sources):
            mxfp8_linear.quantize_input_slice(
                source,
                retained,
                destination_column=index * args.slice_width,
            )
        return mxfp8_linear.mm_quantized_into(
            retained,
            packed_weight,
            tokens=args.tokens,
            out=output,
            expected_m=args.tokens,
        )

    expected = concatenated().clone()
    actual = staged().clone()
    bitwise_equal = torch.equal(actual, expected)
    maximum_difference = float((actual.float() - expected.float()).abs().max().item())
    concatenated()
    staged()
    torch.cuda.synchronize()
    concatenated_samples: list[float] = []
    staged_samples: list[float] = []
    for iteration in range(args.iterations):
        if iteration % 2:
            staged_samples.append(_elapsed_ms(staged))
            concatenated_samples.append(_elapsed_ms(concatenated))
        else:
            concatenated_samples.append(_elapsed_ms(concatenated))
            staged_samples.append(_elapsed_ms(staged))

    sources.clear()
    del expected, actual
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    staged_live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def staged_memory() -> torch.Tensor:
        for index in range(args.slices):
            source = _source(args.tokens, args.slice_width, args.seed + index)
            mxfp8_linear.quantize_input_slice(
                source,
                retained,
                destination_column=index * args.slice_width,
            )
            del source
        return mxfp8_linear.mm_quantized_into(
            retained,
            packed_weight,
            tokens=args.tokens,
            out=output,
            expected_m=args.tokens,
        )

    staged_result = staged_memory()
    torch.cuda.synchronize()
    staged_peak = torch.cuda.max_memory_allocated()
    del staged_result
    report = {
        "schema": "b12x.mxfp8-staged-input.v2",
        "command": " ".join(sys.argv),
        "source": _git_metadata(repository),
        "gpu": _gpu_metadata(),
        "geometry": {
            "tokens": args.tokens,
            "slice_width": args.slice_width,
            "slices": args.slices,
            "input_width": input_width,
            "output_width": args.output_width,
        },
        "output": {
            "bitwise_equal": bitwise_equal,
            "max_absolute_difference": maximum_difference,
        },
        "timing": {
            "samples_interleaved": True,
            "source_construction_included": False,
        },
        "concatenated": {
            "samples_ms": concatenated_samples,
            "median_ms": statistics.median(concatenated_samples),
            "live_before_bytes": concatenated_live,
            "peak_bytes": concatenated_peak,
        },
        "staged": {
            "samples_ms": staged_samples,
            "median_ms": statistics.median(staged_samples),
            "live_before_bytes": staged_live,
            "peak_bytes": staged_peak,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized, end="")
    if not report["output"]["bitwise_equal"]:
        raise SystemExit("staged MXFP8 input changed the linear output")


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(f"elapsed_seconds={time.monotonic() - started:.3f}")
