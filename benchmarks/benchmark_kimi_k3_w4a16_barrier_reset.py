#!/usr/bin/env python3
"""Benchmark reusable W4A16 small-M barrier state at Kimi-K3 TP16 shape.

This is intentionally model-free.  It uses K3's per-rank native-MXFP4
decode shape (M=1, H=3584, I=3072/16=192, top-k=16) and compares CUDA graphs
that contain either the historical two scalar ``zero_`` launches plus MoE, or
only MoE with the kernel's persistent counter/epoch protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys

import torch

from sparkinfer.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_e8m0_native_weights,
)


RESET_ENV = "SPARKINFER_W4A16_SMALL_M_HOST_BARRIER_RESET"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experts",
        type=int,
        default=32,
        help="Local experts to materialize; barrier behavior is expert-count invariant.",
    )
    parser.add_argument("--validation-cases", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.validation_cases < 1:
        parser.error("--validation-cases must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    return args


def _command_output(
    command: list[str],
    *,
    cwd: pathlib.Path | None = None,
    allow_empty: bool = False,
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    output = result.stdout.strip()
    return output if output or allow_empty else "unavailable"


def _provenance(device: torch.device) -> dict[str, object]:
    repo = pathlib.Path(__file__).resolve().parents[1]
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=repo)
    worktree = _command_output(["git", "rev-parse", "--show-toplevel"], cwd=repo)
    status = _command_output(["git", "status", "--short"], cwd=repo, allow_empty=True)
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    uuid = str(getattr(props, "uuid", ""))
    physical_selector = uuid if uuid and uuid != "None" else str(device_index)
    gpu_mode = _command_output(
        [
            "nvidia-smi",
            "-i",
            physical_selector,
            "--query-gpu=index,uuid,pci.bus_id,pstate,clocks.current.graphics,"
            "clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "git_commit": commit,
        "worktree": worktree,
        "git_dirty": None if status == "unavailable" else bool(status),
        "gpu": {
            "requested_device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_index": device_index,
            "name": props.name,
            "uuid": uuid,
            "pci_bus_id": str(getattr(props, "pci_bus_id", "")),
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "mode": gpu_mode,
            "mode_fields": (
                "index,uuid,pci.bus_id,pstate,clocks.current.graphics_mhz,"
                "clocks.current.memory_mhz,power_limit_w"
            ),
        },
        "ratio_direction": "higher_speedup_is_better",
    }


def _capture(run, *, reset: bool) -> torch.cuda.CUDAGraph:
    os.environ[RESET_ENV] = "1" if reset else "0"
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iterations


def main() -> None:
    args = _parse_args()
    if args.experts < 16:
        raise ValueError("--experts must be >= K3 top-k (16)")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    m = 1
    hidden_size = 3584
    intermediate_size = 192
    topk = 16
    rows = 2 * intermediate_size
    e8m0_dtype = torch.float8_e8m0fnu

    w13 = torch.randint(
        0,
        256,
        (args.experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (args.experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )

    def patterned_e8m0(shape: tuple[int, ...], offset: int = 0) -> torch.Tensor:
        numel = 1
        for dim in shape:
            numel *= dim
        storage = (
            torch.arange(numel, dtype=torch.int64, device=device) + offset
        ) % 4 + 119
        return storage.to(torch.uint8).reshape(shape).view(e8m0_dtype)

    w13_scale = patterned_e8m0((args.experts, rows, hidden_size // 32))
    w2_scale = patterned_e8m0(
        (args.experts, hidden_size, intermediate_size // 32), offset=1
    )
    global_scales = torch.ones(args.experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        global_scales,
        w2,
        w2_scale,
        global_scales,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )

    inputs = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=device)
    topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    reset_buffers = make_w4a16_packed_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=device
    )
    persistent_buffers = make_w4a16_packed_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=device
    )
    fc2_chunks = ((intermediate_size // 2) + 127) // 128
    required_inter_u32 = m * fc2_chunks * 128 * topk

    def direct_intermediate(buffers) -> torch.Tensor:
        if buffers.intermediate_cache2.view(torch.uint32).numel() >= required_inter_u32:
            return buffers.intermediate_cache2
        return torch.empty(
            required_inter_u32 * 2,
            dtype=torch.bfloat16,
            device=device,
        )

    reset_intermediate = direct_intermediate(reset_buffers)
    persistent_intermediate = direct_intermediate(persistent_buffers)

    def make_run(buffers, intermediate_cache2: torch.Tensor):
        def run() -> torch.Tensor:
            return run_w4a16_moe(
                inputs,
                prepared,
                topk_weights,
                topk_ids,
                activation="situ",
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=intermediate_cache2,
                output=buffers.output,
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
            )

        return run

    inputs.normal_(generator=generator)
    topk_ids.copy_(
        torch.randperm(args.experts, device=device, generator=generator)[:topk]
    )
    topk_weights.copy_(
        torch.rand((m, topk), device=device, generator=generator).softmax(dim=-1)
    )
    reset_graph = _capture(make_run(reset_buffers, reset_intermediate), reset=True)
    persistent_graph = _capture(
        make_run(persistent_buffers, persistent_intermediate), reset=False
    )

    max_abs = 0.0
    exact_cases = 0
    for _ in range(args.validation_cases):
        inputs.normal_(generator=generator)
        topk_ids.copy_(
            torch.randperm(args.experts, device=device, generator=generator)[:topk]
        )
        topk_weights.copy_(
            torch.rand((m, topk), device=device, generator=generator).softmax(dim=-1)
        )
        reset_graph.replay()
        torch.cuda.synchronize()
        expected = reset_buffers.output.clone()
        persistent_graph.replay()
        torch.cuda.synchronize()
        actual = persistent_buffers.output
        if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
            raise RuntimeError("non-finite output in barrier validation corpus")
        exact_cases += int(torch.equal(expected, actual))
        max_abs = max(
            max_abs,
            float((expected.float() - actual.float()).abs().max()),
        )

    barrier_count = prepared.workspace[-2:-1]
    if int(barrier_count.item()) != 0:
        raise RuntimeError(f"barrier counter did not quiesce: {barrier_count.item()}")

    reset_times: list[float] = []
    persistent_times: list[float] = []
    for trial in range(args.trials):
        if trial % 2 == 0:
            reset_times.append(_time_graph(reset_graph, args.iterations))
            persistent_times.append(_time_graph(persistent_graph, args.iterations))
        else:
            persistent_times.append(_time_graph(persistent_graph, args.iterations))
            reset_times.append(_time_graph(reset_graph, args.iterations))

    reset_median = statistics.median(reset_times)
    persistent_median = statistics.median(persistent_times)
    result = {
        "provenance": _provenance(device),
        "shape": {
            "m": m,
            "hidden_size": hidden_size,
            "intermediate_size_per_tp16_rank": intermediate_size,
            "topk": topk,
            "materialized_experts": args.experts,
        },
        "validation": {
            "cases": args.validation_cases,
            "exact_cases": exact_cases,
            "max_abs": max_abs,
            "barrier_count_after": int(barrier_count.item()),
            "barrier_epoch_after": int(prepared.workspace[-1:].item()),
        },
        "host_reset_us": reset_times,
        "persistent_epoch_us": persistent_times,
        "host_reset_median_us": reset_median,
        "persistent_epoch_median_us": persistent_median,
        "saved_us": reset_median - persistent_median,
        "speedup": reset_median / persistent_median,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if exact_cases != args.validation_cases or max_abs != 0.0:
        raise SystemExit("persistent barrier epoch changed W4A16 output")


if __name__ == "__main__":
    main()
