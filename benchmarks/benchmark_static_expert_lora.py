#!/usr/bin/env python3
"""Performance gate for the hybrid W4A16 expert-LoRA path.

This benchmark deliberately uses the DeepSeek-V4-Flash TP4 expert geometry:
256 experts, H=4096, I_tp=512, top-k=6, BF16 activations, and native ModelOpt
FP4 storage. It compares B12X's untouched small-M fused decode path with the
static rank-4 expert-LoRA implementation under CUDA graph replay. Single-token
decode augments the fused direct kernel; larger batches retain the staged
tensor-core implementation selected by the production dispatcher.

The base FP4 payload is synthetic because its values do not affect launch
geometry.  When ``--adapter`` is supplied, the rank-4 tensors are loaded from
the real adapter and TP-sliced for the selected rank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe._shared.kernels.w4a16.kernel import (
    _small_m_direct_supported,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.lora import W4A16StaticExpertLoRA
from b12x.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_modelopt_native_weights,
)
from b12x.moe._shared.kernels.reference import (
    compare_to_reference,
    moe_reference_w4a16_f32,
)


EXPERTS = 256
HIDDEN_SIZE = 4096
INTERMEDIATE_FULL = 2048
TOPK = 6
RANK = 4
ORACLE_MIN_COS = 0.9975


def _command_metadata(
    argv: list[str], *, cwd: Path | None = None
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"argv": argv, "error": repr(error)}
    return {
        "argv": argv,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _gpu_snapshot() -> dict[str, object]:
    fields = (
        "timestamp,driver_version,index,uuid,name,pstate,compute_mode,"
        "power.draw,power.limit,temperature.gpu,clocks.current.sm,"
        "clocks.current.memory,clocks_event_reasons.active,gpu_recovery_action"
    )
    return _command_metadata(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )


def _selected_gpu_identity() -> dict[str, object]:
    """Record the physical GPU UUID backing visible CUDA device zero."""

    apps = _command_metadata(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    selected_rows: list[str] = []
    stdout = apps.get("stdout")
    if isinstance(stdout, str):
        current_pid = str(os.getpid())
        for row in stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) >= 2 and fields[1] == current_pid:
                selected_rows.append(row)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_index": 0,
        "current_pid": os.getpid(),
        "physical_compute_process_rows": selected_rows,
        "all_compute_processes": apps,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_artifacts() -> dict[str, object]:
    """Inventory CuTe and Triton artifacts produced by this benchmark cache."""

    compile_root_value = os.environ.get("B12X_COMPILE_CACHE_DIR")
    triton_root_value = os.environ.get("TRITON_CACHE_DIR")
    manifests: list[dict[str, object]] = []
    if compile_root_value:
        compile_root = Path(compile_root_value)
        for manifest_path in sorted(compile_root.rglob("*.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            kernel_id = manifest.get("kernel_id")
            if not isinstance(kernel_id, str) or not kernel_id.startswith(
                "moe.w4a16"
            ):
                continue
            manifests.append(
                {
                    "manifest": str(manifest_path.relative_to(compile_root)),
                    "kernel_id": kernel_id,
                    "cache_key": manifest.get("cache_key"),
                    "compile_spec_hash": manifest.get("compile_spec_hash"),
                    "compile_spec_json": manifest.get("compile_spec_json"),
                    "object_sha256": manifest.get("object_sha256"),
                    "artifact_evidence_sha256": manifest.get(
                        "artifact_evidence_sha256"
                    ),
                    "target": manifest.get("target"),
                }
            )

    triton_artifacts: list[dict[str, object]] = []
    if triton_root_value:
        triton_root = Path(triton_root_value)
        for artifact_path in sorted(triton_root.rglob("*")):
            if (
                not artifact_path.is_file()
                or artifact_path.suffix not in {".cubin", ".ptx"}
                or "_w4a16_static_lora_" not in artifact_path.name
            ):
                continue
            triton_artifacts.append(
                {
                    "artifact": str(artifact_path.relative_to(triton_root)),
                    "sha256": _sha256(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                }
            )
    return {
        "b12x_compile_cache_dir": compile_root_value,
        "triton_cache_dir": triton_root_value,
        "cute_manifests": manifests,
        "triton_artifacts": triton_artifacts,
    }


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 0.25 + 0.03125).to(
        torch.float8_e4m3fn
    )


def _make_native_modelopt_weights(
    *, intermediate_size: int
) -> tuple[torch.Tensor, ...]:
    w13_rows = 2 * intermediate_size
    w13 = torch.randint(
        0,
        256,
        (EXPERTS, w13_rows, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (EXPERTS, HIDDEN_SIZE, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((EXPERTS, w13_rows, HIDDEN_SIZE // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((EXPERTS, HIDDEN_SIZE, intermediate_size // 16))
    )
    w13_global_scale = (
        torch.rand(EXPERTS, device="cuda", dtype=torch.float32) * 0.1 + 0.05
    )
    w2_global_scale = (
        torch.rand(EXPERTS, device="cuda", dtype=torch.float32) * 0.1 + 0.05
    )
    return (
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
    )


def _adapter_key(layer: int, suffix: str) -> str:
    return f"base_model.model.model.layers.{layer}.mlp.experts.{suffix}.weight"


def _load_adapter(
    path: Path,
    *,
    layer: int,
    tp_size: int,
    tp_rank: int,
    token_mapping: torch.Tensor,
    split_w13_b: bool,
) -> W4A16StaticExpertLoRA:
    if INTERMEDIATE_FULL % tp_size:
        raise ValueError("full intermediate size must divide evenly across TP")
    intermediate_tp = INTERMEDIATE_FULL // tp_size
    start = tp_rank * intermediate_tp
    stop = start + intermediate_tp
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        w13_a = handle.get_tensor(_adapter_key(layer, "lora_A"))
        w13_b_full = handle.get_tensor(_adapter_key(layer, "lora_B"))
        w2_a_full = handle.get_tensor(_adapter_key(layer, "lora_A_down"))
        w2_b = handle.get_tensor(_adapter_key(layer, "lora_B_down"))

    # The fused projection has two independent I-sized halves. Slice each
    # half for TP, preserving vLLM's independent allocations when requested.
    w13_b_gate = w13_b_full[:, start:stop, :]
    w13_b_up = w13_b_full[
        :, INTERMEDIATE_FULL + start : INTERMEDIATE_FULL + stop, :
    ]
    w2_a = w2_a_full[:, :, start:stop]
    tensors = [w13_a, w13_b_gate, w13_b_up, w2_a, w2_b]
    tensors = [
        tensor.to(device="cuda", dtype=torch.bfloat16).contiguous()
        for tensor in tensors
    ]
    gate_b, up_b = tensors[1], tensors[2]
    return W4A16StaticExpertLoRA(
        w13_a=tensors[0],
        w13_b=gate_b if split_w13_b else torch.cat((gate_b, up_b), dim=1),
        w13_b_up=up_b if split_w13_b else None,
        w2_a=tensors[3],
        w2_b=tensors[4],
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )


def _synthetic_adapter(
    *,
    intermediate_size: int,
    token_mapping: torch.Tensor,
    split_w13_b: bool,
) -> W4A16StaticExpertLoRA:
    def make(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16)

    gate_b = make((EXPERTS, intermediate_size, RANK), 0.04)
    up_b = make((EXPERTS, intermediate_size, RANK), 0.04)
    return W4A16StaticExpertLoRA(
        w13_a=make((EXPERTS, RANK, HIDDEN_SIZE), 0.025),
        w13_b=gate_b if split_w13_b else torch.cat((gate_b, up_b), dim=1),
        w13_b_up=up_b if split_w13_b else None,
        w2_a=make((EXPERTS, RANK, intermediate_size), 0.025),
        w2_b=make((EXPERTS, HIDDEN_SIZE, RANK), 0.04),
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )


def _capture(run: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    return graph


def _time_graph_once(graph: torch.cuda.CUDAGraph, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) * 1000.0 / iterations


def _time_graphs_interleaved(
    base_graph: torch.cuda.CUDAGraph,
    lora_graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Alternate variant order so monotonic clock or thermal drift is balanced."""

    base_samples: list[float] = []
    lora_samples: list[float] = []
    for repeat in range(repeats):
        ordered = (
            (("base", base_graph), ("lora", lora_graph))
            if repeat % 2 == 0
            else (("lora", lora_graph), ("base", base_graph))
        )
        for label, graph in ordered:
            sample = _time_graph_once(graph, iterations=iterations)
            (base_samples if label == "base" else lora_samples).append(sample)
    return base_samples, lora_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--split-w13-b",
        action="store_true",
        help="benchmark vLLM's independent gate/up W13 B allocations",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print a one-replay CUDA kernel profile for each token count",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.tp_rank < 0 or args.tp_rank >= args.tp_size:
        raise ValueError("tp-rank must be in [0, tp-size)")

    torch.manual_seed(20260821)
    torch.cuda.set_device(0)
    intermediate_size = INTERMEDIATE_FULL // args.tp_size
    raw_weights = _make_native_modelopt_weights(
        intermediate_size=intermediate_size
    )
    prepared = prepare_w4a16_modelopt_native_weights(
        *raw_weights,
        activation="silu",
        params_dtype=torch.bfloat16,
        source_format="modelopt_nvfp4",
        w13_layout="up_gate",
    )
    repo_root = Path(__file__).resolve().parents[1]
    gpu_props = torch.cuda.get_device_properties(0)
    evidence: dict[str, object] = {
        "command": [sys.executable, *sys.argv],
        "worktree_path": str(repo_root),
        "container_image": os.environ.get("B12X_BENCH_CONTAINER_IMAGE"),
        "git_commit": _command_metadata(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ),
        "git_status": _command_metadata(
            ["git", "status", "--short", "--branch"], cwd=repo_root
        ),
        "git_source_tree": _command_metadata(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root
        ),
        "comparison_revision": _command_metadata(
            ["git", "merge-base", "HEAD", "origin/master"], cwd=repo_root
        ),
        "ptxas": _command_metadata(
            [
                os.environ.get("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas"),
                "--version",
            ]
        ),
        "gpu_before": _gpu_snapshot(),
        "selected_gpu": _selected_gpu_identity(),
    }

    results: dict[str, object] = {
        "evidence": evidence,
        "gpu": torch.cuda.get_device_name(0),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
            "triton": _package_version("triton"),
            "gpu_capability": [int(gpu_props.major), int(gpu_props.minor)],
            "gpu_multi_processor_count": int(gpu_props.multi_processor_count),
            "gpu_total_memory": int(gpu_props.total_memory),
        },
        "shape": {
            "experts": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_full": INTERMEDIATE_FULL,
            "intermediate_tp": intermediate_size,
            "topk": TOPK,
            "tp_size": args.tp_size,
            "tp_rank": args.tp_rank,
            "activation": "silu",
            "swiglu_limit": 10.0,
            "rank": RANK,
        },
        "adapter": None if args.adapter is None else str(args.adapter),
        "split_w13_b": bool(args.split_w13_b),
        "measurements": [],
    }

    for m in args.tokens:
        x = (torch.randn(m, HIDDEN_SIZE, device="cuda") * 0.2).to(
            torch.bfloat16
        )
        topk_ids = torch.stack(
            [
                (torch.arange(m, device="cuda", dtype=torch.int32) * TOPK + i)
                % EXPERTS
                for i in range(TOPK)
            ],
            dim=1,
        ).contiguous()
        topk_weights = torch.softmax(
            torch.randn(m, TOPK, device="cuda", dtype=torch.float32), dim=-1
        ).contiguous()
        token_mapping = torch.zeros(m, dtype=torch.int32, device="cuda")
        adapter = (
            _load_adapter(
                args.adapter,
                layer=args.layer,
                tp_size=args.tp_size,
                tp_rank=args.tp_rank,
                token_mapping=token_mapping,
                split_w13_b=args.split_w13_b,
            )
            if args.adapter is not None
            else _synthetic_adapter(
                intermediate_size=intermediate_size,
                token_mapping=token_mapping,
                split_w13_b=args.split_w13_b,
            )
        )

        base_buffers = make_w4a16_packed_buffers(
            prepared,
            m=m,
            topk=TOPK,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        lora_buffers = make_w4a16_packed_buffers(
            prepared,
            m=m,
            topk=TOPK,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        rank_scratch = torch.empty(
            m * TOPK, RANK, dtype=torch.bfloat16, device="cuda"
        )

        def run(
            buffers,
            *,
            static_lora=None,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            rank_scratch=rank_scratch,
        ):
            return run_w4a16_moe(
                x,
                prepared,
                topk_weights,
                topk_ids,
                activation="silu",
                fast_math=True,
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=buffers.output,
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
                swiglu_limit=10.0,
                static_lora=static_lora,
                lora_rank_scratch=(
                    rank_scratch if static_lora is not None else None
                ),
            )

        base_graph = _capture(lambda buffers=base_buffers: run(buffers))
        base_output = base_buffers.output.clone()
        lora_graph = _capture(
            lambda buffers=lora_buffers, static_lora=adapter: run(
                buffers,
                static_lora=static_lora,
            )
        )
        lora_output = lora_buffers.output.clone()
        torch.cuda.synchronize()

        base_reference = moe_reference_w4a16_f32(
            x,
            *raw_weights[:3],
            *raw_weights[3:],
            topk_ids,
            topk_weights,
            EXPERTS,
            HIDDEN_SIZE,
            intermediate_size,
            activation="silu",
            swiglu_limit=10.0,
        )
        lora_reference = moe_reference_w4a16_f32(
            x,
            *raw_weights[:3],
            *raw_weights[3:],
            topk_ids,
            topk_weights,
            EXPERTS,
            HIDDEN_SIZE,
            intermediate_size,
            activation="silu",
            swiglu_limit=10.0,
            static_lora=adapter,
        )
        base_oracle = compare_to_reference(base_output, base_reference)
        lora_oracle = compare_to_reference(lora_output, lora_reference)
        for label, output_value, metrics in (
            ("base", base_output, base_oracle),
            ("lora", lora_output, lora_oracle),
        ):
            if not torch.isfinite(output_value).all().item():
                raise RuntimeError(f"{label} output is non-finite before timing")
            if metrics.cos < ORACLE_MIN_COS:
                raise RuntimeError(
                    f"{label} output failed FP32 oracle: "
                    f"cos={metrics.cos:.8f} < {ORACLE_MIN_COS:.8f}; "
                    f"metrics={metrics}"
                )

        allocated_before_replay = torch.cuda.memory_allocated()
        base_us, lora_us = _time_graphs_interleaved(
            base_graph,
            lora_graph,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        torch.cuda.synchronize()
        allocated_after_replay = torch.cuda.memory_allocated()
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            for label, graph in (("base", base_graph), ("lora", lora_graph)):
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as profiler:
                    graph.replay()
                    torch.cuda.synchronize()
                print(
                    f"\n== tokens={m} {label} CUDA profile ==\n"
                    + profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=30
                    ),
                    flush=True,
                )
        base_median = statistics.median(base_us)
        lora_median = statistics.median(lora_us)
        delta = (lora_output.float() - base_output.float()).norm()
        base_norm = base_output.float().norm().clamp_min(1e-30)
        small_m_direct_baseline = _small_m_direct_supported(
            m=m,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=intermediate_size,
            num_experts=EXPERTS,
            topk=TOPK,
            activation="silu",
            apply_router_weight_on_input=False,
            swiglu_limit=10.0,
            swiglu_alpha=None,
            swiglu_beta=None,
            element_dtype="bf16",
            weight_layout=prepared.weight_layout,
            w13_layout=prepared.w13_layout,
            scale_format=prepared.scale_format,
        )
        measurement = {
            "tokens": m,
            "lora_path": (
                "direct_augmented"
                if m == 1 and small_m_direct_baseline
                else "staged"
            ),
            "base_us": base_us,
            "lora_us": lora_us,
            "base_median_us": base_median,
            "lora_median_us": lora_median,
            "slowdown_x": lora_median / base_median,
            "overhead_percent": (lora_median / base_median - 1.0) * 100.0,
            "relative_output_delta": float((delta / base_norm).item()),
            "oracle_min_cos": ORACLE_MIN_COS,
            "base_oracle": asdict(base_oracle),
            "lora_oracle": asdict(lora_oracle),
            "base_oracle_passed": base_oracle.cos >= ORACLE_MIN_COS,
            "lora_oracle_passed": lora_oracle.cos >= ORACLE_MIN_COS,
            "base_finite": bool(torch.isfinite(base_output).all().item()),
            "lora_finite": bool(torch.isfinite(lora_output).all().item()),
            "lora_nonzero_delta": bool(delta.item() > 0.0),
            "small_m_direct_baseline": small_m_direct_baseline,
            "replay_allocated_delta_bytes": (
                allocated_after_replay - allocated_before_replay
            ),
        }
        results["measurements"].append(measurement)
        print(json.dumps(measurement, sort_keys=True), flush=True)

        # Keep only one graph pair live at a time.
        del base_graph, lora_graph, base_buffers, lora_buffers, adapter
        torch.cuda.empty_cache()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    results["memory"] = {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    evidence["gpu_after"] = _gpu_snapshot()
    evidence["compiler_artifacts"] = _compiler_artifacts()
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
