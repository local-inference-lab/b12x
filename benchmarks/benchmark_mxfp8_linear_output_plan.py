#!/usr/bin/env python3
"""Compare output-storage plans for the complete MXFP8 linear operation.

The benchmark measures ``mxfp8_linear.mm`` under CUDA-graph replay, including
BF16 activation quantization and the dense GEMM. It compares two swapped TMA
plans for the Kimi-K3 TP16 projection shape M=3, N=132, K=7168. Both arms must
match the same dequantized reference before timing. Measurement order alternates
between the arms, and the JSON result retains every timing sample.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from cuda.bindings import runtime as cudart

import b12x._lib.dense_gemm as dense_module
from b12x._lib import compiler as cute_compiler
from b12x.gemm import block_fp8_linear as block_fp8
from b12x.gemm import mxfp8_linear
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch


@dataclass(frozen=True)
class OutputPlan:
    name: str
    mma_tiler_mn: tuple[int, int]
    swap_ab: bool = True


PLANS = (
    OutputPlan("tma-64x32-swapped", (64, 32)),
    OutputPlan("tma-64x64-swapped", (64, 64)),
)


def _command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable_metadata(name: str) -> dict[str, object]:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required executable is unavailable: {name}")
    path = pathlib.Path(executable).resolve()
    completed = subprocess.run(
        [str(path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "version": version,
    }


def _compile_artifacts() -> tuple[pathlib.Path, dict[str, dict[str, object]]]:
    root = cute_compiler._cute_compile_cache_dir()
    artifacts: dict[str, dict[str, object]] = {}
    if not root.exists():
        return root, artifacts
    for manifest_path in sorted(root.glob("*/*.json")):
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise RuntimeError(f"compile manifest is not an object: {manifest_path}")
        cache_key = str(manifest.get("cache_key", ""))
        if not cache_key or manifest_path.stem != cache_key:
            raise RuntimeError(
                f"compile manifest cache key is invalid: {manifest_path}"
            )
        object_path = manifest_path.with_suffix(".o")
        if not object_path.is_file():
            raise RuntimeError(f"compile object is missing: {object_path}")
        object_sha256 = _sha256(object_path)
        if object_sha256 != manifest.get("object_sha256"):
            raise RuntimeError(f"compile object hash differs: {object_path}")
        artifacts[cache_key] = {
            "cache_key": cache_key,
            "kernel_id": manifest.get("kernel_id"),
            "compile_spec_hash": manifest.get("compile_spec_hash"),
            "compile_kwargs_hash": manifest.get("compile_kwargs_hash"),
            "target_identity": manifest.get("target_identity"),
            "object_path": str(object_path.relative_to(root)),
            "object_bytes": object_path.stat().st_size,
            "object_sha256": object_sha256,
            "manifest_path": str(manifest_path.relative_to(root)),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_evidence_sha256": manifest.get("artifact_evidence_sha256"),
            "toolchain": manifest.get("toolchain"),
            "compile_options": manifest.get("compile_options"),
            "compile_environment": manifest.get("compile_environment"),
            "launch_metadata": manifest.get("launch_metadata"),
        }
    return root, artifacts


def _source_metadata(repository: pathlib.Path) -> dict[str, object]:
    paths = (
        pathlib.Path(__file__).resolve(),
        repository / "b12x/_lib/dense_gemm.py",
        repository / "b12x/gemm/mxfp8_linear/_kernel.py",
    )
    try:
        revision = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
        status = _command_output(["git", "-C", str(repository), "status", "--short"])
        revision_source = "git"
        worktree_state = "clean" if not status else "modified"
    except subprocess.CalledProcessError:
        revision = os.getenv("B12X_BENCHMARK_SOURCE_REVISION", "unavailable")
        worktree_state = os.getenv("B12X_BENCHMARK_WORKTREE_STATE", "unavailable")
        status = ""
        revision_source = "environment" if revision != "unavailable" else "unavailable"
    return {
        "repository": str(repository),
        "revision": revision,
        "revision_source": revision_source,
        "worktree_state": worktree_state,
        "worktree_status": status.splitlines(),
        "source_sha256": {
            str(path.relative_to(repository)): _sha256(path) for path in paths
        },
    }


def _canonical_pci_bus_id(value: str) -> str:
    domain, bus, device = value.strip().lower().split(":")
    return f"{int(domain, 16):08x}:{bus}:{device}"


def _gpu_metadata() -> dict[str, object]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    error, pci_bus_id_buffer = cudart.cudaDeviceGetPCIBusId(32, device)
    if int(error) != 0:
        raise RuntimeError(
            f"cudaDeviceGetPCIBusId failed for logical CUDA device {device}: {error}"
        )
    pci_bus_id = _canonical_pci_bus_id(
        pci_bus_id_buffer.split(b"\0", 1)[0].decode("ascii")
    )
    query = (
        "index,uuid,pci.bus_id,name,pstate,clocks.current.sm,"
        "clocks.current.memory,power.limit,compute_mode,"
        "clocks_throttle_reasons.active"
    )
    records = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    visible_record = None
    selected_fields: list[str] | None = None
    for record in records:
        fields = [field.strip() for field in record.split(",")]
        if len(fields) >= 3 and _canonical_pci_bus_id(fields[2]) == pci_bus_id:
            visible_record = record
            selected_fields = fields
            break
    if visible_record is None or selected_fields is None or len(selected_fields) != 10:
        raise RuntimeError(
            "nvidia-smi did not report the physical GPU selected by CUDA at "
            f"PCI bus ID {pci_bus_id}"
        )
    driver_version = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    return {
        "cuda_device_index": device,
        "cuda_pci_bus_id": pci_bus_id,
        "compute_capability": [properties.major, properties.minor],
        "torch_device_name": properties.name,
        "nvidia_smi_record": visible_record,
        "physical_index": int(selected_fields[0]),
        "uuid": selected_fields[1],
        "pstate": selected_fields[4],
        "sm_clock_mhz": int(selected_fields[5]),
        "memory_clock_mhz": int(selected_fields[6]),
        "power_limit_watts": float(selected_fields[7]),
        "compute_mode": selected_fields[8],
        "active_throttle_mask": selected_fields[9],
        "driver_version": driver_version,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _quantize_modelopt_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    blocked = source.float().reshape(rows, width // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    scale_base = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(scale_base)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).float()
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _reference(source: torch.Tensor, packed_weight: object) -> torch.Tensor:
    source_q = block_fp8.quantize_input(source)
    source_dequantized = dequantize_mxfp8_rows_torch(
        source_q.values,
        source_q.scale_rows,
    )
    weight_dequantized = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values,
        packed_weight.weight.scale_rows,
    )
    return (source_dequantized @ weight_dequantized.T).to(torch.bfloat16)


@contextlib.contextmanager
def _force_target_plan(
    plan: OutputPlan,
    *,
    tokens: int,
    out_features: int,
    in_features: int,
) -> Iterator[None]:
    """Override only the benchmarked shape while compiling each graph arm."""

    original = dense_module._select_default_dense_gemm_plan

    def select_plan(
        m: int,
        n: int,
        k: int,
        sm_count: int,
        *,
        is_mxfp8: bool,
        is_mxfp6: bool = False,
        expected_m: int | None = None,
        select_swapped_output_storage: bool = False,
    ):
        selected = original(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=is_mxfp8,
            is_mxfp6=is_mxfp6,
            expected_m=expected_m,
            select_swapped_output_storage=select_swapped_output_storage,
        )
        if (
            is_mxfp8
            and (m, n, k) == (tokens, out_features, in_features)
            and select_swapped_output_storage
        ):
            return replace(
                selected,
                mma_tiler_mn=plan.mma_tiler_mn,
                load_path="tma",
                swap_ab=plan.swap_ab,
            )
        return selected

    dense_module._select_default_dense_gemm_plan = select_plan
    try:
        yield
    finally:
        dense_module._select_default_dense_gemm_plan = original


def _capture_plan(
    plan: OutputPlan,
    source: torch.Tensor,
    packed_weight: object,
    *,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, float]:
    tokens, in_features = map(int, source.shape)
    out_features = int(packed_weight.out_features)
    with _force_target_plan(
        plan,
        tokens=tokens,
        out_features=out_features,
        in_features=in_features,
    ):
        for _ in range(warmup):
            mxfp8_linear.mm(source, packed_weight, expected_m=tokens)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = mxfp8_linear.mm(source, packed_weight, expected_m=tokens)
    cold_sample_microseconds = _time_call_us(graph.replay)
    return graph, output, cold_sample_microseconds


def _time_call_us(call: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0)


def _balanced_samples_us(
    first: Callable[[], None],
    second: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    for index in range(warmup):
        (first if index % 2 == 0 else second)()
        (second if index % 2 == 0 else first)()
    torch.cuda.synchronize()

    first_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    second_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def record(
        call: Callable[[], None],
        destination: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> None:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        destination.append((start, stop))

    for index in range(iterations):
        if index % 2 == 0:
            record(first, first_events)
            record(second, second_events)
        else:
            record(second, second_events)
            record(first, first_events)
    torch.cuda.synchronize()
    return (
        [start.elapsed_time(stop) * 1000.0 for start, stop in first_events],
        [start.elapsed_time(stop) * 1000.0 for start, stop in second_events],
    )


def _summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_microseconds": samples,
        "median_microseconds": statistics.median(samples),
        "minimum_microseconds": ordered[0],
        "maximum_microseconds": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=7168)
    parser.add_argument("--out-features", type=int, default=132)
    parser.add_argument("--compile-warmup", type=int, default=3)
    parser.add_argument("--measurement-warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--allow-populated-compile-cache",
        action="store_true",
        help=(
            "run without exact per-arm artifact attribution; the receipt is "
            "marked unqualified"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        raise RuntimeError("an SM120 or SM121 GPU is required")
    if (args.tokens, args.out_features, args.in_features) != (3, 132, 7168):
        raise ValueError(
            "this benchmark qualifies the fixed Kimi-K3 TP16 projection shape "
            "M=3, N=132, K=7168"
        )

    compile_cache_root, initial_artifacts = _compile_artifacts()
    if initial_artifacts and not args.allow_populated_compile_cache:
        raise RuntimeError(
            "B12X_COMPILE_CACHE_DIR must be empty for exact per-arm artifact "
            f"attribution, got {len(initial_artifacts)} objects under "
            f"{compile_cache_root}"
        )

    torch.manual_seed(args.seed)
    source = (
        torch.randn(
            (args.tokens, args.in_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn(
            (args.out_features, args.in_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_rows(weight_bf16)
    packed_weight = mxfp8_linear.pack_weight(weight, weight_scale)
    expected = _reference(source, packed_weight)

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    correctness: dict[str, object] = {}
    cold_samples: dict[str, float] = {}
    artifact_keys_by_plan: dict[str, list[str]] = {}
    observed_artifact_keys = set(initial_artifacts)
    for plan in PLANS:
        graph, output, cold_sample = _capture_plan(
            plan,
            source,
            packed_weight,
            warmup=args.compile_warmup,
        )
        _, observed_artifacts = _compile_artifacts()
        artifact_keys_by_plan[plan.name] = sorted(
            set(observed_artifacts) - observed_artifact_keys
        )
        observed_artifact_keys = set(observed_artifacts)
        close = torch.isclose(output, expected, rtol=1e-2, atol=2e-2)
        difference = (output.float() - expected.float()).abs()
        finite = bool(torch.isfinite(output).all().item())
        mismatches = int((~close).sum().item())
        if not finite or mismatches:
            raise RuntimeError(
                f"{plan.name} failed correctness: finite={finite}, "
                f"mismatches={mismatches}, max_abs_error={float(difference.max())}"
            )
        graphs[plan.name] = graph
        cold_samples[plan.name] = cold_sample
        correctness[plan.name] = {
            "finite": finite,
            "mismatches": mismatches,
            "max_abs_error": float(difference.max()),
            "reference_tolerance": {"rtol": 1e-2, "atol": 2e-2},
        }

    first, second = PLANS
    first_samples, second_samples = _balanced_samples_us(
        graphs[first.name].replay,
        graphs[second.name].replay,
        warmup=args.measurement_warmup,
        iterations=args.iterations,
    )
    first_result = _summary(first_samples)
    second_result = _summary(second_samples)
    first_median = float(first_result["median_microseconds"])
    second_median = float(second_result["median_microseconds"])
    winner = first.name if first_median <= second_median else second.name

    repository = pathlib.Path(__file__).resolve().parents[1]
    source_metadata = _source_metadata(repository)
    physical_gpu = _gpu_metadata()
    compile_cache_root, final_artifacts = _compile_artifacts()
    ptxas = _executable_metadata("ptxas")
    qualification_checks = {
        "source_revision_recorded": source_metadata["revision"] != "unavailable",
        "source_worktree_clean": source_metadata["worktree_state"] == "clean",
        "compile_cache_initially_empty": not initial_artifacts,
        "each_plan_bound_to_new_compile_artifact": all(
            artifact_keys_by_plan[plan.name] for plan in PLANS
        ),
        "compile_artifacts_hash_verified": bool(final_artifacts),
        "physical_gpu_in_p1": physical_gpu["pstate"] == "P1",
        "physical_gpu_default_compute_mode": (
            physical_gpu["compute_mode"] == "Default"
        ),
        "physical_gpu_unthrottled": (
            physical_gpu["active_throttle_mask"] == "0x0000000000000000"
        ),
        "correctness_passed": all(
            entry["finite"] and entry["mismatches"] == 0
            for entry in correctness.values()
        ),
        "raw_cold_samples_recorded": set(cold_samples) == {plan.name for plan in PLANS},
        "raw_warm_samples_recorded": (
            len(first_samples) == args.iterations
            and len(second_samples) == args.iterations
        ),
    }
    status = "qualified" if all(qualification_checks.values()) else "unqualified"
    result = {
        "schema": "b12x.mxfp8-linear-output-plan-benchmark.v1",
        "status": status,
        "qualification_checks": qualification_checks,
        "operation": (
            "mxfp8_linear.mm CUDA-graph replay including BF16 activation "
            "quantization and dense MXFP8 GEMM"
        ),
        "shape_mnk": [args.tokens, args.out_features, args.in_features],
        "measurement": {
            "order": "AB then BA, alternating once per recorded sample pair",
            "compile_warmup_per_arm": args.compile_warmup,
            "measurement_warmup_pairs": args.measurement_warmup,
            "recorded_sample_pairs": args.iterations,
            "timer": "CUDA events around one CUDA-graph replay",
            "cold_sample_definition": (
                "first timed graph replay immediately after graph capture and "
                "before balanced measurement warmup"
            ),
        },
        "plans": {
            first.name: {
                "mma_tiler_mn": list(first.mma_tiler_mn),
                "load_path": "tma",
                "swap_ab": first.swap_ab,
                "correctness": correctness[first.name],
                "cold_replay_samples_microseconds": [cold_samples[first.name]],
                **first_result,
            },
            second.name: {
                "mma_tiler_mn": list(second.mma_tiler_mn),
                "load_path": "tma",
                "swap_ab": second.swap_ab,
                "correctness": correctness[second.name],
                "cold_replay_samples_microseconds": [cold_samples[second.name]],
                **second_result,
            },
        },
        "comparison": {
            "kind": "within-revision output-plan comparison",
            "source_revisions": {
                first.name: source_metadata["revision"],
                second.name: source_metadata["revision"],
            },
            "winner": winner,
            "median_ratio_64x64_over_64x32": second_median / first_median,
            "ratio_direction": (
                "tma-64x64-swapped median divided by "
                "tma-64x32-swapped median; values above one mean 64x64 is slower"
            ),
        },
        "command": {
            "argv": [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            "shell": shlex.join(
                [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]]
            ),
        },
        "container_image": os.getenv(
            "B12X_BENCHMARK_CONTAINER_IMAGE",
            "unavailable",
        ),
        "source": source_metadata,
        "physical_gpu": physical_gpu,
        "cutlass_ptxas_artifact_map": {
            "compile_cache_root": str(compile_cache_root),
            "initial_cache_keys": sorted(initial_artifacts),
            "plan_new_cache_keys": artifact_keys_by_plan,
            "ptxas": ptxas,
            "compile_cache_telemetry": cute_compiler.compile_cache_info(),
            "artifacts": final_artifacts,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
