"""Probe pre-quantized dense-GEMM plans on the Blackwell SM120 architecture.

The default shape represents one tensor-parallel projection with 7,168 input
features and 132 output features per rank. The measurement is a plan-level
proxy that calls ``dense_gemm`` with pre-quantized operands; it does not measure
the complete ``mxfp8_linear`` serving operation. Every plan must reproduce the
dequantized reference before timing begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from cuda.bindings import runtime as cudart

from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.block_fp8 import (
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
from b12x.gemm.mxfp8_linear._kernel import pack_mxfp8_linear_weight


@dataclass(frozen=True)
class Plan:
    name: str
    tile: tuple[int, int]
    load_path: str
    swap_ab: bool


PLANS = (
    Plan("tma-16x64-unswapped", (16, 64), "tma", False),
    Plan("tma-32x64-unswapped", (32, 64), "tma", False),
    Plan("tma-64x64-unswapped", (64, 64), "tma", False),
    Plan("tma-16x128-unswapped", (16, 128), "tma", False),
    Plan("tma-32x128-unswapped", (32, 128), "tma", False),
    Plan("tma-64x128-unswapped", (64, 128), "tma", False),
    Plan("tma-64x32-swapped", (64, 32), "tma", True),
    Plan("tma-64x64-swapped", (64, 64), "tma", True),
    Plan("cpasync-64x64-unswapped", (64, 64), "cpasync", False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_metadata() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__).resolve(),
        repository / "b12x/_lib/dense_gemm.py",
        repository / "b12x/gemm/mxfp8_linear/_kernel.py",
    )
    try:
        revision = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
        status = _command_output(["git", "-C", str(repository), "status", "--short"])
        worktree_state = "clean" if not status else "modified"
        revision_source = "git"
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
            str(path.relative_to(repository)): _sha256(path) for path in source_paths
        },
    }


def _canonical_pci_bus_id(value: str) -> str:
    domain, bus, device = value.strip().lower().split(":")
    return f"{int(domain, 16):08x}:{bus}:{device}"


def _gpu_metadata() -> dict[str, object]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    error, pci_bus_id_buffer = cudart.cudaDeviceGetPCIBusId(32, device_index)
    if int(error) != 0:
        raise RuntimeError(
            "cudaDeviceGetPCIBusId failed for logical CUDA device "
            f"{device_index}: {error}"
        )
    pci_bus_id = _canonical_pci_bus_id(
        pci_bus_id_buffer.split(b"\0", 1)[0].decode("ascii")
    )
    rows = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,pstate,compute_mode",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    selected = None
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) >= 3 and _canonical_pci_bus_id(fields[2]) == pci_bus_id:
            selected = row
            break
    if selected is None:
        raise RuntimeError(
            "nvidia-smi did not report the physical GPU selected by CUDA at "
            f"PCI bus ID {pci_bus_id}"
        )
    return {
        "cuda_device_index": device_index,
        "cuda_pci_bus_id": pci_bus_id,
        "compute_capability": [properties.major, properties.minor],
        "torch_device_name": properties.name,
        "nvidia_smi_record": selected,
    }


def _time_launch(call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def _quantize_modelopt_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _launch(
    plan: Plan,
    x_q,
    packed_weight,
    output: torch.Tensor,
    *,
    tokens: int,
    in_features: int,
) -> torch.Tensor:
    return dense_gemm(
        (
            x_q.values.reshape(tokens, in_features, 1),
            x_q.scale_mma,
        ),
        (
            packed_weight.weight.values.reshape(
                packed_weight.out_features,
                packed_weight.padded_in_features,
                1,
            ),
            packed_weight.weight.scale_mma,
        ),
        out=output,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=tokens,
        mma_tiler_mn=plan.tile,
        load_path=plan.load_path,
        swap_ab=plan.swap_ab,
    )[:, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=7168)
    parser.add_argument("--out-features", type=int, default=132)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--plan",
        action="append",
        choices=tuple(plan.name for plan in PLANS),
        help="Run only the named plan; repeat the option to select several plans.",
    )
    args = parser.parse_args()

    torch.manual_seed(20260814)
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
    packed_weight = pack_mxfp8_linear_weight(weight, weight_scale)
    source_q = quantize_block_fp8_linear_input_mxfp8(source)
    source_dequantized = dequantize_mxfp8_rows_torch(
        source_q.values, source_q.scale_rows
    )
    weight_dequantized = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values,
        packed_weight.weight.scale_rows,
    )
    reference = (source_dequantized @ weight_dequantized.T).to(torch.bfloat16)

    results: list[dict[str, object]] = []
    selected_plans = (
        tuple(plan for plan in PLANS if plan.name in args.plan) if args.plan else PLANS
    )
    for plan in selected_plans:
        output = torch.full(
            (args.tokens, args.out_features, 1),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        try:
            actual: torch.Tensor | None = None

            def launch() -> torch.Tensor:
                nonlocal actual
                actual = _launch(
                    plan,
                    source_q,
                    packed_weight,
                    output,
                    tokens=args.tokens,
                    in_features=packed_weight.padded_in_features,
                )
                return actual

            pre_warmup_milliseconds = _time_launch(launch)
            assert actual is not None
            finite = bool(torch.isfinite(actual).all().item())
            close = torch.isclose(actual, reference, rtol=0.0, atol=0.0)
            mismatches = int((~close).sum().item())
            correct = finite and mismatches == 0
            difference = torch.nan_to_num(
                (actual.float() - reference.float()).abs(),
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            )
            max_abs_error = float(difference.max())
            row_mismatches = [
                int((~close[row]).sum().item()) for row in range(args.tokens)
            ]
            result: dict[str, object] = {
                "plan": plan.name,
                "correct": correct,
                "finite": finite,
                "reference_tolerance": {"rtol": 0.0, "atol": 0.0},
                "mismatches": mismatches,
                "max_abs_error": max_abs_error,
                "row_mismatches": row_mismatches,
                "pre_warmup_samples_milliseconds": [pre_warmup_milliseconds],
            }
            if not correct:
                result["warm_samples_milliseconds"] = None
                results.append(result)
                continue

            for _ in range(args.warmup):
                launch()
            torch.cuda.synchronize()
            warm_samples = [_time_launch(launch) for _ in range(args.repeats)]
            sorted_samples = sorted(warm_samples)
            result.update(
                {
                    "warm_samples_milliseconds": warm_samples,
                    "median_milliseconds": sorted_samples[len(sorted_samples) // 2],
                }
            )
            results.append(result)
        except ValueError as exc:
            results.append(
                {
                    "plan": plan.name,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    timed_results = [
        result
        for result in results
        if isinstance(result.get("median_milliseconds"), float)
    ]
    if timed_results:
        fastest = min(float(result["median_milliseconds"]) for result in timed_results)
        for result in timed_results:
            result["time_ratio_to_fastest"] = (
                float(result["median_milliseconds"]) / fastest
            )

    print(
        json.dumps(
            {
                "artifact_kind": (
                    "plan-level proxy for pre-quantized direct dense_gemm; "
                    "not an mxfp8_linear serving-path benchmark"
                ),
                "command": {
                    "argv": [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        *sys.argv[1:],
                    ],
                    "shell": shlex.join(
                        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
                    ),
                },
                "source": _source_metadata(),
                "physical_gpu": _gpu_metadata(),
                "shape": [args.tokens, args.out_features, args.in_features],
                "warmup": args.warmup,
                "repeats": args.repeats,
                "ratio_definition": (
                    "plan median milliseconds divided by the fastest valid "
                    "plan median; values greater than one are slower"
                ),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
