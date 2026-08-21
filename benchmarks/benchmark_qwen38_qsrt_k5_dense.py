#!/usr/bin/env python3
"""Qualify packed Qwen3.8-27B K5 dense linears with rank-16 corrections.

The benchmark loads one sealed Qwen3.8 dense-MLP QSRT K5 layer and its sealed
activation-weighted low-rank factors.  It compares B12X execution against an
independently decoded BF16 matrix for output and linear-Jacobian parity, proves
bit-exact eager/CUDA-graph replay, and measures packed versus decoded latency.

The decoded matrix is a correctness and timing control only.  The packed case
passes the native trellis payload directly to :mod:`b12x.gemm.trellis_linear`
and never materializes or dispatches through the decoded control.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from safetensors import safe_open
import torch

from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from b12x.gemm import trellis_linear


_RESULT_KIND = "b12x_qwen38_qsrt_k5_dense_qualification"
_SCHEMA_VERSION = 1
_DESCRIPTOR_SHA256 = "17cf4ca9ef1e3a07c3354c12f7ac887b4e081b1668bea61eb37d8f2b410bb968"
_ARTIFACT_MANIFEST_SHA256 = (
    "039845f8176afd254d0f028d3cd72791a5df9580d14ad524d7b3dbb828ddd052"
)
_INITIAL_ADAPTER_MANIFEST_SHA256 = (
    "1c6207cc41b2b3511bdd49970408fddf714419742a60d1ac053762c03f3a74c5"
)
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_RELATIVE_L2_LIMIT = 2.0e-2
_COSINE_LIMIT = 0.999


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "b12x/gemm/trellis_linear/__init__.py",
        root / "b12x/gemm/trellis_linear/_low_rank.py",
        root / "b12x/gemm/trellis_linear/api.py",
        root / "b12x/moe/_shared/kernels/w4a16/kernel.py",
        root / "b12x/moe/_shared/kernels/w4a16/prepare.py",
        root / "b12x/_lib/quant/sqg_fp16_d3l.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if (
        not result
        or len(set(result)) != len(result)
        or any(item <= 0 for item in result)
    ):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def _projection_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not result
        or len(set(result)) != len(result)
        or any(item not in _PROJECTIONS for item in result)
    ):
        raise argparse.ArgumentTypeError(
            f"projections must be a unique subset of {','.join(_PROJECTIONS)}"
        )
    return result


def _tensor_pointers(buffers: trellis_linear.Buffers) -> dict[str, int | None]:
    return {
        name: None if value is None else int(value.data_ptr())
        for name, value in vars(buffers).items()
    }


def _pair_tensor_pointers(
    buffers: trellis_linear.PairBuffers,
) -> dict[str, int | None]:
    result = {
        f"base.{name}": None if value is None else int(value.data_ptr())
        for name, value in vars(buffers.base).items()
    }
    result["output"] = int(buffers.output.data_ptr())
    result["low_rank_hidden"] = int(buffers.low_rank_hidden.data_ptr())
    return result


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual - expected).float()
    reference = expected.float()
    flat_error = difference.abs().flatten()
    reference_norm = float(reference.norm().item())
    relative_l2 = float(difference.norm().item()) / max(reference_norm, 1.0e-30)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference.flatten(),
            dim=0,
        ).item()
    )
    # CUDA's quantile implementation rejects tensors above its element-count
    # limit. Error metrics are outside the timed region, so preserve an exact
    # percentile by moving only oversized reductions to CPU.
    p99_absolute_error: float
    if flat_error.is_cuda and flat_error.numel() > (1 << 24):
        p99_absolute_error = float(np.quantile(flat_error.cpu().numpy(), 0.99))
    else:
        p99_absolute_error = float(torch.quantile(flat_error, 0.99).item())
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "mean_absolute_error": float(flat_error.mean().item()),
        "p99_absolute_error": p99_absolute_error,
        "maximum_absolute_error": float(flat_error.max().item()),
        "passed": (
            bool(torch.isfinite(actual).all())
            and relative_l2 <= _RELATIVE_L2_LIMIT
            and cosine >= _COSINE_LIMIT
        ),
    }


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "minimum_us": ordered[0],
        "maximum_us": ordered[-1],
    }


def _interleaved_times_us(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    replays: int,
    flush_l2: Callable[[], None] | None,
) -> dict[str, list[float]]:
    samples = {name: [] for name in graphs}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    names = list(graphs)
    for replay in range(replays):
        order = names if replay % 2 == 0 else list(reversed(names))
        for name in order:
            if flush_l2 is not None:
                flush_l2()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            events.append((name, begin, end))
    torch.cuda.synchronize()
    for name, begin, end in events:
        samples[name].append(begin.elapsed_time(end) * 1000.0)
    return samples


def _load_qsrt_contract(qsrt_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(qsrt_root))
    from qsrt.qwen38_adapter_archive import (  # noqa: PLC0415
        load_layer_factors,
        validate_final_report as validate_adapter_report,
    )
    from qsrt.qwen38_mlp import (  # noqa: PLC0415
        PROJECTION_BY_NAME,
        decode_qsrt_projection,
    )
    from qsrt.qwen38_mlp_artifact import (  # noqa: PLC0415
        layer_filename,
        validate_completed_layer,
        validate_final_report as validate_artifact_report,
    )
    from qsrt.qwen38_serving_checkpoint import (  # noqa: PLC0415
        validate_recovery_selection,
    )

    return {
        "load_layer_factors": load_layer_factors,
        "validate_adapter_report": validate_adapter_report,
        "projection_by_name": PROJECTION_BY_NAME,
        "decode_qsrt_projection": decode_qsrt_projection,
        "layer_filename": layer_filename,
        "validate_completed_layer": validate_completed_layer,
        "validate_artifact_report": validate_artifact_report,
        "validate_recovery_selection": validate_recovery_selection,
    }


def _validate_archives(
    contract: dict[str, Any],
    *,
    artifact_root: Path,
    adapter_root: Path,
    recovery_root: Path | None,
    quality_selection_report: Path | None,
    expected_adapter_manifest_sha256: str,
) -> dict[str, Any]:
    artifact_manifest, _, artifact_manifest_sha256 = contract[
        "validate_artifact_report"
    ](artifact_root, verify_layer_hashes=False)
    adapter_manifest, adapter_report, adapter_manifest_sha256 = contract[
        "validate_adapter_report"
    ](adapter_root, verify_layer_hashes=False)
    if (
        artifact_manifest_sha256 != _ARTIFACT_MANIFEST_SHA256
        or artifact_manifest.get("codec", {}).get("descriptor_sha256")
        != _DESCRIPTOR_SHA256
        or adapter_manifest_sha256 != expected_adapter_manifest_sha256
        or adapter_manifest.get("k5_artifact", {}).get("manifest_sha256")
        != artifact_manifest_sha256
        or "weighted" not in adapter_report.get("variants", [])
        or 16 not in adapter_report.get("ranks", [])
    ):
        raise ValueError("Qwen K5 artifact and rank-16 adapter identities do not close")
    identity: dict[str, Any] = {
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_report_sha256": _sha256(artifact_root / "report.json"),
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "adapter_report_sha256": _sha256(adapter_root / "report.json"),
        "factor_source": "sealed activation-weighted rank-16 initialization",
    }
    if recovery_root is not None:
        selection = contract["validate_recovery_selection"](
            recovery_root,
            adapter_manifest_sha256=adapter_manifest_sha256,
            quality_selection_report=quality_selection_report,
        )
        overlay_path = selection.overlay
        with safe_open(str(overlay_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("kind") != "qwen38_full_depth_mlp_adapter_overlay_v1"
                or metadata.get("rank") != "16"
                or metadata.get("variant") != "weighted"
            ):
                raise ValueError("selected Qwen recovery overlay metadata differs")
        identity.update(
            {
                "factor_source": (
                    "rank-16 dense-MLP factors selected by the completed "
                    "quality-analysis report"
                    if quality_selection_report is not None
                    else (
                        "rank-16 dense-MLP factors selected by the completed "
                        "full-depth quantization-aware recovery report"
                    )
                ),
                "recovery_report": str(recovery_root / "complete.json"),
                "recovery_report_sha256": selection.report_sha256,
                "selected_overlay": str(overlay_path),
                "selected_overlay_sha256": selection.overlay_sha256,
            }
        )
        if selection.quality_selection_path is not None:
            identity["quality_selection_report"] = str(
                selection.quality_selection_path
            )
            identity["quality_selection_report_sha256"] = (
                selection.quality_selection_sha256
            )
            identity["selected_optimizer_step"] = selection.quality_selection[
                "selected_overlay"
            ]["step"]
    return identity


def _load_projection(
    contract: dict[str, Any],
    *,
    artifact_root: Path,
    adapter_root: Path,
    selected_overlay: Path | None,
    layer: int,
    projection: str,
    device: torch.device,
    artifact_manifest_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    receipt = contract["validate_completed_layer"](
        artifact_root,
        layer=layer,
        manifest_sha256=artifact_manifest_sha256,
        verify_file_hash=True,
    )
    layer_path = artifact_root / "layers" / contract["layer_filename"](layer)
    prefix = f"{projection}.qsrt_k5"
    with safe_open(str(layer_path), framework="pt", device="cpu") as handle:
        tensors = {
            name: handle.get_tensor(f"{prefix}.{name}").contiguous()
            for name in ("trellis", "suh", "svh")
        }
    if selected_overlay is None:
        factors = contract["load_layer_factors"](
            adapter_root,
            layer=layer,
            variant="weighted",
            rank=16,
        )[projection]
    else:
        factor_prefix = f"model.language_model.layers.{layer}.mlp.{projection}"
        with safe_open(str(selected_overlay), framework="pt", device="cpu") as handle:
            factors = (
                handle.get_tensor(f"{factor_prefix}.a"),
                handle.get_tensor(f"{factor_prefix}.b"),
            )
    replay = {
        "bits": 5,
        **{name: tensor.to(device) for name, tensor in tensors.items()},
    }
    decoded_source = contract["decode_qsrt_projection"](
        replay,
        contract["projection_by_name"][projection],
        bits=5,
    )
    gpu = {
        **replay,
        "a": factors[0].to(device).contiguous(),
        "b": factors[1].to(device).contiguous(),
        "decoded_kn": decoded_source.T.to(
            device=device, dtype=torch.bfloat16
        ).contiguous(),
    }
    provenance = {
        "layer_file": str(layer_path),
        "layer_file_bytes": layer_path.stat().st_size,
        "layer_file_sha256": receipt["payload_sha256"],
        "projection": projection,
        "trellis_shape": list(gpu["trellis"].shape),
        "suh_shape": list(gpu["suh"].shape),
        "svh_shape": list(gpu["svh"].shape),
        "a_shape": list(gpu["a"].shape),
        "b_shape": list(gpu["b"].shape),
        "decoded_control_shape": list(gpu["decoded_kn"].shape),
    }
    return gpu, provenance


@torch.inference_mode()
def _run_case(
    tensors: dict[str, torch.Tensor],
    *,
    rows: int,
    warmup: int,
    hot_replays: int,
    cold_replays: int,
    seed: int,
) -> dict[str, Any]:
    device = tensors["trellis"].device
    base = trellis_linear.prepare_weight(
        tensors["trellis"],
        tensors["suh"],
        tensors["svh"],
        codebook="sqg_fp16",
        params_dtype=torch.bfloat16,
    )
    weight = trellis_linear.prepare_additive_weight(base, tensors["a"], tensors["b"])
    packed_buffers = trellis_linear.make_buffers(
        weight,
        size_m=rows,
        input_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        (rows, int(base.in_features)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    direction = torch.randn(
        (rows, int(base.in_features)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)

    def packed_run() -> torch.Tensor:
        return trellis_linear.run_additive(
            x,
            weight,
            low_rank_hidden=packed_buffers.low_rank_hidden,
            **packed_buffers.run_kwargs(),
        )

    for _ in range(warmup):
        packed_run()
    torch.cuda.synchronize(device)
    trellis_before = tensors["trellis"].clone()
    a_t_before = weight.a_t.clone()
    b_before = weight.b.clone()
    eager = packed_run().clone()
    pointers_before = _tensor_pointers(packed_buffers)
    packed_buffers.output.fill_(float("nan"))
    packed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(packed_graph):
        captured = packed_run()
    torch.cuda.synchronize(device)
    capture_preserves_poison = bool(torch.isnan(packed_buffers.output).all())
    allocated_before_replay = torch.cuda.memory_allocated(device)
    packed_graph.replay()
    packed_graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)
    replay_matches_eager = bool(torch.equal(captured, eager))
    pointers_after = _tensor_pointers(packed_buffers)
    immutable_weights = bool(
        torch.equal(tensors["trellis"], trellis_before)
        and torch.equal(weight.a_t, a_t_before)
        and torch.equal(weight.b, b_before)
    )

    reference_output = torch.empty_like(packed_buffers.output)
    reference_hidden = torch.empty(
        (rows, weight.rank),
        dtype=torch.bfloat16,
        device=device,
    )
    reference_correction = torch.empty_like(reference_output)

    def reference_run() -> torch.Tensor:
        torch.mm(x, tensors["decoded_kn"], out=reference_output)
        torch.mm(x, tensors["a"], out=reference_hidden)
        torch.mm(reference_hidden, tensors["b"].T, out=reference_correction)
        reference_output.add_(reference_correction)
        return reference_output

    for _ in range(warmup):
        reference_run()
    expected = reference_run().clone()
    reference_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(reference_graph):
        reference_run()
    torch.cuda.synchronize(device)

    jvp_buffers = trellis_linear.make_buffers(
        weight,
        size_m=rows,
        input_dtype=torch.bfloat16,
    )
    actual_jvp = trellis_linear.run_additive(
        direction,
        weight,
        low_rank_hidden=jvp_buffers.low_rank_hidden,
        **jvp_buffers.run_kwargs(),
    ).clone()
    expected_jvp = direction @ tensors["decoded_kn"]
    expected_jvp.add_((direction @ tensors["a"]) @ tensors["b"].T)
    torch.cuda.synchronize(device)

    hot = _interleaved_times_us(
        {"packed_k5_rank16": packed_graph, "decoded_bf16_rank16": reference_graph},
        replays=hot_replays,
        flush_l2=None,
    )
    cold = None
    if cold_replays:
        cold = _interleaved_times_us(
            {"packed_k5_rank16": packed_graph, "decoded_bf16_rank16": reference_graph},
            replays=cold_replays,
            flush_l2=make_l2_flush_fn(True),
        )
    output_metrics = _metrics(eager, expected)
    jvp_metrics = _metrics(actual_jvp, expected_jvp)
    correctness = {
        "output": output_metrics,
        "linear_jacobian_vector_product": jvp_metrics,
        "capture_preserves_output_poison": capture_preserves_poison,
        "replay_matches_eager_bit_exact": replay_matches_eager,
        "buffer_pointers_stable": pointers_before == pointers_after,
        "replay_allocation_stable": allocated_before_replay == allocated_after_replay,
        "packed_weight_and_factors_immutable": immutable_weights,
    }
    correctness["passed"] = bool(
        output_metrics["passed"]
        and jvp_metrics["passed"]
        and capture_preserves_poison
        and replay_matches_eager
        and pointers_before == pointers_after
        and allocated_before_replay == allocated_after_replay
        and immutable_weights
    )
    timing: dict[str, Any] = {
        "hot": {name: _summary(values) for name, values in hot.items()},
        "hot_samples_us": hot,
        "hot_packed_over_decoded": (
            statistics.median(hot["packed_k5_rank16"])
            / statistics.median(hot["decoded_bf16_rank16"])
        ),
    }
    if cold is not None:
        timing["cold"] = {name: _summary(values) for name, values in cold.items()}
        timing["cold_samples_us"] = cold
        timing["cold_packed_over_decoded"] = statistics.median(
            cold["packed_k5_rank16"]
        ) / statistics.median(cold["decoded_bf16_rank16"])
    return {
        "rows": rows,
        "correctness": correctness,
        "timing": timing,
        "storage": {
            "packed_trellis_bytes": tensors["trellis"].numel()
            * tensors["trellis"].element_size(),
            "rotation_bytes": sum(
                tensors[name].numel() * tensors[name].element_size()
                for name in ("suh", "svh")
            ),
            "adapter_bytes": sum(
                tensors[name].numel() * tensors[name].element_size()
                for name in ("a", "b")
            ),
            "decoded_control_bytes": tensors["decoded_kn"].numel()
            * tensors["decoded_kn"].element_size(),
            "scratch_bytes": sum(
                value.numel() * value.element_size()
                for value in vars(packed_buffers).values()
                if isinstance(value, torch.Tensor)
            ),
        },
    }


@torch.inference_mode()
def _run_pair_case(
    gate: dict[str, torch.Tensor],
    up: dict[str, torch.Tensor],
    *,
    rows: int,
    warmup: int,
    hot_replays: int,
    cold_replays: int,
    seed: int,
) -> dict[str, Any]:
    device = gate["trellis"].device
    bases = [
        trellis_linear.prepare_weight(
            tensors["trellis"],
            tensors["suh"],
            tensors["svh"],
            codebook="sqg_fp16",
            params_dtype=torch.bfloat16,
        )
        for tensors in (gate, up)
    ]
    a = torch.stack((gate["a"], up["a"]), dim=0).contiguous()
    b = torch.stack((gate["b"], up["b"]), dim=0).contiguous()
    weight = trellis_linear.prepare_additive_pair(*bases, a, b)
    packed_buffers = trellis_linear.make_pair_buffers(
        weight,
        size_m=rows,
        input_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        (rows, int(weight.left.in_features)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    direction = torch.randn(
        (rows, int(weight.left.in_features)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)

    def packed_run() -> torch.Tensor:
        return trellis_linear.run_additive_pair(
            x,
            weight,
            buffers=packed_buffers,
        )

    for _ in range(warmup):
        packed_run()
    torch.cuda.synchronize(device)
    trellis_before = [value["trellis"].clone() for value in (gate, up)]
    a_t_before = weight.a_t.clone()
    b_before = weight.b.clone()
    eager = packed_run().clone()
    pointers_before = _pair_tensor_pointers(packed_buffers)
    packed_buffers.output.fill_(float("nan"))
    packed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(packed_graph):
        captured = packed_run()
    torch.cuda.synchronize(device)
    capture_preserves_poison = bool(torch.isnan(packed_buffers.output).all())
    allocated_before_replay = torch.cuda.memory_allocated(device)
    packed_graph.replay()
    packed_graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)
    replay_matches_eager = bool(torch.equal(captured, eager))
    pointers_after = _pair_tensor_pointers(packed_buffers)
    immutable_weights = bool(
        all(
            torch.equal(tensors["trellis"], before)
            for tensors, before in zip((gate, up), trellis_before, strict=True)
        )
        and torch.equal(weight.a_t, a_t_before)
        and torch.equal(weight.b, b_before)
    )

    width = int(weight.left.out_features)
    decoded_pair = torch.cat((gate["decoded_kn"], up["decoded_kn"]), dim=1)
    a_pair = torch.cat((gate["a"], up["a"]), dim=1)
    b_block = torch.zeros(
        (2 * width, 2 * weight.rank),
        dtype=torch.bfloat16,
        device=device,
    )
    b_block[:width, : weight.rank].copy_(gate["b"])
    b_block[width:, weight.rank :].copy_(up["b"])
    reference_output = torch.empty_like(packed_buffers.output)
    reference_hidden = torch.empty(
        (rows, 2 * weight.rank),
        dtype=torch.bfloat16,
        device=device,
    )
    reference_correction = torch.empty_like(reference_output)

    def reference_run() -> torch.Tensor:
        torch.mm(x, decoded_pair, out=reference_output)
        torch.mm(x, a_pair, out=reference_hidden)
        torch.mm(reference_hidden, b_block.T, out=reference_correction)
        reference_output.add_(reference_correction)
        return reference_output

    for _ in range(warmup):
        reference_run()
    expected = reference_run().clone()
    reference_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(reference_graph):
        reference_run()
    torch.cuda.synchronize(device)

    jvp_buffers = trellis_linear.make_pair_buffers(
        weight,
        size_m=rows,
        input_dtype=torch.bfloat16,
    )
    actual_jvp = trellis_linear.run_additive_pair(
        direction,
        weight,
        buffers=jvp_buffers,
    ).clone()
    expected_jvp = direction @ decoded_pair
    expected_jvp.add_((direction @ a_pair) @ b_block.T)
    torch.cuda.synchronize(device)

    names = {
        "packed_pair_k5_rank16": packed_graph,
        "decoded_fused_bf16_rank16": reference_graph,
    }
    hot = _interleaved_times_us(names, replays=hot_replays, flush_l2=None)
    cold = None
    if cold_replays:
        cold = _interleaved_times_us(
            names,
            replays=cold_replays,
            flush_l2=make_l2_flush_fn(True),
        )
    output_metrics = _metrics(eager, expected)
    jvp_metrics = _metrics(actual_jvp, expected_jvp)
    correctness = {
        "output": output_metrics,
        "linear_jacobian_vector_product": jvp_metrics,
        "capture_preserves_output_poison": capture_preserves_poison,
        "replay_matches_eager_bit_exact": replay_matches_eager,
        "buffer_pointers_stable": pointers_before == pointers_after,
        "replay_allocation_stable": allocated_before_replay == allocated_after_replay,
        "packed_weights_and_factors_immutable": immutable_weights,
    }
    correctness["passed"] = bool(
        output_metrics["passed"]
        and jvp_metrics["passed"]
        and capture_preserves_poison
        and replay_matches_eager
        and pointers_before == pointers_after
        and allocated_before_replay == allocated_after_replay
        and immutable_weights
    )
    timing: dict[str, Any] = {
        "hot": {name: _summary(values) for name, values in hot.items()},
        "hot_samples_us": hot,
        "hot_packed_over_decoded": (
            statistics.median(hot["packed_pair_k5_rank16"])
            / statistics.median(hot["decoded_fused_bf16_rank16"])
        ),
    }
    if cold is not None:
        timing["cold"] = {name: _summary(values) for name, values in cold.items()}
        timing["cold_samples_us"] = cold
        timing["cold_packed_over_decoded"] = statistics.median(
            cold["packed_pair_k5_rank16"]
        ) / statistics.median(cold["decoded_fused_bf16_rank16"])
    return {
        "rows": rows,
        "correctness": correctness,
        "timing": timing,
        "storage": {
            "packed_trellis_bytes": sum(
                tensors["trellis"].numel() * tensors["trellis"].element_size()
                for tensors in (gate, up)
            ),
            "rotation_bytes": sum(
                tensors[name].numel() * tensors[name].element_size()
                for tensors in (gate, up)
                for name in ("suh", "svh")
            ),
            "adapter_bytes": a.numel() * a.element_size()
            + b.numel() * b.element_size(),
            "decoded_control_bytes": decoded_pair.numel() * decoded_pair.element_size(),
            "scratch_bytes": sum(
                value.numel() * value.element_size()
                for value in vars(packed_buffers.base).values()
                if isinstance(value, torch.Tensor)
            )
            + packed_buffers.output.numel() * packed_buffers.output.element_size()
            + packed_buffers.low_rank_hidden.numel()
            * packed_buffers.low_rank_hidden.element_size(),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument(
        "--recovery-root",
        type=Path,
        help=(
            "completed full-depth quantization-aware recovery directory whose "
            "report selects the factor overlay"
        ),
    )
    parser.add_argument(
        "--quality-selection-report",
        type=Path,
        help=(
            "content-bound analysis report that selects one saved optimizer "
            "boundary instead of the trainer's screening choice"
        ),
    )
    parser.add_argument("--qsrt-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--projections",
        type=_projection_names,
        default=_projection_names(",".join(_PROJECTIONS)),
    )
    parser.add_argument(
        "--rows", type=_positive_ints, default=_positive_ints("1,8,32,128")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--hot-replays", type=int, default=100)
    parser.add_argument("--cold-replays", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--expected-adapter-manifest-sha256",
        default=_INITIAL_ADAPTER_MANIFEST_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not 0 <= args.layer < 64
        or args.warmup <= 0
        or args.hot_replays <= 0
        or args.cold_replays < 0
        or len(args.expected_adapter_manifest_sha256) != 64
    ):
        raise SystemExit(
            "layer, replay counts, or adapter manifest identity is invalid"
        )
    if args.quality_selection_report is not None and args.recovery_root is None:
        raise SystemExit("--quality-selection-report requires --recovery-root")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")

    device = torch.device("cuda", torch.cuda.current_device())
    contract = _load_qsrt_contract(args.qsrt_root.expanduser().resolve())
    artifact_root = args.artifact_root.expanduser().resolve()
    adapter_root = args.adapter_root.expanduser().resolve()
    recovery_root = (
        None
        if args.recovery_root is None
        else args.recovery_root.expanduser().resolve()
    )
    quality_selection_report = (
        None
        if args.quality_selection_report is None
        else args.quality_selection_report.expanduser().resolve()
    )
    archive_identity = _validate_archives(
        contract,
        artifact_root=artifact_root,
        adapter_root=adapter_root,
        recovery_root=recovery_root,
        quality_selection_report=quality_selection_report,
        expected_adapter_manifest_sha256=args.expected_adapter_manifest_sha256,
    )
    selected_overlay = (
        None
        if recovery_root is None
        else Path(str(archive_identity["selected_overlay"]))
    )
    properties = torch.cuda.get_device_properties(device)
    result: dict[str, Any] = {
        "kind": _RESULT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "status": "running",
        "classification": "qualification",
        "created_unix_ns": time.time_ns(),
        "provenance": {
            "command": shlex.join(
                [sys.executable, *(sys.argv if argv is None else argv)]
            ),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "source_sha256": _source_sha256(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "qsrt_root": str(args.qsrt_root.expanduser().resolve()),
        },
        "contract": {
            "model": "Qwen/Qwen3.8-27B",
            "scope": "all selected dense decoder MLP projections at TP1",
            "layer": args.layer,
            "projections": list(args.projections),
            "rows": list(args.rows),
            "codebook": "sqg_fp16_d3l",
            "rate": 5,
            "adapter_variant": "weighted",
            "adapter_rank": 16,
            "output_relative_l2_limit": _RELATIVE_L2_LIMIT,
            "output_cosine_limit": _COSINE_LIMIT,
            "jvp_relative_l2_limit": _RELATIVE_L2_LIMIT,
            "jvp_cosine_limit": _COSINE_LIMIT,
            "eager_graph_requirement": "bit exact",
            "decoded_weight_use": "independent control only; absent from packed closure",
            "gate_up_factor_execution": (
                "one shared rank-16 projection launch and one shared output launch"
            ),
            "gate_up_factor_kernel_launches_expected": 2,
        },
        "archives": archive_identity,
        "device": {
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "capability": [major, minor],
            "torch": torch.__version__,
            "mode": nvidia_smi_gpu_mode_snapshot(),
        },
        "cases": [],
    }
    loaded: dict[
        str,
        tuple[dict[str, torch.Tensor], dict[str, Any]],
    ] = {}
    for projection_index, projection in enumerate(args.projections):
        tensors, projection_identity = _load_projection(
            contract,
            artifact_root=artifact_root,
            adapter_root=adapter_root,
            selected_overlay=selected_overlay,
            layer=args.layer,
            projection=projection,
            device=device,
            artifact_manifest_sha256=archive_identity["artifact_manifest_sha256"],
        )
        loaded[projection] = (tensors, projection_identity)
        projection_cases = []
        for rows in args.rows:
            case = _run_case(
                tensors,
                rows=rows,
                warmup=args.warmup,
                hot_replays=args.hot_replays,
                cold_replays=args.cold_replays,
                seed=args.seed + 1000 * projection_index + rows,
            )
            projection_cases.append(case)
        result["cases"].append(
            {
                "identity": projection_identity,
                "rows": projection_cases,
                "passed": all(
                    case["correctness"]["passed"] for case in projection_cases
                ),
            }
        )
    paired_gate_up = None
    if {"gate_proj", "up_proj"} <= loaded.keys():
        pair_cases = [
            _run_pair_case(
                loaded["gate_proj"][0],
                loaded["up_proj"][0],
                rows=rows,
                warmup=args.warmup,
                hot_replays=args.hot_replays,
                cold_replays=args.cold_replays,
                seed=args.seed + 900000 + rows,
            )
            for rows in args.rows
        ]
        paired_gate_up = {
            "semantic_role": "vLLM gate_up_proj packed execution",
            "projections": [
                loaded["gate_proj"][1],
                loaded["up_proj"][1],
            ],
            "rows": pair_cases,
            "passed": all(case["correctness"]["passed"] for case in pair_cases),
        }
        result["paired_gate_up"] = paired_gate_up
    passed = all(case["passed"] for case in result["cases"])
    if paired_gate_up is not None:
        passed = passed and paired_gate_up["passed"]
    result["status"] = "passed" if passed else "failed"
    del loaded
    torch.cuda.empty_cache()
    result["completed_unix_ns"] = time.time_ns()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
