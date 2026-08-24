#!/usr/bin/env python3
"""Compare a real packed QSRT layer with decoded-weight replay.

The packed side invokes the production W4A16 MoE preparation and execution
path. The reference side decodes the same rank-local payload into BF16 weight
matrices and replays the coupled activation-boundary transform. Inputs are
real routed-MoE rows from an all-row capture; each sampled row is evaluated as
a single-expert route so the comparison isolates packed expert execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open
import torch

from benchmarks.benchmark_qsrt_checkpoint_profiles import (
    _K2,
    _prepare_profile,
    _read_layer_source,
)
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from tests.moe.test_fused_moe_trellis import (
    _reconstruct_native,
    _reference_coupled_decoded,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def _capture_chunk(root: Path, layer: int, chunk: int) -> Path:
    path = root / f"layer-{layer:05d}" / f"chunk-{chunk:08d}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sample_distinct_top1_rows(
    chunk: Path, count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with safe_open(chunk, framework="pt", device="cpu") as handle:
        inputs = handle.get_tensor("input")
        experts = handle.get_tensor("expert_indices")[:, 0].to(torch.int64)
    if count <= 0:
        raise ValueError("--rows must be positive")
    stride = max(1, int(inputs.shape[0]) // (count * 8))
    selected_rows: list[int] = []
    selected_experts: set[int] = set()
    for row in range(0, int(inputs.shape[0]), stride):
        expert = int(experts[row])
        if expert in selected_experts:
            continue
        selected_rows.append(row)
        selected_experts.add(expert)
        if len(selected_rows) == count:
            break
    if len(selected_rows) != count:
        raise RuntimeError(
            f"capture chunk contains only {len(selected_rows)} sampled distinct experts"
        )
    rows = torch.tensor(selected_rows, dtype=torch.int64)
    return inputs.index_select(0, rows), experts.index_select(0, rows), rows


def _decode_experts(
    prepared: object, experts: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w13 = prepared.w13.view(torch.int16).reshape(2, 896, 224, 16, 32)
    w2 = prepared.w2.view(torch.int16).reshape(896, 16, 224, 32)
    gate: list[torch.Tensor] = []
    up: list[torch.Tensor] = []
    down: list[torch.Tensor] = []
    for expert in experts.tolist():
        gate.append(_reconstruct_native(w13[0, expert], codebook="sqg_e4m3"))
        up.append(_reconstruct_native(w13[1, expert], codebook="sqg_e4m3"))
        down.append(_reconstruct_native(w2[expert], codebook="sqg_e4m3"))
    return tuple(
        torch.stack(values).to(device=device, dtype=torch.bfloat16).contiguous()
        for values in (gate, up, down)
    )


def _production_output(
    source: torch.Tensor,
    experts: torch.Tensor,
    prepared: object,
) -> torch.Tensor:
    device = source.device
    ids = experts.to(device=device, dtype=torch.int32).view(-1, 1).contiguous()
    weights = torch.ones((int(source.shape[0]), 1), device=device, dtype=torch.float32)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=int(source.shape[0]),
        topk=1,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            source,
            prepared,
            weights,
            ids,
            activation="situ",
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
            route_block_size_m=8,
            intermediate_rotation_scales=prepared.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=prepared.gate_suh,
            suh_up_table=prepared.up_suh,
            svh_table=prepared.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    run()
    result = run().clone()
    torch.cuda.synchronize(device)
    return result


def _reference_output(
    source: torch.Tensor,
    experts: torch.Tensor,
    prepared: object,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    device = source.device
    count = int(experts.numel())
    local_ids = torch.arange(count, device=device, dtype=torch.int32).view(-1, 1)
    weights = torch.ones((count, 1), device=device, dtype=torch.float32)
    rotations = prepared.intermediate_rotations.index_select(
        0, experts.to(device=device, dtype=torch.int64)
    )
    return _reference_coupled_decoded(
        source,
        local_ids,
        weights,
        gate,
        up,
        down,
        prepared.gate_suh,
        rotations[:, : 3 * int(prepared.intermediate_size)],
        rotations[:, 3 * int(prepared.intermediate_size) :],
        prepared.down_svh,
        quantize_activations=False,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> None:
    args = _parser().parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    chunk = _capture_chunk(args.capture, args.layer, args.chunk)
    source_cpu, experts, row_indices = _sample_distinct_top1_rows(chunk, args.rows)
    layer_source, provenance = _read_layer_source(args.checkpoint, _K2, args.layer)
    prepared, preparation = _prepare_profile(
        layer_source, tp_rank=args.tp_rank, device=device
    )
    gate, up, down = _decode_experts(prepared, experts, device)
    source = source_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    packed = _production_output(source, experts, prepared)
    reference = _reference_output(source, experts, prepared, gate, up, down)
    delta = packed.float() - reference.float()
    reference_tiles = reference.float().reshape(int(args.rows), -1, 16)
    delta_tiles = delta.reshape(int(args.rows), -1, 16)
    max_abs = delta_tiles.abs().amax(dim=-1)
    relative_l2 = delta_tiles.norm(dim=-1) / reference_tiles.norm(dim=-1).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    tile_records = []
    for row in range(int(args.rows)):
        for tile in range(int(max_abs.shape[1])):
            tile_records.append(
                {
                    "sample": row,
                    "capture_row": int(row_indices[row]),
                    "expert": int(experts[row]),
                    "output_tile": tile,
                    "reference_l2": float(reference_tiles[row, tile].norm()),
                    "max_abs": float(max_abs[row, tile]),
                    "relative_l2": float(relative_l2[row, tile]),
                }
            )
    max_abs_values = [record["max_abs"] for record in tile_records]
    relative_values = [record["relative_l2"] for record in tile_records]
    report = {
        "kind": "b12x_qsrt_packed_numerics",
        "checkpoint": str(args.checkpoint.resolve()),
        "capture": str(args.capture.resolve()),
        "capture_chunk": str(chunk.resolve()),
        "capture_note": (
            "Inputs were captured from the 3.08-bpw resident model and are used "
            "only to supply real activation magnitudes for operator parity."
        ),
        "layer": args.layer,
        "tp_rank": args.tp_rank,
        "samples": int(args.rows),
        "output_tile_width": 16,
        "output_tiles": len(tile_records),
        "experts": [int(value) for value in experts],
        "capture_rows": [int(value) for value in row_indices],
        "packed_vs_decoded_bf16": {
            "full_output_relative_l2": float(delta.norm() / reference.float().norm()),
            "full_output_max_abs": float(delta.abs().max()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    packed.float().flatten(), reference.float().flatten(), dim=0
                )
            ),
            "per_tile_max_abs": _quantiles(max_abs_values),
            "per_tile_relative_l2": _quantiles(relative_values),
        },
        "payload_provenance": provenance,
        "preparation": preparation,
        "tiles": tile_records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
