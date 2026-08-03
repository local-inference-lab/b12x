#!/usr/bin/env python3
"""Validate mixed K3/K4 Trellis execution with one real checkpoint layer."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from sparkinfer.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from sparkinfer.moe._shared.kernels.w4a16.mixed_trellis import (
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
K3_EXPERTS = 192
K4_EXPERTS = 64
TOTAL_EXPERTS = K3_EXPERTS + K4_EXPERTS
DEFAULT_TILE_CONFIG = (128, 128, 32, 512)


def _key(layer: int, expert: int, projection: str, rank: int, field: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.{field}"


def _load_tier(
    checkpoint: Path,
    *,
    layer: int,
    rank: int,
    expert_ids: list[int],
    bits: int,
    tile_config: tuple[int, int, int, int],
    device: torch.device,
):
    shard = checkpoint / f"model-layer-{layer:03d}.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:

        def stack(projection: str, field: str) -> torch.Tensor:
            host = torch.stack(
                [
                    handle.get_tensor(_key(layer, expert, projection, rank, field))
                    for expert in expert_ids
                ]
            )
            return host.to(device=device, non_blocking=False).contiguous()

        gate = stack("gate_proj", "trellis")
        up = stack("up_proj", "trellis")
        down = stack("down_proj", "trellis")
        gate_suh = stack("gate_proj", "suh")
        gate_svh = stack("gate_proj", "svh")
        up_suh = stack("up_proj", "suh")
        up_svh = stack("up_proj", "svh")
        down_suh = stack("down_proj", "suh")
        down_svh = stack("down_proj", "svh")

    expected_last = 16 * bits
    expected_projection = (
        len(expert_ids),
        HIDDEN // 16,
        INTERMEDIATE // 16,
        expected_last,
    )
    expected_down = (len(expert_ids), INTERMEDIATE // 16, HIDDEN // 16, expected_last)
    if (
        tuple(gate.shape) != expected_projection
        or tuple(up.shape) != expected_projection
    ):
        raise ValueError(
            f"K{bits} FC1 geometry mismatch: gate={tuple(gate.shape)} up={tuple(up.shape)} "
            f"expected={expected_projection}"
        )
    if tuple(down.shape) != expected_down:
        raise ValueError(
            f"K{bits} FC2 geometry mismatch: down={tuple(down.shape)} "
            f"expected={expected_down}"
        )

    w13 = torch.stack((gate, up), dim=0).contiguous()
    intermediate_rotations = torch.cat((gate_svh, up_svh, down_suh), dim=1).contiguous()
    return prepare_trellis256_moe_weights(
        w13=w13,
        w2=down,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=len(expert_ids),
        activation="silu",
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


def _serial_state(prepared, expert_map: torch.Tensor, m: int, device: torch.device):
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=TOPK,
        dtype=torch.float16,
        device=device,
        route_num_experts=int(expert_map.numel()),
        full_rotation=True,
        block_size_m=8,
    )
    return prepared, expert_map, buffers


def _run_serial(
    state,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    prepared, expert_map, buffers = state
    assert buffers.rotation_a_gate is not None
    assert buffers.rotation_a_up is not None
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
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
        expert_map=expert_map,
        output_expert_map=expert_map,
        route_block_size_m=8,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


def _time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--tile-config",
        type=int,
        nargs=4,
        metavar=("FC1_K", "FC1_N", "FC2_K", "FC2_N"),
        default=DEFAULT_TILE_CONFIG,
    )
    args = parser.parse_args()
    tile_config = tuple(int(value) for value in args.tile_config)
    if tile_config[0] < 128:
        parser.error("--tile-config FC1_K must be at least 128")
    capacity = args.m if args.capacity is None else int(args.capacity)
    if capacity < args.m:
        raise ValueError("capacity must cover m")

    device = torch.device("cuda", torch.cuda.current_device())
    bitmap = json.loads((args.checkpoint / "tier_bitmap.json").read_text())
    bitrates = [int(bits) for bits in bitmap[str(args.layer)]["k"]]
    k3_ids = [expert for expert, bits in enumerate(bitrates) if bits == 3]
    k4_ids = [expert for expert, bits in enumerate(bitrates) if bits == 4]
    if len(k3_ids) != K3_EXPERTS or len(k4_ids) != K4_EXPERTS:
        raise ValueError(
            f"expected {K3_EXPERTS} K3 and {K4_EXPERTS} K4 experts, "
            f"got {len(k3_ids)} and {len(k4_ids)}"
        )

    print(
        f"loading layer={args.layer} rank={args.rank} K3={len(k3_ids)} K4={len(k4_ids)}"
    )
    tier0 = _load_tier(
        args.checkpoint,
        layer=args.layer,
        rank=args.rank,
        expert_ids=k3_ids,
        bits=3,
        tile_config=tile_config,
        device=device,
    )
    tier1 = _load_tier(
        args.checkpoint,
        layer=args.layer,
        rank=args.rank,
        expert_ids=k4_ids,
        bits=4,
        tile_config=tile_config,
        device=device,
    )
    global_to_combined, descriptor = build_tiered_maps(k3_ids, k4_ids, device=device)
    map0 = torch.full((TOTAL_EXPERTS,), -1, dtype=torch.int32, device=device)
    map1 = torch.full((TOTAL_EXPERTS,), -1, dtype=torch.int32, device=device)
    map0[torch.tensor(k3_ids, dtype=torch.long, device=device)] = torch.arange(
        len(k3_ids), dtype=torch.int32, device=device
    )
    map1[torch.tensor(k4_ids, dtype=torch.long, device=device)] = torch.arange(
        len(k4_ids), dtype=torch.int32, device=device
    )

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    rows = []
    for _ in range(args.m):
        ids = k3_ids[:6] + k4_ids[:2]
        perm = torch.randperm(TOPK, generator=generator).tolist()
        rows.append([ids[index] for index in perm])
    topk_ids = torch.tensor(rows, dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn((args.m, TOPK), dtype=torch.float32, device=device), dim=-1
    )
    x = (torch.randn((args.m, HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)

    serial0 = _serial_state(tier0, map0, args.m, device)
    serial1 = _serial_state(tier1, map1, args.m, device)
    serial_output = torch.empty((args.m, HIDDEN), dtype=torch.float32, device=device)

    def serial_fn() -> torch.Tensor:
        return torch.add(
            _run_serial(serial0, x, topk_weights, topk_ids),
            _run_serial(serial1, x, topk_weights, topk_ids),
            out=serial_output,
        )

    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(capacity * TOPK, 8, TOTAL_EXPERTS)
    launch = compile_mixed_trellis(
        size_m=capacity,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        tier0_num_experts=len(k3_ids),
        tier1_num_experts=len(k4_ids),
        top_k=TOPK,
        max_m_blocks=(route_slots + 7) // 8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=tile_config,
    )
    buffers = make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    rotations = combine_trellis_rotations(tier0, tier1)

    def mixed_fn() -> torch.Tensor:
        return run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            rotations,
            launch,
            buffers,
        )

    reference = serial_fn().clone()
    actual = mixed_fn().clone()
    torch.cuda.synchronize()
    relative = (reference - actual).norm() / reference.norm().clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), actual.flatten(), dim=0
    )
    if float(relative) > 4.0e-3 or float(cosine) < 0.99999:
        raise RuntimeError(
            f"checkpoint correctness failed: rel={float(relative):.6e} "
            f"cos={float(cosine):.9f}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mixed_fn()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(captured, actual):
        raise RuntimeError("mixed checkpoint output changed under CUDA graph replay")

    serial_us = _time_us(serial_fn, args.warmup, args.iterations)
    mixed_us = _time_us(mixed_fn, args.warmup, args.iterations)
    allocated = torch.cuda.memory_allocated(device) / 2**30
    print(
        f"PASS m={args.m} capacity={capacity} tile={tile_config} "
        f"rel={float(relative):.6e} "
        f"cos={float(cosine):.9f} "
        f"serial={serial_us:.2f}us mixed={mixed_us:.2f}us "
        f"speedup={serial_us / mixed_us:.3f}x allocated={allocated:.3f}GiB "
        f"regs={launch.registers_per_thread} local={launch.local_memory_bytes}"
    )


if __name__ == "__main__":
    main()
