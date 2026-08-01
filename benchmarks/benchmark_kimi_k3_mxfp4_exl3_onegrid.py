#!/usr/bin/env python3
"""Synthetic Kimi-K3 TP16 MXFP4+EXL3 one-grid correctness/latency harness.

This deliberately never opens a model checkpoint.  It keeps production decode
geometry (H=3584, I=192, top-k=16 and the TP16 tile plan) while allocating only
two physical experts per tier.  The correctness oracle is the current serial
runtime: one original-MXFP4 launch plus one full-rotation EXL3 launch.
"""

from __future__ import annotations

import argparse
import time

import torch

from sparkinfer.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe,
    run_w4a16_moe,
)
from sparkinfer.moe._shared.kernels.w4a16.mixed_trellis import (
    build_tiered_maps,
    combine_mxfp4_trellis_rotations,
    compile_mixed_mxfp4_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
    prepare_w4a16_e8m0_native_weights,
)


def _e8m0(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("torch.float8_e8m0fnu is required")
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(dtype)


def _build_problem(device: torch.device, *, broadcast_scales: bool):
    experts = 2
    hidden = 3584
    intermediate = 192
    topk = 16
    capacity_m = 1
    tiles = (128, 64, 64, 128)
    torch.manual_seed(20260801)

    tier0 = prepare_w4a16_e8m0_native_weights(
        torch.randint(
            0,
            256,
            (experts, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
            device=device,
        ),
        _e8m0((experts, 2 * intermediate, hidden // 32), device),
        torch.ones(experts, dtype=torch.float32, device=device),
        torch.randint(
            0,
            256,
            (experts, hidden, intermediate // 2),
            dtype=torch.uint8,
            device=device,
        ),
        _e8m0((experts, hidden, intermediate // 32), device),
        torch.ones(experts, dtype=torch.float32, device=device),
        activation="situ",
        params_dtype=torch.float16,
        w13_layout="w31",
    )

    generator = torch.Generator(device=device).manual_seed(33)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).half()

    tier1 = prepare_trellis256_moe_weights(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        fc1_tile_n=tiles[1],
        fc2_tile_n=tiles[3],
        device=device,
        seed=44,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=3,
        gate_suh=scales((1 if broadcast_scales else experts, hidden)),
        up_suh=scales((1 if broadcast_scales else experts, hidden)),
        intermediate_rotations=scales((experts, 3 * intermediate)),
        down_svh=scales((1 if broadcast_scales else experts, hidden)),
        tile_config=tiles,
        tp_local_intermediate_hadamard_tail=64,
    )
    x = (torch.randn((capacity_m, hidden), device=device) * 1.0e-3).bfloat16()
    weights = torch.softmax(
        torch.randn((capacity_m, topk), device=device), dim=-1
    ).float()
    return tier0, tier1, x, weights, hidden, intermediate, topk, capacity_m, tiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--broadcast-scales", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    (
        tier0,
        tier1,
        x,
        topk_weights,
        hidden,
        intermediate,
        topk,
        capacity_m,
        tiles,
    ) = _build_problem(device, broadcast_scales=args.broadcast_scales)
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    logical_tier0 = int(tier0.num_experts)
    logical_tier1 = int(tier1.num_experts)
    logical_experts = logical_tier0 + logical_tier1

    serial0_buffers = make_w4a16_packed_buffers(
        tier0,
        m=capacity_m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        route_num_experts=logical_experts,
        full_rotation=False,
        block_size_m=8,
    )
    serial1_buffers = make_w4a16_packed_buffers(
        tier1,
        m=capacity_m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        route_num_experts=logical_experts,
        full_rotation=True,
        block_size_m=8,
    )
    serial1_launch = compile_w4a16_fused_moe(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=2,
        top_k=topk,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=16,
        element_dtype="fp16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        force_tile_config=tiles,
        intermediate_rotation=True,
        full_rotation=True,
        rotation_input_dtype="bf16",
        broadcast_suh=args.broadcast_scales,
    )
    compile_started = time.perf_counter()
    mixed_launch = compile_mixed_mxfp4_trellis(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=logical_tier0,
        tier1_num_experts=logical_tier1,
        top_k=topk,
        max_m_blocks=(max_packed_route_slots(capacity_m * topk, 8, logical_experts) + 7)
        // 8,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tiles,
        activation="situ",
        intermediate_hadamard_tail=64,
        tier1_broadcast_suh=args.broadcast_scales,
        tier1_broadcast_svh=args.broadcast_scales,
    )
    compile_seconds = time.perf_counter() - compile_started
    global_to_combined, descriptor = build_tiered_maps(
        range(logical_tier0),
        range(logical_tier0, logical_experts),
        device=device,
    )
    rotations = combine_mxfp4_trellis_rotations(logical_tier0, tier1)
    mixed_buffers = make_mixed_trellis_buffers(mixed_launch, device=device, sms=sms)
    if mixed_buffers.fc2.data_ptr() != mixed_buffers.rotation_gate.data_ptr():
        raise RuntimeError("production FC2/rotation buffer alias was lost")

    map0 = torch.full((logical_experts,), -1, dtype=torch.int32, device=device)
    map0[0] = 0
    map0[1] = 1
    map1 = torch.full((logical_experts,), -1, dtype=torch.int32, device=device)
    map1[logical_tier0] = 0
    map1[logical_tier0 + 1] = 1

    def serial(topk_ids: torch.Tensor) -> torch.Tensor:
        out0 = run_w4a16_moe(
            x.half(),
            tier0,
            topk_weights,
            topk_ids,
            activation="situ",
            intermediate_cache13=serial0_buffers.intermediate_cache13,
            intermediate_cache2=serial0_buffers.intermediate_cache2,
            output=serial0_buffers.output,
            fc1_c_tmp=serial0_buffers.fc1_c_tmp,
            fc2_c_tmp=serial0_buffers.fc2_c_tmp,
            packed_route_indices=serial0_buffers.packed_route_indices,
            block_expert_ids=serial0_buffers.block_expert_ids,
            packed_route_count=serial0_buffers.packed_route_count,
            expert_offsets=serial0_buffers.expert_offsets,
            expert_counts=serial0_buffers.expert_counts,
            expert_map=map0,
            route_block_size_m=8,
        ).float()
        out0 = out0.clone()
        for value in (
            serial1_buffers.intermediate_cache13,
            serial1_buffers.intermediate_cache2,
            serial1_buffers.output,
            serial1_buffers.rotation_a_gate,
            serial1_buffers.rotation_a_up,
        ):
            if value is not None:
                value.zero_()
        out1 = run_w4a16_moe(
            x,
            tier1,
            topk_weights,
            topk_ids,
            activation="situ",
            intermediate_cache13=serial1_buffers.intermediate_cache13,
            intermediate_cache2=serial1_buffers.intermediate_cache2,
            output=serial1_buffers.output,
            fc1_c_tmp=serial1_buffers.fc1_c_tmp,
            fc2_c_tmp=serial1_buffers.fc2_c_tmp,
            packed_route_indices=serial1_buffers.packed_route_indices,
            block_expert_ids=serial1_buffers.block_expert_ids,
            packed_route_count=serial1_buffers.packed_route_count,
            expert_offsets=serial1_buffers.expert_offsets,
            expert_counts=serial1_buffers.expert_counts,
            expert_map=map1,
            output_expert_map=map1,
            route_block_size_m=8,
            intermediate_rotation_scales=tier1.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=tier1.gate_suh,
            suh_up_table=tier1.up_suh,
            svh_table=tier1.down_svh,
            rotation_a_gate=serial1_buffers.rotation_a_gate,
            rotation_a_up=serial1_buffers.rotation_a_up,
            fused_launch=serial1_launch,
        ).float()
        if not torch.isfinite(out0).all() or not torch.isfinite(out1).all():
            tier1_census = {}
            for field in (
                "intermediate_cache13",
                "intermediate_cache2",
                "output",
                "rotation_a_gate",
                "rotation_a_up",
            ):
                value = getattr(serial1_buffers, field)
                if value is not None:
                    tier1_census[field] = {
                        "shape": tuple(value.shape),
                        "finite": bool(torch.isfinite(value).all()),
                        "nan": int(torch.isnan(value).sum()),
                    }
            print(
                {
                    "serial_tier0_finite": bool(torch.isfinite(out0).all()),
                    "serial_tier1_finite": bool(torch.isfinite(out1).all()),
                    "serial_tier0_norm": float(out0.norm()),
                    "serial_tier1_norm": float(out1.norm()),
                    "tier1_census": tier1_census,
                }
            )
        return out0 + out1

    def mixed(topk_ids: torch.Tensor) -> torch.Tensor:
        return run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            rotations,
            mixed_launch,
            mixed_buffers,
        )

    def serial1_local(local_ids: torch.Tensor) -> torch.Tensor:
        for value in (
            serial1_buffers.intermediate_cache13,
            serial1_buffers.intermediate_cache2,
            serial1_buffers.output,
            serial1_buffers.rotation_a_gate,
            serial1_buffers.rotation_a_up,
        ):
            if value is not None:
                value.zero_()
        return run_w4a16_moe(
            x,
            tier1,
            topk_weights,
            local_ids,
            activation="situ",
            intermediate_cache13=serial1_buffers.intermediate_cache13,
            intermediate_cache2=serial1_buffers.intermediate_cache2,
            output=serial1_buffers.output,
            fc1_c_tmp=serial1_buffers.fc1_c_tmp,
            fc2_c_tmp=serial1_buffers.fc2_c_tmp,
            packed_route_indices=serial1_buffers.packed_route_indices,
            block_expert_ids=serial1_buffers.block_expert_ids,
            packed_route_count=serial1_buffers.packed_route_count,
            expert_offsets=serial1_buffers.expert_offsets,
            expert_counts=serial1_buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=tier1.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=tier1.gate_suh,
            suh_up_table=tier1.up_suh,
            svh_table=tier1.down_svh,
            rotation_a_gate=serial1_buffers.rotation_a_gate,
            rotation_a_up=serial1_buffers.rotation_a_up,
            fused_launch=serial1_launch,
        ).float()

    cases = {
        "mixed": torch.tensor(
            [[0, logical_tier0, 1, logical_tier0 + 1] * 4],
            dtype=torch.int32,
            device=device,
        ),
        "tier0": torch.tensor([[0, 1] * 8], dtype=torch.int32, device=device),
        "tier1": torch.tensor(
            [[logical_tier0, logical_tier0 + 1] * 8],
            dtype=torch.int32,
            device=device,
        ),
    }
    mapped_tier1 = serial(cases["tier1"])
    local_tier1 = serial1_local(
        torch.tensor([[0, 1] * 8], dtype=torch.int32, device=device)
    )
    torch.cuda.synchronize(device)
    print(
        {
            "mapped_vs_local_tier1_relative_l2": float(
                (mapped_tier1 - local_tier1).norm()
                / local_tier1.norm().clamp_min(1.0e-12)
            ),
            "mapped_tier1_norm": float(mapped_tier1.norm()),
            "local_tier1_norm": float(local_tier1.norm()),
        }
    )
    eager_outputs: dict[str, torch.Tensor] = {}
    for name, ids in cases.items():
        expected = serial(ids)
        actual = mixed(ids)
        torch.cuda.synchronize(device)
        eager_outputs[name] = actual.clone()
        relative = float(
            (actual.float() - expected).norm() / expected.norm().clamp_min(1.0e-12)
        )
        print(
            {
                "case": name,
                "relative_l2": relative,
                "reference_norm": float(expected.norm()),
                "actual_norm": float(actual.float().norm()),
                "finite": bool(torch.isfinite(actual).all()),
            }
        )
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise RuntimeError(f"{name} produced non-finite output")
        if relative >= 3.0e-2:
            raise RuntimeError(f"{name} correctness gate failed: {relative}")

    graph_ids = cases["mixed"].clone()
    graph_expected = eager_outputs["mixed"]
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mixed(graph_ids)
    graph_results = []
    for name in ("mixed", "tier0", "tier1", "mixed", "tier1", "tier0"):
        graph_ids.copy_(cases[name])
        graph.replay()
        torch.cuda.synchronize(device)
        expected = eager_outputs[name]
        graph_relative = float(
            (captured.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1.0e-12)
        )
        graph_results.append((name, graph_relative))
        if graph_relative >= 1.0e-5:
            raise RuntimeError(
                "CUDA graph replay changed the mixed output for "
                f"{name}: {graph_relative}"
            )
    print(
        {
            "graph_route_sequence_relative_l2": graph_results,
            "graph_expected_norm": float(graph_expected.float().norm()),
        }
    )

    def milliseconds(fn) -> float:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / args.iterations

    ids = cases["mixed"]
    serial_ms = milliseconds(lambda: serial(ids))
    mixed_ms = milliseconds(lambda: mixed(ids))
    graph_ids.copy_(ids)
    graph_ms = milliseconds(graph.replay)
    print(
        {
            "compile_seconds": compile_seconds,
            "shared_memory_bytes": mixed_launch.shared_memory_bytes,
            "registers_per_thread": mixed_launch.registers_per_thread,
            "local_memory_bytes": mixed_launch.local_memory_bytes,
            "serial_ms": serial_ms,
            "mixed_eager_ms": mixed_ms,
            "mixed_graph_ms": graph_ms,
            "serial_over_mixed_graph": serial_ms / graph_ms,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
