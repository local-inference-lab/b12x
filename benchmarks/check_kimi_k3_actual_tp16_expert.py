#!/usr/bin/env python3
"""Actual-expert K3 TP16 EXL3 -> SparkInfer closure without a model load.

Only one routed expert's six MXFP4 source tensors are opened.  The script
quantizes it after TP sharding, binds one rank's H128+H64 tiles to the real
SparkInfer preparation path, and compares the cooperative mixed one-grid
result with the serial full-rotation Trellis oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

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


def _synthetic_mxfp4(device: torch.device):
    hidden, intermediate = 3584, 192
    generator = torch.Generator(device=device).manual_seed(81)
    return prepare_w4a16_e8m0_native_weights(
        torch.randint(
            0,
            256,
            (1, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        _e8m0((1, 2 * intermediate, hidden // 32), device),
        torch.ones(1, dtype=torch.float32, device=device),
        torch.randint(
            0,
            256,
            (1, hidden, intermediate // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        _e8m0((1, hidden, intermediate // 32), device),
        torch.ones(1, dtype=torch.float32, device=device),
        activation="situ",
        params_dtype=torch.float16,
        w13_layout="w31",
    )


def _actual_trellis(
    tensors: dict[str, torch.Tensor],
    *,
    layer: int,
    expert: int,
    rank: int,
    device: torch.device,
):
    hidden, global_intermediate, local_intermediate = 3584, 3072, 192
    start = rank * local_intermediate
    stop = start + local_intermediate
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"

    def get(matrix: str, part: str) -> torch.Tensor:
        return tensors[f"{prefix}.{matrix}.exl3_{part}"].to(device)

    w13 = (
        torch.stack(
            (
                get("w1", "trellis")[:, start // 16 : stop // 16],
                get("w3", "trellis")[:, start // 16 : stop // 16],
            ),
            dim=0,
        )
        .unsqueeze(1)
        .contiguous()
    )
    w2 = get("w2", "trellis")[start // 16 : stop // 16].unsqueeze(0).contiguous()
    intermediate_rotations = (
        torch.cat(
            (
                get("w1", "svh")[start:stop],
                get("w3", "svh")[start:stop],
                get("w2", "suh")[start:stop],
            )
        )
        .view(1, 3 * local_intermediate)
        .contiguous()
    )
    assert stop <= global_intermediate
    return prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=local_intermediate,
        num_experts=1,
        activation="situ",
        fc1_tile_n=64,
        fc2_tile_n=128,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=3,
        gate_suh=get("w1", "suh").view(1, hidden).contiguous(),
        up_suh=get("w3", "suh").view(1, hidden).contiguous(),
        intermediate_rotations=intermediate_rotations,
        down_svh=get("w2", "svh").view(1, hidden).contiguous(),
        tile_config=(128, 64, 64, 128),
        tp_local_intermediate_hadamard_tail=64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=7)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--temp-batch-size", type=int, default=16)
    parser.add_argument(
        "--quant-cache",
        type=Path,
        help="reuse/save the tiny one-expert quant output across kernel iterations",
    )
    args = parser.parse_args()
    if not 0 <= args.rank < 16:
        raise ValueError("--rank must be in [0, 15]")
    os.environ["KQUANT_EXL3_SHARED_SU"] = "1"

    from kquant.io.hf_cache import resolve
    from kquant.pack.exl3 import make_shared_h, quantize_layer

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    cache_contract = {
        "layer": args.layer,
        "expert": args.expert,
        "tp_size": 16,
        "hadamard_blocks": [128, 64],
        "bits": 3,
        "codebook": "mcg",
        "shared_su": True,
    }
    cache_meta = (
        args.quant_cache.with_suffix(args.quant_cache.suffix + ".json")
        if args.quant_cache
        else None
    )
    cache_hit = bool(
        args.quant_cache
        and args.quant_cache.is_file()
        and cache_meta
        and cache_meta.is_file()
        and json.loads(cache_meta.read_text()) == cache_contract
    )
    if cache_hit:
        from safetensors.torch import load_file

        tensors = load_file(str(args.quant_cache))
        errors = {}
        quant_seconds = 0.0
    else:
        if args.quant_cache and (args.quant_cache.exists() or cache_meta.exists()):
            raise RuntimeError("--quant-cache exists but its contract does not match")
        quant_started = time.perf_counter()
        tensors, errors = quantize_layer(
            resolve(),
            args.layer,
            [args.expert],
            device,
            {k: make_shared_h(k, device) for k in (3584, 192)},
            batch=1,
            tp_size=16,
            temp_batch_size=args.temp_batch_size,
        )
        quant_seconds = time.perf_counter() - quant_started
        if args.quant_cache:
            from safetensors.torch import save_file

            save_file(tensors, str(args.quant_cache))
            assert cache_meta is not None
            cache_meta.write_text(json.dumps(cache_contract, sort_keys=True))
    tier0 = _synthetic_mxfp4(device)
    tier1 = _actual_trellis(
        tensors,
        layer=args.layer,
        expert=args.expert,
        rank=args.rank,
        device=device,
    )

    hidden, intermediate, topk = 3584, 192, 16
    tiles = (128, 64, 64, 128)
    x = (torch.randn((1, hidden), device=device) * 1.0e-3).bfloat16()
    topk_ids = torch.ones((1, topk), dtype=torch.int32, device=device)
    topk_weights = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=device)
    serial_buffers = make_w4a16_packed_buffers(
        tier1,
        m=1,
        topk=topk,
        dtype=torch.float16,
        device=device,
        route_num_experts=2,
        full_rotation=True,
        block_size_m=8,
    )
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared = int(props.shared_memory_per_block_optin)
    serial_launch = compile_w4a16_fused_moe(
        size_m=1,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=1,
        top_k=topk,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=16,
        element_dtype="fp16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        force_tile_config=tiles,
        intermediate_rotation=True,
        full_rotation=True,
        rotation_input_dtype="bf16",
        broadcast_suh=True,
    )
    serial = run_w4a16_moe(
        x,
        tier1,
        topk_weights,
        topk_ids,
        activation="situ",
        intermediate_cache13=serial_buffers.intermediate_cache13,
        intermediate_cache2=serial_buffers.intermediate_cache2,
        output=serial_buffers.output,
        fc1_c_tmp=serial_buffers.fc1_c_tmp,
        fc2_c_tmp=serial_buffers.fc2_c_tmp,
        packed_route_indices=serial_buffers.packed_route_indices,
        block_expert_ids=serial_buffers.block_expert_ids,
        packed_route_count=serial_buffers.packed_route_count,
        expert_offsets=serial_buffers.expert_offsets,
        expert_counts=serial_buffers.expert_counts,
        expert_map=torch.tensor([-1, 0], dtype=torch.int32, device=device),
        output_expert_map=torch.tensor([-1, 0], dtype=torch.int32, device=device),
        route_block_size_m=8,
        intermediate_rotation_scales=tier1.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=tier1.gate_suh,
        suh_up_table=tier1.up_suh,
        svh_table=tier1.down_svh,
        rotation_a_gate=serial_buffers.rotation_a_gate,
        rotation_a_up=serial_buffers.rotation_a_up,
        fused_launch=serial_launch,
    ).clone()

    compile_started = time.perf_counter()
    mixed_launch = compile_mixed_mxfp4_trellis(
        size_m=1,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=1,
        tier1_num_experts=1,
        top_k=topk,
        max_m_blocks=(max_packed_route_slots(topk, 8, 2) + 7) // 8,
        sms=sms,
        max_shared_mem=max_shared,
        force_tile_config=tiles,
        activation="situ",
        intermediate_hadamard_tail=64,
        tier1_broadcast_suh=True,
        tier1_broadcast_svh=True,
    )
    compile_seconds = time.perf_counter() - compile_started
    global_to_combined, descriptors = build_tiered_maps((0,), (1,), device=device)
    buffers = make_mixed_trellis_buffers(mixed_launch, device=device, sms=sms)
    mixed = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptors,
        combine_mxfp4_trellis_rotations(1, tier1),
        mixed_launch,
        buffers,
    ).clone()
    torch.cuda.synchronize(device)
    relative = float(
        (mixed.float() - serial.float()).norm()
        / serial.float().norm().clamp_min(1.0e-12)
    )
    serial_finite = bool(torch.isfinite(serial).all())
    mixed_finite = bool(torch.isfinite(mixed).all())
    if not serial_finite or not mixed_finite or not math.isfinite(relative):
        raise AssertionError(
            "actual expert kernel closure produced non-finite values: "
            f"serial_finite={serial_finite}, mixed_finite={mixed_finite}, "
            f"serial_nan={int(torch.isnan(serial).sum())}, "
            f"mixed_nan={int(torch.isnan(mixed).sum())}, relative={relative}"
        )
    if relative >= 4.0e-3:
        raise AssertionError(
            f"actual expert kernel closure failed: relative={relative}"
        )

    eager = mixed.clone()
    captured = torch.empty_like(mixed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.copy_(
            run_mixed_trellis(
                x,
                tier0,
                tier1,
                topk_weights,
                topk_ids,
                global_to_combined,
                descriptors,
                combine_mxfp4_trellis_rotations(1, tier1),
                mixed_launch,
                buffers,
            )
        )
    graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(captured, eager):
        raise AssertionError("CUDA graph replay drifted from eager one-grid output")
    print(
        {
            "pass": True,
            "checkpoint_tensors_opened": 0 if cache_hit else 6,
            "quant_cache_hit": cache_hit,
            "layer": args.layer,
            "expert": args.expert,
            "tp_rank_slice": args.rank,
            "quant_seconds": round(quant_seconds, 3),
            "compile_seconds": round(compile_seconds, 3),
            "serial_onegrid_relative_l2": relative,
            "cuda_graph_exact": True,
            "proxy_errors": errors,
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / 2**20, 1
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
