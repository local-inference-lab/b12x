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
import json
import statistics
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


EXPERTS = 256
HIDDEN_SIZE = 4096
INTERMEDIATE_FULL = 2048
TOPK = 6
RANK = 4


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
    # half for TP, then restore the adapter contract's logical [gate, up].
    w13_b = torch.cat(
        (
            w13_b_full[:, start:stop, :],
            w13_b_full[
                :, INTERMEDIATE_FULL + start : INTERMEDIATE_FULL + stop, :
            ],
        ),
        dim=1,
    )
    w2_a = w2_a_full[:, :, start:stop]
    tensors = [w13_a, w13_b, w2_a, w2_b]
    tensors = [
        tensor.to(device="cuda", dtype=torch.bfloat16).contiguous()
        for tensor in tensors
    ]
    return W4A16StaticExpertLoRA(
        w13_a=tensors[0],
        w13_b=tensors[1],
        w2_a=tensors[2],
        w2_b=tensors[3],
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )


def _synthetic_adapter(
    *, intermediate_size: int, token_mapping: torch.Tensor
) -> W4A16StaticExpertLoRA:
    def make(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16)

    return W4A16StaticExpertLoRA(
        w13_a=make((EXPERTS, RANK, HIDDEN_SIZE), 0.025),
        w13_b=make((EXPERTS, 2 * intermediate_size, RANK), 0.04),
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


def _time_graph(
    graph: torch.cuda.CUDAGraph, *, iterations: int, repeats: int
) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) * 1000.0 / iterations)
    return samples


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

    results: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
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
            )
            if args.adapter is not None
            else _synthetic_adapter(
                intermediate_size=intermediate_size,
                token_mapping=token_mapping,
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

        def run(buffers, *, static_lora=None):
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

        base_graph = _capture(lambda: run(base_buffers))
        base_output = base_buffers.output.clone()
        lora_graph = _capture(lambda: run(lora_buffers, static_lora=adapter))
        lora_output = lora_buffers.output.clone()
        torch.cuda.synchronize()

        base_us = _time_graph(
            base_graph, iterations=args.iterations, repeats=args.repeats
        )
        lora_us = _time_graph(
            lora_graph, iterations=args.iterations, repeats=args.repeats
        )
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
        measurement = {
            "tokens": m,
            "lora_path": "direct_augmented" if m == 1 else "staged",
            "base_us": base_us,
            "lora_us": lora_us,
            "base_median_us": base_median,
            "lora_median_us": lora_median,
            "slowdown_x": lora_median / base_median,
            "overhead_percent": (lora_median / base_median - 1.0) * 100.0,
            "relative_output_delta": float((delta / base_norm).item()),
            "base_finite": bool(torch.isfinite(base_output).all().item()),
            "lora_finite": bool(torch.isfinite(lora_output).all().item()),
            "small_m_direct_baseline": _small_m_direct_supported(
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
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
