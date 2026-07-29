"""Fast correctness and latency harness for Kimi K3 TP16 MXFP4 decode.

This reproduces the per-layer shape that matters for single-request decode
without loading the model checkpoint.  It compares SparkInfer's native M=1
microkernel with the numerically-safe M=9 packed-route fallback and an FP32
oracle, then times both GPU paths.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from sparkinfer.moe._shared.kernels.reference import (
    moe_reference_w4a16_fp4_e8m0_k32,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_e8m0_native_weights,
)
from tests._reference.w4a16_reference import compare_to_reference


K3_HIDDEN_SIZE = 3584
K3_TP16_INTERMEDIATE_SIZE = 192
K3_TOPK = 16
PACKED_ROUTE_M = 9


def _pattern_e8m0(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("K3 MXFP4 harness requires torch.float8_e8m0fnu")
    values = torch.arange(torch.tensor(shape).prod().item(), device="cuda")
    values = ((values + offset) % 4 + 119).to(torch.uint8).reshape(shape)
    return values.view(dtype)


def _micro_scratch(m: int, intermediate_size: int, topk: int) -> torch.Tensor:
    # Matches MoEMicroKernelW4A16SmallMDirect's FC1-output scratch contract.
    fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
    return torch.zeros(
        2 * m * fc2_n_chunks * 128 * topk,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _launch(
    x: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    buffers,
    intermediate_cache2: torch.Tensor,
) -> torch.Tensor:
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="situ",
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )


def _bench_ms(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=717)
    parser.add_argument("--hidden-size", type=int, default=K3_HIDDEN_SIZE)
    parser.add_argument(
        "--intermediate-size", type=int, default=K3_TP16_INTERMEDIATE_SIZE
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    experts = args.experts
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    rows = 2 * intermediate_size
    topk = K3_TOPK

    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0(
        (experts, hidden_size, (intermediate_size + 31) // 32), offset=1
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )

    x1 = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")
    ids1 = (torch.arange(topk, device="cuda") % experts).to(torch.int32)[None, :]
    weights1 = torch.rand(1, topk, dtype=torch.float32, device="cuda")
    weights1 /= weights1.sum(dim=1, keepdim=True)

    x9 = torch.zeros(PACKED_ROUTE_M, hidden_size, dtype=torch.bfloat16, device="cuda")
    x9[:1].copy_(x1)
    ids9 = torch.zeros(PACKED_ROUTE_M, topk, dtype=torch.int32, device="cuda")
    ids9[:1].copy_(ids1)
    weights9 = torch.zeros(PACKED_ROUTE_M, topk, dtype=torch.float32, device="cuda")
    weights9[:1].copy_(weights1)

    buffers1 = make_w4a16_packed_buffers(
        prepared,
        m=1,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    buffers9 = make_w4a16_packed_buffers(
        prepared,
        m=PACKED_ROUTE_M,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    micro_cache2 = _micro_scratch(1, intermediate_size, topk)

    def direct_launch() -> torch.Tensor:
        return _launch(x1, prepared, weights1, ids1, buffers1, micro_cache2)

    def packed_launch() -> torch.Tensor:
        return _launch(
            x9,
            prepared,
            weights9,
            ids9,
            buffers9,
            buffers9.intermediate_cache2,
        )

    compile_start = time.perf_counter()
    direct = direct_launch().clone()
    micro_intermediate = micro_cache2.clone()
    packed = packed_launch()[:1].clone()
    torch.cuda.synchronize()
    compile_and_first_launch_s = time.perf_counter() - compile_start

    results: dict[str, object] = {
        "shape": {
            "experts": experts,
            "hidden_size": hidden_size,
            "tp16_intermediate_size": intermediate_size,
            "topk": topk,
        },
        "compile_and_first_launch_s": compile_and_first_launch_s,
        "direct_vs_packed": asdict(compare_to_reference(direct, packed)),
        "finite": {
            "direct": bool(torch.isfinite(direct).all().item()),
            "packed": bool(torch.isfinite(packed).all().item()),
        },
        "diagnostics": {
            "direct_nonzero": int(torch.count_nonzero(direct).item()),
            "direct_abs_max": float(direct.abs().max().item()),
            "packed_nonzero": int(torch.count_nonzero(packed).item()),
            "packed_abs_max": float(packed.abs().max().item()),
            "micro_intermediate_nonzero_u16": int(
                torch.count_nonzero(micro_intermediate.view(torch.uint16)).item()
            ),
        },
    }
    if not args.skip_oracle:
        oracle = moe_reference_w4a16_fp4_e8m0_k32(
            x1,
            w13,
            w13_scale,
            w13_global_scale,
            w2,
            w2_scale,
            w2_global_scale,
            ids1,
            weights1,
            experts,
            hidden_size,
            intermediate_size,
            activation="situ",
            w13_layout="w31",
        )
        results["direct_vs_oracle"] = asdict(compare_to_reference(direct, oracle))
        results["packed_vs_oracle"] = asdict(compare_to_reference(packed, oracle))

    direct_ms = _bench_ms(direct_launch, warmup=args.warmup, iterations=args.iterations)
    packed_ms = _bench_ms(packed_launch, warmup=args.warmup, iterations=args.iterations)
    results["latency_ms"] = {
        "direct_m1": direct_ms,
        "packed_m9": packed_ms,
        "packed_over_direct": packed_ms / direct_ms,
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
