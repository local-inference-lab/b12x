"""Benchmark the Kimi K3 TP12 EXL3 full-rotation decode path.

The default shape is the production per-rank geometry:

* M=1, H=3584, I=3072/TP12=256
* top-k=16 over 896 global experts
* native 3-bit Trellis weights with shared H-side rotations

Only the routed experts are read, and expert count is runtime data for this
kernel.  The default therefore materializes 32 experts to keep the standalone
benchmark small while preserving the exact generated kernel and launch
geometry used by layers containing hundreds of EXL3 experts.
"""

from __future__ import annotations

import argparse
import gc
import statistics

import torch

from sparkinfer.moe import fused_moe


_PRODUCTION_TILES = (64, 256, 64, 256)
_TILE_CANDIDATES = (
    _PRODUCTION_TILES,
    (128, 128, 64, 256),
    (64, 256, 128, 128),
    (128, 128, 128, 128),
    (64, 128, 64, 128),
    (128, 64, 128, 64),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compare-tiles",
        action="store_true",
        help="benchmark every legal 128/256-thread tile candidate",
    )
    return parser.parse_args()


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0 / iterations)
    return samples


def main() -> None:
    args = _parse_args()
    if args.experts < 16:
        raise ValueError("--experts must be at least top-k=16")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        raise ValueError("iterations, rounds, and warmup must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        raise RuntimeError(f"benchmark requires SM12x, got SM{major}{minor}")

    experts = args.experts
    hidden_size = 3_584
    intermediate_size = 256
    topk = 16
    route_experts = 896
    trellis_words = 16 * 3

    w13 = torch.randint(
        -32768,
        32767,
        (
            2,
            experts,
            hidden_size // 16,
            intermediate_size // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (
            experts,
            intermediate_size // 16,
            hidden_size // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
    )
    shared_h_rotation = torch.ones((1, hidden_size), dtype=torch.float16, device=device)
    intermediate_rotations = torch.ones(
        (experts, 3 * intermediate_size), dtype=torch.float16, device=device
    )
    x = (torch.randn((1, hidden_size), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk)
    topk_weights = torch.rand((1, topk), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    expert_map = torch.full((route_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:experts] = torch.arange(experts, dtype=torch.int32, device=device)

    tile_configs = _TILE_CANDIDATES if args.compare_tiles else (_PRODUCTION_TILES,)
    reference: torch.Tensor | None = None
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"shape=M1/H{hidden_size}/I{intermediate_size}/E{route_experts}/K{topk}"
    )
    print("tiles                         median_us    min_us    rel_error    cosine")

    for tile_config in tile_configs:
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="exl3_trellis_mcg",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout="w13",
            trellis_bits=3,
            trellis_tile_config=tile_config,
        )
        weights = fused_moe.prepare_weights(
            plan=weight_plan,
            params_dtype=torch.bfloat16,
            w1_fp4=w13,
            w2_fp4=w2,
            gate_suh=shared_h_rotation,
            up_suh=shared_h_rotation,
            intermediate_rotations=intermediate_rotations,
            down_svh=shared_h_rotation,
            trellis_mcg=0xCBAC1FED,
        )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=1,
                num_topk=topk,
                route_num_experts=route_experts,
                device=device,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=8,
            )
        )
        scratch_spec = plan.scratch_specs()[0]
        scratch = torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=scratch_spec.device,
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=expert_map,
            output_expert_map=expert_map,
        )
        eager = binding.run().clone()
        torch.cuda.synchronize(device)
        if not torch.isfinite(eager).all():
            raise RuntimeError(f"non-finite output for tile config {tile_config}")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = binding.run()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(captured, eager):
            raise RuntimeError(
                f"CUDA-graph output differs from eager for tile config {tile_config}"
            )

        if reference is None:
            reference = eager.clone()
        relative_error = float(
            (eager - reference).norm() / reference.norm().clamp_min(1.0e-9)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                eager.flatten(), reference.flatten(), dim=0
            )
        )
        samples = _time_graph(
            graph,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        print(
            f"{str(tile_config):<28} "
            f"{statistics.median(samples):9.3f} "
            f"{min(samples):9.3f} "
            f"{relative_error:12.3e} "
            f"{cosine:9.6f}"
        )

        del captured, graph, binding, scratch, plan, weights, weight_plan, eager
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
