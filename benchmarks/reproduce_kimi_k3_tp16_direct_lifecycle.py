#!/usr/bin/env python3
"""Reproduce the Kimi-K3 TP16 native-MXFP4 small-M MoE lifecycle.

The vLLM startup sequence first warms M=1, captures an M=1 CUDA graph, and
then executes an eager M=2 sampler warmup while the graph remains alive.  This
script exercises that sequence on one GPU without loading the full model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_e8m0_native_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument(
        "--post-m",
        type=int,
        default=2,
        help="Eager token count launched after the M=1 graph capture.",
    )
    parser.add_argument(
        "--routes",
        choices=("low", "high", "random"),
        default="low",
    )
    parser.add_argument("--route-seed", type=int, default=1)
    parser.add_argument(
        "--route-ids",
        help="Comma-separated expert IDs; overrides --routes and --route-seed.",
    )
    parser.add_argument(
        "--sort-routes",
        action="store_true",
        help="Sort each generated random top-k row by expert ID.",
    )
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument(
        "--weight-byte",
        type=int,
        default=0,
        help="Packed FP4 byte used to initialize both expert weight tensors.",
    )
    parser.add_argument("--scan-start", type=int, default=0)
    parser.add_argument("--scan-stop", type=int)
    parser.add_argument("--scan-step", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=(
            "eager-m2",
            "capture-m1",
            "capture-m1-then-eager-m2",
            "scan-experts",
        ),
        default="capture-m1-then-eager-m2",
    )
    return parser.parse_args()


def _make_runner(
    *,
    prepared: object,
    m: int,
    device: torch.device,
    experts: int,
    routes: str,
    route_seed: int,
    route_ids: tuple[int, ...] | None,
    sort_routes: bool,
    random_input: bool,
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, torch.Tensor]:
    hidden_size = 3_584
    topk = 16
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
    )
    fc2_chunks = ((192 // 2) + 127) // 128
    required_inter_u32 = m * fc2_chunks * 128 * topk
    intermediate_cache2 = buffers.intermediate_cache2
    if intermediate_cache2.view(torch.uint32).numel() < required_inter_u32:
        intermediate_cache2 = torch.empty(
            required_inter_u32 * 2,
            dtype=torch.bfloat16,
            device=device,
        )
    if random_input:
        generator = torch.Generator(device=device).manual_seed(route_seed)
        inputs = torch.randn(
            (m, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    else:
        inputs = torch.zeros((m, hidden_size), dtype=torch.bfloat16, device=device)
    if route_ids is not None:
        if len(route_ids) != topk:
            raise ValueError(f"--route-ids must contain exactly {topk} IDs")
        topk_ids = (
            torch.tensor(route_ids, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(m, -1)
            .contiguous()
        )
    elif routes == "random":
        generator = torch.Generator(device=device).manual_seed(route_seed)
        topk_ids = torch.stack(
            [
                torch.randperm(experts, device=device, generator=generator)[:topk]
                for _ in range(m)
            ]
        ).to(torch.int32)
        if sort_routes:
            topk_ids = topk_ids.sort(dim=-1).values
    else:
        route_start = 0 if routes == "low" else experts - topk
        topk_ids = (
            torch.arange(
                route_start,
                route_start + topk,
                dtype=torch.int32,
                device=device,
            )
            .unsqueeze(0)
            .expand(m, -1)
            .contiguous()
        )
    topk_weights = torch.full(
        (m, topk),
        1.0 / topk,
        dtype=torch.float32,
        device=device,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            inputs,
            prepared,
            topk_weights,
            topk_ids,
            activation="situ",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
        )

    return run, buffers.output, topk_ids


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    hidden_size = 3_584
    intermediate_size = 192
    rows = 2 * intermediate_size
    e8m0_dtype = torch.float8_e8m0fnu
    route_ids = None
    if args.route_ids:
        route_ids = tuple(int(value) for value in args.route_ids.split(","))

    if not 0 <= args.weight_byte <= 255:
        raise ValueError("--weight-byte must be in [0, 255]")
    w13 = torch.full(
        (args.experts, rows, hidden_size // 2),
        args.weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.full(
        (args.experts, hidden_size, intermediate_size // 2),
        args.weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (args.experts, rows, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(e8m0_dtype)
    w2_scale = torch.full(
        (args.experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(e8m0_dtype)
    scales = torch.ones(args.experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        scales,
        w2,
        w2_scale,
        scales,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )

    if args.mode == "scan-experts":
        scan_stop = args.experts if args.scan_stop is None else args.scan_stop
        for expert_id in range(args.scan_start, scan_stop, args.scan_step):
            run_scan, output_scan, _ = _make_runner(
                prepared=prepared,
                m=args.post_m,
                device=device,
                experts=args.experts,
                routes=args.routes,
                route_seed=args.route_seed,
                route_ids=(expert_id,) * 16,
                sort_routes=args.sort_routes,
                random_input=args.random_input,
            )
            print(f"launching expert {expert_id}", flush=True)
            run_scan()
            torch.cuda.synchronize(device)
            print(
                f"expert {expert_id} passed; "
                f"finite={bool(torch.isfinite(output_scan).all())}",
                flush=True,
            )
        return

    run_m1, output_m1, topk_ids_m1 = _make_runner(
        prepared=prepared,
        m=1,
        device=device,
        experts=args.experts,
        routes=args.routes,
        route_seed=args.route_seed,
        route_ids=route_ids,
        sort_routes=args.sort_routes,
        random_input=args.random_input,
    )
    run_post, output_post, topk_ids_post = _make_runner(
        prepared=prepared,
        m=args.post_m,
        device=device,
        experts=args.experts,
        routes=args.routes,
        route_seed=args.route_seed,
        route_ids=route_ids,
        sort_routes=args.sort_routes,
        random_input=args.random_input,
    )
    print(f"M=1 routes: {topk_ids_m1.cpu().tolist()}")
    if args.post_m != 1:
        print(f"M={args.post_m} routes: {topk_ids_post.cpu().tolist()}")

    if args.mode == "eager-m2":
        run_post()
        torch.cuda.synchronize(device)
        print(
            f"eager M={args.post_m} passed; "
            f"finite={bool(torch.isfinite(output_post).all())}"
        )
        return

    run_m1()
    torch.cuda.synchronize(device)
    eager_m1 = output_m1.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_m1()
    torch.cuda.synchronize(device)
    print(f"M=1 capture passed; finite={bool(torch.isfinite(output_m1).all())}")

    if args.mode == "capture-m1":
        output_m1.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        exact = bool(torch.equal(output_m1, eager_m1))
        max_abs = float((output_m1.float() - eager_m1.float()).abs().max())
        print(f"M=1 replay passed; exact={exact}; max_abs={max_abs}")
        return

    run_post()
    torch.cuda.synchronize(device)
    print(
        f"post-capture eager M={args.post_m} passed; "
        f"finite={bool(torch.isfinite(output_post).all())}"
    )


if __name__ == "__main__":
    main()
