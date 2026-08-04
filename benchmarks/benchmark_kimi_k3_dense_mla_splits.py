#!/usr/bin/env python3
"""Benchmark capture-static Kimi-K3 dense-MLA split plans on one GPU.

This isolates the exact TP16/DCP8 per-rank production shape without loading
the model.  Global context length is eight times ``cache_seqlens`` because a
DCP8 rank owns one eighth of the KV cache.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from sparkinfer.attention import dense_mla


FP8 = torch.float8_e4m3fn
HEADS = 48
HEAD_DIM = 576
VALUE_DIM = 512


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--page-size", type=int, default=768)
    parser.add_argument("--max-cache-tokens", type=int, default=131_072)
    parser.add_argument(
        "--splits",
        type=_csv_ints,
        default=_csv_ints("1,8,16,32,64,94"),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=_csv_ints,
        default=_csv_ints("18,256,2048,8192,32768,65536,131072"),
        help="Per-rank DCP cache lengths, not global context lengths.",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if max(args.sequence_lengths) > args.max_cache_tokens:
        raise ValueError("sequence length exceeds --max-cache-tokens")

    page_width = (args.max_cache_tokens + args.page_size - 1) // args.page_size
    cache = torch.full(
        (page_width, args.page_size, HEAD_DIM),
        0.25,
        dtype=FP8,
        device=device,
    )
    q = torch.full((1, HEADS, HEAD_DIM), 0.125, dtype=FP8, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    kv_scale = torch.ones(1, dtype=torch.float32, device=device)
    page_table = torch.arange(page_width, dtype=torch.int32, device=device).view(1, -1)
    cache_seqlens = torch.ones(1, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    reference_outputs: dict[int, torch.Tensor] = {}
    all_results: list[dict[str, float | int]] = []

    for requested_splits in args.splits:
        plan = dense_mla.plan(
            dense_mla.Caps(
                device=device,
                mode="decode",
                kv_dtype=FP8,
                num_q_heads=HEADS,
                page_size=args.page_size,
                max_total_q=1,
                max_batch=1,
                max_cache_tokens=args.max_cache_tokens,
                max_page_table_width=page_width,
                num_cache_pages=page_width,
                use_cuda_graph=True,
                budget=dense_mla.Budget(max_splits=requested_splits),
            )
        )
        (scratch_spec,) = plan.scratch_specs()
        scratch = torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=scratch_spec.device,
        )
        output = torch.empty(
            (1, HEADS, VALUE_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        binding = dense_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        dense_mla.compile(binding=binding)
        dense_mla.run(binding=binding)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output, _ = dense_mla.run(binding=binding)
        torch.cuda.synchronize(device)

        for local_length in args.sequence_lengths:
            cache_seqlens.fill_(local_length)
            for _ in range(args.warmups):
                graph.replay()
            torch.cuda.synchronize(device)

            samples: list[float] = []
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / args.iterations)

            actual = captured_output.float().cpu()
            reference = reference_outputs.setdefault(local_length, actual.clone())
            max_abs = float((actual - reference).abs().max().item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.reshape(1, -1), reference.reshape(1, -1), dim=1
                ).item()
            )
            result: dict[str, float | int] = {
                "requested_max_splits": requested_splits,
                "actual_splits": plan.num_splits,
                "chunks_per_split": plan.chunks_per_split,
                "local_cache_tokens": local_length,
                "global_dcp8_context_tokens": local_length * 8,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "cosine_vs_first_plan": cosine,
                "max_abs_vs_first_plan": max_abs,
            }
            all_results.append(result)
            print(json.dumps(result), flush=True)

        del graph, binding, output, scratch, plan
        torch.cuda.empty_cache()

    print(json.dumps({"results": all_results}), flush=True)


if __name__ == "__main__":
    main()
