"""Compare staged scalar and BF16 tensor-core MXFP4 paged scoring.

Both public plans read identical quantized operands and paged metadata. CUDA
graphs exclude Python overhead; score output must be bitwise equal before any
timing is reported. Selection and tensor-parallel communication are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention import dsa_indexer as api


def measure(rows: int, width: int, heads: int, repeats: int) -> dict:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4132)
    q = torch.randn((rows, heads, 128), device=device).bfloat16()
    keys = torch.randn((width, 128), device=device).bfloat16()
    weights = torch.randn((rows, heads), device=device).bfloat16() / 64
    pages = (width + 63) // 64
    physical = torch.randperm(pages, device=device).int() + 7
    pool = torch.empty((pages + 7, api.index_mxfp4_page_bytes()), device=device, dtype=torch.uint8)
    positions = torch.arange(width, device=device)
    slots = physical[positions // 64].long() * 64 + positions % 64
    api.quantize_write_index_k_mxfp4(keys, index_k_cache=pool, slot_mapping=slots)
    packed = torch.empty((rows, heads, 64), device=device, dtype=torch.uint8)
    scales = torch.empty((rows, heads, 4), device=device, dtype=torch.uint8)
    api.quantize_q_mxfp4(q, q_mxfp4=packed, q_scales=scales)
    shared = dict(
        q_mxfp4=packed, q_scales=scales, query_weights=weights,
        index_k_cache=pool, page_table=physical[None],
        cache_lengths=torch.full((rows,), width, device=device, dtype=torch.int32),
        active_width=torch.tensor([width], device=device, dtype=torch.int32),
        output_indices=torch.empty((rows, 512), device=device, dtype=torch.int32),
    )
    bindings = {}
    for mode in ("decode", "prefill"):
        plan = api.plan(api.Caps(
            device=device, num_q_heads=heads, max_q_rows=max(256, rows),
            max_page_table_width=pages, topk=512, cache_format="mxfp4", mode=mode,
        ))
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        bindings[mode] = api.bind(plan, scratch=scratch, **shared)
    expected = api.score(bindings["decode"]).clone()
    actual = api.score(bindings["prefill"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graphs = {}
    freeze_kernel_resolution("MXFP4 score benchmark fixed-capacity replay")
    try:
        for mode, binding in bindings.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    api.score(binding)
            graphs[mode] = graph
        samples = {mode: [] for mode in bindings}
        for repeat in range(repeats):
            order = ("decode", "prefill") if repeat % 2 else ("prefill", "decode")
            for mode in order:
                graphs[mode].replay()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 10)
        return {
            "rows": rows, "keys": width, "heads": heads,
            "device": torch.cuda.get_device_name(device),
            "score_bitwise_equal": True, "samples_us": samples,
            "median_us": {mode: statistics.median(values) for mode, values in samples.items()},
            "scalar_over_tensorcore": statistics.median(samples["decode"]) / statistics.median(samples["prefill"]),
        }
    finally:
        unfreeze_kernel_resolution()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--keys", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    for rows in args.rows:
        for width in args.keys:
            print(json.dumps(measure(rows, width, args.heads, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
