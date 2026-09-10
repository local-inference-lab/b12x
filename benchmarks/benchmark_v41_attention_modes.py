"""Compare existing V4.1 decode/extend kernels on serving-shaped inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_deepseek_v41_cache_reference,
)
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", default="1,5,6,24,64")
    parser.add_argument("--page-stride-bytes", type=int, default=236544)
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    l2_flush = make_l2_flush_fn(True)
    torch.manual_seed(41083)
    scale = 512**-0.5
    heads, capacity = 8, 64
    swa_kv = torch.randn(128, 512, device=device).bfloat16() * 0.25
    indexed_kv = torch.randn(512, 512, device=device).bfloat16() * 0.25
    q = torch.randn(capacity, heads, 512, device=device).bfloat16() * 0.2
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    records = []

    def cache(values, page, kind):
        packed = pack_deepseek_v41_cache_reference(
            values, page_size=page, cache_kind=kind
        )
        backing = torch.empty(
            packed.shape[0] + 1,
            args.page_stride_bytes,
            device=device,
            dtype=torch.uint8,
        )
        view = backing[:, : packed.shape[1]]
        view[0].fill_(127)
        view[1:].copy_(packed)
        return view

    swa = cache(swa_kv, 32, "swa")
    extra = cache(indexed_kv, 128, "indexed")
    swa_ids = (
        torch.arange(128, device=device, dtype=torch.int32)[None].repeat(capacity, 1)
        + 32
    )
    swa_lengths = torch.full((capacity,), 128, device=device, dtype=torch.int32)
    extra_ids = torch.arange(512, device=device, dtype=torch.int32)[None].repeat(
        capacity, 1
    )
    extra_lengths = torch.full((capacity,), 512, device=device, dtype=torch.int32)
    table = torch.arange(1, 5, device=device, dtype=torch.int32)[None].repeat(
        capacity, 1
    )
    for indexed in (False, True):
        plans = {}
        for mode in ("decode", "extend"):
            plan = mla.plan(
                mla.Caps(
                    device=device,
                    num_q_heads=heads,
                    max_q_rows=capacity,
                    max_width=128 + (512 if indexed else 0),
                    swa_width=128,
                    indexed_width=512 if indexed else 0,
                    swa_page_size=32,
                    indexed_page_size=128,
                    max_page_table_width=4,
                    mode=mode,
                    cache_format="deepseek_v41",
                    use_cuda_graph=True,
                )
            )
            (spec,) = plan.scratch_specs()
            plans[mode] = (
                plan,
                torch.empty(spec.shape, dtype=spec.dtype, device=device),
            )
        for rows in [int(value) for value in args.rows.split(",")]:
            if not 0 < rows <= capacity:
                raise ValueError("rows must fit the serving chunk capacity64")
            expected = compressed_sparse_mla_reference(
                q[:rows],
                swa,
                swa_ids[:rows],
                swa_lengths[:rows],
                extra_k_cache=extra if indexed else None,
                extra_indices=extra_ids[:rows] + 128 if indexed else None,
                extra_topk_lengths=extra_lengths[:rows] if indexed else None,
                swa_page_size=32,
                extra_page_size=128,
                sm_scale=scale,
                attn_sink=sink,
                cache_format="deepseek_v41",
            )
            for mode in ("decode", "extend"):
                plan, scratch = plans[mode]
                kwargs = (
                    dict(
                        indexed_indices=extra_ids[:rows],
                        indexed_lengths=extra_lengths[:rows],
                        indexed_page_table=table[:rows],
                    )
                    if indexed
                    else {}
                )
                binding = mla.bind(
                    plan,
                    scratch=scratch,
                    q=q[:rows],
                    swa_indices=swa_ids[:rows],
                    swa_lengths=swa_lengths[:rows],
                    **kwargs,
                )
                out = torch.empty_like(q[:rows])

                def run():
                    return mla.run(
                        binding=binding,
                        swa_k_cache=swa,
                        indexed_k_cache=extra if indexed else None,
                        swa_page_size=32,
                        indexed_page_size=128,
                        sm_scale=scale,
                        attn_sink=sink,
                        out=out,
                        cache_format="deepseek_v41",
                    )

                run()
                torch.testing.assert_close(out, expected, rtol=0.035, atol=0.035)
                freeze_kernel_resolution("V4.1 attention mode benchmark")
                try:
                    graph = capture_cuda_graph(run, warmup=5)
                    times_us = bench_cuda_graph(
                        graph, replays=30, l2_flush=l2_flush
                    )["replay_us"]
                finally:
                    unfreeze_kernel_resolution()
                row = {
                    "rows": rows,
                    "indexed": indexed,
                    "mode": mode,
                    "median_us": statistics.median(times_us),
                    "planned_max_chunks_per_row": plan.caps.max_chunks_per_row,
                    "samples_us": times_us,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "page_stride_bytes": args.page_stride_bytes,
        "heads": heads,
        "planned_rows": capacity,
        "timer": "CUDA events, CUDA graph replay, cold L2",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
