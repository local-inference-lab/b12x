"""Measure resident Engram lookup through device, mapped-host, or Grace storage.

Reported bandwidth counts logical requested bytes. GPU caches can serve repeat
rows; this is not a measurement of raw C2C link bandwidth. No HBM row cache is
implemented. Compare identical traces across placements and record model-level
decode latency separately.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from b12x.sequence import engram
from b12x._lib.platform import probe_platform
from dataclasses import asdict
from benchmarks.benchmark_roce_oneshot import _source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory", choices=("device", "mapped_host", "grace"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-table-size", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--trace",
        type=Path,
        help="CPU int64 [samples,tokens,24] global row ids saved with torch.save",
    )
    args = parser.parse_args()
    if min(args.tokens, args.samples, args.base_table_size) <= 0:
        parser.error("sizes and samples must be positive")
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = engram.build_geometry(
        base_table_size=args.base_table_size, compressed_vocab_size=32
    )
    plan = engram.plan(
        engram.Caps(
            device=device,
            max_tokens=args.tokens,
            max_seqs=1,
            max_requests=1,
            vocab_size=32,
            tp_size=1,
        ),
        token_map=list(range(32)),
        geometry=geometry,
    )
    storage = engram.allocate_storage(plan, memory=args.memory)
    try:
        # Distinct values by row make stale-data and address errors observable.
        for start in range(0, plan.shard_rows, 65536):
            end = min(start + 65536, plan.shard_rows)
            values = (
                torch.arange(start, end, device=storage.weight_load_view.device) % 15
                - 7
            ).float() / 4
            storage.weight_load_view[start:end].copy_(
                values[:, None].expand(-1, 256).to(torch.float8_e4m3fn)
            )
        storage.scales_load_view.fill_(127)
        trace = (
            torch.load(args.trace, weights_only=True, map_location="cpu")
            if args.trace
            else torch.randint(
                plan.table_rows,
                (args.samples, args.tokens, 24),
                generator=torch.Generator().manual_seed(103),
            )
        )
        if trace.shape != (args.samples, args.tokens, 24) or trace.dtype != torch.int64:
            parser.error("trace must be int64 [samples,tokens,24]")
        if torch.any((trace < 0) | (trace >= plan.table_rows)):
            parser.error("trace row ids exceed the selected table")
        ids = trace[0].to(device)
        live = torch.tensor([args.tokens], device=device, dtype=torch.int32)
        output = torch.empty((args.tokens, 6144), device=device, dtype=torch.bfloat16)
        bound = engram.bind_lookup(
            plan, storage=storage, hash_ids=ids, num_tokens=live, out=output
        )
        for _ in range(5):
            engram.run_lookup(bound)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            engram.run_lookup(bound)
        samples = []
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for rows in trace:
            ids.copy_(rows)
            output.fill_(float("nan"))
            before = torch.cuda.memory_allocated()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            assert torch.cuda.memory_allocated() == before
            expected = (
                ((rows % 15 - 7).float() / 4)
                .to(device)
                .repeat_interleave(256, dim=-1)
                .to(torch.bfloat16)
            )
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            samples.append(start.elapsed_time(end) * 1000)
        graph.reset()
        median = statistics.median(samples)
        receipt = {
            "command": sys.argv,
            **_source_state(),
            "platform": asdict(probe_platform(device)),
            "gpu": str(torch.cuda.get_device_properties(device)),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "torch": torch.__version__,
            "storage": storage.stats(),
            "correctness": "exact",
            "graph_us": samples,
            "median_graph_us": median,
            "lookups_per_token": 24,
            "logical_bytes_per_token": 24 * 264,
            "logical_bandwidth_GB_s": args.tokens * 24 * 264 / median / 1000,
            "hbm_row_cache_hit_rate": None,
            "trace_unique_row_fraction": trace.unique().numel() / trace.numel(),
            "table_rows": plan.table_rows,
            "tokens": args.tokens,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    finally:
        storage.close()


if __name__ == "__main__":
    main()
