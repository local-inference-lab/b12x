"""Kernel time of the top-k selection sort (``b12x.attention.topk_sort``) at
decode row counts (single GPU).

The sparse indexer of a decode step emits, per query row, ``topk`` logical KV
positions in arrival order; ``sort_convert`` rewrites each row ascending and
converts the positions to physical cache slots in place. In a decode step the
sort runs once per indexer layer, so its kernel time bounds what a side
stream has to hide.

For every (rows, seq_len, max_positions) case the sort is captured alone in
a CUDA graph, its replay is checked against ``sort_convert_reference``
(correctness precedes timing; a mismatch aborts the run), and the graph is
replayed ``--iters`` times: the time is the mean of CUDA-event timings around
the replay, and the kernel's own GPU time comes from the profiler. The
selection is rebuilt before every replay (a sorted row would otherwise be
re-sorted as if its slots were positions). ``max_positions`` is the compile
key (bitmap words); the row's own length decides how many bitmap words are
cleared and scanned. There is no baseline arm: the kernel time is absolute
(lower is better).

The JSON output records the command, the source revision and worktree
state, the physical GPU and its operating mode before and after the timed
work, the correctness state, and the raw per-replay timings of every case.

Usage:
  python benchmarks/benchmark_topk_sort.py [--rows 4,8,16]
      [--seq-lens 1024,30000,131072] [--max-positions 32768,524288]
      [--topk 2048] [--iters 50] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _case(rows: int, topk: int, seq_len: int, block_size: int, device, seed: int):
    g = torch.Generator().manual_seed(seed)
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    width = max((seq_len + block_size - 1) // block_size, 1)
    block_table = torch.randperm(1 << 20, generator=g)[: rows * width].view(rows, width)
    for row in range(rows):
        count = min(topk, seq_len)
        indices[row, :count] = torch.randperm(seq_len, generator=g)[:count].to(
            torch.int32
        )
    return (
        indices.to(device),
        torch.full((rows,), seq_len, dtype=torch.int32, device=device),
        block_table.to(torch.int32).to(device),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="4,8,16")
    parser.add_argument("--seq-lens", default="1024,30000,131072")
    parser.add_argument("--max-positions", default="32768,524288")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from b12x.attention import topk_sort
    from benchmarks.common import benchmark_provenance, nvidia_smi_gpu_mode_snapshot

    device = torch.device("cuda")
    provenance = benchmark_provenance(argv, device)
    records = []
    for max_positions in (int(x) for x in args.max_positions.split(",")):
        topk_sort.precompile(max_positions, device)
        for rows in (int(x) for x in args.rows.split(",")):
            for seq_len in (int(x) for x in args.seq_lens.split(",")):
                if seq_len > max_positions:
                    continue
                indices, lens, table = _case(
                    rows, args.topk, seq_len, args.block_size, device, seed=rows
                )
                logical = indices.clone()
                expected = topk_sort.sort_convert_reference(
                    indices.cpu(),
                    lens.cpu(),
                    table.cpu(),
                    args.block_size,
                    max_positions,
                )
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize(device)
                with torch.cuda.graph(graph):
                    topk_sort.sort_convert(
                        indices, lens, table, args.block_size, max_positions
                    )
                torch.cuda.synchronize(device)
                indices.copy_(logical)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.equal(indices.cpu(), expected), "sort result differs"

                times = []
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(args.iters):
                        indices.copy_(logical)
                        torch.cuda.synchronize(device)
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        graph.replay()
                        e.record()
                        torch.cuda.synchronize(device)
                        times.append(s.elapsed_time(e) * 1000.0)
                kernel_us = [
                    ev.device_time
                    for ev in prof.key_averages()
                    if "TopkSortConvert" in ev.key or "topk_sort" in ev.key
                ]
                if not kernel_us:
                    raise SystemExit(
                        "no profiler event matched the sort kernel; kernel time "
                        f"is unavailable for rows={rows} seq_len={seq_len} "
                        f"max_positions={max_positions}"
                    )
                replay_us = sum(times) / len(times)
                record = {
                    "rows": rows,
                    "seq_len": seq_len,
                    "max_positions": max_positions,
                    "topk": args.topk,
                    "correctness": "graph replay equals sort_convert_reference",
                    "replay_us_mean": replay_us,
                    "replay_us_samples": times,
                    # Mean device time per replay summed over every matching
                    # kernel event (one event per launch in the captured graph).
                    "kernel_us_mean": sum(kernel_us),
                    "kernel_us_events": len(kernel_us),
                }
                records.append(record)
                print(
                    f"rows {rows:2d} seq {seq_len:6d} max_positions {max_positions:6d}: "
                    f"kernel {record['kernel_us_mean']:6.2f} us  "
                    f"replay {replay_us:6.2f} us"
                )
                del graph
    if args.json:
        provenance["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot(device)
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "semantic_role": "topk_sort kernel time per launch at decode row counts",
                    # Every case's replay was checked against the reference
                    # before its timing; a mismatch aborts the run before this
                    # record is written.
                    "status": "qualified",
                    "provenance": provenance,
                    "args": vars(args),
                    "correctness_state": (
                        "every case's graph replay matched sort_convert_reference "
                        "before it was timed"
                    ),
                    "comparison": {
                        "baseline": None,
                        "direction": "absolute kernel time per launch; lower is better",
                    },
                    "device": torch.cuda.get_device_name(device),
                    "records": records,
                },
                fh,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
