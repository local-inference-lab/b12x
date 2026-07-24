"""Benchmark the reusable BF16->MX-FP6 TMA quantization kernel module."""

from __future__ import annotations

import argparse
import pathlib
import statistics as _stats
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from sparkinfer.quantization.mxfp6 import allocate_bf16_to_fp6_tma_outputs, compile_bf16_to_fp6_tma


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    m = int(args.M)
    k = int(args.K)
    dev = torch.device("cuda")
    torch.manual_seed(42)
    bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    gs = torch.tensor([1.0], dtype=torch.float32, device=dev)
    rows_padded = ((m + 127) // 128) * 128
    cols_pad_sf = ((k // 32 + 3) // 4) * 4
    inp = (
        bf16
        if rows_padded == m and bf16.is_contiguous()
        else torch.zeros((rows_padded, k), dtype=torch.bfloat16, device=dev)
    )
    if rows_padded != m:
        inp[:m].copy_(bf16)
    # The kernel is compiled for rows_padded rows and its launch closure
    # reshapes the flat code buffer to (1, rows_padded, packed_k); allocate
    # accordingly or any m % 128 != 0 raises a size-mismatch RuntimeError.
    out = allocate_bf16_to_fp6_tma_outputs(rows_padded, k, device=dev)
    compiled = compile_bf16_to_fp6_tma(rows_padded, k)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)" if l2_flush else "off"
    print("Compiled OK (MX-FP6 E3M2)")
    print(f"L2 flush: {flush_desc}")

    def launch() -> None:
        compiled(inp, gs, out.packed_a_flat, out.scale_flat)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        if l2_flush is not None:
            l2_flush()
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for idx in range(args.iters):
        if l2_flush is not None:
            l2_flush()
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    times_ms = [starts[idx].elapsed_time(ends[idx]) for idx in range(args.iters)]
    med_us = _stats.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    packed_k = (k * 3) // 4
    read_bytes = rows_padded * k * 2
    write_bytes = rows_padded * packed_k + rows_padded * cols_pad_sf
    bw = (read_bytes + write_bytes) / (med_us * 1e-6) / 1e9
    print(f"M={m} K={k}  graph replay median: {med_us:.1f} us  (min {min_us:.1f})  BW: {bw:.1f} GB/s")


if __name__ == "__main__":
    main()
