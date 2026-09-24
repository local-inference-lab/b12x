"""Benchmark the dynamic-kernel lut_e4m3 trellis W4A8 path at TP12 geometry.

CUDA-graph replay timing of the single fused launch (decode regimes) or the
routing launch plus the two materialized phase kernels (prefill regimes).
Weights are random trellis payloads; correctness at every measured shape is
covered by tests/moe/test_dynamic_w4a8_trellis.py against the torch
reference in tests/_reference/trellis_moe.py.

Examples:
  python benchmarks/benchmark_dynamic_w4a8_trellis.py --tokens 1
  python benchmarks/benchmark_dynamic_w4a8_trellis.py --tokens 1024 --split
  python benchmarks/benchmark_dynamic_w4a8_trellis.py --recipe w4a8_mx
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--recipe", choices=("w4a8_trellis", "w4a8_mx"), default="w4a8_trellis"
    )
    parser.add_argument("--activation", default="situ")
    parser.add_argument(
        "--split",
        action="store_true",
        help="materialized phase-kernel prefill regime (tile 64)",
    )
    parser.add_argument("--intermediate-hadamard", action="store_true")
    parser.add_argument("--direct-lut", action="store_true")
    parser.add_argument("--share-input", action="store_true")
    parser.add_argument("--mac", type=int, default=188)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    import os

    if args.share_input:
        os.environ["BENCH_SHARE"] = "1"
    from tests.moe.test_dynamic_w4a8_trellis import _run_trellis_dynamic
    from b12x._lib.compiler import compile as b12x_compile
    import b12x._lib.compiler as compiler_mod
    import tests.moe.test_dynamic_w4a8_trellis as harness

    captured: dict = {}
    orig_compile = harness.b12x_compile

    def wrap_compile(*cargs, **kwargs):
        compiled = orig_compile(*cargs, **kwargs)

        def wrapper(*call_args):
            captured["fn"] = compiled
            captured["args"] = call_args
            return compiled(*call_args)

        return wrapper

    harness.b12x_compile = wrap_compile
    keep: list = []
    got, want = _run_trellis_dynamic(
        activation=args.activation,
        E=args.experts,
        m=args.tokens,
        K=args.hidden_size,
        n=args.intermediate_size,
        top_k=args.topk,
        seed=args.seed,
        keepalive=keep,
        mac=args.mac,
        recipe=args.recipe,
        tile_m=64 if args.split else 16,
        split_materialized=args.split,
        intermediate_hadamard=args.intermediate_hadamard,
        direct_lut=args.direct_lut,
    )
    harness.b12x_compile = orig_compile
    if args.recipe == "w4a8_trellis":
        cosine = torch.nn.functional.cosine_similarity(
            got.reshape(1, -1), want.reshape(1, -1)
        ).item()
        print(f"reference cosine at bench shape: {cosine:+.6f}")

    fn, cargs = captured["fn"], captured["args"]
    from b12x.moe.fused_moe._impl import current_cuda_stream

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*cargs[:-3], current_cuda_stream(), cargs[-2], cargs[-1])
    iters = args.iterations or (500 if args.tokens <= 8 else 50)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    print(
        f"{args.recipe}{' direct-lut' if args.direct_lut else ''} "
        f"M{args.tokens} E{args.experts} "
        f"H{args.hidden_size} I{args.intermediate_size} "
        f"split={args.split} intermediate_hadamard={args.intermediate_hadamard}: "
        f"median {statistics.median(samples):.3f} us  "
        f"min {min(samples):.3f}  max {max(samples):.3f}"
    )


if __name__ == "__main__":
    main()
