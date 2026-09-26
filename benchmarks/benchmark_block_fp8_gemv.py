#!/usr/bin/env python3
"""Small-row serialized block-FP8: GEMV regime vs the dense GEMM plan.

Times fixed ``expected_m`` plans (the decode shape vLLM declares per CUDA graph
size) through ``blockscaled.mm_block_fp8``, once with the GEMV regime and once
with it disabled, as CUPTI kernel time (median).  ``--cold`` flushes L2 before
every call, which is what a decode step sees for weights larger than L2.

    python benchmarks/benchmark_block_fp8_gemv.py [--cold] [--rows 1,2,4,8]
        [--shapes 3712x4096,1024x7168]
"""

from __future__ import annotations

import argparse
import contextlib
import statistics

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _block_fp8_gemv
from b12x.preparation import PreparationSession, PreparedCall

DEFAULT_SHAPES = [
    (n, k)
    for k in (1536, 4096, 7168)
    for n in (256, 512, 1024, 2048, 3712, 4096, 6144, 8192, 16384)
]


@contextlib.contextmanager
def _gemv_enabled(enabled: bool):
    original = _block_fp8_gemv.supports
    if not enabled:
        _block_fp8_gemv.supports = lambda *args: False
    try:
        yield
    finally:
        _block_fp8_gemv.supports = original


def _prepare(lhs, lhs_scale, rhs, rhs_scale, rows):
    query = blockscaled.FixedBlockscaledQuery(
        recipe="block_fp8", call_kind="serialized", max_rows=rows,
        in_features=rhs.shape[1], padded_in_features=rhs.shape[1],
        out_features=rhs.shape[0], input_dtype="float8_e4m3fn",
        output_dtype="bfloat16", expected_m=rows,
    )
    plan = blockscaled.plan(query)

    def call(state):
        return PreparedCall(run=lambda: state.run_serialized(
            lhs, lhs_scale, rhs, rhs_scale, None, ab_dtype="float8_e4m3fn",
            sf_dtype="float32", c_dtype="bfloat16", sf_vec_size=128,
            block_fp8=True, stream=None,
        ))

    session = PreparationSession(device=lhs.device, autotune=False, compile_workers=4)
    session.__enter__()
    session.prepare((plan.request(name="bench", prepare_call=call),))
    return session, plan


def _kernel_us(fn, flush, iters, names):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if flush is not None:
                flush.zero_()
            torch.cuda._sleep(20_000)
            fn()
        torch.cuda.synchronize()
    events = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    per_call, current = [], None
    for event in events:
        name = event.name.lower()
        if any(tag in name for tag in ("fill", "memset", "sleep", "spin", "zero")):
            if current is not None:
                per_call.append(current)
                current = None
            continue
        current = (current or 0.0) + (event.time_range.end - event.time_range.start)
        names.add(event.name.split("(")[0][:48])
    if current is not None:
        per_call.append(current)
    return statistics.median(per_call)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,2,4,8")
    parser.add_argument("--shapes", default=None, help="NxK list, e.g. 3712x4096")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--iters", type=int, default=25)
    args = parser.parse_args()
    shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        if args.shapes else DEFAULT_SHAPES
    )
    rows_list = [int(r) for r in args.rows.split(",")]
    device = torch.device("cuda")
    flush = torch.empty(320 << 20, dtype=torch.uint8, device=device) if args.cold else None
    print(f"{'N':>6} {'K':>6} {'M':>2} {'dense_us':>9} {'gemv_us':>8} {'speedup':>7}  eligible")
    for n, k in shapes:
        gen = torch.Generator(device=device).manual_seed(n * 131 + k)
        rhs = torch.randn((n, k), generator=gen, device=device).to(torch.float8_e4m3fn)
        rhs_scale = torch.rand(((n + 127) // 128, k // 128), generator=gen, device=device)
        for rows in rows_list:
            lhs = torch.randn((rows, k), generator=gen, device=device).to(torch.float8_e4m3fn)
            lhs_scale = torch.rand((rows, k // 128), generator=gen, device=device)
            times, kernels = {}, {}
            for label, enabled in (("dense", False), ("gemv", True)):
                with _gemv_enabled(enabled):
                    original_max = _block_fp8_gemv.MAX_OUT_FEATURES
                    _block_fp8_gemv.MAX_OUT_FEATURES = 1 << 30  # time every shape
                    try:
                        session, plan = _prepare(lhs, lhs_scale, rhs, rhs_scale, rows)

                        def fn(plan=plan):
                            return blockscaled.mm_block_fp8(
                                lhs, lhs_scale, rhs, rhs_scale, plan=plan)
                        names = set()
                        times[label] = _kernel_us(fn, flush, args.iters, names)
                        kernels[label] = ",".join(sorted(names))
                        session.__exit__(None, None, None)
                    finally:
                        _block_fp8_gemv.MAX_OUT_FEATURES = original_max
            eligible = _block_fp8_gemv.supports(rows, n, k)
            print(f"{n:6d} {k:6d} {rows:2d} {times['dense']:9.2f} {times['gemv']:8.2f} "
                  f"{times['dense'] / times['gemv']:7.2f}  {'yes' if eligible else 'no'}"
                  f"  [{kernels['dense']} | {kernels['gemv']}]", flush=True)


if __name__ == "__main__":
    main()
