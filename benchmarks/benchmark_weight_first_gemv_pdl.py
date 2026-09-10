"""Exposed time of a decode projection behind a long, HBM-idle kernel with
programmatic dependent launch (single GPU).

In a tensor-parallel decode step the projection after the attention-output
allreduce (router gate plus shared-expert gate_up, ``[768, 6144]`` BF16)
streams a weight that is the same every step, while the fused PCIe one-shot
allreduce before it keeps a handful of CTAs busy for tens of microseconds
with the HBM idle. A spin kernel of that duration stands in for the
allreduce; the projection that follows is either cuBLAS or the weight-first
brick GEMV (``b12x.gemm.weight_first_gemv``), which stages its brick into
shared memory before ``griddepcontrol.wait``. With the
programmatic-stream-serialization launch attribute on the GEMV and a
``griddepcontrol.launch_dependents`` at the start of the spin, the staging
overlaps the spin and only the activation read, the products and the split
reduction remain exposed.

Variants (graph replay of [spin -> projection]; mean replay time from CUDA
events; per-kernel GPU times from the profiler; the L2 is flushed before
every replay unless ``--warm``):

  cublas        spin -> torch.nn.functional.linear
  wf-serial     spin -> brick GEMV without the launch attribute
  wf-pdl-late   spin without trigger -> brick GEMV with the attribute
                (launch overlap only; the wait releases at spin completion)
  wf-pdl        spin with trigger at its start -> brick GEMV with the
                attribute (staging overlaps the spin)

"after spin" is the replay time minus the spin kernel's GPU time: the time
the projection adds to the step. ``cublas`` is the baseline; the ratio
reported per variant is ``after_spin_us(variant) / after_spin_us(cublas)``
and lower is better. The acceptance condition for the GLM-5.3-NVFP4 serving
launch (tensor parallel 8 over PCIe on RTX PRO 6000 Blackwell Max-Q, where
each rank's router gate plus shared-expert gate_up projection is the
``router_shared`` shape ``[768, 6144]``) is that at 4 rows the ``wf-pdl``
ratio is at most 0.5.

Correctness precedes timing: every captured graph is replayed once and its
output checked against the fp32 product before it is timed. The check
rejects non-finite values and any element outside the BF16 bound
``|y - ref| <= |ref| * 2^-8 + (|x| @ |w|^T) * K * 2^-24 + 1e-6`` (one BF16
ulp plus fp32 accumulation noise); a failure aborts the run. The JSON
output records the command, the source revision and worktree state, the
physical GPU and its operating mode before and after the timed work, the
validation result and raw per-replay timings of every variant, and the
ratios against ``cublas``.

Usage:
  python benchmarks/benchmark_weight_first_gemv_pdl.py [--shapes router_shared,q_b]
      [--rows 4,16] [--spin-us 18] [--spin-ctas 4] [--iters 50] [--warm]
      [--bricks 48x768,32x768] [--depth 1] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import globaltimer_ns, st_global_u64
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm.weight_first_gemv._kernel import (
    BRICKS,
    brick_for,
    compile_weight_first_gemv,
    smem_bytes,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.common import (  # noqa: E402
    benchmark_provenance,
    nvidia_smi_gpu_mode_snapshot,
)

#: Elementwise BF16 acceptance bound against the fp32 product: one BF16 ulp
#: of the reference plus the fp32 accumulation noise of a K-term sum.
BF16_ERROR_BOUND = "|y - ref| <= |ref| * 2^-8 + (|x| @ |w|^T) * K * 2^-24 + 1e-6"


def validate_output(
    y: torch.Tensor, ref32: torch.Tensor, mass: torch.Tensor, k: int
) -> dict:
    """Check ``y`` against the fp32 product under ``BF16_ERROR_BOUND``;
    returns the validation record (``status`` is ``pass`` or the reason)."""
    y32 = y.float()
    finite = bool(torch.isfinite(y32).all().item())
    err = (y32 - ref32).abs()
    tol = ref32.abs() * 2.0**-8 + mass * k * 2.0**-24 + 1e-6
    within = bool((err <= tol).all().item()) if finite else False
    record = {
        "bound": BF16_ERROR_BOUND,
        "finite": finite,
        "max_abs_err": float(err.max().item()) if finite else None,
        "max_err_over_bound": float((err / tol).max().item()) if finite else None,
    }
    if not finite:
        record["status"] = "non-finite output"
    elif not within:
        record["status"] = "output outside the BF16 bound"
    else:
        record["status"] = "pass"
    return record


SHAPES = {  # name: (N, K)
    "router": (256, 6144),
    "router_shared": (768, 6144),
    "q_b": (2048, 1536),
    "qkv_a": (2624, 6144),
    "o_proj": (6144, 2048),
}

_SPIN_THREADS = 768


class SpinKernel:
    """``ctas`` CTAs of 768 threads poll ``%globaltimer`` for ``ns``
    nanoseconds; with ``trigger`` set every CTA executes
    ``griddepcontrol.launch_dependents`` first. Thread 0 of each CTA writes
    its measured spin length to ``out[cta]``."""

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        ns: Int64,
        trigger: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(out_ptr, ns, trigger).launch(
            grid=(ctas, 1, 1), block=[_SPIN_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, out_ptr: cute.Pointer, ns: Int64, trigger: Int32):
        if trigger != Int32(0):
            cute.arch.griddepcontrol_launch_dependents()
        t0 = globaltimer_ns()
        t = globaltimer_ns()
        while t - t0 < ns:
            t = globaltimer_ns()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if Int32(tidx) == Int32(0):
            st_global_u64(
                Int64(out_ptr.toint()) + Int64(bidx) * Int64(8),
                cutlass.Uint64(t - t0),
            )


def compile_spin():
    raw = b12x_compile(
        SpinKernel(),
        make_ptr(cutlass.Int64, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        current_cuda_stream(),
    )

    def launch(out: torch.Tensor, ns: int, trigger: bool, ctas: int) -> None:
        raw(
            make_ptr(
                cutlass.Int64, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            int(ns),
            1 if trigger else 0,
            int(ctas),
            current_cuda_stream(),
        )

    return launch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapes", default="router,router_shared,q_b,qkv_a")
    ap.add_argument("--rows", default="4,16")
    ap.add_argument("--spin-us", type=float, default=18.0)
    ap.add_argument("--spin-ctas", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--warm", action="store_true", help="do not flush the L2 before each replay"
    )
    ap.add_argument("--variants", default="cublas,wf-serial,wf-pdl-late,wf-pdl")
    ap.add_argument(
        "--bricks",
        default="",
        help="comma-separated NTxKT list to sweep instead of brick_for()",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="cp.async groups in flight per CTA (0: unbounded)",
    )
    ap.add_argument("--json", default="", help="write the rows as JSON to this path")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    provenance = benchmark_provenance(sys.argv[1:] if argv is None else argv, dev)
    torch.manual_seed(0)
    spin = compile_spin()
    flush_buf = torch.ones(192 << 20, dtype=torch.uint8, device=dev)
    spin_out = torch.empty(64, dtype=torch.int64, device=dev)
    spin_ns = int(args.spin_us * 1000)
    variants = args.variants.split(",")
    rows_list = [int(v) for v in args.rows.split(",")]
    records = []

    def flush_l2():
        flush_buf.view(torch.int32).sum()

    def measure(graph, iters, do_flush):
        from torch.profiler import ProfilerActivity, profile

        starts, ends = [], []
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                if do_flush:
                    flush_l2()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                graph.replay()
                e.record()
                starts.append(s)
                ends.append(e)
            torch.cuda.synchronize()
        times = {}
        for ev in prof.key_averages():
            if ev.device_type.name != "CUDA":
                continue
            if "reduce_kernel" in ev.key and "ReduceOp" in ev.key:
                continue  # the L2 flush
            times[ev.key] = (ev.count / iters, ev.device_time_total / iters)
        samples = [s.elapsed_time(e) * 1000 for s, e in zip(starts, ends, strict=False)]
        total = sum(samples) / iters
        return times, total, samples

    def part(times, *subs):
        return sum(t for key, (_, t) in times.items() if any(s in key for s in subs))

    for name in args.shapes.split(","):
        n, k = SHAPES[name]
        if args.bricks:
            bricks = [
                tuple(int(v) for v in b.split("x")) for b in args.bricks.split(",")
            ]
        else:
            bricks = [brick_for(n, k)]
        nbytes = n * k * 2
        w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
        for m in rows_list:
            x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
            ref32 = torch.nn.functional.linear(x.float(), w.float())
            mass = torch.nn.functional.linear(x.float().abs(), w.float().abs())
            ref = torch.nn.functional.linear(x, w)
            err_cublas = (ref.float() - ref32).abs().max().item()
            for nt, kt in bricks:
                if (nt, kt) not in BRICKS or k % kt:
                    print(f"{name}: brick {nt}x{kt} skipped")
                    continue
                launch = compile_weight_first_gemv(n, k, nt, kt)
                partial = torch.zeros(k // kt, 16, n, dtype=torch.float32, device=dev)
                counters = torch.zeros(
                    (n + nt - 1) // nt, dtype=torch.int32, device=dev
                )
                y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
                ctas = ((n + nt - 1) // nt) * (k // kt)
                spin(spin_out, 1000, False, args.spin_ctas)
                launch(x, w, y, y, partial, counters, m, n, args.depth, False)
                torch.cuda.synchronize()
                err_wf = (y.float() - ref32).abs().max().item()
                print(
                    f"{name} [{n},{k}] {nbytes / 1e6:.1f} MB, M={m}, brick {nt}x{kt} "
                    f"({ctas} CTAs, {smem_bytes(nt, kt) / 1024:.0f} KB smem), spin "
                    f"{args.spin_us:.0f} us x {args.spin_ctas} CTAs; max abs err vs fp32: "
                    f"cublas {err_cublas:.4f}, brick {err_wf:.4f}"
                )

                def build(variant):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        spin(spin_out, spin_ns, variant == "wf-pdl", args.spin_ctas)
                        if variant == "cublas":
                            torch.nn.functional.linear(x, w, out=y)
                        elif variant == "wf-serial":
                            launch(
                                x, w, y, y, partial, counters, m, n, args.depth, False
                            )
                        else:
                            launch(
                                x, w, y, y, partial, counters, m, n, args.depth, True
                            )
                    return graph

                group = []
                for variant in variants:
                    if variant == "cublas" and (nt, kt) != bricks[0]:
                        continue
                    graph = build(variant)
                    for _ in range(3):
                        flush_l2()
                        graph.replay()
                    torch.cuda.synchronize()
                    # Correctness precedes timing: one checked replay on a
                    # cleared output before the timed replays.
                    y.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    validation = validate_output(y, ref32, mass, k)
                    if validation["status"] != "pass":
                        raise SystemExit(
                            f"{name} M={m} brick {nt}x{kt} {variant}: "
                            f"{validation['status']} ({validation})"
                        )
                    y.zero_()
                    times, total, samples = measure(
                        graph, args.iters, do_flush=not args.warm
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    after_timing = validate_output(y, ref32, mass, k)
                    if after_timing["status"] != "pass":
                        raise SystemExit(
                            f"{name} M={m} brick {nt}x{kt} {variant} after timing: "
                            f"{after_timing['status']} ({after_timing})"
                        )
                    err = validation["max_abs_err"]
                    gemm_us = part(
                        times,
                        "nvjet",
                        "splitK",
                        "gemv",
                        "gemm",
                        "weight_first",
                        "cutlass",
                    )
                    spin_us = part(times, "spin", "Spin")
                    after = total - spin_us
                    print(
                        f"  {variant:12s} gemm {gemm_us:6.1f} us  spin {spin_us:5.1f}  graph total "
                        f"{total:6.1f} us  after spin {after:5.1f}  (max abs err {err:.4f})",
                        flush=True,
                    )
                    record = {
                        "shape": name,
                        "n": n,
                        "k": k,
                        "rows": m,
                        "brick": f"{nt}x{kt}",
                        "variant": variant,
                        "depth": args.depth,
                        "spin_us": args.spin_us,
                        "spin_ctas": args.spin_ctas,
                        "gemm_us": gemm_us,
                        "spin_kernel_us": spin_us,
                        "replay_us": total,
                        "replay_us_samples": samples,
                        "after_spin_us": after,
                        "max_abs_err": err,
                        "validation": validation,
                        "validation_after_timing": after_timing,
                        "kernels": {
                            key: {"count": c, "us": t} for key, (c, t) in times.items()
                        },
                    }
                    records.append(record)
                    group.append(record)
                    del graph
                # Ratios against the cuBLAS baseline of the same shape and
                # row count (measured with the first brick).
                baseline = next(
                    (
                        r
                        for r in records
                        if r["shape"] == name
                        and r["rows"] == m
                        and r["variant"] == "cublas"
                    ),
                    None,
                )
                for record in group:
                    record["after_spin_ratio_vs_cublas"] = (
                        None
                        if baseline is None or baseline["after_spin_us"] <= 0
                        else record["after_spin_us"] / baseline["after_spin_us"]
                    )
                    if record["variant"] == "wf-pdl":
                        ratio = record["after_spin_ratio_vs_cublas"]
                        if ratio is None:
                            # The after-spin time subtracts a profiler device
                            # time from an event-timed mean, so a spin-bound
                            # baseline can leave no positive remainder.
                            print(
                                f"  wf-pdl / cublas after-spin ratio at M={m}: "
                                "unavailable (cublas after_spin_us "
                                f"{None if baseline is None else baseline['after_spin_us']})",
                                flush=True,
                            )
                            continue
                        print(
                            f"  wf-pdl / cublas after-spin ratio at M={m}: "
                            f"{ratio:.3f} (acceptance for router_shared at 4 rows: "
                            f"<= 0.5)",
                            flush=True,
                        )
            torch.cuda.empty_cache()
    if args.json:
        provenance["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot(dev)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "semantic_role": (
                        "exposed time of a decode projection behind an HBM-idle "
                        "kernel with programmatic dependent launch"
                    ),
                    # Every variant passed the BF16 bound before and after
                    # its timed replays; a failure aborts the run before this
                    # record is written.
                    "status": "qualified",
                    "provenance": provenance,
                    "args": vars(args),
                    "correctness_state": (
                        "every variant's graph replay passed the BF16 bound "
                        "before and after its timed replays"
                    ),
                    "comparison": {
                        "baseline": "cublas",
                        "metric": "after_spin_us",
                        "direction": (
                            "after_spin_ratio_vs_cublas = variant / cublas; "
                            "lower is better; acceptance: wf-pdl <= 0.5 at 4 rows "
                            "for router_shared"
                        ),
                    },
                    "records": records,
                },
                fh,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
