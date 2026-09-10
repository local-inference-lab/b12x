"""Fused PCIe one-shot allreduce followed by the weight-first projection on
every GPU at once.

In a tensor-parallel decode step the router-gate plus shared-expert
gate_up projection (``[768, 6144]`` BF16, 9.4 MB) follows the
attention-output allreduce. With programmatic dependent launch the
projection's CTAs stage their weight bricks while the allreduce waits on the
fabric, and that staging traffic competes with the peers' reads of this
GPU's memory. This benchmark replays ``[one-shot fused add + RMSNorm ->
projection]`` graphs on every rank concurrently and reports the median
replay latency per variant, so the staging depth (``cp.async`` groups in
flight per CTA) can be chosen where the allreduce is not slowed.

Variants: ``none`` (allreduce alone), ``cublas`` (``torch.nn.functional.linear``),
``wf-serial`` (brick GEMV without the launch attribute), ``wf-d<depth>``
(brick GEMV with the attribute at staging depth ``depth``; 0 = unbounded).

``cublas`` is the baseline. The metric of a variant is the slowest rank's
median replay latency; the ratio reported per variant is
``median_us_max_rank(variant) / median_us_max_rank(cublas)`` and lower is
better. The acceptance condition for the GLM-5.3-NVFP4 serving launch
(tensor parallel 8 over PCIe on RTX PRO 6000 Blackwell Max-Q, hidden size
6144, router gate 256 rows plus shared-expert gate_up 512 rows, decode
batches of 4, 8 and 16 rows) is a ``wf-d1`` ratio below 1 at 4, 8 and 16
rows, with the transport configured as that launch configures it
(``B12X_PCIE_TP8_OWNER_REDUCE=0``; the default, 1, is about twice as slow
for these sizes in this harness). The value in force is recorded in the
JSON output.

Correctness precedes timing: after capture every graph is replayed once
and, on every rank, its fused output is checked against an NCCL fp32
reference of the all-reduce, residual add and RMSNorm, and its projection
outputs against the fp32 product of that fused output with the weights
under the BF16 bound ``|y - ref| <= |ref| * 2^-8 + (|x| @ |w|^T) * K *
2^-24 + 1e-6``; a failure on any rank aborts the run before any median is
taken. The JSON output records the command, the source revision and
worktree state, rank 0's physical GPU and its operating mode before and
after the timed work, every rank's validation result and raw replay
samples, and the ratios against ``cublas``.

Usage (8 GPUs):
  B12X_PCIE_TP8_OWNER_REDUCE=0 B12X_ONESHOT_ROWS=4,8,16 B12X_WF_DEPTHS=0,1,2,4 \\
      python benchmarks/benchmark_pcie_oneshot_weight_first.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmarks.common import (  # noqa: E402
    benchmark_provenance,
    nvidia_smi_gpu_mode_snapshot,
)

BF16_ERROR_BOUND = "|y - ref| <= |ref| * 2^-8 + (|x| @ |w|^T) * K * 2^-24 + 1e-6"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _median_latency(
    graph, device, warmup=50, iterations=300
) -> tuple[float, list[float]]:
    """Median and raw per-replay wall times (us) of ``graph`` on this rank."""
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples), samples


def _fused_reference(x_in, residual, norm_w, eps):
    """fp32 all-reduce (NCCL), residual add and RMSNorm of the fused op."""
    reduced = x_in.float().clone()
    dist.all_reduce(reduced)
    residual_out = reduced + residual.float()
    inv = torch.rsqrt(residual_out.square().mean(dim=-1, keepdim=True) + eps)
    return residual_out * inv * norm_w.float(), residual_out


def _bf16_bound_check(y: torch.Tensor, ref32: torch.Tensor, mass: torch.Tensor, k: int):
    y32 = y.float()
    if not bool(torch.isfinite(y32).all().item()):
        return {"status": "non-finite output", "bound": BF16_ERROR_BOUND}
    err = (y32 - ref32).abs()
    tol = ref32.abs() * 2.0**-8 + mass * k * 2.0**-24 + 1e-6
    record = {
        "bound": BF16_ERROR_BOUND,
        "max_abs_err": float(err.max().item()),
        "max_err_over_bound": float((err / tol).max().item()),
    }
    record["status"] = (
        "pass" if bool((err <= tol).all().item()) else "output outside the BF16 bound"
    )
    return record


def _validate_variant(
    name, out, residual_out, produced, x_in, residual, norm_w, weights, hidden
):
    """Validation record of one captured graph after a checked replay:
    the fused output and residual against the NCCL fp32 reference (BF16
    tolerance 2e-2 relative and absolute), and each projection output
    against the fp32 product of the fused output with its weight."""
    expected_out, expected_residual = _fused_reference(x_in, residual, norm_w, 1e-6)
    record = {"variant": name, "checks": {}}
    for label, actual, expected in (
        ("fused_out", out, expected_out),
        ("residual_out", residual_out, expected_residual),
    ):
        finite = bool(torch.isfinite(actual.float()).all().item())
        err = (actual.float() - expected).abs()
        tol = 2e-2 + 2e-2 * expected.abs()
        record["checks"][label] = {
            "status": "pass"
            if finite and bool((err <= tol).all().item())
            else ("non-finite output" if not finite else "outside tolerance"),
            "max_abs_err": float(err.max().item()) if finite else None,
            "tolerance": "|a - ref| <= 2e-2 + 2e-2 * |ref|",
        }
    for index, (y, w) in enumerate(zip(produced, weights, strict=True)):
        ref32 = torch.nn.functional.linear(out.float(), w.float())
        mass = torch.nn.functional.linear(out.float().abs(), w.float().abs())
        record["checks"][f"projection_{index}"] = _bf16_bound_check(
            y, ref32, mass, hidden
        )
    record["status"] = (
        "pass"
        if all(check["status"] == "pass" for check in record["checks"].values())
        else "fail"
    )
    return record


def _worker(
    rank: int, world_size: int, port: int, json_path: str, argv: list[str]
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    provenance = benchmark_provenance(argv, device) if rank == 0 else None
    from b12x.gemm import weight_first_gemv as wf

    hidden = int(os.getenv("B12X_WF_HIDDEN", "6144"))
    n_first = int(os.getenv("B12X_WF_N0", "256"))
    n_out = int(os.getenv("B12X_WF_N", "768"))
    rows_list = tuple(
        int(r) for r in os.getenv("B12X_ONESHOT_ROWS", "4,8,16").split(",")
    )
    depths = tuple(int(d) for d in os.getenv("B12X_WF_DEPTHS", "0,1,2,4").split(","))
    dtype = torch.bfloat16
    max_bytes = max(128 * 1024, max(rows_list) * hidden * 2)
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_bytes,
        max_size=max_bytes,
        single_channel=True,
    )
    pool.for_stream()
    torch.manual_seed(11 + rank)

    def make_weights():
        w = (torch.randn(n_out, hidden, device=device) * 0.02).to(dtype)
        return [w[:n_first].clone(), w[n_first:].clone()]

    weight_cublas = (torch.randn(n_out, hidden, device=device) * 0.02).to(dtype)
    wf_by_depth = {
        d: wf.WeightFirstProjection(make_weights(), depth=d, pdl=True) for d in depths
    }
    wf_serial = wf.WeightFirstProjection(make_weights(), depth=0, pdl=False)
    norm_w = torch.ones(hidden, dtype=dtype, device=device)
    records = []
    validations = []
    try:
        if rank == 0:
            print("rows,variant,median_us_rank0,median_us_max_rank", flush=True)
        for rows in rows_list:
            shape = (rows, hidden)
            x_in = torch.randn(shape, dtype=dtype, device=device) * 0.01
            residual = torch.randn(shape, dtype=dtype, device=device)
            out = torch.empty_like(x_in)
            residual_out = torch.empty_like(x_in)
            pool.prepare_graph_fused_add_rms_norm(x_in)

            def allreduce():
                pool.all_reduce_fused_add_rms_norm(
                    x_in, residual, norm_w, 1e-6, out=out, residual_out=residual_out
                )

            # (name, projection callable returning its outputs, its weights)
            variants = [
                ("none", lambda: (), ()),
                (
                    "cublas",
                    lambda: (torch.nn.functional.linear(out, weight_cublas),),
                    (weight_cublas,),
                ),
                (
                    "wf-serial",
                    lambda: wf_serial(out),
                    (wf_serial.weight[:n_first], wf_serial.weight[n_first:]),
                ),
            ]
            for d in depths:
                proj = wf_by_depth[d]
                variants.append(
                    (
                        f"wf-d{d}",
                        (lambda proj=proj: proj(out)),
                        (proj.weight[:n_first], proj.weight[n_first:]),
                    )
                )
            for name, fn, weights in variants:
                allreduce()
                fn()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with pool.capture(), torch.cuda.graph(graph):
                    allreduce()
                    produced = fn()
                # Correctness precedes timing: a checked replay on every rank,
                # aborted collectively when any rank fails.
                out.zero_()
                residual_out.zero_()
                for y in produced:
                    y.zero_()
                graph.replay()
                torch.cuda.synchronize(device)
                validation = _validate_variant(
                    name,
                    out,
                    residual_out,
                    produced,
                    x_in,
                    residual,
                    norm_w,
                    weights,
                    hidden,
                )
                validation["rows"] = rows
                all_validations = [None] * world_size
                dist.all_gather_object(all_validations, validation)
                failed = [
                    (source, record)
                    for source, record in enumerate(all_validations)
                    if record["status"] != "pass"
                ]
                if failed:
                    raise SystemExit(
                        f"rows={rows} variant={name}: validation failed on rank(s) "
                        f"{[source for source, _ in failed]}: {failed[0][1]}"
                    )
                us, samples = _median_latency(graph, device)
                all_us = [None] * world_size
                all_samples = [None] * world_size
                dist.all_gather_object(all_us, us)
                dist.all_gather_object(all_samples, samples)
                if rank == 0:
                    print(f"{rows},{name},{us:.2f},{max(all_us):.2f}", flush=True)
                    records.append(
                        {
                            "rows": rows,
                            "variant": name,
                            "validation": "pass on every rank",
                            "median_us_rank0": us,
                            "median_us_per_rank": all_us,
                            "median_us_max_rank": max(all_us),
                            "samples_us_per_rank": all_samples,
                        }
                    )
                    validations.append(all_validations)
                del graph
                torch.cuda.synchronize(device)
            if rank == 0:
                # Ratios of the slowest rank's median against the cuBLAS
                # baseline and the added time over the bare allreduce.
                by_name = {r["variant"]: r for r in records if r["rows"] == rows}
                cublas = by_name.get("cublas")
                none = by_name.get("none")
                for record in by_name.values():
                    record["ratio_vs_cublas"] = (
                        None
                        if cublas is None or cublas["median_us_max_rank"] <= 0
                        else record["median_us_max_rank"] / cublas["median_us_max_rank"]
                    )
                    record["added_us_vs_none"] = (
                        None
                        if none is None
                        else record["median_us_max_rank"] - none["median_us_max_rank"]
                    )
                if cublas is not None and "wf-d1" in by_name:
                    print(
                        f"rows={rows} wf-d1 / cublas (slowest-rank medians): "
                        f"{by_name['wf-d1']['ratio_vs_cublas']:.3f} (acceptance: < 1)",
                        flush=True,
                    )
        if rank == 0 and json_path:
            provenance["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot(device)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "semantic_role": (
                            "fused PCIe one-shot allreduce followed by the "
                            "weight-first projection on every rank"
                        ),
                        # Every variant passed its checks on every rank
                        # before timing; a failure aborts all ranks before
                        # this record is written.
                        "status": "qualified",
                        "provenance": provenance,
                        "hidden": hidden,
                        "n": n_out,
                        "n0": n_first,
                        "rows": list(rows_list),
                        "depths": list(depths),
                        "world_size": world_size,
                        "env": {
                            key: os.environ.get(key)
                            for key in (
                                "B12X_PCIE_TP8_OWNER_REDUCE",
                                "B12X_PCIE_SCATTER_GATHER",
                                "B12X_PCIE_DMA_FP8",
                            )
                        },
                        "correctness_state": (
                            "every variant's graph replay passed the fused-output "
                            "and projection checks on every rank before timing"
                        ),
                        "comparison": {
                            "baseline": "cublas",
                            "metric": "median_us_max_rank",
                            "direction": (
                                "ratio_vs_cublas = variant / cublas; lower is "
                                "better; acceptance: wf-d1 < 1 at 4, 8 and 16 rows"
                            ),
                        },
                        "records": records,
                        "validations": validations,
                    },
                    fh,
                    indent=2,
                )
    finally:
        pool.close()
        dist.destroy_process_group()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--json", default="", help="write per-variant medians (all ranks) to this path"
    )
    ap.add_argument(
        "--world-size", type=int, default=int(os.getenv("B12X_ONESHOT_WORLD_SIZE", "8"))
    )
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.json,
            list(sys.argv[1:] if argv is None else argv),
        ),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
