"""Compare prepared hierarchical BF16 reduction with NCCL on PCIe GPUs.

Run with torchrun and 9, 10, 12 or 16 visible GPUs. Each arm reduces the same
immutable inputs into caller-owned output. Correctness checks use FP32 island
sums and changed-input CUDA graphs before balanced, rank-maximum timings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from b12x.comm.pcie import AllReduce
from b12x.comm.pcie import _owner_preparation as preparation
from b12x.preparation import PreparationSession


def graph_latency(graph, repeats, launches):
    measurements = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        maximum = torch.tensor(start.elapsed_time(end) * 1000 / launches, device="cuda")
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        measurements.append(float(maximum.item()))
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elements", nargs="+", type=int, default=[3584, 7168, 14336, 57344]
    )
    parser.add_argument("--launches", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    world = dist.get_world_size()
    bootstrap = torch.ones(7168, device=device, dtype=torch.bfloat16)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize()
    runtime = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=max(args.elements) * 2,
    )
    if runtime.algorithm != "hierarchical":
        raise AssertionError("benchmark requires bounded-peer hierarchical dispatch")
    properties = torch.cuda.get_device_properties(device)
    nccl_version = ctypes.c_int()
    nccl_path = os.environ.get("VLLM_NCCL_SO_PATH", "libnccl.so.2")
    status = ctypes.CDLL(nccl_path).ncclGetVersion(ctypes.byref(nccl_version))
    if status:
        raise RuntimeError(f"ncclGetVersion failed: {status}")
    report = {
        "world_size": world,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": nccl_version.value,
        "nccl_library": nccl_path,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": str(properties),
        "gpu_uuid": str(properties.uuid),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("B12X_PCIE_", "NCCL_"))
        },
        "conditions": "no concurrent GPU requests; existing allocations are not evicted",
        "free_memory_bytes": torch.cuda.mem_get_info(device)[0],
        "latency_unit": "microseconds per collective; slowest rank",
        "cases": [],
    }
    graphs = {}
    try:
        with PreparationSession(
            device=device, autotune=False, compile_workers=1
        ) as session:
            for elements in args.elements:
                generator = torch.Generator(device=device).manual_seed(113 + rank)
                inp = torch.randn(
                    elements, generator=generator, device=device, dtype=torch.bfloat16
                )
                saved = inp.clone()
                output = torch.empty_like(inp)
                nccl_output = torch.empty_like(inp)
                plan = runtime.plan(inp, out=output)
                request = plan.request(
                    name=f"hierarchical_{elements}",
                    prepare_call=lambda state: preparation.prepared_call(
                        state, inp=inp, out=output
                    ),
                )
                session.prepare((request,))
                gathered = [torch.empty_like(inp) for _ in range(world)]
                dist.all_gather(gathered, inp)
                reference = torch.zeros(elements, device=device, dtype=torch.float32)
                for first in range(0, world, 4):
                    partial = torch.zeros_like(reference)
                    for source in gathered[first : first + 4]:
                        partial.add_(source.float())
                    reference.add_(partial)
                runtime.all_reduce(inp, out=output, plan=plan)
                torch.cuda.synchronize()
                torch.testing.assert_close(output, reference.bfloat16(), rtol=0, atol=0)
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                assert torch.equal(inp, saved)
                error = float((output.float() - reference).abs().max().item())

                for name in ("nccl", "hierarchical"):
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        for _ in range(args.launches):
                            if name == "nccl":
                                nccl_output.copy_(inp)
                                dist.all_reduce(nccl_output)
                            else:
                                runtime.all_reduce(inp, out=output, plan=plan)
                    graphs[name] = graph
                for step in range(8):
                    # Integers keep every NCCL BF16 partial sum exact through
                    # TP16. Random-data accuracy is checked separately above.
                    inp.fill_(rank + 1 + step)
                    output.fill_(float("nan"))
                    nccl_output.fill_(float("nan"))
                    dist.barrier()
                    for graph in graphs.values():
                        graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.full_like(
                        inp, world * (1 + step) + world * (world - 1) / 2
                    )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    torch.testing.assert_close(nccl_output, expected, rtol=0, atol=0)
                inp.copy_(saved)
                for _ in range(4):
                    for graph in graphs.values():
                        graph.replay()
                torch.cuda.synchronize()
                timings = {name: [] for name in graphs}
                for sample in range(args.samples):
                    order = (
                        ("nccl", "hierarchical")
                        if sample % 2 == 0
                        else ("hierarchical", "nccl")
                    )
                    for name in order:
                        timings[name].extend(
                            graph_latency(graphs[name], 1, args.launches)
                        )
                result = {
                    "elements": elements,
                    "bytes": elements * 2,
                    "exact_grouped_fp32_rounding": True,
                    "changed_graph_inputs": 8,
                    "graph_launches_per_replay": args.launches,
                    "bf16_rounding_max_absolute_error": error,
                    "latency_us": timings,
                    "median_us": {
                        name: median(values) for name, values in timings.items()
                    },
                    "nccl_over_hierarchical_speedup": median(timings["nccl"])
                    / median(timings["hierarchical"]),
                }
                report["cases"].append(result)
                if rank == 0:
                    print(json.dumps(result), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
                session.release(plan)
    except BaseException:
        # CUDA/NCCL teardown can block after an assertion inside graph capture.
        # Publish the original failure before entering collective cleanup.
        traceback.print_exc()
        raise
    finally:
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
