"""Profile empty tiers and journaled exchanges on a physical SM120 device.

Instrumentation is outside graph replay. Copy event intervals include stream
idle time during synchronous host copies; they are not isolated DMA timings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from b12x.moe.fused_moe._residency_updates import _CudaTransfer
from b12x.sequence._shared.disk_table import MappedHostAllocation
from .sm120_residency_poc import Experiment, load_layer
from .sm120_residency_spectrum import Graphs, Timer, gpu_snapshot, routes


class TransferTrace(_CudaTransfer):
    """Observe the real exchange without changing its ordering or copy API."""

    def __init__(self, experiment):
        super().__init__(experiment.device)
        self.regions = []
        for label, fields in (
            ("vram", experiment.tiers[0].fields),
            ("mapped_host", experiment.tiers[1].fields),
            ("journal", experiment.updates.journal),
            ("device_map", {"map": experiment.mapping}),
            ("host_before", {"map": experiment.updates.before}),
            ("host_after", {"map": experiment.updates.after}),
        ):
            for name, tensor in fields.items():
                self.regions.append((tensor.data_ptr(), tensor.nbytes, label, name))
        self.rows, self.events = [], []
        self.last = time.perf_counter()
        self.stage = 0

    @property
    def phase(self):
        phases = (
            "initial_drain",
            "map_readback",
            "journal",
            "replacement",
            "map_publication",
        )
        return phases[self.stage] if self.stage < len(phases) else "recovery"

    def identify(self, tensor):
        for pointer, size, label, name in self.regions:
            if pointer <= tensor.data_ptr() < pointer + size:
                return label, name
        raise ValueError("copy outside declared exchange storage")

    def copy(self, destination, source):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin = time.perf_counter()
        a.record()
        api = time.perf_counter()
        super().copy(destination, source)
        done = time.perf_counter()
        b.record()
        row = dict(
            operation="copy",
            phase=self.phase,
            source=self.identify(source),
            destination=self.identify(destination),
            bytes=source.nbytes,
            host_gap_us=(begin - self.last) * 1e6,
            api_wall_us=(done - api) * 1e6,
        )
        self.rows.append(row)
        self.events.append((a, b, row))
        self.last = time.perf_counter()

    def synchronize(self):
        begin = time.perf_counter()
        super().synchronize()
        end = time.perf_counter()
        self.rows.append(
            dict(
                operation="synchronize",
                phase=self.phase,
                host_gap_us=(begin - self.last) * 1e6,
                wall_us=(end - begin) * 1e6,
            )
        )
        self.last = time.perf_counter()
        self.stage += 1

    def finish(self):
        for a, b, row in self.events:
            row["event_interval_us"] = a.elapsed_time(b) * 1000
        return self.rows


def canonical_fill_probe(source, device, repeats=12):
    """Bounded one-expert transport lower bounds, without residency publication.

    Only the staging rows are pinned. This does not establish a cache fill or
    rollback contract and must not be reported as a complete promotion cost.
    """
    # All source fields are byte views; concatenation preserves the native data.
    ordinary = torch.cat([field[0].flatten() for field in source.values()])
    size = ordinary.numel()
    owners = [
        MappedHostAllocation((size,), torch.uint8, device, write_combined=wc)
        for wc in (False, True)
    ]
    destination = torch.empty(size, dtype=torch.uint8, device=device)
    transfer = _CudaTransfer(device)
    try:
        for owner in owners:
            transfer.copy(owner.host_view, ordinary)
        transfer.synchronize()
        arms = {
            "pageable_h2d": ordinary,
            "cached_pinned_h2d": owners[0].host_view,
            "write_combined_pinned_h2d": owners[1].host_view,
            "pageable_staged_h2d": owners[0].host_view,
        }
        rows = []
        for repeat in range(repeats):
            for name in arms if repeat % 2 == 0 else list(arms)[::-1]:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                transfer.synchronize()
                begin = time.perf_counter()
                stage_us = 0.0
                if name == "pageable_staged_h2d":
                    transfer.copy(owners[0].host_view, ordinary)
                    stage_us = (time.perf_counter() - begin) * 1e6
                start.record()
                api = time.perf_counter()
                transfer.copy(destination, arms[name])
                submitted = time.perf_counter()
                end.record()
                end.synchronize()
                total = (time.perf_counter() - begin) * 1e6
                event_us = start.elapsed_time(end) * 1000
                torch.testing.assert_close(destination.cpu(), ordinary, atol=0, rtol=0)
                rows.append(
                    dict(
                        arm=name,
                        repeat=repeat,
                        bytes=size,
                        staging_wall_us=stage_us,
                        copy_api_wall_us=(submitted - api) * 1e6,
                        completion_wall_us=total,
                        event_interval_us=event_us,
                        effective_gbps_wall=size / total / 1000,
                        effective_gbps_event=size / event_us / 1000,
                    )
                )
        return dict(
            scope="transport only; no publication, recovery, or miss service",
            pinned_staging_bytes=2 * size,
            rows=rows,
        )
    finally:
        transfer.synchronize()
        for owner in owners:
            owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--live", type=int, nargs="+", default=[1, 128])
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--hot", type=int, default=256)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ncu", action="store_true")
    p.add_argument(
        "--backing-memory",
        choices=("cached", "write_combined"),
        default="write_combined",
    )
    p.add_argument(
        "--journal-memory",
        choices=("cached", "write_combined"),
        default="write_combined",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        gpu_before=gpu_snapshot(),
        affinity=sorted(os.sched_getaffinity(0)),
        backing_memory=args.backing_memory,
        journal_memory=args.journal_memory,
        torch_threads=torch.get_num_threads(),
        status="running",
    )
    record["command"] = sys.argv
    record["toolchain"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "nvidia-cutlass-dsl", "triton", "cuda-bindings")
    }
    record["torch_cuda"] = torch.version.cuda
    record["configuration"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    record["topology"] = {}
    for name, command in (
        ("gpu", ["nvidia-smi", "topo", "-m"]),
        ("numa", ["numactl", "--hardware"]),
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            record["topology"][name] = dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
        except OSError as error:
            record["topology"][name] = dict(error=str(error))
    record["source_sha256"] = {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(
            [
                *Path("b12x").rglob("*.py"),
                *Path("benchmarks/moe").glob("sm120_residency*.py"),
            ]
        )
    }
    path = args.output / "results.json"
    path.write_text(json.dumps(record, indent=2))
    source, checksum = load_layer(
        args.checkpoint, "model.language_model.layers.0.mlp.experts", args.experts
    )
    record["checkpoint_fields_sha256"] = checksum
    e = Experiment(
        source,
        hot=args.hot,
        capacity=max(args.live),
        topk=args.top_k,
        backing_write_combined=args.backing_memory == "write_combined",
        journal_write_combined=args.journal_memory == "write_combined",
    )
    try:
        timer = Timer(e)
        record["latency"] = []
        for live in args.live:
            graphs = Graphs(e, live)
            try:
                for count in (0, live * args.top_k):
                    graphs.inputs(routes(e.e, args.hot, live, args.top_k, count))
                    check = graphs.validate(allocator=True)
                    if args.ncu:
                        record["profiled_route_count"] = (
                            graphs.bindings[1].packed_route_count.cpu().tolist()
                        )
                        record["profiled_correctness"] = check
                        torch.cuda.cudart().cudaProfilerStart()
                        graphs.graphs["cold_operator"].replay()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        record["status"] = "profiled"
                        return
                    values = {
                        name: [timer.sample(graph, 32) for _ in range(3)]
                        for name, graph in graphs.graphs.items()
                    }
                    label = f"m{live}-cold{count}"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for _ in range(4):
                            graphs.graphs["cold_operator"].replay()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.output / f"{label}.trace.json"))
                    record["latency"].append(
                        dict(live=live, cold=count, correctness=check, timings=values)
                    )
                    print(label, flush=True)
            finally:
                graphs.close()
        record["exchanges"] = []
        for instrumented in (False, True, False, True, False, False, False, False):
            transfer = TransferTrace(e) if instrumented else _CudaTransfer(e.device)
            e.updates.transfer = transfer
            begin = time.perf_counter()
            e.updates.exchange(
                ((0, args.hot),), expected=e.updates.snapshot(), quiescent=True
            )
            elapsed = (time.perf_counter() - begin) * 1e6
            record["exchanges"].append(
                dict(
                    instrumented=instrumented,
                    total_us=elapsed,
                    calls=transfer.finish() if instrumented else [],
                )
            )
            print("exchange", instrumented, elapsed, flush=True)
        record["canonical_fill_transport"] = canonical_fill_probe(source, e.device)
        record["status"] = "passed"
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        record["gpu_after"] = gpu_snapshot()
        path.write_text(json.dumps(record, indent=2) + "\n")
        e.close()


if __name__ == "__main__":
    main()
