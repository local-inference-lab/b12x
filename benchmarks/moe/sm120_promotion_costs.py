"""Complete quiescent canonical fills versus journaled pair exchanges on SM120.

Every timed operation drains the device, checks the device map, transfers native
payload fields, waits, publishes the map and advances a usable generation. The
canonical arms retain an immutable mapped copy of all experts for direct miss
service and recovery. Pageable/staged arms compare fill transports only; they do
not establish an unpinned full-model miss-service implementation.
"""

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

from b12x.moe.fused_moe._residency_storage import align
from b12x.moe.fused_moe._residency_updates import _CudaTransfer
from .sm120_canonical_fill import CanonicalFills
from .sm120_residency_poc import Experiment, load_layer
from .sm120_residency_spectrum import Graphs, gpu_snapshot


class TransactionTrace(_CudaTransfer):
    def __init__(self, e, arm, source):
        super().__init__(e.device)
        self.arm = arm
        self.regions = []
        self.rows = []
        self.events = []
        self.stage = 0
        self.last = time.perf_counter()
        groups = [
            ("vram", e.tiers[0].fields),
            ("mapped", e.tiers[1].fields),
            ("map_device", {"map": e.mapping}),
            ("map_before", {"map": e.updates.before}),
            ("map_after", {"map": e.updates.after}),
        ]
        if arm == "exchange":
            groups.append(("journal", e.updates.journal))
        elif arm != "pinned_fill":
            groups.append(("pageable", source))
        if arm == "staged_fill":
            groups.append(("staging", e.updates.staging))
        for label, fields in groups:
            for name, t in fields.items():
                self.regions.append((t.data_ptr(), t.nbytes, label, name))
        # Mapped CPU/device aliases can differ. Identify both without guessing.
        if arm != "exchange":
            for name, t in e.updates.canonical.items():
                self.regions.append((t.data_ptr(), t.nbytes, "mapped", name))

    def identify(self, tensor):
        for ptr, size, label, name in self.regions:
            if ptr <= tensor.data_ptr() < ptr + size:
                return (label, name)
        raise ValueError("unclassified transaction copy")

    def phase(self):
        names = ["drain", "map_readback"]
        if self.arm == "exchange":
            names += ["journal"]
        if self.arm == "staged_fill":
            names += ["staging"]
        names += ["payload", "publication"]
        return names[self.stage] if self.stage < len(names) else "recovery"

    def copy(self, dst, src):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin = time.perf_counter()
        a.record()
        start = time.perf_counter()
        super().copy(dst, src)
        end = time.perf_counter()
        b.record()
        row = dict(
            operation="copy",
            phase=self.phase(),
            source=self.identify(src),
            destination=self.identify(dst),
            bytes=src.nbytes,
            api_wall_us=(end - start) * 1e6,
            gap_us=(begin - self.last) * 1e6,
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
                phase=self.phase(),
                wall_us=(end - begin) * 1e6,
                gap_us=(begin - self.last) * 1e6,
            )
        )
        self.stage += 1
        self.last = time.perf_counter()

    def finish(self):
        for a, b, row in self.events:
            row["event_interval_us"] = a.elapsed_time(b) * 1000
        return self.rows


def configure_source(e, source, mode):
    if mode == "pinned_fill":
        return
    old = e.updates
    staging = None
    if mode == "staged_fill":
        offset = 0
        staging = {}
        for name, value in source.items():
            n = value[0].numel()
            staging[name] = e.journal_owner.host_view[offset : offset + n].view(
                value.shape[1:]
            )
            offset += align(2 * n)
    e.updates = CanonicalFills(
        resident=old.resident,
        canonical=old.canonical,
        source=source,
        staging=staging,
        mapping=e.mapping,
        expert_map=old.snapshot().expert_map,
        before=old.before,
        after=old.after,
        transfer=old.transfer,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prefix", default="model.language_model.layers.0.mlp.experts")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--hot", type=int, default=256)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=20)
    torch.manual_seed(0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running",
        command=sys.argv,
        activation_seed=0,
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(
                [*Path("b12x").rglob("*.py"), *Path("benchmarks/moe").glob("sm120*.py")]
            )
        },
        gpu_before=gpu_snapshot(),
        affinity=sorted(os.sched_getaffinity(0)),
        torch_cuda=torch.version.cuda,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "triton", "cuda-bindings")
        },
    )
    record["topology"] = {}
    for name, command in [
        ("gpu", ["nvidia-smi", "topo", "-m"]),
        ("numa", ["numactl", "--hardware"]),
        ("pci", ["lspci", "-tv"]),
        ("cmdline", ["cat", "/proc/cmdline"]),
        ("memory_policy", ["numactl", "--show"]),
    ]:
        try:
            r = subprocess.run(command, capture_output=True, text=True)
            record["topology"][name] = dict(
                returncode=r.returncode, stdout=r.stdout, stderr=r.stderr
            )
        except OSError as error:
            record["topology"][name] = dict(error=str(error))
    path = a.output / "results.json"
    path.write_text(json.dumps(record, indent=2))
    experiments = {}
    graphs = {}
    results = []
    try:
        source, digest = load_layer(a.checkpoint, a.prefix, a.experts)
        record["checkpoint_fields_sha256"] = digest
        record["geometry"] = dict(
            experts=a.experts,
            hot=a.hot,
            topk=a.top_k,
            hidden=source["w2"].shape[1],
            intermediate=source["w13"].shape[1] // 2,
            capacity=128,
        )
        for arm in ("exchange", "pinned_fill", "pageable_fill", "staged_fill"):
            e = Experiment(
                source,
                hot=a.hot,
                capacity=128,
                topk=a.top_k,
                backing_write_combined=False,
                journal_write_combined=False,
                canonical_backing=arm != "exchange",
                cache_dir=a.output / "cache",
            )
            experiments[arm] = e
            if arm != "exchange":
                configure_source(e, source, arm)
            graphs[arm] = Graphs(e, 1)
            graphs[arm].inputs(
                torch.tensor([[0, a.hot] * ((a.top_k + 1) // 2)])[:, : a.top_k]
            )
            graphs[arm].validate(allocator=True)
        record["memory"] = {
            arm: dict(
                resident_bytes=e.tiers[0].slab.nbytes,
                mapped_bytes=e.tiers[1].slab.nbytes,
                staging_and_map_allocation=e.journal_owner.host_view.nbytes,
                payload_bytes=e.row_bytes,
                extra_pageable_source_bytes=sum(t.nbytes for t in source.values())
                if arm in ("pageable_fill", "staged_fill")
                else 0,
            )
            for arm, e in experiments.items()
        }
        for repeat in range(a.repeats + 2):
            instrumented = repeat >= a.repeats
            for arm in (
                list(experiments) if repeat % 2 == 0 else list(experiments)[::-1]
            ):
                e = experiments[arm]
                before = e.updates.snapshot()
                ptrs = e.pointers()
                candidate, victim = (
                    (a.hot, 0) if before.expert_map[a.hot][0] else (0, a.hot)
                )
                e.updates.transfer = (
                    TransactionTrace(e, arm, source)
                    if instrumented
                    else _CudaTransfer(e.device)
                )
                start = time.perf_counter()
                if arm == "exchange":
                    after = e.updates.exchange(
                        ((candidate, victim),), expected=before, quiescent=True
                    )
                else:
                    after = e.updates.promote(
                        candidate, victim, expected=before, quiescent=True
                    )
                elapsed = (time.perf_counter() - start) * 1e6
                transfers = e.updates.transfer.finish() if instrumented else None
                assert (
                    after.generation == before.generation + 1 and ptrs == e.pointers()
                )
                validation = graphs[arm].validate(
                    allocator=repeat in (0, a.repeats - 1)
                )
                results.append(
                    dict(
                        arm=arm,
                        repeat=repeat,
                        instrumented=instrumented,
                        transaction_wall_us=elapsed,
                        generation=after.generation,
                        promotion=[candidate, victim],
                        transfers=transfers,
                        validation=validation,
                        gpu=gpu_snapshot(),
                    )
                )
                with (a.output / "transactions.jsonl").open("a") as out:
                    out.write(json.dumps(results[-1]) + "\n")
                print(arm, repeat, round(elapsed, 2), flush=True)
        record["status"] = "passed"
        record["gpu_after"] = gpu_snapshot()
    except BaseException:
        record["status"] = "failed"
        record["failure"] = traceback.format_exc()
        raise
    finally:
        for g in graphs.values():
            g.close()
        for e in experiments.values():
            e.close()
        path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
