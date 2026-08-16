#!/usr/bin/env python3
"""Compare TP16 hierarchical and equal-quarter PCIe all-reduce latency.

The benchmark keeps both B12X runtimes and a decode-context-parallel attention
IPC pool resident. It validates BF16 results against an FP32 sum, captures both
implementations before timing, alternates their measured order, and reports
rank-maximum CUDA-event latency. The generated JSON receipt binds the result to
the clean source checkout, compile artifacts, PTXAS executable, physical GPU
identity and GPU operating mode used for the measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x._lib import compiler as cute_compiler
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from b12x.comm.pcie.pcie_hierarchical import PCIeHierarchicalAllReduce
from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce


WORLD_SIZE = 16
IMPLEMENTATIONS = ("hierarchical", "equal_quarter")
DEFAULT_SHAPES = (7_168, 14_336, 14_338, 28_672, 57_344)
SOURCE_FILES = (
    "benchmarks/benchmark_pcie_island_rs.py",
    "b12x/comm/pcie/_island_rs_cute.py",
    "b12x/comm/pcie/pcie_allreduce.py",
    "b12x/comm/pcie/pcie_hierarchical.py",
    "b12x/comm/pcie/pcie_island_rs.py",
)


def _command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(repository: Path) -> dict[str, object]:
    revision = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    tree = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"])
    status = _command_output(
        ["git", "-C", str(repository), "status", "--short", "--untracked-files=all"]
    )
    return {
        "repository": "local-inference-lab/b12x",
        "worktree": str(repository.resolve()),
        "revision": revision,
        "tree": tree,
        "status_porcelain": status.splitlines(),
        "worktree_state": "clean" if not status else "modified",
        "source_sha256": {
            relative: _sha256(repository / relative) for relative in SOURCE_FILES
        },
    }


def _validate_source(
    observed: dict[str, object],
    *,
    expected_revision: str | None,
    expected_tree: str | None,
) -> None:
    if observed["worktree_state"] != "clean":
        raise RuntimeError(
            "benchmark source checkout must be clean: "
            + ", ".join(observed["status_porcelain"])
        )
    if expected_revision is not None and observed["revision"] != expected_revision:
        raise RuntimeError(
            "observed source revision does not match --source-revision: "
            f"observed={observed['revision']}, expected={expected_revision}"
        )
    if expected_tree is not None and observed["tree"] != expected_tree:
        raise RuntimeError(
            "observed source tree does not match --source-tree: "
            f"observed={observed['tree']}, expected={expected_tree}"
        )


def _executable_metadata(name: str) -> dict[str, object]:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required executable is unavailable: {name}")
    path = Path(executable).resolve()
    completed = subprocess.run(
        [str(path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    return {"path": str(path), "sha256": _sha256(path), "version": version}


def _compile_artifacts() -> tuple[Path, dict[str, dict[str, object]]]:
    root = cute_compiler._cute_compile_cache_dir()
    artifacts: dict[str, dict[str, object]] = {}
    if not root.exists():
        return root, artifacts
    for manifest_path in sorted(root.glob("*/*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError(f"compile manifest is not an object: {manifest_path}")
        cache_key = str(manifest.get("cache_key", ""))
        if not cache_key or manifest_path.stem != cache_key:
            raise RuntimeError(
                f"compile manifest cache key is invalid: {manifest_path}"
            )
        object_path = manifest_path.with_suffix(".o")
        if not object_path.is_file():
            raise RuntimeError(f"compile object is missing: {object_path}")
        object_sha256 = _sha256(object_path)
        if object_sha256 != manifest.get("object_sha256"):
            raise RuntimeError(f"compile object hash differs: {object_path}")
        artifacts[cache_key] = {
            "cache_key": cache_key,
            "kernel_id": manifest.get("kernel_id"),
            "compile_spec_hash": manifest.get("compile_spec_hash"),
            "compile_kwargs_hash": manifest.get("compile_kwargs_hash"),
            "target_identity": manifest.get("target_identity"),
            "object_path": str(object_path.relative_to(root)),
            "object_bytes": object_path.stat().st_size,
            "object_sha256": object_sha256,
            "manifest_path": str(manifest_path.relative_to(root)),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_evidence_sha256": manifest.get("artifact_evidence_sha256"),
            "toolchain": manifest.get("toolchain"),
            "compile_options": manifest.get("compile_options"),
            "compile_environment": manifest.get("compile_environment"),
            "launch_metadata": manifest.get("launch_metadata"),
        }
    return root, artifacts


def _canonical_pci_bus_id(value: str) -> str:
    domain, bus, device = value.strip().lower().split(":")
    return f"{int(domain, 16):08x}:{int(bus, 16):02x}:{device}"


def _pci_bus_number(value: str) -> int:
    return int(_canonical_pci_bus_id(value).split(":")[1], 16)


def _physical_gpu_snapshot(
    expected_bus_ids: tuple[int, ...],
) -> list[dict[str, object]]:
    query = (
        "index,uuid,pci.bus_id,name,pstate,clocks.current.sm,"
        "clocks.current.memory,power.limit,compute_mode,"
        "clocks_throttle_reasons.active,driver_version"
    )
    lines = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    records: dict[int, list[str]] = {}
    for row in csv.reader(lines):
        fields = [field.strip() for field in row]
        if len(fields) != 11:
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        bus_number = _pci_bus_number(fields[2])
        if bus_number in records:
            raise RuntimeError(
                "nvidia-smi reported an ambiguous PCI bus number: "
                f"bus={bus_number:#x}, first_uuid={records[bus_number][1]}, "
                f"second_uuid={fields[1]}"
            )
        records[bus_number] = fields

    snapshot: list[dict[str, object]] = []
    for logical_index, bus_id in enumerate(expected_bus_ids):
        properties = torch.cuda.get_device_properties(logical_index)
        observed_bus = int(properties.pci_bus_id)
        if observed_bus != bus_id:
            raise RuntimeError(
                "visible GPU rank order differs from the declared topology: "
                f"logical_index={logical_index}, observed_bus={observed_bus:#x}, "
                f"expected_bus={bus_id:#x}"
            )
        if bus_id not in records:
            raise RuntimeError(f"nvidia-smi omitted PCI bus {bus_id:#x}")
        fields = records[bus_id]
        snapshot.append(
            {
                "logical_cuda_index": logical_index,
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": _canonical_pci_bus_id(fields[2]),
                "torch_device_name": properties.name,
                "nvidia_smi_device_name": fields[3],
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "pstate": fields[4],
                "sm_clock_mhz": int(fields[5]),
                "memory_clock_mhz": int(fields[6]),
                "power_limit_watts": float(fields[7]),
                "compute_mode": fields[8],
                "active_throttle_mask": fields[9],
                "driver_version": fields[10],
            }
        )
    return snapshot


def _gpu_mode_checks(
    before: list[dict[str, object]],
    after: list[dict[str, object]],
    *,
    required_throttle_mask: int,
) -> dict[str, bool]:
    identity_fields = ("logical_cuda_index", "physical_index", "uuid", "pci_bus_id")
    return {
        "physical_gpu_identity_stable": all(
            all(left[field] == right[field] for field in identity_fields)
            for left, right in zip(before, after, strict=True)
        ),
        "physical_gpus_in_p1": all(
            record["pstate"] == "P1" for record in (*before, *after)
        ),
        "physical_gpus_in_default_compute_mode": all(
            record["compute_mode"] == "Default" for record in (*before, *after)
        ),
        "physical_gpu_throttle_masks_match_policy": all(
            int(str(record["active_throttle_mask"]), 0) == required_throttle_mask
            for record in (*before, *after)
        ),
        "physical_gpu_memory_clocks_stable": all(
            left["memory_clock_mhz"] == right["memory_clock_mhz"]
            for left, right in zip(before, after, strict=True)
        ),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_islands(value: str) -> tuple[tuple[int, ...], ...]:
    islands = tuple(
        tuple(int(bus.strip(), 0) for bus in island.split(","))
        for island in value.split("|")
    )
    if len(islands) != 4 or any(len(island) != 4 for island in islands):
        raise ValueError(
            "expected four rank-ordered groups of four hexadecimal PCI bus IDs"
        )
    flattened = tuple(bus for island in islands for bus in island)
    if len(set(flattened)) != WORLD_SIZE:
        raise ValueError("each PCI bus ID must appear exactly once")
    return islands


def _reference(inp: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    result = torch.zeros_like(inp, dtype=torch.float32)
    for value in gathered:
        result.add_(value.float())
    return result


def _validate_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
    inp: torch.Tensor,
    input_before: torch.Tensor,
) -> dict[str, float | bool]:
    input_immutable = bool(torch.equal(inp, input_before))
    rank_zero = actual.clone()
    dist.broadcast(rank_zero, src=0)
    replicated = bool(torch.equal(actual, rank_zero))
    max_abs_error = (actual.float() - expected).abs().max()
    dist.all_reduce(max_abs_error, op=dist.ReduceOp.MAX)
    close = bool(torch.allclose(actual.float(), expected, rtol=0.02, atol=0.125))
    status = torch.tensor(
        int(input_immutable and replicated and close),
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise AssertionError(
            "all-reduce correctness failed: "
            f"input_immutable={input_immutable}, replicated={replicated}, "
            f"within_tolerance={close}, max_abs_error={max_abs_error.item()}"
        )
    return {
        "input_immutable": input_immutable,
        "max_abs_error_against_fp32": float(max_abs_error.item()),
        "replicated_output_bit_exact": replicated,
        "rtol": 0.02,
        "atol": 0.125,
        "within_tolerance": close,
    }


def _capture_graph(
    implementation: object,
    inp: torch.Tensor,
    out: torch.Tensor,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    with (
        implementation.capture(stream=stream, channel_id="benchmark"),
        torch.cuda.graph(graph, stream=stream),
    ):
        implementation.all_reduce(
            inp,
            out=out,
            stream=stream,
            channel_id="benchmark",
        )
    torch.cuda.synchronize(device)
    return graph


def _rank_max_graph_latency(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    iterations: int,
) -> float:
    dist.barrier(device_ids=[device.index])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    rank_max = torch.tensor(
        start.elapsed_time(end) * 1_000 / iterations,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    return float(rank_max.item())


def _balanced_rank_max_timings(
    graphs: dict[str, torch.cuda.CUDAGraph],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    first, second = IMPLEMENTATIONS
    for index in range(warmup):
        order = (first, second) if index % 2 == 0 else (second, first)
        for name in order:
            graphs[name].replay()
    torch.cuda.synchronize(device)

    timings = {name: [] for name in IMPLEMENTATIONS}
    measured_order: list[list[str]] = []
    for index in range(samples):
        order = (first, second) if index % 2 == 0 else (second, first)
        measured_order.append(list(order))
        for name in order:
            timings[name].append(
                _rank_max_graph_latency(
                    graphs[name],
                    device,
                    iterations=iterations,
                )
            )
    return timings, measured_order


def _artifact_keys_after_barrier(rank: int) -> set[str]:
    dist.barrier()
    keys = set(_compile_artifacts()[1]) if rank == 0 else set()
    dist.barrier()
    return keys


def _precondition_gpu_mode(device: torch.device) -> None:
    """Put every measured GPU under load before an operating-mode snapshot."""

    torch.cuda._sleep(50_000_000)
    torch.cuda.synchronize(device)


def _gpu_mode_snapshot_under_load(
    rank: int,
    device: torch.device,
    expected_bus_ids: tuple[int, ...],
) -> list[dict[str, object]]:
    """Read physical GPU state while every rank continuously submits GPU work."""

    ready = threading.Event()
    stop = threading.Event()
    load_errors: list[Exception] = []

    def sustain_load() -> None:
        try:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream(device=device)
            first_submission = True
            while not stop.is_set():
                with torch.cuda.stream(stream):
                    for _ in range(4):
                        torch.cuda._sleep(50_000_000)
                if first_submission:
                    ready.set()
                    first_submission = False
                stream.synchronize()
        except Exception as error:
            load_errors.append(error)
            ready.set()

    load_thread = threading.Thread(
        target=sustain_load,
        name=f"gpu-mode-load-rank-{rank}",
        daemon=True,
    )
    load_thread.start()
    if not ready.wait(timeout=30):
        stop.set()
        load_thread.join(timeout=30)
        raise RuntimeError(f"rank {rank} did not start GPU snapshot load")
    if load_errors:
        raise RuntimeError(f"rank {rank} GPU snapshot load failed") from load_errors[0]

    try:
        dist.barrier()
        snapshot = _physical_gpu_snapshot(expected_bus_ids) if rank == 0 else []
        dist.barrier()
    finally:
        stop.set()
        load_thread.join(timeout=30)
        if load_thread.is_alive():
            raise RuntimeError(f"rank {rank} GPU snapshot load did not stop")
        if load_errors:
            raise RuntimeError(
                f"rank {rank} GPU snapshot load failed"
            ) from load_errors[0]
    return snapshot


def _worker(
    rank: int,
    port: int,
    shapes: tuple[int, ...],
    warmup: int,
    iterations: int,
    samples: int,
    expected_islands: tuple[tuple[int, ...], ...],
    measurement_path: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
        device_id=device,
    )

    bootstrap = torch.zeros(max(shapes), dtype=torch.bfloat16, device=device)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize(device)

    dcp_pool = None
    runtime = None
    hierarchical = None
    equal_quarter = None
    try:
        initial_keys = _artifact_keys_after_barrier(rank)
        dcp_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=1,
            total_heads=96,
            head_dim=512,
            query_head_dim=576,
        )
        dcp_keys = _artifact_keys_after_barrier(rank)
        hierarchical = PCIeHierarchicalAllReduce(
            exchange_group=dist.group.WORLD,
            device=device,
            max_elements=max(shapes),
        )
        hierarchical_keys = _artifact_keys_after_barrier(rank)
        equal_quarter = PCIeIslandRSAllReduce(
            exchange_group=dist.group.WORLD,
            device=device,
            max_elements=max(shapes),
        )
        equal_quarter_keys = _artifact_keys_after_barrier(rank)
        runtime = PCIeAllReduce(
            hierarchical,
            "hierarchical",
            equal_quarter,
            algorithm_override="island_rs",
        )
        hierarchical = None
        equal_quarter = None
        implementations = {
            "hierarchical": runtime._runtime,
            "equal_quarter": runtime._island_rs,
        }

        _precondition_gpu_mode(device)
        gpu_before_timing = _gpu_mode_snapshot_under_load(
            rank,
            device,
            tuple(bus for island in expected_islands for bus in island),
        )

        results: list[dict[str, object]] = []
        for shape_index, elements in enumerate(shapes):
            generator = torch.Generator(device=device).manual_seed(
                0x15_1A_4D + rank * 17 + elements
            )
            inp = torch.randn(
                elements,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            input_before = inp.clone()
            expected = _reference(inp, WORLD_SIZE)
            outputs: dict[str, torch.Tensor] = {}
            graphs: dict[str, torch.cuda.CUDAGraph] = {}
            correctness: dict[str, dict[str, object]] = {}

            for name in IMPLEMENTATIONS:
                implementation = implementations[name]
                out = torch.empty_like(inp)
                implementation.all_reduce(inp, out=out)
                torch.cuda.synchronize(device)
                eager = _validate_result(out, expected, inp, input_before)
                graph = _capture_graph(implementation, inp, out, device)
                out.zero_()
                graph.replay()
                torch.cuda.synchronize(device)
                captured = _validate_result(out, expected, inp, input_before)
                outputs[name] = out
                graphs[name] = graph
                correctness[name] = {"eager": eager, "capture": captured}

            first_timed_replay_order = (
                IMPLEMENTATIONS
                if shape_index % 2 == 0
                else tuple(reversed(IMPLEMENTATIONS))
            )
            first_timed_replay_samples: dict[str, list[float]] = {
                name: [] for name in IMPLEMENTATIONS
            }
            for name in first_timed_replay_order:
                first_timed_replay_samples[name].append(
                    _rank_max_graph_latency(graphs[name], device, iterations=1)
                )

            warm_samples, measured_order = _balanced_rank_max_timings(
                graphs,
                device,
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            )
            for name in IMPLEMENTATIONS:
                graphs[name].replay()
                torch.cuda.synchronize(device)
                correctness[name]["post_measurement"] = _validate_result(
                    outputs[name], expected, inp, input_before
                )

            arms: dict[str, dict[str, object]] = {}
            for name in IMPLEMENTATIONS:
                raw = warm_samples[name]
                arms[name] = {
                    "correctness": correctness[name],
                    "input_data_ptr": inp.data_ptr(),
                    "output_data_ptr": outputs[name].data_ptr(),
                    "first_timed_replay_rank_max_microseconds": (
                        first_timed_replay_samples[name]
                    ),
                    "warm_rank_max_microseconds": raw,
                    "median_warm_rank_max_microseconds": float(statistics.median(raw)),
                    "minimum_warm_rank_max_microseconds": min(raw),
                    "maximum_warm_rank_max_microseconds": max(raw),
                }

            hierarchical_us = float(
                arms["hierarchical"]["median_warm_rank_max_microseconds"]
            )
            equal_quarter_us = float(
                arms["equal_quarter"]["median_warm_rank_max_microseconds"]
            )
            ratio = hierarchical_us / equal_quarter_us
            size_routed_dispatch = (
                "equal_quarter" if runtime._use_island_rs(inp) else "hierarchical"
            )
            results.append(
                {
                    "size_routed_dispatch": size_routed_dispatch,
                    "elements": elements,
                    "bytes": elements * torch.bfloat16.itemsize,
                    "first_timed_replay_measurement_order": list(
                        first_timed_replay_order
                    ),
                    "warm_sample_measurement_order": measured_order,
                    "implementations": arms,
                    "hierarchical_latency_divided_by_equal_quarter_latency": ratio,
                    "ratio_direction": (
                        "hierarchical median divided by equal-quarter median; "
                        "values above one mean equal-quarter is faster"
                    ),
                    "lower_median_latency": (
                        "equal_quarter" if ratio > 1.0 else "hierarchical"
                    ),
                }
            )

            del graphs, outputs, inp, input_before, expected
            torch.cuda.synchronize(device)

        _precondition_gpu_mode(device)
        gpu_after_timing = _gpu_mode_snapshot_under_load(
            rank,
            device,
            tuple(bus for island in expected_islands for bus in island),
        )
        if rank == 0:
            artifact_groups = {
                "initial": sorted(initial_keys),
                "dcp_ipc_pool": sorted(dcp_keys - initial_keys),
                "hierarchical": sorted(hierarchical_keys - dcp_keys),
                "equal_quarter": sorted(equal_quarter_keys - hierarchical_keys),
            }
            measurement = {
                "artifact_groups": artifact_groups,
                "physical_gpus": {
                    "before_timing": gpu_before_timing,
                    "after_timing": gpu_after_timing,
                },
                "rank_ordered_pci_bus_islands": expected_islands,
                "results": results,
            }
            Path(measurement_path).write_text(
                json.dumps(measurement, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        if runtime is not None:
            runtime.close()
        else:
            if equal_quarter is not None:
                equal_quarter.close()
            if hierarchical is not None:
                hierarchical.close()
        if dcp_pool is not None:
            dcp_pool.close()
        dist.destroy_process_group()


def _all_correctness_passed(results: list[dict[str, object]]) -> bool:
    for result in results:
        implementations = result["implementations"]
        for arm in implementations.values():
            for stage in arm["correctness"].values():
                if not (
                    stage["input_immutable"]
                    and stage["replicated_output_bit_exact"]
                    and stage["within_tolerance"]
                ):
                    return False
    return True


def _balanced_orders_recorded(results: list[dict[str, object]], samples: int) -> bool:
    for result in results:
        orders = result["warm_sample_measurement_order"]
        if len(orders) != samples:
            return False
        first_positions = [order[0] for order in orders]
        if any(sorted(order) != sorted(IMPLEMENTATIONS) for order in orders):
            return False
        if any(first_positions.count(name) != samples // 2 for name in IMPLEMENTATIONS):
            return False
    return True


def _captured_graph_replays_validated(results: list[dict[str, object]]) -> bool:
    return all(
        set(result["implementations"]) == set(IMPLEMENTATIONS)
        and all(
            "capture" in arm["correctness"]
            for arm in result["implementations"].values()
        )
        for result in results
    )


def _render_report(receipt: dict[str, object], receipt_path: Path) -> str:
    source = receipt["source"]
    runtime = receipt["runtime"]
    rows = []
    for result in receipt["results"]:
        arms = result["implementations"]
        rows.append(
            "| {elements:,} | {dispatch} | {hierarchical:.3f} | "
            "{equal_quarter:.3f} | {ratio:.3f}× |".format(
                elements=result["elements"],
                dispatch=result["size_routed_dispatch"],
                hierarchical=arms["hierarchical"]["median_warm_rank_max_microseconds"],
                equal_quarter=arms["equal_quarter"][
                    "median_warm_rank_max_microseconds"
                ],
                ratio=result["hierarchical_latency_divided_by_equal_quarter_latency"],
            )
        )
    command = receipt["command"]["shell"]
    checks = receipt["qualification_checks"]
    failed = [name for name, passed in checks.items() if not passed]
    failed_text = "none" if not failed else ", ".join(failed)
    invariants = receipt["enforced_invariants"]
    invariant_names = ", ".join(invariants)
    source_state = (
        "clean and unchanged"
        if invariants["source_checkout_clean_before_timing"]
        and invariants["source_checkout_clean_after_timing"]
        and checks["source_checkout_unchanged_during_timing"]
        else "not clean and unchanged; see failed qualification checks"
    )
    return f"""# TP16 PCIe equal-quarter all-reduce qualification

Status: **{receipt["status"]}**.

## Purpose

This record compares the B12X hierarchical and equal-quarter BF16 all-reduce
implementations on one TP16 PCIe topology. The comparison measures rank-maximum
CUDA-graph replay latency while a decode-context-parallel attention IPC pool
with 96 query heads, 512-dimensional latent heads, and 576-dimensional query
heads is resident. It does not measure end-to-end model throughput.

## Conditions

- Source repository: `{source["repository"]}`
- Source revision: `{source["revision"]}`
- Source tree: `{source["tree"]}`
- Measured worktree: `{source["worktree"]}`
- Worktree state before and after timing: {source_state}
- Container image: `{receipt["container_image"]}`
- CUDA runtime: `{runtime["cuda_runtime"]}`
- PyTorch: `{runtime["pytorch"]}`
- Driver: `{runtime["driver_version"]}`
- Physical GPUs: {runtime["world_size"]} × `{runtime["nvidia_smi_device_name"]}`
- Required GPU mode: P1, Default compute mode, active throttle mask
  `{runtime["required_active_throttle_mask"]}`
- Warm measurement: {runtime["warmup_graph_replay_pairs"]} alternating warmup
  pairs, {runtime["iterations_per_warm_sample"]} graph replays per sample,
  {runtime["warm_samples_per_implementation"]} samples per implementation
- Receipt: `{receipt_path}`

Both implementations use the same source revision. Each captured CUDA graph is
replayed and correctness-validated before timed samples. Warm samples alternate
AB and BA order with equal position counts. The first timed replay follows the
correctness replay and is not a cold-start measurement. The receipt records
every first-timed-replay and warm rank-maximum sample, the per-sample order,
source hashes, compile manifests and objects, PTXAS identity, physical GPU
UUIDs, clocks, modes, and correctness results.

## Results

| BF16 elements | Size-routed dispatch | Hierarchical µs | Equal-quarter µs | Hierarchical/equal-quarter |
| ---: | :--- | ---: | ---: | ---: |
{chr(10).join(rows)}

The ratio is hierarchical median latency divided by equal-quarter median
latency. Values above one mean equal-quarter is faster.

## Enforced invariants and qualification checks

Every eager result, captured result, and post-measurement result preserves the
input, is bit-identical across all 16 ranks, and matches the FP32 accumulation
reference at `rtol=0.02` and `atol=0.125`. Receipt construction aborts instead
of writing an artifact when any enforced invariant fails. Enforced invariants:
{invariant_names}. Reportable qualification checks that failed: {failed_text}.

## Reproduction

```bash
{command}
```

The command requires exactly 16 visible GPUs in the PCI bus order declared by
`--expected-pci-bus-islands`, an empty `B12X_COMPILE_CACHE_DIR`, and a clean
checkout at the recorded source revision.

## Limitations

Status applies only to the recorded TP16 topology, source revision, container,
driver, GPU mode, tensor shapes, BF16 data type, and CUDA-graph execution. The
receipt does not qualify other topology orders, GPU modes, message sizes,
dtypes, or end-to-end serving workloads.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--expected-pci-bus-islands", required=True)
    parser.add_argument("--shapes", type=int, nargs="+", default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--required-active-throttle-mask",
        type=lambda value: int(value, 0),
        default=0,
    )
    parser.add_argument(
        "--allow-populated-compile-cache",
        action="store_true",
        help="run without exact per-runtime compile-artifact attribution",
    )
    args = parser.parse_args()

    if min(args.shapes) <= 0 or any(elements % 2 for elements in args.shapes):
        parser.error("shapes must be positive even BF16 element counts")
    if min(args.warmup, args.iterations, args.samples) <= 0:
        parser.error("warmup, iterations and samples must be positive")
    if args.samples % 2:
        parser.error("samples must be even so both arms occupy each order equally")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            f"benchmark requires exactly {WORLD_SIZE} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )

    repository = Path(__file__).resolve().parents[1]
    source_before = _source_metadata(repository)
    _validate_source(
        source_before,
        expected_revision=args.source_revision,
        expected_tree=args.source_tree,
    )
    expected_islands = _parse_islands(args.expected_pci_bus_islands)
    expected_bus_ids = tuple(bus for island in expected_islands for bus in island)

    compile_cache_root, initial_artifacts = _compile_artifacts()
    if initial_artifacts and not args.allow_populated_compile_cache:
        raise RuntimeError(
            "B12X_COMPILE_CACHE_DIR must be empty for exact runtime artifact "
            f"attribution, got {len(initial_artifacts)} objects under "
            f"{compile_cache_root}"
        )

    ptxas = _executable_metadata("ptxas")
    actual_bus_ids = tuple(
        int(torch.cuda.get_device_properties(index).pci_bus_id)
        for index in range(WORLD_SIZE)
    )
    if actual_bus_ids != expected_bus_ids:
        raise RuntimeError(
            "visible GPU rank order differs from --expected-pci-bus-islands: "
            f"observed={actual_bus_ids}, expected={expected_bus_ids}"
        )
    with tempfile.NamedTemporaryFile(
        prefix="b12x-tp16-pcie-allreduce-",
        suffix=".json",
        delete=False,
    ) as temporary:
        measurement_path = Path(temporary.name)
    measurement_path.unlink()
    try:
        mp.spawn(
            _worker,
            args=(
                _free_port(),
                tuple(args.shapes),
                args.warmup,
                args.iterations,
                args.samples,
                expected_islands,
                str(measurement_path),
            ),
            nprocs=WORLD_SIZE,
            join=True,
        )
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    finally:
        measurement_path.unlink(missing_ok=True)

    source_after = _source_metadata(repository)
    _validate_source(
        source_after,
        expected_revision=args.source_revision,
        expected_tree=args.source_tree,
    )
    compile_cache_root, final_artifacts = _compile_artifacts()
    artifact_groups = measurement["artifact_groups"]
    results = measurement["results"]
    gpu_before = measurement["physical_gpus"]["before_timing"]
    gpu_after = measurement["physical_gpus"]["after_timing"]
    gpu_checks = _gpu_mode_checks(
        gpu_before,
        gpu_after,
        required_throttle_mask=args.required_active_throttle_mask,
    )
    enforced_invariants = {
        "source_checkout_clean_before_timing": (
            source_before["worktree_state"] == "clean"
        ),
        "source_checkout_clean_after_timing": (
            source_after["worktree_state"] == "clean"
        ),
        "compile_artifact_object_hashes_match_manifests": all(
            artifact["object_sha256"]
            == json.loads(
                (compile_cache_root / artifact["manifest_path"]).read_text(
                    encoding="utf-8"
                )
            )["object_sha256"]
            for artifact in final_artifacts.values()
        ),
        "captured_graph_replays_validated_before_timing": (
            _captured_graph_replays_validated(results)
        ),
        "first_timed_replay_samples_recorded": all(
            len(arm["first_timed_replay_rank_max_microseconds"]) == 1
            for result in results
            for arm in result["implementations"].values()
        ),
        "raw_warm_samples_recorded": all(
            len(arm["warm_rank_max_microseconds"]) == args.samples
            for result in results
            for arm in result["implementations"].values()
        ),
        "correctness_passed": _all_correctness_passed(results),
    }
    violated_invariants = [
        name for name, passed in enforced_invariants.items() if not passed
    ]
    if violated_invariants:
        raise RuntimeError(
            "benchmark receipt invariant failed: " + ", ".join(violated_invariants)
        )
    qualification_checks = {
        "source_checkout_unchanged_during_timing": source_before == source_after,
        "compile_cache_initially_empty": not initial_artifacts,
        "hierarchical_runtime_bound_to_new_compile_artifacts": bool(
            artifact_groups["hierarchical"]
        ),
        "equal_quarter_runtime_bound_to_new_compile_artifacts": bool(
            artifact_groups["equal_quarter"]
        ),
        "all_runtime_artifact_keys_present": all(
            key in final_artifacts
            for group in artifact_groups.values()
            for key in group
        ),
        "balanced_warm_measurement_order_recorded": _balanced_orders_recorded(
            results, args.samples
        ),
        **gpu_checks,
    }
    status = "qualified" if all(qualification_checks.values()) else "unqualified"
    command_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    receipt = {
        "schema": "b12x.tp16-pcie-equal-quarter-benchmark.v4",
        "status": status,
        "artifact_kind": "B12X TP16 PCIe all-reduce benchmark receipt",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "enforced_invariants": enforced_invariants,
        "qualification_checks": qualification_checks,
        "comparison": {
            "kind": "within-revision all-reduce implementation comparison",
            "source_revisions": {
                name: source_before["revision"] for name in IMPLEMENTATIONS
            },
            "source_identities": {
                name: {
                    "revision": source_before["revision"],
                    "tree": source_before["tree"],
                    "worktree": source_before["worktree"],
                }
                for name in IMPLEMENTATIONS
            },
            "ratio_direction": (
                "hierarchical median divided by equal-quarter median; values "
                "above one mean equal-quarter is faster"
            ),
        },
        "command": {"argv": command_argv, "shell": shlex.join(command_argv)},
        "container_image": os.getenv("B12X_BENCHMARK_CONTAINER_IMAGE", "unavailable"),
        "source": source_before,
        "runtime": {
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
            "driver_version": gpu_before[0]["driver_version"],
            "world_size": WORLD_SIZE,
            "nvidia_smi_device_name": gpu_before[0]["nvidia_smi_device_name"],
            "torch_device_name": gpu_before[0]["torch_device_name"],
            "rank_ordered_pci_bus_islands": expected_islands,
            "dcp_ipc_pool_co_resident": True,
            "warmup_graph_replay_pairs": args.warmup,
            "iterations_per_warm_sample": args.iterations,
            "warm_samples_per_implementation": args.samples,
            "required_active_throttle_mask": (
                f"0x{args.required_active_throttle_mask:016x}"
            ),
        },
        "physical_gpus": {"before_timing": gpu_before, "after_timing": gpu_after},
        "cutlass_ptxas_artifact_map": {
            "compile_cache_root": str(compile_cache_root),
            "initial_cache_keys": sorted(initial_artifacts),
            "runtime_new_cache_keys": artifact_groups,
            "ptxas": ptxas,
            "compile_cache_telemetry": cute_compiler.compile_cache_info(),
            "artifacts": final_artifacts,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_render_report(receipt, args.output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
