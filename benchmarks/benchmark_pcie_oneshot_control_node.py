"""Immutable one-run worker for the PCIe oneshot control-node A/B harness.

Use ``run_pcie_oneshot_control_node_ab.py`` rather than comparing two ad-hoc
JSON files. That external controller invokes this exact file for both source
trees in an alternating AB/BA sequence and validates every recorded contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from cuda.bindings import runtime as cudart

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


SCHEMA_NAME = "sparkinfer.pcie_oneshot_control_node.run"
SCHEMA_VERSION = 1
DTYPE_NAME = "bfloat16"
TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "identity",
    "software",
    "hardware",
    "environment",
    "expected",
    "per_rank",
    "slowest_rank",
}
CONTRACT_KEYS = {"world_size", "numel", "dtype", "warmup", "iters"}
IDENTITY_KEYS = {
    "label",
    "variant",
    "run_index",
    "implementation_root",
    "implementation_git_sha",
    "implementation_git_dirty",
    "harness_sha256",
    "started_at_utc",
}
METRIC_KEYS = {"cold_us", "mean_us", "p50_us", "p95_us", "min_us", "max_us"}
SOFTWARE_KEYS = {
    "python",
    "platform",
    "torch",
    "torch_cuda",
    "cuda_runtime_version",
    "cuda_driver_version",
}
HARDWARE_KEYS = {
    "hostname",
    "devices",
    "nvidia_smi_identity",
    "nvidia_smi_sample_fields",
    "nvidia_smi_samples",
    "nvidia_smi_topology",
}
EXPECTED_KEYS = {"staged_graph_kernel_nodes", "registered_graph_kernel_nodes"}
DEVICE_KEYS = {
    "index",
    "name",
    "uuid",
    "compute_capability",
    "total_memory_bytes",
    "multi_processor_count",
}
NVIDIA_QUERY_KEYS = {"fields", "returncode", "rows", "stderr"}
NVIDIA_SAMPLE_KEYS = {"monotonic_ns", "query"}
NVIDIA_TOPOLOGY_KEYS = {"returncode", "stdout", "stderr"}
NVIDIA_IDENTITY_FIELDS = ("index", "uuid", "name", "driver_version")
NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "power.limit",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
)
EXPLICIT_ENV_KEYS = {
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_EXTENSIONS_DIR",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-graph-kernel-nodes", type=int, required=True)
    parser.add_argument("--numel", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sampler-interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def _run_command(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _git_identity(root: Path) -> tuple[str, bool]:
    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    dirty = bool(_run_command(["git", "status", "--porcelain=v1"], cwd=root))
    return sha, dirty


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_version(call: Callable[[], tuple[object, int]]) -> int:
    result, version = call()
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA version query failed: {result}")
    return int(version)


def _nvidia_smi_query(fields: tuple[str, ...]) -> dict[str, object]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "fields": list(fields),
        "returncode": result.returncode,
        "rows": [
            [value.strip() for value in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ],
        "stderr": result.stderr.strip(),
    }


def _nvidia_smi_topology() -> dict[str, object]:
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.rstrip(),
        "stderr": result.stderr.strip(),
    }


class _NvidiaSmiSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "query": _nvidia_smi_query(NVIDIA_SMI_FIELDS),
                }
            )
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(
            {
                "monotonic_ns": time.monotonic_ns(),
                "query": _nvidia_smi_query(NVIDIA_SMI_FIELDS),
            }
        )


def _environment_toggles() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in EXPLICIT_ENV_KEYS
        or key.startswith("SPARKINFER_")
        or key.startswith("NCCL_")
        or key.startswith("TORCH_NCCL_")
    }


def _graph_kernel_count(graph: torch.cuda.CUDAGraph) -> int:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes(size) failed: {result}")
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    if result != cudart.cudaError_t.cudaSuccess or returned_nodes != num_nodes:
        raise RuntimeError(
            f"cudaGraphGetNodes(data) failed: {result}, {returned_nodes=}, {num_nodes=}"
        )
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    count = 0
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphNodeGetType failed: {result}")
        count += node_type == kernel_type
    return count


def _measure(
    operation: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iters: int,
) -> tuple[float, list[float]]:
    cold_start = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    cold_start.record(stream)
    operation()
    cold_end.record(stream)
    stream.synchronize()
    cold_us = float(cold_start.elapsed_time(cold_end) * 1000)

    for _ in range(warmup):
        operation()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        operation()
        end.record(stream)
    stream.synchronize()
    samples = [
        float(start.elapsed_time(end) * 1000)
        for start, end in zip(starts, ends, strict=True)
    ]
    return cold_us, samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "mean_us": sum(ordered) / len(ordered),
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _gather_rank_metrics(
    eager_cold_us: float,
    eager_samples: list[float],
    graph_cold_us: float,
    graph_samples: list[float],
    graph_kernel_nodes: int,
    device: torch.device,
) -> list[dict[str, object]]:
    eager = _summary(eager_samples)
    graph = _summary(graph_samples)
    names = ("mean_us", "p50_us", "p95_us", "min_us", "max_us")
    local = torch.tensor(
        [
            eager_cold_us,
            *(eager[name] for name in names),
            graph_cold_us,
            *(graph[name] for name in names),
            float(graph_kernel_nodes),
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)

    result = []
    for rank, values_tensor in enumerate(gathered):
        values = values_tensor.cpu().tolist()
        result.append(
            {
                "rank": rank,
                "eager": {
                    "cold_us": values[0],
                    **dict(zip(names, values[1:6], strict=True)),
                },
                "graph": {
                    "cold_us": values[6],
                    **dict(zip(names, values[7:12], strict=True)),
                    "kernel_nodes": int(values[12]),
                },
            }
        )
    return result


def _slowest_rank_summary(per_rank: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode in ("eager", "graph"):
        mode_rows = [row[mode] for row in per_rank]
        result[mode] = {
            key: max(float(row[key]) for row in mode_rows) for key in METRIC_KEYS
        }
    result["graph"]["kernel_nodes"] = max(
        int(row["graph"]["kernel_nodes"]) for row in per_rank
    )
    return result


def _device_metadata() -> list[dict[str, object]]:
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "unavailable")),
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return devices


def _validate_record(record: dict[str, object], contract: dict[str, object]) -> None:
    if set(record) != TOP_LEVEL_KEYS:
        raise ValueError(
            f"benchmark schema keys differ: {set(record) ^ TOP_LEVEL_KEYS}"
        )
    if record["schema"] != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ValueError(f"unsupported benchmark schema: {record['schema']}")
    if set(record["contract"]) != CONTRACT_KEYS or record["contract"] != contract:
        raise ValueError(
            f"benchmark contract mismatch: {record['contract']} != {contract}"
        )
    if set(record["identity"]) != IDENTITY_KEYS:
        raise ValueError("benchmark identity schema mismatch")
    if set(record["software"]) != SOFTWARE_KEYS:
        raise ValueError("benchmark software schema mismatch")
    if set(record["hardware"]) != HARDWARE_KEYS:
        raise ValueError("benchmark hardware schema mismatch")
    if set(record["expected"]) != EXPECTED_KEYS:
        raise ValueError("benchmark expectation schema mismatch")
    if not isinstance(record["environment"], dict):
        raise ValueError("benchmark environment must be an object")
    if not isinstance(record["per_rank"], list) or len(record["per_rank"]) != int(
        contract["world_size"]
    ):
        raise ValueError("benchmark per-rank cardinality mismatch")
    hardware = record["hardware"]
    if not isinstance(hardware["devices"], list) or not hardware["devices"]:
        raise ValueError("benchmark device inventory must be a non-empty list")
    for device in hardware["devices"]:
        if set(device) != DEVICE_KEYS:
            raise ValueError("benchmark device schema mismatch")
    identity_query = hardware["nvidia_smi_identity"]
    if set(identity_query) != NVIDIA_QUERY_KEYS:
        raise ValueError("benchmark NVIDIA identity schema mismatch")
    if identity_query["fields"] != list(NVIDIA_IDENTITY_FIELDS):
        raise ValueError("benchmark NVIDIA identity field contract mismatch")
    if identity_query["returncode"] != 0:
        raise ValueError(f"benchmark NVIDIA identity query failed: {identity_query}")
    if len(identity_query["rows"]) != len(hardware["devices"]):
        raise ValueError("benchmark NVIDIA identity/device cardinality mismatch")
    if any(len(row) != len(NVIDIA_IDENTITY_FIELDS) for row in identity_query["rows"]):
        raise ValueError("benchmark NVIDIA identity row schema mismatch")
    if hardware["nvidia_smi_sample_fields"] != list(NVIDIA_SMI_FIELDS):
        raise ValueError("benchmark NVIDIA sample field contract mismatch")
    if (
        not isinstance(hardware["nvidia_smi_samples"], list)
        or not hardware["nvidia_smi_samples"]
    ):
        raise ValueError("benchmark NVIDIA samples must be a non-empty list")
    successful_samples = 0
    for sample in hardware["nvidia_smi_samples"]:
        if set(sample) != NVIDIA_SAMPLE_KEYS:
            raise ValueError("benchmark NVIDIA sample schema mismatch")
        if set(sample["query"]) != NVIDIA_QUERY_KEYS:
            raise ValueError("benchmark NVIDIA sample query schema mismatch")
        if sample["query"]["fields"] != list(NVIDIA_SMI_FIELDS):
            raise ValueError("benchmark NVIDIA sample query field mismatch")
        if sample["query"]["returncode"] == 0:
            successful_samples += 1
            if any(
                len(row) != len(NVIDIA_SMI_FIELDS) for row in sample["query"]["rows"]
            ):
                raise ValueError("benchmark NVIDIA sample row schema mismatch")
    if successful_samples == 0:
        raise ValueError("benchmark collected no successful NVIDIA telemetry samples")
    topology = hardware["nvidia_smi_topology"]
    if set(topology) != NVIDIA_TOPOLOGY_KEYS:
        raise ValueError("benchmark NVIDIA topology schema mismatch")
    if topology["returncode"] != 0 or not topology["stdout"]:
        raise ValueError(f"benchmark NVIDIA topology query failed: {topology}")
    if set(record["slowest_rank"]) != {"eager", "graph"}:
        raise ValueError("slowest-rank mode schema mismatch")
    if set(record["slowest_rank"]["eager"]) != METRIC_KEYS:
        raise ValueError("slowest eager metric schema mismatch")
    if set(record["slowest_rank"]["graph"]) != METRIC_KEYS | {"kernel_nodes"}:
        raise ValueError("slowest graph metric schema mismatch")
    for row in record["per_rank"]:
        if set(row) != {"rank", "eager", "graph"}:
            raise ValueError("per-rank schema mismatch")
        if set(row["eager"]) != METRIC_KEYS:
            raise ValueError("eager metric schema mismatch")
        if set(row["graph"]) != METRIC_KEYS | {"kernel_nodes"}:
            raise ValueError("graph metric schema mismatch")
    if {int(row["rank"]) for row in record["per_rank"]} != set(
        range(int(contract["world_size"]))
    ):
        raise ValueError("per-rank identities do not match world size")


def main() -> None:
    args = _parse_args()
    if (
        args.iters <= 0
        or args.warmup < 0
        or args.numel <= 0
        or args.run_index < 0
        or args.expected_graph_kernel_nodes <= 0
    ):
        raise ValueError(
            "numel/iters/run-index/node-count must be valid and warmup non-negative"
        )
    if args.sampler_interval <= 0:
        raise ValueError("sampler interval must be positive")

    implementation_root = args.implementation_root.resolve()
    git_sha, git_dirty = _git_identity(implementation_root)
    if git_sha != args.expected_git_sha:
        raise ValueError(f"implementation SHA {git_sha} != {args.expected_git_sha}")
    if git_dirty and not args.allow_dirty:
        raise ValueError(f"implementation worktree is dirty: {implementation_root}")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    contract = {
        "world_size": world_size,
        "numel": args.numel,
        "dtype": DTYPE_NAME,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    sampler = _NvidiaSmiSampler(args.sampler_interval) if rank == 0 else None
    if sampler is not None:
        sampler.start()

    pool = None
    try:
        nbytes = args.numel * torch.bfloat16.itemsize
        pool = PCIeOneshotAllReducePool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_input_bytes=nbytes,
            max_size=nbytes,
        )
        stream = torch.cuda.Stream(device=device)
        channel = pool.for_stream(stream)
        eager_inp = torch.full(
            (args.numel,),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        eager_out = torch.empty_like(eager_inp)

        def eager_operation() -> None:
            with torch.cuda.stream(stream):
                channel.all_reduce(eager_inp, out=eager_out)

        dist.barrier()
        eager_cold_us, eager_samples = _measure(
            eager_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        expected = world_size * (world_size + 1) // 2
        torch.testing.assert_close(
            eager_out,
            torch.full_like(eager_out, expected),
            rtol=0,
            atol=0,
        )

        graph_inp = torch.full_like(eager_inp, rank + 2)
        graph_out = torch.empty_like(graph_inp)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            pool.capture(stream) as graph_channel,
            torch.cuda.graph(
                graph,
                stream=stream,
            ),
        ):
            graph_channel.all_reduce(graph_inp, out=graph_out)
        graph_kernel_nodes = _graph_kernel_count(graph)
        if graph_kernel_nodes != args.expected_graph_kernel_nodes:
            raise AssertionError(
                f"graph has {graph_kernel_nodes} kernel nodes, expected "
                f"{args.expected_graph_kernel_nodes}"
            )

        def graph_operation() -> None:
            with torch.cuda.stream(stream):
                graph.replay()

        dist.barrier()
        graph_cold_us, graph_samples = _measure(
            graph_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        expected += world_size
        torch.testing.assert_close(
            graph_out,
            torch.full_like(graph_out, expected),
            rtol=0,
            atol=0,
        )

        per_rank = _gather_rank_metrics(
            eager_cold_us,
            eager_samples,
            graph_cold_us,
            graph_samples,
            graph_kernel_nodes,
            device,
        )
        observed_nodes = {int(row["graph"]["kernel_nodes"]) for row in per_rank}
        if observed_nodes != {args.expected_graph_kernel_nodes}:
            raise AssertionError(
                f"rank graph-node counts {observed_nodes} do not match "
                f"{args.expected_graph_kernel_nodes}"
            )
        torch.cuda.synchronize(device)
        dist.barrier()

        if sampler is not None:
            sampler.stop()
        if rank == 0:
            harness_path = Path(__file__).resolve()
            record = {
                "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
                "contract": contract,
                "identity": {
                    "label": args.label,
                    "variant": args.variant,
                    "run_index": args.run_index,
                    "implementation_root": str(implementation_root),
                    "implementation_git_sha": git_sha,
                    "implementation_git_dirty": git_dirty,
                    "harness_sha256": _sha256(harness_path),
                    "started_at_utc": started_at,
                },
                "software": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_runtime_version": _cuda_version(cudart.cudaRuntimeGetVersion),
                    "cuda_driver_version": _cuda_version(cudart.cudaDriverGetVersion),
                },
                "hardware": {
                    "hostname": socket.gethostname(),
                    "devices": _device_metadata(),
                    "nvidia_smi_identity": _nvidia_smi_query(NVIDIA_IDENTITY_FIELDS),
                    "nvidia_smi_sample_fields": list(NVIDIA_SMI_FIELDS),
                    "nvidia_smi_samples": sampler.samples,
                    "nvidia_smi_topology": _nvidia_smi_topology(),
                },
                "environment": _environment_toggles(),
                "expected": {
                    "staged_graph_kernel_nodes": args.expected_graph_kernel_nodes,
                    "registered_graph_kernel_nodes": 1,
                },
                "per_rank": per_rank,
                "slowest_rank": _slowest_rank_summary(per_rank),
            }
            _validate_record(record, contract)
            rendered = json.dumps(record, indent=2, sort_keys=True)
            print(rendered, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        if sampler is not None and sampler._thread.is_alive():
            sampler.stop()
        if pool is not None:
            pool.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
