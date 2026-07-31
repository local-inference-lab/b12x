"""Run repeated immutable AB/BA measurements with one external harness.

The measurement script lives outside both implementation worktrees and is
hashed into every record. Each pair reverses execution order (AB, then BA) to
reduce monotonic thermal/clock drift. Raw records are schema- and
contract-validated before the aggregate report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


RUN_SCHEMA = {"name": "sparkinfer.pcie_oneshot_control_node.run", "version": 1}
SUITE_SCHEMA = {"name": "sparkinfer.pcie_oneshot_control_node.ab_suite", "version": 1}
RUN_TOP_LEVEL_KEYS = {
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
METRICS = ("mean_us", "p50_us", "p95_us")
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
METRIC_KEYS = {"cold_us", "mean_us", "p50_us", "p95_us", "min_us", "max_us"}
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
NVIDIA_IDENTITY_FIELDS = ["index", "uuid", "name", "driver_version"]
NVIDIA_SMI_FIELDS = [
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
]
SUITE_TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "harness",
    "implementations",
    "sequence",
    "records",
    "paired_deltas",
    "aggregate",
}
HARNESS_KEYS = {
    "source_path",
    "frozen_path",
    "sha256",
    "identical_for_all_runs",
}
IMPLEMENTATION_KEYS = {"root", "git_sha"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-worktree", type=Path, required=True)
    parser.add_argument("--candidate-worktree", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--numel", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sampler-interval", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("benchmark_pcie_oneshot_control_node.py"),
    )
    return parser.parse_args()


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}"
        )
    return result.stdout.strip()


def _git_sha(worktree: Path) -> str:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    dirty = _run(["git", "status", "--porcelain=v1"], cwd=worktree)
    if dirty:
        raise ValueError(f"benchmark worktree is dirty: {worktree}\n{dirty}")
    return sha


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "world_size": args.world_size,
        "numel": args.numel,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iters": args.iters,
    }


def _validate_run(
    record: dict[str, object],
    *,
    contract: dict[str, object],
    variant: str,
    sha: str,
    run_index: int,
    harness_sha256: str,
    expected_nodes: int,
    implementation_root: Path,
) -> None:
    if set(record) != RUN_TOP_LEVEL_KEYS:
        raise ValueError(f"run schema keys differ: {set(record) ^ RUN_TOP_LEVEL_KEYS}")
    if record["schema"] != RUN_SCHEMA:
        raise ValueError(f"run schema mismatch: {record['schema']}")
    if set(record["contract"]) != CONTRACT_KEYS or record["contract"] != contract:
        raise ValueError(f"run contract mismatch: {record['contract']} != {contract}")
    identity = record["identity"]
    if set(identity) != IDENTITY_KEYS:
        raise ValueError("run identity schema mismatch")
    expected_identity = {
        "variant": variant,
        "run_index": run_index,
        "implementation_root": str(implementation_root),
        "implementation_git_sha": sha,
        "implementation_git_dirty": False,
        "harness_sha256": harness_sha256,
    }
    for key, expected in expected_identity.items():
        if identity[key] != expected:
            raise ValueError(f"identity {key}={identity[key]!r}, expected {expected!r}")
    if set(record["expected"]) != EXPECTED_KEYS or record["expected"] != {
        "staged_graph_kernel_nodes": expected_nodes,
        "registered_graph_kernel_nodes": 1,
    }:
        raise ValueError(f"node contract mismatch: {record['expected']}")
    if set(record["software"]) != SOFTWARE_KEYS:
        raise ValueError("run software schema mismatch")
    if set(record["hardware"]) != HARDWARE_KEYS:
        raise ValueError("run hardware schema mismatch")
    if not isinstance(record["environment"], dict):
        raise ValueError("run environment must be an object")
    if not isinstance(record["per_rank"], list) or len(record["per_rank"]) != int(
        contract["world_size"]
    ):
        raise ValueError("run per-rank cardinality mismatch")
    hardware = record["hardware"]
    if not isinstance(hardware["devices"], list) or not hardware["devices"]:
        raise ValueError("run device inventory must be a non-empty list")
    for device in hardware["devices"]:
        if set(device) != DEVICE_KEYS:
            raise ValueError("run device schema mismatch")
    identity_query = hardware["nvidia_smi_identity"]
    if set(identity_query) != NVIDIA_QUERY_KEYS:
        raise ValueError("run NVIDIA identity schema mismatch")
    if identity_query["fields"] != NVIDIA_IDENTITY_FIELDS:
        raise ValueError("run NVIDIA identity field contract mismatch")
    if identity_query["returncode"] != 0:
        raise ValueError(f"run NVIDIA identity query failed: {identity_query}")
    if len(identity_query["rows"]) != len(hardware["devices"]):
        raise ValueError("run NVIDIA identity/device cardinality mismatch")
    if any(len(row) != len(NVIDIA_IDENTITY_FIELDS) for row in identity_query["rows"]):
        raise ValueError("run NVIDIA identity row schema mismatch")
    if hardware["nvidia_smi_sample_fields"] != NVIDIA_SMI_FIELDS:
        raise ValueError("run NVIDIA sample field contract mismatch")
    if (
        not isinstance(hardware["nvidia_smi_samples"], list)
        or not hardware["nvidia_smi_samples"]
    ):
        raise ValueError("run NVIDIA samples must be a non-empty list")
    successful_samples = 0
    for sample in hardware["nvidia_smi_samples"]:
        if set(sample) != NVIDIA_SAMPLE_KEYS:
            raise ValueError("run NVIDIA sample schema mismatch")
        if set(sample["query"]) != NVIDIA_QUERY_KEYS:
            raise ValueError("run NVIDIA sample query schema mismatch")
        if sample["query"]["fields"] != NVIDIA_SMI_FIELDS:
            raise ValueError("run NVIDIA sample query field mismatch")
        if sample["query"]["returncode"] == 0:
            successful_samples += 1
            if any(
                len(row) != len(NVIDIA_SMI_FIELDS) for row in sample["query"]["rows"]
            ):
                raise ValueError("run NVIDIA sample row schema mismatch")
    if successful_samples == 0:
        raise ValueError("run collected no successful NVIDIA telemetry samples")
    topology = hardware["nvidia_smi_topology"]
    if set(topology) != NVIDIA_TOPOLOGY_KEYS:
        raise ValueError("run NVIDIA topology schema mismatch")
    if topology["returncode"] != 0 or not topology["stdout"]:
        raise ValueError(f"run NVIDIA topology query failed: {topology}")
    if set(record["slowest_rank"]) != {"eager", "graph"}:
        raise ValueError("run slowest-rank schema mismatch")
    if set(record["slowest_rank"]["eager"]) != METRIC_KEYS:
        raise ValueError("run eager metric schema mismatch")
    if set(record["slowest_rank"]["graph"]) != METRIC_KEYS | {"kernel_nodes"}:
        raise ValueError("run graph metric schema mismatch")
    for row in record["per_rank"]:
        if set(row) != {"rank", "eager", "graph"}:
            raise ValueError("run per-rank schema mismatch")
        if set(row["eager"]) != METRIC_KEYS:
            raise ValueError("run per-rank eager metric schema mismatch")
        if set(row["graph"]) != METRIC_KEYS | {"kernel_nodes"}:
            raise ValueError("run per-rank graph metric schema mismatch")
    if {int(row["rank"]) for row in record["per_rank"]} != set(
        range(int(contract["world_size"]))
    ):
        raise ValueError("run rank identities do not match world size")
    if {int(row["graph"]["kernel_nodes"]) for row in record["per_rank"]} != {
        expected_nodes
    }:
        raise ValueError("observed graph-node count differs across ranks")


def _outside_worktrees(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    resolved = path.resolve()
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            raise ValueError(f"{label} must be outside measured worktree {root}")
    return resolved


def _stable_environment(record: dict[str, object]) -> dict[str, object]:
    environment = dict(record["environment"])
    environment.pop("TORCH_EXTENSIONS_DIR", None)
    return environment


def _stable_hardware(record: dict[str, object]) -> dict[str, object]:
    hardware = record["hardware"]
    return {
        "hostname": hardware["hostname"],
        "devices": hardware["devices"],
        "nvidia_smi_identity": hardware["nvidia_smi_identity"],
        "nvidia_smi_sample_fields": hardware["nvidia_smi_sample_fields"],
        "nvidia_smi_topology": hardware["nvidia_smi_topology"],
    }


def _paired_deltas(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deltas = []
    for pair_index in range(len(records) // 2):
        pair = records[2 * pair_index : 2 * pair_index + 2]
        by_variant = {record["identity"]["variant"]: record for record in pair}
        if set(by_variant) != {"baseline", "candidate"}:
            raise ValueError(f"pair {pair_index} does not contain one A and one B")
        delta = {"pair_index": pair_index, "metrics": {}}
        for mode in ("eager", "graph"):
            delta["metrics"][mode] = {}
            for metric in METRICS:
                before = float(by_variant["baseline"]["slowest_rank"][mode][metric])
                after = float(by_variant["candidate"]["slowest_rank"][mode][metric])
                delta["metrics"][mode][metric] = {
                    "baseline_us": before,
                    "candidate_us": after,
                    "delta_us": after - before,
                    "delta_percent": (after / before - 1) * 100,
                }
        deltas.append(delta)
    return deltas


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for variant in ("baseline", "candidate"):
        variant_records = [
            record for record in records if record["identity"]["variant"] == variant
        ]
        result[variant] = {}
        for mode in ("eager", "graph"):
            result[variant][mode] = {}
            for metric in METRICS:
                values = [
                    float(record["slowest_rank"][mode][metric])
                    for record in variant_records
                ]
                result[variant][mode][metric] = {
                    "mean_us": statistics.mean(values),
                    "median_us": statistics.median(values),
                    "min_us": min(values),
                    "max_us": max(values),
                }
    return result


def main() -> None:
    args = _parse_args()
    if (
        args.pairs <= 0
        or args.world_size <= 0
        or args.numel <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.sampler_interval <= 0
    ):
        raise ValueError("invalid A/B workload contract")
    baseline_root = args.baseline_worktree.resolve()
    candidate_root = args.candidate_worktree.resolve()
    harness_source = args.harness.resolve()
    roots = (baseline_root, candidate_root)
    if baseline_root == candidate_root:
        raise ValueError("baseline and candidate worktrees must differ")
    output_dir = _outside_worktrees(args.output_dir, roots, "output directory")
    output = _outside_worktrees(args.output, roots, "suite output")
    if not harness_source.is_file():
        raise ValueError(f"harness does not exist: {harness_source}")
    baseline_sha = _git_sha(baseline_root)
    candidate_sha = _git_sha(candidate_root)
    if baseline_sha == candidate_sha:
        raise ValueError("baseline and candidate Git SHAs must differ")
    harness_sha256 = _sha256(harness_source)
    contract = _expected_contract(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    harness = output_dir / f"harness-{harness_sha256[:16]}.py"
    harness_bytes = harness_source.read_bytes()
    if harness.exists() and harness.read_bytes() != harness_bytes:
        raise ValueError(f"frozen harness collision at {harness}")
    if not harness.exists():
        harness.write_bytes(harness_bytes)
    if _sha256(harness) != harness_sha256:
        raise ValueError("frozen harness hash differs from source harness")

    variants = {
        "baseline": {
            "root": baseline_root,
            "sha": baseline_sha,
            "expected_nodes": 1,
        },
        "candidate": {
            "root": candidate_root,
            "sha": candidate_sha,
            "expected_nodes": 2,
        },
    }
    records = []
    sequence = []
    for pair_index in range(args.pairs):
        order = (
            ("baseline", "candidate")
            if pair_index % 2 == 0
            else ("candidate", "baseline")
        )
        for variant in order:
            run_index = len(sequence)
            sequence.append(variant)
            config = variants[variant]
            raw_path = output_dir / f"{run_index:02d}-{variant}.json"
            extension_dir = (
                output_dir / "torch-extensions" / f"{variant}-{config['sha'][:12]}"
            )
            extension_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(config["root"]) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            env["TORCH_EXTENSIONS_DIR"] = str(extension_dir)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.world_size}",
                str(harness),
                "--label",
                f"{variant}-{config['sha'][:12]}-{run_index}",
                "--variant",
                variant,
                "--run-index",
                str(run_index),
                "--implementation-root",
                str(config["root"]),
                "--expected-git-sha",
                config["sha"],
                "--expected-graph-kernel-nodes",
                str(config["expected_nodes"]),
                "--numel",
                str(args.numel),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--sampler-interval",
                str(args.sampler_interval),
                "--output",
                str(raw_path),
            ]
            print(
                f"[{run_index + 1}/{2 * args.pairs}] {variant} {config['sha'][:12]}",
                flush=True,
            )
            _run(command, cwd=config["root"], env=env)
            record = json.loads(raw_path.read_text(encoding="utf-8"))
            _validate_run(
                record,
                contract=contract,
                variant=variant,
                sha=config["sha"],
                run_index=run_index,
                harness_sha256=harness_sha256,
                expected_nodes=config["expected_nodes"],
                implementation_root=config["root"],
            )
            records.append(record)

    software_fingerprints = {
        json.dumps(record["software"], sort_keys=True) for record in records
    }
    hardware_fingerprints = {
        json.dumps(_stable_hardware(record), sort_keys=True) for record in records
    }
    environment_fingerprints = {
        json.dumps(_stable_environment(record), sort_keys=True) for record in records
    }
    if len(software_fingerprints) != 1:
        raise ValueError("software stack changed during alternating A/B suite")
    if len(hardware_fingerprints) != 1:
        raise ValueError(
            "hardware/driver/topology changed during alternating A/B suite"
        )
    if len(environment_fingerprints) != 1:
        raise ValueError("environment toggles changed during alternating A/B suite")

    suite = {
        "schema": SUITE_SCHEMA,
        "contract": contract,
        "harness": {
            "source_path": str(harness_source),
            "frozen_path": str(harness),
            "sha256": harness_sha256,
            "identical_for_all_runs": True,
        },
        "implementations": {
            "baseline": {"root": str(baseline_root), "git_sha": baseline_sha},
            "candidate": {"root": str(candidate_root), "git_sha": candidate_sha},
        },
        "sequence": sequence,
        "records": records,
        "paired_deltas": _paired_deltas(records),
        "aggregate": _aggregate(records),
    }
    if set(suite) != SUITE_TOP_LEVEL_KEYS:
        raise ValueError("suite top-level schema mismatch")
    if suite["schema"] != SUITE_SCHEMA or suite["contract"] != contract:
        raise ValueError("suite identity/contract mismatch")
    if set(suite["harness"]) != HARNESS_KEYS:
        raise ValueError("suite harness schema mismatch")
    if set(suite["implementations"]) != {"baseline", "candidate"}:
        raise ValueError("suite implementation variants mismatch")
    for implementation in suite["implementations"].values():
        if set(implementation) != IMPLEMENTATION_KEYS:
            raise ValueError("suite implementation schema mismatch")
    if len(suite["sequence"]) != args.pairs * 2:
        raise ValueError("suite sequence cardinality mismatch")
    if len(suite["records"]) != args.pairs * 2:
        raise ValueError("suite record cardinality mismatch")
    if len(suite["paired_deltas"]) != args.pairs:
        raise ValueError("suite paired-delta cardinality mismatch")
    rendered = json.dumps(suite, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
