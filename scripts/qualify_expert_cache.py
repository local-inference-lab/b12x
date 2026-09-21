#!/usr/bin/env python3
"""Run source-bound host, SM120, or serving acceptance using existing tests."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._sm103_source import source_identity
from scripts.qualify_sm103 import junit_counts
from b12x.testing.artifacts import sha256

HOST = [
    "tests/test_registry.py",
    "tests/architecture",
    "tests/preparation",
    *[
        f"tests/moe/test_{name}.py"
        for name in (
            "shared_residency",
            "residency_cache",
            "residency_epoch",
            "vllm_residency_epoch",
            "expert_cache_serving",
            "expert_cache_capacity",
            "routing_health",
            "prepared_expert_cache",
            "residency_updates",
        )
    ],
]
GPU = [
    "tests/moe/test_prepared_expert_cache.py",
    "tests/moe/test_routing_profile_gpu.py",
    "tests/moe/test_residency_epoch_gpu.py",
]
SANITIZER = [
    "tests/moe/test_prepared_expert_cache.py::test_prepared_native_cache_graph_matches_resident_reference[geometry0-dtype0]",
    "tests/moe/test_routing_profile_gpu.py::test_pending_health_close_reconstructs_without_retained_allocations",
]


def source_files():
    """Bind package, tests, harnesses and acceptance commands, including exports."""
    files = {}
    for folder in ("b12x", "tests", "scripts", "benchmarks/moe", "ci/expert_cache"):
        for path in sorted((ROOT / folder).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                files[str(path.relative_to(ROOT))] = sha256(path)
    return files


def require_complete(path, log):
    from benchmarks.moe.summarize_expert_cache_serving import summarize

    text = log.read_text()
    if any(
        word in text
        for word in (
            "Exception ignored in:",
            "did not exit",
            "SIGKILL",
            "force killing",
        )
    ):
        raise ValueError(f"unclean shutdown: {log}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not any(row["kind"] == "shutdown" for row in records):
        raise ValueError("receipt lacks explicit shutdown acknowledgement")
    return summarize(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("host", "gpu", "serving"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device-uuid")
    p.add_argument("--model", type=Path)
    p.add_argument("--profile", type=Path)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--calibration-prompts", type=Path)
    p.add_argument("--build-manifest", type=Path)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--workload", default="general")
    p.add_argument(
        "--all-resident-static",
        action="store_true",
        help="Qualify repeated all-resident static runs; no adaptive acceptance claim",
    )
    p.add_argument("--cache-gib", type=int, default=8)
    p.add_argument("--host-gib", type=int, default=40)
    p.add_argument("--kv-gib", type=int, default=2)
    p.add_argument("--sanitizer", type=Path)
    p.add_argument(
        "--sanitizer-tool", choices=("memcheck", "synccheck"), default="memcheck"
    )
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("acceptance output must be a new directory")
    if args.tier != "host" and not args.device_uuid:
        p.error("GPU execution requires an explicit physical GPU UUID")
    if args.tier == "serving" and (
        not all((args.model, args.profile, args.prompts, args.build_manifest))
        or args.pairs < 1
    ):
        p.error(
            "serving requires model, profile, prompts, build manifest and positive pairs"
        )
    if args.tier == "gpu" and not args.model:
        p.error("GPU acceptance requires the supplied NVFP4 checkpoint")
    if args.sanitizer and args.tier != "gpu":
        p.error("sanitizer is a separate GPU diagnostic")
    if args.all_resident_static and args.tier != "serving":
        p.error("all-resident static is a serving qualification scope")
    args.output.mkdir(parents=True)
    out = args.output.resolve()
    receipt = dict(
        schema="b12x-expert-cache-acceptance/v1",
        status="running",
        tier=args.tier,
        serving_scope=(
            "all-resident-static" if args.all_resident_static else "static-adaptive"
        )
        if args.tier == "serving"
        else None,
        command=sys.argv if argv is None else argv,
        source=source_identity(ROOT),
        source_files=source_files(),
        results=[],
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    )
    manifest = out / "acceptance.json"

    def save():
        manifest.write_text(json.dumps(receipt, indent=2) + "\n")

    def execute(name, command):
        log = out / f"{name}.log"
        row = dict(name=name, command=command)
        receipt["results"].append(row)
        save()
        begin = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        row.update(
            returncode=result.returncode,
            wall_s=time.monotonic() - begin,
            log_sha256=sha256(log),
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {log}")
        return row, log

    save()
    try:
        if args.tier != "host":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_uuid
            import torch

            if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
                12,
                0,
            ):
                raise RuntimeError(
                    "physical SM120 is required; SM103 uses qualify_sm103.py"
                )
            receipt["gpu_uuid"] = args.device_uuid
            receipt["cuda"] = torch.version.cuda
            for label, command in (
                ("gpu-before", ["nvidia-smi", "-q", "-i", args.device_uuid]),
                ("topology", ["nvidia-smi", "topo", "-m"]),
                ("numa", ["lscpu"]),
            ):
                execute(label, command)
            os.environ["B12X_TEST_NVFP4_CHECKPOINT"] = str(args.model.resolve())
        if args.tier in ("host", "gpu"):
            command = [
                sys.executable,
                "-m",
                "pytest",
                *(
                    HOST
                    if args.tier == "host"
                    else SANITIZER
                    if args.sanitizer
                    else GPU
                ),
                "-q",
                "-ra",
                f"--junitxml={out / 'tests.xml'}",
            ]
            if args.sanitizer:
                command = [
                    str(args.sanitizer.resolve()),
                    "--tool",
                    args.sanitizer_tool,
                    "--error-exitcode",
                    "99",
                    *command,
                ]
            row, _ = execute("tests", command)
            row["tests"] = junit_counts(out / "tests.xml")
            if (
                not row["tests"]["tests"]
                or row["tests"]["failures"]
                or row["tests"]["errors"]
                or (args.tier == "gpu" and row["tests"]["skipped"])
            ):
                raise RuntimeError(
                    "required tests failed, were absent, or GPU coverage skipped"
                )
            # Host skips are retained individually in JUnit; they do not qualify GPU work.
        else:
            from b12x.testing.artifacts import verify_from_file

            receipt["companion"] = json.loads(args.build_manifest.read_text())
            receipt["installed"] = verify_from_file(args.build_manifest)
            os.environ["B12X_ACCEPTANCE_BUILD_MANIFEST"] = str(
                args.build_manifest.resolve()
            )
            os.environ["B12X_SERVING_SOURCE_RECEIPT"] = str(manifest)
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            os.environ["B12X_AUTOTUNE"] = "0"
            os.environ["B12X_WEIGHTS_COMPILE_WORKERS"] = "0"
            profile = args.profile.resolve()
            if args.calibration_prompts:
                evaluation = {
                    json.loads(v)["text"] for v in args.prompts.read_text().splitlines()
                }
                calibration = {
                    json.loads(v)["text"]
                    for v in args.calibration_prompts.read_text().splitlines()
                }
                if evaluation & calibration or profile.exists():
                    raise ValueError(
                        "calibration requires disjoint prompts and a new profile path"
                    )
            elif not profile.is_file():
                raise ValueError("supply a profile or separate calibration prompts")

            def serve(name, mode, prompts, tokens):
                path = out / f"{name}.jsonl"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.moe.expert_cache_serving",
                    "--model",
                    str(args.model.resolve()),
                    "--profile",
                    str(profile),
                    "--prompts",
                    str(prompts.resolve()),
                    "--output",
                    str(path),
                    "--resources",
                    str(out / f"{name}-resources.jsonl"),
                    "--mode",
                    mode,
                    "--workload",
                    args.workload,
                    "--tokens",
                    str(tokens),
                    "--concurrency",
                    str(args.concurrency),
                    "--admission",
                    "together",
                    "--cache-gib",
                    str(args.cache_gib),
                    "--host-gib",
                    str(args.host_gib),
                    "--kv-gib",
                    str(args.kv_gib),
                ]
                if mode == "adaptive":
                    command += [
                        "--control",
                        "health",
                        "--epoch-tokens",
                        "16",
                        "--cold-threshold",
                        "0.15",
                        "--epoch-pairs",
                        "32",
                        "--epoch-mib",
                        "128",
                        "--history-depth",
                        "0",
                    ]
                row, log = execute(name, command)
                row["summary"] = require_complete(path, log)
                resources = [
                    json.loads(s)
                    for s in (out / f"{name}-resources.jsonl").read_text().splitlines()
                ]
                closed = [r for r in resources if r["stage"] == "after_worker_shutdown"]
                if len(closed) != 1 or any(
                    closed[0][k]
                    for k in (
                        "mapped_bytes",
                        "cpu_source_bytes",
                        "graph_owners",
                        "pending_health",
                    )
                ):
                    raise RuntimeError(
                        "worker retained cache owners after explicit shutdown"
                    )
                if not any(r.get("artifacts", {}).get("native") for r in resources):
                    raise RuntimeError("worker did not attest loaded native libraries")
                save()
                return row["summary"]

            if args.calibration_prompts:
                serve("calibration", "profile", args.calibration_prompts, 128)
            receipt["profile_sha256"] = sha256(profile)
            receipt["prompts_sha256"] = sha256(args.prompts)
            serve("ordinary", "native", args.prompts, 16)
            static_hash = None
            for pair in range(args.pairs):
                if args.all_resident_static:
                    name = f"pair-{pair}-static"
                    summary = serve(name, "static", args.prompts, args.tokens)
                    records = [
                        json.loads(r)
                        for r in (out / f"{name}.jsonl").read_text().splitlines()
                    ]
                    status = next(
                        r["status"] for r in records if r["kind"] == "prepared"
                    )
                    if not status["layers"] or any(
                        row["resident"] != row["experts"]
                        or row["max_pairs"]
                        or row["generation"]
                        for row in status["layers"].values()
                    ):
                        raise RuntimeError(
                            "all-resident static requires every expert resident and no updates"
                        )
                    if (
                        static_hash is not None
                        and static_hash != summary["output_token_sha256"]
                    ):
                        raise RuntimeError("repeated all-resident output mismatch")
                    static_hash = summary["output_token_sha256"]
                    if sha256(profile) != receipt["profile_sha256"]:
                        raise RuntimeError("learned profile changed during serving")
                    continue
                arms = (
                    ("static", "adaptive") if pair % 2 == 0 else ("adaptive", "static")
                )
                summaries = {
                    mode: serve(f"pair-{pair}-{mode}", mode, args.prompts, args.tokens)
                    for mode in arms
                }
                if (
                    summaries["static"]["output_token_sha256"]
                    != summaries["adaptive"]["output_token_sha256"]
                ):
                    raise RuntimeError("controlled-admission output mismatch")
                if not summaries["adaptive"]["epochs"]["promotions"]:
                    raise RuntimeError("adaptive serving did not exercise a promotion")
                if sha256(profile) != receipt["profile_sha256"]:
                    raise RuntimeError("learned profile changed during serving")
        if source_files() != receipt["source_files"]:
            raise RuntimeError("source changed during acceptance")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
