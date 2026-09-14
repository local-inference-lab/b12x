#!/usr/bin/env python3
"""Prepare or execute the implemented SM103 operator qualification suite.

Preparation requires no CUDA context. Execution requires an explicitly selected
physical SM103 GPU and records every subprocess result. This suite does not
qualify complete models, performance, Grace memory, or Station RDMA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._sm103_source import package_source_sha256 as _package_source_sha256
from scripts._sm103_source import source_identity

SUITES = {
    "moe": ["tests/moe/test_sm103_pointwise.py", "tests/moe/test_sm103_nvfp4.py"],
    "trellis_reconstruction": ["tests/moe/test_sm103_trellis.py"],
    "trellis_projection": ["tests/moe/test_sm103_trellis_gemm.py", "tests/moe/test_sm103_trellis_staging.py"],
    "trellis_moe": ["tests/moe/test_sm103_trellis_moe.py", "tests/moe/test_sm103_trellis_transforms.py", "tests/moe/test_trellis_extents.py", "tests/moe/test_sm103_trellis_input_halves.py"],
    "trellis_atoms": ["tests/moe/test_sm103_trellis_atoms_moe.py", "tests/moe/test_sm103_trellis_atoms_staging.py", "tests/moe/test_sm103_trellis_transforms.py"],
    "trellis_mixed": ["tests/moe/test_sm103_trellis_mixed_moe.py", "tests/moe/test_sm103_trellis_mixed_staging.py", "tests/moe/test_sm103_trellis_transforms.py", "tests/moe/test_trellis_extents.py", "tests/moe/test_sm103_trellis_input_halves.py", "-k", "mixed or prepared_expert_map or extent or cross_half or column_selection"],
    "kda_decode": ["tests/sequence/test_gdn_decode_kda_cute.py"],
    "gdn_decode": ["tests/sequence/test_gdn_decode.py"],
    "kda_prefill": ["tests/sequence/test_kda_prefill.py"],
    "gdn_prefill": ["tests/sequence/test_gdn_prefill.py"],
    "mtp_feedback": ["tests/sequence/test_mtp_feedback.py", "-k", "not standalone_cute_norm"],
    "dense_mla": [
        "tests/attention/test_dense_mla.py",
        "tests/attention/test_dense_mla_window.py",
    ],
    "sparse_mla": ["tests/attention/test_sm103_sparse_mla.py"],
    "mhc": ["tests/norm/test_mhc.py", "tests/norm/test_residual_mhc.py",
            "tests/norm/test_mhc_lagged_parallel.py", "tests/norm/test_sm103_mhc.py"],
    "compressed_mla": [
        "tests/attention/test_sm103_compressed_mla.py",
        "tests/attention/test_mla_kv_cache.py::test_v41_writer_recipes_odd_pages_and_int64_pool_offsets",
        "tests/attention/test_mla_kv_cache.py::test_v41_writer_precompile_dynamic_rows_and_graph_replay",
    ],
    "dsa_indexer": [
        "tests/attention/test_sm103_dsa_indexer.py",
        "tests/attention/test_fused_indexer.py",
    ],
    "projection": ["tests/gemm/test_bf16_gemv.py", "-k", "not installed_plugins"],
    "blockscaled": ["tests/gemm/test_sm103_blockscaled.py", "tests/gemm/test_mxfp4_packing.py"],
    "fp8": ["tests/gemm/test_sm103_fp8.py"],
    "block_fp8_linear": ["tests/gemm/test_sm103_block_fp8_linear.py"],
    "wo_projection": ["tests/gemm/test_sm103_wo_projection.py"],
    "fp6": ["tests/gemm/test_sm103_fp6.py", "tests/quantization/test_fp6_workspace.py", "tests/gemm/test_fp6_smem_packing.py"],
}


def package_source_sha256():
    return _package_source_sha256(ROOT)


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def verify_compile_artifacts(manifest_path, compiled):
    root = manifest_path.resolve().parent
    artifacts = compiled.get("artifacts", [])
    if not artifacts or len(artifacts) != compiled.get("callables"):
        raise ValueError("compile manifest has incomplete callable coverage")
    for artifact in artifacts:
        files = artifact.get("files", {})
        if not any(name.endswith(".ptx") for name in files) or not any(
            name.endswith(".cubin") for name in files
        ):
            raise ValueError("each compiled callable requires PTX and cubin artifacts")
        for name, identity in files.items():
            path = (root / artifact["name"] / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("artifact path escapes the compile directory")
            payload = path.read_bytes()
            if (
                len(payload) != identity["bytes"]
                or hashlib.sha256(payload).hexdigest() != identity["sha256"]
            ):
                raise ValueError(f"compile artifact identity mismatch: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--device-uuid", help="physical GPU UUID; mandatory for execution"
    )
    parser.add_argument("--compile-manifest", type=Path)
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument(
        "--sanitizer-tool",
        choices=("memcheck", "synccheck", "racecheck"),
        default="memcheck",
    )
    args = parser.parse_args(argv)
    if args.execute and not args.device_uuid:
        parser.error(
            "--execute requires --device-uuid to select the physical SM103 GPU"
        )
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output-dir must be empty")
    if args.sanitizer and not args.sanitizer.is_file():
        parser.error("--sanitizer must name an installed compute-sanitizer executable")
    source_hash = package_source_sha256()
    compile_identity = None
    if args.compile_manifest:
        payload = args.compile_manifest.read_bytes()
        compiled = json.loads(payload)
        if (
            compiled.get("target") != "sm_103a"
            or compiled.get("status") != "cross-compiled"
        ):
            parser.error("compile manifest must describe successful SM103 compilation")
        if compiled.get("source_sha256") != source_hash:
            parser.error("compile manifest does not match the package source")
        verify_compile_artifacts(args.compile_manifest, compiled)
        compile_identity = {
            "path": str(args.compile_manifest.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    out.mkdir(parents=True, exist_ok=True)
    components = list(SUITES) if args.component == "all" else [args.component]
    receipt = {
        "status": "prepared",
        "runtime_qualified": False,
        "scope": "implemented operators",
        **source_identity(ROOT),
        "source_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((ROOT / "tests").rglob("*.py")))
        ).hexdigest(),
        "command": sys.argv if argv is None else argv,
        "compile_manifest": compile_identity,
        "components": components,
        "results": [],
        "excluded": [
            "complete GLM/V4.1 serving",
            "DFlash2 integration",
            "frozen QSRT coupled high-rate conversion",
            "Grace memory",
            "Station RDMA",
            "FlashInfer/vLLM plugin installation",
            "performance qualification",
        ],
    }
    if args.sanitizer:
        receipt["sanitizer"] = {
            "path": str(args.sanitizer.resolve()),
            "sha256": hashlib.sha256(args.sanitizer.read_bytes()).hexdigest(),
            "version": subprocess.check_output(
                [str(args.sanitizer.resolve()), "--version"], text=True
            ),
            "tool": args.sanitizer_tool,
        }
    for component in components:
        command = [
            sys.executable,
            "-m",
            "pytest",
            *SUITES[component],
            "-q",
            "-ra",
            f"--junitxml={out / (component + '.xml')}",
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
        receipt["results"].append(
            {"component": component, "command": command, "status": "prepared"}
        )
    manifest = out / "qualification.json"

    def save():
        manifest.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    if not args.execute:
        for result in receipt["results"]:
            print(shlex.join(result["command"]))
        print(f"Prepared operator qualification in {manifest}; no GPU work executed.")
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_uuid
    os.environ["CUTE_DSL_ARCH"] = "sm_103a"
    try:
        import importlib.metadata
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (
            10,
            3,
        ):
            raise RuntimeError("execution requires a physical SM103 GPU")
        props = torch.cuda.get_device_properties(0)
        receipt["device"] = {
            "uuid": args.device_uuid,
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "sm_count": props.multi_processor_count,
            "total_memory": props.total_memory,
        }
        receipt["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "cuda-python", "nvidia-cutlass-dsl", "triton")
        }
        snapshot = subprocess.run(
            ["nvidia-smi", "-q", "-i", args.device_uuid],
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "gpu-before.txt").write_text(snapshot.stdout)
        receipt["status"] = "running"
        save()
        for result in receipt["results"]:
            log = out / (result["component"] + ".log")
            with log.open("w") as stream:
                completed = subprocess.run(
                    result["command"],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result.update(
                returncode=completed.returncode,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            )
            xml = out / (result["component"] + ".xml")
            counts = junit_counts(xml) if xml.exists() else {}
            result["tests"] = counts
            passed = (
                completed.returncode == 0
                and counts.get("tests", 0) > 0
                and all(
                    counts.get(key, 0) == 0 for key in ("failures", "errors", "skipped")
                )
            )
            result["status"] = "passed" if passed else "failed"
            save()
            if not passed:
                raise RuntimeError(
                    f"{result['component']} failed or skipped required qualification; see {log}"
                )
        if package_source_sha256() != source_hash:
            raise RuntimeError("package source changed during qualification")
        if (
            hashlib.sha256(
                b"".join(p.read_bytes() for p in sorted((ROOT / "tests").rglob("*.py")))
            ).hexdigest()
            != receipt["test_source_sha256"]
        ):
            raise RuntimeError("qualification test source changed during execution")
        snapshot = subprocess.run(
            ["nvidia-smi", "-q", "-i", args.device_uuid],
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "gpu-after.txt").write_text(snapshot.stdout)
        receipt.update(status="operator-qualification-passed", runtime_qualified=True)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    save()
    print(f"Implemented operator suite passed; receipt: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
