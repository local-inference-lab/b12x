#!/usr/bin/env python3
"""Reassemble flagged SM103 packing PTX with one assembler without GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_sm103_resources import metrics, read_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if comparison["baseline_sha256"] != digest(args.baseline):
        parser.error("comparison does not bind the baseline manifest")
    if comparison["manifest_sha256"] != digest(args.current):
        parser.error("comparison does not bind the current manifest")
    names = sorted(
        name for name in comparison["positive_existing_deltas"]
        if name.startswith("activation_pack_")
    )
    if not names:
        parser.error("comparison contains no flagged activation packing kernels")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "reassembled without GPU execution",
        "runtime_qualified": False,
        "performance_claim": False,
        "command": sys.argv,
        "script_sha256": digest(Path(__file__)),
        "comparison_sha256": digest(args.comparison),
        "tools": {
            name: {
                "path": str(path.resolve()), "sha256": digest(path),
                "version": subprocess.check_output([str(path), "--version"], text=True),
            }
            for name, path in (
                ("ptxas", args.ptxas), ("cuobjdump", args.cuobjdump),
                ("nvdisasm", args.nvdisasm),
            )
        },
        "cases": {},
    }
    for arm, manifest_path in (("baseline", args.baseline), ("current", args.current)):
        manifest = read_manifest(manifest_path)
        artifacts = {item["name"]: item for item in manifest["artifacts"]}
        for name in names:
            original = artifacts[name]
            source_dir = manifest_path.parent / name
            metadata = json.loads((source_dir / (name + ".metadata.json")).read_text())
            options = metadata["options"]
            case = report["cases"].setdefault(name, {"options": options})
            if case["options"] != options:
                raise ValueError(f"{name}: compile options differ between arms")
            (ptx_name,) = [p for p in original["files"] if p.endswith(".ptx")]
            source_ptx = source_dir / ptx_name
            source_hash = digest(source_ptx)
            directory = out / arm / name
            directory.mkdir(parents=True)
            ptx = directory / (name + ".ptx")
            ptx.write_bytes(source_ptx.read_bytes())
            cubin = directory / (name + ".cubin")
            command = [str(args.ptxas.resolve()), "-lineinfo", "-v",
                       "--regAllocOptLevel=2", "--gpu-name=sm_103a"]
            if not options["enable_fp_fusion"]:
                command.append("--fmad=false")
            command += [str(ptx), "-o", str(cubin)]
            with (directory / "ptxas.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            for suffix, tool in (
                ("resources.txt", [str(args.cuobjdump.resolve()), "--dump-resource-usage"]),
                ("sass", [str(args.nvdisasm.resolve())]),
            ):
                (directory / (name + "." + suffix)).write_text(
                    subprocess.check_output([*tool, str(cubin)], text=True)
                )
            if digest(source_ptx) != source_hash or digest(ptx) != source_hash:
                raise RuntimeError(f"{name}: PTX identity changed during assembly")
            files = {p.name: {"sha256": digest(p)} for p in directory.iterdir()}
            measured = metrics(out / arm, {"name": name, "files": files, "native_mma": False})
            case[arm] = {"command": command, "source_ptx_sha256": source_hash,
                         "files": files, "metrics": measured}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(names), "output": str(out)}))


if __name__ == "__main__":
    main()
