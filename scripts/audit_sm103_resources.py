#!/usr/bin/env python3
"""Compare hash-verified SM103 compile artifacts and retain every resource delta."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qualify_sm103 import verify_compile_artifacts


def read_manifest(path):
    value = json.loads(path.read_text())
    if value["status"] != "cross-compiled" or value["target"] != "sm_103a":
        raise ValueError("resource audit requires successful SM103 compilation")
    verify_compile_artifacts(path, value)
    names = [a["name"] for a in value["artifacts"]]
    if len(set(names)) != len(names):
        raise ValueError("compile manifest contains duplicate callable names")
    return value


def metrics(root, artifact):
    name = artifact["name"]
    files = artifact["files"]
    for suffix in (".resources.txt", ".sass"):
        if name + suffix not in files:
            raise ValueError(f"{name}: resource/SASS report is not manifest-bound")
    resources = (root / name / (name + ".resources.txt")).read_text()
    entries = [
        {k: int(v) for k, v in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", line)}
        for line in resources.splitlines()
        if "REG:" in line
    ]
    if not entries or any(
        set(entry) != {"REG", "STACK", "SHARED", "LOCAL"} for entry in entries
    ):
        raise ValueError(f"{name}: incomplete allocated-resource report")
    sass = (root / name / (name + ".sass")).read_text()
    instructions = "\n".join(
        re.findall(r"^\s*/\*[0-9a-fA-F]+\*/.*;\s*$", sass, re.MULTILINE)
    )
    if not instructions:
        raise ValueError(f"{name}: empty SASS instruction stream")
    (ptx_file,) = [p for p in files if p.endswith(".ptx")]
    ptx = (root / name / ptx_file).read_bytes()
    readers = b"tcgen05.ld." in ptx
    if readers and b"tcgen05.wait::ld.sync.aligned;" not in ptx:
        raise ValueError(f"{name}: missing TMEM load-completion wait")
    if (
        artifact["native_mma"]
        and "UTCHMMA" not in instructions
        and "UTCOMMA" not in instructions
    ):
        raise ValueError(f"{name}: native MMA absent from SASS")
    return {
        "entries": entries,
        "register_sets": {
            kind: sorted(
                set(map(int, re.findall(r"\b" + kind + r"(\d+)\b", instructions)))
            )
            for kind in ("R", "UR", "P", "UP")
        },
        "instructions": len(instructions.splitlines()),
        "local_loads": len(re.findall(r"\bLDL(?:\.|\b)", instructions)),
        "local_stores": len(re.findall(r"\bSTL(?:\.|\b)", instructions)),
        "tmem_reader_with_wait": readers,
        "native_mma": artifact["native_mma"],
    }


def audit(current, baseline):
    new, old = read_manifest(current), read_manifest(baseline)
    prior = {a["name"]: a for a in old["artifacts"]}
    records, added, changed, positive = {}, [], {}, {}
    equal_ptx, equal_cubins = 0, 0
    for artifact in new["artifacts"]:
        name = artifact["name"]
        measured = metrics(current.parent, artifact)
        records[name] = measured
        if name not in prior:
            added.append(name)
            continue
        previous = metrics(baseline.parent, prior[name])
        identities = {}
        for extension in (".ptx", ".cubin"):
            before = sorted(
                v["sha256"]
                for k, v in prior[name]["files"].items()
                if k.endswith(extension)
            )
            after = sorted(
                v["sha256"]
                for k, v in artifact["files"].items()
                if k.endswith(extension)
            )
            identities[extension] = {
                "before": before,
                "after": after,
                "identical": before == after,
            }
        equal_ptx += identities[".ptx"]["identical"]
        equal_cubins += identities[".cubin"]["identical"]
        if not identities[".cubin"]["identical"]:
            changed[name] = {
                "identities": identities,
                "metrics_equal": previous == measured,
            }
        deltas = {}
        for entry, (after, before) in enumerate(
            zip(measured["entries"], previous["entries"], strict=True)
        ):
            for key in after:
                if after[key] > before[key]:
                    deltas[f"{entry}.{key}"] = after[key] - before[key]
        for kind, registers in measured["register_sets"].items():
            increase = len(registers) - len(previous["register_sets"][kind])
            if increase > 0:
                deltas[kind] = increase
        if deltas:
            positive[name] = deltas
    removed = sorted(set(prior) - set(records))
    return {
        "schema": "b12x.sm103.resource_comparison.v1",
        "command": sys.argv,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "source_sha256": new["source_sha256"],
        "callables": len(records),
        "cuda_entries": sum(len(v["entries"]) for v in records.values()),
        "identical_existing_ptx": equal_ptx,
        "identical_existing_cubins": equal_cubins,
        "added": added,
        "removed": removed,
        "changed_cubins": changed,
        "positive_existing_deltas": positive,
        "tmem_readers_with_wait": sum(
            v["tmem_reader_with_wait"] for v in records.values()
        ),
        "stack_or_local_flags": {
            name: value
            for name, value in records.items()
            if any(e["STACK"] or e["LOCAL"] for e in value["entries"])
            or value["local_loads"]
            or value["local_stores"]
        },
        "metrics": records,
        "runtime_qualified": False,
        "performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest.resolve(), args.baseline.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"metrics", "changed_cubins"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
