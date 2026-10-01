"""Compare manifest-bound CUDA objects from identical b12x source.

ELF extraction and nvdisasm parsing retain the checked routines from
b12x revision 070af6100^, evidence/kernel_resources.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path

_CUDA_ELF_MAGIC = b"\x7fELF\x02\x01\x01\x41"

def _section(disassembly: str, section: str) -> str:
    marker = f"//--------------------- {section} "
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _attribute_blocks(section: str, attribute: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*//----- nvinfo : {re.escape(attribute)}[ \t]*\r?$"
        rf"(?P<body>.*?)"
        rf"(?=^[ \t]*//----- nvinfo :|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    return [match.group("body") for match in pattern.finditer(section)]


def _attribute_block(section: str, attribute: str) -> str:
    blocks = _attribute_blocks(section, attribute)
    if len(blocks) > 1:
        raise ValueError(f"nvdisasm reported duplicate per-kernel {attribute} blocks")
    return blocks[0] if blocks else ""


def _resource_values(disassembly: str, attribute: str) -> dict[str, int]:
    """Parse one bounded global .nv.info resource-attribute block."""

    global_info = _section(disassembly, ".nv.info")
    if not global_info:
        raise ValueError("nvdisasm omitted the global .nv.info section")
    blocks = _attribute_blocks(global_info, attribute)
    if not blocks:
        raise ValueError(f"nvdisasm omitted the global {attribute} block")

    entries: dict[str, int] = {}
    for block_number, block in enumerate(blocks, start=1):
        pending_kernel: str | None = None
        for line in block.splitlines():
            index_match = re.search(r"\.word\s+index@\(([^)]+)\)", line)
            value_match = re.search(r"\.word\s+0x([0-9a-fA-F]+)", line)
            if index_match is not None:
                if pending_kernel is not None:
                    raise ValueError(
                        f"{attribute} block {block_number} has index "
                        f"{pending_kernel!r} without a value"
                    )
                pending_kernel = index_match.group(1)
            elif value_match is not None:
                if pending_kernel is None:
                    raise ValueError(
                        f"{attribute} block {block_number} has a value without "
                        "a kernel/function index"
                    )
                if pending_kernel in entries:
                    raise ValueError(
                        f"{attribute} has duplicate entry for {pending_kernel}"
                    )
                entries[pending_kernel] = int(value_match.group(1), 16)
                pending_kernel = None
        if pending_kernel is not None:
            raise ValueError(
                f"{attribute} block {block_number} has index {pending_kernel!r} "
                "without a value"
            )
    if not entries:
        raise ValueError(f"nvdisasm reported no entries in {attribute}")
    return entries


def _kernel_code(disassembly: str, kernel: str) -> str:
    marker = f"//--------------------- .text.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _kernel_info(disassembly: str, kernel: str) -> str:
    marker = f"//--------------------- .nv.info.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _attribute_words(kernel_info: str, attribute: str) -> list[int]:
    block = _attribute_block(kernel_info, attribute)
    return [int(value, 16) for value in re.findall(r"\.word\s+0x([0-9a-fA-F]+)", block)]


def _attribute_short(kernel_info: str, attribute: str) -> int:
    block = _attribute_block(kernel_info, attribute)
    match = re.search(r"\.short\s+0x([0-9a-fA-F]+)", block)
    return int(match.group(1), 16) if match else 0


def _cubin_shared_section_bytes(disassembly: str, kernel: str) -> int:
    marker = f"//--------------------- .nv.shared.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return 0
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    section = (
        disassembly[start:] if next_section < 0 else disassembly[start:next_section]
    )
    return sum(
        int(value, 0) for value in re.findall(r"\.zero\s+([0-9xa-fA-F]+)", section)
    )


def _ptxas_metadata(disassembly: str) -> tuple[str, str]:
    version_match = re.search(
        r'\.string\s+"(Cuda compilation tools,[^"]+)"', disassembly
    )
    flags_match = next(
        (
            match
            for match in re.finditer(r'\.string\s+"([^"\r\n]+)"', disassembly)
            if re.search(r"(?:^|\s)-O\s+\d+(?:\s|$)", match.group(1))
            and re.search(r"(?:^|\s)-arch\s+\S+", match.group(1))
        ),
        None,
    )
    return (
        version_match.group(1) if version_match else "",
        flags_match.group(1).strip() if flags_match else "",
    )


def _embedded_cuda_elf(object_bytes: bytes) -> bytes:
    """Extract the exact embedded ELF64 CUDA object, excluding wrapper bytes."""

    embedded_cubin_count = object_bytes.count(_CUDA_ELF_MAGIC)
    if embedded_cubin_count != 1:
        raise ValueError(
            f"expected exactly one embedded CUDA ELF, found {embedded_cubin_count}"
        )
    cubin_start = object_bytes.find(_CUDA_ELF_MAGIC)
    available = len(object_bytes) - cubin_start
    if available < 64:
        raise ValueError("embedded CUDA ELF header is truncated")

    # CUDA cubins are ELF64 little-endian objects. Extended table counts would
    # require consulting section zero; reject them instead of guessing an
    # extent that could absorb bytes belonging to the host-object wrapper.
    e_phoff = struct.unpack_from("<Q", object_bytes, cubin_start + 0x20)[0]
    e_shoff = struct.unpack_from("<Q", object_bytes, cubin_start + 0x28)[0]
    e_ehsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x34)[0]
    e_phentsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x36)[0]
    e_phnum = struct.unpack_from("<H", object_bytes, cubin_start + 0x38)[0]
    e_shentsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", object_bytes, cubin_start + 0x3C)[0]
    if e_ehsize < 64:
        raise ValueError(f"embedded CUDA ELF has invalid header size {e_ehsize}")
    if e_phnum == 0xFFFF or e_shnum == 0:
        raise ValueError("embedded CUDA ELF uses unsupported extended table counts")
    if e_phnum and (not e_phoff or e_phentsize < 56):
        raise ValueError("embedded CUDA ELF has an invalid program-header table")
    if e_shnum and (not e_shoff or e_shentsize < 64):
        raise ValueError("embedded CUDA ELF has an invalid section-header table")

    def checked_end(offset: int, size: int, label: str) -> int:
        end = offset + size
        if offset < 0 or size < 0 or end < offset or end > available:
            raise ValueError(f"embedded CUDA ELF {label} is truncated")
        return end

    extent = checked_end(0, e_ehsize, "header")
    if e_phnum:
        extent = max(
            extent,
            checked_end(e_phoff, e_phentsize * e_phnum, "program-header table"),
        )
        for index in range(e_phnum):
            header = cubin_start + e_phoff + index * e_phentsize
            p_offset = struct.unpack_from("<Q", object_bytes, header + 0x08)[0]
            p_filesz = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
            extent = max(
                extent,
                checked_end(p_offset, p_filesz, f"program segment {index}"),
            )
    extent = max(
        extent,
        checked_end(e_shoff, e_shentsize * e_shnum, "section-header table"),
    )
    for index in range(e_shnum):
        header = cubin_start + e_shoff + index * e_shentsize
        sh_type = struct.unpack_from("<I", object_bytes, header + 0x04)[0]
        sh_offset = struct.unpack_from("<Q", object_bytes, header + 0x18)[0]
        sh_size = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
        if sh_type != 8:  # SHT_NOBITS occupies memory but has no file payload.
            extent = max(
                extent,
                checked_end(sh_offset, sh_size, f"section {index}"),
            )
    return object_bytes[cubin_start : cubin_start + extent]


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_digest(value):
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode())


def comparison_key(payload):
    """Normalize only disabled PyIR and the package-owned runtime library path.

    CUTLASS 4.7 records its disabled PyIR frontend explicitly. CUTLASS 4.6
    instead mutates CUTE_DSL_LIBS at import. Neither coordinate selects a
    different user program. Raw manifests and artifact hashes stay intact.
    """
    payload = json.loads(json.dumps(payload))
    payload["compile_options"] = [
        value for value in payload["compile_options"] if value != "enable-pyir=false"
    ]
    kwargs = payload.get("compile_kwargs", {})
    if "__dsl_compile_options_key" in kwargs:
        kwargs["__dsl_compile_options_key"] = [
            value for value in kwargs["__dsl_compile_options_key"]
            if value != "enable-pyir=false"
        ]
        payload["compile_kwargs_hash"] = _json_digest(kwargs)
    environment = []
    for name, value in payload["compile_environment"]:
        if name == "CUTE_DSL_LIBS":
            components = value.split(":")
            value = ":".join(component for component in components if not (
                Path(component).name == "libcute_dsl_runtime.so"
                and "nvidia_cutlass_dsl" in Path(component).parts
            ))
            if not value:
                continue
        environment.append([name, value])
    payload["compile_environment"] = environment
    return _json_digest(payload)


def collect(cache: Path, output: Path, nvdisasm: str):
    """Audit every manifest/object pair without loading or modifying the cache."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    manifests = sorted(cache.rglob("*.json"))
    if not manifests:
        raise ValueError(f"no manifests in {cache}")
    objects = {p.resolve() for p in cache.rglob("*.o")}
    if objects != {p.with_suffix(".o").resolve() for p in manifests}:
        raise ValueError("object/manifest coverage differs")
    for path in manifests:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
        obj = path.with_suffix(".o").read_bytes()
        if manifest["schema"] != "b12x._lib.compile_manifest.v3":
            raise ValueError(f"unknown manifest schema: {path}")
        if manifest["cache_key"] != path.stem:
            raise ValueError(f"cache identity mismatch: {path}")
        if _digest(obj) != manifest["object_sha256"]:
            raise ValueError(f"object hash mismatch: {path}")
        if _json_digest(manifest["semantic_payload"]) != manifest["semantic_key"]:
            raise ValueError(f"semantic hash mismatch: {path}")
        evidence = {key: manifest[key] for key in (
            "cache_key", "object_sha256", "launch_metadata",
        )}
        if _json_digest(evidence) != manifest["artifact_evidence_sha256"]:
            raise ValueError(f"artifact evidence hash mismatch: {path}")
        launch = manifest["launch_metadata"]
        if launch["status"] != "exact":
            raise ValueError(f"launch metadata is not exact: {path}")
        cubin = _embedded_cuda_elf(obj)
        cubin_path = output / f"{path.stem}.cubin"
        cubin_path.write_bytes(cubin)
        sass = subprocess.check_output([nvdisasm, str(cubin_path)], text=True, timeout=60)
        (output / f"{path.stem}.sass").write_text(sass)
        registers = _resource_values(sass, "EIATTR_REGCOUNT")
        frames = _resource_values(sass, "EIATTR_FRAME_SIZE")
        stacks = _resource_values(sass, "EIATTR_MIN_STACK_SIZE")
        kernels = set(re.findall(r"^//-+ \.text\.(\S+)\s", sass, re.MULTILINE))
        if kernels != set(launch["launch_dynamic_smem_bytes"]):
            raise ValueError(f"CUDA entry-point/launch coverage differs: {path}")
        ptxas, flags = _ptxas_metadata(sass)
        for kernel in sorted(kernels):
            code = _kernel_code(sass, kernel)
            instructions = re.findall(r"/\*([0-9a-fA-F]+)\*/\s+([^\n]*;)", code)
            if not instructions:
                raise ValueError(f"no SASS instructions: {kernel}")
            instruction_text = "\n".join(line for _, line in instructions)
            sets = {kind: sorted({int(v) for v in re.findall(
                rf"\b{kind}([0-9]+)\b", instruction_text,
            )}) for kind in ("R", "UR", "P", "UP")}
            smem = launch["launch_dynamic_smem_bytes"][kernel]
            if len(set(smem)) != 1:
                raise ValueError(f"multiple launch SMEM values: {kernel}")
            row = {
                "kernel_id": manifest.get("kernel_id"),
                "semantic_key": manifest["semantic_key"],
                "semantic_payload": manifest["semantic_payload"],
                "cache_key": manifest["cache_key"], "kernel": kernel,
                "package_fingerprint": manifest["package_fingerprint"],
                "toolchain": manifest["toolchain"],
                "manifest": str(path.resolve()), "manifest_sha256": _digest(manifest_bytes),
                "object_sha256": _digest(obj), "cubin_sha256": _digest(cubin),
                "sass_sha256": _digest(sass.encode()), "ptxas": ptxas, "ptxas_flags": flags,
                "allocated_gpr": registers[kernel], "frame_bytes": frames[kernel],
                "min_stack_bytes": stacks[kernel],
                "launch_smem_bytes": smem[0],
                "static_smem_bytes": _cubin_shared_section_bytes(sass, kernel),
                "threads": _attribute_words(_kernel_info(sass, kernel), "EIATTR_REQNTID"),
                "register_sets": sets,
                "local_loads": len(re.findall(r"\bLDL\b", instruction_text)),
                "local_stores": len(re.findall(r"\bSTL\b", instruction_text)),
                "setmaxnreg": re.findall(r"SETMAXNREG[^;]*;", instruction_text),
                "instruction_count": len(instructions),
                "code_bytes": max(int(offset, 16) for offset, _ in instructions) + 16,
            }
            rows.append(row)
        if _digest(path.with_suffix(".o").read_bytes()) != manifest["object_sha256"]:
            raise ValueError(f"object changed during audit: {path}")
    report = {"schema": "b12x.cutlass.resource_census.v1", "cache": str(cache.resolve()),
              "nvdisasm": subprocess.check_output([nvdisasm, "--version"], text=True),
              "occupancy": "not measured", "rows": rows}
    (output / "resources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"objects": len(manifests), "entry_points": len(rows),
                      "kernel_ids": sorted({r["kernel_id"] for r in rows})}))


def compare(baseline: Path, candidate: Path, output: Path):
    """Pair exact semantic identities; retain every positive resource delta."""
    def load(path):
        report = json.loads(path.read_text())
        if report["schema"] != "b12x.cutlass.resource_census.v1":
            raise ValueError(f"unknown census schema: {path}")
        index = {(comparison_key(r["semantic_payload"]), r["kernel"]): r
                 for r in report["rows"]}
        if len(index) != len(report["rows"]):
            raise ValueError(f"duplicate semantic entry point: {path}")
        return index

    before, after = load(baseline), load(candidate)
    if set(before) != set(after):
        raise ValueError(f"unmatched entries: baseline-only={len(before.keys() - after.keys())}, "
                         f"candidate-only={len(after.keys() - before.keys())}")
    fingerprints = {r["package_fingerprint"] for r in (*before.values(), *after.values())}
    if len(fingerprints) != 1:
        raise ValueError("source fingerprints differ")
    deltas = []
    metrics = ("allocated_gpr", "frame_bytes", "min_stack_bytes", "local_loads",
               "local_stores", "launch_smem_bytes", "static_smem_bytes", "code_bytes",
               "instruction_count")
    for key, a in before.items():
        b = after[key]
        compiler_fields = {
            "cutlass_dsl", "cutlass_dsl_libs_base", "cutlass_dsl_libs_core",
            "cutlass_dsl_libs_cu12", "cutlass_dsl_libs_cu13",
        }
        if ([v for v in a["toolchain"] if v[0] not in compiler_fields]
                != [v for v in b["toolchain"] if v[0] not in compiler_fields]):
            raise ValueError("non-CUTLASS toolchain fields differ")
        changes = {name: b[name] - a[name] for name in metrics}
        register_changes = {}
        for kind in ("R", "UR", "P", "UP"):
            sa, sb = set(a["register_sets"][kind]), set(b["register_sets"][kind])
            register_changes[kind] = {"added": sorted(sb - sa), "removed": sorted(sa - sb)}
            changes[f"{kind}_count"] = len(sb) - len(sa)
            changes[f"{kind}_span"] = max(sb, default=-1) - max(sa, default=-1)
        deltas.append({"comparison_key": key[0], "kernel": key[1], "kernel_id": a["kernel_id"],
                       "before_semantic_key": a["semantic_key"], "after_semantic_key": b["semantic_key"],
                       "delta": changes, "positive": [n for n, d in changes.items() if d > 0],
                       "register_set_changes": register_changes,
                       "threads_changed": a["threads"] != b["threads"],
                       "setmaxnreg_changed": a["setmaxnreg"] != b["setmaxnreg"],
                       "before": a["cache_key"], "after": b["cache_key"]})
    counts = Counter(n for row in deltas for n in row["positive"])
    report = {"schema": "b12x.cutlass.resource_delta.v1", "matched": len(deltas),
              "baseline_sha256": _digest(baseline.read_bytes()),
              "candidate_sha256": _digest(candidate.read_bytes()),
              "positive_counts": dict(counts), "rows": deltas}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matched": len(deltas), "positive_counts": dict(counts)}))


def occupancy(report_path: Path, output: Path):
    """Query driver occupancy using verified cubin copies and exact launch sizes."""
    from cuda.bindings import driver

    def checked(result):
        error, *values = result
        if error != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {error}")
        return values[0] if len(values) == 1 else values

    report = json.loads(report_path.read_text())
    checked(driver.cuInit(0))
    device = checked(driver.cuDeviceGet(0))
    context = checked(driver.cuDevicePrimaryCtxRetain(device))
    checked(driver.cuCtxSetCurrent(context))
    rows = []
    try:
        for row in report["rows"]:
            binary = (report_path.parent / (row["cache_key"] + ".cubin")).read_bytes()
            if _digest(binary) != row["cubin_sha256"]:
                raise ValueError("cubin copy hash mismatch")
            if len(row["threads"]) != 3:
                raise ValueError("exact required thread dimensions are missing")
            module = checked(driver.cuModuleLoadData(binary))
            try:
                function = checked(driver.cuModuleGetFunction(module, row["kernel"].encode()))
                smem = row["launch_smem_bytes"]
                if smem:
                    checked(driver.cuFuncSetAttribute(function,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
                blocks = checked(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    function, math.prod(row["threads"]), smem,
                ))
                rows.append({"comparison_key": comparison_key(row["semantic_payload"]),
                             "kernel": row["kernel"], "blocks_per_sm": blocks,
                             "cache_key": row["cache_key"]})
            finally:
                checked(driver.cuModuleUnload(module))
        result = {"schema": "b12x.cutlass.occupancy.v1", "resource_sha256": _digest(report_path.read_bytes()),
                  "device_uuid": str(checked(driver.cuDeviceGetUuid(device))),
                  "driver_version": checked(driver.cuDriverGetVersion()), "rows": rows}
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"entry_points": len(rows), "blocks_per_sm": dict(Counter(r["blocks_per_sm"] for r in rows))}))
    finally:
        checked(driver.cuDevicePrimaryCtxRelease(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("cache", type=Path)
    collect_parser.add_argument("output", type=Path)
    collect_parser.add_argument("--nvdisasm", default="nvdisasm")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("output", type=Path)
    occupancy_parser = commands.add_parser("occupancy")
    occupancy_parser.add_argument("report", type=Path)
    occupancy_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.cache, args.output, args.nvdisasm)
    elif args.command == "compare":
        compare(args.baseline, args.candidate, args.output)
    else:
        occupancy(args.report, args.output)


if __name__ == "__main__":
    main()
