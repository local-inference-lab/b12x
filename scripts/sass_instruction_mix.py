#!/usr/bin/env python3
"""Census the SASS instruction mix of cached sparkinfer CuTe kernels.

ncu says WHERE the issue slots go (stalls, throughputs); this says WHAT the
instructions are. Phase G needs both: the FP6 prefill GEMM runs the MMA pipe at
72-75% where FP8 runs it at 93-95%, and 45% of FP6's stall cycles sit in
memory-pipe and barrier categories FP8 barely touches. That is enough to know
the MMA is starved, and not enough to know what is starving it.

Point ``SPARKINFER_COMPILE_CACHE_DIR`` at a fresh directory, run the workload,
then pass that directory here. The cache stores the compiled CUDA ELF inside
each host object, so this reads the exact cubin that ran rather than a
recompiled approximation.

The mainloop, not the whole kernel, is the number that matters. A GEMM spends
essentially all its time in one loop, and whole-kernel totals bury it under
prologue and epilogue code. Loop bodies are recovered from backward branches in
the disassembly.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.cutlass_migration.evidence.kernel_resources import (  # noqa: E402
    _TEXT_KERNEL_SECTION_RE,
    _embedded_cuda_elf,
    _kernel_code,
)

_TOOL_TIMEOUT_SECONDS = 120

# /*0a30*/ @!P0 LDSM.16.M88.4 R8, [R25+0x800] ;
_INSTRUCTION_RE = re.compile(
    r"^\s*/\*(?P<offset>[0-9a-fA-F]+)\*/\s*"
    r"(?P<predicate>@!?U?P[T0-9]+\s+)?"
    r"(?P<opcode>[A-Z][A-Z0-9_]*)"
    r"(?P<modifiers>(?:\.[A-Za-z0-9_]+)*)"
    r"(?P<operands>[^;]*);\s*$",
    re.MULTILINE,
)
# nvdisasm writes branch targets symbolically - ``@!P0 BRA `(.L_x_12) ;`` - and
# emits ``.L_x_12:`` on its own line ahead of the target instruction. Absolute
# hex targets appear only in some builds/flag combinations, so accept both.
_BRANCH_LABEL_RE = re.compile(r"`\((?P<label>\.L[A-Za-z0-9_$.]+)\)")
_BRANCH_HEX_RE = re.compile(r"\b0x(?P<target>[0-9a-fA-F]+)\b")
_LABEL_DEF_RE = re.compile(r"^\s*(?P<label>\.L[A-Za-z0-9_$.]+):", re.MULTILINE)

# Ordered: the first matching category wins, so put narrow patterns first.
# Category names are the vocabulary the Phase G writeup uses; keep them stable.
_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Tensor-core math. On SM120 the mxf8f6f4 block-scaled MMA disassembles as
    # a QGMMA/HGMMA/OMMA family member depending on operand kind.
    ("mma", re.compile(r"^(?:HMMA|IMMA|OMMA|QMMA|HGMMA|QGMMA|BMMA|MMA)$")),
    # Shared-memory operand feed. LDSM is the ldmatrix that fills MMA operand
    # registers; plain LDS/STS is everything else touching smem.
    ("ldsm", re.compile(r"^(?:LDSM|MOVM)$")),
    ("lds", re.compile(r"^LDS$")),
    ("sts", re.compile(r"^(?:STS|STSM)$")),
    # Global traffic. LDGSTS is cp.async; UTMA* is the TMA unit.
    ("tma", re.compile(r"^(?:UTMALDG|UTMASTG|UTMACCTL|UBLKCP)$")),
    ("ldgsts", re.compile(r"^(?:LDGSTS|LDGDEPBAR)$")),
    ("ldg", re.compile(r"^(?:LDG|LD)$")),
    ("stg", re.compile(r"^(?:STG|ST|RED|ATOM|ATOMG|ATOMS)$")),
    # Spills. Any nonzero count here is a finding on its own.
    ("local", re.compile(r"^(?:LDL|STL)$")),
    # Constant-bank reads. Cheap and broadcast, but they are how a kernel
    # re-reads kernel params it failed to hoist, so keep them visible rather
    # than folded into int_addr.
    ("const_load", re.compile(r"^(?:LDC|ULDC|LDCU)$")),
    # Synchronization. NANOSLEEP is what ncu reports as the `sleeping` stall.
    ("barrier", re.compile(r"^(?:BAR|BARRIER|BSSY|BSYNC|DEPBAR|MEMBAR|ERRBAR"
                           r"|ARRIVES|ARRIVELDS|ELECT|NANOSLEEP|YIELD|FENCE"
                           r"|SYNCS|ACQBULK|ENDCOLLECTIVE|CCTL|CCTLL|CCTLT)$")),
    # Predicate logic. Separate from int_addr because predicate pressure and
    # address pressure call for different fixes.
    ("pred", re.compile(r"^(?:PLOP3|UPLOP3|PSETP|UPSETP|P2R|R2P)$")),
    # Scalar float math: the epilogue and any in-kernel scaling.
    ("fp_math", re.compile(r"^(?:FADD|FMUL|FFMA|FSEL|FSETP|FMNMX|MUFU|FCHK"
                           r"|HADD2|HMUL2|HFMA2|HSETP2|HMNMX2|F2F|F2FP|F2I|I2F"
                           r"|I2FP|FRND|CVT)$")),
    # Integer and address arithmetic. This is the bucket that grows when a
    # kernel recomputes addresses per k-block instead of hoisting them.
    ("int_addr", re.compile(r"^(?:IMAD|IADD|IADD3|IABS|ISETP|LOP3|LEA|SHF|SGXT"
                            r"|BREV|FLO|POPC|PRMT|SEL|ICMP|IMNMX|VIADD"
                            r"|UIMAD|UIADD3|UISETP|ULOP3|ULEA|USHF|USEL"
                            r"|UPRMT|UFLO|UPOPC)$")),
    ("move", re.compile(r"^(?:MOV|UMOV|MOV32I|S2R|S2UR|CS2R|R2UR|UR2UP|R2B|B2R"
                        r"|SHFL|VOTE|VOTEU|REDUX|MATCH)$")),
    ("control", re.compile(r"^(?:BRA|BRX|JMP|JMX|CALL|RET|EXIT|NOP|PBK|BPT"
                           r"|SSY|SYNC|WARPSYNC|SETMAXREG|USETMAXREG"
                           r"|SETCTAID|PMTRIG)$")),
)


@dataclass
class Loop:
    """A SASS loop body recovered from a backward branch."""

    start: int
    end: int
    instructions: list[tuple[int, str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.instructions)


def _categorize(opcode: str) -> str:
    for name, pattern in _CATEGORIES:
        if pattern.match(opcode):
            return name
    return "other"


def _disassemble(object_path: Path, nvdisasm: str) -> str:
    cubin_bytes = _embedded_cuda_elf(object_path.read_bytes())
    # NamedTemporaryFile(delete=True) cannot be reopened by another process on
    # Windows, and the CUTLASS loader is known to patch ELFs it opens; write a
    # throwaway copy and disassemble that.
    with tempfile.TemporaryDirectory() as tmp:
        cubin_path = Path(tmp) / "kernel.cubin"
        cubin_path.write_bytes(cubin_bytes)
        result = subprocess.run(
            # No flags, matching the audited resource reader: the section
            # markers that _kernel_code keys on only appear in the default
            # full-object output.
            [nvdisasm, str(cubin_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_TOOL_TIMEOUT_SECONDS,
        )
    return result.stdout


def _instructions(code: str) -> list[tuple[int, str, str]]:
    """(offset, opcode, full text) for every SASS instruction, in order."""
    out: list[tuple[int, str, str]] = []
    for match in _INSTRUCTION_RE.finditer(code):
        out.append(
            (
                int(match.group("offset"), 16),
                match.group("opcode"),
                match.group(0).strip(),
            )
        )
    return out


def _modifiers_by_offset(code: str) -> dict[int, str]:
    """Offset -> the instruction's modifier suffix, e.g. ``.U8`` or ``.128``.

    Instruction COUNTS cannot distinguish 96 byte-wide shared loads from 96
    128-bit ones, and that distinction is the difference between a real
    vectorization win and none at all.
    """
    return {
        int(match.group("offset"), 16): match.group("modifiers")
        for match in _INSTRUCTION_RE.finditer(code)
    }


def _print_access_widths(
    body: list[tuple[int, str]],
    modifiers: dict[int, str],
    categories: tuple[str, ...] = ("lds", "ldsm", "sts", "ldg", "stg", "local"),
) -> None:
    """Break memory instructions down by their width modifier."""
    rows: Counter = Counter()
    for offset, opcode in body:
        if _categorize(opcode) not in categories:
            continue
        rows[f"{opcode}{modifiers.get(offset, '')}"] += 1
    if not rows:
        return
    print("\n  memory access widths")
    for name, count in sorted(rows.items(), key=lambda item: -item[1]):
        print(f"    {name:<28} {count:>6}")


def _label_offsets(code: str) -> dict[str, int]:
    """Map each branch label to the offset of the instruction it precedes."""
    labels: dict[str, int] = {}
    pending: list[str] = []
    for line in code.splitlines():
        label = _LABEL_DEF_RE.match(line)
        if label is not None:
            pending.append(label.group("label"))
            continue
        instruction = _INSTRUCTION_RE.match(line)
        if instruction is None or not pending:
            continue
        offset = int(instruction.group("offset"), 16)
        for name in pending:
            labels[name] = offset
        pending.clear()
    return labels


def _branch_targets(text: str, labels: dict[str, int]) -> list[int]:
    """Resolve every branch target in one instruction to an offset."""
    targets = [
        labels[match.group("label")]
        for match in _BRANCH_LABEL_RE.finditer(text)
        if match.group("label") in labels
    ]
    if targets:
        return targets
    # Hex fallback. Only consulted when no symbolic target resolved, because an
    # instruction can carry unrelated hex immediates that would read as targets.
    return [int(match.group("target"), 16) for match in _BRANCH_HEX_RE.finditer(text)]


def _find_loops(
    instructions: list[tuple[int, str, str]], labels: dict[str, int]
) -> list[Loop]:
    """Recover loop bodies from backward branches.

    A branch whose target is at or before its own offset closes a loop running
    from the target to the branch. Nested loops therefore appear as separate,
    overlapping entries; callers pick by size rather than by nesting, because
    the GEMM mainloop is the largest body by a wide margin.
    """
    by_offset = {offset: index for index, (offset, _, _) in enumerate(instructions)}
    loops: list[Loop] = []
    for index, (offset, opcode, text) in enumerate(instructions):
        if opcode not in {"BRA", "BRX", "JMP"}:
            continue
        targets = _branch_targets(text, labels)
        backward = [t for t in targets if t <= offset and t in by_offset]
        if not backward:
            continue
        start_index = by_offset[min(backward)]
        body = [(o, op) for o, op, _ in instructions[start_index : index + 1]]
        loops.append(Loop(start=min(backward), end=offset, instructions=body))
    return loops


def _dedupe_loops(loops: list[Loop]) -> list[Loop]:
    """Collapse loops that share a body; a multi-exit loop emits one per exit."""
    unique: dict[tuple[int, int], Loop] = {}
    for loop in loops:
        unique.setdefault((loop.start, loop.end), loop)
    return sorted(unique.values(), key=lambda loop: (loop.start, -loop.end))


def _mma_count(loop: Loop) -> int:
    return sum(1 for _, opcode in loop.instructions if _categorize(opcode) == "mma")


def _select_mainloop(loops: list[Loop]) -> Loop | None:
    """Pick the innermost loop that still issues MMA.

    Not the largest loop. These kernels are persistent, so the outermost loop
    is the tile scheduler and its body contains a whole tile's prologue,
    mainloop and epilogue - censusing it reports epilogue conversions and
    stores as though they were per-MMA mainloop cost. The k-loop is the
    innermost body that retains MMA, and it is the only one whose per-MMA
    ratios mean what Phase G needs them to mean.
    """
    with_mma = [loop for loop in loops if _mma_count(loop) > 0]
    if not with_mma:
        return None
    return min(with_mma, key=lambda loop: loop.size)


def _print_loop_table(loops: list[Loop], total: int) -> None:
    print(f"\n  loop nest ({len(loops)} distinct bodies)")
    print(f"    {'range':<21} {'insns':>6} {'% kernel':>9} {'mma':>6}")
    print(f"    {'-' * 21} {'-' * 6} {'-' * 9} {'-' * 6}")
    for loop in loops:
        share = 100.0 * loop.size / total if total else 0.0
        label = f"0x{loop.start:x}..0x{loop.end:x}"
        print(f"    {label:<21} {loop.size:>6} {share:>8.1f}% {_mma_count(loop):>6}")


def _census(body: list[tuple[int, str]]) -> Counter:
    counts: Counter = Counter()
    for _, opcode in body:
        counts[_categorize(opcode)] += 1
    return counts


def _opcode_census(body: list[tuple[int, str]]) -> Counter:
    counts: Counter = Counter()
    for _, opcode in body:
        counts[opcode] += 1
    return counts


def _print_census(title: str, counts: Counter, total: int, mma: int) -> None:
    print(f"\n{title}")
    print(f"  {'category':<12} {'count':>7} {'%':>7} {'per MMA':>9}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 9}")
    for name, _ in _CATEGORIES:
        count = counts.get(name, 0)
        if count == 0:
            continue
        share = 100.0 * count / total if total else 0.0
        per_mma = f"{count / mma:.2f}" if mma else "-"
        print(f"  {name:<12} {count:>7} {share:>6.1f}% {per_mma:>9}")
    if counts.get("other"):
        share = 100.0 * counts["other"] / total if total else 0.0
        per_mma = f"{counts['other'] / mma:.2f}" if mma else "-"
        print(f"  {'other':<12} {counts['other']:>7} {share:>6.1f}% {per_mma:>9}")
    print(f"  {'TOTAL':<12} {total:>7}")


def _print_branches(
    instructions: list[tuple[int, str, str]], labels: dict[str, int]
) -> None:
    """Dump raw branch lines when loop recovery finds nothing.

    Branch-target syntax varies across nvdisasm builds, so a zero-loop result
    is at least as likely to be a parsing failure as a genuinely branch-free
    kernel. Print the evidence rather than reporting the whole-kernel census as
    though it were the mainloop.
    """
    branches = [
        (offset, text)
        for offset, opcode, text in instructions
        if opcode in {"BRA", "BRX", "JMP", "CALL", "RET"}
    ]
    print(f"  branch instructions: {len(branches)}, resolvable labels: {len(labels)}")
    for offset, text in branches[:10]:
        print(f"    /*{offset:04x}*/ {text}")
    if len(branches) > 10:
        print(f"    ... {len(branches) - 10} more")


def _print_unclassified(body: list[tuple[int, str]]) -> None:
    """Name every opcode that fell into ``other``.

    An unexplained bucket is not evidence. If a category is large enough to
    matter it has to be nameable, and the opcode tables here are hand-written
    and certain to be incomplete for SM120.
    """
    unknown = Counter(
        opcode for _, opcode in body if _categorize(opcode) == "other"
    )
    if not unknown:
        return
    print("  unclassified opcodes (fix _CATEGORIES if any of these matter):")
    for opcode, count in unknown.most_common():
        print(f"    {opcode:<16} {count:>6}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="cache directories or .o object files to census",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="only census entry points whose mangled name contains this substring",
    )
    parser.add_argument(
        "--top-opcodes",
        type=int,
        default=12,
        help="show this many individual opcodes for the mainloop (0 disables)",
    )
    parser.add_argument(
        "--whole-kernel",
        action="store_true",
        help="also census the whole entry point, not just its largest loop",
    )
    parser.add_argument(
        "--dump-sass",
        type=Path,
        default=None,
        help=(
            "also write each censused entry point's disassembly here, so the "
            "instruction stream can be read directly; the cubin is embedded in "
            "the host object and nvdisasm cannot open the .o itself"
        ),
    )
    args = parser.parse_args()

    if args.dump_sass is not None:
        args.dump_sass.mkdir(parents=True, exist_ok=True)

    nvdisasm = shutil.which("nvdisasm")
    if nvdisasm is None:
        parser.error("nvdisasm is required (install the CUDA toolkit or add it to PATH)")

    object_paths: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            object_paths.extend(sorted(path.rglob("*.o")))
        else:
            object_paths.append(path)
    if not object_paths:
        parser.error(f"no .o objects found under {', '.join(map(str, args.paths))}")

    censused = 0
    for object_path in object_paths:
        try:
            disassembly = _disassemble(object_path, nvdisasm)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"SKIP {object_path.name}: {type(exc).__name__}: {exc}")
            continue

        kernels = sorted(
            {m.group("kernel") for m in _TEXT_KERNEL_SECTION_RE.finditer(disassembly)}
        )
        for kernel in kernels:
            if args.kernel and args.kernel not in kernel:
                continue
            code = _kernel_code(disassembly, kernel)
            instructions = _instructions(code)
            if not instructions:
                print(f"SKIP {kernel}: no SASS instructions parsed")
                continue

            print("=" * 78)
            print(f"object: {object_path.name}")
            print(f"kernel: {kernel}")
            print(f"instructions: {len(instructions)}")

            if args.dump_sass is not None:
                dump = args.dump_sass / f"{object_path.stem[:16]}_{kernel[:80]}.sass"
                dump.write_text(code)
                print(f"sass: {dump}")

            labels = _label_offsets(code)
            modifiers = _modifiers_by_offset(code)
            loops = _dedupe_loops(_find_loops(instructions, labels))
            print(f"labels: {len(labels)}  distinct loops: {len(loops)}")
            if loops:
                _print_loop_table(loops, len(instructions))
            mainloop = _select_mainloop(loops) if loops else None
            if mainloop is not None:
                counts = _census(mainloop.instructions)
                mma = counts.get("mma", 0)
                _print_census(
                    f"mainloop [0x{mainloop.start:x} .. 0x{mainloop.end:x}] "
                    f"({mainloop.size} instructions, {len(loops)} loops found)",
                    counts,
                    mainloop.size,
                    mma,
                )
                if counts.get("local"):
                    print(
                        f"  WARNING: {counts['local']} local-memory instructions in "
                        "the mainloop - the kernel is spilling."
                    )
                _print_unclassified(mainloop.instructions)
                _print_access_widths(mainloop.instructions, modifiers)
                if args.top_opcodes:
                    print(f"\n  top {args.top_opcodes} mainloop opcodes")
                    for opcode, count in _opcode_census(
                        mainloop.instructions
                    ).most_common(args.top_opcodes):
                        print(f"    {opcode:<16} {count:>6}")
            elif loops:
                print(
                    "\n  loops found but none issue MMA - not the GEMM entry point?"
                )
            else:
                print("\n  no loops found; censusing the whole kernel instead.")
                _print_branches(instructions, labels)

            # The outermost loop is the persistent tile scheduler: one iteration
            # is one output tile, so this is where the epilogue cost lives.
            outer = max(loops, key=lambda loop: loop.size) if loops else None
            if outer is not None and mainloop is not None and outer is not mainloop:
                counts = _census(outer.instructions)
                _print_census(
                    f"tile loop [0x{outer.start:x} .. 0x{outer.end:x}] "
                    f"({outer.size} instructions, mainloop + epilogue)",
                    counts,
                    outer.size,
                    counts.get("mma", 0),
                )

            if args.whole_kernel or not loops:
                body = [(offset, opcode) for offset, opcode, _ in instructions]
                counts = _census(body)
                _print_census(
                    "whole kernel", counts, len(body), counts.get("mma", 0)
                )
                _print_unclassified(body)
            censused += 1

    if censused == 0:
        print("no matching kernels censused", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
