"""Summarize the executed SASS instructions in an Nsight Compute source CSV.

Usage:

    sudo ncu -i report.ncu-rep --page source --csv \
        | python benchmarks/analyze_ncu_source.py

The source page contains one row per SASS instruction and reports dynamic
``Instructions Executed`` counts.  Grouping those counts by opcode is a useful
way to compare kernels with different implementations but equivalent work.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict


_PREDICATE_RE = re.compile(r"^@!?[A-Za-z0-9_.]+\s+")


def _opcode(source: str, *, keep_suffixes: bool) -> str | None:
    source = _PREDICATE_RE.sub("", source.strip())
    if not source:
        return None
    opcode = source.split(None, 1)[0]
    if opcode.endswith(":"):
        return None
    return opcode if keep_suffixes else opcode.split(".", 1)[0]


def _parse_count(value: str) -> int:
    value = value.strip().replace(",", "")
    if not value or value == "-":
        return 0
    return int(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="Instructions Executed",
        help="NCU source-page metric to aggregate",
    )
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--keep-suffixes",
        action="store_true",
        help="Keep opcode modifiers such as .E and .WIDE",
    )
    parser.add_argument(
        "--top-sites",
        type=int,
        default=0,
        help="Also print the highest-count individual SASS sites",
    )
    parser.add_argument(
        "--kernel-regex",
        help="Only include source rows from kernel names matching this regex",
    )
    args = parser.parse_args()

    reader = csv.reader(sys.stdin)
    header: list[str] | None = None
    current_kernel = ""
    kernel_re = re.compile(args.kernel_regex) if args.kernel_regex else None
    totals: dict[str, int] = defaultdict(int)
    static_sites: dict[str, int] = defaultdict(int)
    sites: list[tuple[int, str, str]] = []

    for row in reader:
        if not row:
            continue
        if row[0] == "Kernel Name":
            current_kernel = row[1] if len(row) > 1 else ""
            header = None
            continue
        if row[0] == "Address":
            header = row
            continue
        if kernel_re is not None and not kernel_re.search(current_kernel):
            continue
        if header is None or len(row) != len(header):
            continue
        fields = dict(zip(header, row))
        opcode = _opcode(fields.get("Source", ""), keep_suffixes=args.keep_suffixes)
        if opcode is None:
            continue
        count = _parse_count(fields.get(args.metric, "0"))
        totals[opcode] += count
        static_sites[opcode] += 1
        if args.top_sites and count:
            sites.append(
                (
                    count,
                    fields.get("Address", "").strip(),
                    fields.get("Source", "").strip(),
                )
            )

    grand_total = sum(totals.values())
    print("opcode,executed,percent,static_sites")
    for opcode, count in sorted(
        totals.items(), key=lambda item: (-item[1], item[0])
    )[: args.top]:
        percent = 100.0 * count / grand_total if grand_total else 0.0
        print(f"{opcode},{count},{percent:.3f},{static_sites[opcode]}")
    print(f"TOTAL,{grand_total},100.000,{sum(static_sites.values())}")
    if args.top_sites:
        print()
        print("site_count,address,source")
        for count, address, source in sorted(sites, reverse=True)[
            : args.top_sites
        ]:
            print(f"{count},{address},{source}")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
