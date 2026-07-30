#!/usr/bin/env python3
"""Tabulate the headline metrics from `ncu --page details --csv` exports.

`scripts/ncu_profile_fp6_gemm.sh` writes one details CSV per shard. Each CSV is
one row per metric with the full kernel name repeated on every row, which is
unreadable directly. This pulls the metrics that decide a memory-bound GEMM's
fate into one row per report, plus the warp stall breakdown.

Usage::

    python scripts/summarize_ncu_details.py /tmp/fp6_ncu_*/
    python scripts/summarize_ncu_details.py a.details.csv b.details.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

# (column label, metric name as ncu spells it)
HEADLINE: tuple[tuple[str, str], ...] = (
    ("dur_us", "Duration [us]"),
    ("dram_%", "DRAM Throughput"),
    ("sm_%", "Compute (SM) Throughput"),
    ("mem_TB/s", "Memory Throughput"),
    ("occ_%", "Achieved Occupancy"),
    ("theo_occ_%", "Theoretical Occupancy"),
    ("grid", "Grid Size"),
    ("block", "Block Size"),
    ("waves/SM", "Waves Per SM"),
    ("regs", "Registers Per Thread"),
    ("smem_KB", "Dynamic Shared Memory Per Block [KB]"),
    ("lim_smem", "Block Limit Shared Mem"),
    ("lim_regs", "Block Limit Registers"),
    ("lim_warps", "Block Limit Warps"),
    ("L2_hit_%", "L2 Hit Rate"),
    ("L1_hit_%", "L1/TEX Hit Rate"),
)


_STALL_PREFIX = "smsp__average_warps_issue_stalled_"
_STALL_SUFFIX = "_per_issue_active.ratio"

# (N, K) per Behemoth TP=2 shard, mirroring scripts/_behemoth_tp2_shapes.sh
# (which this file cannot source) and BEHEMOTH_TP2_SHAPES in the benchmark.
# Used only to turn a BYTES=1 capture into a read-amplification ratio, which
# needs the shard's value count.
_SHARD_NK: dict[str, tuple[int, int]] = {
    "qkv": (7168, 12288),
    "o": (12288, 6144),
    "gate_up": (28672, 12288),
    "down": (12288, 14336),
}


def _weight_bytes(report_name: str) -> tuple[str, float] | None:
    """Theoretical DRAM bytes for one pass over a shard's weights.

    Report names look like ``decode_fp6_gate_up_m1``. Returns (arm, bytes) or
    None when the name does not identify a known shard/arm, in which case the
    amplification column is left blank rather than guessed.
    """
    stem = report_name
    # Longest first: "o" is a substring of nothing here, but keeping the order
    # explicit avoids a future shard name shadowing another.
    shard = next(
        (s for s in sorted(_SHARD_NK, key=len, reverse=True) if f"_{s}_" in stem),
        None,
    )
    if shard is None:
        return None
    n, k = _SHARD_NK[shard]
    values = float(n) * float(k)
    if "_fp8_" in stem:
        # FP8 block-scaled: 1 byte per value plus a per-128x128-block scale,
        # which rounds to nothing at these sizes but is included for honesty.
        return "fp8", values + values / (128.0 * 128.0) * 4.0
    if "_fp6_" in stem:
        # Packed MX-FP6: 6 bits per value plus one UE8M0 byte per 32 values.
        return "fp6", values * 6.0 / 8.0 + values / 32.0
    return None


def _read(path: pathlib.Path) -> tuple[dict[str, str], list[tuple[str, float]]]:
    """Return (metric -> value, sorted stall reasons)."""
    metrics: dict[str, str] = {}
    stalls: list[tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Metric Name") or "").strip()
            value = (row.get("Metric Value") or "").strip()
            if not name:
                continue
            # Duplicate metric names appear in several sections (e.g. "Memory
            # Throughput" in both SOL and Memory Workload) with different units;
            # keep the first, which is the SOL percentage form.
            metrics.setdefault(name, value)
            # ncu picks the unit per magnitude, so a slow kernel reports
            # Gbyte/s where a fast one reports Tbyte/s. Normalize, or the
            # column silently blanks out on exactly the regressions we care
            # about most.
            unit = (row.get("Metric Unit") or "").strip()
            # Same magnitude-dependent unit choice bites Duration: a decode
            # kernel reports usecond, a prefill kernel at the same shape family
            # reports msecond, and reading the raw number makes a 6.5 ms GEMM
            # look like 6.5 us.
            if name == "Duration":
                scale = {
                    "nsecond": 1e-3,
                    "ns": 1e-3,
                    "usecond": 1.0,
                    "us": 1.0,
                    "msecond": 1e3,
                    "ms": 1e3,
                    "second": 1e6,
                    "s": 1e6,
                }.get(unit)
                if scale is not None:
                    try:
                        metrics["Duration [us]"] = (
                            f"{float(value.replace(',', '')) * scale:.2f}"
                        )
                    except ValueError:
                        pass
            # And again for shared memory: ncu reports byte for a small kernel
            # and Kbyte for a large one, so an unnormalized column labelled KB
            # mixes magnitudes on exactly the comparison smem work depends on.
            if name == "Dynamic Shared Memory Per Block":
                scale = {
                    "byte": 1.0 / 1024.0,
                    "Kbyte": 1.0,
                    "Mbyte": 1024.0,
                }.get(unit)
                if scale is not None:
                    try:
                        metrics["Dynamic Shared Memory Per Block [KB]"] = (
                            f"{float(value.replace(',', '')) * scale:.2f}"
                        )
                    except ValueError:
                        pass
            if name == "Memory Throughput" and unit in ("Tbyte/s", "Gbyte/s"):
                try:
                    tb = float(value.replace(",", ""))
                except ValueError:
                    continue
                if unit == "Gbyte/s":
                    tb /= 1000.0
                metrics["Memory Throughput [TB/s]"] = f"{tb:.2f}"
            # Two spellings reach this file. The WarpStateStats SECTION emits
            # "Stall Long Scoreboard"; an explicit --metrics request (what
            # STALLS=1 in ncu_profile_fp6_gemm.sh does, because the section
            # alone omits them from the details page) emits the raw counter
            # name. Accept both or STALLS=1 silently produces no stall table.
            reason = None
            if name.startswith("Stall "):
                reason = name[6:]
            elif name.startswith(_STALL_PREFIX) and name.endswith(_STALL_SUFFIX):
                reason = name[len(_STALL_PREFIX) : -len(_STALL_SUFFIX)]
            if reason is not None:
                try:
                    stalls.append((reason, float(value.replace(",", ""))))
                except ValueError:
                    pass
    stalls.sort(key=lambda kv: kv[1], reverse=True)
    return metrics, stalls


def _num(metrics: dict[str, str], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _dram_read_bytes(metrics: dict[str, str]) -> float | None:
    """Bytes fetched from DRAM, by counter if available and by L2 miss if not.

    The `dram__*` counters report "n/a" on sm_120 workstation parts, so the
    direct read is unavailable exactly where we need it. L2 read sectors times
    the sector miss rate times 32 reconstructs it: every read sector that misses
    L2 is a DRAM fetch. Validated against the SOL "Memory Throughput" figure on
    the Jul 28 decode capture, where the two agree to within 4%.
    """
    direct = _num(metrics, "dram__bytes_read.sum")
    if direct is not None:
        return direct
    sectors = _num(metrics, "lts__t_sectors_srcunit_tex_op_read.sum")
    hit_pct = _num(metrics, "lts__t_sector_hit_rate.pct")
    if sectors is None or hit_pct is None:
        return None
    return sectors * (1.0 - hit_pct / 100.0) * 32.0


def _print_bytes(files: list[pathlib.Path]) -> None:
    """Read-amplification table, printed only for BYTES=1 captures."""
    rows: list[tuple[str, float, float, float, str]] = []
    for f in files:
        metrics, _ = _read(f)
        name = f.name.replace(".details.csv", "")
        expected = _weight_bytes(name)
        if expected is None:
            continue
        actual = _dram_read_bytes(metrics)
        if actual is None:
            continue
        _, want = expected
        dur = _num(metrics, "Duration [us]")
        tbs = f"{actual / dur / 1e6:.2f}" if dur else "-"
        rows.append((name, actual / 1e6, want / 1e6, actual / want, tbs))
    if not rows:
        return
    width = max(len(r[0]) for r in rows)
    print(f"\n{'report':{width}s} {'read_MB':>10s} {'weight_MB':>10s} "
          f"{'amplif':>8s} {'TB/s':>8s}")
    for name, got, want, ratio, tbs in rows:
        print(f"{name:{width}s} {got:10.1f} {want:10.1f} {ratio:8.3f} {tbs:>8s}")
    print("\namplif = DRAM bytes read / bytes the shard's packed weights occupy.")
    print("1.00 means the layout is clean and the roofline is the hardware.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--stalls", type=int, default=6, help="Top stall reasons")
    args = ap.parse_args()

    files: list[pathlib.Path] = []
    for raw in args.paths:
        p = pathlib.Path(raw)
        files.extend(sorted(p.rglob("*.details.csv")) if p.is_dir() else [p])
    if not files:
        sys.exit("no *.details.csv found")

    labels = [lbl for lbl, _ in HEADLINE]
    width = max(len(f.name.replace(".details.csv", "")) for f in files)
    print(f"{'report':{width}s} " + " ".join(f"{lbl:>10s}" for lbl in labels))
    all_stalls: list[tuple[str, list[tuple[str, float]]]] = []
    for f in files:
        metrics, stalls = _read(f)
        cells = []
        for _, metric in HEADLINE:
            key = (
                "Memory Throughput [TB/s]"
                if metric == "Memory Throughput"
                else metric
            )
            cells.append(f"{metrics.get(key, '-'):>10s}")
        print(f"{f.name.replace('.details.csv', ''):{width}s} " + " ".join(cells))
        all_stalls.append((f.name.replace(".details.csv", ""), stalls))

    _print_bytes(files)

    for name, stalls in all_stalls:
        if not stalls:
            continue
        print(f"\n{name} - top warp stall reasons (cycles/instruction):")
        for reason, val in stalls[: args.stalls]:
            print(f"  {val:8.2f}  {reason}")


if __name__ == "__main__":
    main()
