"""Summarize an isolated MoE workspace snapshot containing per-CTA timestamps.

Save the int32 workspace after synchronizing its execution stream:
``workspace.cpu().numpy().tofile(path)``. Pass the actual launch grid size;
unused rows in a reduced grid are not valid samples. Times are nanoseconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct


def summarize(data: bytes, *, sms: int, ctas: int) -> dict:
    if not 0 < ctas <= sms:
        raise ValueError("actual CTA count must be positive and no greater than SMS")
    offset_bytes = ((4 * sms + 5) // 4 * 4) * 4
    required = offset_bytes + ctas * 80
    if len(data) < required:
        raise ValueError(f"workspace needs at least {required} bytes")
    rows = [
        struct.unpack_from("<10Q", data, offset_bytes + i * 80) for i in range(ctas)
    ]
    for index, row in enumerate(rows):
        if row[0] == 0 or any(a > b for a, b in zip(row[:7], row[1:8], strict=True)):
            raise ValueError(f"CTA {index} has missing or non-monotonic timestamps")
        if row[8] > row[3] - row[2]:
            raise ValueError(f"CTA {index} lock polling exceeds its FC1 phase")

    def stats(values):
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    phases = {
        name: stats([row[end] - row[start] for row in rows])
        for name, start, end in (
            ("rotation", 0, 1),
            ("fc1_including_lock_polling", 2, 3),
            ("activation", 4, 5),
            ("fc2", 6, 7),
        )
    }
    barriers = {}
    for name, arrive, release in (
        ("rotation", 1, 2),
        ("fc1", 3, 4),
        ("activation", 5, 6),
    ):
        arrivals = [row[arrive] for row in rows]
        releases = [row[release] for row in rows]
        barriers[name] = {
            "arrival_spread_ns": max(arrivals) - min(arrivals),
            "last_release_after_last_arrival_ns": max(releases) - max(arrivals),
            "per_cta_barrier_ns": stats(
                [r - a for a, r in zip(arrivals, releases, strict=True)]
            ),
        }
    return {
        "schema": "b12x.w4a16.phase_profile.v1",
        "sms": sms,
        "ctas": ctas,
        "workspace_sha256": hashlib.sha256(data).hexdigest(),
        "body_envelope_ns": max(row[7] for row in rows) - min(row[0] for row in rows),
        "phase_per_cta_ns": phases,
        "barriers": barriers,
        "fc1_lock_poll_per_cta_ns": stats([row[8] for row in rows]),
        "fc1_lock_poll_calls_per_cta": stats([row[9] for row in rows]),
        "records": [list(row) for row in rows],
        "limitations": (
            "Instrumented single-launch diagnostics, excluding LUT staging and route sum. "
            "Barrier durations exclude surrounding CTA syncs. Lock polling includes the "
            "first successful load. Per-CTA durations overlap and must not be summed "
            "into step latency. Compare timings with an uninstrumented control."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--sms", type=int, required=True)
    parser.add_argument("--ctas", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.workspace.read_bytes(), sms=args.sms, ctas=args.ctas)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
