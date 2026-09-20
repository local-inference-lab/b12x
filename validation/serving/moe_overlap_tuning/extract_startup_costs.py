"""Attribute cumulative preparation timing without double-counting requests.

Preparation requests interleave. Their start/end counter differences overlap;
only successive counter increments can be attributed to individual races.
These host timing counters are not kernel latency or total server startup time.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def summarize(path):
    """Attribute each counter increment once, to its completed tuning batch."""
    phases = []
    phase = None
    queries = {}
    previous = 0.0
    groups = defaultdict(float)
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "begin":
            if phase is not None:
                raise ValueError(f"Unfinished preparation phase in {path}")
            phase = {"rank": event["rank"], "measurement_s": 0.0}
            previous = 0.0
            queries = {}
            groups = defaultdict(float)
        if phase is None:
            raise ValueError(f"Event outside a preparation phase in {path}")
        if event["event"] == "request_begin":
            queries[event["request"]] = (
                event["component"], event.get("query", {}).get("num_tokens"))
        measured = event.get("seconds", {}).get("autotuning", previous)
        delta = measured - previous
        if delta < -1e-9:
            raise ValueError(f"Nonmonotonic preparation counter in {path}")
        if delta > 0:
            if event["event"] != "batch_end" or event["request"] not in queries:
                raise ValueError(f"Unattributable measurement increment in {path}")
            groups[queries[event["request"]]] += delta
            phase["measurement_s"] += delta
        previous = measured
        if event["event"] == "complete":
            phase.update({
                "rank": event["rank"],
                "elapsed_s": event["elapsed_s"],
                "failed": event.get("failed"),
                "cumulative_seconds": event["seconds"],
                "small_moe_measurement_s": sum(
                    seconds for (component, tokens), seconds in groups.items()
                    if component == "moe.decode" and tokens is not None
                    and 2 <= tokens <= 8),
                "races": [{"component": component, "tokens": tokens,
                           "measurement_s": seconds}
                          for (component, tokens), seconds in sorted(
                              groups.items(), key=lambda pair: str(pair[0]))],
            })
            phases.append(phase)
            phase = None
    if phase is not None:
        raise ValueError(f"Incomplete preparation trace in {path}")
    return {"trace": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "phases": phases}


def main():
    """Export identified preparation counters without overwriting evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": __doc__, "traces": [summarize(p) for p in args.trace]}
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    for trace in result["traces"]:
        print(json.dumps({"trace": trace["trace"],
            "phase_count": len(trace["phases"]), "measured_phases": [{
            key: phase[key] for key in ("rank", "elapsed_s", "measurement_s",
                                        "small_moe_measurement_s")}
            for phase in trace["phases"] if phase["measurement_s"]]}))


if __name__ == "__main__":
    main()
