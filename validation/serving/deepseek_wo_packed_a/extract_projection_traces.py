#!/usr/bin/env python3
"""Identify attention-output projections by stream order in Torch CUDA traces.

WO-A consumes inverse-RoPE quantization and WO-B consumes WO-A. Both can have
the same generated kernel class name; the name is not a projection identity.
Report costs and launch geometry without treating profiled time as throughput.
"""

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path, include_calls=False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    streams = collections.defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel" and "dur" in event:
            args = event["args"]
            streams[(args["device"], args["context"], args["stream"])].append(event)
    roles = {"first_projection_wo_a": [], "second_projection_wo_b": []}
    samples = []
    for stream_key, kernels in streams.items():
        kernels.sort(key=lambda event: event["ts"])
        for index, event in enumerate(kernels):
            if event["name"] != "_quantize_attention_inv_rope_to_tdg_kernel":
                continue
            following = kernels[index + 1 : index + 7]
            dense = []
            for item in following:
                if (
                    "OneshotLaunch" in item["name"]
                    or "_quantize_attention_inv_rope" in item["name"]
                ):
                    break
                if "dense_gemmDenseGemmKernel" in item["name"]:
                    dense.append(item)
            if len(dense) != 2:
                raise ValueError(
                    f"Expected two ordered projections after quantization at {event['ts']}"
                )
            for role, kernel in zip(roles, dense, strict=True):
                roles[role].append(kernel)
            if len(samples) < 4:
                samples.append(
                    {
                        "stream": stream_key,
                        "quantization_ts_us": event["ts"],
                        "ordered_kernels": [
                            {
                                "name": item["name"],
                                "ts_us": item["ts"],
                                "duration_us": item["dur"],
                                "block": item["args"]["block"],
                                "grid": item["args"]["grid"],
                            }
                            for item in dense
                        ],
                    }
                )
    report = {}
    for role, kernels in roles.items():
        if not kernels:
            raise ValueError("Trace has no ordered inverse-RoPE WO projection pair")
        geometries = collections.Counter(
            (
                tuple(item["args"]["block"]),
                tuple(item["args"]["grid"]),
                item["args"].get("registers per thread"),
                item["args"].get("shared memory"),
            )
            for item in kernels
        )
        durations = [item["dur"] for item in kernels]
        report[role] = {
            "count": len(kernels),
            "mean_us": statistics.mean(durations),
            "median_us": statistics.median(durations),
            "sum_us": sum(durations),
            "launches": [
                {
                    "block": key[0],
                    "grid": key[1],
                    "registers_per_thread": key[2],
                    "shared_memory_bytes": key[3],
                    "count": count,
                }
                for key, count in geometries.items()
            ],
        }
        if include_calls:
            report[role]["call_fields"] = ["timestamp_us", "duration_us"]
            report[role]["calls"] = [[item["ts"], item["dur"]] for item in kernels]
    return {
        "trace": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "projections": report,
        "ordered_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", action="append", required=True, help="label=trace.json[.gz]"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-calls",
        action="store_true",
        help="Retain every ordered projection sample for review",
    )
    parser.add_argument(
        "--serving-provenance",
        action="store_true",
        help="Attach image, hardware and launch records adjacent to each capture",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    traces = dict(value.split("=", 1) for value in args.trace)
    report = {
        "method": "Two dense launches following inverse-RoPE quantization on the same CUDA stream.",
        "limit": "Profile costs identify components; use unprofiled windows for speed claims.",
        "arms": {
            label: summarize(Path(path), args.include_calls)
            for label, path in traces.items()
        },
    }
    if args.serving_provenance:
        for label, path in traces.items():
            directory = Path(path).parents[3]
            launch = json.loads((directory / "launch.json").read_text())
            runtime = json.loads((directory / "runtime.json").read_text())[0]
            report["arms"][label]["serving"] = {
                key: launch[key]
                for key in (
                    "image",
                    "image_id",
                    "container_id",
                    "gpus",
                    "repository",
                    "checkpoint_revision",
                    "memory_offset",
                    "graphics_offset",
                    "gpu_identity_before",
                    "command",
                )
            }
            report["arms"][label]["serving"].update(
                native_libraries_sha256=runtime["libraries_sha256"],
                source_receipt_sha256=hashlib.sha256(
                    (directory / "launch.json").read_bytes()
                ).hexdigest(),
            )
            qualification = json.loads((directory / "qualification.json").read_text())
            report["arms"][label]["serving"]["decode_windows"] = []
            for command in qualification["commands"]:
                if not command["label"].startswith("decode"):
                    continue
                argv = command["argv"]
                raw_path = Path(argv[argv.index("--output") + 1])
                raw = json.loads(raw_path.read_text())
                report["arms"][label]["serving"]["decode_windows"].append(
                    {
                        "command": command,
                        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        "cells": [
                            {
                                key: cell.get(key)
                                for key in (
                                    "concurrency",
                                    "context_tokens",
                                    "aggregate_tps",
                                    "server_steps_per_s",
                                    "server_spec_accept_length",
                                    "failure_reason",
                                    "underfilled",
                                    "capacity_limited",
                                )
                            }
                            for cell in raw["results"]
                        ],
                    }
                )
        report["ratio_direction"] = (
            "Time ratios are candidate duration / comparison duration; these are profiled diagnostics, not serving throughput ratios."
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                label: {
                    role: {
                        key: value for key, value in values.items() if key != "calls"
                    }
                    for role, values in arm["projections"].items()
                }
                for label, arm in report["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
