#!/usr/bin/env python3
"""Time real prepared GDN/KDA prefill transactions over the native case corpus.

This is a workload benchmark, not an offline configuration generator.  Every
case is declared, prepared, bound, and then replayed through the production
prefill operation; mutable recurrent state is restored outside timing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.testing.delta_prefill_cases import (
    GDN_PREFILL_CASES,
    KDA_PREFILL_CASES,
    check_binding,
    make_inputs,
    oracle,
    prepared_binding,
    run_binding,
)
from benchmarks.common import require_sm120


def _cases(recipe: str, names: str):
    available = GDN_PREFILL_CASES if recipe == "gdn" else KDA_PREFILL_CASES
    by_name = {case.name: case for case in available}
    selected = tuple(by_name) if names == "all" else tuple(part.strip() for part in names.split(",") if part.strip())
    if not selected or len(set(selected)) != len(selected) or set(selected) - by_name.keys():
        raise ValueError(f"unknown or duplicate cases; available: {sorted(by_name)}")
    return tuple(by_name[name] for name in selected)


def _time(case, binding, initial_state, *, warmup: int, iterations: int):
    for _ in range(warmup):
        binding.recurrent_state.copy_(initial_state)
        run_binding(case.recipe, binding)
    torch.cuda.synchronize(binding.recurrent_state.device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        binding.recurrent_state.copy_(initial_state)
        start.record()
        run_binding(case.recipe, binding)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("gdn", "kda"), default="gdn")
    parser.add_argument("--cases", default="all", help="comma-separated PrefillCase names")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    device = require_sm120()
    cases = _cases(args.recipe, args.cases)
    report = {"recipe": args.recipe, "cases": [], "timed_operation": "prepared production prefill"}
    for index, case in enumerate(cases):
        tensors = make_inputs(case, device=device, seed=args.seed + index)
        initial_state = tensors["recurrent_state"].clone()
        expected_output, expected_state = oracle(case, tensors)
        with prepared_binding(case, tensors, checkpoint_export=args.checkpoint) as binding:
            samples = _time(case, binding, initial_state, warmup=args.warmup, iterations=args.iterations)
            metrics = check_binding(case, binding, expected_output, expected_state, initial_state)
            binding.scratch.fill_(0xFF)
            graph = torch.cuda.CUDAGraph()
            with kernel_resolution_guard("delta prefill benchmark graph capture"):
                with torch.cuda.graph(graph):
                    run_binding(case.recipe, binding)
            binding.recurrent_state.copy_(initial_state)
            graph.replay()
            torch.cuda.synchronize(device)
            metrics = check_binding(case, binding, expected_output, expected_state, initial_state)
        row = {"case": case.name, "samples_us": samples, "median_us": statistics.median(samples), "correctness": metrics}
        report["cases"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    raise SystemExit(main())
