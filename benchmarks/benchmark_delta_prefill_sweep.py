#!/usr/bin/env python3
"""Run source-bound, resumable configuration races for delta-rule prefill."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x.policy.device import detect_device
from b12x.policy.generation import CheckpointStore, GenerationContext, GenerationSettings
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.delta_prefill import (
    GdnPrefillGenerator, KdaPrefillGenerator, prefill_candidates, prefill_cases,
)
from benchmarks._gdn_prefill_race import _source, _versions
from benchmarks.benchmark_gdn_decode import _device_provenance, _git_provenance
from benchmarks.common import nvidia_smi_gpu_mode_snapshot, require_sm120


def source_manifest(recipe: str) -> list[dict[str, str]]:
    sources = {Path(__file__).resolve()}
    for directory in (
        ROOT / "b12x/sequence/_shared/delta_prefill",
        ROOT / f"b12x/sequence/{recipe}_prefill",
        ROOT / "b12x/_lib",
    ):
        sources.update(directory.glob("*.py"))
    sources.update((
        ROOT / "b12x/policy/generation/providers/delta_prefill.py",
        ROOT / "b12x/policy/generation/delta_prefill_cases.py",
        ROOT / "b12x/policy/generation/sweep.py",
    ))
    return [_source(path) for path in sorted(sources)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("gdn", "kda"), default="gdn")
    parser.add_argument("--cases", default="all", help="comma-separated PrefillCase names")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cases = tuple(case for case in prefill_cases(args.recipe)
                  if bool(case.query["checkpoint_export"]) == args.checkpoint)
    if args.cases != "all":
        names = args.cases.split(",")
        selected = {case.case_id.rsplit("-", 1)[0]: case for case in cases}
        if len(names) != len(set(names)) or set(names) - selected.keys():
            parser.error(f"unknown or duplicate cases; available: {sorted(selected)}")
        cases = tuple(selected[name] for name in names)
    device = require_sm120()
    detected = detect_device(device)
    if detected.identity is None or detected.ordinal is None:
        parser.error("a physical CUDA device is required")
    settings = GenerationSettings(warmup=args.warmup, groups=1,
                                  repetitions=args.iterations, cold_l2=not args.warm_l2)
    sources = source_manifest(args.recipe)
    identity = {"sources": sources, "toolchain": _versions(),
                "device": _device_provenance(device), "settings": settings.to_dict()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    work_dir = args.work_dir.resolve() / fingerprint
    context = GenerationContext(device=detected.identity, device_ordinal=detected.ordinal,
                                work_dir=work_dir, source_revision=_git_provenance()["commit"],
                                settings=settings)
    cls = GdnPrefillGenerator if args.recipe == "gdn" else KdaPrefillGenerator
    generator = cls(cases=cases)
    estimate = {"cases": len(cases), "candidates": sum(len(prefill_candidates(c)) for c in cases),
                "per_case": {c.case_id: len(prefill_candidates(c)) for c in cases},
                "work_dir": str(work_dir)}
    print(json.dumps(estimate, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    work_dir.mkdir(parents=True, exist_ok=True)
    before = nvidia_smi_gpu_mode_snapshot()
    result = generator.generate(context, progress=NullProgressReporter(),
                                checkpoints=CheckpointStore(work_dir / "checkpoints"))
    if source_manifest(args.recipe) != sources:
        raise RuntimeError("prefill source changed during measurement; results cannot be published")
    report = {"provenance": identity, "git": _git_provenance(), "estimate": estimate,
              "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
              "gpu_mode_before": before, "gpu_mode_after": nvidia_smi_gpu_mode_snapshot(),
              "component": dict(result.component), "evidence": dict(result.evidence),
              "metric_direction": "lower latency_us is better"}
    report_path = work_dir / f"result-{time.time_ns()}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(str(report_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
