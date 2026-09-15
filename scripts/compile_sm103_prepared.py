#!/usr/bin/env python3
"""Compile session preparation declarations for SM103 without a CUDA context."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compile_case(case):
    from b12x._lib.compile_pool import describe_compilation, _run_job
    from b12x._lib.compile_plan import evict_planning_artifacts
    from b12x.preparation.types import _plan_scope
    from scripts._sm103_preparation_corpus import declare, IDENTITY, DEVICE
    import torch

    started = time.monotonic()
    plan = declare(case)
    selected = plan.contract.configure(plan.query, device=IDENTITY, search=False).default
    with _plan_scope(plan):
        memory = plan._memory_requirements(selected, DEVICE)
    descriptions = tuple(describe_compilation(job) for job in plan._compile_jobs(selected, DEVICE))
    required = frozenset(program for item in descriptions for program in item.programs)
    evict_planning_artifacts(required)
    from b12x._lib import compiler
    original_compile = compiler._call_cute_compile

    def capture(compile_callable, func, args, kwargs, *, compile_spec, cache_key):
        directory = Path(os.environ["B12X_SM103_NATIVE_DUMP"]) / cache_key
        directory.mkdir(parents=True, exist_ok=True)
        options = kwargs.get("options", "")
        if not isinstance(options, str):
            raise TypeError("native artifact export requires string compiler options")
        compiled = original_compile(
            compile_callable, func, args,
            {**kwargs, "options": options + f" --keep-ptx --keep-cubin --dump-dir={directory}"},
            compile_spec=compile_spec, cache_key=cache_key,
        )
        files = tuple(directory.glob("*.ptx")) + tuple(directory.glob("*.cubin"))
        if len(files) != 2 or not any(".target sm_103a" in p.read_text() for p in files if p.suffix == ".ptx"):
            raise RuntimeError("native export did not produce one SM103 PTX/cubin pair")
        (directory / "metadata.json").write_text(json.dumps({
            "program_key": cache_key, "name": compile_spec.kernel_id if compile_spec else None,
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        }, indent=2) + "\n")
        return compiled

    with patch.object(compiler, "_call_cute_compile", capture):
        for item in descriptions:
            _run_job(pickle.dumps(item.job), item.programs)
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compilation initialized CUDA")
    return dict(case=case, status="compiled", component=plan.component_id,
                config=dict(plan.contract.config_payload(selected)),
                scratch_bytes=memory.scratch_nbytes,
                programs=[asdict(p) for p in sorted(required, key=lambda p: (p.dialect, p.key))],
                elapsed_seconds=time.monotonic() - started, cuda_initialized=False)


def main():
    from scripts._sm103_preparation_corpus import CASES, DEVICE, IDENTITY
    from scripts._sm103_source import package_source_sha256, source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["B12X_COMPILE_CACHE_DIR"] = str(output / "compile-cache")
    os.environ["B12X_SM103_NATIVE_DUMP"] = str(output / "native")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from b12x._lib.compile_pool import _initialize_worker, _offline_compiler_spawn_environment
    cases = args.case or CASES
    manifest = dict(source=source_identity(ROOT), source_sha256=package_source_sha256(ROOT),
                    command=sys.argv, target=asdict(IDENTITY), cases=list(cases),
                    status="running", qualification="offline compilation only",
                    toolchain={name: importlib.metadata.version(name) for name in
                               ("torch", "nvidia-cutlass-dsl", "triton")})
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    context = multiprocessing.get_context("spawn")
    activity = context.Array("q", (0, 0))
    failures = 0
    with _offline_compiler_spawn_environment():
        pool = context.Pool(args.workers, initializer=_initialize_worker,
                            initargs=(0, (10, 3), DEVICE.uuid, IDENTITY.product_name, 148,
                                      DEVICE.max_shared_memory_per_block,
                                      DEVICE.max_shared_memory_per_multiprocessor, activity))
    try:
        pending = [(case, pool.apply_async(compile_case, (case,))) for case in cases]
        with (output / "cases.jsonl").open("w") as records:
            for case, job in pending:
                try:
                    result = job.get()
                except Exception:
                    failures += 1
                    result = dict(case=case, status="failed", error=traceback.format_exc())
                records.write(json.dumps(result) + "\n")
                records.flush()
                print(f"{case}: {result['status']}", flush=True)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    unchanged = package_source_sha256(ROOT) == manifest["source_sha256"]
    required_cute = {
        program["key"]
        for line in (output / "cases.jsonl").read_text().splitlines()
        for program in json.loads(line).get("programs", ())
        if program["dialect"] == "cute"
    }
    exported = {p.parent.name for p in (output / "native").glob("*/metadata.json")}
    missing = sorted(required_cute - exported)
    manifest.update(status="passed" if not failures and unchanged and not missing else "failed",
                    failures=failures, source_unchanged=unchanged,
                    required_cute_programs=len(required_cute), native_exports=len(exported),
                    missing_native_exports=missing)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return int(failures > 0 or not unchanged or bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
