"""Trace the unchanged frozen suite with worker-local frozen kernel resolution.

Use CUDA mode under Nsight, or worker-local torch/CUPTI mode.
Callable worker RPC requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 in this trusted,
local offline process only; do not enable it on an externally exposed server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import verify


def worker_snapshot(worker, label, arm=False, synchronize=False):
    import dataclasses
    import torch
    import b12x
    from b12x._lib.compiler import compile_cache_info
    from vllm.compilation.counter import compilation_counter
    from vllm.utils.jit_monitor import is_active
    from b12x.sequence.engram import DiskTable

    if arm:
        if not is_active():
            raise RuntimeError(
                "JIT monitor must be active before serving qualification"
            )
        os.environ["B12X_ENGINE_STARTED"] = "1"
        os.environ["B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START"] = "1"
        b12x.freeze_kernel_resolution("frozen V4.1 serving qualification")
    if synchronize:
        torch.cuda.synchronize()
    with torch.profiler.record_function(label):
        torch.cuda.nvtx.mark(label)
    draft = worker.get_draft_model()
    disk_tables = {}
    for name, module in worker.get_model().named_modules():
        table = getattr(module, "disk_table", None)
        if isinstance(table, DiskTable):
            disk_tables[name] = {
                "backend": type(table._cache).__module__
                + "."
                + type(table._cache).__qualname__,
                "table_rows": table.plan.table_rows,
                "shard_start": table.plan.shard_start,
                "shard_end": table.plan.shard_end,
                "resident_weight_parameter": module.weight is not None,
                "resident_scale_parameter": module.weight_scale_inv is not None,
                "stats": table.stats(),
            }
    return {
        "rank": worker.rank,
        "local_rank": worker.local_rank,
        "pid": os.getpid(),
        "device": str(worker.device),
        "runner": type(worker.model_runner).__qualname__,
        "model": type(worker.get_model()).__module__
        + "."
        + type(worker.get_model()).__qualname__,
        "draft": None
        if draft is None
        else type(draft).__module__ + "." + type(draft).__qualname__,
        "cudagraph_mode": str(worker.compilation_config.cudagraph_mode),
        "jit_monitor_active": is_active(),
        "b12x_frozen": b12x.kernel_resolution_frozen(),
        "profiler_running": worker.profiler is not None and worker.profiler.is_running,
        "b12x_compile": compile_cache_info(),
        "vllm_compile": dataclasses.asdict(compilation_counter),
        "engram_disk_tables": disk_tables,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", default=str(Path(__file__).with_name("quality_suite.json"))
    )
    parser.add_argument(
        "--engine", default=str(Path(__file__).with_name("engine_tp4_ssd.json"))
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profiler", choices=("cuda", "torch"), default="cuda")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    suite, _ = verify._checked_suite(args.suite)
    profiler_config = {"profiler": args.profiler}
    if args.profiler == "torch":
        profiler_config.update(
            torch_profiler_dir=str(out / "torch"),
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False,
            torch_profiler_with_memory=False,
            torch_profiler_with_flops=False,
            torch_profiler_dump_cuda_time_total=False,
            torch_profiler_use_gzip=True,
            ignore_frontend=True,
        )
    audit = {
        "suite_sha256": hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
        "instrumentation": {
            "profiler_config": profiler_config,
            "jit_monitor_mode": "error",
            "jit_monitor_verbose": False,
            "freeze_after_init": True,
        },
        "boundaries": [],
        "completed": False,
    }
    live = []
    import vllm

    base_llm = vllm.LLM

    class TracedLLM(base_llm):
        def __init__(self, **kwargs):
            kwargs.update(
                {
                    "profiler_config": profiler_config,
                    "jit_monitor_mode": "error",
                    "jit_monitor_verbose": False,
                }
            )
            super().__init__(**kwargs)
            self.trace_batch = 0
            live.append(self)
            audit["post_init"] = self.collective_rpc(
                worker_snapshot, args=("dsv41/post_init", True, True)
            )
            self.start_profile(f"dsv41_tp{kwargs['tensor_parallel_size']}")
            started = self.collective_rpc(worker_snapshot, args=("dsv41/trace_begin",))
            audit["trace_begin"] = started
            expected_ranks = kwargs["tensor_parallel_size"]
            if len(started) != expected_ranks or not all(
                r["profiler_running"] for r in started
            ):
                raise RuntimeError("Profiler did not start on every worker")

        def generate(self, *pargs, **kwargs):
            index = self.trace_batch
            self.trace_batch += 1
            batch_count = len(suite["batches"])
            label = f"dsv41/repeat={index // batch_count}/batch={suite['batches'][index % batch_count]['id']}"
            row = {
                "label": label,
                "begin": self.collective_rpc(worker_snapshot, args=(label + "/begin",)),
            }
            audit["boundaries"].append(row)
            try:
                return super().generate(*pargs, **kwargs)
            finally:
                row["end"] = self.collective_rpc(
                    worker_snapshot, args=(label + "/end",)
                )

    vllm.LLM = TracedLLM
    try:
        verify.run_vllm(
            argparse.Namespace(
                suite=args.suite,
                engine=args.engine,
                repeats=args.repeats,
                output=str(out / "vllm_results.json"),
            )
        )
        audit["completed"] = True
    finally:
        try:
            for llm in live:
                try:
                    audit["trace_end"] = llm.collective_rpc(
                        worker_snapshot, args=("dsv41/trace_end", False, True)
                    )
                finally:
                    llm.stop_profile()
        finally:
            vllm.LLM = base_llm
            (out / "instrumentation.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
