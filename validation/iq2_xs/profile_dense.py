"""Profile checkpoint dense graphs using launch choices from benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.checkpoint_dense import (
    checkpoint_cases,
    check_output,
    load_weight,
    oracle,
)
from benchmarks.common import make_l2_flush_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1])
    parser.add_argument("--recipe", choices=("iq2_xs", "nvfp4"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = [json.loads(line) for line in args.evidence.read_text().splitlines()]
    selected = {
        (row["weight"], row["rows"]): row
        for row in records
        if row["kind"] == "case" and row["rows"] in args.rows
    }
    index, cases = checkpoint_cases(args.model_path)
    flush = make_l2_flush_fn(enabled=True)
    for case in cases:
        if args.recipe is not None and case["recipe"] != args.recipe:
            continue
        weight, decoded, multiplier, hashes = load_weight(args.model_path, index, case)
        for rows in args.rows:
            record = selected[case["weight"], rows]
            if hashes != record["payload_sha256"]:
                raise ValueError("checkpoint payload differs from benchmark evidence")
            torch.manual_seed(42 + rows)
            source = (
                torch.randn((rows, case["k"]), device="cuda", dtype=torch.bfloat16)
                * 0.25
            )
            expected = oracle(source, decoded, multiplier)
            plan = blockscaled.plan(
                blockscaled.BlockscaledQuery(**record["query"]),
                override=blockscaled.BlockscaledConfig(**record["config"]),
            )
            values = weight.values
            scales = weight.metadata if case["recipe"] == "iq2_xs" else weight.scale_mma
            global_scale = None if case["recipe"] == "iq2_xs" else weight.global_scale

            def prepare(state):
                scratch = (
                    torch.empty(state.required_workspace, device="cuda", dtype=torch.uint8)
                    if state.required_workspace
                    else None
                )
                return PreparedCall(
                    run=lambda: state.run(
                        source, values, scales, global_scale, workspace=scratch
                    )
                )

            with PreparationSession(
                device=source.device, autotune=False, compile_workers=0
            ) as session:
                session.prepare((plan.request(name=case["weight"], prepare_call=prepare),))
                session.freeze()
                state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
                scratch = (
                    torch.empty(state.required_workspace, device="cuda", dtype=torch.uint8)
                    if state.required_workspace
                    else None
                )
                with kernel_resolution_guard("checkpoint dense profiling"):
                    for _ in range(3):
                        result = blockscaled.mm(
                            source, weight, plan=plan, workspace=scratch
                        )
                    check_output(result, expected)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = blockscaled.mm(
                            source, weight, plan=plan, workspace=scratch
                        )
                    torch.cuda.synchronize()
                    flush()
                    torch.cuda.synchronize()
                    print(
                        json.dumps(
                            dict(
                                weight=case["weight"],
                                rows=rows,
                                config=record["config"],
                            )
                        ),
                        flush=True,
                    )
                    torch.cuda.profiler.start()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    check_output(output, expected)
                    del graph



if __name__ == "__main__":
    main()
