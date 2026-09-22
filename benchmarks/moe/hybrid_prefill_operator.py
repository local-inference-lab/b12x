"""Matched resident/mapped W4A16 graphs using retained checkpoint-layer routes.

Lengths beyond the saved route fixture repeat its rows. They measure controlled
operator scaling and reuse, not natural long-prompt serving or model quality.
Timing includes both cache tiers, route packing and ordered output reduction.
Profiler traces are separate from timed samples.
"""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.compile_plan import observe_programs
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.artifacts import sha256
from scripts._sm103_source import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Replay each tier inside an NVTX range for an external profiler; no timing claim",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this physical control requires SM120")
    if min(*args.rows, args.samples, args.replays) < 1:
        raise ValueError("row counts and repetitions must be positive")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "checkpoint_oracle", root / "tests/moe/test_next80_checkpoint.py"
    )
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    config, source, _ = oracle.load_layer(args.checkpoint, args.layer)
    trace = torch.load(args.routes, weights_only=True, map_location="cpu")
    original = {k: trace[k].contiguous() for k in ("x", "ids", "weights")}
    e = source.plan.geometry.num_experts
    used = set(original["ids"].flatten().tolist())
    unused = next((n for n in range(e) if n not in used), None)
    if unused is None:
        raise ValueError("all-cold control needs an unused resident expert")
    report = dict(
        source=source_identity(root),
        checkpoint_layer_sha256=source.weights.checkpoint_fingerprint,
        route_file_sha256=sha256(args.routes),
        route_rows=len(original["x"]),
        recipe="nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum",
        layer=args.layer,
        geometry=asdict(source.plan.geometry),
        command=__import__("sys").argv,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_before=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        measurement_scope="external profiler"
        if args.diagnostic_only
        else "CUDA event timing; separate profiler traces",
        rows=[],
    )
    for rows in args.rows:
        inputs = {
            k: value.repeat((rows + len(value) - 1) // len(value), 1)[:rows]
            .contiguous()
            .cuda()
            for k, value in original.items()
        }
        inputs["ids"] = inputs["ids"].to(torch.int32)
        plans, bindings, graphs, programs = {}, {}, {}, {}
        for mode, residents in (("resident", tuple(range(e))), ("mapped", (unused,))):
            plan = moe.plan_execution(
                experts=source,
                capacity=moe.ExecutionCapacity(
                    max_tokens=rows, top_k=config["num_experts_per_tok"]
                ),
                placement=moe.ExpertResidencyPlan(
                    total_experts=e,
                    hbm_expert_ids=residents,
                    grace_expert_ids=tuple(n for n in range(e) if n not in residents),
                    layer=source.weights.layer_name,
                    model_fingerprint=source.weights.checkpoint_fingerprint,
                    workload="retained-prefill-routes",
                    provenance="matched operator tier control",
                ),
                memory_budget=moe.ExpertMemoryBudget(
                    hbm_bytes=4 << 30, grace_bytes=4 << 30
                ),
            )
            plans[mode] = plan

        def prepare(state, mode):
            binding = state.bind(
                a=inputs["x"], topk_ids=inputs["ids"], topk_weights=inputs["weights"]
            )
            bindings[mode] = binding
            return PreparedCall(
                run=binding.run,
                output=binding.output,
                owners=(binding,),
                close=state.close,
            )

        with PreparationSession(autotune=False, compile_workers=0) as session:
            session.prepare(
                tuple(
                    p.request(name=n, prepare_call=lambda s, n=n: prepare(s, n))
                    for n, p in plans.items()
                )
            )
            pointers = {n: p.prepared.state.pointers() for n, p in plans.items()}
            for mode, binding in bindings.items():
                graph = torch.cuda.CUDAGraph()
                with (
                    observe_programs() as observed,
                    session.capture(),
                    torch.cuda.graph(graph),
                ):
                    binding.run()
                graphs[mode] = graph
                programs[mode] = [
                    asdict(p)
                    for p in sorted(observed, key=lambda p: (p.dialect, p.key))
                ]
                graph.replay()
            torch.cuda.synchronize()
            session.freeze()
            # Profiler-only runs have no timing samples, but must establish the
            # same replay ownership and allocation invariants before reporting them.
            for mode, graph in graphs.items():
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                with kernel_resolution_guard("matched prefill replay qualification"):
                    graph.replay()
                    torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                assert pointers[mode] == plans[mode].prepared.state.pointers()
            resident = bindings["resident"].output.cpu()
            torch.testing.assert_close(
                resident, bindings["mapped"].output.cpu(), atol=0, rtol=0
            )
            assert torch.isfinite(resident).all() and torch.count_nonzero(resident)
            expected = oracle.routed_oracle(
                source,
                inputs["x"][: min(rows, 4)],
                inputs["ids"][: min(rows, 4)],
                inputs["weights"][: min(rows, 4)],
                device="cpu",
            )
            torch.testing.assert_close(
                resident[: len(expected)], expected.cpu(), atol=0.001, rtol=0.03
            )
            samples = {mode: [] for mode in graphs}
            for sample in range(0 if args.diagnostic_only else args.samples):
                for mode in (
                    ("resident", "mapped")
                    if sample % 2 == 0
                    else ("mapped", "resident")
                ):
                    graph = graphs[mode]
                    for _ in range(3):
                        graph.replay()
                    start, stop = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    with kernel_resolution_guard("matched prefill operator"):
                        start.record()
                        for _ in range(args.replays):
                            graph.replay()
                        stop.record()
                    stop.synchronize()
                    assert (
                        before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    )
                    assert pointers[mode] == plans[mode].prepared.state.pointers()
                    samples[mode].append(start.elapsed_time(stop) * 1000 / args.replays)
            for mode, graph in graphs.items():
                if args.diagnostic_only:
                    with torch.cuda.nvtx.range(f"w4a16_{mode}_m{rows}"):
                        graph.replay()
                        torch.cuda.synchronize()
                else:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        graph.replay()
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(
                        str(args.output / f"rows-{rows}-{mode}.json")
                    )
                graph.reset()
            report["rows"].append(
                dict(
                    rows=rows,
                    repeated_fixture=rows > len(original["x"]),
                    samples_us=samples,
                    exact_tier_equality=True,
                    independent_oracle_rows=len(expected),
                    no_replay_allocation=True,
                    fixed_pointers=True,
                    memory={
                        n: asdict(p.prepared.state.memory) for n, p in plans.items()
                    },
                    programs=programs,
                )
            )
        assert all(p.prepared is None for p in plans.values())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["gpu_after"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
