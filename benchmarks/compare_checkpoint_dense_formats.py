"""Compare IQ2_XS checkpoint matrices with NVFP4 requantizations at equal shapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._a16 import scale_storage
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.benchmark_blockscaled_precision import _paired
from benchmarks.checkpoint_dense import (
    checkpoint_cases, check_output, load_weight, oracle, source_manifest,
)
from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot


def requantize_nvfp4(decoded):
    n, k = decoded.shape
    gain = (2688.0 / decoded.abs().amax().float()).reshape(1)
    codes, scales = quantize_grouped_nvfp4_torch(
        decoded[None], torch.tensor([n], device=decoded.device), gain,
    )
    multiplier = gain.reciprocal()
    weight = blockscaled.pack_weight(
        codes[:, :, 0], scales, recipe="nvfp4", global_scale=multiplier,
    )
    compact = scale_storage(scales, n, k, 16).view(torch.float8_e4m3fn)
    compact = compact.view((n + 127) // 128, k // 64, 32, 4, 4)
    compact = compact.permute(0, 3, 2, 1, 4).reshape(-1, k // 16)[:n].float()
    unpacked = torch.stack((weight.values & 15, weight.values >> 4), -1).reshape(n, k).long()
    lut = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
        device=decoded.device,
    )
    local = (lut[unpacked] * compact.repeat_interleave(16, 1)).bfloat16()
    hashes = {
        name: hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for name, tensor in (("values", weight.values), ("scales", scales), ("global_scale", multiplier))
    }
    return weight, local, multiplier, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 20 or args.iters < 100 or min(args.batch_sizes) <= 0:
        raise ValueError("comparison requires positive rows, 20 warmups, and 100 trials")
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        raise RuntimeError("comparison requires SM120/SM121")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    index, cases = checkpoint_cases(args.model_path)
    cases = [case for case in cases if case["recipe"] == "iq2_xs"]
    if not cases:
        raise ValueError("checkpoint has no IQ2_XS dense matrices")
    flush = make_l2_flush_fn(enabled=True)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    medians = {recipe: [] for recipe in ("iq2_xs", "nvfp4")}
    with args.evidence.open("x") as evidence:
        def record(row):
            evidence.write(json.dumps(row) + "\n")
            evidence.flush()

        manifest = source_manifest()
        for name in ("benchmarks/compare_checkpoint_dense_formats.py",
                     "benchmarks/benchmark_blockscaled_precision.py"):
            manifest["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        record(dict(kind="manifest", **manifest, command=sys.argv, device=nvidia_smi_gpu_mode_snapshot(),
                    torch=torch.__version__, cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                    triton=importlib.metadata.version("triton"),
                    tuning_cache_version=os.environ.get("B12X_TUNING_CACHE_VERSION", "1"),
                    counts=args.batch_sizes, warmup=args.warmup, trials=args.iters,
                    metric="cold-L2 single-op graph microseconds; minimize",
                    ratio="NVFP4 latency / IQ2_XS latency; greater than one favors IQ2_XS",
                    nvfp4_source="requantized independently decoded BF16 IQ2_XS checkpoint weights"))
        for case in cases:
            iq2 = load_weight(args.model_path, index, case)
            operands = {"iq2_xs": iq2, "nvfp4": requantize_nvfp4(iq2[1])}
            for m in args.batch_sizes:
                source = torch.empty((m, case["k"]), device="cuda", dtype=torch.bfloat16)
                plans, requests = {}, []
                for recipe, (weight, _, _, _) in operands.items():
                    query = blockscaled.BlockscaledQuery(
                        recipe=recipe, num_tokens=m, in_features=case["k"],
                        padded_in_features=case["k"], out_features=case["n"],
                        activation_mode="a16", workspace_form="provided",
                        workspace_nbytes=2_000_000_000, expected_m=m,
                    )
                    plan = blockscaled.plan(query)
                    plans[recipe] = plan
                    scales = weight.metadata if recipe == "iq2_xs" else weight.scale_mma
                    global_scale = None if recipe == "iq2_xs" else weight.global_scale

                    def prepare(state, weight=weight, scales=scales, global_scale=global_scale):
                        scratch = (torch.empty(state.required_workspace, device="cuda", dtype=torch.uint8)
                                   if state.required_workspace else None)
                        return PreparedCall(
                            run=lambda: state.run(source, weight.values, scales, global_scale, workspace=scratch),
                            produce=lambda: source.fill_(0.125), owners=(weight, scales), capture_safe=False,
                        )

                    requests.append(plan.request(name=f"{recipe}:{case['weight']}",
                                                 prepare_call=prepare, benchmark_call=prepare))
                with PreparationSession(device=source.device, autotune=True, compile_workers=1) as session:
                    session.prepare(tuple(requests))
                    session.freeze()
                    torch.manual_seed(42 + m)
                    source.normal_(std=0.25)
                    graphs, outputs, workspaces, expected, states, checks = {}, {}, {}, {}, {}, {}
                    with kernel_resolution_guard("equal-shape format comparison"):
                        for recipe, (weight, decoded, multiplier, _) in operands.items():
                            state = require_prepared(plans[recipe], "gemm.blockscaled_precision", source.device)
                            states[recipe] = state
                            scratch = (torch.empty(state.required_workspace, device="cuda", dtype=torch.uint8)
                                       if state.required_workspace else None)
                            workspaces[recipe] = scratch
                            expected[recipe] = oracle(source, decoded, multiplier)
                            result = blockscaled.mm(source, weight, plan=plans[recipe], workspace=scratch)
                            check_output(result, expected[recipe])
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                output = blockscaled.mm(source, weight, plan=plans[recipe], workspace=scratch)
                            graphs[recipe], outputs[recipe] = graph, output
                        programs = {name: state.programs["gemm"] for name, state in states.items()}
                        buffers = {
                            name: (source, item[0].values,
                                   item[0].metadata if name == "iq2_xs" else item[0].scale_mma,
                                   outputs[name], workspaces[name])
                            for name, item in operands.items()
                        }
                        pointers = {name: tuple(t.data_ptr() if t is not None else None for t in tensors)
                                    for name, tensors in buffers.items()}
                        source.neg_()
                        for recipe in graphs:
                            outputs[recipe].fill_(float("nan"))
                            if workspaces[recipe] is not None:
                                workspaces[recipe].fill_(255)
                        allocated = torch.cuda.memory_allocated()
                        for graph in graphs.values():
                            graph.replay()
                        torch.cuda.synchronize()
                        assert allocated == torch.cuda.memory_allocated()
                        for recipe in graphs:
                            checks[recipe] = check_output(outputs[recipe], -expected[recipe])
                        before = nvidia_smi_gpu_mode_snapshot()
                        pairs = _paired(graphs, args.warmup, args.iters, flush)
                        after = nvidia_smi_gpu_mode_snapshot()
                        for recipe in graphs:
                            check_output(outputs[recipe], -expected[recipe])
                            assert programs[recipe] is states[recipe].programs["gemm"]
                            assert pointers[recipe] == tuple(t.data_ptr() if t is not None else None
                                                             for t in buffers[recipe])
                    latency = {recipe: statistics.median(pair[recipe] for pair in pairs) for recipe in graphs}
                    for recipe, value in latency.items():
                        medians[recipe].append(value)
                    record(dict(kind="case", weight=case["weight"], n=case["n"], k=case["k"], rows=m,
                                medians_us=latency, samples_us=pairs, correctness=checks,
                                configs={name: state.config.to_dict() for name, state in states.items()},
                                payload_sha256={name: item[3] for name, item in operands.items()},
                                packed_weight_bytes={name: item[0].values.numel() +
                                    (item[0].metadata.numel() if name == "iq2_xs" else item[0].scale_mma.numel())
                                    for name, item in operands.items()},
                                stable_callable=True, stable_addresses=True, zero_replay_allocation=True,
                                device_before=before, device_after=after))
                    print(f"{case['role']} M={m}: IQ2_XS {latency['iq2_xs']:.3f} us; "
                          f"NVFP4 {latency['nvfp4']:.3f} us; "
                          f"NVFP4/IQ2_XS {latency['nvfp4']/latency['iq2_xs']:.3f}x", flush=True)
                    del graphs, outputs, workspaces, expected, states, buffers
            del operands, iq2
        means = {name: math.exp(sum(map(math.log, values)) / len(values)) for name, values in medians.items()}
        record(dict(kind="summary", geomean_us=means, nvfp4_over_iq2_xs=means["nvfp4"] / means["iq2_xs"]))
        print(f"geo mean: {means}; NVFP4/IQ2_XS {means['nvfp4']/means['iq2_xs']:.3f}x", flush=True)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
