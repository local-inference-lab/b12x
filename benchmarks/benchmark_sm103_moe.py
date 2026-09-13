"""Qualify and time native SM103 MoE through the canonical planning API.

A bundle contains ``weights`` (PackedWeights constructor fields), ``w13_layout``
and BF16 ``activations``. Shapes determine E/K/N; --top-k determines routing.
All bundle tensors are CPU tensors from the loader's unmodified NVFP4 source.
Without a bundle, this benchmark labels the weights and inputs as synthetic.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import b12x
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.materialized_nvfp4_reference import reference
from tests.moe.test_sm103_nvfp4 import case
from benchmarks.benchmark_roce_oneshot import _source_state


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timed(fn, samples, repetitions):
    values = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(samples):
        start.record()
        for _ in range(repetitions):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repetitions)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 128])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if min(*args.tokens, args.samples, args.repetitions) <= 0:
        parser.error("token counts, samples, and repetitions must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        parser.error("this benchmark requires physical SM103 hardware")
    torch.manual_seed(103)
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = max(args.tokens)
    bundle = None
    if args.bundle:
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
        weights = fused_moe.PackedWeights(
            **{k: v.to(device) for k, v in bundle["weights"].items()}
        )
        e, two_n, half_k = weights.w13.shape
        wp = fused_moe.plan_weights(
            source=fused_moe.PackedSource(
                format="modelopt_nvfp4", w13_layout=bundle["w13_layout"]
            ),
            activation=fused_moe.ActivationSpec(
                mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=e, hidden_size=half_k * 2, intermediate_size=two_n // 2
            ),
        )
        prepared = fused_moe.prepare_weights(plan=wp, weights=weights)
        plan = fused_moe.plan_execution(
            experts=prepared,
            capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=args.top_k),
        )
        fused_moe.prewarm(plan)
        scratch = torch.empty(plan.scratch.nbytes, dtype=torch.uint8, device=device)
        source = bundle["activations"].to(device)
        if (
            source.dtype != torch.bfloat16
            or source.shape[0] < capacity
            or source.shape[1] != half_k * 2
        ):
            parser.error("bundle activations must be BF16 [at least capacity,K]")
    else:
        prepared, plan, scratch = case(
            device,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            top_k=args.top_k,
            capacity=capacity,
        )
        source = (
            torch.randn((capacity, args.hidden), device=device, dtype=torch.bfloat16)
            * 0.1
        )
    raw = prepared._impl
    receipt = {
        "status": "qualifying",
        "command": sys.argv,
        **_source_state(),
        "worktree": str(Path.cwd()),
        "gpu": str(torch.cuda.get_device_properties(device)),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weight_source": "checkpoint_bundle" if bundle is not None else "synthetic",
        "bundle_sha256": file_hash(args.bundle) if args.bundle else None,
        "geometry": {
            "hidden": raw.hidden_size,
            "intermediate": raw.intermediate_size,
            "experts": raw.num_experts,
            "top_k": args.top_k,
        },
        "capacity": capacity,
        "scratch_bytes": plan.scratch.nbytes,
        "rows": [],
    }
    b12x.freeze_kernel_resolution("SM103 benchmark")
    try:
        for m in args.tokens:
            a = source[:m].clone()
            ids = torch.randint(
                raw.num_experts, (m, args.top_k), device=device, dtype=torch.int64
            )
            ids[0, 0] = -1
            weights = torch.softmax(torch.randn(m, args.top_k, device=device), dim=-1)
            bound = fused_moe.bind(
                plan,
                scratch=scratch,
                experts=prepared,
                a=a,
                topk_ids=ids,
                topk_weights=weights,
            )
            expected = reference(a, raw, ids, weights)
            output = fused_moe.run(binding=bound)
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.03)
            cosine = F.cosine_similarity(
                output.float().flatten(), expected.float().flatten(), dim=0
            ).item()
            if (
                not cosine > 0.999
                or not torch.isfinite(output).all()
                or not output.count_nonzero()
            ):
                raise RuntimeError("finite/nonzero/cosine correctness gate failed")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=bound)
            a.mul_(0.75)
            ids.copy_((ids + 1) % raw.num_experts)
            expected = reference(a, raw, ids, weights)
            output.fill_(float("nan"))
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert before == torch.cuda.memory_allocated()
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.03)
            for _ in range(10):
                graph.replay()
            samples = timed(graph.replay, args.samples, args.repetitions)
            receipt["rows"].append(
                {
                    "tokens": m,
                    "cosine": cosine,
                    "correctness": "passed",
                    "graph_us": samples,
                    "median_graph_us": statistics.median(samples),
                    "graph_allocations": 0,
                    "output_pointer": output.data_ptr(),
                }
            )
            graph.reset()
        receipt["status"] = "passed"
    finally:
        b12x.unfreeze_kernel_resolution()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
