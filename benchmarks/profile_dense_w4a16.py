"""Qualify and profile the Super3 Mamba W4A16 kernels and controlled ablations.

Use --profile-arm under ncu --profile-from-start off to collect a single
cold-L2 graph replay per shape. Without it, collect paired graph timings.
FlashInfer BF16 MMA is a diagnostic specialization, not its public default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import flashinfer

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _a16
from b12x.preparation import PreparationSession
from benchmarks.benchmark_blockscaled_precision import (
    _checked_graph, _clock_checks, _git, _paired, _prepared_call,
    _reference_weight, _snapshot, _warmup,
)
from benchmarks.benchmark_dense_gemm import SUPER3_MAMBA_GEMM_SPECS
from benchmarks.common import make_l2_flush_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("in_proj", "out_proj", "both"), default="both")
    parser.add_argument("--config", nargs=3, type=int, action="append", metavar=("N", "K", "SPLIT"))
    parser.add_argument("--profile-arm", choices=("b12x", "flashinfer", "flashinfer-bf16", "all"))
    parser.add_argument("--b12x-stages", type=int,
                        help="Diagnostic override of mainloop stage count; isolated compile cache.")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.b12x_stages is not None:
        if args.b12x_stages <= 0:
            parser.error("--b12x-stages must be positive")
        # Process-local compile-time ablation, never a runtime or serving knob.
        # Its artifacts cannot alias the production compilation cache.
        os.environ["B12X_COMPILE_CACHE_DIR"] = str(args.evidence.parent / "stage-compile-cache")
        os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
        from b12x._lib.dense_gemm import DenseGemmKernel
        original_stages = DenseGemmKernel._compute_stages

        def stages(*a, **kw):
            _, epilogue = original_stages(*a, **kw)
            return args.b12x_stages, epilogue

        DenseGemmKernel._compute_stages = staticmethod(stages)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    configs = args.config or [(128, 64, 4)]
    fi_root = pathlib.Path(flashinfer.__file__).parent
    paths = [pathlib.Path(__file__), ROOT / "benchmarks/benchmark_blockscaled_precision.py",
             ROOT / "b12x/_lib/dense_gemm.py", ROOT / "b12x/_lib/intrinsics.py",
             *sorted((ROOT / "b12x/gemm/blockscaled").glob("*.py")),
             *sorted(fi_root.rglob("*bf16_fp4*.py"))]
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as evidence:
        def record(row):
            evidence.write(json.dumps(row) + "\n")
            evidence.flush()

        record(dict(kind="manifest", command=[sys.executable, *sys.argv], cwd=str(ROOT),
                    revision=_git("rev-parse", "HEAD"), dirty=_git("status", "--porcelain"),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    environment={k: os.environ.get(k) for k in (
                        "CUTE_DSL_ARCH", "FLASHINFER_WORKSPACE_BASE", "B12X_BENCH_SPONGE",
                        "B12X_COMPILE_CACHE_DIR", "B12X_COMPILE_MEMORY_CACHE")},
                    device=_snapshot(), torch=torch.__version__, flashinfer=flashinfer.__version__,
                    profiled=bool(args.profile_arm), ratio="b12x_us / flashinfer_us; >1 favors FlashInfer"))
        flush = make_l2_flush_fn(enabled=True)
        for name, k, n, _ in SUPER3_MAMBA_GEMM_SPECS:
            if args.case != "both" and not name.endswith(args.case):
                continue
            torch.manual_seed(42)
            weight, local, multiplier = _reference_weight("nvfp4", n, k)
            source = torch.randn(1, k, device="cuda", dtype=torch.bfloat16) * 0.25
            reference = (source.float() @ local.to(torch.bfloat16).float().T) * multiplier
            fi_weight = flashinfer.prepare_bf16_fp4_weights(
                weight.values, _a16.scale_storage(weight.scale_mma, n, k, 16),
                multiplier, backend="cute-dsl",
            )
            graphs, correctness, owners = {}, {}, []

            def add(label, call):
                graph, check, output = _checked_graph(call, reference, label)
                graphs[label], correctness[label] = graph, check
                owners.extend((call, output))

            with PreparationSession(device="cuda", autotune=False, compile_workers=0) as session:
                for tile_n, tile_k, split_k in configs:
                    output = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
                    call, scratch = _prepared_call(
                        session, source, weight, output,
                        blockscaled.BlockscaledConfig(mode="a16", tile_n=tile_n,
                                                      tile_k=tile_k, split_k=split_k),
                    )
                    add(f"b12x_{tile_n}_{tile_k}_{split_k}", call)
                    owners.append(scratch)
                fi_out = torch.empty_like(output)
                add("flashinfer", lambda: flashinfer.mm_bf16_fp4(
                    source, *fi_weight, backend="cute-dsl", out=fi_out))

                from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import (
                    _get_cute_dsl_bf16_fp4_gemm, _select_bf16_fp4_tile_shape,
                )
                tile, atoms = _select_bf16_fp4_tile_shape(1, n, k)
                bf16_kernel = _get_cute_dsl_bf16_fp4_gemm(
                    tile, source.dtype, source.dtype, atoms, use_fp16_mma=0,
                )
                bf16_out = torch.empty_like(output)

                def fi_bf16():
                    bf16_kernel(source, fi_weight[0], fi_weight[1], bf16_out, fi_weight[2])
                    return bf16_out

                add("flashinfer-bf16", fi_bf16)
                session.freeze()
                _warmup(graphs, args.warmup, flush)
                before = _snapshot()
                if args.profile_arm:
                    for label, graph in graphs.items():
                        if args.profile_arm not in ("all", label, "b12x" if label.startswith("b12x_") else ""):
                            continue
                        flush()
                        torch.cuda.synchronize()
                        print(f"PROFILE {name}: {label}", flush=True)
                        torch.cuda.profiler.start()
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                    samples = []
                else:
                    samples = _paired(graphs, args.iters, flush)
                after = _snapshot()
                medians = {label: statistics.median(row[label] for row in samples) for label in graphs} if samples else {}
                record(dict(kind="case", name=name, m=1, n=n, k=k, correctness=correctness,
                            medians_us=medians, samples_us=samples,
                            snapshot_before=before, snapshot_after=after,
                            clock_validation=_clock_checks(before, after)))
                print(json.dumps(dict(name=name, medians_us=medians)), flush=True)
                graphs.clear()
                owners.clear()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
