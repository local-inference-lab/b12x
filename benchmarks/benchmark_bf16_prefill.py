"""Qualify BF16 tensor-core projection against the scalar path and FP64 math.

Explicitly invokes scalar and tensor-core kernels rather than timing automatic
serving dispatch. Timings include caller binding and retain raw graph samples.
"""

import json
import statistics
import argparse

import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x.gemm import bf16_gemv


def main():
    torch.manual_seed(415121)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-features", type=int, default=512)
    args = parser.parse_args()
    capacity, n, k = 4096, args.out_features, 5120
    x = torch.randn((capacity, k), device=device).bfloat16()
    weight = (torch.randn((n, k), device=device) / k**0.5).bfloat16()
    scalar = torch.empty((capacity, n), device=device)
    candidate = torch.empty_like(scalar)
    session = PreparationSession(device=x.device, autotune=False, compile_workers=2)
    plans = {}
    for name, backend, output in (("scalar", "simt", scalar), ("tensorcore", "prefill", candidate)):
        plan = bf16_gemv.plan(bf16_gemv.query_from_call(x, weight, out=output),
                              override=bf16_gemv.GemvConfig(backend=backend))
        def prepare(state):
            trial_output = torch.empty_like(output)
            return PreparedCall(run=lambda: state.run(x, weight, out=trial_output), output=trial_output)
        session.prepare((plan.request(name=name, prepare_call=prepare),))
        plans[name] = plan
    session.freeze()
    graphs = {}
    try:
        for rows in (1, 8, 64, 256, 513, 1024, 4096):
            scalar.fill_(float("nan"))
            candidate.fill_(float("nan"))
            bf16_gemv.mm(x[:rows], weight, out=scalar[:rows], plan=plans["scalar"])
            bf16_gemv.mm(x[:rows], weight, out=candidate[:rows], plan=plans["tensorcore"])
            oracle = x[:rows].double() @ weight.double().T
            errors = {}
            for name, value in (("scalar", scalar), ("tensorcore", candidate)):
                delta = value[:rows].double() - oracle
                errors[name] = {
                    "max_abs": float(delta.abs().max()),
                    "rmse": float(delta.square().mean().sqrt()),
                }
                assert torch.isfinite(value[:rows]).all()
                assert torch.isnan(value[rows:]).all()
                torch.testing.assert_close(value[:rows].double(), oracle,
                                           atol=2e-5, rtol=2e-5)
            graphs = {}
            for name in ("scalar", "tensorcore"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(5):
                        if name == "scalar":
                            bf16_gemv.mm(x[:rows], weight, out=scalar[:rows], plan=plans["scalar"])
                        else:
                            bf16_gemv.mm(x[:rows], weight, out=candidate[:rows], plan=plans["tensorcore"])
                graphs[name] = graph
            samples = {name: [] for name in graphs}
            for repeat in range(7):
                order = ("scalar", "tensorcore") if repeat % 2 else ("tensorcore", "scalar")
                for name in order:
                    graphs[name].replay()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 10)
            print(json.dumps({
                "rows": rows, "n": n, "k": k, "errors": errors,
                "samples_us": samples,
                "median_us": {name: statistics.median(v) for name, v in samples.items()},
            }), flush=True)
            for graph in graphs.values():
                graph.reset()
    finally:
        for graph in graphs.values():
            graph.reset()
        session.close()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
