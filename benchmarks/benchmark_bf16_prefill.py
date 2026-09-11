"""Qualify BF16 tensor-core projection against the scalar path and FP64 math.

Explicitly invokes scalar and tensor-core kernels rather than timing automatic
serving dispatch. Timings include caller binding and retain raw graph samples.
"""

import json
import statistics
import argparse

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import bf16_gemv
from b12x.gemm.bf16_gemv._kernel import _launch_scalar
from b12x.gemm.bf16_gemv._prefill import prefill_mm


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
prefill_mm(x, weight, candidate)
bf16_gemv.mm(x[:1], weight, out=scalar[:1])
freeze_kernel_resolution("BF16 prefill dynamic-row qualification")
try:
    for rows in (1, 8, 64, 256, 513, 1024, 4096):
        scalar.fill_(float("nan"))
        candidate.fill_(float("nan"))
        _launch_scalar(x[:rows], weight, scalar[:rows])
        prefill_mm(x[:rows], weight, candidate[:rows])
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
                        _launch_scalar(x[:rows], weight, scalar[:rows])
                    else:
                        prefill_mm(x[:rows], weight, candidate[:rows])
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
finally:
    unfreeze_kernel_resolution()
