"""Replay recorded unquantized vLLM router GEMMs with checkpoint weights.

This diagnostic invokes the existing Torch operation used by vLLM. It is not
a b12x compute backend, replacement numerical recipe or throughput benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--traces", type=Path, nargs=2, required=True)
    p.add_argument("--step", type=int, default=130)
    p.add_argument("--modules", nargs="+", default=["model.layers.0.mlp.gate"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh diagnostic receipt")
    records = [
        next(r for r in torch.load(t, weights_only=True) if r["step"] == args.step)
        for t in args.traces
    ]
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    receipt = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arguments": vars(args),
        "rows": [],
    }
    for name in args.modules:
        key = name + ".weight"
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
            source = f.get_tensor(key)
        weight = source.cuda()
        inputs = [r["modules"][name]["args"][0].cuda() for r in records]
        expected = [r["modules"][name]["output"][0].cuda() for r in records]
        assert torch.equal(inputs[0][0], inputs[1][0]), "logical router inputs differ"
        outputs = [torch.nn.functional.linear(x, weight) for x in inputs]
        for observed, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        # Repeat the same shape independently; no scheduler, cache or maintenance.
        for x, output in zip(inputs, outputs, strict=True):
            for _ in range(10):
                torch.testing.assert_close(
                    torch.nn.functional.linear(x, weight), output, atol=0, rtol=0
                )
        oracle = inputs[0][0].double() @ weight.double().T
        differing = torch.where(outputs[0][0] != outputs[1][0])[0]
        reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            full = [torch.nn.functional.linear(x, weight) for x in inputs]
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
        receipt["rows"].append(
            {
                "module": name,
                "weight_sha256": hashlib.sha256(
                    source.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "dtype": str(weight.dtype),
                "shapes": [list(x.shape) for x in inputs],
                "same_shape_exact": True,
                "matches_serving_exact": True,
                "different_columns": differing.tolist(),
                "values": [output[0, differing].float().tolist() for output in outputs],
                "fp64_dot": oracle[differing].tolist(),
                "rounded_fp64_dot": oracle[differing]
                .to(outputs[0].dtype)
                .float()
                .tolist(),
                "default_reduced_bf16_reduction": reduced,
                "full_precision_reduction": {
                    "cross_shape_exact": torch.equal(full[0][0], full[1][0]),
                    "rounded_fp64_exact": [
                        torch.equal(v[0], oracle.to(v.dtype)) for v in full
                    ],
                    "different_columns": torch.where(full[0][0] != full[1][0])[
                        0
                    ].tolist(),
                },
            }
        )
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2, default=str)


if __name__ == "__main__":
    main()
