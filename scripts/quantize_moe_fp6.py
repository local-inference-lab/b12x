#!/usr/bin/env python
"""Offline-quantize BF16 MoE expert weights to MX-FP6 (W6A6) and save them.

The output file fed the historical ``b12x.integration.tp_moe.b12x_moe_fp6`` host
API directly; upstream the consumer is ``sparkinfer.moe.fused_moe`` (load with
``sparkinfer.quantization.mxfp6.load_fp6_moe_weights``). Activations are NOT quantized here:
they are quantized on the fly inside the kernel at inference time.

Input checkpoint (``--input``) must be a safetensors file containing BF16
tensors (safetensors only; pickle .pt inputs are not supported):
    w1 / w13 : FC1 weights (E, 2*N, K)  (gate+up; (E, N, K) for relu2)
    w2 / w_down / down : FC2 weights (E, K, N)

Examples
--------
Quantize a checkpoint:
    python scripts/quantize_moe_fp6.py --input experts_bf16.safetensors \
        --output experts_fp6.safetensors

Generate + quantize random weights (no checkpoint needed) to smoke-test the path:
    python scripts/quantize_moe_fp6.py --demo --experts 4 --k 128 --n 128 \
        --output demo_fp6.safetensors
"""
from __future__ import annotations

import argparse

import torch

from sparkinfer.quantization.mxfp6 import quantize_moe_weights_to_fp6, save_fp6_moe_weights


def _pick(d: dict, *keys: str) -> torch.Tensor:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} found in checkpoint; keys={list(d)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="safetensors checkpoint with w1/w13 and w2/down")
    parser.add_argument("--output", required=True, help="destination .safetensors path")
    parser.add_argument("--source-format", default="mxfp6_default")
    parser.add_argument("--activation", default="silu", choices=["silu", "relu2"])
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="use the slow torch reference quantizer instead of the GPU kernel",
    )
    parser.add_argument("--demo", action="store_true", help="generate random weights")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    if args.demo:
        torch.manual_seed(args.seed)
        w1_rows = (2 if args.activation == "silu" else 1) * args.n
        w1 = torch.randn(args.experts, w1_rows, args.k, device=device, dtype=torch.bfloat16) * 0.15
        w2 = torch.randn(args.experts, args.k, args.n, device=device, dtype=torch.bfloat16) * 0.15
    else:
        if not args.input:
            parser.error("--input is required unless --demo is set")
        from safetensors.torch import load_file

        ckpt = load_file(args.input, device=str(device))
        w1 = _pick(ckpt, "w1", "w13", "w1_bf16").to(device=device, dtype=torch.bfloat16)
        w2 = _pick(ckpt, "w2", "w_down", "down", "w2_bf16").to(device=device, dtype=torch.bfloat16)

    print(f"quantizing: w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
          f"source_format={args.source_format} activation={args.activation} device={device}")
    weights = quantize_moe_weights_to_fp6(
        w1,
        w2,
        source_format=args.source_format,
        activation=args.activation,
        use_gpu=not args.no_gpu and device.type == "cuda",
    )
    save_fp6_moe_weights(weights, args.output)
    print(
        f"saved -> {args.output}\n"
        f"  w1_fp6={tuple(weights.w1_fp6.shape)} w1_blockscale={tuple(weights.w1_blockscale.shape)}\n"
        f"  w2_fp6={tuple(weights.w2_fp6.shape)} w2_blockscale={tuple(weights.w2_blockscale.shape)}\n"
        f"  E={weights.num_experts} K={weights.k} N={weights.n} weight_fmt={weights.weight_fmt}"
    )


if __name__ == "__main__":
    main()
