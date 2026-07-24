#!/usr/bin/env python
"""Decode a sparkinfer FP6 checkpoint back to a plain BF16 HF checkpoint.

Inverse of ``scripts/quantize_model_fp6.py``: each quantized linear's
packed-code + UE8M0-scale tensors are dequantized to one BF16 ``.weight``;
everything else is copied through and ``quantization_config`` is removed.

The output loads on stock vLLM with no sparkinfer involvement, which is the point:
a KLD of the dequantized model against the original BF16 model measures the
*weight* quantization error alone. Comparing that with the full sparkinfer FP6 KLD
(same reference logits) isolates how much the runtime W6A6 *activation*
quantization contributes:

    KLD(dequant-BF16)  ~= weight error only
    KLD(sparkinfer FP6)      ~= weight error + runtime activation error

Example:
    python scripts/dequantize_fp6_to_bf16.py \
        --model /media/fmodels/TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6 \
        --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6-DequantBF16
"""
from __future__ import annotations

import argparse

import torch

from sparkinfer.quantization.mxfp6 import dequantize_fp6_checkpoint_to_bf16


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="sparkinfer FP6 checkpoint directory")
    parser.add_argument("--out", required=True, help="output BF16 checkpoint directory")
    parser.add_argument(
        "--no-gpu", action="store_true", help="dequantize on CPU (slower)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    report = dequantize_fp6_checkpoint_to_bf16(args.model, args.out, device=device)
    print(
        f"\ndone: dequantized={report.quantized_tensors} "
        f"copied={report.copied_tensors} shards={report.shards} "
        f"bytes={report.total_bytes} -> {report.out_dir}"
    )


if __name__ == "__main__":
    main()
