#!/usr/bin/env python
"""Point-at-a-model offline MX-FP6 (W6A6) converter for HF checkpoints.

Walks a Hugging Face model directory and quantizes its quantizable weights to
packed MX-FP6. Two output formats:

* ``--format safetensors`` (default): a full, loadable HF checkpoint mirroring
  ModelOpt NVFP4 keys (``.weight`` / ``.weight_scale`` / ``.weight_scale_2`` /
  ``.input_scale``) with ``quantization_config.quant_algo="W6A6"``. Routed MoE
  experts are written per-expert. Sharded ``model-*.safetensors`` + index +
  patched ``config.json`` + copied tokenizer/aux files.
* ``--format pt``: per-layer ``.moe_fp6.safetensors`` / ``.dense_fp6.safetensors``
  artifacts for kernel validation (load MoE layers with
  ``sparkinfer.quantization.mxfp6.load_fp6_moe_weights``). The flag name is
  historical; output is safetensors (pickle persistence is not supported).

Routed experts plus the non-expert 2-D Linears (attention, shared experts; and
for dense, MLP) are quantized via the golden-rule walk; norms, embeddings,
router gates, ``lm_head``/``mtp``, SSM/linear-attention and the vision tower stay
BF16 (opt the last two in with ``--include-linear-attn`` / ``--include-vision``)
and are recorded in ``quantization_config.exclude_modules``.

The default ``--source-format`` is ``mxfp6_w6a8``: E2M3 weights (on disk) plus
FP8 E4M3 runtime activations in the MoE kernel (best MoE KLD, no post-export
``config.json`` edit). Legacy W6A6 pairings remain available via
``--source-format mxfp6_default`` / ``mxfp6_e2m3`` / ``mxfp6_e3m2``.

Examples
--------
Inspect the plan first (no GPU, no full-tensor loads):
    python scripts/quantize_model_fp6.py --model /path/Qwen3.6-35B-A3B --out out_moe --arch auto --dry-run

Export a full FP6 HF checkpoint (MoE):
    python scripts/quantize_model_fp6.py --model /path/Qwen3.6-35B-A3B --out qwen_fp6 --arch moe

Export a full FP6 HF checkpoint (dense, MLP + attention):
    python scripts/quantize_model_fp6.py --model /path/Qwen3.6-27B --out qwen27_fp6 --arch dense

Per-layer kernel-validation artifacts (safetensors):
    python scripts/quantize_model_fp6.py --model /path/Qwen3.6-35B-A3B --out out_moe --arch moe --format pt
"""
from __future__ import annotations

import argparse

import torch

from sparkinfer.quantization.mxfp6 import (
    SafetensorsModel,
    convert_dense_model_to_fp6,
    convert_moe_model_to_fp6,
    discover_moe_experts,
    export_dense_model_to_fp6_safetensors,
    export_moe_model_to_fp6_safetensors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model directory (config.json + *.safetensors)")
    parser.add_argument("--out", help="output directory for FP6 artifacts (required unless --inspect)")
    parser.add_argument("--arch", default="auto", choices=["auto", "moe", "dense"])
    parser.add_argument(
        "--format",
        dest="out_format",
        default="safetensors",
        choices=["safetensors", "pt"],
        help="safetensors: full HF checkpoint (ModelOpt-mirror W6A6, default); "
        "pt: legacy per-layer artifacts for kernel validation",
    )
    parser.add_argument(
        "--source-format",
        default="mxfp6_w6a8",
        choices=[
            "mxfp6_w6a8",
            "mxfp6_default",
            "mxfp6_e2m3",
            "mxfp6_e3m2",
            "mxfp6_mixed",
        ],
        help="mxfp6_w6a8 (default): E2M3 weights + E4M3 runtime activations; "
        "mxfp6_default: E2M3 weights + E3M2 activations (legacy W6A6); "
        "mxfp6_e2m3: symmetric E2M3 W6A6",
    )
    parser.add_argument("--activation", default="silu", choices=["silu", "relu2"], help="MoE only")
    parser.add_argument(
        "--gate-up-order",
        default="gate_up",
        choices=["gate_up", "up_gate"],
        help="MoE only: source fused FC1 row order (HF default gate_up)",
    )
    parser.add_argument("--no-attention", action="store_true", help="skip attention projections (dense + moe)")
    parser.add_argument(
        "--include-linear-attn",
        action="store_true",
        help="also quantize linear-attention 2-D proj matmuls (in_proj_*/out_proj) "
        "for dense + moe. Shrinks multimodal-hybrid checkpoints below FP8; "
        "quality trade-off, validate downstream.",
    )
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="also quantize vision-tower block linears (attn qkv/proj, mlp fc) for dense + moe.",
    )
    parser.add_argument(
        "--skip-experts",
        action="store_true",
        help="MoE sensitivity sweep: copy routed experts through BF16 "
        "(everything else still quantizes). Diagnostic, not for production.",
    )
    parser.add_argument(
        "--skip-shared-expert",
        action="store_true",
        help="MoE sensitivity sweep: keep shared_expert linears BF16.",
    )
    parser.add_argument(
        "--report-error",
        action="store_true",
        help="dequantize every emitted tensor and print a per-group "
        "relative-RMSE table (experts.gate/up/down, shared_expert, self_attn, "
        "linear_attn, ...)",
    )
    parser.add_argument(
        "--block-scale-rule",
        default="mse",
        choices=["mse", "ceil"],
        help="offline UE8M0 weight exponent selection: mse (default, Phase 1.2: "
        "per-block ceil vs ceil-1, keep lower MSE) or ceil (amax containment only). "
        "Runtime activation quant is unaffected.",
    )
    parser.add_argument("--limit-layers", type=int, default=None, help="convert only the first N layers")
    parser.add_argument("--no-gpu", action="store_true", help="MoE only: use the slow torch quantizer")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; do not quantize or write")
    parser.add_argument(
        "--inspect",
        type=int,
        default=None,
        metavar="LAYER",
        help="print every checkpoint key+shape under .layers.{LAYER}. and exit (debug layout)",
    )
    args = parser.parse_args()

    if args.inspect is not None:
        model = SafetensorsModel(args.model)
        needle = f".layers.{args.inspect}."
        rows = sorted(k for k in model.keys() if needle in k)
        print(f"keys under {needle} ({len(rows)}):")
        for k in rows:
            print(f"  {k}\t{tuple(model.shape_of(k))}")
        return

    if not args.out:
        parser.error("--out is required unless --inspect is set")

    arch = args.arch
    if arch == "auto":
        arch = "moe" if discover_moe_experts(SafetensorsModel(args.model)) is not None else "dense"
        print(f"[auto] detected arch: {arch}")

    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    use_gpu = not args.no_gpu and device == "cuda"

    if args.out_format == "safetensors":
        if arch == "moe":
            report = export_moe_model_to_fp6_safetensors(
                args.model,
                args.out,
                source_format=args.source_format,
                activation=args.activation,
                gate_up_order=args.gate_up_order,
                include_attention=not args.no_attention,
                include_linear_attn=args.include_linear_attn,
                include_vision=args.include_vision,
                skip_experts=args.skip_experts,
                skip_shared_expert=args.skip_shared_expert,
                report_error=args.report_error,
                limit_layers=args.limit_layers,
                device=device,
                use_gpu=use_gpu,
                block_scale_rule=args.block_scale_rule,
                dry_run=args.dry_run,
            )
        else:
            if args.skip_experts or args.skip_shared_expert:
                parser.error("--skip-experts/--skip-shared-expert are MoE-only")
            report = export_dense_model_to_fp6_safetensors(
                args.model,
                args.out,
                source_format=args.source_format,
                include_attention=not args.no_attention,
                include_linear_attn=args.include_linear_attn,
                include_vision=args.include_vision,
                report_error=args.report_error,
                limit_layers=args.limit_layers,
                device=device,
                use_gpu=use_gpu,
                block_scale_rule=args.block_scale_rule,
                dry_run=args.dry_run,
            )
        print(
            f"\n{'[dry-run] plan only' if args.dry_run else 'done'}: "
            f"arch={report.arch} layers={len(report.layers)} "
            f"quantized={report.quantized_tensors} copied={report.copied_tensors} "
            f"shards={report.shards} bytes={report.total_bytes} -> {report.out_dir}"
        )
        return

    if arch == "moe":
        report = convert_moe_model_to_fp6(
            args.model,
            args.out,
            source_format=args.source_format,
            activation=args.activation,
            gate_up_order=args.gate_up_order,
            limit_layers=args.limit_layers,
            device=device,
            use_gpu=use_gpu,
            dry_run=args.dry_run,
        )
    else:
        report = convert_dense_model_to_fp6(
            args.model,
            args.out,
            source_format=args.source_format,
            include_attention=not args.no_attention,
            limit_layers=args.limit_layers,
            device=device,
            dry_run=args.dry_run,
        )

    print(
        f"\n{'[dry-run] plan only' if args.dry_run else 'done'}: arch={report.arch} "
        f"layers={len(report.layers)} tensors_written={report.tensors_written} "
        f"artifacts={len(report.artifacts)} -> {report.out_dir}"
    )


if __name__ == "__main__":
    main()