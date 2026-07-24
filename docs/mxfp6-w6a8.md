# MX-FP6 (W6A6/W6A8) dense + MoE

MX-FP6 serving support for SM120/SM121: 6-bit E2M3 weights with UE8M0 group-32
block scales, activations quantized at runtime to FP6 (W6A6) or FP8 E4M3
(W6A8, the production default). Both operand recipes ride the same
`mxf8f6f4` `m16n8k32` block-scaled MMA the MXFP8 path uses; only the emitted
instruction's type qualifier distinguishes FP6 from FP8 operands.

## Format

- **Weights on disk**: packed MX-FP6 E2M3 codes (`uint8`, 4 values in 3 bytes
  along K) + unswizzled UE8M0 K/32 block scales + per-tensor (dense) or
  per-expert (MoE) f32 global scales. ModelOpt-mirror safetensors schema
  (`quant_method=modelopt`, `quant_algo=W6A6`); schema id
  `b12x_fp6_safetensors_v1` (historical, kept for checkpoint compatibility).
- **Runtime activations**: quantized each forward. W6A8 uses FP8 E4M3 codes
  with UE8M0 K/32 scales — the same activation encoding as `w4a8_mx`, so the
  MoE path reuses that machinery unchanged.
- There is no FP6 torch dtype: FP6 tensors travel as `uint8` byte-containers
  (one code per byte in smem/registers) or 3:4-packed storage; the CUTLASS
  element type is injected at pointer-construction time
  (`Float6E2M3FN`/`Float6E3M2FN` in `_lib/utils.py::get_cutlass_dtype`).

## Dense (`_lib/dense_gemm.py` + `gemm/mxfp6_linear`)

The dense block-scaled GEMM accepts `ab_dtype="float6_e2m3fn"` (plus per-operand
`a_fmt`/`b_fmt` so W6A8 can mix an FP8 A with an FP6 B). Key mechanics:

- **Byte-container mainloop**: FP6 codes flow through the FP8-shaped
  TMA/ldmatrix machinery; the per-K-block MMA is emitted by
  `_lib/dense_gemm_mxfp6.py::emit_mxfp6_dense_mma_k_block` as inline PTX with
  explicit format qualifiers.
- **Packed-B streaming** (`b_packed=True`): B is TMA-loaded in its 3:4 packed
  form and expanded to byte-container in smem, saving 25% of B-side HBM
  traffic. Gated by `SPARKINFER_PACKED_B_MIN_N` (large-N layers only).
- **Small-M decode tiles**: m<=16 activations use a small-M quantizer (only
  real rows computed) and narrow MMA tiles.
- **Per-row activation global scale**: per-tensor amax makes output depend on
  batch composition (chunked prefill changes results); per-row scaling makes
  each row's quantization independent, matching BF16 cuBLAS semantics.
  Default on; `SPARKINFER_DENSE_PER_ROW_GS=0` for A/B.
- **Fused quant prologue** (`SPARKINFER_DENSE_FUSED_QUANT=1`): producer warp
  quantizes BF16 x directly into sA/sSFA. Measured ~10% slower at M=1 than
  the separate quantizer on RTX PRO 6000; numerically bit-identical. Kept as
  an off-by-default gate.

`gemm.mxfp6_linear` wraps this as the fused-linear op (weight pack + runtime
activation quant + GEMM + alpha epilogue). `gemm.bf16_gemv` covers narrow
BF16 projections (e.g. GDN `in_proj_ba`, N<=1024) where a GEMM tile wastes
the CTA.

## MoE (`moe.fused_moe`, quant mode `w6a8_mx`)

`quant_mode="w6a8_mx"` with `source_format="mxfp6_e2m3"`, following the
`w4a8_mx` shape: FP6 packed weights + swizzled UE8M0 scales + per-expert
runtime alphas prepared once (`kernels/w6a8/weights.py`), MXFP8-E4M3
activation quantization reused from the w4a8 path, FP6 MMA emitted per K
block via `kernels/mxfp6_moe.py`. Scheduling is the unified dynamic
persistent-grid backend; the combine is the standard atomic scatter by
default, or the deterministic `route_output` + top-k sum when
`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1` / `Caps.deterministic_output=True`
(required for bit-reproducible KLD scoring; measured 1.3-4.4x slower on the
combine — scoring only, not serving).

## Accuracy (measured, Qwen3.6, Wikitext KLD vs BF16, eager + deterministic)

| Model | KLD | Position |
|-------|-----|----------|
| Qwen3.6-27B dense (W6A8) | 0.034599 | measured W6A8 dense floor: weight slice ~0.023 + activation slice ~0.011 |
| Qwen3.6-35B-A3B MoE (W6A8) | 0.015388 | FP8 band (FP8 ~0.0158) |

Constants re-baselined Jul 23 2026 after moving every per-row scale
derivation to correctly-rounded division (host f64-divide-then-cast,
kernel `div.rn.f32`); see `fp6-results.md` for the history.

Reproducibility requires eager scoring (no inductor) and the deterministic
MoE combine; both runs then match bit-for-bit. Levers measured and closed:
MSE block scales (tie), GPTQ (skipped, ~0.002 est.), group-32 Hadamard
rotation (NO-GO: +0.00012 worse).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_PACKED_B_MIN_N` | 12288 | Min out_features for packed-B streaming |
| `SPARKINFER_DENSE_PER_ROW_GS` | on | Per-row activation global scale (dense, m>1) |
| `SPARKINFER_DENSE_PERSISTENT_SCRATCH` | on | Persistent decode-quant scratch |
| `SPARKINFER_DENSE_FUSED_QUANT` | off | Fused quant prologue (slower; A/B gate) |
| `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` | off | Deterministic MoE combine (scoring) |
| `SPARKINFER_ENABLE_FP6_MICRO` | off | BS1 MoE micro-kernel opt-in |
| `SPARKINFER_FP6_ACT_FMT_OVERRIDES` | unset | `pat=fmt` fnmatch per-linear activation format overrides |
| `SPARKINFER_DISABLE_BF16_GEMV` | off | Disable small-N GEMV routing |

## Quantization tooling

`scripts/quantize_model_fp6.py` (dense) and `scripts/quantize_moe_fp6.py`
(MoE) export HF checkpoints to the FP6 schema (safetensors only, architecture
discovery, MSE-refined block scales). `scripts/dequantize_fp6_to_bf16.py`
produces the weight-only-error control checkpoint used to separate weight
floor from runtime slice in KLD attribution.
