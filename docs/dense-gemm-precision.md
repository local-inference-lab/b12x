# Dense GEMM activation precision

Status: implemented on SM120/SM121. `gemm.blockscaled` accepts BF16 activations with
NVFP4 or MXFP8 weights and selects an activation precision for dense linear
projections. MoE routing and expert GEMMs have separate implementations.

`mode="a16"` uses the BF16 warp-MMA specialization of
`b12x/_lib/dense_gemm.py::DenseGemmKernel`. The specialization retains the dense
engine's TMA producer, shared-memory pipeline, tile scheduler, accumulator
epilogue, compiler, and launch wrapper. It loads compressed weights and their
scales into shared memory, converts weight pairs directly into BF16 MMA
registers, and accumulates in FP32. Split-K writes FP32 partials and reduces
them into BF16 output. Activations remain BF16 throughout this route.

The implementation follows the inline weight conversion and narrow-M warp-MMA
patterns in `b12x/moe/_shared/kernels/w4a16/kernel.py` and the native FP8
conversion helpers in `b12x/_lib/intrinsics.py`. The MoE engine and its prepared
scale layout are references, not dependencies of the dense launch path. Triton
is used only for supporting activation quantization and packing in
`b12x/gemm/blockscaled/_quantize.py`.

## Shared weight contract

| Recipe | Stored values | Block scales | Reconstructed weight |
| --- | --- | --- | --- |
| NVFP4 | `uint8[N,K/2]`, low nibble first | E4M3, one per 16 K values | `E2M1 * block_scale * global_scale` |
| MXFP8 | `float8_e4m3fn[N,K]` | UE8M0, one per 32 K values | `E4M3 * block_scale` |

Both activation precision routes accept the same F8_128x4-swizzled weight
scale storage. `w4a16`/`w8a16` accept its flat physical storage or native six
dimensional MMA view. `pack_weight` also supports the established compact
MXFP8 scale input, which it swizzles during weight preparation.

NVFP4 `pack_weight` borrows the packed values and scale tensors without
rewriting them. `global_scale_kind="reciprocal"` interprets the supplied weight
global scale as a quantizer multiplier and divides by it in the epilogue.
Neither mode creates a second weight-scale tensor. Global scales must be
finite and positive; reconstructed weights must fit BF16.

A16 requires contiguous, 16-byte-aligned CUDA tensors, N divisible by 8,
stored K divisible by 32, and input K divisible by 8. Its native packed BF16
conversions require PTX 9.2 (CUDA 13.3). Quantized activation execution requires
stored K divisible by 128. MXFP8's established functional path remains available
for its other supported layouts and devices.

## Preparation and graph capture

Packed and raw dense calls use the unified
[preparation lifecycle](gpu-profiles.md). `blockscaled.query_from_call` describes
the actual source/weight/output/workspace ABI; `blockscaled.plan(query,
override=...)` returns an allocation-free declaration. Prepare a request with
the real parameter tensors and a representative activation producer, then pass
the prepared plan to `blockscaled.mm(..., plan=plan)`. Standalone W4A16
likewise takes its prepared plan.

The query includes exact planned M, logical/stored K, recipe, activation mode,
scale availability and interpretation, layout/alignment eligibility, output
form, workspace form and expected-M semantics. Functional MXFP8 and provided
output/workspace calls keep their existing distinct quantization paths.
Already-quantized activations retain their supplied precision.

Explicit `activation_mode="a16"` or `"quantized"` constrains the declaration.
The per-query `BlockscaledConfig` selects the actual mode and applicable
`tile_n`, `tile_k`, and `split_k`; inactive quantized fields are `None`.
For eligible uncovered BF16 inputs, the integrated default retains A16 at M1–8
with `(tile_n, tile_k, split_k) = (128, 64, 4)`. Forced A16 without that default
route retains `(64, 64, 1)`, including M16. Native K-slice clamping and layout
restrictions still apply. Missing required activation scales are errors.

Enabled startup search races the complete eligible set on cache misses.
Explicit valid pins and completed cached choices take precedence. Disabled or
cancelled tuning still prepares the validated default; it does not publish a
measured winner. No embedded precision table or separate policy mode participates.

Timings include the actual activation production/scale, quantization and GEMM
path used by the invocation. Weight preparation remains outside timing.
Correctness, poisoned-buffer replay and quantization semantics precede timing
interpretation. Standalone GEMM-only measurements are diagnostics, not
end-to-end precision-selection evidence.

Prepare every graph-visible exact M. Scratch is caller-owned and reusable
across sequential calls; concurrent calls require disjoint output/workspace.
Capture under `session.capture()`, retain the plans for the graph lifetime,
and destroy graphs before releasing the plans or closing the session.
Capture and replay execute retained launchers without policy or compiler
resolution.

## Super3.5 Mamba decode benchmark

The dense benchmark's `super3-mamba` shape profile isolates the two NVFP4
projections in `nvidia/NVIDIA_Super3_5_VL_IQ2XXS-Packed`. Each occurs 40 times:

| Projection | Input K | Output N |
| --- | ---: | ---: |
| Mamba `in_proj` | 4096 | 18560 |
| Mamba `out_proj` | 8192 | 4096 |

Reproduce the untuned single-token serving configuration (16-row tile,
N128/K64, four K slices) on GB10:

```bash
CUTE_DSL_ARCH=sm_121a .venv/bin/python benchmarks/benchmark_dense_gemm.py \
  --profile super3-mamba --dtype fp4-a16 --batch-sizes 1 \
  --a16-config 128 64 4 --warmup 20 --iters 100 \
  --evidence /tmp/super3-mamba-m1.jsonl
```

Add `--flashinfer-w4a16 cute-dsl` to race FlashInfer's native BF16 × NVFP4
backend against the same weights and independent oracle. Use a Python
environment containing FlashInfer with `mm_bf16_fp4` and
`prepare_bf16_fp4_weights` (the Super3 vLLM virtual environment has these).
`--flashinfer-w4a16 cudnn` selects its cuDNN backend when supported by the
installed cuDNN version. Weight repacking happens before timing. Both libraries
use their prepared graph replay paths with autotuning disabled; b12x uses the
explicit configuration above and FlashInfer uses its default tactic.
FlashInfer 0.6.17's CuTe DSL default accepts BF16 inputs but uses FP16 MMA
internally; passing the oracle on these synthetic inputs does not establish
the same numerical range as b12x's BF16 MMA.

The profile supplies geometry; `fp4-a16` selects BF16 activations with inline
NVFP4 weight decoding. Weights are synthetic, independently decoded for the
correctness oracle. Timings use CUDA graph replay with L2 eviction, including
split-K reduction, and evidence records raw samples and GPU/source identity.
Every arm must pass finite/nonzero, relative-error and cosine checks, including
replay after poisoning the captured output. Trials alternate backend order;
ratios are **b12x A16 time / baseline time**, so values below one favor b12x.
The A16 arm is compared with separately validated activation-quantized arms;
those arms have different activation precision and are not the serving baseline.
Use a fresh evidence filename for each run. Add `2` to `--batch-sizes` for
two-token decode; replace `--a16-config 128 64 4` with `--tune-a16` to race
the existing tile/split configurations. Keep the GPU free of serving requests
while collecting isolated timings.

For controlled diagnosis of the Mamba gap, use
`benchmarks/profile_dense_w4a16.py --config 128 64 4 --config 128 128 4
--evidence /tmp/mamba-k-tile.jsonl` in the same FlashInfer environment. It
compares the production b12x plans, FlashInfer's public default, and a
diagnostic FlashInfer BF16-MMA specialization. Add `--profile-arm all` under
Nsight Compute with `--profile-from-start off` to expose one qualified,
cold-L2 graph replay per arm and shape. Use the unprofiled paired timings for
latency comparisons; profiler replay affects timing.

The GB10 investigation on 2026-09-24 found that K64 was the dominant cause of
the serving gap. Row-major NVFP4 makes each K64 weight-row transfer only 32
bytes. K128 doubles the transfer width and nearly halves L2 read requests:
1.373 million to 0.780 million for `in_proj`, with identical read-sector
and read-miss counts. Producer memory-dependency and barrier waits dropped
sharply. Holding N128, split-K4 and BF16 arithmetic fixed, the confirmation
run measured 304.1 → 175.6 µs for `in_proj` and 158.9 → 89.5 µs for `out_proj`;
FlashInfer measured 190.5 and 88.1 µs. Its BF16-MMA specialization measured
189.4 and 86.0 µs, ruling out its FP16 arithmetic as the source of this gap.
These are synthetic, M1 geometry results. The generic fallback uses K64, but
Super3 should run with autotuning enabled: its launcher must not set
`B12X_AUTOTUNE=0` or `enable_b12x_autotune=false`. The existing A16 search
already races N64/N128, K64/K128/K256 and split-K 1/2/4/8: 24 candidates
per decode geometry, expanding to 72 with M16/M32/M64 at the 256-row
prefill capacity. The integration supplies geometry and workspace capacity
without pinning those choices, and preserves this checkpoint's BF16 activation
requirement. Pipeline depth follows the tile's shared-memory budget. Wide Q8_0
tiles with FP32 split-K output use a smaller epilogue tile when necessary to
fit even one input stage; they remain eligible for tuning.

Raw timing evidence is in
`/tmp/super3-root-cause-20260924T054121Z`; NCU reports, metrics, and SASS are in
`/tmp/super3-root-cause-20260924T053659Z` on the investigation machine.
