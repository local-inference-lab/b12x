# FP6 in SparkInfer (B12X): design, decisions and implementation history

This document is the complete engineering record of adding MX-FP6 serving
support to SparkInfer (the library formerly named B12X): what we built, what
we reused, every consequential decision and the reasoning behind it. It is
the narrative companion to the reference docs:

* `mxfp6-w6a8.md` — kernel/format reference
* `mxfp6-vllm-integration.md` — vLLM plugin internals
* `fp6-user-guide.md` — how to make and serve an FP6 quant
* `fp6-results.md` — classification, KLD and performance numbers

---

## 1. Goal and guiding principle

The goal was to add a production FP6 (6-bit weight) serving path for
**SM12.0** GPUs (Blackwell workstation/consumer: RTX PRO 6000, GeForce
RTX 50 series) to SparkInfer, *following the existing developer's lead*:
reuse the library's proven block-scaled GEMM machinery, its MoE execution
model, its compile/cache infrastructure and its checkpoint conventions, and
add only what FP6 genuinely requires. FP6 was to slot in beside the
existing MXFP8, NVFP4 and W4A8 paths as a peer — same facades, same op
registration, same integration pattern — not as a parallel universe.

Hard requirements we held ourselves to throughout:

1. **Bit-deterministic accuracy scoring.** KLD of a served FP6 model must
   be exactly reproducible run to run, and numerically identical across
   every equivalent execution path (fused vs unfused, small-M vs large-M).
   Any deviation is treated as a bug, never as acceptable noise.
2. **No BF16 fallback on the quantized path.** Covered Linears execute
   their matmuls in 6/8-bit end to end.
3. **Safetensors only.** No pickle (`.pt`) persistence anywhere in the
   tooling or artifacts.
4. **Calibration-free.** A quant is produced from weights alone; no
   calibration dataset, no static activation scales.

## 2. Hardware target: SM12.0 first and foremost

Everything here is built on one instruction: the Blackwell block-scaled
`mxf8f6f4` MMA (`m16n8k32`), which consumes mixed FP8/FP6/FP4 operands with
UE8M0 group-of-32 block scales *natively* in the tensor core. SparkInfer
already drove this instruction for MXFP8; FP6 rides the identical
machinery with a different type qualifier on the emitted instruction.

Support posture:

* **SM12.0 — the target.** All development, validation, KLD scoring and
  benchmarking was done on RTX PRO 6000 (Blackwell, sm_120, 96 GB).
* **SM12.1 — expected to work, unvalidated.** It shares the SM12.0
  instruction set (the format reference says SM120/SM121), but no FP6
  run has been performed on one.
* **SM10.0 / SM10.3 (datacenter Blackwell) — not targeted.** Those parts
  use the `tcgen05`/tensor-memory programming model; SparkInfer's dense
  GEMM here is the SM120 port (see `sm120_dense_fp8_deepgemm_port.md`),
  and none of the FP6 work wires the tcgen05 path. It may be portable in
  principle; nothing about it is claimed.
* **Hopper and earlier — impossible as designed.** There is no
  `mxf8f6f4` MMA; an FP6 path there would mean software dequantization,
  which defeats the purpose.

## 3. The format: what lives where, and why

### 3.1 On disk: W6A16 (weight-only FP6)

The checkpoint stores **only weights** quantized: packed MX-FP6 E2M3 codes
(4 values in 3 bytes along K), unswizzled UE8M0 K/32 block scales, and a
per-tensor (dense) / per-expert (MoE) f32 global scale. Everything else —
norms, embeddings, `lm_head`, MTP head, router gates, and by default the
linear-attention projections and vision tower — stays BF16. Activations
are **never** stored quantized, so the checkpoint's interface precision is
BF16: **W6A16** in storage nomenclature, ~6.25 bits/weight-value for the
covered Linears.

Why weight-only on disk:

* **Activations are data-dependent.** Their scales cannot be known until
  the forward pass sees real data. Storing activation scales would mean
  *calibration* — running a proxy dataset through the model and hoping
  deployment traffic matches it. We deliberately rejected that: dynamic
  scales computed per forward (per row, per 32-block) adapt to the actual
  input every time, are robust to distribution shift, and make quant
  production a pure weight transform anyone can run in minutes.
* **The checkpoint stays self-describing and portable.** It loads (and
  can be dequantized for inspection) without any runtime component.
* The `config.json` declares
  `quantization_config = {"quant_method": "modelopt", "quant_algo": "W6A6"}`
  with ModelOpt-mirror tensor keys (`.weight` / `.weight_scale` /
  `.weight_scale_2` / `.input_scale`). This is a **detection tag** chosen
  so the serving plugin can claim the checkpoint through the same
  override hook the NVFP4/ModelOpt path uses — it names the weight scheme,
  not the runtime activation width (which is recorded separately by the
  export and resolved per layer at load).

### 3.2 At runtime: W6A8, not W6A6

Each forward pass quantizes the BF16 activations on the fly to **FP8
E4M3** codes with UE8M0 K/32 block scales, and the GEMM executes with an
8-bit A operand against the 6-bit B operand — **W6A8**. A true-W6A6 mode
(E3M2 or E2M3 activations) exists and remains selectable at export time
(`--source-format mxfp6_default` / `mxfp6_e2m3`), but W6A8 is the
production default, for measured reasons:

* **Activations are the hard operand.** Weights are static, smooth, and
  quantize well at 6 bits — especially with the MSE-refined block-scale
  selection (per block, try exponent ceil vs ceil−1, keep the lower
  error). Activations are heavy-tailed and spiky: within one 32-element
  block the dynamic range routinely exceeds what a 6-bit mantissa/exponent
  budget covers. E4M3's wider exponent range plus the extra mantissa bit
  roughly halves the runtime error slice. In the measured KLD attribution
  for Qwen3.6-27B, the total ~0.033-0.035 splits into a ~0.023 weight
  floor (irreducible without changing the weight format) and a ~0.011
  activation slice — E4M3 is what keeps that second slice small. With
  6-bit activations the activation slice, not the weight floor, dominates.
* **Zero extra silicon cost.** The `mxf8f6f4` MMA accepts an FP8 A operand
  against an FP6 B operand in the same instruction at the same throughput
  — mixed 8/6-bit is a first-class operand combination, not a
  special-cased slow path. Choosing E4M3 activations costs nothing at the
  tensor core.
* **Massive machinery reuse.** FP8-E4M3-with-UE8M0-scales is exactly the
  activation encoding of the existing `w4a8_mx` MoE path. Choosing it let
  the FP6 MoE reuse that activation quantizer, scratch layout and
  scheduling unchanged (§5).

The one place 6-bit activation codes still appear is the legacy pairing
for A/B experiments; nothing in production emits them.

### 3.3 Why the in-kernel activation scaling costs no performance

A reasonable worry: "you re-quantize every activation on every forward —
doesn't that eat the FP6 win?" It does not, for four reasons:

1. **The quant is tiny relative to the GEMM it feeds.** At decode
   (M ≤ 16), the small-M quantizer touches only the real rows — one thread
   per 32-element block, plain global loads — and measures ~4-5 µs per
   launch in serving profiles, against dense GEMMs an order of magnitude
   larger. At prefill, the TMA quantizer tiles M in 128 and amortizes to
   noise. The amax scan the scale derivation needs is a sub-microsecond
   L2-resident read at decode sizes and is computed redundantly per CTA
   precisely so no cross-CTA synchronization is ever needed.
2. **Everything around the quant is fused.** The entire per-row scaling
   recipe — row amax, `gs_r = numerator/amax_r`, the BF16 pre-scale, the
   quantization itself, the inverse-scale output for the epilogue
   correction, and the GEMM's alpha — happens inside the one quant kernel
   (`SmallMQuantKernel`, `per_row=True`). This was the hard-won lesson of
   the post-rebase performance regression: the same recipe as ~12 eager
   torch launches per linear cost ~1.7 ms/token at 27B decode scale
   (TPOT 11.47 → 9.79 ms/token once fused). Zero host-side launches
   remain on the hot path.
3. **Quantized operands *reduce* memory traffic.** The A operand leaves
   the quantizer at 1 byte/value instead of 2 (BF16); the B operand
   streams from HBM in its 3:4-packed 6-bit form and expands to
   byte-containers in shared memory (`b_packed=True`), saving 25% of
   B-side HBM traffic versus even the byte-container layout. Decode-side
   GEMMs are bandwidth-bound; the quant pays for itself.
4. **We never leave the 6/8-bit path.** There is no dequantize-to-BF16
   step anywhere between the quantizer and the accumulator: FP6 codes
   travel as `uint8` byte-containers through the FP8-shaped TMA/ldmatrix
   pipeline (there is no FP6 torch dtype; the CUTLASS element type is
   injected at pointer-construction time), and the per-K-block MMA is
   emitted as inline PTX with explicit format qualifiers
   (`emit_mxfp6_dense_mma_k_block`). The tensor core consumes FP8 A and
   FP6 B directly; scales are applied by the block-scale hardware and the
   f32 alpha epilogue. BF16 exists only at the input edge (activation
   load) and the output edge (accumulator writeback).

## 4. What we reused from SparkInfer (B12X), and why

The single biggest design decision was to treat FP6 as a *format variant*
of existing paths rather than a new engine. Reused wholesale:

* **The dense block-scaled GEMM** (`_lib/dense_gemm.py`): TMA loads, smem
  swizzles, ldmatrix, the persistent tile scheduler, warp specialization,
  the alpha epilogue — the entire MXFP8 pipeline. FP6 enters via
  `ab_dtype="float6_e2m3fn"` plus per-operand `a_fmt`/`b_fmt` (so W6A8 can
  mix an FP8 A with an FP6 B) and swaps only the MMA emission.
* **The `w4a8_mx` MoE machinery** (`moe.fused_moe`, quant mode
  `w6a8_mx`): the MXFP8-E4M3 activation quantization, per-expert alpha
  preparation, the unified dynamic persistent-grid scheduler, and the
  scatter/combine — all unchanged. FP6 contributes packed expert weights,
  swizzled scales, and the per-K-block FP6 MMA (`kernels/mxfp6_moe.py`).
  This reuse is *why* W6A8's E4M3 choice was so cheap to adopt.
* **The compile/cache infrastructure** (`_lib/compiler.py`):
  `KernelCompileSpec` explicit-fact cache keys, the content-addressed disk
  cache keyed on a whole-package source fingerprint plus toolchain, the
  in-memory caches. Every new FP6 kernel compiles through it. (During
  debugging we audited this end to end — the fingerprinting is sound, and
  it exonerated the cache when we were hunting a bit-exactness bug that
  turned out to be arithmetic; §6.3.)
* **The `plan`/`bind`/`run` facade and OpManifest lazy registration** —
  the upstream SparkInfer architecture (post-rename). All FP6 entry
  points register into the `sparkinfer::` custom-op namespace the same
  way the other quant modes do, which is what makes torch.compile mode-3
  and CUDA graphs work without special-casing (the FP6 linear is an
  opaque `sparkinfer::fp6_dense_linear` custom op).
* **The intrinsics library** (`_lib/intrinsics.py`): warp/block
  reductions, vectorized global loads, PTX store helpers — extended, not
  replaced (§5).
* **Checkpoint conventions**: the ModelOpt-mirror safetensors schema and
  the plugin-claims-modelopt detection trick mirror how the NVFP4 path is
  wired, so the vLLM side of FP6 looks exactly like the vLLM side of FP4.

## 5. What we wrote new, and why

* **FP6 MMA emission** (`_lib/dense_gemm_mxfp6.py`): the byte-container
  layout is not a workaround — it is the operand format the hardware
  mandates. The PTX ISA requires `e2m3`/`e3m2` MMA operands as one 6-bit
  value per byte with 2 zero padding bits, and `.kind::mxf8f6f4` does not
  accept fully-compressed sub-byte smem (that exists only for FP4 via
  `mxf4`/`mxf4nvf4`). A "native" 6-bit path therefore cannot exist on
  SM120; FP6's only density win is in HBM, which packed-B captures. The
  per-K-block MMA is emitted as inline PTX with explicit FP6/FP8 type
  qualifiers while the rest of the kernel reuses the FP8-shaped machinery.
* **Packed-B smem expansion** (`b_packed=True`): TMA-load B in 3:4-packed
  form, expand to byte containers in smem. Gated by
  `SPARKINFER_PACKED_B_MIN_N` (default 12288) because the expansion chain
  loses at small N where the GEMM is latency- not bandwidth-bound
  (measured crossover on RTX PRO 6000).
* **The activation quantizers**:
  * a TMA-based large-M BF16→FP6/FP8 quantizer (M tiled in 128) for
    prefill, and
  * `SmallMQuantKernel` (M ≤ 16) for decode/MTP-verify: one thread per
    32-block, only real rows written, in-kernel amax, in-kernel per-row
    scale recipe, alpha and inverse-scale outputs (§3.3, §6).
* **New PTX intrinsics** in support of bit-exactness: `div_rn_f32`
  (IEEE round-to-nearest f32 division — the DSL's `/` may lower to an
  approximate division), `cvt_f32_to_bf16_bits`, `st_global_u16`.
* **Offline weight quantization + loaders**
  (`quantization/mxfp6/fp6_dense_weights.py`, `fp6_moe_weights.py`,
  `model_fp6.py`, the safetensors export): golden-rule model walk,
  MSE-refined UE8M0 block scales, per-expert MoE packing, full HF
  checkpoint export with patched config and copied tokenizer files.
* **Tooling** (`scripts/quantize_model_fp6.py` and friends): point-at-a-
  model conversion with dry-run/inspect/error-report modes and the
  sensitivity knobs documented in `fp6-user-guide.md`.
* **The vLLM integration** (`sparkinfer/integration/vllm/`): a thin
  plugin (`plugin.py`) registered through vLLM's `general_plugins` entry
  point, plus a framework-agnostic serving layer (`fp6_serving.py`) that
  a fork of any engine can call. Upstream SparkInfer ships no integration
  layers, so this mirrors how the maintainer wires NVFP4 privately: glue
  in `integration/`, calling only public `plan`/`bind`/`run` APIs. A
  process-wide MoE scratch/plan cache deduplicated by geometry (not by
  layer object) keeps the footprint flat across ~40 identical MoE layers
  — per-layer caching multiplied scratch by the layer count and was fatal
  under vLLM's CUDA-graph memory estimator.
* **A small-N BF16 GEMV** (`gemm.bf16_gemv`) for narrow projections
  (e.g. GDN `in_proj_ba`, N ≤ 1024) where a GEMM tile wastes the CTA —
  these layers aren't FP6-quantized, but the hybrid Qwen3.6 stack made
  their cost visible once the FP6 GEMMs got fast.
* **A deterministic MoE combine mode**
  (`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1`): `route_output` + ordered
  top-k sum instead of atomic scatter-add. 1.3-4.4x slower on the combine,
  used for scoring only — it exists because requirement #1 (§1) is
  unsatisfiable with atomics deciding summation order.

## 6. The bit-exactness campaign

Three separate problems had to be solved to make "KLD is identical every
run, on every path" true. Each produced a design rule.

### 6.1 Batch-composition independence: per-row activation scaling

A per-tensor activation amax makes each row's quantization depend on every
*other* row in the batch — chunked prefill (vLLM splitting a prompt
differently between runs) then changes the logits. The fix: a per-row
global scale. Each row is pre-scaled so its amax maps to the format
numerator (E4M3: 200704 = 448², E2M3: 3360, E3M2: 12544 — all exactly
representable in f32), quantized with a unit global scale, and corrected
after the GEMM by `bf16(1/gs_r)`. M=1 deliberately takes the same
rounding chain so decode rows match their prefill twins bit-for-bit.

### 6.2 Compile-time and scheduling nondeterminism

Inductor's compile-time kernel selection perturbed logits between runs —
scoring therefore runs eager (`TORCH_COMPILE_DISABLE=1`); serving keeps
full compile+graphs since serving has no bit-repro contract. MoE's atomic
scatter-add combine was replaced for scoring by the deterministic combine
(§5). With both in place, KLD reproduces exactly.

### 6.3 Correctly-rounded arithmetic: the division story

The subtlest bug of the project. Fusing the per-row recipe into the
kernel produced a *single* code byte that differed from the host chain on
one row — deterministically. The trail led through (in order): suspected
stale compile caches (audited, sound), suspected kernel races (none), a
diagnostic that compared the wrong activation format, and finally to
this: **torch's CUDA f32 scalar/tensor division is not always correctly
rounded.** For `200704 / 2.625` torch returns `0x47955556`, one ulp above
the correctly-rounded `0x47955555` that the kernel's `div.rn.f32`
produces; that one ulp flips the bf16 rounding of one pre-scaled element
and one quant code. We refused to replicate torch's error in the kernel
(unspecified, version-dependent). Instead the host chain now divides in
f64 and casts to f32 — provably bit-identical to a correctly-rounded f32
division (the 2p+2 double-rounding rule) — so host and kernel agree on
every operand *by construction*. The rule that fell out: **every scale
derivation, host or device, must be correctly rounded; nothing may depend
on a library's unspecified rounding.** This re-baselined the KLD
constants (dense 0.034423 → 0.034599, MoE 0.011016 → 0.015388), verified
bit-identical across the fused path, the host-chain fallback and repeated
runs.

## 7. The B12X → SparkInfer rebase

Mid-project, upstream renamed B12X to SparkInfer and rewrote its
architecture (the `plan`/`bind`/`run` facade, the `sparkinfer::` op
namespace, OpManifest lazy loading, CUTLASS DSL 4.6). A `git rebase` was
unworkable against a rewrite, so the FP6 work was **forward-ported
feature by feature**: `_lib` plumbing and env-var renames first
(`B12X_*` → `SPARKINFER_*`, with legacy names still accepted), then dense
GEMM, the quantization stack, bf16 GEMV and MoE in parallel, then the
vLLM layer — rebuilt as the native `integration/` package rather than the
old external plugin, matching upstream's FP4 pattern. Two upstream shifts
mattered most: CUTLASS DSL 4.6's stricter tracing (several kernels needed
constexpr/dynamic-value hygiene fixes) and the lazy op registration
(anything touching `torch.ops.sparkinfer.*` must touch the module
attribute first). FP6 KV-cache attention work was dropped from scope
during the port; the FP6 linear/MoE path has no KV-cache dependency.

## 8. Validation summary

Full numbers in `fp6-results.md`; the shape of the evidence:

* **Unit level**: quantizer-vs-TMA bit-equality, fused-vs-host per-row
  bit-equality (every m ∈ {1,3,5,16} × format), end-to-end
  small-M-vs-M=128 row equality, compile guards.
* **Accuracy**: Qwen3.6-27B dense KLD **0.034599** (three runs identical,
  including the host-chain fallback); Qwen3.6-35B-A3B MoE **0.015388**
  (two runs identical). Dense sits at the measured W6A8 floor
  (~0.023 weight + ~0.011 activation); MoE lands in the FP8 band.
* **Performance** (RTX PRO 6000): dense 27B decode 9.79 ms/token TPOT at
  TP=1 (106.94 tok/s single-stream decode-dominated, parity with the
  pre-rebase baseline), 7.49 ms at TP=2; MoE 4.28 ms TPOT at TP=1,
  3.94 ms at TP=2; full 1k-32k context sweeps recorded for all four
  configurations. MoE gains little from TP=2 at single stream (~3B
  active params leave too little per-GPU work); dense gains ~24%.

## 9. Known limits and honest caveats

* SM12.0-validated only (§2). SM12.1 is expected-compatible, unproven;
  datacenter Blackwell and everything older are out of scope.
* The MoE deterministic combine is a scoring tool, not a serving default
  (combine cost 1.3-4.4x).
* The dense KLD floor (~0.023) is a property of 6-bit weights + UE8M0
  group-32 scales; levers measured and closed: MSE block scales (kept —
  ties or wins), GPTQ (skipped, ~0.002 est. gain), group-32 Hadamard
  rotation (rejected — measurably worse). Getting materially below the
  floor means changing the weight format, not tuning this one.
* Blackwell sm_120 + vLLM TP>1 requires `--disable-custom-all-reduce`
  (vLLM's custom all-reduce crashes during CUDA-graph capture; NCCL
  fallback costs ~1-3%).
* Seven pre-existing upstream regression-test failures (non-FP6 paths)
  were present before the port and are unrelated to it.
