# SparkInfer MX-FP6: classification, accuracy (KLD) and performance results

Reference numbers for the SparkInfer MX-FP6 quantization as validated on the
Qwen3.6 family. Companion document: `fp6-user-guide.md` (how to make and run
an FP6 quant). Kernel-level format details: `mxfp6-w6a8.md`; vLLM wiring:
`mxfp6-vllm-integration.md`.

**Test hardware:** 1-2x NVIDIA RTX PRO 6000 (Blackwell, sm_120, 96 GiB).
**Models:** `Qwen3.6-27B` (dense hybrid, 64 layers) and `Qwen3.6-35B-A3B`
(MoE hybrid, 256 experts / top-8, ~3B active).

---

## 1. How the quant is classified

The precision story has two distinct stages — on disk and at runtime — and
both matter when describing the quant:

| Stage | Weights | Activations | Shorthand |
|---|---|---|---|
| On disk (storage) | MX-FP6 E2M3, packed 6-bit + UE8M0 block scale per 32 values | not quantized (model I/O stays BF16) | **W6A16** (weight-only) |
| At runtime (execution) | MX-FP6 E2M3, streamed to the GEMM | quantized on the fly per forward to FP8 **E4M3** + UE8M0 block scales + per-row global scale | **W6A8** |

Points worth spelling out:

* **On disk it is a weight-only quant.** Only 2-D Linear weights are FP6;
  activations are never stored quantized, so the checkpoint's interface
  precision is BF16 — W6A16 in the usual storage nomenclature. Norms,
  embeddings, `lm_head`, the MTP head, router gates and (by default) the
  vision tower remain BF16.
* **At runtime the GEMMs execute W6A8.** The default export
  (`--source-format mxfp6_w6a8`) records FP8 E4M3 as the runtime activation
  format; every FP6 linear and MoE expert quantizes its BF16 input
  activations on the fly (in-kernel for the decode path) and runs the
  matmul on the Blackwell `mxf8f6f4` block-scaled MMA. Legacy pairings
  (E3M2/E2M3 activations, true W6A6) remain selectable at export time.
* **The `config.json` tag says `W6A6`.** The checkpoint declares
  `quantization_config = {"quant_method": "modelopt", "quant_algo": "W6A6"}`.
  This is the *detection tag* the vLLM plugin keys on (mirroring ModelOpt
  NVFP4 key layout), not a precise statement of runtime activation width —
  the actual activation format is carried separately in the export and
  resolved per layer at load time.

Per-value weight cost is 6 bits + 8/32 bits of block scale ≈ **6.25
bits/value**, i.e. ~2.56x smaller than BF16 and below FP8 checkpoints for
the covered Linears.

---

## 2. Accuracy — KLD vs the BF16 reference

Kullback-Leibler divergence of the FP6 model's logits against the BF16
reference model, wikitext-2-raw-v1, context 2048, stride 512, deterministic
eager scoring (`TORCH_COMPILE_DISABLE=1`), measured Jul 23 2026 after the
correctly-rounded-division re-baseline:

| Model | Mean KLD | Determinism |
|---|---|---|
| Qwen3.6-27B (dense, W6A8 runtime) | **0.034599** | bit-identical across 3 runs, including the `SPARKINFER_DENSE_PER_ROW_IN_KERNEL=0` host-chain fallback |
| Qwen3.6-35B-A3B (MoE, W6A8 runtime) | **0.015388** | bit-identical across 2 runs |

Notes:

* KLD is **bit-deterministic**: repeated runs under the documented scoring
  configuration reproduce the exact value. Any deviation indicates a bug,
  not noise.
* The fused in-kernel per-row activation quantizer and the host-side chain
  produce bit-identical outputs (validated by the dense three-way run and by
  the unit suite `tests/quantization/test_fp6_small_m_quant.py`).
* MoE scores substantially better than dense: the BF16 router, per-expert
  quantization scope, expert redundancy, and the smaller quantized share of
  the forward pass all reduce the divergence.
* History: prior to Jul 23 the constants were 0.034423 (dense) and 0.011016
  (MoE), measured with torch's not-always-correctly-rounded CUDA f32
  division in the per-row scale chain. The chain now uses correctly rounded
  division (bit-identical to the kernel's `div.rn.f32`), which re-baselined
  both constants.

---

## 3. Performance — single-stream serving benchmarks

All numbers from `vllm bench serve` against a warm server, random dataset,
output 256 tokens, `--max-concurrency 1 --temperature 0`, MTP speculative
decoding enabled (`qwen3_next_mtp`, 2 speculative tokens). TPOT (time per
output token) is the cleanest kernel-level metric; headline tok/s at 256
output tokens is dragged down by TTFT amortization, and MTP acceptance
varies run to run with the random prompts.

### 3.1 Qwen3.6-27B dense, TP=1 — context sweep (Jul 23, post-fix, warm)

| Input ctx | Output tok/s | Mean TPOT (ms) | Mean ITL (ms) | Mean TTFT (ms) | MTP accept len |
|---|---|---|---|---|---|
| 1k  | 92.27 | 9.80  | 33.79 | 274  | 3.47 |
| 4k  | 99.23 | 8.93  | 35.78 | 303  | 4.04 |
| 8k  | 94.75 | 9.37  | 38.46 | 312  | 4.14 |
| 16k | 85.54 | 10.39 | 43.89 | 343  | 4.26 |
| 32k | 66.19 | 13.41 | 54.92 | 449  | 4.12 |

Decode-dominated confirmation (input 1024 / output 1024, TTFT amortized):
**106.94 tok/s**, TPOT 9.09 ms, acceptance length 3.75 — at parity with the
pre-rebase 108.5 tok/s baseline. The TPOT/ITL growth across the sweep is
KV-attention cost, not quantization overhead.

### 3.2 Qwen3.6-27B dense, TP=2 — context sweep (Jul 24, post-fix)

| Input ctx | Output tok/s | Mean TPOT (ms) | Mean ITL (ms) | Mean TTFT (ms) | MTP accept len |
|---|---|---|---|---|---|
| 1k  | 107.11 | 7.49 | 25.25 | 481  | 3.39 |
| 4k  | 100.38 | 6.51 | 26.28 | 891  | 4.06 |
| 8k  | 87.56  | 7.06 | 28.93 | 1123 | 4.15 |
| 16k | 66.42  | 8.30 | 34.33 | 1738 | 4.18 |
| 32k | 42.73  | 9.86 | 45.30 | 3477 | 4.63 |

### 3.3 Qwen3.6-35B-A3B MoE, TP=1 — context sweep (Jul 24, post-fix)

| Input ctx | Output tok/s | Mean TPOT (ms) | Mean ITL (ms) | Mean TTFT (ms) | MTP accept len |
|---|---|---|---|---|---|
| 1k  | 155.33 | 4.28 | 16.69 | 557  | 3.93 |
| 4k  | 147.84 | 4.34 | 17.85 | 625  | 4.14 |
| 8k  | 143.69 | 5.16 | 19.54 | 465  | 3.81 |
| 16k | 109.73 | 6.28 | 23.14 | 731  | 3.70 |
| 32k | 87.46  | 6.86 | 30.17 | 1177 | 4.44 |

### 3.4 Qwen3.6-35B-A3B MoE, TP=2 — context sweep (Jul 24, post-fix)

| Input ctx | Output tok/s | Mean TPOT (ms) | Mean ITL (ms) | Mean TTFT (ms) | MTP accept len |
|---|---|---|---|---|---|
| 1k  | 183.50 | 3.94 | 16.42 | 391 | 4.21 |
| 4k  | 187.03 | 3.90 | 16.28 | 373 | 4.19 |
| 8k  | 168.65 | 4.57 | 17.97 | 352 | 3.96 |
| 16k | 128.62 | 5.51 | 21.40 | 585 | 3.91 |
| 32k | 91.10  | 7.11 | 28.33 | 997 | 4.00 |

### 3.5 Reading the matrix

* **Dense TP=2 helps decode latency:** TPOT 9.80 -> 7.49 ms at ctx-1k
  (~24% faster per token) — the 27B GEMMs are large enough that splitting
  them beats the NCCL all-reduce overhead. TTFT is higher at TP=2 in these
  runs (prefill pays the disabled custom all-reduce and per-launch sync).
* **MoE gains less from TP=2:** TPOT 4.28 -> 3.94 ms at ctx-1k (~8%). With
  only ~3B active parameters per token, per-GPU compute is small and sync
  overhead eats most of the split; TP=2's main value for this model is
  capacity, not single-stream latency.
* MoE decode is ~2.3x faster than dense at short context (TPOT 4.28 vs
  9.80 ms), consistent with the active-parameter ratio.
* Headline tok/s at 256 output tokens mixes TTFT into the average — when
  comparing configurations, TPOT is the kernel-level metric; MTP acceptance
  (which varies run to run with the random prompts) explains most residual
  tok/s spread at equal TPOT.
