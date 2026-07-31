# SparkInfer MX-FP6: classification, accuracy (KLD) and performance results

Reference numbers for the SparkInfer MX-FP6 quantization as validated on the
Qwen3.6 family. Companion document: `fp6-user-guide.md` (how to make and run
an FP6 quant). Kernel-level format details: `mxfp6-w6a8.md`; vLLM wiring:
`mxfp6-vllm-integration.md`.

**Test hardware:** 1-2x NVIDIA RTX PRO 6000 (Blackwell, sm_120, 96 GiB).
**Models:** `Qwen3.6-27B` (dense hybrid, 64 layers) and `Qwen3.6-35B-A3B`
(MoE hybrid, 256 experts / top-8, ~3B active). Section 3.5 adds
`Behemoth-R1-123B-v2` (dense, 88 layers) as a large-model data point.

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
eager scoring (`TORCH_COMPILE_DISABLE=1` and `VLLM_USE_V2_MODEL_RUNNER=0`).
Both Qwen rows were re-baselined on Jul 29 against the current nightly; the
Behemoth row is the Jul 28 value, reproduced exactly across the 3.6c epilogue
change.

| Model | Mean KLD | Determinism |
|---|---|---|
| Qwen3.6-27B (dense, W6A8 runtime) | **0.033892** | bit-identical across 2 runs (Jul 29 re-baseline; see below) |
| Qwen3.6-35B-A3B (MoE, W6A8 runtime) | **0.014908** | bit-identical across 2 runs (Jul 29 re-baseline; same cause as Qwen dense) |
| Behemoth-R1-123B-v2 (dense, W6A8 runtime) | **0.009547** | reproduced exactly across the 3.6c epilogue change (TP=4) |

Notes:

* KLD is **bit-deterministic**: repeated runs under the documented scoring
  configuration reproduce the exact value. Any deviation indicates a bug,
  not noise.
* The Behemoth row is the accuracy gate on the 3.6c epilogue change. It is a
  reproduction, not a tolerance check: the same 0.009547 to all six digits,
  which is what a change that moves output staging without touching
  accumulation order is required to produce. The four-shard `check vs` lines
  were already identical; this confirms it survives 88 layers of composition.
* Scoring requires `VLLM_USE_V2_MODEL_RUNNER=0`. Score mode (`kld_mode`,
  `return_prompt_logits`) exists only in the V1 model runner; the V2 runner
  builds its `ModelRunnerOutput` without `kld_result_dict`, so every window
  silently contributes zero positions and the run ends in "No valid positions
  for KLD calculation" after a full pass. Recent vLLM defaults the Mistral
  architectures to V2, so this is now load-bearing for Behemoth.
* The fused in-kernel per-row activation quantizer and the host-side chain
  produce bit-identical outputs (validated by the dense three-way run and by
  the unit suite `tests/quantization/test_fp6_small_m_quant.py`).
* MoE scores substantially better than dense: the BF16 router, per-expert
  quantization scope, expert redundancy, and the smaller quantized share of
  the forward pass all reduce the divergence.
* Jul 29 Qwen re-baseline, 0.034599 -> 0.033892, cause outside SparkInfer.
  The checkpoint and the reference logits directory are byte-for-byte the ones
  that produced 0.034599, and the run reproduces 0.033892 exactly. It is not
  the epilogue policy: `SPARKINFER_DENSE_EPI_TILE=128x128` restores the
  pre-policy full-tile epilogue on every tile shape and still gives 0.033892.
  It is not anything else SparkInfer landed since Jul 23 either — reverting
  `LARGE_M_TILE`, `ROW_SCALE_EPILOGUE`, `SF_COPY_MODE`, `LARGE_M_UNROLL`,
  `DECODE_STAGE3`, `PACKED_B_EXPAND_LARGE_M`, `PACKED_B_EXPAND_AHEAD` and
  `PERSISTENT_SCRATCH` to their pre-fix values *simultaneously* still gives
  0.033892. The remaining variable is the vLLM nightly, which advanced
  substantially over that window and now reports `Enabled custom fusions:
  norm_quant, act_quant` — fusions that change rounding in the norm and
  activation path, which SparkInfer does not own. Not isolated further, since
  the shift is small, in the favourable direction, and outside our code.
  Behemoth is unaffected because its 0.009547 baseline was taken recently.
* Jul 29 MoE re-baseline, 0.015388 -> 0.014908, same picture and not isolated
  separately: two runs agree exactly, the shift is small and favourable, and
  the epilogue work never reached the MoE kernels at all (3.6d), so it tracks
  the same nightly drift as the dense row.
* History: prior to Jul 23 the constants were 0.034423 (dense) and 0.011016
  (MoE), measured with torch's not-always-correctly-rounded CUDA f32
  division in the per-row scale chain. The chain now uses correctly rounded
  division (bit-identical to the kernel's `div.rn.f32`), which re-baselined
  both constants.

---

## 3. Performance — single-stream serving benchmarks

Sections 3.1-3.4 come from `vllm bench serve` against a warm server, random
dataset, output 256 tokens, `--max-concurrency 1 --temperature 0`, MTP
speculative decoding enabled (`qwen3_next_mtp`, 4 speculative tokens). TPOT
(time per output token) is the cleanest kernel-level metric; headline tok/s at
256 output tokens is dragged down by TTFT amortization, and MTP acceptance
varies run to run with the random prompts.

Sections 3.1b, 3.3b and 3.5 use `bench-32k-sweep.sh`, run three times against
a server launched with `ENABLE_PREFIX_CACHING=0`:

```bash
ENABLE_PREFIX_CACHING=0 CUDA_VISIBLE_DEVICES=0 MODEL_DIR=<fp6 checkpoint> ./<model>-fp6.sh <hash>
./bench-32k-sweep.sh <hash> <served-name> <fp6 checkpoint> http://localhost:8001
```

Prefix caching off is mandatory, not hygiene — see `fp6-user-guide.md` 2.6b for
the 7.5x phantom prefill it produces otherwise.

**Two harnesses appear in this section and their columns are not
interchangeable.** Sections 3.1-3.4 report TTFT and an amortized output tok/s;
3.1b, 3.3b and 3.5 use the later true-prefill sweep script shown above, which
separates prefill throughput from decode. The older TTFT column is not a
prefill measurement:
3.1 records 449 ms for a 32k prefill, i.e. ~73k tok/s, which on a 27B dense
model is about 3.9 PFLOP/s and therefore above what the card can deliver.
Prefix caching was evidently on. Compare within a harness, never across.

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

### 3.1b Qwen3.6-27B dense, TP=1 — true-prefill sweep (Jul 29, post-3.6c)

Same script as 3.5, 3 passes, median shown. Run-to-run spread was 0.4% on
TTFT at 1k and under 0.1% elsewhere; decode repeated to the hundredth of a
tok/s at every context.

| Input ctx | TTFT (ms) | True prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| 1k  | 186.54  | 5489 | 110.86 |
| 4k  | 582.74  | 7029 | 101.01 |
| 8k  | 1104.83 | 7415 | 109.41 |
| 16k | 2243.13 | 7304 | 100.30 |
| 32k | 4723.62 | 6937 | 82.58  |

This is the post-epilogue-policy state, not a before/after: no true-prefill
sweep of Qwen exists from before 3.6c, and 3.1 is a different harness. Nor is
decode comparable across the two: 3.1's decode-dominated check ran 1024 in /
1024 out with TTFT amortized in, while 3.1b reports a separated decode rate at
each context, and the prompt set and MTP acceptance differ. 106.94 there and
110.86 at 1k here are the same order, which is all that can be claimed — this
section is **not** a decode regression check. Prefill shows no hole at any
context, which is the signature a bad `_choose_epilogue` pick would leave.

Two observations recorded rather than chased:

* Decode is non-monotonic — 4k (101.01) sits below 8k (109.41) — and it
  reproduces to the hundredth across all three passes, so it is deterministic
  rather than noise. Most likely MTP acceptance length differing on the fixed
  prompt set for those contexts.
* Fitting the 1k and 4k points gives a fixed cost near **55 ms** and a
  marginal rate near 7800 tok/s, so roughly 29% of the 1k TTFT is overhead
  that does not scale with tokens. Consistent with the standing short-context
  diagnosis: fixed per-chunk cost, not a GEMM gap, and not addressable by tile
  or epilogue tuning.

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

### 3.3b Qwen3.6-35B-A3B MoE, TP=1 — true-prefill sweep (Jul 29, pre-MoE-port)

Same script as 3.1b and 3.5, `ENABLE_PREFIX_CACHING=0`, 3 passes, median shown.
This is a **pre-port baseline**: the MoE kernel still hardwires its epilogue to
the MMA tile (`dynamic.py:904`), so 3.6c has not touched this path.

| Input ctx | TTFT (ms) | True prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| 1k  | 103.15  | 9927  | 255.75 |
| 4k  | 192.04  | 21329 | 216.45 |
| 8k  | 364.98  | 22445 | 206.19 |
| 16k | 722.49  | 22677 | 192.68 |
| 32k | 1557.95 | 21033 | 134.41 |

Pass 1 is a warm-up outlier at 4k and 8k (13576 and 14951 tok/s against ~21400
and ~22450 on passes 2 and 3, which agree with each other to 0.7% and 0.1%);
1k, 16k and 32k are stable from the first pass. Take the median of three, or
discard pass 1. Decode carries a wider spread than the dense model — 238.66 to
268.10 at 1k, about 11% — which is MTP acceptance varying with the prompt set,
so treat decode differences below ~10% here as noise.

Shape of the curve: prefill saturates around 22.5k tok/s by 8k and holds, with
1k at 9927 — 44% of saturation. The same fixed-overhead story as the dense
model, and more pronounced, since a 3B-active forward pass amortizes a fixed
cost over less work.

### 3.4 Qwen3.6-35B-A3B MoE, TP=2 — context sweep (Jul 24, post-fix)

| Input ctx | Output tok/s | Mean TPOT (ms) | Mean ITL (ms) | Mean TTFT (ms) | MTP accept len |
|---|---|---|---|---|---|
| 1k  | 183.50 | 3.94 | 16.42 | 391 | 4.21 |
| 4k  | 187.03 | 3.90 | 16.28 | 373 | 4.19 |
| 8k  | 168.65 | 4.57 | 17.97 | 352 | 3.96 |
| 16k | 128.62 | 5.51 | 21.40 | 585 | 3.91 |
| 32k | 91.10  | 7.11 | 28.33 | 997 | 4.00 |

### 3.5 Behemoth-R1-123B-v2 dense, TP=2 — context sweep (Jul 28)

A second, much larger dense model on 2x RTX PRO 6000. Measured with
`bench-32k-sweep.sh`, `--max-num-seqs 2`, `max-num-batched-tokens 8192`,
`cudagraph_mode=full_decode_only`, `custom_ops=["+rms_norm","+silu_and_mul"]`,
and `--disable-custom-all-reduce` (forced: vLLM's custom all-reduce crashes
during graph capture on sm_120 at TP>1). Three consecutive passes; the
run-to-run spread is +/-0.03 tok/s on decode.

| Input ctx | TTFT (ms) | True prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| 1k  | 450.2   | 2274 | 26.98 |
| 4k  | 1477.5  | 2772 | 26.77 |
| 8k  | 2998.8  | 2732 | 26.41 |
| 16k | 6451.6  | 2539 | 25.69 |
| 32k | 14773.4 | 2218 | 24.38 |

Prefill here is 8.1-14.2% above the same sweep taken before the epilogue fix in
3.6c (2104 / 2428 / 2429 / 2281 / 2024 tok/s, TTFT 486.6 / 1687.2 / 3373.2 /
7183.6 / 16190.2 ms). Decode rose 0.6-0.7% across the board, which is *not*
attributable to that change — the decode tile is provably untouched by it — and
is most likely the card starting the decode phase cooler now that prefill
finishes sooner.

### 3.6 Behemoth decode is at the weight-streaming DRAM roofline

At 123B parameters the FP6 checkpoint is ~91 GiB, or ~46 GiB per GPU after
the TP=2 shard, and single-stream decode reads essentially all of it per
token. 46 GiB at 26.8 tok/s requires **1.32 TB/s** sustained; Nsight Compute
measures 1.07-1.38 TB/s achieved on the four decode GEMM shards. Decode is
therefore bandwidth-bound, not latency-bound, on models of this size.

The practical consequence is that decode GEMM latency work has a poor
conversion rate here. The shape-dependent 3-stage decode policy cut isolated
decode GEMM time 6.1% (455.9 -> 427.9 us across the four shards) and returned
+1.9% end to end; what it actually bought was achieved bandwidth (DRAM
throughput 59-71% -> 64-75%), not the latency it removed. Optimizations that
do not either reduce bytes moved or raise achieved bandwidth should not be
expected to move decode on a model this large.

Note this is a size effect, not an FP6 property: the Qwen3.6-27B sweeps above
are far from this wall, which is why occupancy and pipeline work paid there.

### 3.6b Negative result: the 32-wide N decode tile cannot raise CTA count

Achieved occupancy on the decode shards is ~6.6% against a theoretical 12.5%,
with shared memory permitting 2 blocks/SM. The obvious reading is that the
shards launch too few CTAs (`down`/`o` 192, `qkv` 112, against 188 SMs) to ever
place a second block, so a 32-wide N tile was built to double them.

**It does not work, and the premise was wrong.** Decode runs the persistent
tile scheduler, whose grid is sized from resident-CTA capacity rather than the
output-tile count. Halving the tile width doubles the work tiles each CTA loops
over and creates no CTAs. Measured on `down` (1x12288x14336), Jul 28 2026, RTX
PRO 6000 GPU-41235b51:

| | (16,64) | (16,32) |
|---|---|---|
| grid | 192 | **188** (not 384) |
| DRAM throughput | 71.3% | 39.8% |
| duration (ncu) | 109.1 us | 195.7 us |
| duration (bench) | 96.2 us | 145.2 us |
| dynamic smem/block | 37.89 KB | 38.91 KB |

Shared memory per block *rises* because `sm120_make_smem_layout_sfb` rounds any
tile up to a full 128-column SF block, so narrowing N frees nothing and
quadruples SFB reads. Numerics were byte-identical (cos 0.9992541075, max_abs
0.68877006), as expected: tile width changes column ownership, not accumulation
order.

The occupancy diagnosis above survives; only this fix for it is dead. The one
remaining lever that genuinely changes CTA count is split-K, which bypasses the
persistent scheduler for a `(1, slices, n_tiles)` grid — at the cost of
changing FP32 accumulation order, so it is not bit-identical to the current
kernel and would need a fresh KLD gate rather than a byte comparison.

### 3.6c The epilogue was sizing the mainloop pipeline

The epilogue staging tile used to be hardwired to the MMA tile. Because
`epi_stage_max` is `(tile_m/epi_m) * (tile_n/epi_n)`, that pinned `epi_stage` at
1 and reserved the whole output tile in shared memory. At (128,64) it costs
16384 B of 99328 and is invisible. At (128,128) it costs 32768 B and leaves
66560 against 33792 per mainloop stage — `ab_stage=1`, no pipeline at all.

`_choose_epilogue` now takes the largest epilogue that does not cost a mainloop
stage, which leaves (128,64) and the (16,64) decode tile exactly as they were
and moves (128,128) to a (64,64) tile at 2 stages, 16384 B. The wide-N prefill
default moved to (128,128). Behemoth TP=2 shards at M=8192, expanded-B, 200
iters, `check vs` identical between arms on every shard:

| shard | (128,64), 3 stages | (128,128), 2 stages | |
|---|---|---|---|
| qkv     | 2552.8 us / 565 TF | 2090.0 us / 691 TF | **-18.1%** |
| o       | 2170.9 us / 570 TF | 1754.1 us / 705 TF | **-19.2%** |
| down    | 5357.6 us / 539 TF | 4427.9 us / 652 TF | **-17.4%** |
| gate_up | 10435.5 us / 553 TF | 8532.0 us / 677 TF | **-18.2%** |

Three earlier records were confounded by this and are retired, not merely
outvoted. The Jul 26 tile sweep that chose (128,64) as the M-independent
prefill winner, the "(128,128) is 12-19% slower" re-sweep, and the
"(128,128) +36% at every stage depth" cross-product all measured an
unpipelined wide tile against a pipelined narrow one. The lesson is that a
tile sweep is only valid if the resolved `ab_stage` is recorded alongside each
arm; `_compute_stages` now logs it at DEBUG for exactly this reason.

Two preconditions on the epilogue tile are load-bearing and neither fails
loudly. It must be a whole number of MMA atom tiles, because the accumulator
copy loop trips `epi_m // mma_tile_m` times and a sub-atom tile floors that to
zero, TMA-storing uninitialized shared memory. And the epilogue must actually
stage through shared memory: the `m=1` path (`use_m1_non_tma_c`) stores
straight out of registers, so a sub-tiled multi-stage buffer describes
something it does not do. Both produce NaN or garbage rows rather than an
error, and both were caught only by the `atol=0.0` bit-exactness tests.

Accuracy gate: Behemoth KLD reproduces at exactly 0.009547 after the change
(section 2), at TP=4 rather than the TP=2 used for the shard benchmarks. Every
Behemoth shard stays above the N>1536 wide-N threshold at either TP, so the
selector resolves (128,128) in both cases and this is not a new tile regime —
but the row-parallel shards do run a different K, and the value held.

### 3.6d The MoE epilogue is hardwired too, and it does not matter

MoE carries the same construct 3.6c fixed — `dynamic.py:904` pins `epi_tile` to
the MMA tile, and the resulting `sC` enters the fixed shared-memory
reservation at `:1163`. The FP6 MoE kernel is also unpipelined. Measured:

```
dynamic stages: tile=(128, 128, 128) gated=True ab_stage=1 (dense suggested 1,
  max_fit 1) epi_tile=(128, 128) epi_stage=1 per_stage=50736 fixed=44352
  (sC 32768) capacity=101376
```

But the epilogue is not the binding constraint here, and porting
`_choose_epilogue` would be a no-op. The per-stage footprint is twice dense's:

| | bytes |
|---|---|
| sA (128x128, one E4M3 byte per element) | 16384 |
| sB x2 — gated FC1 keeps gate and up | 32768 |
| sSFA + sSFB x2 | ~1536 |
| mbar | 48 |
| **per stage** | **50736** |

`(101376 - 44352) / 50736 = 1.12`, hence one stage. Delete the epilogue
*entirely*, not merely shrink it, and it is `(101376 - 11584) / 50736 = 1.77`
— still one stage. Two stages need `per_stage <= 44896`, an 11.5% cut. So the
policy's two candidates would both fail the probe and the full tile would win
on the tie, producing a byte-identical kernel.

The contrast with 3.6c is the whole point: dense's per-stage was 25600 against
a 32768 epilogue, so the epilogue was *larger* than a mainloop stage and
freeing it bought one outright. Here it is two thirds of a stage. Same
construct, opposite conclusion — which is why the budget has to be computed
per kernel rather than inferred from the dense result.

Two consequences. There is no confounded MoE tile record to retire, because
only (128,128) is ever built for `w6a8_mx` (the ctor rejects the rest and
`_select_dynamic_tile_mn` returns the fixed tile unconditionally), so no MoE
tile sweep was ever run under this. And if MoE pipelining is worth pursuing at
all, the lever is the doubled B buffer — 32768 of the 50736, 65% of the
per-stage cost — which is a structural change to gated FC1 staging, not a
policy port. That should not be started without a profile establishing the
kernel is pipeline-starved rather than bound elsewhere; `ab_stage=1` proves
the pipeline is absent, not that adding one would pay.

### 3.7 Reading the matrix

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
