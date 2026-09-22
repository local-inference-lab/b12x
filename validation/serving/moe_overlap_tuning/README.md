# MoE tuning with concurrent shared experts and varied routing

Startup selection measures small routed-expert candidates together with the
shared FP8 expert and supported routing gate that overlap them in serving.
Five deterministic route-sharing levels prevent the race from considering
only one distinct-expert count. B12X chooses the launch configuration; no
model-specific or GPU-specific grid is pinned.

Status: **implemented**; the DeepSeek V4 Flash serving and CPU checks below
are **qualified**.
Cross-model release qualification is recorded in the
[serving report](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/karmic-kraken-serving.md).
That matrix qualifies the composed release, not this PR in isolation. Loaded
microbenchmarks outside their declared clock envelope are **research-only**
and are not the source of the serving speedup claim.

## Conditions and result

DeepSeek V4 Flash uses two-way tensor parallelism (TP2) and DSpark speculative
decoding with up to five draft tokens per verification step. Attention KV is
FP8; the batch budget is 4096 tokens with eight request slots and request
temperature 1/top-p 1. C1 and C8 mean one and eight concurrent requests.
Hardware: two RTX PRO 6000 Blackwell Max-Q
Workstation GPUs, VRAM +6000, automatic graphics clocks and 325 W limits.
Each cell has five warmed, unprofiled 30-second decode windows or five cold
32K prefill windows. Prefill is prompt tokens divided by client time to first
token (TTFT), with
a unique prefix per request. C8 output is aggregate throughput.

| Median | Isolated routed-kernel tuning | Concurrent context and varied routes | Change |
| --- | ---: | ---: | ---: |
| 32K prefill, tokens/s | 11,427 | 11,392 | -0.31% |
| C1 output, tokens/s | 196.171 | 212.517 | +8.33% |
| C1 verifier, steps/s | 71.899 | 75.453 | +4.94% |
| C1 accepted length | 2.732 | 2.815 | +3.02% |
| C8 output, tokens/s | 660.109 | 667.505 | +1.12% |
| C8 verifier, steps/s | 247.371 | 249.682 | +0.93% |

C1 verifier relative standard deviation is 0.289%; C8 is 1.138%. These
unseeded requests have stochastic acceptance, so +8.33% output is not a
guaranteed token-rate gain. All arithmetic, cold/repeated/changed-prefix,
loop, admission and drain checks pass. Both images expose 1,301,500 logical
KV tokens. There is no measured prefill/C8 deficit beyond run variability.

[Raw results](results.json) include both source/image identities, checkpoint
revision, complete commands, all samples and busy GPU telemetry. The candidate
contains the activation-aware context key. The comparison retains the same
packed attention-output projection control from
[B12X #396](https://github.com/local-inference-lab/b12x/pull/396) and
[vLLM #812](https://github.com/local-inference-lab/vllm/pull/812), and the
resident-grid candidate extension from
[B12X #397](https://github.com/local-inference-lab/b12x/pull/397).
Those controls are unchanged between the two source-identified arms in
`results.json`. The comparison measures the **joint** context/corpus change,
not the corpus in isolation.

Recompute every median and reject failed/missing cells with:

```bash
uv run python validation/serving/moe_overlap_tuning/audit_evidence.py
```

## Why the execution context matters

A wide routed grid can delay the concurrent shared expert. A cooperative
thread array (CTA) is one CUDA thread block. The isolated
188-CTA selection has shared-down means of 52.62/48.77 microseconds on the
two ranks, while the automatic 128-CTA selection has 8.17/8.14 microseconds.
Each mean contains 172 target-layer calls. The 12 draft calls per rank retain
240 CTAs in both arms. [All intervals](intervals.json) are tied to the same
image/source identities as the unprofiled comparison.
These intervals overlap and must not be added to predict throughput.

`extract_intervals.py` extracts every routed interval through its PCIe
collective and records concurrent kernels without discarding initial steps.
Pass all four rank traces with repeated `--trace baseline-rank0=path`,
`--trace baseline-rank1=path`, `--trace candidate-rank0=path` and
`--trace candidate-rank1=path`, together with `--results results.json
--target-layers 43 --draft-layers 3 --compact --output intervals.json`.
The output retains trace SHA-256, launch geometry and every interval; the
compact format stores repeated kernel names once.

## Startup cost and correctness

[Startup counters](startup-cost.json) compare fixed-sharing and varied-sharing
races while retaining the same shared/gate context. Small-MoE measurement
time increases by **0.359/0.333 seconds per rank**, from 1.315/1.447 to
1.674/1.780 seconds.
These are incremental measurement counters, not a total startup-time claim.
Cached decisions skip the race; corpus versioning invalidates selections,
not compiled kernels. Serving qualification is external to startup.
`extract_startup_costs.py` attributes successive cumulative-counter increments
at `batch_end`; overlapping request start/end differences are not summed.
It rejects failed or unmarked preparation phases and nonfinite timing values
before exporting any summary.
The committed summary groups the extractor's traces under source-identified
arms and retains phases with nonzero `small_moe_measurement_s`. Within each
phase it renames `measurement_s` to `all_measurement_s`, removes the verified
`failed: false` flag and
`cumulative_seconds`, and retains only `moe.decode` races with token counts
2 through 8. Trace paths are relative to the validation workspace; their
SHA-256 values remain unchanged. No timing values are rounded or recomputed
by this summary transformation.

The PR's corpus/variant file passes 21 CPU tests. The composed source reports
161 CPU passes and eight CUDA skips; vLLM reports 19 context/ownership passes
and 180 startup passes with two skips. The five sharing regression cases fail
with the fixed-sharing generator. These tests cover deterministic expert IDs,
per-token uniqueness, bounded IDs, small expert pools, unchanged non-verifier
inputs, context cache identity, stream joins and bounded trial ownership.

```bash
uv run python -m pytest tests/moe/test_fused_moe_variant_selection.py -q
```

Kernel math, checkpoint values, the public `shared_40` workload, and routing
for single-token/prefill trials are unchanged. The five levels are a coverage
corpus, not a claim that production routing is uniformly distributed.
