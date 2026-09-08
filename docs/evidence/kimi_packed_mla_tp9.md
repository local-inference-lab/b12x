# Packed Kimi MLA verification on TP9

Status: **qualified for the recorded kernel cases**. The measurement record
`kimi_packed_mla_tp9.json` contains source identities, GPU operating state,
individual replay samples, and output/LSE digests from 2026-09-08.

Kimi-K3 under TP9 gathers 99 effective query heads. The dense adapter's
eight-head padding gives 104 heads, while the packed reader executes 16-head
tiles. Six full tiles and a remainder require two launches. Padding to 112
heads closes the tile; the adapter removes zero heads before DCP reduction.

The recorded workload uses four query rows, a 116,736-token local capacity,
64 split slots, 1,536-token pages, and 656-byte packed KV records. Each variant
uses the same 99 semantic heads and the same generated query/KV values.
All visible KV tokens remain selected.

| Reader | Padded heads | Split policy / partials | Correctness |
| --- | ---: | --- | --- |
| Reference source `04f246d7e5e6` | 104 | Static / BF16 | Reference digest |
| Vector shared-memory loads | 104 | Static / BF16 | Output and LSE bytes equal |
| Vector loads with whole head tiles | 112 | Static / BF16 | Output and LSE bytes equal |
| Vector loads with balanced work | 112 | Balanced / FP32 | Different association; recorded FP32-reference errors |

At 2,048/8,192/16,384 local tokens and two query amplitudes, the six static
cases preserve output and LSE digests for all effective heads. The raw timing
samples are in the JSON record; they are isolated kernel measurements with
sequential arm ordering, not a universal model-throughput claim. The balanced
FP32 arm is **research-only for serving** and is excluded from the bit-identity
claim. Hardware residual dequantization is disabled in these comparisons.

The candidate's `b12x/attention/_shared/mla/` and
`b12x/attention/sparse_mla/` source trees are identical to PR #311 revision
`0edbaef99ffa6f03588e0ca46b4bd65a143ca3fb`. Importantly, this includes the S4
return-state fix; the intermediate fast-path revision `242d6ca` is not a valid
comparison or deployment candidate.

## Reproduction

Run on an idle SM12x GPU using the desired source checkout and record its
revision. The benchmark retains raw replay samples, GPU state, and digests.

```bash
B12X_MLA_SM120_GLM_FASTPATH=1 \
B12X_MLA_SM120_GLM_W_HW_DEQUANT=0 \
python benchmarks/benchmark_kimi_packed_mla.py \
  --heads 112 --policy static --partial bf16 \
  --lengths 2048,8192,16384 --seeds 42 --amplitudes 0.25,4 \
  --flush-l2 --output packed-112.json
```

Use 104 heads for the tail-launch comparison. Disable the fast path on the
reference source for the scalar-load comparison. Do not interpret timings
taken concurrently with model serving as an isolated-kernel result.

`validation/attention/check_kimi_packed_mla_high_pages.py` independently checks
physical byte addressing beyond signed 32-bit range. It compares low pages
with page 2,133 at byte offset 2,149,244,928 using four rows and 112 padded
heads. Output and LSE are bit-identical in the recorded frozen-source run.
The script needs more than 2 GiB of free device memory.

```bash
B12X_MLA_SM120_GLM_FASTPATH=1 \
B12X_MLA_SM120_GLM_W_HW_DEQUANT=0 \
python validation/attention/check_kimi_packed_mla_high_pages.py
```

## Serving boundary

The serving composition pairs vLLM `fa6ea71c01fd`, B12X `0bf9f177b237`, and
LMCache `9f8514c680e7`. At approximately 64 Ki context, rank-zero target graph
duration changes from 35.35 to 30.85 ms and packed MLA kernel sum from 7.50 to
2.83 ms per step. The query-head tail disappears: 192 to 96 reader launches
over four steps and 24 layers. Kernel sums across streams are not critical-path
durations, and these traces use different generated continuations.

The 128 Ki serving baseline is **invalid**: it contains no client output and
borrowed another request's global server counters. No 128 Ki model speedup is
claimed. The benchmark ownership correction is
[llm-inference-bench #16](https://github.com/local-inference-lab/llm-inference-bench/pull/16).
