# NVFP4-CSF expert decoding and batch serving

Cooperative scale reconstruction removes most of the measured GLM/Qwen
batch-serving regression while preserving the compressed checkpoint's exact
scale bytes. Dynamic A4 SiLU kernels reconstruct native scale operands in shared
memory; preparation chooses between that path and complete-plane expansion.
Short-route expansion uses a prepared exception-word index. The
[operand layout and synchronization contract](nvfp4-scale-operands.md) describes
the implementation. vLLM retains checkpoint reading and TP slicing.

**Implemented:** cooperative operands, N64 tails, indexed short-route expansion,
typed preparation selection and native route/addressing corrections. Canonical
checkpoint schemas, FP4 weight codes, FP32 calibration and A4 arithmetic are
unchanged. No checkpoint conversion or launch flag change is required. The
paired loading API requires vLLM #956 and B12X #450. Runtime source remains in
those PRs; the measurement does not switch production services.

**Qualified:** component correctness and the bounded serving requests described
below. **Research-only:** every timing figure, because NVML reports clock-event
mask `0x400`, graphics clocks are dynamic and each decode cell has two short
windows. **Unsupported by this evidence:** full-model bit parity or KLD,
maximum-context stress, other hardware and unmeasured geometries. Small-batch
overhead remains; the result is not a zero-cost compression claim.

## Conditions and artifact identities

Frank1 uses RTX PRO 6000 Blackwell Workstation Edition 96 GB GPUs, 600 W,
driver 615.71.09 and CUTLASS DSL 4.7.1. Both arms of each model use the same
physical GPUs. GLM uses GPUs 12–15 at loaded memory clock 16,365 MHz; Qwen uses
GPUs 10/11 at 13,365 MHz. Both tests run concurrently on disjoint GPU sets;
other services on the host remain running. The per-GPU UUIDs, observed clocks
and clock-event masks are in the [measurement receipt](nvfp4-serving-evidence.json).

| Model | Native HF revision | CSF HF revision |
| --- | --- | --- |
| local-inference-lab/GLM-5.3-Flash-NVFP4 | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` |
| local-inference-lab/Qwen3.8-Flash-Next-NVFP4 | `b797d2e1160b9596b2570e56c1d3590faa09d4ed` | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` |

CSF repository names append `-CSF` to the native names. The native revisions
match the compressed manifests' source; the GLM QAD revision is not used.
Checkpoints are read-only.

| Serving setting | GLM | Qwen |
| --- | --- | --- |
| TP / DCP | 4 / 1 | 2 / 1 |
| Target experts / MTP | Native A4 / three draft tokens | Native A4 / three draft tokens |
| Draft MoE backend | Marlin | B12X |
| KV cache dtype | FP8 | FP8 |
| Maximum context | 1,048,576 | 262,144 |
| Maximum batched tokens / sequences | 4096 / 32 | 6019 / 16 |
| GPU memory utilization | 0.93 | 0.96 |
| CUDA graph mode | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |

All arms use image
`sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`.
B12X implementation commit is
`2fa9fefc929bb6c93a34d65084eb6aa891465c1e`; the receipt records its changed runtime
and test file SHA256 values. The vLLM overlay composes
the reader/activation changes onto control
`45f20fa3a1c2272783cdad2693371195dea9b080`. Each arm retains the complete
per-file source manifest and Docker command; the receipt authenticates both.

GLM native/CSF and Qwen CSF have identical runtime source manifests. Qwen native
predates one preparation-only correction: declaring the indexed scale decoder
and the deterministic reduction. Its native atomic launch math and policy are
identical. No decoder or expert-compute source differs between these arms.

## Decode measurement

Benchmark 0.7.3 measures two valid 30-second sustained windows at initial context
zero for each concurrency. Each table cell is their arithmetic mean. Output
tokens/s is the user-visible throughput. Steps/s is a diagnostic derived from
server output/speculation counters; it is not isolated GPU kernel time.
Acceptance can change output rate even when the step rate is similar.
All ratios are CSF/native minus one.

### GLM TP4 / MTP3

| Concurrency | Native tokens/s | CSF tokens/s | Output delta | Native steps/s | CSF steps/s | Step delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 295.56 | 289.61 | -2.01% | 118.32 | 116.04 | -1.93% |
| 8 | 990.76 | 995.93 | +0.52% | 401.03 | 405.52 | +1.12% |
| 16 | 1419.15 | 1458.23 | +2.75% | 575.68 | 587.83 | +2.11% |
| 32 | 2192.64 | 2234.44 | +1.91% | 886.54 | 904.18 | +1.99% |

### Qwen TP2 / MTP3

| Concurrency | Native tokens/s | CSF tokens/s | Output delta | Native steps/s | CSF steps/s | Step delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 241.75 | 234.38 | -3.05% | 105.61 | 102.81 | -2.64% |
| 8 | 1082.94 | 1028.32 | -5.04% | 468.36 | 453.56 | -3.16% |
| 16 | 1639.68 | 1586.82 | -3.22% | 710.71 | 692.14 | -2.61% |

GLM batch decode is approximately at native speed in this comparison. Qwen
retains a measurable C8/C16 deficit: 5.04/3.22% in output rate and 3.16/2.61%
in step rate. The repair reduces that regression; it does not eliminate it.
C1 output overhead is 2.01% for GLM and 3.05% for Qwen in these paired runs.


The source-pinned regression control used vLLM `45f20fa3a1` and B12X `8f0f0182`.
It measured GLM output losses of 9.0/13.2/11.0% at C8/16/32 and Qwen losses of
11.3/14.7% at C8/16. Those controls reproduce the batch regression; they are
distinct from the matched implementation comparison above.

Independent GLM controls illustrate timing sensitivity. With cooperative
operands before indexed short-route expansion, C8/16/32 step-rate differences
against a native repeat were −1.28/+0.35/+0.73%, while C1 was −7.81%.
The API-based comparison before indexed expansion measured
−2.62/+1.37/+2.39/+1.64% at C1/8/16/32. These are separately retained runs, not
pooled with the table. Small positive batch differences do not establish a
general speedup, and C1 still has material overhead in some runs.

## Prefill and memory

| Model | Prompt size | Native prefill tokens/s | CSF prefill tokens/s | Delta |
| --- | ---: | ---: | ---: | ---: |
| GLM TP4 | 8K | 14,901 | 14,679 | -1.49% |
| GLM TP4 | 32K | 16,240 | 15,949 | -1.79% |
| Qwen TP2 | 8K | 17,558 | 17,523 | -0.20% |
| Qwen TP2 | 32K | 17,279 | 17,138 | -0.82% |

| Model | Native model GiB/rank | CSF model GiB/rank | Native KV tokens | CSF KV tokens | KV change |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM TP4 | 46.93 | 45.26 | 5,614,647 | 6,022,639 | +7.27% |
| Qwen TP2 | 39.01 | 37.71 | 4,819,379 | 5,064,058 | +5.08% |

Prefill is the client's prompt-token count divided by time to first token,
using repeated cold prompts near 8K and 32K tokens. Actual prompt lengths and
sample counts are in the receipt. It includes request/first-token overhead and
does not isolate the prefill kernel. KV capacities come from server startup,
not the benchmark's generic single-cache estimate. The prepared exception
index consumes part of the on-disk compression saving.

## Correctness and replay

- **Qualified component corpus:** `nv-publication-tests-2.log` records 408
  passing decoding, expert, preparation-policy and native-oracle tests. The two
  strict declaration cases in `nv-indexed-declarations-3.log` additionally pass
  for atomic and deterministic output. Registry checks pass nine cases.
- Exact byte oracles cover zero/sparse/dense exceptions, reused shared slots,
  partial row/K atoms and native F8_128x4 order. Indexed expansion covers both
  routing integer ABIs, route counts 1/5/31/63/64/128, invalid and repeated IDs,
  route mutation, inactive output preservation and barrier initialization.
- Expert tests compare native/CSF A4 and A16 execution with poisoned scratch,
  frozen CUDA graphs and zero replay allocations. Intermediate dimensions
  64/128/192/320 include compact tails. The original Qwen layer-3 TP2 rank-0
  oracle is exact at 4/8/32/64 tokens with 512 experts, hidden dimension 2560,
  intermediate dimension 320 and top-10 routing (`nv-qwen-tail-real-1.json`).
- The [native correctness report](dynamic-moe-correctness.md) records the
  independently reproduced shared route-broadcast race, deterministic MXFP8
  gather address defect and preparation declaration invariants. Existing
  numerical oracle thresholds are unchanged.
- Both decode passes, prefill, bounded generation probes and trace requests
  complete for all four serving arms. These are operational checks, not a
  full-vocabulary quality comparison.

The component indexed-decoder experiment on real GLM TP4 layer-3 scales reduced
eight-route expansion from 4.70 to 2.54 microseconds and 16-route expansion from
4.84 to 3.83 microseconds. At 32 routes the difference was small. Those timings
use 64 launches per CUDA graph to avoid CPU launch-rate limits; they do not
predict a proportional full-model speedup (`nv-indexed-expansion-4.json`).

## Evidence and reproduction

Raw logs, traces, source manifests and launch commands reside under
`/data/trellis-quant/csf-batch-performance-20261001`. The receipt lists every
measured arm and SHA256 authenticates its decode, prefill, source, command,
server-log and telemetry files. The harness is
`/root/vllm/kimi/csf-batch-performance-20261001/profile_serving.py`;
`summarize_nv_serving_evidence.py` checks the measurements and source identities.
Those paths are retained operator evidence, not a package dependency.

Run the focused GPU suites in a matching CUDA/CUTLASS environment:

```bash
python -m pytest -q \
  tests/quantization/test_nvfp4_csf_inline.py \
  tests/moe/test_nvfp4_csf.py \
  tests/moe/test_nvfp4_csf_dispatch.py \
  tests/preparation/test_tuning_predicates.py \
  tests/preparation/test_precision_choices.py \
  tests/quantization/test_nvfp4_csf.py \
  tests/quantization/test_mxfp4_csf.py \
  tests/moe/test_mxfp4_csf.py \
  tests/quantization/test_csf_routing.py \
  tests/moe/test_w4a8_migration_corpus.py::test_w4a8_materialized_routing_phase1_phase2_matches_oracle_under_graph
```

DS4.1's separate [MXFP8 expert report](mxfp8-serving.md) covers its activation
policy correction. Its BF16 attention and Kimi's BF16 experts remain unchanged.
