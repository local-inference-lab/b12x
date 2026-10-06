# MXFP4-CSF expert activation and batch-serving validation

MXFP4-CSF weight preparation supports native MXFP8 expert activations without
retaining expanded scales per model layer. vLLM selects this path for
DeepSeek-V4.1-Flash unless `VLLM_B12X_MOE_FP4_FORCE_A16=1` requests BF16.
Kimi retains BF16 expert activations. Attention and logits precision do not
change. Checkpoint bytes and canonical CSF schemas remain compatible; restoring
activation quantization can change outputs relative to BF16 expert execution.

**Implemented:** B12X prepares compressed scales in native W4A8 order and expands
selected experts into caller-owned shared scratch. Both canonical weight handles
and the prepared representation retain that scratch. Compact N64 tails and
padded N256 layouts use the same E8M0 clamp as native preparation. A bounded
presence mask avoids redundant route-prefix scans and skips absent experts.
NVFP4 expansion uses the same presence-mask mechanism.

**Qualified components:** the composed B12X runtime passes 70 GPU tests covering
MXFP4/NVFP4 decoding, activation modes, exact expert output, invalid/repeated
routes, poisoned shared scratch, frozen CUDA graphs and zero replay allocation.
The paired vLLM overlay passes 61 loading and policy tests. A separate oracle
reads original DS4.1 layer 3, TP4 rank 0: all 384 experts have identical FP4
weight bytes and prepared scale bytes. Expert output is bit-identical for
1, 17, 33 and 128 tokens, top-8 routing, fast math and SwiGLU clamp 10.

## Conditions and measurements

Serving uses Frank1 RTX PRO 6000 Blackwell GPUs 12–15, 600 W, loaded memory clock
16,365 MHz, TP4/DCP1, DSpark with maximum depth 7, BF16 attention, FP8 KV,
4096 batched tokens, 32 sequences and maximum context 1,048,576. The original
checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277`; CSF is
`local-inference-lab/DeepSeek-V4.1-Flash-MXFP4-CSF` revision
`872da235166458bd6ffa9ee3f3c5c4771b63159c`.

The image is
`sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`,
with CUTLASS DSL 4.7.1. The native control uses vLLM
`45f20fa3a1c2272783cdad2693371195dea9b080` and B12X
`8f0f01829168f51cca424ac2f57c51b7a25a7b67`. The repair composes the CSF loading
and preparation APIs from vLLM `80c4201c6cdc724fb3c361e88454eadde1269b9c`
and B12X `096464c43ac413755b7dd071ab5266a9d0663c4e` onto those controls, followed
by the activation and decoder changes. Per-file SHA256 source manifests are
retained with each serving arm. Production services were not changed.

Each decode cell averages two valid 30-second windows at context zero using
benchmark 0.7.3. Ratios are CSF/native minus one. NVML reports clock-event mask
`0x400`, so all timing observations are **research-only**, not release performance
qualification.

| Concurrency | Native tokens/s | CSF tokens/s | Native steps/s | CSF steps/s | Step-rate delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 298.26 | 301.59 | 113.86 | 110.34 | −3.09% |
| 8 | 1029.09 | 992.46 | 427.56 | 429.70 | +0.50% |
| 16 | 1378.17 | 1342.04 | 633.79 | 626.04 | −1.22% |
| 32 | 1969.41 | 1927.01 | 885.40 | 873.00 | −1.40% |

The initial native-MXFP8 versus CSF-BF16 comparison lost approximately
11–12% of step rate at concurrency 8–32. Restoring MXFP8 removes most of that
gap in the measured configuration. Step rate is a server-counter diagnostic,
not isolated GPU compute time. Aggregate output rate also depends on speculative
acceptance; identical tokens/s or full-model logits are not established.

The server reports 9,733,390 native KV tokens and 13,105,035 CSF KV tokens
(+34.64%). These startup capacities include the model's complete cache layout;
the benchmark's generic single-cache metric is not interchangeable with them.
Both repair decode passes, 8K/32K prefill, text arithmetic, tool use and vision
smoke checks complete without errors.

## Limits and evidence

Repeated temperature-zero prompts without speculation vary within both native
and CSF serving arms. Maximum selected-logprob differences on their common
prefixes are 0.482 native, 0.510 CSF and 0.578 across arms. These eight short
paired probes do not establish full-vocabulary KLD or attribute nondeterminism
to a particular kernel. Component exactness is qualified separately.

**Unsupported by these claims:** complete-model logit parity, 1M-context stress,
other hardware modes and unmeasured serving geometries. These DS4.1 measurements
use presence-based scale expansion and do not qualify NVFP4 operand
reconstruction. GLM/Qwen implementations, measurements and remaining overhead
are covered by the separately pinned [NVFP4 serving report](nvfp4-serving.md).

Raw receipts reside under `/data/trellis-quant/csf-batch-performance-20261001`:
`ds41-api-serving-means.json`, `api-mxfp8-tests.log`, `api-vllm-tests-2.log`,
`ds41-real-mxfp8-parity.json`, `ds41-nospec-probe-comparison.json`, and serving
directories `ds41-native-c8-profile-repeat-1` /
`ds41-csf-c8-profile-api-mxfp8-2`. The harness and detailed report are in
`/root/vllm/kimi/csf-batch-performance-20261001/{profile_serving.py,REPORT.md}`.
These paths identify retained operator evidence, not files shipped in the package.
