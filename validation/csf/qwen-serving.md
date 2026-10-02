# Qwen NVFP4-CSF operand reconstruction and serving

Qwen3.8-Flash-Next batch execution with compressed scales has a measured TP2
engine-step penalty of 1.53–1.86% after reducing scale reconstruction work.
The matched baseline penalty is 3.74–3.83%. Actual generated-token throughput
remains 2.71–3.29% below native at C8/C16 because MTP acceptance also differs.
TP1 C8 is within the observed variation of native throughput. TP2 C1 retains
3.34% engine-step overhead; that remaining cost is not resolved by this change.

**Implemented:** ready gate/up operands share two consumer barriers, split
128-row gates reconstruct only their consumed 64-row halves, and partial K
atoms select exact zero padding inside the shared scale read. The
[operand contract](nvfp4-scale-operands.md) describes synchronization and
padding. Native MMA loads, activation precision, checkpoint schemas, and
prepared scale storage sizes are unchanged. Test fixtures own separate native
and compressed weight tensors so preparation cannot contaminate the oracle.

**Qualified correctness:** 105 byte/expert cases pass, including empty and
dense exception streams, partial atoms, slot reuse, runtime expert/route counts,
poisoned scratch, frozen graph replay, and allocation checks. Explicit split-gate
cases cover M16/32/64/128 consumer configurations. Compute Sanitizer racecheck
passes all four cases with zero errors, warnings, or hazards. Four online
A4/A16 preparation cases pass when composing these runtime changes with
[B12X #458](https://github.com/local-inference-lab/b12x/pull/458).

**Research-only performance:** all measurements below use RTX PRO 6000 Blackwell
96 GB GPUs at 600 W with NVML clock-event mask `0x400`. They are not clean-clock
release qualification. Dynamic clocks, two short generation windows per point,
and MTP acceptance variation limit precision. Full-model KLD, long-context
performance, and complete GLM serving on this implementation were not measured.

**Unsupported for cooperative operand reconstruction:** A16, non-SiLU experts,
hidden sizes not divisible by 128, or intermediate sizes not divisible by 64.
Their existing execution paths and qualification contracts remain in place.

## Serving conditions

Native and CSF arms use the same physical GPUs within each TP comparison,
FP8 KV, MTP with three speculative tokens, `VLLM_PLE_CPU_OFFLOAD=1`, prefill
capacity 4096, and GPU memory utilization 0.96. The PLE/ngram table uses
26.82 GiB of host memory at TP1 or 13.41 GiB per rank at TP2.
TP1 uses GPU 10, memory 13,365 MHz, maximum
model length 262,144 and sequence capacity 16. TP2 uses GPUs 12/13, memory
16,365 MHz, maximum model length 131,072 and sequence capacity 32.
These different TP configurations are compared only against their matched
native controls, not against each other.

For each concurrency, the decode harness warms the server and measures two
30-second windows with zero added context and an 8192-token output limit.
All reported windows sustain the requested concurrency, with no queued requests,
capacity limits, loop detections, or request failures. Four generation/logprob
smoke requests pass per arm; they do not establish full-vocabulary logit parity.

The benchmark command inside the serving container is:

```bash
python /opt/lil/bench/llm_decode_bench.py --port 8041 \
  --model Qwen3.8-Flash-Next --concurrency 1,8,16 --contexts 0 \
  --duration 30 --skip-prefill --max-tokens 8192 --display-mode plain \
  --no-hw-monitor --output /results/decode.json
```

TP1 uses port 8042 and `--concurrency 1,8`. Run each command twice with distinct
output files. Full launch options are retained with the local receipts;
[the evidence JSON](qwen-serving-evidence.json) binds their hashes and the
source-manifest hashes.

## Serving results

Positive deltas mean CSF is faster. Values are arithmetic means of the two
windows. Engine steps remove the multiplicative effect of accepted MTP length;
they still reflect the real serving workload rather than an isolated decoder.

| TP | Concurrent requests | Native tok/s | CSF tok/s | Output delta | Native steps/s | CSF steps/s | Step delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 196.02 | 201.29 | +2.69% | 86.15 | 84.74 | -1.64% |
| 1 | 8 | 797.99 | 802.96 | +0.62% | 346.39 | 351.62 | +1.51% |
| 2 | 1 | 274.58 | 267.62 | -2.53% | 118.03 | 114.09 | -3.34% |
| 2 | 8 | 1233.38 | 1199.91 | -2.71% | 528.99 | 519.17 | -1.86% |
| 2 | 16 | 1884.44 | 1822.36 | -3.29% | 810.76 | 798.37 | -1.53% |

The baseline CSF implementation on the same TP2 GPUs measures 1173.82 and
1790.20 tok/s at C8/C16, or −4.83%/−5.00% against native. The operand changes
increase output throughput by 2.22%/1.80% and engine-step rate by 1.95%/2.39%
relative to that baseline. C1 engine-step rate is effectively unchanged
(+0.04%); its output delta alone would misrepresent the implementation effect.

TP2 MTP accepted length in the final C8/C16 arms is 0.87%/1.79% lower than
native. TP1 C1 accepted length is 4.39% higher. This explains why generated-token
and engine-step deltas must both remain visible. Two windows cannot establish
whether an acceptance difference is systematic; no quality improvement or
regression is inferred from these throughput measurements.

| TP | Native loaded model GiB/rank | CSF loaded model GiB/rank | Native KV tokens | CSF KV tokens | KV change |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.86 | 71.20 | 660986 | 859746 | +30.07% |
| 2 | 38.41 | 37.12 | 3912587 | 4096315 | +4.70% |

These are startup observations under the listed capacities, not guaranteed
long-context concurrency. The prepared scale allocation is unchanged by the
operand optimization. The separate [NVFP4 serving report](nvfp4-serving.md)
retains the GLM TP4 and Qwen TP2 measurements for its declared source revision.

## Component and architecture evidence

A real Qwen expert layer uses 512 experts, hidden size 2560, intermediate size
320 on TP2 rank 0, and top-10 routing. Layer 3 from the checkpoint below is
compared with an independently reconstructed native owner. Exact deterministic
outputs survive input/route mutation and poisoned, allocation-free graph replay.
Six alternating-order samples measure 100 replays each on GPU 14.

| Tokens | Native microseconds | Baseline inline CSF microseconds | Optimized inline CSF microseconds |
|---:|---:|---:|---:|
| 8 | 93.07 | 120.72 | 110.68 |
| 32 | 337.42 | 358.04 | 347.73 |
| 64 | 504.24 | 487.31 | 484.10 |

Random routes in this component probe do not reproduce the full model's expert
reuse. The faster CSF result at 64 tokens is not a general serving speedup claim.
A matched GLM layer-3 check on GPU 14 retains exact outputs and changes inline
latency from 183.30/445.56/609.66 to 181.35/444.80/608.52 microseconds at
8/32/64 tokens. This qualifies the component path, not complete GLM throughput.

Nsight Compute on the Qwen 32-token component shows 167 allocated registers
per thread for optimized CSF versus 146 for native, with the same 62.46 KB
of dynamic shared memory and approximately 6.24% achieved occupancy. No shared
memory spilling is reported. These diagnostic profiles are retained locally;
profiling durations are excluded from the serving tables.

## Reproduction and identities

Run the correctness gates in a B12X CUDA environment:

```bash
python -m pytest tests/moe/test_nvfp4_csf.py \
  tests/quantization/test_nvfp4_csf.py \
  tests/quantization/test_nvfp4_csf_inline.py -q -p no:cacheprovider
compute-sanitizer --tool racecheck --error-exitcode 42 --target-processes all \
  python -m pytest tests/moe/test_nvfp4_csf.py -k split_gate -q -p no:cacheprovider
```

Runtime implementation: `0cb79c43d2678de8aa6e71ff9d706caea9a11b1e`.
Baseline: `bb5f4c4a43b420a1451f38c8aca0b0d9e6385662`.
Image: `sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`,
CUDA 13.4 / CUTLASS DSL 4.7.1. Serving uses a compatible beta vLLM overlay;
source manifests identify all Python bytes rather than equating that overlay
with a bare repository commit. All vLLM Python files and all unchanged B12X
modules match within each native/CSF pair. The three changed runtime modules
match the implementation commit. Online composition uses #458 commit
`c41b25529b2c4153f58aeab81d91bb662f1a3e39` plus these runtime changes.

| Checkpoint role | Hugging Face repository | Revision |
|---|---|---|
| Qwen native control | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `b797d2e1160b9596b2570e56c1d3590faa09d4ed` |
| Qwen compressed scales | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF` | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` |
| GLM component probe | `local-inference-lab/GLM-5.3-Flash-NVFP4-CSF` | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` |

Raw receipts reside at `/data/trellis-quant/csf-qwen-performance-20261002`.
Each serving arm retains its complete launch, Python source manifest, logs,
telemetry, two decode results, smoke responses, and a bounded C8 trace. Public
JSON retains raw timing samples, means, ratios, GPU UUIDs, clock ranges and
artifact hashes. Changed-code lint adds no diagnostics; the existing `_impl.py`
F841/B905 findings are recorded against the unchanged baseline. Deployed services
were not replaced by the benchmark containers.
