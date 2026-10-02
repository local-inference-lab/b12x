# Qwen NVFP4-CSF reconstruction and serving

Qwen3.8-Flash-Next compressed-scale execution reduces the matched TP2 C8/C16
output-throughput penalty from 4.83%/5.00% to 1.44%/2.91%.
Engine-step penalties are 1.69%/1.84%.
C1 engine overhead is 2.12% against the table's native control, or
1.92% against an independently repeated native C1 control. These results
place the measured overhead near the DS4.1/GLM measurements in the
[NVFP4](nvfp4-serving.md) and [MXFP8](mxfp8-serving.md) reports; they do not imply
zero overhead or parity for every model, context length or concurrency.

**Implemented:** ready gate/up operands share two consumer barriers, split
128-row gates reconstruct only their consumed 64-row halves, and partial K
atoms select exact zero padding inside the shared reader. Complete-plane
indexed reconstruction loads codes and bases for four adjacent words together,
shares their bitmap/prefix lookup, and stores one aligned sixteen-byte vector.
The [scale contract](nvfp4-scale-operands.md) explains selection, storage,
synchronization and address invariants. Native MMA loads, activation precision,
checkpoint schemas, prepared scale storage sizes and decoder grids are unchanged.

**Qualified correctness:** 121 byte/expert cases pass. Coverage includes empty
and dense exception streams, partial atoms and CTAs, shared-slot reuse, invalid
and repeated int64 routes, frozen graph replay, poisoned scratch and allocation
checks. A selected expert's fixed stream starts beyond 2 GiB and its output at
4 GiB. Compute Sanitizer passes 16 indexed-reader memcheck cases and 22 indexed
racecheck cases with zero errors, warnings or hazards. The four split-gate
racecheck cases from the unchanged shared-reader implementation also pass.
Four online A4/A16 preparation composition cases pass with
[B12X #458](https://github.com/local-inference-lab/b12x/pull/458).

**Research-only performance:** the RTX PRO 6000 Blackwell 96 GB GPUs report NVML
clock-event mask `0x400` at 600 W. These are not clean-clock release measurements.
Dynamic clocks, two short generation windows per point and MTP acceptance
variation limit precision. Full-model KLD, long-context performance and complete
GLM serving on this implementation remain unmeasured.

**Unsupported for cooperative operand reconstruction:** A16, non-SiLU experts,
hidden dimensions not divisible by 128, or intermediate dimensions not divisible
by 64. Their existing complete-plane execution contracts remain in place.

## Serving conditions and results

Native and CSF arms use the same physical GPUs within each TP comparison,
FP8 KV, MTP with three speculative tokens, `VLLM_PLE_CPU_OFFLOAD=1`, prefill
capacity 4096 and GPU memory utilization 0.96. TP1 uses GPU 10, memory clock
13,365 MHz, maximum model length 262,144 and sequence capacity 16. TP2 uses GPUs
12/13, memory clock 16,365 MHz, maximum model length 131,072 and sequence capacity
32. The PLE/ngram table uses 26.82 GiB of host memory at TP1 or 13.41 GiB per rank
at TP2. Each configuration is compared against its matched native control.

Each concurrency has two warmed 30-second windows, zero added context and an
8192-token output limit. Every window sustains its requested concurrency without
queued requests, capacity limits, exact-loop detections or request failures.
Four generation/logprob smoke requests pass per arm; they do not establish
full-vocabulary logit parity. Positive deltas mean CSF is faster; values are
arithmetic means of the two windows.

| TP | Concurrent requests | Native tok/s | CSF tok/s | Output delta | Native steps/s | CSF steps/s | Step delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 196.02 | 197.47 | +0.74% | 86.15 | 85.18 | -1.12% |
| 1 | 8 | 797.99 | 807.12 | +1.14% | 346.39 | 350.55 | +1.20% |
| 2 | 1 | 274.58 | 275.98 | +0.51% | 118.03 | 115.53 | -2.12% |
| 2 | 8 | 1233.38 | 1215.59 | -1.44% | 528.99 | 520.04 | -1.69% |
| 2 | 16 | 1884.44 | 1829.53 | -2.91% | 810.76 | 795.86 | -1.84% |

Engine-step rates separate the multiplicative contribution of accepted MTP
length, while still reflecting the real serving workload. TP2 C8/C16 accepted
length changes by +0.25%/-1.09%. The output and step
metrics therefore remain separate. Two windows cannot establish whether an
acceptance difference is systematic; no quality improvement or regression is
inferred from this comparison.

Against the matched CSF baseline, TP2 C8/C16 generated-token throughput improves
3.56%/2.20% and engine-step rates improve 2.12%/2.07%.
The evidence retains the operand-only implementation and both repeated C1
controls, so the indexed-reader effect can be checked independently. Its C1
engine-step gain over the repeated operand-only control is
1.29%.

| TP | Native loaded GiB/rank | CSF loaded GiB/rank | Native KV tokens | CSF KV tokens | KV change |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.86 | 71.20 | 660986 | 859746 | +30.07% |
| 2 | 38.41 | 37.12 | 3912587 | 4096315 | +4.70% |

These startup capacities are observations under the listed settings, not
long-context concurrency guarantees. Runtime optimization does not change the
prepared compressed-scale allocation.

## Decoder and expert evidence

A C1 TP2 trace identifies 48 separate indexed scale expansions per model step.
With the operand-only implementation their mean duration is 5.58 microseconds;
vector reconstruction reduces it to 4.15 microseconds. The corresponding MoE
kernel remains near 37.3 microseconds. These diagnostic traces explain the C1
improvement; profiling durations are excluded from the throughput table.

A real Qwen layer-3 probe uses 512 experts, hidden size 2560, TP2 intermediate
size 320 and top-10 routing. Native weights own an independent copy. Four-token
serving-policy execution selects complete-plane expansion. On GPU 14, its
optimized CSF latency is 40.99 microseconds versus 41.81 for the scalar indexed
reader; paired native controls are 38.85/38.83 microseconds. Atomic reductions
retain their numerical tolerance and cosine gate; separate deterministic
expert tests retain exact equality. Replayed graphs mutate inputs/routes,
poison scratch and assert stable allocation. A larger-span decoder probe
passes its byte oracles but loses performance, so the 4096-byte CTA span is
retained.

The cooperative path retains its independently qualified Qwen random-route
layer-3 timings at 8/32/64 tokens: native 93.07/337.42/504.24 microseconds,
baseline CSF 120.72/358.04/487.31, optimized CSF 110.68/347.73/484.10.
Those routes do not reproduce serving expert reuse. The matched GLM component
also retains exact outputs, with optimized latency
181.35/444.80/608.52 microseconds versus 183.30/445.56/609.66 for its baseline.
This qualifies the component, not complete GLM throughput.

Nsight Compute on the Qwen 32-token cooperative component reports 167 registers
per thread for CSF versus 146 for native, the same 62.46 KB dynamic shared
memory, approximately 6.24% achieved occupancy and no shared-memory spilling.
The indexed-reader change does not alter that cooperative kernel's source.

## Reproduction and identities

Run correctness gates in the specified CUDA environment:

```bash
python -m pytest tests/moe/test_nvfp4_csf.py \
  tests/quantization/test_nvfp4_csf.py \
  tests/quantization/test_nvfp4_csf_inline.py -q -p no:cacheprovider
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/quantization/test_nvfp4_csf_inline.py \
  -k indexed_vectors -q -p no:cacheprovider
compute-sanitizer --tool racecheck --error-exitcode 99 --target-processes all \
  python -m pytest tests/quantization/test_nvfp4_csf_inline.py \
  -k indexed -q -p no:cacheprovider
```

The serving benchmark command runs twice with separate output files:

```bash
python /opt/lil/bench/llm_decode_bench.py --port 8041 \
  --model Qwen3.8-Flash-Next --concurrency 1,8,16 --contexts 0 \
  --duration 30 --skip-prefill --max-tokens 8192 --display-mode plain \
  --no-hw-monitor --output /results/decode.json
```

TP1 uses port 8042 and `--concurrency 1,8`. The complete Docker launches and
source manifests are hash-bound in [the evidence JSON](qwen-serving-evidence.json).
Runtime implementation: `e646d0d16d09d4c2f2c3d751d68a447c71086b75`;
operand-only control: `0cb79c43d2678de8aa6e71ff9d706caea9a11b1e`;
matched baseline: `bb5f4c4a43b420a1451f38c8aca0b0d9e6385662`.
Image: `sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`,
CUDA 13.4 / CUTLASS DSL 4.7.1. Serving uses a compatible beta vLLM overlay;
source manifests identify all Python bytes rather than equating that overlay
with a bare repository commit. All vLLM Python files and unchanged B12X modules
match across controls; the only runtime differences are the three declared
CSF modules. Online composition uses #458 commit
`c41b25529b2c4153f58aeab81d91bb662f1a3e39` plus these runtime changes.

| Checkpoint role | Hugging Face repository | Revision |
|---|---|---|
| Qwen native | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `b797d2e1160b9596b2570e56c1d3590faa09d4ed` |
| Qwen CSF | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF` | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` |
| GLM component | `local-inference-lab/GLM-5.3-Flash-NVFP4-CSF` | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` |

Raw receipts reside at `/data/trellis-quant/csf-qwen-performance-20261002`.
Each serving arm retains launch arguments, the source manifest, logs, telemetry,
two decode results, smoke responses and a bounded C1 or C8 trace. Public JSON
retains individual windows, means, ratios, GPU identities, clocks and artifact
hashes. Changed-code lint adds no diagnostics; the existing `_impl.py` F841/B905
findings remain recorded against the unchanged baseline. Benchmark containers
are removed after measurement; deployed services are not replaced.
