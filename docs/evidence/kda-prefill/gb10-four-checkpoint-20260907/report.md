# GLM prefill: four configurations

bounded semantic/cache smoke checks and performance observations; not full numerical/model-quality qualification.

Four GB10 Sparks, TP4/DCP4, MTP3, fixed 8192-token budget. Runtime vLLM `abb715f132bdccb592a34b2596a3d3a8d757ffbc`, B12X `70fe41974ef4b18f61caaa2579c81cdc05d1265f`.

Both native split-page settings are 512 in every arm. Diagnostics are enabled throughout.

## Cold prefill

Three samples per size; prompt tokens / median first-token latency. Positive percentages indicate higher throughput than Neither.

| Prompt | Neither tok/s | Coalescing tok/s | Sharded mHC tok/s | Both tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 1949.6 (+0.0%) | 2947.3 (+51.2%) | 1958.5 (+0.5%) | 3063.5 (+57.1%) |
| 16384 | 2358.9 (+0.0%) | 2928.3 (+24.1%) | 2390.8 (+1.4%) | 3060.8 (+29.8%) |
| 32768 | 2614.5 (+0.0%) | 2923.5 (+11.8%) | 2681.8 (+2.6%) | 3049.6 (+16.6%) |

## Decode

One 20-second observation per cell. Each entry is aggregate tok/s / acceptance-normalized steps/s.

| Context / concurrency | Neither | Coalescing | Sharded mHC | Both |
| --- | ---: | ---: | ---: | ---: |
| 8192 / C1 | 49.17 / 17.86 | 47.37 / 17.91 | 48.46 / 17.86 | 46.29 / 17.89 |
| 8192 / C4 | 122.19 / 43.70 | 120.18 / 44.54 | 121.17 / 44.99 | 119.37 / 42.99 |
| 32768 / C1 | 46.18 / 17.82 | 45.92 / 17.73 | 47.31 / 17.62 | 47.11 / 17.57 |
| 32768 / C4 | 123.44 / 44.34 | 125.20 / 45.64 | 123.90 / 45.84 | 124.25 / 45.00 |

## Execution and correctness evidence

| Configuration | Exact-answer/cache checks | Four-checkpoint dispatch | mHC common dispatches across four ranks |
| --- | ---: | --- | ---: |
| Neither | 5/5 passed | Disabled | 0 |
| Coalescing | 5/5 passed | Observed | 0 |
| Sharded mHC | 5/5 passed | Disabled | 20 |
| Both | 5/5 passed | Observed | 35 |

The JSON contains every prefill sample, sanitized decode request measurements, source/artifact hashes and deltas. No prompts, answers, private addresses or container identifiers are included.

## Limits

- Three cold samples per prefill size; one 20-second decode observation per cell.
- Sequential arm order and stochastic decode acceptance limit small-difference conclusions.
- Two recovery reboots occurred between arms: one between Neither and Coalescing, and another between Coalescing and Sharded mHC. Each receipt records matching driver, GPU clock settings, persistence and CPU0 governor before and after its reboot. Reboots can change allocator/cache state and thermal conditions; this was not an interleaved A/B experiment.
- Prefill throughput is prompt tokens divided by first-token latency, including request overhead.
- MTP-normalized steps/s is aggregate tok/s divided by measured acceptance length; server steps/s is retained separately.
- Dispatch logs plus completed requests corroborate asynchronous execution without per-kernel GPU completion events.
- Exact-answer and cache-reuse tests are smoke coverage, not full model-quality or numerical-equivalence evaluation.
- Diagnostic logging remains enabled in all arms and adds host work.
- Deployment readiness/source checks and exclusive access are recorded attestations; public output excludes private deployment identifiers.
