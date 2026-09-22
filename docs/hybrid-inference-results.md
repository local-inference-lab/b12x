# Hybrid inference qualification

Status: **serving and numerical gates passed; real-checkpoint sanitizer gates
remain incomplete**. This report records the format-neutral
storage contract, tensor-parallel residency, and phase measurements for the
pinned Qwen3-Next-80B checkpoint. Completed measurements remain attached to the
sources that produced them. Incomplete gates are not acceptance claims.

The repeated C4 measurements establish three distinct results. At TP1, adaptive
residency improves generated-token throughput from 59.28 to 70.03 tok/s, with
longer delivery tails. At TP2, the larger learned placement reaches 208.41 tok/s
and the unchanged adaptive controller reduces throughput to 199.29 tok/s.
Ordinary selective UVA offload reaches 41.35 and 157.69 tok/s, respectively,
using a different W4A4 recipe. It is faster on the measured prompt-processing
series. Neither adaptation nor one numerical backend wins every serving phase.

The format-neutral declarations and arbitrary-N transaction are implemented;
NVFP4 is the only physically exercised storage adapter. Physical model evidence
covers TP1/TP2. Software tests at N=3/4 do not confer additional model or hardware
qualification. All-resident TP2 is rejected under the fixed useful reservations.

## Sources and artifacts

The working branches began clean at b12x
`ce103e4c8c7a7db4d7c2ebd18459109452ebcca3` and companion vLLM
`771c44da5338940687a4f507ad0fca6532b55bb9`. Inspection included their live remote
heads and default branches. The b12x default head was
`c2dc1cf02295b8241fc6a7728be7b6e8c23dda2f`, with merge base
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`; vLLM main was
`47ccf6c57d92f03630ebcbad3809450545825488`, with merge base
`e12b91b032daed2afc34d77cca20902cef957b3c`. Neither default branch was merged
or rewritten.

A final live-ref check found b12x master advanced to
`b631ac19d2abcbec85947f423c844087fdb1fb84` during qualification. The added
commits include IQ2_XS support, NVFP4 decode tuning, preparation compatibility,
GDN binding checks and attention changes. The merge base remains `0f3a8cbf`.
Those changes are outside the frozen experiment and were not merged or used to
relabel its measurements. Companion main remains at the head recorded above.

The serving measurement export is b12x
`1fed1fa77ef05e2c205097fd75bbdb9812bbf3b7`. Companion source is
`7e3471fc0feb58a264433fc78ddf0a30ad3228a1`, built as a complete wheel with
SHA-256 `0222927943a77df15db945c62c856c258414d5411bfc7e3444711ff91f79f0a2`.
Balanced-profile measurements use b12x
`41e4fc273286fc7b8fbbf9120fbd069bf764289b`, which corrects JSON tuple/list
validation without changing the stored profile or its ranking. Other primary
measurements retain the frozen `1fed1fa7` export. Every engine verifies loaded
Python modules and native libraries against that wheel. A version suffix is not used as build identity. Earlier bring-up runs
retain their distinct source exports and wheels.

An independent receipt check matches all recorded loaded Python and native
library hashes from the 27 workers in the 18 principal engines to the wheel
manifest. The initial pre-import verification alone contains no loaded native
libraries; the worker receipts supply that evidence. The consolidated result is
`loaded-artifacts-check.json`.

Final core tests, layer oracles and cancellation runs use
`572d3f5bfdd0dae43389f3b69d91514c809f9bef`. Later changes add explicit storage
transform/rollback validation, per-rank lifecycle
accounting, explicit layer-oracle error measurements, the compact sanitizer
fixture's shard identity, and all-rank post-cancellation validation. They do not
change the measured arithmetic or replacement policy. The first TP1 pair retains
its frozen launch/export identity but predates the additional per-launch source
file manifest. Continuous telemetry also has gaps for that pair and ordinary
TP2 trials 2/3; these gaps are reported rather than reconstructed.

Profiler retirement controls and the four-cycle reconstruction use
`a994091ae10adfb7d8b68882c71e65b128c7d694`. This harness-only change disables
vLLM's in-worker summary-table aggregation while preserving complete trace
export. The companion wheel, kernels and non-profiled serving configuration
remain unchanged. Earlier profiler failures retain their original source.

Raw evidence is retained on ripper under
`/home/jasonc/b12x-hybrid-20260922`. Source archives, exact runners, artifact
manifests, checkpoint/profile identities, request token IDs, resource receipts,
failed attempts, sanitizer logs and profiler traces belong to that bundle.
These paths identify evidence; portable runner arguments accept explicit model,
profile, artifact and output paths.

The model is `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`, revision
`8fb2682f136cf94d932a498f18cb1e428832a912`, content fingerprint
`f22fdcef6ae16e9a85415e35ec55069ba7ef7eab8220f48343747fc2adb4ec2e`.
The initial TP1 profile is
`16b14690fa88d1532a39042420afedf81c082e58cc0fc709602235694fb232b2`;
the TP2 maximum-capacity profile is
`363243aabe9f53380f8bafee3e29069c47669c689c48306c4cc78acb284ba926`.
Both bind the `nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum` recipe and
`next80-mixed-calibration-v1` workload, with their own admitted geometry.
The TP2 artifact is constructed from the verified calibration counts in
`bea288cb375cb73cd8cd4f170f675c4c98bcf9ec60819bfc106065c11ee24f7d`;
that lower-capacity calibration artifact is not the serving placement.

The complete snapshot and validated identity receipt are reused. No model was
downloaded during this pass. Qwen3.8, PLE integration, MTP and speculation remain
deferred.

## Hardware and topology

Both devices are RTX PRO 4000 Blackwell SM120 GPUs, with 25,151,012,864 bytes of
usable CUDA memory each. UUIDs are:

- rank 0: `GPU-47363510-b87a-13a5-4824-2542e97df76c`;
- rank 1: `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`.

The PCIe addresses are `41:00.0` and `61:00.0`. Both negotiate **Gen4 x16 under
load**, share NUMA node 0, and support peer access in both directions. There is
no NVLink claim. Measurements use driver 580.173.02, Torch 2.13.0, CUDA 13.3,
NVCC 13.3.73 and NCCL 2.31.2. Exact loaded package and native-library identities
are retained, including the CUTLASS DSL/binding environment.

The topology diagnostic checks copied values, uses independent buffers for
simultaneous directions, and retains raw samples. The first incorrect duplex
experiment and the corrected concurrent measurement are both retained.
A 64-MiB directional P2P transfer reaches about 27.7 GB/s; true simultaneous
bidirectional transfer reaches 49.54 GB/s aggregate. These are physical Gen4
measurements, not estimates of Gen5 or Grace performance. NCCL all-reduce,
all-gather and reduce-scatter are measured separately from model execution.
At 4 KiB, all-reduce/all-gather/reduce-scatter take approximately 35/33/34 us;
at 4 MiB, 235/252/139 us; at 64 MiB, 3,547/3,796/2,145 us. Host-to-device and
device-to-host copies approach 28 GB/s at 64 MiB. Raw samples retain direction,
size and rank. Small-message microbenchmark latency includes its own launch
arrangement and must not be substituted for captured model collective time.

Available serving-interval telemetry records P1 and 13,365-MHz memory clocks on
both GPUs. GPU0 SM clocks span 2,100–2,475 MHz and GPU1 2,047–2,452 MHz;
sampled power spans 32.5–105.36 W and 35.43–113.82 W, respectively. These ranges
include active low-utilization/control intervals. Clocks are observed rather
than claimed fixed. Per-engine samples and the telemetry gaps noted above
remain explicit in `telemetry-summary.json`.

No explicitly configured Station/B300 endpoint was available. No network scan
or capability spoofing was performed. SM103 still requires its separate native
qualification sequence.

## Format and ownership contract

The [architecture guide](hybrid-inference.md) specifies the implemented
`ExpertStorageContract` and `ExpertStorageSource`. Logical expert identity,
rank-local geometry, source/backing/resident byte geometry, execution recipe,
transforms, cold-execution capability and rollback requirements are distinct.
The generic residency controller consumes IDs, membership, observations,
generations and declared movement costs. It does not inspect quantization
formats.

| Capability | Status |
| --- | --- |
| NVFP4 source, prepared canonical backing, BF16 whole-K W4A16 | Physical qualification path |
| Direct executable backing | Represented explicitly |
| Prepare-on-promotion backing | Host contract; executable adapter absent |
| Resident-only storage | Host contract; executable residency adapter absent |
| FP8, MXFP4, EXL3/BTX-style distinct representations | Host capability fixtures only |

NVFP4-specific validation and preparation remain in its adapter: native packed
bytes, K16 scales, equal gate/up global scales, finite positive scales, SiLU,
canonical scale swizzling and fixed-address whole-K execution. No arithmetic,
replacement policy or production default is changed. A physical FP8 adapter
still needs real checkpoint validation, executable cold/resident representations,
bounded preparation/promotion and rollback, and arithmetic/graph/lifecycle
qualification. Declaring its storage geometry does not provide those kernels.

## Tensor-parallel contract

The engine owns one logical policy authority, using rank 0 observations. Every
rank retains counters for agreement checks, one local shard of each expert and
its local executable backing. No expert payload is broadcast. Logical IDs,
slot membership and generations agree; addresses and shard payloads are local.
The model-wide epoch has one transaction token and an agreed vector of per-layer
generations. A layer advances only when it moves experts; every participating
rank must acknowledge the same resulting vector.

| Tensor class | Maintained model TP behavior |
| --- | --- |
| Routed gate/up weights and block scales | Intermediate output rows shard |
| Routed down weights and block scales | Intermediate input columns shard |
| Routed global/input scales | Small metadata replicates |
| Router and shared-expert sigmoid gate | Replicated projections |
| Shared MLP | Column/row parallel; external MoE runner owns composition and final reduction |
| GDN projections, convolution and recurrent state | Established head/channel TP partitioning |
| Attention Q/K/V | Head partitioning; KV replication when required by head geometry |
| Attention/GDN output projections | Row-parallel output reduction |
| Embedding and output head | Vocabulary partitioning |
| Normalization | Replicated state |

Checkpoint/model divisibility and backend alignment constrain valid shard
geometries. For example, a world-size-agnostic coordinator does not make a
512-wide intermediate dimension divisible by three. Software transaction tests
exercise N=1/2/3/4, including a delayed arbitrary participant, reversed responses,
profile/generation/routing disagreement, rollback failure, cancellation, missing
acknowledgements and uncertain publication. Physical execution covers N=1/2.

The transaction drains readers, validates every participant, stages reversible
local fills, waits for all staged acknowledgements, publishes, checks every map
and generation, acknowledges every publication, then retires staging. Missing
or uncertain acknowledgements leave scheduling paused and require reload.
The pair cap counts logical promotions. The 128-MiB byte limit remains **per
rank**; receipts distinguish local, maximum-rank and aggregate physical bytes.

The existing default custom all-reduce path failed CUDA graph capture during
standalone bring-up. The qualified configuration explicitly selects the
maintained PYNCCL path with custom all-reduce disabled. The failure is retained;
no custom collective implementation was added.

## Ordinary W4A4 numerical classification

Classification: **bounded reduction variation on the frozen layer-0 fixture**.
This does not establish a universal error bound for arbitrary model inputs.

Real layer-0 checkpoint bytes, input rows, top-10 IDs, route weights, processed
weights/scales and execution shape are frozen. Each run includes 20 eager,
20 captured-graph and 10 allocator-perturbed replays. Untuned FlashInfer CUTLASS
runs are bitwise stable across fresh processes and selective UVA placement.
After autotuning, 50/50 replays produce distinct hashes, with a maximum difference
of 0.001953125. Replaying the recorded tactics reproduces the variation while
all postprocessed weight and scale hashes remain unchanged.

FlashInfer's installed source documents that its default fused GEMM2 finalization
uses non-associative atomic top-k reduction. The companion calls that default.
The independent dequantization/GEMM/reduction diagnostic retains the native
activation quantizer. Across the tactic replay, relative L2 error stays below
0.421%, within the declared 1% diagnostic bound; maximum absolute error stays
below 0.004. This explains why matched ordinary fresh engines can choose different
greedy tokens near a logit tie without establishing a corrupted checkpoint.

The executable diagnostic is `benchmarks/moe/nvfp4_repeatability.py`.
FLASHINFER_TRTLLM and FLASHINFER_CUTEDSL reject this SM120 configuration; their
rejections are not execution results. VLLM_CUTLASS is also examined as an existing
alternative: its selective-UVA fixture has one output hash across 50 replays,
maximum absolute reference error 0.0009765625 and relative L2 error 0.3183%.
FlashInfer CUTLASS remains the physically measured Tier-1 serving backend; its
final frozen-fixture relative L2 error is 0.4189%. Choosing that established
backend preserves the existing loading/offload path. Basic-offload throughput remains a **W4A4 deployment comparison**
with the b12x W4A16 tiers, never an isolated cache-overhead comparison.

## Memory admission and lifecycle

The TP2 all-resident configuration is rejected before canonical materialization
under the fixed context-2048, 2-GiB KV, 512-MiB graph and 1-GiB safety reservations.
The admitted maximum profile contains 397–398 of 512 experts per layer:
**77.551% of logical expert payload resident**. The expert envelope is
18,075,163,404 bytes per rank, including workspace and metadata; it is not a
payload fraction. TP1 retains its qualified 34.680% payload placement and
16,325,990,988-byte envelope.

Each TP2 worker retains 21,743,861,760 source bytes and 21,743,468,544 mapped
canonical bytes. The aggregate is approximately the TP1 routed representation,
plus legitimately replicated scalar metadata. Loading does not duplicate the
full 40.5-GiB source on every rank. Dense/shared/attention/router loading allocates
about 1.91 GiB of Torch device memory per worker before cache preparation.

Initial TP2 bring-up exposed two companion lifecycle defects: peer error replies
could remain in RPC queues after one rank failed, and executor shutdown could
terminate workers before large cache owners were released. The companion now
drains all peer replies before raising and explicitly waits for worker shutdown
acknowledgements before retiring the worker processes. Worker release is
idempotent only after successful completion. The failed bring-up and forced
cleanup remain recorded separately from the corrected normal path.

The full-model TP1 and TP2 cancellation receipts observe caller cancellation
during engine-owned maintenance, then revalidate every rank's checkpoint/profile,
graphs, pointers and generation agreement. Every participating worker reports zero mapped
bytes, source bytes, graph owners and pending health storage after release.
This uses the `572d3f5b` all-rank cancellation check.

The corrected admission rejection and full-model TP2 static run both report
zero mapped bytes, CPU expert-source bytes, graph owners and pending health work
after release, on both workers. Residual Torch/context pools are reported
separately. Clean process exit alone is not the ownership gate.

The ordinary TP2 offload envelope is 3.5 GiB, selecting 3,892,379,648 actual
postprocessed routed-parameter bytes per rank. An inventory-based 3.25-GiB
projection leaves less than the fixed 1-GiB device reserve and is rejected
arithmetically. The 4-GiB bring-up control is retained separately. Pointer
attributes confirm that shared experts, routers, attention and other ordinary
weights remain resident. W4A4 parameter placement is coarse and differs from
learned per-expert W4A16 residency.

Representative resource checkpoints from the frozen decode trials are:

| Configuration | Torch allocated after graphs, GiB/rank | Device free after serving, GiB/rank | Routed CPU source, GiB/rank | Canonical mapped backing, GiB/rank | Ordinary UVA parameters, GiB/rank | Worker RSS after serving, GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 basic | 21.06 | 1.38 | — | — | 26.06 | 28.96 |
| TP1 static | 21.83 | 1.014 | 40.50 | 40.50 | — | 83.92 |
| TP1 adaptive | 21.83 | 1.010 | 40.50 | 40.50 | — | 84.05 |
| TP2 basic | 21.44 | 1.206 | — | — | 3.625 | 6.96 |
| TP2 static | 21.65 | 1.018 | 20.25 | 20.25 | — | 43.87 |
| TP2 adaptive | 21.65 | 1.016–1.018 | 20.25 | 20.25 | — | 44.03 |

These are observed checkpoints, not replacements for the full admission plan.
CUDA free memory includes non-Torch allocations; Torch allocated memory alone
does not measure total device use. RSS includes ordinary allocations and runtime
state. CUDA pinning is measured through owned mappings and parameter pointer
attributes, not inferred from `VmLck`. TP2 aggregate host and HBM use is the sum
of its rank-local owners, never a pooled GPU address space. The canonical cache
retains considerably more host storage than ordinary UVA offload.

For those representative engines, peak Torch device allocations are
21.086/21.851/21.851 GiB for TP1 basic/static/adaptive, and
21.456/21.660/21.660 GiB per rank for TP2. Process RSS high-water marks are
32.27/83.92/84.05 GiB for TP1 and at most 9.28/43.87/44.03 GiB per TP2 worker.
The ordinary loader's transient RSS exceeds its steady checkpoint. These
high-water marks and explicitly owned mappings are retained separately from
sampled total GPU use; Torch's peak does not include every native allocation.

The final TP2 reconstruction test runs four complete engines sequentially in
one client process. Worker processes are reconstructed each cycle. All eight
worker releases report zero mapped backing, CPU expert sources, graph owners
and pending health storage. Each cycle submits a health read before shutdown;
logical promotion counts are 32, 32, 0 and 32. No forced process termination or
ignored destructor exception is observed.

The existing lifecycle checker passes its unchanged 64-MiB post-warmup growth
limit. Client RSS is 1,562,619,904 / 1,676,361,728 / 1,698,926,592 /
1,706,184,704 bytes: the first warmup increase remains visible, followed by
28.44 MiB across the remaining cycles. Post-release CUDA allocations stay at
514,861,056 bytes on rank 0 and 514,861,568 on rank 1, with 572,522,496 reserved
bytes on each rank. Post-warmup worker RSS growth is below 64 MiB on both ranks.
These are bounded four-cycle observations, not an indefinite-runtime leak
guarantee. Residual runtime pools are separate from released cache ownership.
The earlier two-cycle receipt remains available with its original source.

All four final-source engines also reproduce the first 64 output IDs of each
of the 16 requests from the frozen TP2 static trial. This retained prefix check
links the later diagnostic source to the measured source under the recorded
controlled C4 admission. It is not an arbitrary output-length or batch-shape
invariance claim, and it does not relabel the earlier timings.

## C4 decode comparison

Each arm uses three fresh engines, 16 held-out requests and 4,096 generated
tokens. Static/adaptive order is S/A, A/S, S/A. Context, KV, prepared capacity,
checkpoint and the initial profile are fixed within a topology. No history,
specialist protection or anchor recovery is enabled. Health uses a configured
16-delivered-token interval, threshold 0.15 and maximum interval 1,024. The
logical cap is 32 pairs; the copy cap is 128 MiB per rank.

Values are means ± sample standard deviations across engines. Tokens within
an engine are not replication units. Overall time includes complete control
work and the pending final-control tail. General/code intervals retain their
own request spans; the complete-run rate remains the primary wall-time result.

| TP | Tier | Overall generated tok/s | General tok/s | Code tok/s |
| --- | --- | ---: | ---: | ---: |
| 1 | Basic, ordinary W4A4 | 41.350 ± 0.133 | 41.297 ± 0.217 | 41.442 ± 0.131 |
| 1 | Learned static, W4A16 | 59.284 ± 0.003 | 61.457 ± 0.023 | 57.331 ± 0.026 |
| 1 | Adaptive, W4A16 | 70.035 ± 1.152 | 79.182 ± 0.681 | 62.886 ± 1.435 |
| 2 | Basic, ordinary W4A4 | 157.685 ± 0.296 | 157.721 ± 0.362 | 158.199 ± 0.261 |
| 2 | Learned static, W4A16 | 208.411 ± 0.026 | 210.225 ± 0.031 | 207.575 ± 0.031 |
| 2 | Adaptive, W4A16 | 199.291 ± 1.454 | 203.705 ± 0.262 | 198.179 ± 1.069 |

| TP | Basic → static | Static → adaptive | Basic → adaptive |
| --- | ---: | ---: | ---: |
| 1 | +43.37% | +18.14% | +69.37% |
| 2 | +32.17% | -4.38% | +26.39% |

Basic-offload comparisons cross W4A4/W4A16 numerical recipes. They do not
isolate cache overhead or establish quality equivalence. TP2 also changes
resident capacity and reduction boundaries; its speedup over TP1 is not a
fixed-capacity strong-scaling result.

All six same-TP static/adaptive pairs have exact output IDs. The TP1 digest is
`44804de497f4fe66580e172127765ca77b35cb35c28df1dad362613636222d58`;
TP2 is
`8b3ea693f5aabe7e659a89e0c0c2662a222cd0f4c58c440b8ef4253d847f467c`.
The TP1 digest also matches the preceding `ce103e4c` qualification. This is a
direct output regression check across the storage/TP integration changes.
Cross-TP or W4A4/W4A16 bitwise equality is not asserted. Every b12x run retains
its graph/cache addresses and validates generation agreement.

| TP | Tier | TTFT p50 / p95, ms | Delivery gap p50 / p95 / p99, ms |
| --- | --- | ---: | ---: |
| 1 | Basic | 605.57 / 642.99 | 95.27 / 101.49 / 103.26 |
| 1 | Static | 517.00 / 588.52 | 65.32 / 83.46 / 89.62 |
| 1 | Adaptive | 533.43 / 663.18 | 48.31 / 112.43 / 134.26 |
| 2 | Basic | 195.76 / 272.62 | 24.51 / 25.44 / 25.74 |
| 2 | Static | 271.88 / 372.85 | 17.74 / 20.28 / 21.50 |
| 2 | Adaptive | 277.42 / 379.19 | 18.05 / 20.99 / 22.25 |

Latency entries average each engine's percentile; raw per-engine distributions
remain in `three-tier-summary.json`. Adaptive TP1 improves median delivery and
throughput while worsening delivery tails. Basic TP2 has shorter TTFT than
either b12x tier despite lower sustained generated-token throughput.

TP1 paired adaptive gains are +17.00%, +20.37% and +17.03%. Each run moves
3,104 logical experts in 97 maintenance operations. Blocked scheduler intervals
total 11.67, 9.84 and 11.63 seconds; these intervals include draining useful
already-submitted work and are not added again to serving wall time.

TP2 paired gains are −4.11%, −3.86% and −5.16%. Its initial learned map has
6.32% decode cold selections in the separate generation-zero diagnostic.
The adaptive runs perform three, three and four maximum-interval maintenance
operations, selecting 96, 96 and 128 logical promotions. Blocked intervals
total 0.492, 0.476 and 0.531 seconds. The third run includes a 176-ms final
control tail. This is a negative result for the frozen controller at this
placement and workload; no threshold or cadence is retuned to remove it.

Each TP1 adaptive trial submits 138 health probes and copies 5,511,889,152
physical bytes. TP2 submits 128 probes and copies 171,214,336, 171,214,336 and
228,182,016 aggregate bytes, respectively; each worker moves half of those
bytes. All movement epochs select the full 32-pair allowance. Median backlog
is 64 pair-cap skips from 48 proposing layers, with zero byte-cap skips. The
TP2 loss demonstrates why a policy backlog is not itself evidence that more
movement would improve serving.

Completed full-observation windows contain 1,947,840 selections per TP1
adaptive trial, with 21.2296% cold selections. TP2 trials contain 1,513,920 /
1,512,000 / 1,954,560 selections, with 6.0571% / 6.0611% / 6.1366% cold
selections. These window totals exclude any final interval not covered by a
full snapshot; health summaries are not added again. The separate static-map
route replay covers 1,958,400 decode selections, at 34.266% cold for TP1 and
6.316% for TP2. Different observation boundaries prevent treating those
fractions as an exactly paired per-token cold-rate comparison.

TP1 final pending-control tails are 6.17 / 5.58 / 5.18 ms. TP2 tails are
1.32 / 6.16 / 175.69 ms. They are already included in complete serving time;
none is added again to the reported throughput denominator.

Startup and shutdown are outside serving throughput. They include engine
construction/preparation and explicit resource release, respectively:

| TP | Tier | Mean per-request decode tok/s | Startup, s | Shutdown, s |
| --- | --- | ---: | ---: | ---: |
| 1 | Basic | 10.58 | 105.69 | 11.00 |
| 1 | Static | 15.32 | 136.28 | 15.85 |
| 1 | Adaptive | 18.98 | 138.35 | 16.09 |
| 2 | Basic | 40.66 | 90.44 | 9.62 |
| 2 | Static | 55.29 | 133.15 | 13.99 |
| 2 | Adaptive | 53.47 | 135.29 | 14.18 |

## Prefill and balanced placement

The complete prompt series uses a reproducible constructed public-service
ledger, the pinned tokenizer/template and no prefix caching. Actual C1 lengths
are 242, 498, 1,010 and 1,522 tokens. C4 uses four requests at each of 498, 1,010
and 1,522 tokens. Output length is one token. Prepared capacity stays 64, so
chunk sizes and queueing are part of the measured system. Each tier/topology/
concurrency sample uses one fresh engine; these prompt results have no
independent replication estimate. All same-TP static/adaptive output IDs agree.

The diagnostic clocks distinguish pure model CUDA duration, the elapsed span
across prompt chunks, and client TTFT. Model events stop before logits/sampling;
they are not a separate measurement of complete first-token GPU latency.
Mixed iterations remain indivisible. Static timing has no routing observer.

C1 cells show **direct prompt tok/s / client TTFT in seconds**:

| TP | Tier | 242 tokens | 498 tokens | 1,010 tokens | 1,522 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Basic W4A4 | 176.55 / 1.406 | 175.20 / 2.861 | 173.44 / 5.855 | 174.56 / 8.765 |
| 1 | Static W4A16 | 172.65 / 1.436 | 148.77 / 3.366 | 131.31 / 7.723 | 124.28 / 12.291 |
| 1 | Adaptive W4A16 | 172.69 / 1.436 | 149.32 / 3.354 | 131.22 / 7.728 | 124.43 / 12.277 |
| 2 | Basic W4A4 | 459.50 / 0.563 | 493.52 / 1.023 | 500.08 / 2.042 | 502.27 / 3.063 |
| 2 | Static W4A16 | 334.89 / 0.756 | 348.66 / 1.442 | 351.61 / 2.896 | 349.56 / 4.386 |
| 2 | Adaptive W4A16 | 334.85 / 0.757 | 348.03 / 1.446 | 351.67 / 2.898 | 352.80 / 4.351 |

C4 cells show **direct aggregate prompt tok/s / first–last client TTFT in
seconds**. Each column contains four independently completed requests:

| TP | Tier | 498 tokens/request | 1,010 tokens/request | 1,522 tokens/request |
| --- | --- | ---: | ---: | ---: |
| 1 | Basic W4A4 | 175.64 / 3.28–11.42 | 173.13 / 6.26–23.44 | 174.23 / 9.18–35.10 |
| 1 | Static W4A16 | 149.57 / 3.67–13.40 | 131.45 / 8.00–30.84 | 124.36 / 12.57–49.12 |
| 1 | Adaptive W4A16 | 149.73 / 3.66–13.38 | 131.36 / 8.00–30.86 | 124.30 / 12.57–49.14 |
| 2 | Basic W4A4 | 493.03 / 1.07–4.12 | 504.29 / 2.04–8.14 | 506.77 / 3.06–12.21 |
| 2 | Static W4A16 | 351.17 / 1.46–5.73 | 355.58 / 2.86–11.44 | 357.63 / 4.31–17.13 |
| 2 | Adaptive W4A16 | 348.68 / 1.47–5.79 | 350.91 / 2.91–11.60 | 352.03 / 4.36–17.42 |

Basic offload is faster on this prompt fixture despite its lower sustained C4
decode throughput. The comparison crosses numerical recipes and placements;
it does not isolate kernel quality. Adaptive prompt runs perform zero health
probes and zero promotions because their short outputs never reach the decode
control interval. They measure the unchanged initial placement with adaptive
state enabled, not prefill adaptation.

C1 has 4/8/16/24 pure-prefill iterations: 64-token chunks followed by a 50-token
remainder. C4 has 32/64/96 iterations. At request boundaries a 64-token scheduler
budget can split between prompts; the complete nonzero remainder sizes are
14, 50, 28, 36, 22, 42 and 8. For example, TP1 static's 1,522-token C1 prompt
uses 12.247 seconds of model CUDA time and a 12.281-second prompt-processing
span, versus 12.291-second client TTFT. TP2's corresponding values are
4.354, 4.379 and 4.386 seconds. Client admission timestamps and model host-submit
boundaries are retained; a separate scheduler-queue service time and isolated
first-token sampling duration are not inferred from these events.
Raw per-rank event durations, chunks and every
request's TTFT remain in `prefill-comparison.json` and the serving receipts.
The 64-token capacity makes C4 largely sequential; this pass does not claim an
optimized prefill batching configuration.

Untimed controls preserve exact output IDs and complete normal release. TP1's
timed-versus-untimed TTFT differences range from −0.09% to +0.25%. The first
TP2 comparison differs by +0.84% to +2.17%; a second pair on the same diagnostic
source differs by +0.14% to +0.78%. These few fresh-engine controls retain
measurement variation rather than establishing a universal overhead bound.
Direct prompt rates retain the event instrumentation; headline decode trials
do not enable it.

Observation-only controls match their same-TP static output IDs, keep generation
zero, and agree across TP ranks. They count 1,570,560 prompt selections for C1
and 5,817,600 for C4. C1 prefill cold fractions are:

| Placement | 242 tokens | 498 tokens | 1,010 tokens | 1,522 tokens |
| --- | ---: | ---: | ---: | ---: |
| TP1 original | 34.59% | 44.27% | 54.47% | 60.16% |
| TP1 balanced | 30.82% | 41.78% | 52.36% | 58.08% |
| TP2 original | 7.69% | 11.71% | 16.15% | 18.86% |

C4 cold fractions are 44.22/54.48/60.17% for TP1 original,
41.77/52.40/58.11% for TP1 balanced, and 11.65/16.14/18.84% for TP2.
The long ledger prompts exercise a different distribution from the short
held-out general/code prompts. Their higher cold pressure must not be attributed
to sequence length alone. The basic tier has coarse parameter offload rather
than a logical expert map; no b12x counter trace is substituted for its W4A4
routes or host traffic.

For the original held-out general/code fixture, TP1 prefill/decode cold fractions
are 28.17/32.85% on general and 30.39/35.69% on code. TP2 values are 5.14/6.19%
and 5.58/6.44%. These separate diagnostics explain initial placement coverage;
static headline serving remains uninstrumented.

One balanced placement uses equal normalized phase weights, without a weight
sweep or evaluation routes. Calibration cold fractions change from 22.74% to
9.42% for prefill and from 12.22% to 15.32% for decode. These are calibration
coverage values, not serving performance. The profile retains both raw phase
count sets, the objective and construction identity. A decode-only profile
constructed from the same phase-calibration run controls for the slight
difference from historical calibration completion.

The equal-phase artifact is
`3be7108123b093197cc4745298e617fe74526bdd6aea681b864cddc6e7ddf59f`;
its decode-only control is
`6c615ba578a6db160534519d802d4d81347c732627a2d3fffa9223251a14c1e6`.
Calibration retains 137,280 prefill and 241,920 decode selections per model-wide
count set, from eight disjoint mixed-domain requests and 512 generated tokens.
Both placements contain 8,523 resident experts. The balanced objective changes
1,281 memberships relative to its decode-only control. That control differs by
64 memberships from the historical profile because its observed completion
boundary differs; the artifacts are not relabeled as identical calibration.

The sensitivity comparison uses one engine per cell and shows a tradeoff:

| Initial placement | Static generated tok/s | Adaptive generated tok/s | C1 prompt tok/s at 242 / 498 / 1,010 / 1,522 tokens |
| --- | ---: | ---: | --- |
| Decode-only control | 59.177 | 69.508 | 173.19 / 149.26 / 131.45 / 124.42 |
| Equal normalized phases | 56.788 | 68.259 | 187.70 / 156.84 / 135.79 / 128.23 |

C4 prompt rates change from 149.88/131.51/124.37 to
156.34/135.57/127.89 tok/s at 498/1,010/1,522 tokens per request. All
same-recipe output IDs agree. Static generated-token throughput falls 4.04%,
and adaptive throughput falls 1.80%, while prompt execution improves about
3–8%. This single sensitivity comparison does not establish a preferred
production weighting.

Separate generation-zero diagnostics retain exactly equal per-expert phase
counts across the historical and balanced placements. Scoring those same held-out
routes against the decode-only control and balanced profile changes prefill cold
coverage from 29.30% to 15.76%, but decode cold coverage from 34.36% to 37.04%.
These are routing-placement measurements, not predicted throughput. The
original TP1 profile's top-resident-count prefill and decode hot sets overlap
by 70.43%, versus 83.47% at TP2's larger resident count. This metric is the
intersection divided by the admitted resident count, aggregated across layers.
The phases do not request identical working sets, and the two topology values
use different set sizes.

The decode-only adaptive engine completes its output, storage and release gates,
but its outer shell subsequently exits 127 because a launcher was edited while
that shell was still reading it. The engine/source receipt records exit zero;
`phase-control-wrapper-failure.json` retains the separate orchestration failure.
Remaining launches use immutable runner copies. No failed model launch is
reclassified as a successful timing sample.

## Kernel attribution and profiler lifecycle

Separate CUPTI diagnostics retain scheduler phase annotations and actual kernel
launches. Attribution checks the prepared resident-then-backing launch order:
each 48-layer model iteration must contain 96 routed W4A16 launches. Mixed
iterations are retained separately. Kernel durations can overlap and are not
added to serving wall time. Profiling overhead makes the traced model spans
unsuitable replacements for the unprofiled throughput measurements above.

The C4 code diagnostic uses four held-out requests with 16 output tokens each.
It contains one pure-prefill iteration, one mixed iteration with 51 prompt and
two decode tokens, and 15 pure-decode iterations. Kernel-time fractions are:

| Topology and phase | Mapped routed experts | Resident routed experts | NCCL collectives | Other kernels |
| --- | ---: | ---: | ---: | ---: |
| TP1 pure prefill | 89.34% | 5.52% | — | 5.14% |
| TP1 pure decode | 65.22% | 10.42% | — | 24.36% |
| TP2 rank 0 pure prefill | 44.47% | 20.02% | 19.05% | 16.46% |
| TP2 rank 1 pure prefill | 50.87% | 23.51% | 6.65% | 18.97% |
| TP2 rank 0 pure decode | 20.51% | 21.33% | 4.32% | 53.84% |
| TP2 rank 1 pure decode | 20.48% | 21.32% | 4.44% | 53.76% |

Here, “other kernels” includes identified routing, GDN and attention operations
as well as unclassified projections and pointwise work. Eager CPU correlation
identifies shared-MLP linears; captured decode does not retain enough correlation
to separate every shared/dense projection reliably. In TP2 decode, about 45.5%
of total kernel time remains in that unclassified group. It would be incorrect
to assign it all to shared experts or attention.

Mapped service dominates this TP1 diagnostic. TP2 has substantially more
resident capacity and shorter mapped execution; collectives do not dominate its
C4 decode kernel time. NCCL elapsed time includes waiting for another rank and
load imbalance, not just transport. The prefill rank asymmetry illustrates why
one rank's collective time cannot establish network cost.

A separate TP2 C1 trace covers the complete prompt-length series. Per-rank
collective tensor payloads are 96,149,504 / 197,861,376 / 401,285,120 /
604,708,864 bytes at 242 / 498 / 1,010 / 1,522 prompt tokens. These are
correlated all-reduce input tensor sizes, not wire bytes. Mapped-kernel time on
rank 0 grows from 146.5 to 1,826.2 ms across those prompts; rank 1 grows from
145.5 to 1,785.9 ms. Summed collective kernel time is 85.7 / 321.8 / 822.5 /
759.3 ms on rank 0 and 18.7 / 103.6 / 371.8 / 826.8 ms on rank 1. No monotonic
communication-cost model is inferred from these instrumented samples. Captured
decode collective payloads lack CPU shape correlation and remain unreported.

Board PCIe receive/transmit samples are retained with the diagnostics. They
include more than expert traffic, and TP2 includes peer traffic. They do not
provide isolated cold-expert byte counts. The attribution artifacts are
`trace-comparison.json`, the per-rank summaries and their hashed raw traces.

Profiling exposed a distinct shutdown problem. A long TP2 cache profile and an
ordinary non-cache profile exported their traces but exceeded the normal
shutdown deadline. Earlier short profiles also required forced process cleanup.
The deployed Torch implementation builds a retained Python event tree when
vLLM requests its optional summary table; that aggregation was followed by
expensive finalization. The harness now sets the supported
`torch_profiler_dump_cuda_time_total=False` option and analyzes exported traces
offline. It does not extend shutdown timeouts, suppress exceptions or modify
the non-profiled serving path.

The same long TP2 prompt series, an ordinary TP1 control, and the TP1/TP2 C4
diagnostics then export complete traces, release every cache owner and retire
without forced termination. The before/after source and strict close receipts
are retained in `profiler-retention-source.json`. Earlier failures remain
failures; their trace contents do not qualify their shutdown lifecycle.

## Acceptance status

Independent host CI passes at
[`a994091a`](https://github.com/local-inference-lab/b12x/actions/runs/35766194090):
1,195 passed, 40 hardware-dependent skips and 27 documented fixed-contract or
compile-planning exclusions. Registry tests pass; no registry failure is ignored.
Physical portable SM120 acceptance
passes 44 tests with no skips at both `1fed1fa7` and the final core source
`572d3f5b`. Companion tests pass 36 cases
against the rebuilt wheel. Real-checkpoint TP2 layer bring-up passed early,
middle and late layers, mixed placements, repeated/reordered routes, actual
shared-expert composition, fixed-address promotion and allocation-free replay;
final-source reruns and targeted sanitizers are tracked in the evidence bundle.

The `572d3f5b` TP2 rerun covers layers 0, 24 and 47 with 512, 256 and one
resident logical expert. Both ranks pass exact placement/shared-composition
checks and replay-allocation checks. Maximum errors against the independent
reference across ordinary, reordered and duplicate routes are:

| Layer | Reduced max absolute error | Reduced relative L2 | Minimum cosine |
| --- | ---: | ---: | ---: |
| 0 | 0.00003052 | 0.6237% | 0.99998069 |
| 24 | 0.00004578 | 0.6949% | 0.99997598 |
| 47 | 0.00007629 | 0.7131% | 0.99997461 |

These reduction-reference tolerances are distinct from exact same-recipe
placement and serving-output equality. The layer controls use complete local
expert shards and the actual checkpoint shared MLP/gate, not a synthetic shared
module or an all-resident full-model claim.

The final Qwen3-30 regression on `572d3f5b` passes ordinary non-cache serving
and a learned-static/adaptive pair, including actual promotions, unchanged
graph/cache addresses and clean explicit owner release. Both cache arms produce
the same 1,024 token IDs, digest
`8586034a6cf697bad192540d2ea02acba5dcb5309d61926a11707a89a0deb3ae`.
This is a bounded regression control, not a repeat of its historical performance
matrix. The separate large-model TP1/TP2 cancellation tests and four-cycle TP2
reconstruction cover the new distributed lifecycle.

Sanitizer acceptance is narrower than numerical and serving acceptance. The
portable prepared-cache/health-owner cases pass memcheck and synccheck with
zero reported errors. Initial unfiltered real-checkpoint attempts do not:
the long TP2 attempt was terminated without a completed layer receipt, and
single-rank real-layer memcheck/synccheck each reached their 900-second deadline.
Periodic stacks show progress through preparation, capture and the independent
Torch arithmetic oracle; a timeout is not a clean pass or proof of deadlock.

An unfiltered minimal BF16 PyNCCL collective completes on both ranks but exits
99 under memcheck, with all 240 diagnostics retained. They comprise 80
unavailable-image errors at `cudaFuncGetAttributes`, 80 at `cudaGetLastError`,
40 missing FP8 symmetric-kernel compilation diagnostics and 40 accompanying
JIT-log diagnostics. All recorded error stacks reference `enqueue.cc:87`.
Both ranks load NCCL 2.31.2 with SHA-256
`d028ea782ce1798e6ad751d1e14f4b4516a8211a6289579a92f3bcde5e634a79`.

The upstream [NCCL initialization source](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/enqueue/enqueue.cc#L55)
probes kernel attributes and continues after unavailable kernels. This supports
attribution to initialization probes in the minimal control; it does not turn
the failed sanitizer run into a pass or explain every earlier timeout. No API
error is suppressed, and a full distributed zero-error sanitizer gate remains
uncompleted. The bounded production-kernel diagnostics retain their explicit
filters, independent arithmetic assertions and all API errors; their scope is
not whole-program instrumentation.

The bounded TP2 real-layer memcheck also reaches its 600-second deadline,
returning 124 without either rank's layer-completion receipt. Its last visible
site is Torch's additional all-gather communicator initialization, with NCCL
kernel-attribute diagnostics. The dependent TP2 synccheck is not run because
that prerequisite did not complete. `tp2-real-sanitizer-final.json` retains the
exact command, filter scope, log hash and classification. This is an explicit
remaining qualification gate, not a passing distributed sanitizer result.

The compact single-rank production-kernel memcheck and synccheck each reach
their 600-second deadline. They exercise real checkpoint bytes with 16 experts and
top-k 10, preserving the ordinary shared expert and independent reference
assertions. Neither emits a memory/API error heading, but neither produces a
completed test receipt or sanitizer summary. Periodic snapshots progress through actual
W4A16 composition into the reference calculation. This remains incomplete;
the absence of a reported error before termination is not acceptance. No further
sanitizer variants were launched. All experiment containers and GPU compute
processes have exited; only this experiment's telemetry processes were stopped.

## Interpretation and next priority

The remaining acceptance work is a completed real-checkpoint sanitizer run and
a clean distributed sanitizer path through the recorded NCCL/toolchain. Those
gates must close before claiming complete native sanitizer qualification; the
passing portable cases and uninstrumented model tests do not replace them.

The three tiers have different strengths on this model and topology. TP1
learned residency improves on ordinary selective offload, and adaptation
improves on the same learned placement. TP2 admits much more of the working set;
learned static is the fastest measured decode arm and the frozen adaptive
controller adds cost. The basic W4A4 tier is faster on the prompt series. These
results support separate prefill and decode evaluation rather than one
universal tier ranking.

The balanced-profile experiment confirms that phase demand matters, but it
does not remove that execution tradeoff: better prompt coverage costs decode
coverage. No runtime phase switching or prefill replacement is justified by
this single sensitivity experiment. Likewise, TP2 proposal backlog does not
justify increasing its movement budget when the current adaptive arm loses.

The strongest next performance task is a focused W4A16 prefill investigation
using the retained real routes and matched resident/mapped operator controls.
TP1 mapped execution dominates measured kernel time, and basic offload remains
faster on long prompts at both topologies. The investigation should separate
execution cost from launch gaps and TP waiting before choosing a kernel or
packing change. This pass does not establish that asynchronous fills, a new
policy, or communication tuning is the right solution. The unclassified TP2
decode work also prevents a precise shared-expert or attention bottleneck claim.

Format separation, TP-group logical identity, all-rank failure handling and
phase-labelled observations are reusable foundations for Qwen3.8. They do not
qualify its mixed dispatch, FP8 PLE storage, combined admission, PLE teardown or
MTP. Those remain the explicit gates in the preserved Qwen3.8 audit. NVFP4 is
the only physical adapter here, and N=2 is the only physical multi-rank result.

Policy and ownership conclusions are distinct from hardware timing. The need
for one logical map, per-rank admission, explicit recipe identity and separate
phase evidence is portable. Throughput, copy cost, mapped service and collective
timings belong to these two SM120 devices on Gen4 x16. No Gen5 or Grace value is
inferred.

## Evidence map

The external bundle retains the full launch commands and raw data. Useful entry
points are:

| Evidence | Contents |
| --- | --- |
| `artifact.json`, `installed-verification.json`, per-launch `*-source.json` | Source exports, complete companion wheel and loaded-library identities |
| `loaded-artifacts-check.json`, `primary-matrix-check.json` | All primary worker artifacts, exact paired outputs and frozen settings |
| `tp2-capacity-plan.json`, `serving-profile-index.json` | Per-rank admission, rejected all-resident case and actual serving profile |
| `tp2-clean-coverage-0.json`, `tp2-clean-coverage-1.json` | Complete per-rank target tensor coverage and optional MTP exclusion |
| `three-tier-summary.json` | Raw fresh-engine rates, paired differences, latency and control costs |
| `prefill-comparison.json`, `phase-event-overhead.json` | Direct prompt rates, chunks, TTFT and instrumentation controls |
| `phase-profile-comparison.json` | Phase count identity, placement coverage and balanced-profile comparison |
| `ordinary-final/repeatability.json` | Frozen W4A4 arithmetic and repeatability diagnostic |
| `tp2-real-layer-errors-summary.json` | Early/middle/late TP2 reference errors and placement invariants |
| `trace-comparison.json`, `profiler-retention-source.json` | Kernel attribution, raw trace identities and profiler lifecycle controls |
| `tp2-reconstruct-four-final-check.json` | Four-engine owner release and unchanged resource-growth acceptance |
| `nccl-minimal-sanitizer-classification.json`, `tp2-real-sanitizer-final.json` | Unfiltered NCCL diagnostics and incomplete distributed sanitizer gate |
| `next80-filtered-memcheck-result.json`, `next80-filtered-synccheck-result.json` | Bounded real-checkpoint deadlines and retained scope |
| `final-process-cleanup.json` | Empty GPU compute inventory and retirement of experiment-owned telemetry |

The [architecture guide](hybrid-inference.md#reproduce-the-added-gates) gives the
portable commands and capability limits. Historical evidence stays attached to
its original source; this report does not relabel earlier qualification as a
new performance improvement.
