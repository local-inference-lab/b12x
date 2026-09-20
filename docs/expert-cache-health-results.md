# Expert-cache health serving evidence

Status: **experimental SM120 serving evidence, 2026-09-20 UTC**. Read-only health
probes substantially reduce full maintenance while the learned cache is healthy.
They react much sooner than blind backoff on these transitions. They do not
consistently outperform fixed maintenance: less frequent policy observations can
give back some transition performance. No adaptive production default changes.

The [health contract](expert-cache-health.md) describes preparation, ownership,
generation checks, cancellation, configuration and the distinction between
health probes and policy windows. BF16 router arithmetic and W4A16 whole-K
execution remain unchanged.

## Source and qualification

The matrix uses frozen b12x archive `source-04.tar.gz`, based on
`173d0746f80d317ee8b2ac9d4d463716fe065849`, with archive SHA256
`604368dc0ad51a713ecd84cf8454a97a015d2191b9b98e216a1d4260a4389b81`.
The final package differs only in the public threshold export and SM103 module
admission; timing receipts retain the archive identity. Final package SHA256 is
`a895d1e279365bbb72d6e72b84930879f9940422a5828a9ad4156b2bb47f5c97`.

The companion remains `3e45b530e58186046383e7294e611c2f6bf5cfb8`. The complete
source-built wheel SHA256 is
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
Fresh verification checks all eleven installed native libraries against that
wheel and the changed companion Python files against current HEAD. The wheel's
version label retains `1d1f870b`; its source-matched build identity is documented
by the file hashes, not inferred from that label.

Hardware is `ripper`, RTX PRO 4000 Blackwell 24 GB,
UUID `GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02,
Torch 2.13.0/CUDA 13.3, CUTLASS DSL 4.6.2 and Triton
3.7.1+gitf797708c.nv26.7. The host is a
single-NUMA Threadripper PRO 5975WX. The link negotiates **PCIe Gen4 x16** under
load. Dynamic clocks and power states are retained at 500 ms intervals.
This is one run per arm; small differences are not statistical winners.

The real checkpoint is Qwen3-30B-A3B-NVFP4, fingerprint
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`.
All arms start from the same learned general profile,
`20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`,
with 58 of 128 experts resident in each of 48 layers. This is qualification
geometry, not a public API restriction. Serving uses real routing weights,
an 8 GiB expert-device envelope, 2 GiB BF16 KV, context 2048, prepared capacity
64, FlashInfer attention and full decode graphs for sizes 1 and C. Greedy
requests generate 128 tokens with EOS ignored; prefix caching is disabled.

Each corpus has eight stable requests followed by eight transition requests.
The 48 independently authored prompts are disjoint from the retained calibration
and earlier evaluation fixtures. Controlled group admission is identical within
each comparison. Arm order is reversed at C4; the prose corpus uses another
order. Every arm starts a fresh engine with the original learned placement.

All **45 runs, 92,160 generated tokens and 8,512 promotions** pass exact paired
output equality and retain captured graph and cache addresses. These are direct
AsyncLLM serving measurements, including control and group-admission costs,
without an HTTP transport. They are not unconstrained-arrival or quality tests.

## Overall throughput

Units are generated tokens per second; higher is better. Static has no observer.
Counters adds observation only. Fixed checks every 32 delivered tokens. Backoff
extends healthy checks from 32 to 256. Health probes every 32 delivered tokens,
with full maintenance on global cold fraction at least 15% or a 1024-token
maximum interval. Every adaptive arm uses the existing experimental decayed-LFU
policy, at most 16 model-wide pairs and 64 MiB copy bytes per maintenance.

| Corpus | C | Static | Counters | Fixed | Backoff | Health |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat → code | 1 | 55.03 | 54.66 | 67.61 | 63.74 | 68.08 |
| Chat → code | 4 | 90.44 | 90.20 | 104.52 | 100.04 | 104.56 |
| Chat → code | 8 | 130.95 | 130.87 | 141.51 | 136.30 | 140.74 |
| Prose → math | 1 | 47.08 | 46.85 | 53.37 | 51.03 | 53.35 |
| Prose → math | 4 | 66.42 | 66.24 | 74.80 | 71.24 | 74.18 |
| Prose → math | 8 | 83.88 | 83.65 | 88.57 | 88.93 | 87.81 |
| English → multilingual | 1 | 57.84 | 57.44 | 64.64 | 62.90 | 66.20 |
| English → multilingual | 4 | 87.98 | 87.59 | 98.87 | 95.02 | 99.55 |
| English → multilingual | 8 | 119.30 | 118.65 | 129.29 | 123.92 | 128.24 |

Health exceeds static by 4.7–23.7% over these complete sequences. It exceeds
backoff in eight of nine cells, but does not dominate fixed checks. In
particular, prose-to-math C8 favors backoff over health. Preserve that outcome;
the experiment does not select a universal cadence.

## Stable intervals

| Corpus | C | Static tok/s | Health tok/s | Fixed / health full checks | Health promotions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chat → code | 1 | 90.35 | 89.17 | 30 / 2 | 32 |
| Chat → code | 4 | 162.68 | 161.74 | 25 / 0 | 0 |
| Chat → code | 8 | 252.96 | 252.04 | 21 / 0 | 0 |
| Prose → math | 1 | 90.44 | 89.95 | 29 / 2 | 32 |
| Prose → math | 4 | 125.19 | 127.41 | 25 / 4 | 64 |
| Prose → math | 8 | 164.82 | 164.90 | 21 / 0 | 0 |
| English → multilingual | 1 | 85.30 | 85.36 | 29 / 4 | 64 |
| English → multilingual | 4 | 162.70 | 161.70 | 25 / 0 | 0 |
| English → multilingual | 8 | 218.42 | 217.80 | 21 / 0 | 0 |

Across the matrix, fixed performs 226 stable-interval full checks; health
performs twelve. Health stable throughput ranges from 1.3% below static to 1.8%
above it. Some stable prompts still create routing pressure and useful policy
opportunities; the label does not mean the learned profile is perfect for every
new request. Small apparent gains remain subject to dynamic-clock noise.

## Transition intervals

| Corpus | C | Static tok/s | Fixed tok/s | Backoff tok/s | Health tok/s | First pressure: fixed / backoff / health (s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chat → code | 1 | 39.64 | 55.91 | 49.92 | 55.20 | 1.02 / 5.73 / 1.22 |
| Chat → code | 4 | 62.80 | 80.03 | 73.11 | 77.54 | 0.65 / 4.54 / 0.63 |
| Chat → code | 8 | 88.68 | 102.36 | 94.50 | 98.13 | 1.11 / 4.19 / 1.09 |
| Prose → math | 1 | 31.86 | 38.55 | 35.90 | 37.99 | 0.46 / 6.84 / 1.27 |
| Prose → math | 4 | 45.29 | 54.22 | 50.18 | 52.53 | 1.13 / 6.61 / 0.88 |
| Prose → math | 8 | 56.40 | 62.09 | 61.42 | 60.03 | 1.68 / 2.20 / 1.67 |
| English → multilingual | 1 | 43.84 | 53.24 | 50.51 | 54.20 | 1.04 / 6.89 / 1.22 |
| English → multilingual | 4 | 60.45 | 73.70 | 67.91 | 72.14 | 0.84 / 4.85 / 0.82 |
| English → multilingual | 8 | 82.36 | 94.39 | 87.57 | 91.29 | 1.94 / 4.63 / 2.11 |

First pressure is the time the client receives a pressure result after workload
admission. It is not the GPU's first cold selection. Health detects pressure in
0.63–2.11 seconds in these cells, generally near fixed checks and earlier than
backoff. Prose-to-math C1 is an exception to equally fast fixed detection:
fixed detects at 0.46 seconds and health at 1.27 seconds.

The response can arrive while further tokens are generated. Delivered-token
intervals are therefore approximate; the raw records retain actual completion
counts. The health arm sometimes performs fewer full observations and movements
during the transition. Its long-window LFU state is not equivalent to fixed
checks or the earlier offline four-step policy.

## Control cost and latency tails

The matrix contains 425 health probes. Their complete worker and client costs
are distinct from the CUDA-event intervals:

| Measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Reduction event interval | 4.10 µs | 5.73 µs | 151.84 µs |
| 2,304-byte copy event interval | 5.60 µs | 9.73 µs | 58.37 µs |
| Worker submission, including owner checks | 1.32 ms | 1.97 ms | 2.05 ms |
| Worker poll and result decoding | 0.99 ms | 1.59 ms | 1.66 ms |
| Client response while serving continues | 48.77 ms | 161.68 ms | 271.33 ms |

Event intervals can include submission gaps; they are not a profiler's isolated
instruction-body timing. The outliers remain in the receipts. Client response
includes utility queuing and GPU completion while normal work continues; it is
not a pause. Worker times include generation and pointer validation. There is
no claim of free observation.

For the chat corpus's fixed no-movement checks, median worker RPC is about
14.1–14.2 ms: policy takes 6.8–7.1 ms, full snapshot takes about 2 ms, and its
host decode takes about 1.84–1.86 ms. Snapshot D2H is about 0.06 ms. Scheduler
drain medians rise from 8.72 ms at C1 to 29.75 ms at C8; that includes waiting
for already-submitted useful work. Health skips those stages when pressure is
low. Counters-only throughput is 0.06–0.70% below static in the nine cells,
within the range where dynamic-clock noise limits precise attribution.

Full maintenance still creates latency spikes when movement is needed. These
are client delivery gaps, not CUDA decode-iteration timings. The full analysis
retains p50/p95/p99 distributions for every arm and interval, including gaps
overlapping maintenance and the final control tail.

| Corpus / C | TTFT p50: static / health (ms) | Delivery gap p95: static / health (ms) | Health maintenance p95 (ms) |
| --- | ---: | ---: | ---: |
| chat_code-c1 | 148.87 / 131.99 | 32.46 / 25.90 | 53.01 |
| chat_code-c4 | 347.64 / 308.17 | 72.27 / 69.62 | 93.70 |
| chat_code-c8 | 418.74 / 421.17 | 100.91 / 98.04 | 126.00 |
| prose_math-c1 | 156.52 / 142.30 | 43.48 / 37.21 | 70.54 |
| prose_math-c4 | 365.58 / 338.61 | 100.98 / 94.27 | 119.68 |
| prose_math-c8 | 429.70 / 432.67 | 152.61 / 148.45 | 173.97 |
| english_multilingual-c1 | 139.42 / 132.51 | 29.67 / 26.15 | 57.80 |
| english_multilingual-c4 | 329.58 / 297.09 | 75.08 / 67.03 | 95.18 |
| english_multilingual-c8 | 409.21 / 411.54 | 99.97 / 102.12 | 112.40 |

## Health-signal diagnostics

The global 15% rule reports pressure at twelve of 224 stable probes and 169 of
201 transition probes. Requiring at least half the layers to exceed 15% retains
eleven stable and 163 transition observations. Requiring 80% retains zero
stable and 116 transition observations. These rules are evaluated on the
recorded maps produced by the global rule; different decisions would change
future maps, so this is not a serving result for the alternative gates.

Repeated cold selections also overlap substantially: among pressure probes,
per-cell median repeated shares of cold selections range from about 63–77%
during stable traffic and 68–78% after transition. They do not establish a
cleaner trigger here. The simple global rule remains the experimental baseline;
no classifier or worst-layer default is added. The maximum-interval fallback
fires once in the matrix. Longer healthy regimes remain a separate test.

## Regime duration and retrospective payback

The C1 chat-to-code duration experiment keeps the eight-request stable prefix
and the code prefix fixed. It tests two, eight and sixteen code requests, each
generating 128 tokens. The longest fixture adds eight independently authored
coding prompts. Static code intervals last approximately 6.01, 25.83 and
54.66 seconds. This does not measure a homogeneous workload lasting minutes or
hours.

Positive complete-run gain means static took longer. Payback is the first
matched delivered-token point that remains ahead through the recorded code
interval, after charging the preceding stable interval's relative cost. Time
is measured from code admission on the candidate clock. Complete-run gain also
charges outstanding control at the end; payback curves stop at token delivery.

| Code tokens | Control | Complete-run gain (s) | Payback: seconds / code tokens | Transition promotions |
| ---: | --- | ---: | --- | ---: |
| 256 | Fixed | 0.09 | 5.05 / 237 | 112 |
| 256 | Backoff | -0.27 | No crossing | 16 |
| 256 | Health | 0.41 | 4.00 / 168 | 96 |
| 1024 | Fixed | 6.92 | 4.87 / 230 | 320 |
| 1024 | Backoff | 5.08 | 8.48 / 378 | 192 |
| 1024 | Health | 7.13 | 4.09 / 174 | 304 |
| 2048 | Fixed | 19.77 | 4.93 / 232 | 512 |
| 2048 | Backoff | 15.87 | 8.27 / 371 | 336 |
| 2048 | Health | 20.15 | 3.98 / 166 | 512 |

For this prefix, health repays the earlier cost at roughly four seconds or
166–174 code tokens. Backoff detects the short regime only after about 5.74
seconds, performs one late sixteen-pair maintenance and never repays before the
regime ends. The small fixed-check gain in the short run is within the range
where repetition is needed. These are retrospective, topology-specific
observations, not a production dwell-time threshold or a prediction for a new
workload. Other corpora can already be ahead at transition because earlier
promotions improved their stable interval.

## Probe interval experiment

The additional chat-to-code C4 runs vary only the nominal delivered-token probe
interval. The 32-token row is the matrix arm. All retain exact paired outputs
and unchanged graph/cache addresses.

| Probe interval | Overall tok/s | Stable tok/s | Code tok/s | Stable probes / full checks | Code full checks / promotions |
| ---: | ---: | ---: | ---: | --- | --- |
| 16 | 108.51 | 161.77 | 82.14 | 42 / 0 | 23 / 368 |
| 32 | 104.56 | 161.74 | 77.54 | 25 / 0 | 18 / 288 |
| 64 | 99.19 | 161.79 | 71.90 | 14 / 0 | 13 / 208 |
| 128 | 94.98 | 161.80 | 67.46 | 7 / 0 | 7 / 112 |

Static is 90.44 tok/s overall and 162.68 during stable traffic. The measured
stable cost is nearly unchanged across these intervals, while more frequent
probes permit more full policy updates during code. The 16-token arm also takes
a maximum-interval snapshot at 0.52 seconds after code admission, before its
first pressure report at 0.95 seconds. Its first sixteen-pair publication
finishes at 0.60 seconds. That fallback contributes to the comparison; the
result cannot be attributed solely to faster pressure detection.

Sixteen tokens is the best recorded interval in this one C4 fixture, not a new
default. Repetition across other corpora and concurrency levels is required
before selecting a general setting. The public control remains explicit and
the benchmark's default interval remains 32.

## Validation and retained failures

The complete serving set is **56 runs, 115,712 generated tokens and 10,912
promotions**, including the matrix, two additional regime lengths and three
additional probe intervals. All paired outputs match. Common prompt prefixes
also match exactly across duration experiments. Every cache run verifies
unchanged captured graphs and expert-storage addresses.

Focused host validation passes 132 tests; preparation/admission validation
passes another 99. The final prepared-cache/counter suite passes 25 tests in
55.26 seconds, including real checkpoint bytes. Four health cases pass
Compute Sanitizer memcheck in 37.67 seconds and synccheck in 16.87 seconds,
with zero errors. They cover empty and resident/cold traffic, invalid and
duplicate IDs, 7/128/385-expert geometries, graph replay with frozen resolution,
stable addresses, zero Torch CUDA allocator events, generation changes,
counter resets, unsigned totals above the signed range and overflow rejection.

Ten source-built companion maintenance/loader tests pass in 1.34 seconds. An
ordinary V2 non-cache smoke generates 512 tokens with two captured decode
graphs, no cache configuration and no worker extension. Loaded native-library
hashes match the verified wheel. It completes normal engine shutdown.

The final SM103 health declaration compiles four native objects at the package
hash above, without a CUDA context. The reduction uses 32 registers, zero
stack/local memory, 1 KiB reported static shared memory and a 6 KiB dynamic
shared allocation. The metadata inventory is 87 declarations/245 programs;
this targeted compile does not replace historical full-corpus receipts.

Initial CuTe attempts failed on unsigned control-flow type joins and on passing
a shape tuple instead of a layout to shared allocation. The first SM103 attempt
failed module admission. Those sources and logs remain retained. An earlier GPU
bundle skipped the checkpoint test because it omitted the checkpoint mount;
the final 25-test run closes that gap. GPU-test formatting after validation is
verified to preserve the exact tested AST.

Two ordinary-smoke driver attempts are retained: a missing multiprocessing
entry guard failed startup, and a diagnostic callable RPC was rejected by the
engine's default serialization guard. The successful driver uses ordinary
request APIs and external process-library inspection; it does not enable
insecure serialization. Existing optional Triton import warnings and ignored
interpreter-teardown warnings in other runs remain in their logs. Broad lint
also retains twelve pre-existing export/compact-test diagnostics; new code is
lint-clean. No engine, transport or numerical fix is attributed to these
driver/setup failures. Independent PR CI and physical B300 remain open gates.

## Next experiments

1. Repeat the 16-token probe result across the other corpora and concurrency
   levels. It produced more useful full updates in one fixture without a
   measurable stable penalty; it is not yet a general setting.
2. Study the actual policy observation intervals. Prose-to-math C8 remains a
   losing comparison against backoff, despite equally fast pressure detection.
   Shorter-history counters and larger movement budgets address different
   causes and should be isolated before either changes.
3. Measure whether an engine completion-output hook can avoid the two utility
   round trips and repeated owner checks while preserving result ownership.
   The worker work, not the microsecond reduction, is the remaining probe cost.
4. Extend dwell and arrival-pattern coverage. The existing payback curves do
   not establish behavior under sustained mixed traffic or hour-long regimes.

No evidence here calls for asynchronous replacement, a new PCIe transport or
changed arithmetic. Gen5 and B300 experiments must retain their separate
topology and native-execution qualification contracts.

## Evidence locations and reproduction

Raw receipts, commands, telemetry, failures, source archives and analysis are
retained in `/home/jasonc/b12x-health-evidence-20260920`, with physical originals
in `/home/jasonc/b12x-health-results-20260920` on `ripper`. `matrix.sh` and
`run-serving.sh` record the complete container invocation, mounts, environment,
GPU selection and arm order. The repository fixtures and benchmark CLI are
described in the health contract. `benchmarks.moe.summarize_expert_health`
validates exact paired token hashes and retains per-interval latency, pressure,
movement and retrospective payback curves. Historical receipts are unchanged.

Control-stage timings describe this engine/CPU as well as the GPU. Expert fill,
direct-host miss cost and total throughput remain specific to the recorded
Gen4 topology. There are no inferred Gen5 results and no physical SM103/B300
qualification claims.
