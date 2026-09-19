# Expert-cache locality and canonical-fill experiments

Status: **research-only**. Offline routing analysis and SM120 canonical-fill
benchmarks extend the fixed-address cache experiment. Static serving, automatic
profile activation, shared cache policy, SM103 storage, and numerical kernels
are unchanged. No concurrent replacement or serving-engine cache is added.

The main result is that a good initial placement and a good replacement decision
are separate requirements. Real routes exhibit locality, and a complete
canonical fill costs less than a reversible exchange. Nevertheless, the measured
layer-zero adaptive replay loses to its learned static initial placement because
the chosen victims cause more misses than the promoted experts avoid.

## Routing evidence and its limits

The corpus contains 18 sequential requests to the Qwen3.8-Flash-Next-NVFP4
checkpoint: three prompts each for prose, code, math, multilingual, chat, and
agent/tool-planning text. The agent prompts ask for plans and example commands;
no real tools run. These are authored evaluation prompts, not production logs.
Each request generates 192 tokens. Native vLLM routed-expert export supplies
48 target layers, 512 canonical experts and top-k 10. Speculation is disabled
and concurrency is one. Prompt rows and the final unprocessed output token are
excluded, leaving 191 decode invocations per request.

One request per class supplies training counts. The remaining two are held out:
1,146 training and 2,292 evaluation invocations per layer, respectively 550,080
and 1,100,160 selections across the model. Training counts never include the
held-out requests. The mixed-workload experiment explicitly blends the six
training classes. Separate workload experiments use only their own training
request. Neither produces or overwrites an automatic serving profile.

Routes were collected on the four-Spark TP4 lane with vLLM `76061de4` and b12x
`d2d5368d`, using its native `--enable-return-routed-experts` option. Each
returned record represents one canonical selection, not four replicated TP
observations. The original lane scripts and speculative configuration were
restored; health and a generation smoke check passed. Instrumented capture is
not throughput evidence. No production monkeypatch or trace node is installed
by this change.

The checkpoint export manifest is
`8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397`.
Independent reads on the capture host and ripper agree on the layer-zero native
field SHA256:
`05384d5b0bbe71843464786f15391673847f5eaa9fa08e2a2e8309ab80c6c90e`.
The layer-zero physical replay uses H=2560, I=640, E=512, top-k=10, full expert
rows and synthetic activations. TP4 route observation does not turn this into a
TP4 kernel benchmark or complete-model quality evaluation.

## Locality and policy replay

[`residency_replay.py`](../b12x/testing/residency_replay.py) computes per-layer
selection counts, unique experts per invocation/window, original-order stack
distance, invocation inter-arrival distance, hot-set stability, and top-N
coverage. Coverage ranked from evaluation counts is a descriptive upper bound;
it is not the training-only initial placement used in policy comparisons.

For each unique cold expert touch it retains time to next reuse and later route
selections/distinct invocations over configurable horizons. Duplicate selections
in a call count independently but do not count as future invocations. Reuse
statistics stop at request boundaries. A horizon extending past a request is
censored, not classified as a failed reuse. The 512-step result is unavailable
for this corpus; the analyzer supports longer captures without guessing it.

Across layers, the probability that a touched expert appears again within four
invocations ranges from **21.49% to 78.99%**, median **58.56%**. A 16-invocation
window touches, on average, between **36.63 and 111.50** distinct experts,
depending on layer; the median layer touches **62.30**. A single locality model
for every layer would discard this variation.

Conditioned specifically on a miss under the learned 256-expert static map:

| Later invocations | Eligible cold touches | Reuse probability | Mean later selections |
| --- | ---: | ---: | ---: |
| 1 | 242,450 | 38.04% | 0.380 |
| 4 | 238,118 | 56.14% | 1.225 |
| 16 | 220,590 | 73.25% | 3.915 |
| 128 | 69,266 | 90.77% | 24.304 |
| 512 | 0 | unavailable | unavailable |

The sweep contains **3,744 offline policy fixtures** across all 48 layers:
resident counts 128/256/384, learned/positional starts, and decision windows
4/16/128. A separate 216-fixture comparison covers six workload classes on
layers 0/12/24/47 with 256 resident experts and windows 4/16. These are count
simulations, not GPU benchmark fixtures. Replay explicitly concatenates test
requests in manifest order (split, then workload name); this is a declared
serial workload-transition scenario, not a reconstruction of capture timing.

All adaptive comparisons admit at most one pair at a window boundary, require
two cold observations, and apply a one-window residency guard. Frequency
comparisons require a score advantage of two. The b12x arm invokes the actual
`ResidencyCacheController`, including generation/acknowledgement validation.
Offline LRU ranks last-observed invocation; LFU ranks cumulative evaluation
counts; decayed LFU halves accumulated counts once per decision window. LRU
requires a more recent touch rather than a frequency margin. These comparison
policies are not added to the serving API.

With 256 resident experts, aggregated held-out cold-selection fractions are:

| Initial population / policy | Window | Cold selections | Promotions | Completed zero-hit promotions |
| --- | ---: | ---: | ---: | ---: |
| Positional / static | — | 48.84% | 0 | 0 |
| Learned / static | — | 22.16% | 0 | 0 |
| Learned / b12x | 4 | 19.66% | 18,525 | 5,835 / 14,810 completed |
| Learned / b12x | 16 | 19.35% | 6,712 | 1,138 / 3,982 completed |
| Learned / LRU | 4 | 14.05% | 12,944 | 510 / 6,098 completed |
| Learned / LFU | 4 | 15.24% | 8,229 | 482 / 2,952 completed |
| Learned / decayed LFU | 4 | 13.67% | 12,680 | 507 / 5,900 completed |
| Learned / LFU | 16 | 17.53% | 5,707 | 265 / 1,326 completed |

A completed promotion ends at eviction. Promotions still resident at trace end
remain censored; zero-hit censored records are reported separately. No promotion
is issued after the final invocation. Learned static cold fractions for 128,
256 and 384 experts are respectively **47.22%, 22.16%, and 8.08%**.

This corpus supports testing LRU/decayed frequency and longer frequency history
before changing the production policy. It does not establish their throughput
ranking: each avoided miss and each movement can have different costs, and
these requests are short. Separate workload tables and per-window membership
records are retained in the evidence directory, including unfavorable outcomes.

`retrospective_cost` provides an explicit linear sensitivity calculation from
caller-supplied costs. It reports useful hits, residence length and time to
amortize a promotion. Gross earned-hit attribution excludes victim harm;
net-versus-static uses the difference in cold selections and therefore includes
that harm. No SM120 or GB300 timing constant is built into the analyzer. A
constant cost per cold selection is not a model of nonlinear MoE execution.

## Complete canonical-fill transaction

[`CanonicalFills`](../benchmarks/moe/sm120_canonical_fill.py) is a benchmark
primitive, separate from the unchanged journaled exchange. Every expert retains
canonical native bytes in backing row E, including resident experts. Only the
resident slot and expert-to-execution-row map change:

1. The engine/operator harness pauses all producers, including raw graph replay.
2. Drain the device and read back the mapping to validate its expected generation.
3. Optionally copy the candidate's six native fields into bounded pinned staging.
4. Overwrite the selected VRAM row from canonical source and wait for completion.
5. Publish the fixed-address map: candidate to VRAM row; victim to backing row V.
6. Wait for publication, then advance the host generation and permit resumption.

After any potentially submitted slot write, recovery reloads the victim from
immutable canonical backing, waits, restores the original map, and waits again.
The old map alone is insufficient after overwrite. A failed recovery or a
foreign device map poisons the lane; resuming its graphs is forbidden. Stale
snapshots, invalid pairs and capture-time mutations fail closed. No copy/map
publication occurs inside graph replay. Preparation and the experiment retain
backing, staging, map, programs and scratch owners until graphs are released.

The canonical/source views are verified during preparation and must remain
immutable for the experiment's lifetime. This is an ownership contract, not
hardware write protection against arbitrary tensor mutation. All fields move:
w13, w2, s13, s2 and the two FP32 global scales. Router rows, logical identity,
source quantization and original-top-k finalization remain unchanged.

The complete transaction benchmark compares cacheable pinned source, warm
pageable source, and warm pageable source through one bounded cacheable pinned
row. Each arm includes validation, completion, publication and generation
advance, rather than only a memcpy. Raw event intervals and API/synchronization
wall time are separate. Instrumentation increases host time; uninstrumented
interleaved samples provide the headline pauses. Events around synchronous host
copies can include device-stream idle time and are not isolated DMA rates.

One expert is **2,764,808 bytes**. Successful exclusive exchange issues four
payload copies (11,059,232 bytes): VRAM→journal, mapped backing→journal,
journal→VRAM and journal→mapped backing. Canonical fill issues one payload H2D;
staged fill additionally copies one payload CPU→staging. Both mechanisms read
and publish an E×2 int32 map, 8,192 bytes total for E=512. Recovery traffic is
additional and occurs only on failure.

Submitted-write fault injection measures failed-transaction-plus-recovery wall
time separately, with four samples per point and transport. Failure after the
first payload field or after map publication restores canonical outputs and the
original generation. Pinned-source medians are 0.887/0.854 ms; pageable-source
medians 0.844/0.898 ms; staged-source medians 1.023/0.996 ms. These are benign
Python-injected exceptions after successful CUDA submission, not a latency
model for real device faults. A sticky CUDA error may prevent recovery entirely.

Canonical backing increases mapped memory by **707,790,848 bytes per layer**
when 256 experts are resident. Across 48 such layers this is approximately
**31.64 GiB**. The benchmark also retains the shared source loader's pageable
fields and an all-VRAM correctness control. Its allocation receipt distinguishes
these from execution storage. The staged arm uses one payload row within the
existing bounded allocation; it does not pin an entire checkpoint for staging.
However, every canonical arm still retains a fully mapped canonical layer for
miss execution and recovery. An ordinary-RAM or mmap-only full-model backing
service is **not implemented** by this experiment.

The shared `ExpertPlacement`/controller validation presently describes exclusive,
dense physical rows. Canonical backing has unused backup rows for hot experts,
so its snapshots cannot be handed directly to that validator. Physical replay
uses the causal offline controller's canonical-ID decisions and a separate
benchmark fill mechanism. Productizing this requires an explicit backing/row
capacity contract and a backend-owned miss-service implementation; weakening the
existing exclusive-map checks would hide that missing contract.

## Physical SM120 results

Twenty uninstrumented transactions per arm alternate ordering on the same card.
The final seeded run uses RTX PRO 4000 UUID
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. It reports complete wall time:

| Transaction | Median | Minimum–maximum | Relative to matched exchange |
| --- | ---: | ---: | ---: |
| Exclusive pair exchange | 1.244 ms | 1.080–1.301 ms | 1.000 |
| Cacheable pinned canonical fill | 0.538 ms | 0.478–0.691 ms | 0.432 |
| Warm pageable canonical fill | 0.601 ms | 0.546–0.778 ms | 0.483 |
| Warm pageable → bounded pinned staging → fill | 0.784 ms | 0.728–0.888 ms | 0.630 |

A separate unseeded run on UUID `GPU-47363510-b87a-13a5-4824-2542e97df76c`
reports medians 1.268/0.626/0.673/0.846 ms in the same arm order. It is a
supplementary physical check, not a matched two-card performance comparison.
The preceding cost receipt's 0.860 ms exchange remains historical; these matched
runs do not assume host wall time is invariant across experiments.

Canonical fill removes the two host-to-host payload copies and the victim's
VRAM-to-journal transfer. It also retains per-row field views and omits the
journal-completion boundary. Per-field raw records distinguish those savings
from Python bookkeeping, map transfers and synchronization. The complete staged
path remains faster than exchange in this experiment, although its extra CPU
copy gives it less margin than directly pinned canonical backing. No file-backed
page fault, temporary registration, disk bandwidth or hardware peak is measured.

The final physical real-route replay covers all 2,292 held-out layer-zero
invocations with a learned 256-expert initial map, a 16-invocation decision
window, one pair per decision and the unchanged b12x policy. The three arms
alternate ordering, use the same seeded activations and apply equal validation
work at decision boundaries. They replay fixed captured graphs with changing
IDs. Reported time is the sum of graph CUDA-event intervals and complete
transaction wall intervals; input copies, offline policy calculation, validation
and serving scheduling are excluded.

| Arm | Cold selections | Graph time sum | Transaction time sum | Operator + transaction / static |
| --- | ---: | ---: | ---: | ---: |
| Learned static | 7,040 | 1,127.54 ms | 0 | 1.000 |
| Adaptive exclusive exchange | 7,323 | 1,160.48 ms | 159.82 ms | 1.171 |
| Adaptive canonical fill | 7,323 | 1,154.61 ms | 51.15 ms | 1.069 |

Ratios above one are worse. This is an operator diagnostic, not observed serving
slowdown. Both adaptive mechanisms make the same 142 decisions and earn 1,706
later resident selections. Of 86 completed promotion lifetimes, 35 earn no
later hit; three additional zero-hit promotions remain censored at the end.
Victim losses outweigh earned hits by 283 selections relative to static. A
cheaper fill therefore cannot make this particular policy/trace profitable.
Individual promotion profitability cannot be inferred from the grouped MoE
latency by charging every route an identical fraction of the operator time.

Driver 580.173.02, Torch 2.13.0/CUDA 13.3, CUTLASS DSL 4.6.2, Triton
`3.7.1+gitf797708c.nv26.7` and cuda-bindings 13.0.3 match the preceding physical
experiment. Both cards negotiate PCIe Gen4 ×16 under load. Root-complex topology,
IOMMU-related boot configuration, NUMA inventory, CPU affinity and per-round
GPU snapshots are retained. There is one NUMA node; local/remote comparison is
unavailable. Clocks are dynamic, power limits remain 145 W, and the control
thread permits CPUs 0–63. No claim of fixed-clock release acceptance is made.

## Scheduling and deferred experiments

Canonical counters plus an unchanged map generation are sufficient to count
resident and cold selections at a boundary. No additional per-route atomic is
added. These counts cannot recover order, next-use distance or per-step unique
touches; rich invocation traces supply those research observations.

No asynchronous overlap result is claimed. The accepted-token export has no GPU
layer timestamps, and its C1 contract contains neither verifier groups nor
concurrent requests. The transaction uses a device-wide drain, so it cannot
overlap useful compute by construction. Historical SM120 hot-operation times
and transfer intervals can bound an idealized overlap study, but cannot prove
that a fill, hot compute and backing reads coexist without contention.

A spare slot would require 2,764,808 additional VRAM bytes per qualification
layer, or one fewer hot expert at a fixed budget. The analyzer can compare
budgets 255 and 256 before allocating it. Safe asynchronous publication still
needs reader retirement, stream events and engine scheduling; none is supplied
by a spare tensor alone. No spare-slot or hot-first replacement is implemented.

Priorities supported by this evidence:

1. Evaluate restrained LRU/decayed frequency and victim-aware profitability on
   longer held-out traces. The present recent-window policy creates substantially
   more zero-hit evictions in this corpus.
2. Measure the strongest offline candidates on several real-route layers using
   complete fill costs. A lower miss rate alone is not acceptance.
3. Add timestamped invocation traces for verifier/concurrent schedules. Measure
   safe copy overlap only after the usable window and lifetime contract exist.
4. Design ordinary/mmap backing with bounded pinned staging and explicit miss
   service, then compare it with full mapped canonical backing under memory
   pressure. Warm pageable-copy timing does not include disk or page faults.
5. Qualify the unchanged SM103 static HBM/Grace path on B300 before extrapolating
   cache economics or implementing concurrent Grace replacement.

## Reproduction and validation

Native vLLM response conversion requires a manifest with `execution` equal to
`{"concurrency": 1, "speculative": false}`, checkpoint identity, `experts`,
`hidden`, `intermediate`, `layers`, and ordered `requests`. Each request names
its response path/SHA256, request ID, workload and `train`/`test` split. Use a
worker invocation trace instead for accepted/rejected speculative work or
concurrent batching; accepted-token exports cannot recover that schedule.

```sh
python scripts/import_vllm_expert_trace.py /evidence/import-manifest.json \
  --output /evidence/invocations.json
PYTHONPATH=. python scripts/analyze_expert_cache.py /evidence/invocations.json \
  --output /evidence/replay --budgets 128 256 384 --windows 4 16 128 \
  --initial learned positional
PYTHONPATH=. python scripts/analyze_expert_cache.py /evidence/invocations.json \
  --output /evidence/code --workloads code --budgets 256 --windows 4 16 \
  --initial learned --details
python -m benchmarks.moe.sm120_promotion_costs \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 --output /evidence/fills
python -m benchmarks.moe.sm120_trace_replay \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 --trace /evidence/invocations.json \
  --hot 256 --window 16 --output /evidence/native-trace
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest -q \
  tests/moe/test_sm120_canonical_fill.py::test_native_fill_sanitizer
compute-sanitizer --tool synccheck --error-exitcode 99 python -m pytest -q \
  tests/moe/test_sm120_canonical_fill.py::test_native_fill_sanitizer
```

The [ledger](expert-residency-ledger.md#routing-locality-and-canonical-fill-evidence)
records source exports, exact local commands, outcomes and limitations. The
historical [141-fixture spectrum](expert-residency-sm120-spectrum.md) and
[181-fixture corrected spectrum](expert-residency-sm120-costs.md) remain intact.
No declaration, tuning query, candidate contract or kernel is changed. Source
and artifact identity checks remain enabled. The 85-declaration/241-program SM103 compiler census remains tied to
its original source; this work makes no additional SM103 compile or performance
claim.

Validation results: **108 host tests passed, three CUDA cases skipped**; **32
tests passed in the physical SM120 suite** (11 GPU cases and 21 host cases). The native canonical-fill tests cover int32/int64
IDs, M=1/2/4 under one capacity, changed activations/routes, repeated
promotion/eviction, retained pointers, frozen resolution, no Torch allocator
events during replay, and failure after payload/publication writes. The small
native graph test passes separately under memcheck (39.73 s) and synccheck
(18.61 s), each with zero errors. Earlier full-suite sanitizer timeouts remain
incomplete gates; the bounded case does not erase or replace them. B300, Grace
TMA, concurrent replacement and complete-model caching remain unqualified.

Changed implementation and tests:

| File | Purpose |
| --- | --- |
| `b12x/testing/residency_replay.py` | Platform-neutral offline locality, causal policy replay and retrospective sensitivity calculations. |
| `scripts/analyze_expert_cache.py` | Budget/window/workload sweeps with trace integrity and source manifests. |
| `scripts/import_vllm_expert_trace.py` | Hash-checked serial nonspeculative native-export conversion. |
| `benchmarks/moe/sm120_canonical_fill.py` | Separate recoverable, generation-checked canonical-fill prototype. |
| `benchmarks/moe/sm120_residency_poc.py` | Explicit canonical backing and initial hot population for research lanes; original defaults retained. |
| `benchmarks/moe/sm120_promotion_costs.py` | Complete transaction comparison and per-copy diagnostics. |
| `benchmarks/moe/sm120_trace_replay.py` | Paired physical replay of observed routes with synthetic activations. |
| `tests/moe/test_residency_replay.py` | Censoring, held-out priors, exact count/reuse, policy, integrity and import tests. |
| `tests/moe/test_sm120_canonical_fill.py` | Fault recovery, generation/pointer invariants and native graph/sanitizer cases. |

Documentation files are `docs/expert-cache-evolution.md`,
`docs/expert-residency-ledger.md`, `docs/expert-residency-sm120-costs.md`,
`docs/expert-residency-subsystem.md`, `docs/sm103-readiness-report.md`, and
`docs/sm103-change-summary.md`. No public `b12x.moe.residency` API, prepared numerical contract or serving
default changes in this pass.
