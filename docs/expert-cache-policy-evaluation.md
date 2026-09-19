# Held-out routing and SM120 cache policy evaluation

Status: **research-only**. The offline LRU, LFU and decayed-LFU comparisons do
not add serving policies. Static residency, the shared recent-frequency
controller, SM103 preparation and numerical kernels are unchanged. Physical
measurements use native SM120 operators with real checkpoint weights and routes,
but synthetic activations and uniform `1/top_k` route weights. Native export
provides expert IDs, not the original gate weights. These are not
serving-throughput or full-model numerical measurements.

Decayed LFU with a four-invocation decision window reduces the measured
operator-plus-fill total by **25.8–35.3%** on the three selected layers. LFU with
a 16-invocation window improves by **8.4–22.3%**; the layer-24 improvement repeats
on both GPUs. These results support further evaluation of policies that retain
recent history. They do not justify enabling adaptive serving by default.

## Experiment contract

The source baseline is `bc5a4316cbe2d41beeceab8a605d8a84737533db`, based on master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. The earlier
[locality and canonical-fill report](expert-cache-evolution.md) remains a
separate, immutable comparison. Its short requests motivated a longer held-out
corpus and physical tests beyond layer zero.

The protocol fixes these choices before collecting the longer routes:

- Four independent requests per class: prose, code, math, multilingual, agent
  planning and general chat. The first two train the initial placement; the
  other two are held out. Requests allow natural EOS and at most 1,024 generated
  tokens. Prompts request detailed answers so that long reuse horizons can be
  observed without forcing generation past EOS.
- Offline evaluation covers all 48 layers, resident counts 128/256/384,
  positional and learned starts, and decision windows 4/16. Static,
  recent-frequency, LRU, LFU and decayed LFU use the existing offline semantics.
  No thresholds are fitted to held-out requests.
- Physical replay covers layers 12, 24 and 47, with LFU at window 16 and decayed
  LFU at window 4. A layer-24 recent-frequency/window-16 run anchors the shared
  controller. Each adaptive arm has its own paired learned-static control.
- Physical evaluation uses the first held-out prose request followed by the
  first held-out code request. All twelve training requests determine the
  initial 256-expert resident population. A fill occurs after the observed
  invocation and affects subsequent invocations only.

Before physical timing, the protocol also fixes a matched backing-store control:
both static and adaptive arms retain a canonical row for every expert. The
exclusive static tier has fewer backing rows and would otherwise confound
policy effects with cold-kernel geometry. A layer-24 LFU/window-16 repeat on the
second GPU is specified before physical results are available.

The authored prompts are not production logs. Agent requests contain
hypothetical investigation plans, not actual tool interactions. Native vLLM
routed-expert export runs on the four-Spark TP4 lane with speculation disabled
and concurrency one. Each canonical route is counted once. Prompt rows and the
unprocessed final output token are excluded. These exports cannot reconstruct
verifier work or concurrent engine batches.

The offline schedule deliberately groups held-out requests by workload in the
order prose, code, math, multilingual, agent and chat. It differs from capture
order and exposes explicit workload shifts. Cache contents carry across request
boundaries. Within-request future-reuse statistics censor those boundaries and
exclude horizons extending beyond observed data. Training counts never enter
evaluation statistics.
The mixed-class training prior is an explicit experiment, not an automatic
merge of workload-specific serving profiles. Comparisons between LFU/window-16
and decayed-LFU/window-4 describe complete policy/window combinations. The
offline sweep includes both windows for each policy to expose that distinction.

The corpus has 12,276 training and 11,447 held-out decode invocations per layer:
5,892,480 training and 5,494,560 held-out selections across 48 layers. Twenty-three
responses reach the 1,024-token limit. One held-out code response stops at 195
tokens after stating an intention to inspect a project; it does not complete the
requested implementation. That response is retained, and long reuse horizons
are censored. This corpus measures observed routing, not answer quality.

## Complete offline sweep

All 2,592 policy fixtures and 48 locality analyses complete. Learned placement
provides a large initial benefit independently of adaptation:

| Resident experts per layer | Positional static cold selections | Learned static cold selections |
| --- | ---: | ---: |
| 128 | 74.36% | 46.99% |
| 256 | 48.21% | 20.14% |
| 384 | 23.59% | 5.73% |

At 256 resident experts, the complete six-class held-out schedule gives:

| Policy | Window | Cold selections | Promotions | Completed zero-hit lifetimes / completed lifetimes | Layers with fewer cold selections than static |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned static | — | 20.14% | 0 | — | — |
| Recent-frequency | 4 | 24.63% | 105,605 | 39,656 / 100,254 | 9 / 48 |
| Recent-frequency | 16 | 21.91% | 33,752 | 8,164 / 29,102 | 12 / 48 |
| LFU | 4 | 16.32% | 15,499 | 736 / 8,694 | 48 / 48 |
| LFU | 16 | 17.40% | 14,571 | 776 / 7,993 | 46 / 48 |
| Decayed LFU | 4 | 8.29% | 36,740 | 1,468 / 26,731 | 48 / 48 |
| Decayed LFU | 16 | 12.35% | 27,214 | 970 / 18,133 | 48 / 48 |
| LRU | 4 | 8.38% | 37,350 | 1,571 / 27,292 | 48 / 48 |
| LRU | 16 | 13.44% | 27,623 | 1,036 / 18,484 | 48 / 48 |

LRU and decayed LFU have similar offline results. Only decayed LFU is physically
measured here; the small miss-rate difference does not establish a general
winner between them. Each policy permits at most one promotion per decision,
requires two cold observations, and uses the existing residence guard. Frequency
policies require a score gain of two; LRU requires a more recent observation.
Decayed LFU halves its history each decision window. These are experimental
settings, not recommended deployment defaults.

Conditioned on a cold touch under learned static placement, the probability of
reuse within 4/16/128/512 invocations is **48.59% / 64.90% / 87.67% / 96.01%**.
The 512-step estimate has 501,326 eligible cold touches and averages 47.38 later
route selections per eligible touch. It excludes request tails without a full
horizon. A long-horizon reuse probability alone does not imply profitable
promotion before eviction.

## Workload shifts

At 256 resident experts, the complete held-out schedule gives these selection
counts. Lower cold counts are better; this table contains no GPU timing.

| Layer | Learned static | Recent-frequency, window 16 | LFU, window 16 | Decayed LFU, window 4 | LRU, window 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 21,337 | 26,423 | 21,103 | 10,330 | 10,489 |
| 24 | 22,112 | 25,775 | 18,997 | 8,392 | 8,514 |
| 47 | 14,697 | 19,865 | 14,003 | 7,531 | 7,521 |

The existing recent-frequency controller increases misses on all three layers
under this schedule. Cumulative LFU improves the total but can retain experts
from an earlier workload too long. For layer 24, LFU reduces prose cold
selections from 18.51% to 5.13%, but code cold selections rise from 25.41% to
26.22%. Its first 128 code invocations contain 444 cold selections versus 212
under static placement. Decayed LFU also has a transition spike, at 326 cold
selections, but reduces the complete code interval to 13.85%.

Across the full layer-24 schedule, recent-frequency makes 714 promotions;
cumulative LFU makes 308; decayed LFU makes 788. Their completed zero-hit
promotion counts are 173, 19 and 40, respectively. These counts exclude
unfinished lifetimes from the zero-hit failure classification. Reduced churn
alone does not identify the best policy: the question is whether useful reuse
pays for each completed transaction, including harm caused by evicting a victim.

## Physical policy replay

Each row below covers 2,046 invocations: one held-out prose request followed by
one held-out code request. Times are sums in milliseconds. The adaptive total
is graph time plus complete canonical-fill wall time. The ratio is
**adaptive/static; lower is better**. Every row has its own alternating static
control with identical canonical backing geometry.

| Layer | Policy / window | GPU | Static total ms | Adaptive graph ms | Fill wall ms | Adaptive/static |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 12 | LFU / 16 | 1 | 674.142 | 576.735 | 40.979 | 0.9163 |
| 12 | Decayed LFU / 4 | 1 | 677.413 | 414.413 | 50.375 | 0.6861 |
| 24 | LFU / 16 | 1 | 736.242 | 544.486 | 36.376 | 0.7890 |
| 24 | Decayed LFU / 4 | 1 | 742.226 | 421.255 | 58.903 | 0.6469 |
| 47 | LFU / 16 | 1 | 485.570 | 350.269 | 26.928 | 0.7768 |
| 47 | Decayed LFU / 4 | 1 | 487.851 | 332.252 | 29.491 | 0.7415 |
| 24 | Recent-frequency / 16 | 1 | 737.340 | 640.125 | 41.086 | 0.9239 |
| 24 | LFU / 16 repeat | 0 | 738.555 | 545.719 | 42.599 | 0.7966 |

GPU 1 is `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`; GPU 0 is
`GPU-47363510-b87a-13a5-4824-2542e97df76c`. The physical trace SHA256 is
`9d9b26436f6ebb466b32ede86a1fdab8f4baf6157762b6d6b2812e4f28755b08`.
The record retains its parent trace and selected request identities.

The shorter physical schedule is deliberately distinct from the complete
six-class schedule. Recent-frequency improves the layer-24 total here by 7.6%,
even though it increases cold selections across the complete offline schedule.
LFU on layer 12 wins overall but loses during the code interval: 424.419 ms
including fills versus 376.014 ms static. Decayed LFU improves both intervals
on each selected layer. A favorable aggregate must not hide transition harm.

Decayed LFU makes 157/182/90 promotions on layers 12/24/47 and earns
5,571/6,790/2,658 subsequent resident selections, respectively. That is
35.48/37.31/29.53 later selections per promotion over this observation period.
Its completed zero-hit lifetimes are 6/41, 9/48 and 2/11; another 4/11/4
zero-hit lifetimes are right-censored. An individual promotion's causal profit
cannot be isolated from these totals because victim harm and multi-route cold
execution interact. The report measures policy-level net cost instead of
assigning every later hit a universal saving.

Complete in-loop fill medians range from 0.320 to 0.385 ms across these runs.
The fill implementation is unchanged from the canonical-fill report. These
figures are different execution conditions, not a transport optimization over
its standalone 0.538 ms median. The historical standalone and reversible-exchange
receipts remain separate.

## Reproduction and implementation

The [trace importer](../scripts/import_vllm_expert_trace.py) verifies response
hashes and emits `b12x-routing-invocations-v1`. The
[offline analyzer](../scripts/analyze_expert_cache.py) preserves per-layer
identity and training/test separation. Its comparison policies remain analysis
code; only the `b12x` arm calls the shared production host controller.

```sh
PYTHONPATH=. python scripts/analyze_expert_cache.py invocations.json \
  --output analysis --budgets 128 256 384 --windows 4 16 \
  --initial learned positional

PYTHONPATH=. python -m benchmarks.moe.sm120_trace_replay \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --trace physical-trace.json --trace-layer layer.24 \
  --prefix model.language_model.layers.24.mlp.experts \
  --hot 256 --policy lfu --window 16 --transports canonical \
  --static-backing canonical \
  --expected-fields-sha256 \
    66f3be017fa99808411e5cc35a914863e72896d03de8b99abd560ccb667dc4e7 \
  --output physical-layer24-lfu16
```

The [physical harness](../benchmarks/moe/sm120_trace_replay.py) accepts explicit
offline policy and transport selection. Defaults preserve the original
recent-frequency comparison with both reversible exchange and canonical fill.
The static control is always present and defaults to exclusive backing.
`--static-backing canonical` holds backing-row geometry fixed for policy
comparisons. An optional independently obtained
source-field fingerprint rejects a different checkpoint layer before GPU
preparation.

Every timed invocation uses the same captured graph with changed canonical route
IDs. Complete quiescent canonical-fill wall time includes drain, payload copy,
completion, map publication and transaction bookkeeping. CUDA-event graph time
plus transaction wall time is an operator diagnostic. Trace input copies,
offline policy computation, observation counters, graph preparation, validation
and engine scheduling are excluded. No fill overlaps execution.

Arms alternate order each invocation. Both receive identical seeded synthetic
activations and matching validation at decision boundaries. Validation checks
native all-VRAM parity, finite/nonzero outputs, exact ordered reduction,
stable pointers and allocation-free replay. Lifetime-wide pointer checks span
every promotion. Final physical cold counts must equal the causal offline
schedule, independent of backing-row numbering. GPU state is recorded before,
during and after replay; dynamic clocks remain a measurement limitation.

Canonical fills retain a full cacheable mapped backing copy, including hot
experts, and restore a victim from those verified bytes if overwrite fails.
This experiment does not implement a whole-model pageable/mmap miss service,
spare slots, concurrent replacement or automatic engine activation. The
additional backing-memory cost documented in the canonical-fill report applies.
The 256-expert budget is a per-layer experiment, not whole-model admission to a
24 GiB GPU. Model-wide placement must also admit every layer, dense weights,
workspace, KV storage and safety reserves. Policy computation, observation and
pauses across all layers remain unmeasured serving costs.

## Source and hardware identity

The offline and initial test archive is `source-01.tar.gz`, SHA256
`56e363514b3df7dcc4880d591d805337bf3f590979dddfa383bf86ae8b90bba3`.
The physical replay archive with the matched static-backing selector is
`source-02.tar.gz`, SHA256
`9f3ca9c044247933414cff5b9a8b5926dcb1d785367a7216fe79b518fe16c391`.
The normalized complete trace SHA256 is
`aa0ad4ad5dbccb0c1cb6e855f88b860ed30f50a592be46deb7f8bcddc3dfa7ab`.
The checkpoint export manifest remains
`8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397`.
Independent reads on the Spark capture host provide these source-field hashes
for physical replay admission:

| Layer | Native checkpoint field SHA256 |
| --- | --- |
| 12 | `c89084e9084c40220e1fcfa6dda0399c2f0706f0392c50e15d788e62bb593741` |
| 24 | `66f3be017fa99808411e5cc35a914863e72896d03de8b99abd560ccb667dc4e7` |
| 47 | `8261295ca76651891b6fed92a0e8f2712f5b4ac699c8fb8733b5ddd9e3eed3f6` |

The workstation is ripper, with two RTX PRO 4000 Blackwell GPUs, driver
580.173.02, PCIe Gen4 x16 under load and one host NUMA node. CPU affinity spans
logical CPUs 0–63. The container identity is
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`,
with Torch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2, Triton
`3.7.1+gitf797708c.nv26.7` and cuda-bindings 13.0.3. Raw topology, clocks,
power, throttle state, source hashes and command lines remain in the receipts.
No local-versus-remote NUMA comparison is possible on this host configuration.
During replay and at completion, recorded samples are P1, with a 13,365 MHz
reported memory clock, PCIe Gen4 x16 and zero active throttle flags. Recorded
SM clocks vary from 2,055 to 2,182 MHz. Pre-preparation snapshots include idle
states and a software-power-cap flag; they remain in the receipts. Clocks are
not locked, and this is diagnostic evidence rather than formal release timing.

## Validation and disposition

Raw evidence is retained at
`/home/jasonc/b12x-cache-policy-evidence-20260919/` and
`ripper:/home/jasonc/b12x-cache-policy-results-20260919/`.
The immutable `evidence.tar.gz` bundle has SHA256
`8a940b27a23fcf47cc334209ac41c339c5c1a8e51494c47940673d01c52d8e79`.
All 164 receipt files are hash-verified in the archive and administration copy.
The protocol, launchers, raw samples, workload-shift records and final audit
remain available independently of this report.

Source-01 host tests pass **53**, with **3 CUDA-only skips**. Its GPU-hosted
suite passes **41** tests, including native fill/recovery and replay checks.
Source-02 focused host schedule tests pass **9**. The final physical source
passes **8 paired comparisons**, **32,736 timed graph replays** and **964
complete promotions** on two physical GPUs. The receipt audit verifies every
recorded implementation hash against source-02 and every offline analysis hash
against the same unchanged analyzer source.

Across the physical receipts, 1,960 numerical validation calls pass. Maximum
relative L2 error against the native all-VRAM control is 0.001210 or less;
minimum cosine is 0.9999992 or greater. Ordered reduction is exact. All lifetime
pointer checks pass. The first and last replay checks in each arm total 32
allocator measurements, with zero allocation/free events. Physical cold counts
match the offline schedules exactly. No kernel or copy implementation changes,
so the preceding bounded memcheck/synccheck receipts retain their original
identity rather than being relabeled as new sanitizer runs.

The first summary attempt encountered an in-progress physical receipt before
its replay file existed. Its failure is retained; the summary skips unfinished
runs, and final acceptance separately requires all eight to pass. No measured
fixture or failed model answer is discarded. The historical 141/181-fixture
spectra and sanitizer timeouts remain intact. The original Spark lane scripts
are restored byte-for-byte on all four hosts; health and generation checks pass.

The evidence favors evaluating decayed LFU or LRU with measured control-plane
cost before extending the shared production controller. The ranked remaining
questions are:

1. Measure observation, decision and scheduler-pause costs in a real serving
   lane with model-wide memory admission. Single-layer savings exclude these
   costs and do not predict tokens per second.
2. Test held-out production traffic and concurrent/verifier invocation traces.
   LRU is close to decayed LFU offline; their relative value is unresolved.
3. Measure canonical backing/staging capacity for the complete model before
   productizing fills. The experiment retains full mapped backing and recovery
   copies for hot experts.
4. Add scheduling timestamps to quantify overlap opportunity. Invocation-order
   traces contain no GPU schedule, so they cannot justify spare slots or
   asynchronous replacement.
5. Qualify direct Grace execution and exchange on physical B300 before applying
   PCIe policy economics to SM103. Static placement remains the serving default.

## SM103 boundary

Policy observations and canonical expert IDs are portable concepts. The measured
transport and direct-host miss service are specific to discrete SM120 PCIe
hardware. No result establishes Grace-backed TMA legality, SM103 performance or
an overlap benefit. Physical B300 qualification still starts with all-HBM,
all-Grace and mixed native operator correctness before comparing cache policies.
The 85-declaration/241-program SM103 compiler census retains its original source
identity; this benchmark-only change adds no kernel or preparation contract.
