# Movement capacity and learned-anchor recovery on SM120

Status: **research-only**, with source-built serving and portable correctness
evidence. The [anchor contract](expert-cache-anchor.md) defines the optional
comparison and bounded re-centering mechanism. Adaptive serving remains opt-in.

The evidence supports a bounded, opt-in recovery mechanism. Larger movement
allowances improve broad specialist transitions but worsen subsequent general
traffic. Comparing the current map with the learned anchor detects that return;
bounded restoration recovers part of the loss without forcing a reset on mixed
return traffic. It does not restore static performance in the short return phases.

**53 complete serving runs, 140,288 generated tokens and 30,608 promotions** pass
exact paired-output and fixed-address gates. These comprise 31 runs on immutable
inspected HEAD, 20 on frozen recovery source 03 and two on final source 04. The
excluded overlap and superseded prototype remain separate. Ordinary non-cache
smoke tokens are not included in those totals.

## Source and measurement contract

Inspection found b12x `8fb334f39c0ee64bdbe125223aa7e293bae88937`, containing master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, and companion vLLM
`3e45b530e58186046383e7294e611c2f6bf5cfb8`. Companion main is
`47ccf6c57d92f03630ebcbad3809450545825488`; its only absent change removes
inherited CI workflows. No companion source change is required.

The movement-capacity matrix and diagnostic counterfactuals use an immutable
archive of the inspected b12x HEAD. Recovery serving uses frozen source archive
03, with runtime package digest
`0c83a66db6c0451eda92a59f3aa37f735a1e18af65ea490e8e383d43ca60f198`.
Analysis and documentation completed after measurement do not relabel either
source. The source archives, exact commands and per-file manifests are retained.
The final host-only cleanup is archive 04, runtime digest
`e812e2e6aa06fd250e40375172022ff412874213fa73eb277fac12f69eb4b3f2`, archive SHA256
`f024a2453ef7136e492821ba82c2bed7f8cf3f6393573192e15fb3370e16d98d`.
It removes unnecessary anchor rechecks from ordinary adaptation; its separate
follow-up results do not replace the source-03 matrix.

Archive 03 SHA256 is
`89a47fe28b1e70223019c597f17c67856d0e75563f4ee3ed5d1b7b9f0ffccff0`;
the immutable inspected-HEAD archive is
`fc4d7e0665db408eb0a6245158fe274fc5cdbe5aae14bdbffda332bccb131d00`.

The complete source-built companion wheel SHA256 is
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
All eleven installed native libraries and maintained companion Python files were
verified before serving. The wheel filename retains an earlier commit suffix;
the verified source/library manifest defines the build identity.

Hardware is `ripper`, RTX PRO 4000 Blackwell UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02, Torch 2.13.0,
CUDA 13.3 and CUTLASS DSL 4.6.2. The host negotiates **PCIe Gen4 x16 under load**.
Clocks, power, throttle state and link state are sampled every 500 ms. The host
has one NUMA domain. These are topology-specific movement measurements, not a
Gen5 ceiling or physical B300 result. Dynamic clocks and single-run cells limit
interpretation of small differences.

All serving arms use Qwen3-30B-A3B-NVFP4, checkpoint fingerprint
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`, and learned
profile `20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.
Initial placement holds 58/128 experts in each of 48 layers. Settings are 8 GiB
expert admission, 2 GiB BF16 KV, context 2048, capacity 64, FlashInfer attention,
full decode graphs, controlled group admission and 128 greedy output tokens per
request with `ignore_eos=True`. Canonical backing is cacheable pinned host memory;
retained CPU checkpoint sources and prepared fill storage keep their existing
ownership. Actual route weights, BF16 router arithmetic and whole-K W4A16 remain
unchanged. Main research arms use 16-token health probes, history depth zero,
15% absolute cold pressure, maximum interval 1,024 delivered tokens, and at most
two pairs per layer. Global budgets are 16 pairs/64 MiB or 32 pairs/128 MiB.

Raw receipts, telemetry, tokens, commands and failed attempts are retained at
`/home/jasonc/b12x-anchor-evidence-20260920`; the physical copy is on `ripper` at
`/home/jasonc/b12x-anchor-results-20260920`. Historical health/history receipts
remain immutable in their separate evidence directories. GPU serving and final
qualification share a file lock; no two accepted timing arms overlap GPU work.

## Routing counterfactual before implementation

The retained compact health records do not contain individual counts for the
healthy return tail. A source-built fixed-maintenance diagnostic on immutable
HEAD therefore records complete counters across the existing mixed-return
fixture. The analyzer reconstructs live placement from completed movements and
verifies every window's cold count before comparing it with the learned map.

During 25 return-to-general windows, the live map produces **34,889 cold
selections out of 382,464 (9.12%)**. The learned anchor would produce **20,036
(5.24%)** on those same selections. It is better in **38–48 of 48 layers** per
window. Most windows remain below the absolute 15% health threshold. The learned
map is therefore a useful reference that absolute cold pressure misses.

The reference does not uniformly favor restoration. In the observed specialist
intervals, current/anchor cold fractions are approximately **20.06%/37.52% for
code**, **28.07%/45.13% for math**, and **20.13%/33.18% for multilingual**. The
adapted map covers more specialist routes. During the mixed interval, current
and anchor cold counts are 173,642 and 234,994 over 761,856 selections. This
supports an optional recovery experiment rather than an unconditional reset.

Global advantage and breadth are retained separately. The experiment uses an
explicit two-percentage-point advantage and at least 75% of layers favoring the
anchor. These settings express a broad return hypothesis; the corpus does not
establish uniquely optimal thresholds or prove that breadth is necessary.

## Offline return replay

The replay starts from the actual pre-return map and policy state. Earlier
recorded decisions are reproduced exactly. Identical subsequent count windows
are consumed by each alternative. This is a routing simulation: model outputs,
request timing and GPU costs are not simulated. The hard reset knows the regime
boundary and is only a diagnostic control.

| Return alternative | Cold selections | Movements | Admitted copy bytes | Final anchor overlap |
|---|---:|---:|---:|---:|
| Recorded pressure-gated policy | 34,889 | 16 | 42,492,032 | 81.25% |
| Freeze adapted map | 37,591 | 0 | 0 | 80.93% |
| Hard anchor restore | 21,200 | 531 | 1,409,487,000 | 100% |
| Bounded anchor, 16 pairs | 28,076 | 128 | 339,964,928 | 85.52% |
| Bounded anchor, 32 pairs | 26,202 | 160 | 424,928,512 | 86.67% |
| Normal LFU at every retained window | 22,625 | 400 | 1,062,413,440 | 88.15% |

The hard restore pays its 531 moves after the first return observation, so its
cold total is not the all-window anchor counterfactual. Bounded re-centering
keeps the existing candidate/victim scores and guards; it does not force complete
restoration. More normal maintenance can reduce misses further, but incurs more
movement and control. No timing result is inferred from this table.

## Serving experiments

The capacity sweep completes before choosing recovery behavior: it changes only
the global movement allowance on immutable inspected source. The recovery matrix
then compares normal adaptation and anchor recovery at a fixed 16-pair allowance.
A separate mixed-return experiment combines recovery with 32 pairs. It does not
replace either isolated comparison. C4 capacity arms reverse the C1/C8 arm order;
all comparisons use exact paired outputs and the same learned initial placement.

## Movement capacity on clean transitions

The paired matrix changes only the global pair/copy-byte allowance. Higher tok/s is better. These arms use immutable inspected source, 16-token probes and no history or anchor recovery.

| Transition | C | Static overall | 16-pair overall | 32-pair overall | 16-pair specialist | 32-pair specialist |
|---|---:|---:|---:|---:|---:|---:|
| chat → code | 1 | 55.03 | 71.55 | 75.38 | 59.91 | 65.39 |
| chat → code | 4 | 90.42 | 108.37 | 115.90 | 81.95 | 90.96 |
| chat → code | 8 | 130.98 | 143.53 | 151.65 | 101.11 | 109.05 |
| prose → math | 1 | 47.06 | 57.13 | 62.43 | 42.00 | 47.85 |
| prose → math | 4 | 66.40 | 77.01 | 84.80 | 54.90 | 62.96 |
| prose → math | 8 | 83.83 | 89.38 | 95.81 | 61.62 | 68.06 |
| english → multilingual | 1 | 57.86 | 68.70 | 71.06 | 57.54 | 60.91 |
| english → multilingual | 4 | 87.76 | 103.20 | 110.98 | 76.09 | 84.82 |
| english → multilingual | 8 | 119.08 | 130.87 | 139.96 | 94.17 | 103.59 |

Stable throughput is measured separately:

| Transition | C | Static stable | 16-pair stable | 32-pair stable |
|---|---:|---:|---:|---:|
| chat → code | 1 | 90.31 | 89.19 | 89.34 |
| chat → code | 4 | 162.84 | 161.90 | 161.79 |
| chat → code | 8 | 253.17 | 251.86 | 251.88 |
| prose → math | 1 | 90.42 | 89.96 | 90.16 |
| prose → math | 4 | 125.04 | 129.73 | 131.20 |
| prose → math | 8 | 164.60 | 164.82 | 164.89 |
| english → multilingual | 1 | 85.32 | 85.55 | 85.83 |
| english → multilingual | 4 | 162.04 | 161.74 | 161.70 |
| english → multilingual | 8 | 218.42 | 216.69 | 217.77 |

Movement and control cost remain separate from throughput. Copy bytes are admitted payload/map bytes; blocked scheduling includes the engine drain and is not an additional cost to add to end-to-end wall time.

| Transition / C | Promotions 16 / 32 | Epochs 16 / 32 | Copy MiB 16 / 32 | Apply ms 16 / 32 | Blocked ms 16 / 32 |
|---|---:|---:|---:|---:|---:|
| chat → code / 1 | 448 / 576 | 28 / 18 | 1134.8 / 1458.9 | 200.2 / 220.8 | 1241.0 / 908.9 |
| chat → code / 4 | 368 / 512 | 23 / 16 | 932.1 / 1296.8 | 158.7 / 195.5 | 1766.9 / 1329.5 |
| chat → code / 8 | 304 / 384 | 19 / 12 | 770.0 / 972.6 | 133.4 / 151.8 | 1878.0 / 1259.4 |
| prose → math / 1 | 832 / 1408 | 52 / 44 | 2107.4 / 3566.2 | 356.9 / 549.2 | 2512.9 / 2330.0 |
| prose → math / 4 | 592 / 1088 | 37 / 34 | 1499.5 / 2755.7 | 253.5 / 434.6 | 3354.4 / 3113.6 |
| prose → math / 8 | 352 / 704 | 22 / 22 | 891.6 / 1783.1 | 159.4 / 281.2 | 3084.0 / 2923.5 |
| english → multilingual / 1 | 528 / 832 | 33 / 26 | 1337.4 / 2107.3 | 228.2 / 321.8 | 1466.4 / 1344.3 |
| english → multilingual / 4 | 400 / 704 | 25 / 22 | 1013.2 / 1783.1 | 171.4 / 280.0 | 1966.1 / 1746.5 |
| english → multilingual / 8 | 304 / 384 | 19 / 12 | 770.0 / 972.6 | 131.0 / 149.4 | 1956.2 / 1295.4 |

All **464 promotion epochs** in the completed clean-transition sweep saturate
their pair cap; every copy-byte envelope still has slack. Thus this experiment primarily
measures pair capacity, not a byte-limited transport. The separate full-policy
diagnostic on math C8/32 pairs records 96 proposals per epoch at the median,
1,408 total budget skips, and a median 41 layers with at least one skipped
proposal. Existing per-layer limits remain two pairs. Proposal counts therefore
describe bounded layer proposals, not every potentially useful expert.

The math C8 arms both perform 22 epochs. Doubling selected pairs increases apply
work from 159.4 to 281.2 ms but improves specialist throughput from 61.62 to
68.06 tok/s. Other cells often need fewer maintenance boundaries with 32 pairs.
Blocked scheduling includes the drain of useful work already submitted; it is
not synonymous with b12x computation and must not be added again to serving wall
time. All stage distributions remain in the receipts.

The nominal stable prose interval at C4 is not a pure observation-overhead arm:
it triggers six 16-pair or four 32-pair maintenance operations before math begins.
Those moves explain why its adaptive stable rate can exceed static. Do not
interpret that cell as negative instrumentation cost.

## Return and partial-return serving

All return comparisons use frozen source 03 at C4, the same initial learned profile and controlled admission. Recovery retains the 16-pair budget. Higher tok/s is better.

| Fixture | Arm | Overall | Initial general | Specialist/mixed | Return | Promotions |
|---|---|---:|---:|---:|---:|---:|
| history mixed return | Static | 85.21 | 162.55 | 56.59 | 185.65 | 0 |
| history mixed return | p16 | 92.38 | 161.93 | 67.31 | 136.87 | 1024 |
| history mixed return | p32 | 94.78 | 162.01 | 70.91 | 128.88 | 1824 |
| history mixed return | recenter | 93.01 | 161.87 | 66.86 | 146.74 | 1152 |
| anchor general code general | Static | 99.66 | 165.33 | 55.95 | 163.54 | 0 |
| anchor general code general | p16 | 106.89 | 164.55 | 72.07 | 124.28 | 448 |
| anchor general code general | p32 | 111.28 | 164.51 | 81.93 | 116.23 | 704 |
| anchor general code general | recenter | 109.15 | 164.62 | 71.73 | 135.76 | 496 |
| anchor general math general | Static | 93.70 | 165.21 | 50.55 | 163.49 | 0 |
| anchor general math general | p16 | 95.96 | 164.54 | 62.73 | 109.38 | 672 |
| anchor general math general | p32 | 100.65 | 164.43 | 71.31 | 103.84 | 1216 |
| anchor general math general | recenter | 98.99 | 164.57 | 62.50 | 123.10 | 752 |
| anchor general code partial | Static | 78.07 | 165.39 | 55.90 | 69.32 | 0 |
| anchor general code partial | p16 | 93.77 | 164.59 | 72.27 | 83.24 | 752 |
| anchor general code partial | p32 | 99.69 | 164.56 | 81.96 | 85.16 | 1152 |
| anchor general code partial | recenter | 93.26 | 164.67 | 71.68 | 82.79 | 752 |

Recovery measurements retain declined requests and incomplete restoration:

| Fixture | First anchor indication ms | First restore ms | Restored pairs | Declined checks | Overlap before → after return | Reached 90% / 95% |
|---|---:|---:|---:|---:|---:|---|
| history mixed return | 329.3 | 433.2 | 160 | 4 | 79.09% → 84.84% | — / — |
| anchor general code general | 534.2 | 842.5 | 112 | 7 | 86.42% → 90.45% | 7581.1 / — |
| anchor general math general | 495.0 | 602.5 | 224 | 4 | 82.26% → 90.19% | 5702.7 / — |
| anchor general code partial | — | — | 0 | 0 | 86.42% → 82.22% | — / — |

The larger normal budget improves every specialist/mixed interval here, but
makes all three clean general returns worse. For example, the mixed-return
fixture falls from 136.87 to 128.88 tok/s as the allowance doubles. The learned
static arm reaches 185.65 tok/s. This is the recovery problem, not a fill failure.

Bounded recovery improves the three clean returns to 146.74, 135.76 and
123.10 tok/s respectively, while retaining the preceding specialist benefit.
It does **not** regain static return performance within these short regimes.
The existing score margin, demand threshold and residency guards still apply;
returning to 100% anchor overlap is not an objective of the implementation.

Partial-return traffic does not trigger re-centering at all. The recovery arm
makes the same 752 normal promotions as its 16-pair control. Its return rate is
82.79 versus 83.24 tok/s, a small negative result retained rather than hidden.
The 32-pair normal arm reaches 85.16 tok/s on this mixed return. The anchor is a
reference for measured demand, not a mandate to evict all specialist experts.

### Combined recovery and movement, and churn

The separate mixed-return 32-pair recovery arm reaches **96.23 overall,
70.52 mixed and 143.52 return tok/s**, versus 94.78/70.91/128.88 for normal
32-pair adaptation. It performs 2,048 promotions, including 288 return-phase
restores. Return overlap rises from 74.17% to 84.52%; neither 90% nor 95% is
reached. Its first anchor indication arrives at 570 ms and first completed
restore at 694 ms. Four re-centering checks decline movement.

| Fixture | Normal 16: promotions / re-promotions / zero-hit evicted lifetimes | Normal 32 | Recovery 16 |
|---|---:|---:|---:|
| Mixed return | 1,024 / 65 / 13 | 1,824 / 209 / 32 | 1,152 / 73 / 14 |
| General → code → general | 448 / 0 / 2 | 704 / 3 / 8 | 496 / 0 / 2 |
| General → math → general | 672 / 1 / 10 | 1,216 / 9 / 28 | 752 / 0 / 12 |
| General → code → partial return | 752 / 13 / 7 | 1,152 / 39 / 16 | 752 / 13 / 7 |

A zero-hit lifetime is counted only after eviction. Still-resident promotions
are right-censored. The raw analysis also retains per-expert eviction counts,
movement bytes, hits and overlap trajectories. Larger capacity buys useful
specialist coverage but also causes more churn; no pressure-sensitive budget or
new default follows from this matrix.

Relative to normal adaptation, recovery repays its earlier-interval penalty in
the retained mixed fixture after **729 return tokens / 5.31 s at 16 pairs**,
and **481 tokens / 3.77 s at 32 pairs**. Final gains are 293 and 651 ms. These
are retrospective matched-delivery curves including prior interval cost, not
universal dwell thresholds. The return interval alone remains much slower than
static; the complete sequence remains faster because of the specialist interval.

| Mixed-return C4 arm | TTFT p50 ms | Delivery gap p50 / p95 / p99 ms | Epochs | Epoch p50 / p95 ms | Blocked total ms | Copy MiB |
|---|---:|---:|---:|---:|---:|---:|
| Static | 316.70 | 38.60 / 86.45 / 98.61 | 0 | — | 0.0 | 0.0 |
| Normal 16 | 314.35 | 36.07 / 82.37 / 104.49 | 64 | 79.69 / 111.31 | 5199.6 | 2593.8 |
| Normal 32 | 319.61 | 35.41 / 82.41 / 108.62 | 57 | 86.03 / 119.96 | 5071.8 | 4619.9 |
| Recovery 16 | 316.81 | 36.06 / 83.78 / 106.88 | 76 | 80.59 / 114.11 | 6088.5 | 2918.0 |
| Recovery 32 | 319.16 | 34.68 / 83.89 / 109.86 | 68 | 86.36 / 120.38 | 5855.3 | 5187.2 |

Delivery gaps are client-visible ITL diagnostics, not CUDA iteration timings.
The tail cost remains visible: recovery improves total delivery time without
improving every latency percentile. Per-corpus TTFT, delivery distributions,
request decode rates and nested maintenance stages are retained in the JSON
analysis. The blocked interval includes worker drain and cannot be interpreted
as copy time.

### Observation cost and trigger meaning

An anchor-observation-only mixed-return control reaches 92.31 overall and
136.87 return tok/s, versus 92.38 and 136.87 without the mask. Both perform
1,024 identical-count promotions. This small difference is below what a single
run with dynamic clocks can establish as a stable performance change.

In these serving diagnostics the prepared health kernel median changes from
4.096 to 6.112 microseconds and its compact copy from 4.480 to 5.536 microseconds.
The payload grows from 2,304 to 2,688 bytes. The anchor adds 6,912 device bytes
and 384 pinned bytes at this geometry. It adds no per-route launch or atomic.
These are measured probe stages, not complete RPC latency or a claim of zero cost.

Short probes and full policy windows can disagree. Recovery therefore records
both assessments, explicit declined checks, and `anchor_advantage`, `pressure`
and `maximum_interval` triggers. For the mixed 32-pair recovery run, one maximum
interval and 54 pressure triggers drive the mixed phase; 13 anchor triggers drive
the return phase. Do not attribute its first movement entirely to pressure latency.


### Cadence control and final host cleanup

The retained math C8/32-pair control uses 32-token probes: **92.58 overall and
64.58 math tok/s**, with 512 promotions. At 16 tokens the same budget reaches
95.81/68.06 with 704 promotions. The 32-token arm has 16 pressure-triggered epochs
and no maximum-interval trigger; first pressure is detected at 1.66 s and first
maintenance completes at 1.95 s. This supports 16 tokens as the research cadence
for this fixture, not as a production default or a universal optimum.

Source 03 needlessly recomputed the full host anchor comparison during normal
adaptation whenever recovery thresholds were configured. Frozen source 04 limits
that check to explicit re-centering requests. No GPU program or policy arithmetic
changes. The mixed-return repeat retains 1,152 promotions and exact outputs:
**93.32 overall, 67.19 mixed and 146.65 return tok/s**. Normal-adaptation pressure
checking falls from 6.25 to 2.63 ms median; re-centering validation remains about
6 ms. This is a measured host-stage reduction, not a newly qualified whole-matrix
speedup. Its partial-return repeat reaches 93.70 overall and 83.31 return tok/s,
with 752 promotions and no re-centering. Both source-04 repeats retain exactly
the source-03 transaction sequence, as well as paired token equality. The main
source-03 results remain unchanged above.

## Correctness and compilation

Frozen runtime source passes the following focused gates:

| Gate | Result | Receipt |
|---|---|---|
| Host policy, admission, integration and SM103 declaration contracts | 331 passed, 2 CUDA-only skips in 5.63 s | `host-compiler-final-03.log` |
| Physical prepared counters and cache execution | 33 passed in 58.67 s | `gpu-qualified-03.log` |
| Anchor memcheck | 3 passed in 31.39 s; zero errors | `memcheck-qualified-03.log` |
| Anchor synccheck | 3 passed in 13.86 s; zero errors | `synccheck-qualified-03.log` |
| Source-built companion loader/maintenance tests | 10 passed | `companion-qualified-03.log` |
| Ordinary non-cache graph serving | 512 tokens; five loaded native libraries match the verified wheel | `ordinary-qualified-03.json` |
| SM103 health and anchor declarations | Both cases cross-compile four required programs without a CUDA context | `sm103-final-03/manifest.json` |
| Complete SM103 metadata census | 88 declarations, 246 distinct programs | `census.json` |

Final runtime source 04 independently passes 331 host/compiler tests with two
CUDA-only skips in 5.99 s, 33 GPU tests in 58.72 s, three memcheck cases in
31.70 s and three synccheck cases in 13.72 s, both with zero errors. Ten companion
tests pass in 1.33 s. Its ordinary non-cache graph smoke generates 512 tokens
and verifies five loaded native-library hashes against the source-built wheel.
Its separate SM103 health/anchor compile receipt and metadata
census retain 88 declarations/246 programs. Source 03 and source 04 differ in one
runtime file: the conditional host anchor check in `residency_maintenance.py`.
All runtime files in the final worktree match archive 04. Two subsequent test
formatting changes preserve the test AST; they do not alter the measured runtime.

Tests cover current/anchor equality and disagreement, broad versus one-layer
advantage, empty windows, invalid and duplicated routes, nonuniform expert
counts, profile mismatch, generation rebasing, counter reset/overflow, immutable
anchor identity, static opt-out, score preservation, later normal adaptation,
budget rejection reasons and declined-recovery behavior. Physical tests retain
the same captured graph with stable pointers and no replay allocations. Existing
transaction failure, stale generation, cancellation and recovery-poisoning
checks remain in the host/backend suites.

The two host skips are the existing CUDA rich-trace collector tests on the
development host, not silently accepted physical anchor cases. Ruff's broader changed-file check retains the pre-existing B905 diagnostic and,
with that excluded, twelve existing F401/E701 diagnostics. A comparison against
HEAD finds no newly introduced diagnostics after two test-formatting cleanups.
The runtime package is unchanged by those formatting edits. `git diff --check`
passes. GitHub reports zero check runs and zero commit statuses for the
inspected branch source; local evidence remains independent of CI acceptance.

## Retained failures and limits

An initial GPU-qualification wait used `rg`, which is not installed on `ripper`.
It failed to wait and overlapped the immutable chat→code C4/16-pair timing arm.
That original receipt is retained and excluded from performance analysis. Its
isolated replacement uses the common GPU lock. The initial GPU correctness
results are supplemented by isolated final qualification.

The first recovery prototype could fall back to normal adaptation when a short
anchor probe triggered but the longer policy window declined restoration. Its
outputs and pointers passed, but that ambiguous control behavior is excluded
from recovery conclusions. Frozen source 03 carries an explicit movement intent;
a declined re-centering operation cannot substitute a normal proposal.

An initial fixture copy used shortened filenames that did not exist. An offline
simulation invocation lacked `PYTHONPATH`; the failure ledger records these
command-transcript errors separately from the retained successful output. An
initial metadata census lacked the offline compiler identity and failed
without a CUDA device; the corrected census uses the repository's compiler-worker
initialization. A queued qualification script with a wrong test filename was
cancelled before execution and replaced. Existing ignored engine-destructor
warnings remain in raw shutdown logs. No failure receipt is deleted.

The anchor remains one validated learned profile, not a workload classifier or
an optimal cache. Recovery decisions can be declined, and restored experts can
be evicted by subsequent normal adaptation. Short return regimes need not repay
extra control. Full restoration, distributed recovery serving, multi-profile
selection and asynchronous replacement are not implemented by this experiment.

SM103 evidence is limited to metadata and cross-compilation. Physical acceptance
still starts with all-HBM, all-Grace and mixed native correctness, Grace TMA
legality, same-graph updates and sanitizers before miss-service and adaptive
serving measurements.

## Engineering disposition

The reusable additions are immutable anchor identity, an optional prepared health
mask, an explicit bounded re-centering intent, and model-wide backlog diagnostics.
The serving harness owns experimental trigger settings. vLLM's existing scheduler
boundary requires no change. None of these options becomes a serving default.

The strongest next measurement is longer clean and partial returns: current
recovery improves throughput but usually stops below 90% anchor overlap in the
short fixtures. That experiment should distinguish score/guard limitations from
regimes too short to justify further restoration. A larger restoration budget
must be tested separately from larger specialist-adaptation budgets. Repeated,
alternating-order trials should precede claims about sub-percent overhead.

Deeper history, a variable movement budget, multiple learned profiles and a
completion-output health hook are deferred. This evidence does not require them.
Numerical scheduling, expert transports and asynchronous replacement remain
unchanged. Policy and counterfactual conclusions are portable concepts; measured
copy/apply cost and end-to-end rates remain specific to this PCIe Gen4 x16 host.
A Gen5 comparison must retain the same checkpoint, profile, routes, movements,
budgets, host-memory mode and admission settings and measure its own results.
