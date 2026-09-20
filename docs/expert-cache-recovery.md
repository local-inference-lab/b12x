# Recovery duration and intent-specific movement budgets

Status: **implemented; research-only**. The learned-anchor control retains one
immutable reference placement and one quiescent mutation protocol. An optional
recovery budget changes how many already-eligible pairs a re-centering operation
may admit. It does not change scores, thresholds, residency guards, per-layer
prepared capacity, health cadence or model arithmetic.

The clean C4 returns show two different limits: early recovery is constrained by
movement opportunity, while later recovery stops as the anchor's advantage
shrinks. Larger explicit recovery budgets improve both return fixtures. Late
serving can approach static with only about 91–93% set overlap; demand-weighted
coverage is substantially higher. The evidence does not support relaxing scores,
forcing complete anchor restoration or selecting a production default.
The [specialist-retention study](expert-cache-retention.md) separately tests
whether a small temporary victim guard reduces rebuilding after another
specialist shift. Its source and measurements are distinct from this budget
comparison.

## Source and experiment contract

Inspection finds b12x `9bbc1eec48badb3c792f7e6a967a4f7c140af609`, containing master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, and companion vLLM
`3e45b530e58186046383e7294e611c2f6bf5cfb8`. Companion main is
`47ccf6c57d92f03630ebcbad3809450545825488`; its absent CI-removal change affects
no serving source. The companion requires no modification.

The inspected HEAD is retained as an immutable comparison archive. The separate
budget implementation is frozen as source archive 01, runtime package digest
`d93c1f772cea923edca01de349ff7a8b3b9f1749e76a9341043e61036d7144ca` and archive
SHA256 `debf794bd56a4cc8d68467dfbc80466832ffbb5ad10b861673825736ab184d5d`.
Measurements identify the archive actually executed; later documentation and
analysis do not relabel baseline runs as modified-source qualification.

All serving uses the verified source-built companion wheel SHA256
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
The eleven installed native libraries and maintained Python files are checked
against the wheel. The wheel filename retains an earlier source suffix; the
verified file manifest defines the build identity.

The physical host is `ripper`, RTX PRO 4000 Blackwell UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02, Torch 2.13.0,
CUDA 13.3 and CUTLASS DSL 4.6.2. It uses **PCIe Gen4 x16 under load**, one NUMA
domain and dynamic clocks. Telemetry records link state, clocks, power and
throttle state every 500 ms. Serving and GPU qualification serialize through one
file lock. No other GPU job runs alongside accepted timing arms.

Checkpoint Qwen3-30B-A3B-NVFP4 has fingerprint
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`. Learned
profile `20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`
holds 58/128 experts in each of 48 layers. The experiment retains 8 GiB expert
admission, 2 GiB BF16 KV, context 2048, capacity 64, C4, FlashInfer attention,
full decode graphs, actual router weights, whole-K W4A16, unchanged BF16 router
arithmetic and greedy 128-token requests with ignored EOS. Prefix caching is off.
Controlled group admission and exact paired token equality remain mandatory.

All adaptive arms use 16-token health probes, history depth zero, 15% cold
pressure, maximum interval 1,024 delivered tokens, two pairs per layer, and
normal adaptation limited to 32 pairs/128 MiB. Anchor recovery requires at least
two percentage points of global advantage and 75% of layers favoring the anchor.
Its explicit envelopes are 16 pairs/64 MiB and 32 pairs/128 MiB. A targeted
64-pair/256-MiB code arm is admitted after the recorded 32-pair recovery shows ten
movement epochs selecting 32 out of 54–83 proposals. Math subsequently meets the
same condition: thirteen movement epochs select 32 out of 58–95 proposals.
Neither experiment reaches the byte cap. Per-layer capacity stays two.

Normal-adaptive controls enable read-only anchor health so they retain the same
counterfactual observation as recovery arms; they cannot request re-centering.
Static controls allocate no adaptive observer or anchor health state. These are
control-policy experiments, not a production-default selection.

## Workload and diagnostic scope

The fresh `expert_recovery_code.jsonl` and `expert_recovery_math.jsonl` fixtures
contain eight initial general requests, eight specialist requests and 32 return
requests: 1,024/1,024/4,096 output tokens. Both share the same general sections.
The shorter code fixture is their exact first 24 requests, ending after 1,024
return tokens. Delivered-token prefixes retain intermediate duration economics;
the separate shorter run checks whether ending the regime changes the prefix.

`expert_recovery_partial.jsonl` replaces the clean return with two general and
two specialist requests per admission group. `expert_recovery_repeat.jsonl`
adds eight fresh code requests after the long clean return. Shared prompts
between these fixtures isolate the changed regime; no text overlaps the earlier
repository evaluation fixtures. The learned calibration profile is unchanged.

Full policy diagnostics run separately from headline timings. Each missing
anchor expert receives one primary category: unobserved, below minimum count,
score margin, protected/no victim, per-layer cap, global budget, anchor gate,
or selected. The analyzer first checks score eligibility beyond the layer cap;
those later candidates are counterfactual opportunities, not rejections executed
by a loop that already stopped at its cap. Global pair/byte rejection totals
remain the coordinator's authoritative accounting.

Traffic-weighted anchor coverage is:

```text
selections to anchor experts that are currently resident
-------------------------------------------------------
selections to all experts resident in the learned anchor
```

It measures how much of the anchor's observed route coverage is retained. It is
not a latency estimate. Set overlap weights every expert equally and therefore
need not reach 100% when the useful anchor traffic is almost fully covered.
Full-count windows crossing a regime boundary are retained and identified by
their observation interval; healthy unsnapshotted tails are not invented.

Raw source archives, commands, failed attempts, build identities, token IDs,
counter trajectories and telemetry are retained in
`/home/jasonc/b12x-recovery-evidence-20260920` and
`ripper:/home/jasonc/b12x-recovery-results-20260920`. Prior anchor measurements
remain immutable in their separate evidence directory.

In the verified source-built environment, a recovery-budget arm is:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 --profile placement.json \
  --prompts benchmarks/moe/fixtures/expert_recovery_code.jsonl \
  --output code-recover32.jsonl --mode adaptive --control health \
  --tokens 128 --concurrency 4 --admission together --cache-gib 8 \
  --epoch-tokens 16 --health-max-tokens 1024 --history-depth 0 \
  --cold-threshold 0.15 --anchor-advantage 0.02 --anchor-breadth 0.75 \
  --epoch-pairs 32 --epoch-mib 128 --recenter-pairs 32 --recenter-mib 128
```

The raw launcher records the container, mounts, environment and telemetry command.
Full `--policy-diagnostics` runs are separate from headline timing arms.

## Clean-return measurements

Rates are generated output tokens/s for complete C4 intervals, including control
cost. All adaptive arms use the same 32-pair specialist allowance. The initial
2,048-token transaction sequences match exactly across recovery budgets, so the
return phase starts from the same adapted placement. These are single paired
runs on the recorded fixtures and dynamic clocks, not a universal budget ranking.

| Code fixture | Complete sequence | Specialist | Return general | Return / static |
| --- | ---: | ---: | ---: | ---: |
| Learned static | 128.68 | 57.04 | 171.17 | 100% |
| Adapt32, no recovery | 126.71 | 84.24 | 134.57 | 78.6% |
| Adapt32 / recover16 | 136.81 | 84.24 | 152.53 | 89.1% |
| Adapt32 / recover32 | 138.91 | 84.23 | 156.53 | 91.4% |
| Adapt32 / recover64 | 142.38 | 84.21 | 163.24 | 95.4% |

| Math fixture | Complete sequence | Specialist | Return general | Return / static |
| --- | ---: | ---: | ---: | ---: |
| Learned static | 125.51 | 53.55 | 170.98 | 100% |
| Adapt32, no recovery | 118.59 | 69.75 | 131.14 | 76.7% |
| Adapt32 / recover16 | 124.53 | 69.73 | 142.45 | 83.3% |
| Adapt32 / recover32 | 128.59 | 69.69 | 150.70 | 88.1% |
| Adapt32 / recover64 | 130.41 | 69.76 | 154.35 | 90.3% |

Normal adaptation loses to static over both longer complete sequences. Recover16
still loses slightly overall on math. Those negative outcomes are retained.
Recovery64 is the best measured complete-sequence choice in both clean fixtures,
but neither return average equals static. Math recovery reaches 90% overlap after
14.38/11.53/3.59 seconds at budgets 16/32/64. Final overlap is 91.02/91.49/92.56%;
none reaches 95%. Its last four recover64/static block ratios are approximately
0.978, 0.979, 0.982 and 1.021. A low whole-return average therefore does not imply
that the final placement remains equally bad.

Larger recovery batches also avoid repeated control work. Code recovery alone
uses 28/21/11 checks, of which 18/10/6 move experts, for budgets 16/32/64.
Scheduler-blocked time is 1.636/1.210/0.656 seconds. Committed API copy volume
rises from 765 to 850 to 929 MB, while aggregate apply time falls from 126 to
121 to 104 ms alongside the reduction in transaction count. These nested timings
must not be added to scheduler time. Two maximum-interval normal adaptations
also occur in each 16/32 return; the 64 arm has two and one pressure-triggered
normal adaptation. They are included in interval throughput and retained as
separate movement intents.

At 32, every useful code recovery epoch selects 32 pairs from 54–83 proposals.
At 64, the six selected counts are 64, 64, 44, 64, 57 and 57. Three operations
therefore admit every layer proposal. Raising the model-wide cap further would
not help those operations. The per-layer limit stays two; no byte cap binds.

Across each complete 6,144-token clean fixture, movement accounting is:

| Fixture / recovery | Promotions / 1,000 tokens | Re-promotions | Completed zero-hit lifetimes | Active, right-censored | API copy GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code / none | 140.62 | 10 | 22 | 627 | 2.137 |
| Code / 16 | 151.04 | 10 | 10 | 575 | 2.295 |
| Code / 32 | 156.25 | 12 | 10 | 573 | 2.374 |
| Code / 64 | 166.34 | 18 | 10 | 581 | 2.528 |
| Math / none | 187.50 | 16 | 18 | 798 | 2.849 |
| Math / 16 | 200.52 | 19 | 22 | 734 | 3.047 |
| Math / 32 | 203.12 | 23 | 23 | 730 | 3.087 |
| Math / 64 | 207.03 | 20 | 25 | 733 | 3.146 |

The largest recovery budget increases movement and math's completed zero-hit
lifetimes. Per-expert eviction counts are retained; their maximum is two in these
clean fixtures. Unfinished lifetimes are not counted as wasted promotions.
These bytes are transaction API accounting, not isolated PCIe traffic or line
bandwidth. Timing remains specific to the measured Gen4 x16 host.

## Recovery payback

Payback below compares recovery against normal adapt32 at matched delivered-token
counts, including any wall-time difference accumulated before return admission.
The durable crossing is the first lead that survives the remainder of this finite
fixture. It is not a forecast for a longer or different regime.

| Return | Recovery cap | Durable lead after return admission | Return tokens | Final lead over normal |
| --- | ---: | ---: | ---: | ---: |
| Code | 16 | 2.75 s | 309 | 3.59 s |
| Code | 32 | 1.89 s | 201 | 4.27 s |
| Code | 64 | 1.33 s | 133 | 5.34 s |
| Math | 16 | 8.38 s | 969 | 2.48 s |
| Math | 32 | 3.37 s | 317 | 4.04 s |
| Math | 64 | 1.10 s | 69 | 4.70 s |

These returns last roughly 24–31 seconds. None recovers the cumulative return-phase
lead of learned static within 4,096 tokens. Code still trails static by
2.92/2.24/1.16 seconds and math by 4.80/3.22/2.58 seconds at caps 16/32/64, even
though later block rates are much closer. Complete-sequence gains also include
the preceding specialist benefit; they must not be described as complete return
recovery. No minute- or hour-scale behavior is inferred.

Observed return prefixes make the duration effect explicit. The following rates
are prefixes of the same running experiment, not independently restarted arms:

| Return tokens | Code static | Code recover32 | Math static | Math recover32 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 185.89 | 117.03 | 185.32 | 88.31 |
| 1,024 | 191.22 | 149.54 | 190.68 | 134.28 |
| 2,048 | 179.73 | 147.91 | 178.94 | 140.69 |
| 4,096 | 171.17 | 156.53 | 170.98 | 150.70 |

An independent code run ending after 1,024 return tokens gives 190.96 tok/s static
and 149.49 recover32. Both exact token prefixes and every transaction before the
common endpoint match the longer runs. This checks the prefix interpretation
without adding a large cross-product of duration experiments. The later prompts
differ from earlier prompts in both paired arms, so raw rates need not increase
monotonically even as recovery progresses.

## Why restoration stops

The separate code diagnostic reproduces every recorded proposal and transaction.
At the first fully general observation, 563 anchor experts are missing. Of these,
298 are unobserved, 52 fall below the count threshold, 98 fail the score margin,
33 lie beyond per-layer capacity, 50 lose to the global budget and 32 move.
Score rejection exists, but 82 proposals already compete for 32 places.
No victim is rejected by the configured residence guard.

Later the decisive constraint changes. At 85.5% set overlap, weighted anchor
coverage is already 97.9%; a full-window advantage of 1.36 percentage points
correctly declines recovery despite 152 otherwise eligible anchor pairs. The
last two full snapshots have 91.3/91.2% overlap and 98.8/99.1% weighted coverage.
Their anchor advantage is only +0.09 and then **−0.44 percentage points**, with
28 and 15 layers favoring the anchor. The current map is no longer broadly worse
than the reference. These are maximum-interval adaptation windows, not rejected
re-centering proposals.

The first full snapshot after return also illustrates boundary contamination:
it still includes preceding specialist traffic and reports −25.31 points of
anchor advantage. It declines movement. A fresh subsequent window qualifies.
The cheap indication, full-window recheck and actual fill are separate events.

The late serving trajectory supports this interpretation. In the last four
512-token code-return blocks, recover32/static throughput ratios are approximately
0.988, 1.018, 1.008 and 1.062 while overlap remains near 91%. Recover64 reaches
90% overlap after 7.39 seconds, versus 12.01 at recover32 and 13.29 at recover16;
none reaches 95%. Full set restoration is unnecessary for these later blocks.
Changing prompts and time-varying control work prevent a causal throughput claim
from overlap alone; weighted coverage explains which missing experts actually
carry demand.

Offline replay keeps the recorded counts, observation opportunities and 32-pair
budget fixed. It includes only fully return-phase windows, totaling 1,479,552
selections; the unsnapshotted tail is excluded. The unchanged variant exactly
reproduces every transaction before comparing alternatives.

| Code offline recovery | Cold selections | Moves | Final overlap |
| --- | ---: | ---: | ---: |
| Existing semantics | 103,482 (6.994%) | 384 | 91.27% |
| Remove score margin | 103,482 (6.994%) | 384 | 91.27% |
| Recent-count ranking | 107,682 (7.278%) | 352 | 90.34% |
| Direct bounded anchor restoration | 105,293 (7.117%) | 384 | 91.34% |
| One complete hard restore | 93,602 (6.326%) | 563 | 100% |

The hard restore knows the phase boundary and freezes the anchor afterward. Its
accounting includes about 1,426 MiB of API copies, versus 973 MiB for the existing
sequence, with map publication charged in prepared batches of at most two pairs
per layer. It is a routing reference, not measured serving or an optimal-placement
oracle. No serving hard-reset shortcut is added. Neither removing the margin nor
changing the ranking earns a production policy change from this evidence.

Math's first fully general observation has 638 missing anchor experts: 227
unobserved, 41 below minimum count, 126 rejected by score margin, 149 beyond
per-layer capacity, 63 beyond the global budget and 32 selected. Again, the
95 proposals exceed the budget despite the score rejections. No residence guard
binds. Its final two full snapshots show 98.57/98.70% weighted coverage at
92.03/91.59% overlap; anchor advantage falls to +0.46/+0.02 points. One normal
pressure-triggered movement at the return boundary still reflects preceding
math traffic and precedes the first re-centering operation. This remains visible
alongside the two later maximum-interval adaptations.

The math offline control covers 1,482,624 selections:

| Math offline recovery | Cold selections | Moves | Final overlap |
| --- | ---: | ---: | ---: |
| Existing semantics | 114,143 (7.699%) | 480 | 91.49% |
| Remove score margin | 114,143 (7.699%) | 480 | 91.49% |
| Recent-count ranking | 118,075 (7.964%) | 480 | 92.21% |
| Direct bounded anchor restoration | 116,050 (7.827%) | 480 | 91.85% |
| One complete hard restore | 95,317 (6.429%) | 638 | 100% |

Hard restoration accounts for about 1,616 MiB versus 1,216 MiB under the existing
sequence. More set overlap under the recent-count alternative does not imply
better routing coverage. Retrospective window correlations also do not establish
weighted coverage as a superior predictor: across 22 unequal observation windows
per fixture, Pearson correlation with static-normalized delivery rate is
0.866/0.863 for set overlap/weighted coverage in code and 0.961/0.882 in math.
Time, prompt mix, window length and control cost all co-vary. Weighted coverage
is useful for identifying demand among missing experts, not as a validated
throughput model.

## Restraint on partial return

The longer partial return contains 4,096 output tokens, taking about 33.7 seconds
under adaptation. It triggers **zero re-centering operations**. Its sampled
current-map cold fraction is 11.65%, versus 24.00% under the anchor, and final
anchor overlap is 77.41%. Recovery-enabled and normal-adaptive arms execute the
same complete transaction sequence: 27 ordinary pressure-driven operations in
partial return, plus eighteen in the preceding specialist interval.

Partial-return rates are 86.34 tok/s static, 121.47 normal adaptive and 121.45
recovery-enabled. Complete rates are 86.24, 118.66 and 118.65 respectively. Both
adaptive arms make 1,440 promotions, including 207 re-promotions and 22 completed
zero-hit promoted lifetimes; 893 lifetimes remain right-censored. This preserves
the churn cost alongside the serving gain. Elapsed time alone does not force
restoration when recent traffic favors the adapted map.

## Returning to the specialist again

The general→code→general→code fixture adds fresh second-specialist prompts after
the identical long return. Its first three phases retain exact output prefixes
and transaction sequences before the final return admission group.

| Repeat fixture | Complete sequence tok/s | Return tok/s | Second specialist tok/s |
| --- | ---: | ---: | ---: |
| Learned static | 106.93 | 171.86 | 52.73 |
| Adapt32, no recovery | 120.75 | 134.71 | 94.11 |
| Adapt32 / recover32 | 128.36 | 156.76 | 87.93 |

Recovery does not stick: second-specialist pressure appears after 0.75 seconds
and first maintenance completes after 0.90 seconds. Eighteen normal adaptation
operations promote 576 experts, reducing anchor overlap from 91.27% to 73.74%.
No re-centering occurs in that phase. Its sampled current-map cold rate is 16.84%,
versus 46.46% under the anchor.

The cost of recovering first remains visible. Second-specialist throughput is
6.6% lower than normal adaptation, which kept more specialist state. In the
recovery arm, 279 initially resident anchor experts are evicted in the first
specialist phase, restored during return and evicted again; 260 specialist experts
are promoted, evicted during return and re-promoted afterward. The corresponding
normal-adaptive counts are 168 and 145. Complete recovery-arm churn is 1,536
promotions, 285 re-promotions and 17 completed zero-hit lifetimes, with 812 active
lifetimes right-censored. Movement stays within the explicit budgets, but its
reuse cost is real. The complete-sequence gain does not erase this negative
second-shift result.

## Implementation and validation

Twenty accepted serving runs generate **119,808 tokens and 17,478 promotions**.
All pass exact paired-output and fixed graph/cache address gates. This includes
two separate full-policy diagnostic runs; their timings are excluded from the
headline tables. Historical evidence remains at its original source identity.

`ResidencyEpochCoordinator.observe(..., budget=...)` accepts an explicit typed
envelope for one observation. Omission uses the coordinator's normal envelope;
an override never changes that default. The worker adapter selects the recovery
envelope only for explicit re-centering and records the selected limits in each
backlog receipt. A zero pair or byte allowance admits no movement. Subsequent
normal adaptation still uses its original limits.

The serving harness exposes `--recenter-pairs` and `--recenter-mib` together.
Neither is required; omission preserves the previous behavior. Policy diagnostics
also record their configuration once at baseline creation. The analyzer checks
recorded proposals against the existing scoring rules before assigning blockers.
There is no additional replay work, allocation, device state or compiled program.

Frozen source 01 passes 342 host/compiler tests with two CUDA-only skips, 33 GPU
tests, three targeted cases under each of memcheck and synccheck with zero errors,
and ten source-built companion tests. An ordinary non-cache graph smoke generates
512 tokens and verifies loaded native libraries against the wheel. The metadata
census remains 88 declarations/246 programs with CUDA uninitialized. Both health
variants cross-compile for SM103 from the same runtime source; this does not
qualify physical B300 execution or Grace-backed TMA.

The failure ledger retains two initial test-fixture failures that acknowledged
unapplied fake transactions, a shell argument error rejected before engine
creation, and a launcher-file editing error after one completed baseline run.
The latter stopped the remaining batch, which is resumed without replacing its
completed receipts. Launch scripts remain immutable while running. Existing
ignored destructor warnings at engine shutdown remain in raw logs. None of these
failures is removed or relabeled as a serving correctness pass.
