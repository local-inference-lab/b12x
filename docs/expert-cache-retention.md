# Temporary specialist retention during anchor recovery

Status: **implemented; research-only**. Optional victim protection uses hits
already earned by promoted residents. It applies only to anchor re-centering,
expires after an explicit number of nonempty policy observations, and adds no
routing instrumentation, GPU allocation or mutation protocol. Normal adaptation
can evict protected residents. Both protection settings default to zero.

The repeated-shift experiment tests whether a small residue of previously useful
non-anchor experts can reduce rebuilding when specialist demand returns. Its
primary comparison fixes adaptation at 32 pairs/128 MiB and recovery at
64 pairs/256 MiB. The result is a modest tradeoff, not elimination of cache
thrashing or a reason to enable a serving default.

## Source and physical contract

Inspection finds b12x `36b6f34f367d5327b19c64cbc0bdf3eb97c81eff`, master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, companion vLLM
`3e45b530e58186046383e7294e611c2f6bf5cfb8` and companion main
`47ccf6c57d92f03630ebcbad3809450545825488`. The companion requires no change.

The unchanged source is archived before implementation. Policy/serving source
archive 01 has SHA256
`db44a35929214f29ababed1148f513ec6d6bfbdadaf83009233fdebadfdaa1ff`
and b12x package digest
`8e655890325d8b03a5a8b72338585a23793b781acdbb0e76911b25bfbe0efb43`.
The per-file manifest distinguishes subsequent documentation, analysis and
fixture additions from the frozen runtime. A separate diagnostic archive adds
only request-boundary reads to the unchanged policy.

All runs use the source-built companion wheel SHA256
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
The wheel, eleven native libraries and maintained Python files are verified
against its manifest. Ordinary non-cache graph serving separately checks loaded
native-library hashes. The wheel filename's earlier revision suffix is not its
source identity.

Hardware is `ripper`, RTX PRO 4000 Blackwell UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02, Torch 2.13.0,
CUDA 13.3, CUTLASS DSL 4.6.2, one NUMA domain and **PCIe Gen4 x16 under load**.
Runs serialize through one GPU lock and retain 500-ms clock, power, throttle and
link telemetry. Transport timings describe this topology, not a Gen5 ceiling.

Checkpoint Qwen3-30B-A3B-NVFP4 fingerprint is
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`.
The learned profile is
`20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.
Geometry remains 58/128 resident experts in 48 layers, 8 GiB expert admission,
2 GiB BF16 KV, context 2048, capacity 64, FlashInfer attention and full decode
graphs. C4 requests use controlled group admission and 128 greedy tokens with
EOS ignored. Actual routing weights, whole-K W4A16 and BF16 router arithmetic
remain unchanged. Health probes run every 16 delivered tokens, history depth is
zero, cold pressure is 15%, the maximum snapshot interval is 1,024 tokens, and
anchor recovery requires two percentage points of advantage across at least 75%
of layers. Prepared per-layer transaction capacity remains two pairs.

Raw source archives, commands, build identities, policy windows, phase counters,
token IDs, telemetry and failures are retained in
`/home/jasonc/b12x-protection-evidence-20260920` and
`ripper:/home/jasonc/b12x-protection-results-20260920`.
The earlier [recovery-budget receipts](expert-cache-recovery.md) remain immutable.

## Victim reconstruction and oracle scope

The unchanged recover32 diagnostic reproduces the retained repeated-code run's
entire transaction sequence and exact output tokens. Complete policy windows
alone cannot precisely attribute healthy tails or windows crossing a workload
boundary. The optional `--routing-diagnostics` worker therefore reads existing
cumulative counters at the already-paused controlled-admission boundaries. It
validates owners, pointers and generations, completes pending device work, and
reads without consuming a policy window. Such runs are excluded from timing
claims. No per-route statistic or graph node is added.

`benchmarks/moe/analyze_specialist_retention.py` validates the learned profile
and reconstructs each recovery victim, its per-phase demand, earned hits,
promotion window, first subsequent request group with demand and re-promotions.
Group-level first use is not an exact first-token reuse distance.

The recover64 diagnostic evicts 350 non-anchor residents during re-centering.
They represent 53,885 of 387,840 subsequent code selections, or 13.89%. Of those
victims, 326 are used in the first second-code request group, twelve only in the
next group and twelve never in the recorded second-code interval. There are
221 recovery-evicted experts subsequently re-promoted. Those are finite-corpus
observations; an expert not reused in this receipt is not known to be useless
forever.

| Ranking of recovery victims | Subsequent code selections in top 48 |
| --- | ---: |
| Future-demand oracle | 21,114 |
| Preceding specialist demand | 18,928 |
| Hits since promotion, available at eviction | 18,642 |
| LFU score at eviction | 3,954 |
| Most recent promotion | 2,238 |
| Expected uniform random 48 of 350 | 7,390 |

Earned hits identify useful prior residents substantially better than promotion
age on this trace. The runtime signal uses no workload label or future demand.
The preceding-specialist and future columns are retrospective controls only.
Scores from different eviction windows are not normalized; this is a correlation
check, not a comparison of complete replacement policies.

The victim-only simulation holds recorded observation times and selected anchor
candidates fixed. Its unrestricted oracle selects the lowest future-demand
victim among those satisfying the unchanged score margin and residence guard.
It is a feasible greedy control, not proof of a global optimum. Later recorded
non-recovery transactions can become infeasible after victim substitution;
these are counted as skipped, not replaced with invented adaptation. No latency
or scheduler model is applied.

Unchanged replay reproduces every transaction. The oracle raises second-code
precoverage from 64.23% to 68.30%, but increases observed return cold selections
from 105,671 to 110,433 and skips fifteen later non-recovery transactions.
Protecting the two highest-hit non-anchor residents per layer for sixteen
observations reaches 67.21% in this constrained replay, with 108,151 return cold
selections and 26 skipped transactions. These controls justify a small physical
prototype; their coverage gains are not serving predictions. Four-window
protection changes no hit-ranked victims, because ordinary recovery already
retains the highest-value specialists early on.

## Experimental guard semantics

`ResidencyCacheConfig.recenter_protected_experts` limits protection per layer.
`recenter_protection_windows` sets its lifetime in **nonempty policy observation
windows**, not tokens, elapsed seconds or health probes. Both must be explicitly
positive to enable the experiment.

At the first permitted re-centering observation, each layer selects up to the
configured number of non-anchor residents with positive hits since promotion,
ranked by those hits and then canonical ID. It freezes that small set and a window
expiry. Subsequent re-centering selects its ordinary lowest-score victims from
the unprotected set. Candidate scoring, margins, thresholds, pair capacity and
model-wide budgets remain unchanged.

Empty counter windows do not age protection. A declined anchor gate performs no
movement and cannot start protection. A health probe neither ages nor renews it.
Expiry removes the guard; it does not force eviction or full anchor restoration.
Normal adaptation ignores the guard throughout. Committed normal promotion of a
non-anchor candidate clears the retained set and permits a later recovery episode.
A normal no-op or an anchor promotion cannot keep renewing old protection.

The anchor reference remains immutable within the policy session. Invalid
counter/generation observations fail before protection state advances, and
model-wide reference validation precedes consumption of any layer observation.
Partial-rank/layer failure, restoration poisoning and cancellation retain the
existing fail-closed transaction behavior.

The serving harness exposes the opt-in settings as:

```text
--epoch-pairs 32 --epoch-mib 128
--recenter-pairs 64 --recenter-mib 256
--recenter-protect 2 --recenter-protect-windows 16
```

The small retained set is host policy metadata. Static serving creates no policy
controller and pays no additional replay work. Backend transport, captured
programs and source ownership remain unchanged. The rule is portable host logic;
it supplies no physical SM103/B300 qualification.

## C4 repeated-code serving

Rates are generated tokens/s, including maintenance. Each primary run contains
1,024 general, 1,024 code, 4,096 return-general and 1,024 second-code tokens.
All seven timing arms below have exact paired output equality and unchanged
per-run graph/cache addresses. The final pair reverses arm order. Dynamic clocks
and ordinary control timing remain sources of variation.

| Arm | Return general | Second code | Complete sequence |
| --- | ---: | ---: | ---: |
| Learned static | 172.03 | 52.64 | 106.84 |
| Adapt32, no recovery | 134.75 | 94.06 | 120.75 |
| Recover64, first pair | 162.66 | 84.76 | 129.55 |
| Recover64, protect one/layer | 162.29 | 84.97 | 129.46 |
| Recover64, protect two/layer, first pair | 161.91 | 85.99 | 129.66 |
| Recover64, reverse-order pair | 163.10 | 85.09 | 129.77 |
| Recover64, protect two/layer, reverse-order pair | 161.93 | 85.83 | 129.61 |

One protected expert has no useful complete-sequence gain. Two improve second
code by 1.45% and 0.87% in the two pairs, but reduce general-return throughput
by 0.46% and 0.72%. The complete sequence changes by +0.08% and −0.12%:
**no consistent end-to-end throughput improvement is demonstrated**. This is not
sufficient evidence to recommend protection, expand its budget, change ordinary
adaptation or claim a general cache-thrashing fix.

The transaction sequences and end-of-return placements reproduce across the two
runs of each arm. The retained diagnostic's second-code counts score these maps:

| Placement before second code | Anchor set overlap | Anchor traffic coverage | Second-code precoverage |
| --- | ---: | ---: | ---: |
| No recovery | 85.38% | 97.61% | 76.49% |
| Recover64 | 91.92% | 99.31% | 64.23% |
| Protect one/layer | 91.95% | 99.30% | 64.64% |
| Protect two/layer | 91.74% | 99.24% | 65.41% |

Anchor traffic coverage scores the end-of-return placement against all observed
return-general counts; it is not a mean of the changing placement's coverage.
Second-code precoverage scores the same placement against future code counts and
is explicitly retrospective. Actual host policy changes global rankings and
health-triggered observations, so the fixed-candidate offline replay is not an
exact prediction of the physical protection arm.

## Rebuilding, movement and the remaining limitation

| Complete repeated-code run | Recover64 | Protect two/layer |
| --- | ---: | ---: |
| Promotions | 1,598 | 1,583 |
| Re-promotions | 320 | 317 |
| Completed zero-hit lifetimes | 14 | 14 |
| Active, right-censored lifetimes | 800 | 801 |
| API-copy bytes | 4,243,874,288 | 4,204,052,856 |
| Recovery-evicted experts re-promoted in second code | 221 | 207 |
| Second-code promotions | 576 | 576 |
| Second-code maintenance operations | 18 | 18 |

Each expert payload contains 2,654,216 copied bytes. The recovery-eviction/reload
subset accounts for 559.41 MiB of second-code payload reloads without protection
and 523.97 MiB with protection. This attribution excludes shared map-publication
bytes; the complete API totals above include them. Fewer repeated expert IDs do
not imply fewer total second-code fills: both arms still perform 576 promotions.

In the first pair, second-code blocked scheduling falls from 1,556 to 1,532 ms;
return blocking is effectively unchanged at 849/851 ms. First pressure is seen
at 805/784 ms, with first completed movement at 964/942 ms. These small timing
differences are not evidence of a different trigger or faster transport. The
256-token second-code blocks improve from 69.01/92.25/99.71/84.51 to
70.59/93.63/100.71/85.17 tok/s. The benefit remains modest throughout the short
specialist interval.

The first protection pair spends about 116 ms more in general return and earns
about 173 ms back in second code. Its finite-receipt durable lead, including the
return penalty, starts around 456 second-code tokens and 5.63 seconds, ending
only about 57 ms ahead. The reverse-order complete-sequence result loses. This is
not a robust payback threshold.

The main remaining limitation is visible in the victim lifetimes. Ordinary
recovery already keeps many high-hit residents: only 31 of the initially
protected 96 would have been re-centering victims in the reference run. Later,
**45 of that initial protected set are evicted by ordinary adaptation during the
general return**. They represent 21,629 subsequent code selections. The guard
works as specified; its deliberately limited scope permits these evictions.
Extending it into normal adaptation would change the requested policy contract
and is not justified by this result.

## Guardrails and negative controls

The independent clean code-return run with two protected experts produces
161.83 return tok/s and 1,007 promotions. Its corresponding prefix in the
repeated-code run is 161.91 tok/s; the immutable clean-return recover64 reference
is 163.24 tok/s. It retains most recovery performance but does not improve clean
return. The profile and exact output IDs are unchanged.

An eight-window expiry diagnostic uses the same long general fixture. All 48
layers reach expiry at the recorded window 28, with no active protection. At the
following permitted recovery operation, 60 pairs move, including 24 formerly
protected residents. Later normal adaptation also evicts former members. This
proves expiry and eligibility; it is not a proposed eight-window default.

The long partial return performs **zero re-centering and zero protection**. Its
transactions match the retained normal-adaptive receipt exactly, with 1,440
promotions, 207 re-promotions and 121.36 partial-return tok/s. The retained normal
reference is 121.47 tok/s. Duration alone does not activate recovery.

The fresh `expert_retention_different.jsonl` fixture preserves the initial
code/general sections and ends with eight independently authored mathematical
requests. Math throughput is 78.32 tok/s with ordinary recovery and 78.38 with
protection; static is 51.93. Both adaptive arms have exact paired output equality.
The protected run's first code interval is 2.6% faster **before any protection**,
despite an identical transaction prefix. Its larger complete-sequence number is
therefore not attributed to the guard. Raw timing variation is retained.

The limited complete-sequence result does not justify expanding to math-repeat,
C1/C8, larger protected sets or altered ordinary-adaptation semantics. The small
option remains available for explicit research; zero protection remains the
baseline. No workload classifier, additional learned anchor, deeper history,
asynchronous replacement or transport optimization is introduced.

## Qualification and reproduction

Fifteen source-built serving receipts contain **104,448 generated tokens and
19,434 promotions**. All pass exact paired-output, graph/cache pointer and
per-layer generation-accounting gates. Twelve are timing runs; two full-count
reconstructions and the expiry diagnostic remain separately labeled. Code and
partial-return output qualification uses immutable prior static receipts with
the same prompts and controlled-admission geometry.

Validation includes 141 focused host tests, 18 physical GPU tests, two bounded
model-wide graph cases under each of memcheck and synccheck with zero errors,
ten companion tests, and 512 ordinary non-cache graph-served tokens. No sanitizer
timeout occurs. The host suite covers expiry, no-hit promotions, changed anchors,
stale generations, declined movement, normal eviction and later re-arming.

All fifteen cache logs retain the optional Triton-import message, shutdown force
termination and ignored interpreter-teardown `AsyncLLM.__del__` exception also
present in the unchanged and historical cache harness. Receipts complete and
processes exit successfully, but these are not warning-free lifecycle runs.
No serving failure or numerical mismatch is hidden. An initial host command with
a nonexistent test filename is retained alongside the corrected passing suites.

Telemetry samples during accepted serving remain P1, Gen4 x16, with no reported
throttle reason. Control-policy choices and counterfactual routing are the
portable observations. Apply time, API-copy cost and serving throughput remain
specific to this GPU, host, engine build, admission contract and finite corpus.
No Gen5 or B300 performance is inferred. The native B300 acceptance order remains
all-HBM, all-Grace, mixed, Grace TMA, same-graph updates, sanitizers, Grace miss
cost, then adaptive serving.

The source-built launcher and every arm's full arguments are retained with the
receipts. A diagnostic reconstruction is:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 --profile placement.json \
  --prompts benchmarks/moe/fixtures/expert_recovery_repeat.jsonl \
  --output repeat-diagnostic.jsonl --mode adaptive --control health \
  --tokens 128 --concurrency 4 --admission together --cache-gib 8 \
  --epoch-tokens 16 --health-max-tokens 1024 --history-depth 0 \
  --cold-threshold 0.15 --anchor-advantage 0.02 --anchor-breadth 0.75 \
  --epoch-pairs 32 --epoch-mib 128 --recenter-pairs 64 --recenter-mib 256 \
  --policy-diagnostics --routing-diagnostics

python -m benchmarks.moe.analyze_specialist_retention repeat-diagnostic.jsonl \
  --profile placement.json --output retention-analysis.json
```

Timing arms omit both diagnostic flags. The protection arm additionally supplies
`--recenter-protect 2 --recenter-protect-windows 16`. Every arm starts a fresh
engine from the same immutable learned placement.
