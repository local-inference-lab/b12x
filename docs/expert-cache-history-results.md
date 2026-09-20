# Routing history and maintenance cadence on SM120

Status: **research-only**, with source-built SM120 correctness and serving evidence.

The 16-token health-probe repeat improves overall throughput in all nine
retained comparisons without a consistent stable-workload penalty. Retaining
short policy windows changes the first promotion decision, but does not by
itself recover the transition benefit of additional movement opportunities.
History remains an optional experiment; no serving default or numerical recipe
changes.

## Source and measurement contract

Inspection found b12x `f6daf48fb8484f68e62dee9d40212a3546a8cf4b`, already containing
master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, and companion vLLM
`3e45b530e58186046383e7294e611c2f6bf5cfb8`. Companion main is
`47ccf6c57d92f03630ebcbad3809450545825488`; its only absent commit removes inherited
CI workflows and changes no serving code. This experiment needs no companion
source changes.

The nine initial repeats use an immutable archive of the inspected b12x HEAD.
The history qualification uses source archive 03, SHA256
`e450946b1ca74b2909fad75727df3c0305b9f88697db720f73938b6b5f42f456`.
Its runtime package digest is
`8244104e55c8f8e3df8ee983d990c979b4e87fbe4a6f49ec9ebc8d29f4ba6f16`.
Later documentation and analysis changes do not relabel these frozen sources.

The complete source-built companion wheel is
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
All eleven native libraries and the maintained companion Python changes were
verified again. The wheel version label retains an older commit suffix; the
verified installed source and library manifest define the build identity.

Physical serving uses `ripper`, RTX PRO 4000 Blackwell UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02, Torch 2.13.0,
CUDA 13.3 and CUTLASS DSL 4.6.2. The host negotiates **PCIe Gen4 x16 under load**.
Dynamic clocks, power state and negotiated link state are sampled every 500 ms.
Idle Gen1 snapshots are retained. There are no Gen5 or physical B300 results.

All arms use Qwen3-30B-A3B-NVFP4, checkpoint fingerprint
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`, and learned
profile `20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.
The profile initially holds 58/128 experts in each of 48 layers. Settings remain
8 GiB expert admission, 2 GiB BF16 KV, context 2048, capacity 64, FlashInfer
attention, full decode graphs, controlled group admission and 128 greedy output
tokens per request. Actual route weights and whole-K W4A16 arithmetic are
unchanged. History comparisons retain 16 pairs and 64 MiB per maintenance.

Raw commands, logs, token IDs, telemetry, source archives and receipts are in
`/home/jasonc/b12x-history-evidence-20260920`; the physical copy is on `ripper` at
`/home/jasonc/b12x-history-results-20260920`. Historical health receipts remain
untouched in `b12x-health-evidence-20260920`. These are single-run serving
experiments with dynamic clocks, not statistical estimates of small differences.

## Sixteen-token health probes before implementation

All nine repeats pass exact paired output equality and fixed-address checks.
The 32-token reference is the immutable preceding experiment; only the 16-token
column is newly measured here. Higher tok/s is better.

| Transition | Concurrency | Historical health 32 tok/s | Repeated health 16 tok/s | Stable change vs historical static |
|---|---:|---:|---:|---:|
| Chat → code | 1 | 68.08 | 72.07 | -1.65% |
| Chat → code | 4 | 104.56 | 108.70 | -0.46% |
| Chat → code | 8 | 140.74 | 143.82 | -0.37% |
| Prose → math | 1 | 53.35 | 57.05 | -0.75% |
| Prose → math | 4 | 74.18 | 76.90 | +3.17% |
| Prose → math | 8 | 87.81 | 89.51 | +0.13% |
| English → multilingual | 1 | 66.20 | 68.41 | +0.25% |
| English → multilingual | 4 | 99.55 | 103.14 | -0.60% |
| English → multilingual | 8 | 128.24 | 131.21 | -0.35% |

First transition-pressure responses range from 0.42 to 1.66 seconds. Stable
maintenance is still occasional: 24 operations across these nine runs, versus
twelve in the earlier health-32 matrix. Several C1 stable intervals cross the
unchanged pressure threshold and move experts. Thus more frequent probes do not
guarantee either no healthy movement or a universal optimal interval. The
qualifying repeats cover 18,432 generated tokens.

## What history changes

The original prose-to-math C8 comparison confounds history and movement
opportunity. Fixed control performs 21 transition promotion epochs, moving 336
pairs; health performs sixteen, moving 256. Both repeatedly select the maximum
16 of approximately 96 proposals. The first fixed window contains 11,520
selections, whereas the first health window contains 398,592. A coarse first
window can change ranking, but its correction cannot recreate the missing
earlier promotions.

The nominal trigger interval is not the actual observation interval. In the
frozen-source C8 math runs, after the first maintenance:

| Control | Nominal delivered-token trigger | Median decode tokens per complete snapshot | Transition maintenance operations |
|---|---:|---:|---:|
| Fixed | 32 | 48 | 21 |
| Health | 32 | 64 | 16 |
| Health + depth 8 | 32 | 64 | 16 |
| Immutable-source health repeat | 16 | 48 | 21 |

The engine continues useful serving during asynchronous probe completion. The
client subsequently requests maintenance and sets the following trigger from
the advanced delivered-token count. In this C8 transition the median probe
response is about 243 ms, not a scheduler pause. This scheduling/probe interaction
is visible in `observed-window-cadence.json`; equal trigger arguments do not
establish equal policy windows or equal movement opportunity.

An engine completion-output hook might reduce this interval expansion, but its
benefit has not been measured. The existing 16-token control already tests a
smaller interval without an engine change. Adding an output-lifetime protocol
is deferred; these results do not establish that such a hook is unnecessary or
predict how much it would save.

In the inspected companion, `vllm/v1/worker/gpu/async_utils.py` already completes
sample-output copies in `AsyncOutput.get_output`, and the engine consumes the
result before `scheduler.update_from_output`. That is a plausible observation
delivery boundary. However, `EngineCoreOutputs.utility_output` resolves a
registered utility call ID; it is not an unsolicited health-result channel.
A completion hook needs explicit typed delivery and ownership of every pending
result slot across cancellation and queued model outputs. No arbitrary callable
RPC or callback is installed to bypass that contract. b12x would still interpret
health and policy; the engine would only deliver completed observations.

The implementation records cumulative counter cuts with a bounded prepared D2D
ring. At maintenance, the existing shared controller consumes earlier cuts with
movement prohibited, then evaluates the final window under the normal budget.
See [the history contract](expert-cache-history.md) for ownership, memory
admission, wrap semantics and generation validation.

A separate diagnostic replays the same observed counter cuts through the same
controller both with and without retained boundaries. The short-history result
reconstructs the recorded decision exactly. Only five of sixteen selected pairs
overlap the coarse decision on that same observation endpoint.

| Subsequent maintenance windows | Short-history candidate-minus-victim selections | Coarse-history candidate-minus-victim selections |
|---:|---:|---:|
| 1 | 220 | 84 |
| 2 | 383 | 143 |
| 4 | 705 | 260 |
| 8 | 1,643 | 637 |

These are retrospective routing balances with later replacement ignored. They
are evidence of useful ranking information, not measured counterfactual cache
hits or saved execution time. Full per-layer scores, count deltas, protected
victims and selected pairs are retained in `same-cut-policy-analysis.json` and
the diagnostic serving receipt. Diagnostic serialization is excluded from the
headline timing arms.

## Depth, cost and the targeted transition

The frozen-source prose-to-math C8 depth sweep keeps the 32-token probe cadence,
initial placement and movement budget fixed.

| Retained depth | Overall tok/s | Promotions | Extra replayed windows | Coalesced checkpoint boundaries |
|---:|---:|---:|---:|---:|
| 0 | 87.76 | 256 | 0 | 0 |
| 2 | 87.62 | 256 | 1 | 20 |
| 4 | 87.84 | 256 | 3 | 18 |
| 8 | 88.01 | 256 | 7 | 14 |
| 16 | 87.86 | 256 | 15 | 6 |

The entire range is under 0.5%; it establishes no throughput winner. Each history
arm records 37 checkpoints and sixteen maintenance operations. Only the first
maintenance has multiple retained cuts. Once pressure persists, every probe
already triggers maintenance, so there are no additional short windows to
recover. This explains why a better first ranking need not materially change
the complete interval.

One checkpoint copies 51,472 device bytes in this geometry. The median retained checkpoint CUDA-event
interval is about 4.10 microseconds. Overwritten slots and an unconsumed healthy
tail have no retained event sample; raw probe receipts separately count those
recorded checkpoints. Depth 8 admits 411,776 bytes on the device
and another 411,776 pinned host bytes. There is no history graph node, per-route
atomic, compiler specialization or checkpoint D2H transfer.

Full-ring readback event medians are approximately 39, 40, 46 and 61 microseconds
at depths 2, 4, 8 and 16. Host decoding plus deferred policy replay is materially
larger: approximately 4.1 ms at the typical one-cut maintenance, with first
multi-cut maxima near 16, 36, 56 and 90 ms respectively. These stage intervals
include their declared control work; they are not isolated transport bandwidth.
The 44.9 microsecond checkpoint outlier in the depth-2 arm remains in the receipt.

An overrun coalesces omitted boundaries into the first retained cumulative
window. It preserves selection totals but cannot reconstruct the omitted decay
steps. Depth 1 adds no extra policy window because the newest cut is combined
with the maintenance tail. Larger rings therefore cost both memory and host
interpretation time, without demonstrated proportional benefit.

## Complete prose-to-math control comparison

These five arms use the same frozen runtime and source-built companion. C4 arm
order is reversed to reduce consistent ordering bias. The C8 health/history
entries reuse the depth-0/depth-8 measurements above rather than timing an
identical configuration again. `Counters` has no health probe or maintenance;
`History` adds depth 8 to health-32. Checks exclude baseline establishment.

| C | Arm | Overall tok/s | Stable tok/s | Math tok/s | Stable checks | Math checks | Total promotions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Static | 47.05 | 90.44 | 31.84 | 0 | 0 | 0 |
| 1 | Counters | 46.80 | 89.41 | 31.74 | 0 | 0 | 0 |
| 1 | Fixed | 53.18 | 86.54 | 38.50 | 29 | 31 | 496 |
| 1 | Health | 53.62 | 89.98 | 38.26 | 2 | 28 | 480 |
| 1 | History | 53.28 | 89.26 | 38.04 | 2 | 28 | 480 |
| 4 | Static | 66.41 | 125.05 | 45.30 | 0 | 0 | 0 |
| 4 | Counters | 66.20 | 124.60 | 45.16 | 0 | 0 | 0 |
| 4 | Fixed | 75.90 | 121.79 | 55.30 | 25 | 26 | 464 |
| 4 | Health | 74.08 | 127.32 | 52.44 | 4 | 22 | 416 |
| 4 | History | 73.86 | 126.07 | 52.45 | 4 | 22 | 416 |
| 8 | Static | 83.87 | 164.79 | 56.39 | 0 | 0 | 0 |
| 8 | Counters | 83.82 | 164.56 | 56.37 | 0 | 0 | 0 |
| 8 | Fixed | 88.61 | 155.59 | 62.11 | 21 | 21 | 336 |
| 8 | Health | 87.76 | 164.71 | 60.01 | 0 | 16 | 256 |
| 8 | History | 88.01 | 164.87 | 60.20 | 0 | 16 | 256 |

Counters-only overhead is 0.06–0.53% overall in this repeat. History does not
materially improve math throughput: its change versus health is -0.6%, +0.02%
and +0.3% at C1/C4/C8. Stable history throughput remains near static, but the
larger transition benefit of fixed control at C4/C8 is still present. Retaining
evidence solves the cadence coupling contract; it does not demonstrate that
history depth is the dominant remaining performance lever.

## Additional transition comparisons

These are fresh health-32 versus depth-8 history pairs on the same frozen
source. C4 order is reversed. Static output IDs are checked against the retained
identical-admission references; the table does not attribute those historical
static timings to this source. Paired cells show health / history.

| Transition | C | Health overall tok/s | History overall tok/s | Stable tok/s | Transition tok/s | Promotions |
|---|---:|---:|---:|---|---|---|
| Chat → code | 1 | 68.30 | 67.63 | 89.79 / 89.03 | 55.27 / 54.67 | 336 / 320 |
| Chat → code | 4 | 104.52 | 105.08 | 161.79 / 161.55 | 77.48 / 78.15 | 288 / 272 |
| Chat → code | 8 | 140.79 | 141.42 | 252.04 / 251.93 | 98.11 / 98.79 | 240 / 240 |
| English → multilingual | 1 | 66.21 | 65.42 | 85.33 / 84.88 | 54.22 / 53.36 | 368 / 352 |
| English → multilingual | 4 | 99.56 | 99.35 | 161.61 / 161.80 | 72.16 / 71.92 | 304 / 304 |
| English → multilingual | 8 | 128.24 | 127.88 | 217.68 / 217.73 | 91.32 / 90.95 | 256 / 256 |

History has no consistent throughput advantage. The C1 losses and small C4/C8
gains remain in the comparison. Additional deferred observations range from
seven to 25 per run; their existence alone does not imply profitable movement.

Two additional frozen-source 16-token comparisons preserve the same movement
budget. They repeat the cadence effect while isolating history at that cadence:

| Transition / C | Arm | Overall tok/s | Stable tok/s | Transition tok/s | Promotions |
|---|---|---:|---:|---:|---:|
| Chat → code / 4 | Health 16 | 108.47 | 161.81 | 82.09 | 368 |
| Chat → code / 4 | Health 16 + depth 8 | 109.12 | 161.78 | 82.86 | 352 |
| Prose → math / 8 | Health 16 | 89.35 | 164.90 | 61.59 | 352 |
| Prose → math / 8 | Health 16 + depth 8 | 89.57 | 164.93 | 61.95 | 352 |

The larger difference is cadence, not depth. Probe frequency also changes which
brief pressure excursions are observed: the original math-C8 16-token repeat
performs a promotion at the end of the stable interval, 22 ms before math
admission. Therefore its benefit cannot be attributed only to faster detection
of the named transition. Thresholds and maximum-snapshot behavior remain
unchanged, and sixteen is not selected as a production default.

## Regime duration and payback

The C1 duration controls use the same eight-request stable prefix, followed by
two or eight held-out specialist requests. Each request generates 128 tokens.
The short fixtures contain no calibration text. Long English comparisons use
an additional fresh static run, not a historical timing reference.

Payback is the last crossing into a lead at matched delivered-token counts,
charging the preceding stable interval. It must persist through the recorded
regime. This retrospective finite-corpus metric is not a production predictor.
Complete-run gain also includes the final control tail; positive seconds favor
adaptation.

| Regime | Specialist tokens | Arm | Regime duration s | Payback s / tokens | Complete-run gain s | Transition promotions | Transition blocked-scheduling s |
|---|---:|---|---:|---|---:|---:|---:|
| Math | 256 | Health | 7.15 | 6.14 / 221 | +0.132 | 112 | 0.397 |
| Math | 256 | History | 7.11 | 6.66 / 239 | +0.089 | 112 | 0.430 |
| Math | 1,024 | Health | 26.76 | 5.76 / 209 | +5.336 | 448 | 1.539 |
| Math | 1,024 | History | 26.92 | 6.63 / 238 | +5.089 | 448 | 1.683 |
| Multilingual | 256 | Health | 6.76 | 1.40 / 52 | +0.610 | 96 | 0.319 |
| Multilingual | 256 | History | 6.79 | 3.00 / 123 | +0.424 | 112 | 0.414 |
| Multilingual | 1,024 | Health | 18.88 | 1.34 / 49 | +4.476 | 288 | 0.841 |
| Multilingual | 1,024 | History | 19.19 | 2.53 / 102 | +4.105 | 304 | 1.065 |

All these complete regimes finish ahead, but the short math margins are too
small for a robust general performance claim. Its first 128-token prefix does
not reach durable payback; that is a delivery-curve observation, not a separately
stopped serving run. History delays measured payback in these cases. The earlier
[health-duration experiment](expert-cache-health-results.md) retains its
short-code backoff loss. No losing or late-payback case is removed.

## Mixed traffic and return to general requests

`expert_history_mixed_return.jsonl` contains eight stable-prefix requests,
sixteen mixed requests across general, code, math, multilingual and tool-planning
styles, then eight independently authored general requests. Its new prompts do
not overlap calibration. The following C4 timings exclude the separate
full-score diagnostic run.

| Arm | Overall tok/s | Stable tok/s | Mixed tok/s | Return to general tok/s | Total blocked-scheduling s |
|---|---:|---:|---:|---:|---:|
| Static | 85.32 | 162.64 | 56.70 | 185.42 | 0 |
| Counters | 85.06 | 161.75 | 56.55 | 184.78 | 0 |
| Fixed | 90.63 | 151.92 | 66.87 | 132.05 | 6.391 |
| Health | 91.68 | 161.84 | 65.75 | 143.70 | 3.695 |
| History | 91.21 | 161.84 | 65.35 | 143.00 | 3.938 |

The complete sequence benefits from adaptation, but return-to-general traffic
is substantially slower than static. In the fixed arm, 24 of 26 return checks
are below the unchanged 15% threshold, and the observed full-window cold
fraction is 9.45%. Only 32 promotions occur there. Health/history each perform
one return maintenance and sixteen promotions. A routing map can satisfy this
experimental health threshold while still performing materially worse than the
learned general placement. History cannot repair a policy that elects not to
move. No threshold is lowered to hide this result.

| Arm | Promotions / 1,000 tokens | Re-promotions | Completed promoted lifetimes | Zero-hit completed lifetimes | Still-resident lifetimes | Final resident-set turnover | API copy GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 203.1 | 41 | 214 | 5 | 618 | 18.75% | 2.06 |
| Health | 168.0 | 22 | 140 | 0 | 548 | 17.46% | 1.70 |
| History | 168.0 | 28 | 149 | 0 | 539 | 17.10% | 1.70 |

Re-promotions count an expert promoted previously within the recorded run.
Each expert is evicted at most twice in these arms. Median completed lifetimes
earn 66.5, 65.5 and 74 observed resident selections respectively. Still-resident
lifetimes are right-censored, so zero completed zero-hit lifetimes does not mean
every promotion was useful. Copy volume is successful API movement accounting,
not measured PCIe traffic.

History records 94 health-associated checkpoints; 69 reach completed maintenance
and eight extra cuts are replayed. The 25-cut healthy tail remains unconsumed.
The separate diagnostic preserves subsequent demand for previously evicted
experts, full scores and actual selected pairs. It is excluded from the timing
comparison and does not estimate hypothetical throughput.

Client-visible delivery-gap p50/p95/p99 is 37.96/86.26/98.04 ms for static,
36.67/83.14/99.55 ms for health, and 37.14/84.30/104.46 ms for history. Median
TTFT is 316.42, 313.96 and 312.73 ms respectively. These are delivery metrics,
not CUDA iteration latency. Raw receipts retain request-level gaps, all epoch
stages, cold-pressure series and the losses during return traffic.

## Separate movement-budget control

After the fixed-budget history comparison, one additional math-C8 arm keeps
history disabled and increases only the global movement allowance to 32 pairs /
128 MiB. The larger byte ceiling admits 32 expert payloads; the prepared
per-layer capacity remains two pairs. Cadence, profile, arithmetic and backend
are unchanged. This arm is not pooled into the history-depth comparison.

| Health-32 control | Overall tok/s | Stable tok/s | Math tok/s | Math epochs | Promotions | API copy GiB | Apply wall ms | Blocked-scheduling s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 pairs / 64 MiB | 87.76 | 164.71 | 60.01 | 16 | 256 | 0.633 | 111.50 | 2.390 |
| 32 pairs / 128 MiB | 92.55 | 164.57 | 64.57 | 16 | 512 | 1.266 | 204.44 | 2.350 |

The larger allowance improves this transition by 7.6%, with exact paired outputs
and unchanged addresses. Its last observed cold fraction is 20.39% versus
28.07%. Median SM clocks are 2,475 MHz in both arms; memory clocks are unchanged.
Apply cost nearly doubles. The blocked interval also contains already-submitted
useful work, so its slight reduction does not mean movement becomes cheaper.

Together with the depth sweep and observed 48-versus-64-token windows, this
supports movement opportunity/capacity as the stronger limitation in this case.
It does not select a larger production budget. The mixed-traffic return penalty
shows why additional movement requires a separate churn and reuse evaluation.

## Implementation and qualification limits

The reusable change is small: prepared counter history, cumulative-cut
validation, deferred interpretation through the existing shared coordinator,
and opt-in serving diagnostics. `history_depth=0` allocates no ring. Static mode
rejects history; health without history retains its existing path. The engine
still owns quiescence, and only maintenance can change expert slots or maps.

Host tests cover memory admission, opt-out, malformed/reset/decreasing cuts,
deferred score evolution, empty observations, generation guards, movement only
at the final window, corrupt-tail failure and cancellation with or without
history. Physical GPU tests cover depths 1/2/4/8/16, nonuniform expert counts,
invalid and duplicated IDs, overflow, reset, wrap, rebase, frozen kernel
resolution, unchanged pointers and zero allocator events around same-graph
replays and checkpoints. Both memcheck and synccheck finish the five-depth
targeted cases with zero errors.

Final results are 275 host tests, thirty physical GPU tests, ten source-built
companion tests and five cases under each sanitizer. Ordinary non-cache serving
generates 512 tokens through captured graphs with no cache extension; hashes of
its five loaded native libraries match the verified wheel. The serving
qualification contains sixty complete runs, 130,560 generated tokens and 18,400
promotions, all with exact paired outputs and retained addresses. This total
includes nine immutable-source cadence repeats and separate diagnostic arms;
it excludes the two retained source-02 accounting-bug receipts.

All sampled request intervals use P1, PCIe Gen4 x16 and throttle mask `0x0`.
Raw clocks and power samples remain available, including startup/idle states.
Dynamic clocks and single-run cells still limit interpretation of small changes.
GitHub exposes no independent check/status acceptance for the inspected branch
source; local source-bound evidence is not a substitute for that integration gate.

The history copy path is portable CUDA storage infrastructure. The SM103
preparation case exports all four required native counter/health programs with
no CUDA context and an unchanged runtime digest. The metadata corpus remains
87 declarations/245 programs. This is compile evidence, not physical B300
acceptance. GB300 qualification still starts with all-HBM, all-Grace and mixed
native correctness, Grace-backed TMA legality, graph reuse and native sanitizers
before measuring adaptive serving.

History is limited to one serialized owner-rank decode stream in the implemented
serving adapter. Distributed history aggregation, independent producers sharing
a ring, archival retention and exact reconstruction after ring overrun are
unsupported. Allocator/event overhead uses the existing reserved headroom in
addition to admitted tensor payload bytes.

No changes are made to router reduction, whole-K arithmetic, checkpoint layout,
canonical fills, DMA, spare slots, copy overlap or production adaptive defaults.
The policy/control observations are useful beyond PCIe4, but their end-to-end
timings still depend on this model, GPU, engine and host. Neither throughput nor
copy behavior is extrapolated to PCIe5 or coherent Grace memory.

## Engineering decision

Keep learned static placement as the default and keep history opt-in. The ring
provides a validated observation contract, but a larger ring or a device-side
replacement policy is not supported by the measured benefit. Shallow retained
history is available for experiments that need it.

The next control experiment should account for actual observation intervals and
bounded movement opportunity, then test return traffic before selecting a
cadence or budget. Sixteen-token probes are a stronger research candidate than
increasing depth in these fixtures. A completion-output observation channel is
a possible way to reduce interval expansion, with an unmeasured benefit and a
real ownership contract to implement. Expert-copy optimization and numerical
schedule changes remain separate work.
