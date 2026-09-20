# Expert-cache cadence and execution-shape qualification

Status: **research-only, physically exercised on SM120**. Static learned
placement remains the serving default. Cadence changes do not alter cache
payloads until a bounded maintenance transaction commits. They can nevertheless
change request admission, prefill chunking and the engine's numerical execution
shape.

## C4 divergence: isolated cause

The retained healthy-backoff failure is reproduced with the unchanged b12x
and companion sources before diagnostic changes. Requests 4, 5, 6 and 7 first
differ from the static reference at generated-token indices 36, 12, 5 and 14,
respectively. Expert generations remain zero throughout that interval.

A complete source-built engine reproduces those same four output sequences with
**static cache placement, no observer and no maintenance**, by admitting each
four-request group together. Ordinary non-cache ModelOpt NVFP4 serving also
changes outputs between the admission patterns. That separate control uses the
ordinary A4 recipe; it is not a numerical equivalence claim between A4 and the
cache's W4A16 recipe.

The controlled cache comparison identifies this execution difference:

| Second request group | Streamed admission | Group admission |
| --- | --- | --- |
| First prefill | Request 4 alone, 20 rows | 64 rows; request 7 receives its first 6 prompt tokens |
| Following mixed step | 1 decode row + 56 prefill rows | 3 decode rows + 12 remaining prefill rows |
| Total rows in mixed step | 57 | 15 |
| Subsequent full decode graph | 4 rows | 4 rows |

For request 4's first decode input, layer 0's embedding, normalization, QKV
projection and attention output are bitwise equal. The first observed numerical
difference is the unquantized BF16 router projection: eight gate logits differ
by one BF16 step, at most 0.03125. Layer 0 selects the same expert IDs and its
final MoE output remains equal. Layer 1 also has identical attention and MoE
inputs, but twelve router logits differ. Its selected expert IDs remain equal;
route weights differ by up to 0.0012212172. The resulting MoE output differs in
988 elements, with maximum absolute difference 0.00048828125. Differences then
propagate through later layers before changing greedy choices.

The independent router replay uses the exact checkpoint gate weights and
captured activations. The engine's existing `torch.nn.functional.linear`
operation reproduces both recorded outputs exactly, without a cache, scheduler
or maintenance operation. Ten repetitions at each fixed shape are exact. In
this diagnostic, disabling Torch's reduced-precision BF16 reductions makes the
two shapes agree and match the BF16-rounded FP64 dot product. This isolates the
router reduction behavior; it does not qualify a different whole-model recipe.
Serving defaults and router arithmetic are unchanged.

For the earliest changed output, request 6 at generated-token index 5, the
static trace ranks token 198 at 27.625 above token 27315 at 27.25. The changed
admission trace ranks token 27315 at 26.625 above token 198 at 26.25. The earlier
first sampled hidden state is exact; divergence accumulates after the mixed
step. This is not evidence of expert identity corruption.

The unchanged failing run has no scheduler trace. The controlled admission
experiment establishes a sufficient cause and reproduces its output sequences;
it does not retroactively supply missing metadata to the historical receipt.
Both untraced and traced zero-promotion backoff attempts that do **not** reproduce
the failure are retained. Instrumentation itself can change admission timing.

## Correctness contract

Exact output equality remains the paired performance gate. Comparisons also
state their admission contract; equal prompts alone do not guarantee equal
execution shapes in this engine configuration.

With group admission, static and zero-promotion backoff have identical CPU
execution signatures across 516 steps. All 42 selected device records agree
exactly for input IDs, positions, sequence lengths, sampled hidden states and
full logits. All 2,048 generated token IDs agree. The no-op control changes
neither placement generation nor captured pointers.

The whole-K diagnostic tests M=1,2,4,8,16,32,64,128 with unrelated, repeated and
reversed routes. Both native whole-K and all-resident cache outputs retain the
same logical row exactly: 48 comparisons per source row. This passes with
checkpoint layer 12 and with the actual layer 1 activation, expert IDs and gate
weights from the affected serving step. The latter also matches the recorded
mixed-residency serving output exactly. Kernel resolution is frozen during
replay. These checks preserve the placement-invariant W4A16 contract; they do
not promise batch-invariant arithmetic for every operation in vLLM.

## Diagnostic tooling

`benchmarks.moe.expert_cache_serving` offers explicit experimental controls:

| Option | Purpose |
| --- | --- |
| `--epoch-pairs 0` | Exercise maintenance and policy history without promotion |
| `--admission together` | Pause at each idle request-group boundary, acknowledge every request addition, then resume |
| `--execution-trace PATH` | Record bounded CPU batch metadata through the research worker extension |
| `--trace-requests PREFIX ...` | Additionally retain selected device inputs, hidden states and logits |
| `--trace-modules NAME ...` | Capture selected eager prefill module inputs/outputs and actual cache routes |
| `--mode native` | Ordinary non-cache FlashInfer CUTLASS ModelOpt control |
| `--measure-control-floor` | After traffic, measure idle scheduler RPC, scalar readback, full snapshot and no-op maintenance; requires traced zero-movement maintenance |

The research worker wraps V2 runner methods only when explicitly selected. It
is not loaded by ordinary cache serving. Module hooks are installed after
graph capture and observe eager prefill; they do not claim to capture internal
decode-graph intermediates. Device copies and hook overhead invalidate timing
claims for traced runs. Overflow is reported, and the comparator rejects a
truncated trace. Source/profile/map identities and retained addresses remain
separate from batch metadata.

Compare logical sampled rows with:

```bash
python -m benchmarks.moe.compare_execution_traces left-trace.json right-trace.json \
  --output comparison.json
```

Replay the isolated router and recorded MoE row on physical SM120:

```bash
python -m benchmarks.moe.router_shape_replay \
  --checkpoint /models/Qwen3-30B-A3B-NVFP4 \
  --traces streamed-trace.json.pt together-trace.json.pt \
  --modules model.layers.0.mlp.gate model.layers.1.mlp.gate \
  --step 130 --output router-replay.json

python -m benchmarks.moe.whole_k_schedule \
  --checkpoint /models/Qwen3-30B-A3B-NVFP4 \
  --layer model.layers.1.mlp.experts --top-k 8 \
  --check-shape-invariance --logical-route-trace streamed-trace.json.pt \
  --trace-step 130 --trace-row 0 --output route-shapes.json
```

Use the recorded runtime environment and a fresh output path. Step numbers,
request prefixes and module names identify this receipt, not a public geometry
contract. The source-bound shell commands remain with the raw evidence.

## Health-check evidence

Idle measurements on the source-built engine give these median lower bounds:

| Operation | Median |
| --- | ---: |
| Scheduler-only RPC round trip | 0.684 ms |
| One existing 8-byte counter read | 0.0157 ms |
| Full 51,472-byte counter snapshot | 0.812 ms |
| Complete no-movement maintenance | 6.79 ms |

The snapshot's median D2H portion is 0.0164 ms and host decoding is 0.763 ms.
These are idle diagnostics, not additional serving pauses or an implemented
device health signal. An arbitrary scalar does not encode model-wide cold
pressure. In-flight maintenance additionally drains useful submitted work.

Per-layer counter analysis does not justify replacing the global pressure gate.
Healthy windows have global cold fractions of 3.3–13.8%, yet 1–17 layers exceed
15%. A worst-layer-only threshold would therefore enter maintenance during
healthy traffic. A positive unprotected candidate/victim score gap occurs in
45–48 layers even then. After the transition, 47–48 layers exceed 15% and the
global fraction rises to roughly 36–51% without movement. Repeated selections
are available from the same counters; unique engine-step touches are not.

No new production trigger, device counter bank or per-step host read is added.
The existing backoff remains experimental: fewer healthy checks save control
cost, while the absence of an intermediate pressure signal delays detection.

## Source-built serving comparison

The controlled-admission fixture runs Qwen3-30B-A3B-NVFP4 with the same learned
58/128 placement in every arm, 48 MoE layers, BF16 W4A16 whole-K execution,
FlashInfer attention, 2 GiB KV and an 8 GiB expert-device budget. Sixteen prompts
generate 128 tokens each: eight general prompts followed by eight code prompts.
Decode graph sizes are 1 and the selected concurrency; prefill capacity is 64.
Greedy generation ignores EOS and prefix caching is disabled.

All arms use `--admission together`. This barrier controls admission at idle
request-group boundaries, including the ordinary administrative pause settling
cost in elapsed time. It is a controlled serving fixture, not unconstrained
streamed admission. Its absolute throughput must not be spliced into the
historical streamed-admission table. Static has no observer or residency checks;
the counters-only arm has neither policy evaluation nor maintenance.

Fixed maintenance checks after 32 delivered tokens, with a 15% cold-fraction
gate and at most 16 promotions/64 MiB per epoch. Experimental backoff doubles
the healthy interval up to 256 tokens and resets it on pressure. Delivered
tokens are a host trigger, not a count of scheduler iterations or policy windows.
All timing runs disable execution tracing.

| C | Arm | Overall tok/s | Stable tok/s | Code tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Learned static | 52.37 | 90.43 | 36.91 |
| 1 | Counters only | 51.99 | 89.32 | 36.72 |
| 1 | Fixed conditional | 62.01 | 84.50 | 49.08 |
| 1 | Healthy backoff | 59.44 | 88.46 | 44.86 |
| 4 | Learned static | 79.60 | 136.05 | 56.40 |
| 4 | Counters only | 79.48 | 135.73 | 56.32 |
| 4 | Fixed conditional | 91.83 | 128.23 | 71.92 |
| 4 | Healthy backoff | 87.08 | 134.23 | 64.66 |
| 8 | Learned static | 100.18 | 201.88 | 66.81 |
| 8 | Counters only | 99.97 | 201.01 | 66.72 |
| 8 | Fixed conditional | 107.06 | 187.44 | 75.23 |
| 8 | Healthy backoff | 103.15 | 198.07 | 70.00 |

All four arms have identical output token IDs within each concurrency group.
All 24,576 tokens complete with stable graph and cache addresses; adaptive arms
perform 2,016 promotions. These are complete serving measurements including
control cost, not extrapolations from operator timings.

Backoff reduces the measured stable penalty to 2.2%, 1.3% and 1.9% at C1/C4/C8,
versus 6.6%, 5.7% and 7.2% for fixed checks. It also delays useful adaptation:

| C | Cadence | Stable checks / no-op checks | Stable promotions | Code checks / promotions | First code pressure detection | Code tokens delivered then |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Fixed | 30 / 26 | 64 | 30 / 384 | 0.99 s | 33 |
| 1 | Backoff | 6 / 6 | 0 | 22 / 320 | 5.78 s | 238 |
| 4 | Fixed | 25 / 25 | 0 | 26 / 368 | 0.81 s | 26 |
| 4 | Backoff | 5 / 5 | 0 | 20 / 304 | 5.23 s | 286 |
| 8 | Fixed | 21 / 21 | 0 | 21 / 336 | 1.13 s | 39 |
| 8 | Backoff | 5 / 5 | 0 | 16 / 240 | 5.04 s | 343 |

Detection time is measured from admission of the first code group to completion
of the first pressure check. It includes prefill and maintenance. The raw summary
also retains the last healthy check, intervening checks and cold-rate series;
snapshot windows may straddle the workload boundary. Observed code cold fractions
are 24.9%, 23.1%, 28.7% with fixed checks versus 28.9%, 26.1%, 30.8% with backoff.
Static has no routing snapshot, so no static cold rate is inferred here.

| C | Cadence | No-op median | Promotion median | All-check p95 | Total blocked scheduling |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Fixed | 23.38 ms | 47.36 ms | 62.04 ms | 2.144 s |
| 1 | Backoff | 24.50 ms | 47.67 ms | 68.44 ms | 1.208 s |
| 4 | Fixed | 44.22 ms | 80.77 ms | 98.64 ms | 3.124 s |
| 4 | Backoff | 36.84 ms | 81.46 ms | 93.68 ms | 1.762 s |
| 8 | Fixed | 52.63 ms | 122.15 ms | 140.19 ms | 3.720 s |
| 8 | Backoff | 51.66 ms | 131.21 ms | 156.57 ms | 2.384 s |

These intervals include draining already-submitted useful work; they must not
be interpreted as additive idle time or payload-copy time. The source-bound
`qualified-matrix-summary.json` retains TTFT, per-request decode rate, delivery
gap p50/p95/p99, maintenance-adjacent gaps, stage timings, movement bytes and
per-check cold fractions. Delivery gaps are not GPU iteration timings. For
example, C8 backoff's delivery-gap p99 is 158.11 ms versus static's 155.71 ms;
the whole-run gain does not imply every latency percentile improves.

This is one corpus and one pass at dynamic clocks. GPU telemetry is retained
every 500 ms. C4 arm ordering is reversed relative to C1/C8; some earlier C1/C4
runs share host resources with diagnostics on the other GPU. Differences around
1% are not a general overhead bound. Backoff saves healthy checks but retains
less transition benefit. Neither automatic cadence tuning nor a universal
adaptive default is qualified.

## Validation and remaining work

- Focused host/controller suite: **64 passed**.
- Prepared cache and routing-counter GPU suite: **21 passed** on physical SM120.
- Source-built companion engine/loader suite: **10 passed**.
- Whole-K checkpoint-row checks: **48 exact comparisons** for each of two
  source rows, including the actual divergent serving step.
- Twelve serving arms: **24,576 tokens**, exact paired outputs, fixed addresses.
- Ordinary non-cache graph controls: **4,096 tokens** with no cache observer.

The first GPU evidence archive omitted `scripts/`; its helper-import failure
after twenty passes remains retained. The corrected archive passes all 21 tests.
Completed workers can emit the existing ignored `AsyncLLM.__del__` teardown
`TypeError`; this investigation does not claim to fix interpreter shutdown.
The historical C4 failure and non-reproducing attempts remain intact.

The evidence ranks further work as follows: first, test cadence reaction delay
on more independently authored transitions; second, measure a compact health
summary at an existing safe boundary before adding device state; third, reduce
host snapshot decoding and policy cost if those stages remain material. Keep
whole-K arithmetic, transports and asynchronous fills separate. Physical B300
qualification remains a prerequisite for native SM103 performance claims.

## Source and hardware boundaries

The inspected b12x source is `6343ecc9b479425149fbc689b9b0732eed406303`, containing
master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. The maintained companion is
`3e45b530e58186046383e7294e611c2f6bf5cfb8`; its main branch is
`47ccf6c57d92f03630ebcbad3809450545825488`. No engine or core cache/kernel code is
changed for this investigation.

The complete source-built wheel has SHA256
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
Experiments retain its native-library identities, source archives, GPU UUIDs,
driver/toolchain, profile/checkpoint hashes, prompts, tokens and telemetry.
Raw evidence is under `/home/jasonc/b12x-c4-evidence-20260920` locally and
`/home/jasonc/b12x-c4-results-20260920` on `ripper`. Historical control evidence
under `b12x-control-evidence-20260919` remains immutable.

Timed serving uses source archive 05, SHA256
`9920ef355d9adaa985f271eedf18ec23160946d7bba672b78147686dd8aac442`;
subsequent diagnostic/CLI cleanup is not relabeled as the measured archive.
The final GPU suite uses archive 07 with the required script helpers included.
The raw manifest binds all archives, scripts, receipts and loaded extensions.
The serving GPU is RTX PRO 4000 Blackwell UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver 580.173.02,
CUDA 13.3, Torch 2.13, CUTLASS DSL 4.6.2 and Triton 3.7.1. The diagnostic GPU is
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. Both use PCIe Gen4 x16 on the
single-NUMA Threadripper PRO 5975WX host; clock/power variation remains recorded.
Checkpoint fingerprint is
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`, and learned
profile identity is
`20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.

Physical SM103 remains unqualified. The scheduler boundary is reusable, but
SM120 router and PCIe measurements imply nothing about Grace-backed TMA or B300
performance. Static all-HBM, all-Grace and mixed native correctness precede
SM103 graph-update and serving-control qualification.
