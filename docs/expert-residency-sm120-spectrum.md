# SM120 expert residency test spectrum

Status: **research-only, measured single-layer diagnostics**. This suite extends
the [SM120 cache proof of concept](expert-residency-sm120-poc.md) with a spectrum
of cold-route fractions, live token counts, residency budgets and reuse windows.
It measures real native W4A16 execution on checkpoint expert weights. Activations
and routing distributions are synthetic; no complete model or serving engine is
executed.

The [cost diagnosis](expert-residency-sm120-costs.md) identifies unnecessary
Trellis-table reads in the NVFP4 path and CPU reads from write-combined journals.
It records corrected-source measurements separately from the baseline below.

## Reproduce

The runner writes an immutable output directory containing `manifest.json` and
an incrementally flushed `cases.jsonl`. It refuses to overwrite an existing run.
Numerical failures retain their traceback and receive no accepted timing result.
Fatal setup/CUDA failures leave an explicit failed or incomplete run.

```bash
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.moe.sm120_residency_spectrum \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --experts 512 --hot-experts 256 --top-k 10 \
  --live 1 2 4 8 16 32 64 128 \
  --cold-fractions 0 .015625 .05 .25 1 \
  --rounds 3 --repeats 8 \
  --policy-live 1 8 32 --periods 1 16 128 --epochs 8 \
  --source-revision "$(git rev-parse HEAD)" --output /tmp/sm120-spectrum
```

Use `--hot-experts 128` or `384` for smaller/larger resident budgets, and
`--top-k 2` or `6` for additional routing geometries. These are experimental
operator inputs, not recommended model configuration changes. Both tiers must
contain at least top-k distinct experts. The default layer prefix selects
`model.language_model.layers.0.mlp.experts`; `--prefix` selects another compatible
exported layer.

Host allocation experiments select `--backing-memory cached` and
`--journal-memory cached` explicitly. Both default to `write_combined`, preserving
the original exchange storage choice. The manifest records each choice. Exact
reproduction of the original kernel also requires its source revision; the
corrected kernel does not stage an unused Trellis table for NVFP4.

Every capacity is declared and prepared before graph capture. Live counts select
views within that capacity; they do not compile additional programs. Keep the
same maximum `--live` value when comparing sweeps to preserve the prepared
capacity and native kernel configuration.

## Recorded dimensions

| Dimension | Meaning |
| --- | --- |
| Live M | Tokens in one operator invocation; not end-to-end serving concurrency |
| Top-k | Distinct canonical expert selections per token |
| Cold fraction | Exact fraction of selections assigned to host slots at the start of a static case |
| Resident budget | Number of physical VRAM expert rows, with the remaining rows in mapped host memory |
| Cache condition | Repeated warm replay, or an L2 scrub before each measured replay |
| Policy period | Number of graph replays between out-of-band observations/decisions |
| Workload pattern | Stable initial hot set, stable initially cold set, one hot-to-cold shift, or continuously rotating cold experts |

Cold fractions are quantized by M × top-k. Positive requested fractions smaller
than one selection become one cold selection. Duplicate resulting counts are
tested once. For M=1/top-k=10, one cold selection means **10%**, not 1.5625%.
The receipt always reports `cold_count` and `actual_cold_fraction`.
Static latency fixtures spread expert IDs across tokens; policy fixtures repeat
one top-k set across the batch to study reuse. Rotating fixtures change that set
each epoch. The initial placement is a controlled positional split, not a learned
workload profile.

## Correctness and graph gates

Before timing each static case, the suite checks finite/nonzero native output,
relative L2 ≤0.005 and cosine ≥0.9999 against the all-VRAM native control. An
independent CPU sum of the weighted BF16 route outputs must match the finalizer
bitwise in original top-k order. Each case checks graph pointer stability and
zero allocator allocation/free events during static and profiled replay.

Adaptive windows check the actual completed invocation before publishing any
exchange. Device count deltas must exactly match the generated selection count
and the cold selections implied by the unchanged map generation. Warmup and
validation observations are excluded. Each comparison arm starts from the same
placement: the runner reverses committed exchanges through the existing journaled
transaction after the arm finishes. It never restores a map without restoring
payloads.

The suite reuses the existing experimental policy with a two-selection admission
threshold, two-selection score margin, one-window minimum residence and at most
one pair per boundary. At M=1 with a one-replay period, each distinct expert has
only one observation, so the threshold prevents promotion. Longer windows allow
promotion. These fixed test thresholds are not tuned production defaults.

## Timing method and interpretation

The latency arms are:

- **all_vram**: one native prepared operation with all expert fields in VRAM.
- **static**: both native tier operations, map derivation and ordered reduction;
  no profiling node.
- **profiled**: the static graph plus the prepared routing counter.

Isolated graphs additionally record map derivation, hot operator, cold operator,
ordered reduction and counter latency. Each tier operation includes its native
FC1/activation/FC2 and redundant tier finalization. These are not individual
GEMM kernel timings. Stage measurements are diagnostic and need not sum to full
operator latency because cache state and launch boundaries differ.

CUDA events bracket replay submissions. Warm samples average repeated replays;
cache-scrubbed samples exclude a preceding write over twice the queried L2 size
(at least 64 MiB). This is a cache-eviction proxy, not a measurement of PCIe
traffic. Very short isolated stages can include submission gaps; the full
operator comparisons are more informative than sub-microsecond differences
between those stages. Arm and cache-condition order alternate between rounds.
Raw samples remain in the receipt.

Policy comparisons record both device replay time and serialized host wall time.
The reported amortized wall time includes replay submission, completion wait,
counter snapshot, policy computation and journaled exchange. It excludes input
fixture changes, correctness probes and restoring the baseline between arms.
Exchange time is reported separately and includes the existing Python/CUDA
transaction. It must not be described as isolated DMA bandwidth or a serving
scheduler's complete pause time.

GPU UUID, driver, clocks, performance state, power, temperature, throttle flags
and PCIe link state are retained. These runs use default dynamic clocks and are
exploratory measurements, not release tuning acceptance. Small differences near
measurement noise do not establish a performance winner. Ratios explicitly use
**candidate time / baseline time**: values above one mean slower.

The policy records include observed cold selections, unique cold experts,
promotion count, VRAM hits after promotion, slot generation, API copy-byte totals,
per-window pause cost and numerical error. VRAM hits after promotion count route
selections; they are not equivalent to distinct expert weight fetches.

## Scope

Each card runs independently. There is no TP, engine request scheduling,
checkpoint quality assessment, continuous concurrent migration or B300 result.
The native cold path is PCIe mapped-host execution. Normal SM103 serving and the
shared policy implementation are unchanged. Full native sanitizer qualification
remains separate from these numerical and replay gates.

## Recorded results

The retained source export is based on `f7d1c532` and has SHA256
`5b174e0cc54e716ac217469cc53da9696e0aeb8c565628d63d88b374d63751dd`.
The [engineering ledger](expert-residency-ledger.md#sm120-residency-spectrum-evidence)
binds commands, raw receipts, source hashes, environment and device UUIDs.
All seven recorded sweeps passed: **141 fixtures**, comprising **101 latency
fixtures** with warm/scrubbed measurements and **40 policy fixtures**. An earlier
17-fixture pilot is retained separately. No fixture was dropped for failing
correctness or producing an unfavorable performance result.

| Sweep | Geometry or traffic variation | Passed fixtures |
| --- | --- | ---: |
| Main | M=1,2,4,8,16,32,64,128; top-k=10; 256 hot; five requested cold fractions; policy periods 1/16/128 at M=1/8/32 | 74 |
| Smaller resident budget | 128 hot; M=1/8/32/128; zero, approximately 5%, and all-cold routing | 12 |
| Larger resident budget | 384 hot; same live counts and route fractions | 12 |
| Smaller top-k | top-k=2; 256 hot; same live counts and fractions | 12 |
| Intermediate top-k | top-k=6; 256 hot; same live counts and fractions | 12 |
| Long reuse windows | M=1 policy decisions every 512 replays, plus M=1/128 latency controls | 10 |
| Second physical card | M=1/8/128; top-k=10; 256 hot; three route fractions | 9 |

All use one actual E=512/H=2560/I=640 checkpoint layer and a prepared capacity
of 128. Each latency arm has three samples, each averaging eight graph replays.
Each policy fixture has three paired static/adaptive rounds, eight epochs per
arm, and alternates arm order. Independent pytest results are **40 host passes,
3 CUDA skips**, and **12 passes on the second physical card**.

### Direct execution costs

Selected warm-replay medians from the main card, in microseconds. The all-VRAM
control receives exactly the same canonical routes and weights as the static
two-tier operation in each row.

| M | Actual cold selections | All-VRAM | Static two-tier |
| ---: | ---: | ---: | ---: |
| 1 | 0% | 49.9 | 234.9 |
| 1 | 10% | 50.0 | 340.1 |
| 1 | 100% | 49.8 | 1,593.0 |
| 8 | 0% | 420.3 | 612.5 |
| 8 | 5% | 420.3 | 1,081.0 |
| 8 | 100% | 420.2 | 12,100.6 |
| 32 | 1.5625% | 1,303.8 | 2,133.6 |
| 128 | 0% | 1,369.7 | 1,596.3 |
| 128 | 1.5625% | 1,448.6 | 4,498.0 |
| 128 | 5% | 1,669.6 | 11,051.1 |
| 128 | 100% | 1,350.4 | 38,455.6 |

The current cold operator remains expensive even with zero cold routes: roughly
172–198 µs across these selected main-card cases. This identifies empty-cold
execution as a concrete optimization target; the measurements do not identify
which internal kernel or memory access causes that floor. The prototype still
runs both complete native operators and redundant tier finalization.

The other card corroborates the scale of these costs: at M=1, zero-cold static
execution is 225.9 µs versus a 50.0 µs all-VRAM control; at M=128, all-cold static
execution is 39,747.1 µs. The cards are independent runs, not a TP comparison.

The CUDA-event counter-only measurements are approximately 3–7 µs across the
main shapes, including graph submission effects. Some full profiled samples are
faster than unprofiled samples under dynamic clocks; that is insufficient
resolution/control to assign a small profiler overhead, and does not show that
profiling is free. Cache-scrubbed results and every raw sample remain in the CSV
and JSONL receipts rather than being collapsed into these selected warm medians.

### When promotions pay for themselves

For M=1, the following ratios compare **adaptive amortized wall time / static
amortized wall time**. They include counter snapshots, decisions and exchanges.
Less than one means the adaptive experiment was faster.

| Workload | Every 16 replays | Every 128 replays | Every 512 replays |
| --- | ---: | ---: | ---: |
| Already-hot working set | 1.081 | 1.018 | 1.008 |
| Repeated initially cold set | 2.506 | 0.917 | 0.761 |
| Hot-to-cold workload shift | 2.497 | 1.095 | 0.944 |
| Rotating cold experts, no reuse | 2.816 | 1.229 | 1.065 |

A successful exchange takes **45.6–47.0 ms** in these policy sweeps, with a median
of **45.8 ms**. This includes journaling, both payload copies, map publication,
CUDA synchronization and Python transaction work. It is not a DMA-only timing.
The successful API copy accounting is 11,059,232 payload bytes plus 8,192 map
bytes per pair; physical PCIe traffic was not measured.

At a 512-replay period, the repeatedly cold M=1 fixture earns **14,336 subsequent
VRAM route selections across eight promotions**, or 1,792 per promotion. Its
three paired wall-time ratios range from 0.7607 to 0.7617. The rotating fixture
also performs eight promotions but earns **zero subsequent VRAM hits**, and loses
time. At M=8, exchanging every single replay makes the repeated-cold fixture
29.7× slower, showing why the quiescent mechanism must not imply a per-step
production policy.

Larger batches can amortize a pause sooner in these fixtures. At M=32 and a
128-replay period, repeated-cold and shifted traffic ratios are 0.686 and 0.883;
rotating traffic still loses at 1.057. These are observations for the specified
synthetic distributions and native operator, not recommended production periods
or a cost model for arbitrary workloads.

### Evidence-driven follow-up

1. Investigate the empty-cold operator floor and safe launch elision. This cost
   is present even when every selected expert is resident.
2. Decompose the roughly 46 ms exchange transaction before tuning policy or
   introducing concurrent movement. Payload copying, pinned-memory access and
   Python/map work need separate measurements.
3. Repeat promising cases with real routing traces and an execution schedule
   that includes other layers. Static learned placement remains the deployment
   baseline; the controlled positional split here is not its replacement.

Keep profiler fusion and concurrent cache replacement behind those measurements.
No production default, SM103 backend behavior or public residency API changes as
a result of this testing work.
