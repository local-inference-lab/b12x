# Read-only expert-cache health probes

Status: **experimental**. Health probes use existing canonical routing counters
and the device expert map to detect routing pressure. They do not predict a
throughput gain, score replacement candidates or move experts. Static learned
placement remains the default. BF16 router arithmetic and the cache's whole-K
W4A16 numerical recipe are unchanged.

The [source-built serving results](expert-cache-health-results.md) retain all
three transition corpora, concurrency levels and control arms, including cases
where health control is slower than fixed maintenance.

## Three independent cadences

1. Routing observation increments the existing prepared per-expert counters on
   decode graph execution. Static serving has no observer.
2. An optional health probe reduces those counters against the current map,
   copies a compact result asynchronously and polls a CUDA event. It does not
   advance LFU scores, age residency guards or create a policy decision.
3. Full maintenance uses the existing scheduler-owned drain, complete counter
   snapshot, policy update and bounded promotion transaction. Only this step
   advances the policy observation window.

`RoutingProfileQuery(health_summary=True)` declares the reduction, device
baseline, descriptors, result storage and pinned result buffer. This option
requires owner-rank, decode-only counters. Its query schema is version 4;
health-disabled queries retain the original counting programs and storage.
The registered `moe.routing_profile` preparation contract owns both programs.
No kernel is resolved, compiled or allocated during graph replay.

The serving integration enables this declaration through the explicit
`ExpertCacheServingConfig.health_probes=True` option, supported only with
`mode="adaptive"`. Whole-model admission charges all health buffers once.
For L layers and a total of E logical experts, additional device tensor payload is
`16*E + 80*L` bytes and pinned result payload is `48*L` bytes. The device formula
includes the preparation-only map used to prime the real reduction program.
For 48 layers with 128 experts each, this is 102,144 device bytes and 2,304
pinned bytes. Allocator pool rounding still requires reserved headroom. Host
metadata and CUDA event objects also remain owned by the prepared state; they
are not model-scale storage.

Optional [deferred routing history](expert-cache-history.md) retains short counter
windows without advancing policy at probe time. Full maintenance can replay those
windows before its final movement decision. Health without history allocates no
history storage.

## Summary and baseline semantics

Each layer reports valid selections, cold selections, cold experts selected
once, cold experts selected repeatedly, and repeated cold selections excluding
one initial selection per expert in the probe interval. Invalid IDs are already
ignored by the existing counter kernel; duplicate valid IDs count independently.
These are selection statistics, not unique engine-step touch statistics.

The CuTe reduction reads each expert's cumulative count and subtracts its own
previous-probe baseline on device. It then checks the current map's tier for
that canonical expert. No new per-route atomic or production router variant is
introduced. The reduction has one 128-thread CTA per layer and 6 KiB dynamic
shared memory; the SM103 object also reports 1 KiB static shared memory.
Layer count and expert offsets are runtime values with 64-bit pointer arithmetic.

A baseline is bound to the counter epoch and every participating preparation ID
and map generation. Successful maintenance enqueues a fresh baseline after
publication and before another graph submission. Counts accrued under different
maps are never subtracted as if the meaning of cold residency were unchanged.
A counter reset, stale generation, decreasing counter, invalid tier or unsigned
sum overflow fails closed. Sticky overflow from the counter kernel is retained.

The host-only `b12x.moe.residency.RoutingHealthThresholds` contract evaluates a
summary with an explicit global cold threshold and optional minimum breadth of
layers above a layer threshold. Empty or incompletely observed layers return
`unobserved`. The rule is observational; replacement decisions remain with
`ResidencyCacheController` and the model-wide coordinator.

## Engine ordering and result ownership

The implemented adapter uses the maintained single-process, single-worker vLLM
executor. Model calls and worker utilities are serialized by that executor.
The start utility enqueues the reduction, a nonblocking copy into prepared
pinned storage, and a completion event on the model runner's producer stream.
It returns without waiting for the GPU. Subsequent graph submissions may
continue on that stream. Poll utilities read the pinned buffer only after the
completion event reports success.

There is no scheduler pause, device-wide synchronization or policy update in a
health probe. Output delivery remains active. The producer stream orders the
reduction after preceding counter writes and before later writes, so the device
baseline is a coherent observation cut. This contract does not admit independent
counter-producing streams, multiple serving lanes sharing one counter plan, or
TP/DP/PP/EP health aggregation. Those require an explicit completion protocol.

One prepared result slot permits one outstanding read. `VllmResidencyHealth`
serializes its calls and shields the submitted read from caller cancellation
until the result slot has been consumed. The serving harness additionally
serializes probes with maintenance and controlled admission. A pending read
rejects maintenance before any mutation. Owner/pointer validation and generation
checks occur on the worker for both start and poll. Unknown outcomes disable
further health control until reload. Existing uncertain-maintenance and failed
victim-recovery behavior remains unchanged.

No companion engine code changes are needed: the adapter uses its existing
worker utility and producer stream. Ordinary non-cache serving imports no
health callback, allocates no health storage and submits no health reduction.

The engine owner first prepares the cache with `health_probes=True`, captures
its ordinary serving graphs, and establishes the initial maintenance baseline.
The existing `VllmResidencyMaintenance` object owns policy and movement budgets.
A control loop can then compose the read-only adapter with that object:

```python
from b12x.integration.vllm.residency_health import VllmResidencyHealth
from b12x.moe.residency import RoutingHealthThresholds

health = VllmResidencyHealth(engine)
thresholds = RoutingHealthThresholds(cold_fraction=0.15)
await maintenance.run()  # Initial counter, policy and health baseline.

# Invoked periodically by the engine owner, outside graph replay.
async with control_lock:
    probe = await health.probe()
    if thresholds.assess(probe["summary"])["health"] == "pressure":
        await maintenance.run()
```

`control_lock` is shared with other maintenance and admission operations in the
owner's control loop. The example installs no background task or production
cadence. The benchmark additionally implements an explicit maximum interval;
it checks that interval after a completed probe, so it is not a hard deadline.

## Serving experiment

The explicit benchmark control is:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 \
  --mode adaptive --control health \
  --profile placement.json \
  --prompts benchmarks/moe/fixtures/expert_health_chat_code.jsonl \
  --output health-serving.jsonl \
  --admission together --concurrency 4 --tokens 128 \
  --epoch-tokens 32 --health-max-tokens 1024 \
  --cold-threshold 0.15 --epoch-pairs 16 --epoch-mib 64
```

Here `--epoch-tokens` is the health-probe interval measured in delivered tokens.
Pressure or the maximum full-snapshot interval triggers maintenance. The health
arm delegates movement to the existing bounded policy without a second global
cold-fraction gate over the longer full-snapshot window. Thus a maintenance
operation can legitimately decide to move nothing; a maximum-interval snapshot
can also identify opportunities that the coarse global health rule missed.
Thresholds and intervals are experiment settings, not production defaults.

The fixed and backoff controls retain their existing full-snapshot pressure gate.
Their LFU windows differ from the health arm's less frequent full observations.
Aggregate counters cannot reconstruct missing short history, so this experiment
does not claim to reproduce a four-step offline decay window.

Fixtures are independently authored general-to-code, prose-to-math and
English-to-multilingual sequences. Their text is disjoint from the retained
calibration and evaluation fixtures. Every arm starts from the same learned
checkpoint-bound general profile; no evaluation routes train that initial map.
The multilingual interval mixes languages deliberately and is not a homogeneous
post-transition distribution. These are routing experiments, not answer-quality
evaluations.

Paired numerical qualification uses acknowledged group admission and exact output
IDs. Equal prompts under unconstrained admission do not ensure equal BF16
execution shapes; see [cadence qualification](expert-cache-cadence.md). Traced
runs and unconstrained arrival experiments must be labeled separately.

## Retrospective economics

`benchmarks.moe.summarize_expert_health` compares qualified arms against learned
static, preserves per-corpus/per-interval results, and reports cumulative time
advantage at matched delivered-token counts. It reports the earliest crossing
that remains nonnegative through the retained interval, both for the interval
alone and after charging earlier intervals' time penalty. A missing crossing is
a valid result. It is not evidence that a longer unseen regime would repay.

Delivery curves include prefill and user-visible control costs. They are not
counterfactual GPU execution times, and cross-request delivery order can differ.
The report keeps asynchronous probe response latency, worker submission/poll
cost, GPU reduction/copy time and blocked scheduling time separate. In-flight
maintenance drains useful work, so its entire blocked interval must not be added
again to measured request time as if it were idle overhead.
Overall serving time also includes completion of any outstanding final control
operation. Delivery curves end at the last token; the analyzer reports that
final control tail separately and retains complete-run wall-time gain.
An adaptive cache may already differ from the learned map at the transition.
Earlier-interval gains can therefore put the cumulative curve ahead before the
first new pressure indication or promotion. Such a crossing is whole-run
payback, not a causal estimate of the new regime's minimum required dwell time.

## PCIe topology and portability

`ripper` is a single-NUMA Threadripper PRO 5975WX system with RTX PRO 4000
Blackwell GPUs. The serving link negotiates **PCIe Gen4 x16 under load**; idle
power management can report Gen1. These transport measurements do not establish
a Gen5 x16 workstation ceiling. Doubling nominal link bandwidth does not justify
doubling measured cache or serving performance.

A future Gen4/Gen5 comparison must keep checkpoint/profile identity, GPU class
where possible, resident capacity, host backing mode, actual route and promotion
sequences, concurrency and numerical recipe fixed. Retain negotiated link state,
NUMA/page locality, CPU affinity, root topology, driver/toolchain and power/clock
samples. Measure H2D fill completion, direct-host miss execution, complete
promotion, control cost and serving separately. A route/promotion mismatch makes
the experiment a whole-system comparison, not an isolated transport comparison.
No Gen5 throughput or fill timing is inferred here.

## SM103 boundary

The reduction is architecture-neutral metadata work and has a dedicated SM103
preparation declaration. Compilation and SM120 sanitizer evidence do not qualify
Grace-backed TMA or physical B300 serving. SM103 still requires static all-HBM,
all-Grace and mixed correctness, TMA legality, graph update, native sanitizers
and direct Grace miss measurements before adaptive serving qualification.

## Limits and next evidence

This is a single-worker control experiment with a single counter-producing
stream. Distributed health aggregation, multiple readers sharing a result slot
and multiple producing streams are unsupported. Health never authorizes a
storage mutation by itself; the existing full transaction still owns drain,
recovery, generation publication and fail-closed behavior.

The corpus matrix uses finite authored workloads, fixed output lengths, one
learned general profile and controlled admission. Dynamic clocks and one run per
arm limit conclusions about small differences. It does not measure arbitrary
arrival patterns, speculation, long-lived mixed traffic or answer quality.
The short/long regime experiment varies post-transition traffic while keeping
the stable prefix fixed; it does not establish minute- or hour-scale economics.

There is no new production cadence, replacement policy, device-side LFU decay,
asynchronous fill or transport optimization. A future completion-output health
hook could reduce the two utility round trips, but would require an engine
ownership contract and another serving comparison. A recent-count history bank
would address a different limitation: coarse full-snapshot windows cannot
preserve unobserved short-window recency. Neither change follows automatically
from faster health detection.
