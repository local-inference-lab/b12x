# Scheduler-owned expert-cache maintenance

Status: **experimental, opt-in, single-rank SM120 serving**. Static placement
remains the default. The shared policy and engine boundary are suitable for
other residency backends, but they do not qualify SM103 execution or Grace TMA.

The maintained vLLM companion exposes `AsyncLLM.residency_maintenance(config)`.
One engine utility request drains submitted model work, runs worker-local
residency control and resumes scheduling. Output delivery remains active.
The existing administrative pause API and distributed residency transaction
adapter retain their separate contracts.

## Ownership and safety

The engine sets `PAUSED_ALL`, drains its existing execution queue, and invokes
the worker's device synchronization. The worker cannot observe counters or
overwrite slots until that boundary completes. New requests may queue, but no
model graph is submitted while maintenance owns the boundary. The engine
rejects competing pause/resume requests during maintenance.

All participating layers share one counter snapshot and one bounded model-wide
decision. Before any fill, the worker checks every retained owner, pointer and
placement generation. It verifies all completed generations before
acknowledging policy outcomes. Unchanged immutable maps are compared with their
previously validated snapshots; changed maps receive full structural validation.
Expert slabs, maps, workspaces and captured graph objects retain their addresses.

Any uncertain worker result, partial layer failure or failed drain leaves the
engine paused and requires reload. A transport rollback alone does not prove
model-wide rollback. Client cancellation waits for the engine-owned operation;
it cannot release the scheduling boundary midway through a fill. Destroying the
worker destroys its controller and preparation owners. A fresh worker establishes
a fresh counter baseline.

The public administrative pause includes a 20 ms delay for output-delivery
ordering after drain acknowledgement. That delay is not the GPU reader barrier.
Maintenance retains the real scheduler/device drain and does not invoke the
administrative frontend's output-settling delay. Its completion callback can
resume queued requests without waiting for another client message.

TP, DP, PP and EP are unsupported by this worker-local operation. The existing
`VllmResidencyEpochs` adapter remains available for the separate all-rank
preflight/apply/acknowledge protocol. Controllers cannot switch between local
maintenance and that external protocol without reloading the lane.

## Routing pressure and history

`LocalResidencyMaintenance` derives recent cold selections from canonical
counter deltas and the unchanged placement generation. No additional device
counter, kernel, atomic or per-token host operation is added. Invalid counter
epochs and placement changes inside an observation window fail closed.

An optional explicit `cold_fraction_threshold` suppresses movement proposals
when recent routing remains below the threshold. The shared coordinator still
advances decayed scores, residence windows and useful-hit accounting. Above the
threshold it proposes movements, applies the existing global pair/copy-byte
budgets and invokes backend fills. The signal describes routing pressure, not
predicted throughput. It does not distinguish one-time cold touches from future
reuse, and a model-wide fraction can mask an individual layer's drift.

The gate avoids proposal ranking, repeated transaction RPCs and fills on a
healthy cache. It **still pays one engine boundary, counter readback and host
history update per check**. It is not a device interrupt or a free health check.
No threshold or automatic cadence is installed as a production default.

A policy window is the interval between actual snapshots. Gated checks retain
that history; a longer interval cannot reconstruct omitted short windows.
Device counter banks and device decay are deferred until serving evidence
justifies their replay cost and additional state.

The serving harness also offers an explicit research cadence through
`--healthy-check-max-tokens`. A healthy check doubles the delivered-token
interval up to that cap; pressure or absent observations restores
`--epoch-tokens`. It adds no device work and installs no production default.
There is no drift signal between checks, so a longer healthy interval can delay
reaction. Decay still occurs per observed window, whose length now varies;
this is not equivalent to fixed four-iteration LFU history. Compare fixed and
backed-off maintenance separately before selecting a cadence.

## Application control

After loading an adaptive cache and completing warmup, construct the driver
from the same layer policies and global budget used by external epochs:

```python
from b12x.integration.vllm.residency_maintenance import VllmResidencyMaintenance

maintenance = VllmResidencyMaintenance(
    engine,
    configs=layer_policies,
    budget=global_promotion_budget,
    cold_fraction_threshold=experiment_threshold,
)
await maintenance.run()  # Establish the post-warmup counter baseline.
# At an application-selected check interval:
receipt = await maintenance.run()
```

The driver serializes its calls and records lock waiting separately. It installs
no scheduler, timer or background task. Static mode prepares no observer and
invokes no maintenance. The benchmark's token trigger counts delivered output
tokens across requests, not scheduler iterations; concurrency changes the
relationship between those quantities.

## Measurements and reproduction

The [serving harness](../benchmarks/moe/expert_cache_serving.py) retains its
external pause/RPC control as the baseline. Use the same learned profile,
checkpoint, authored evaluation prompts and engine configuration in every arm:

| Arm | Harness options |
| --- | --- |
| Learned static | `--mode static` |
| Counters only | `--mode adaptive --control observe` |
| External control | `--mode adaptive --control external --epoch-tokens 32` or `128` |
| Worker-local maintenance | `--mode adaptive --control maintenance --epoch-tokens 32` |
| Conditional maintenance | Add an explicit `--cold-threshold` to maintenance |

`observe` retains the adaptive graph's counter and phase-setter nodes but
performs no policy, snapshots, epochs or promotions. It isolates observer cost;
it is not the uninstrumented static control.

Receipts distinguish frontend lock/roundtrip time, engine scheduler drain,
device-drain RPC, worker RPC, resume, counter synchronization/D2H/decoding,
pressure computation, layer policy, opportunity ranking, preflight, fills and
acknowledgement. Fill receipts further distinguish map verification, payload
enqueue/completion and map publication/completion. Nested intervals overlap:
device drain is inside scheduler drain, and worker sub-stages are inside worker
RPC. Do not sum them as disjoint costs. Scheduler drain includes the remaining
useful work of an already-submitted iteration.

The summarizer separates no-op and promotion checks, stable and code intervals,
and client delivery gaps that overlap an epoch. Delivery gaps are not GPU
iteration latency. Output token hashes, pointer assertions, raw timing samples,
GPU telemetry and exact source/build identities remain independent gates.

The [whole-K diagnostic](../benchmarks/moe/whole_k_schedule.py) compares native
preferred scheduling, deterministic whole-K and an all-resident prepared cache.
It uses checkpoint bytes and randomized routes, preserves numerical differences
from the preferred schedule, and rejects cache/whole-K disagreement before
timing. It does not change the serving numerical recipe to improve a benchmark.
For the tested checkpoint at top-k=8/M=1, preferred native, whole-K and
all-resident cache medians are 26.64, 38.93 and 51.22 microseconds. Preferred
small-M execution uses a different direct kernel, so the ratio includes route
preparation and execution structure rather than isolating split-K arithmetic.
The full shape/resource table is in the ledger.

## Serving evidence

Physical RTX PRO 4000 Blackwell experiments use Qwen3-30B-A3B-NVFP4, the same
learned 58/128-expert placement in every layer, and 16 authored requests with
128 generated tokens each. Eight general requests precede eight code requests.
Full decode graphs retain their addresses across every promotion. Output token
IDs match across the fixed-cadence and counters-only arms below at each
concurrency. The image supplies binary extensions;
the maintained companion supplies modified Python sources. These measurements
are evidence for that recorded mixed build, not a source-matched binary release.

Aggregate generated tokens/s, higher is better:

| Control | C1 | C4 | C8 |
| --- | ---: | ---: | ---: |
| Learned static, no observer | 52.84 | 78.94 | 102.78 |
| Counters only, no maintenance | 52.47 | 78.72 | 102.69 |
| External epochs, 32 delivered tokens | 52.33 | 72.96 | 89.83 |
| External epochs, 128 delivered tokens | 54.21 | 78.37 | 98.79 |
| Conditional maintenance, 32 delivered tokens | 63.02 | 90.27 | 110.26 |

Conditional maintenance uses an explicit 0.15 cold-fraction threshold and at
most 16 promotions per decision. Its gains over static are 19.3%, 14.3% and 7.3%
for this sequence. The interval split matters:

| Concurrency | Stable static | Stable maintenance | Code static | Code maintenance |
| --- | ---: | ---: | ---: | ---: |
| C1 | 91.96 | 86.21 | 37.07 | 49.70 |
| C4 | 131.28 | 124.27 | 56.45 | 70.93 |
| C8 | 190.85 | 178.40 | 70.34 | 79.85 |

The fixed check interval still costs 5–7% on the stable interval. Counters alone
are within 1% of static in these runs; that observation does not establish zero
overhead under other shapes or clocks. A repeated C1 pair gives static 52.82
and conditional maintenance 63.00 tokens/s. Dynamic clocks, the small authored
corpus and decode-only traffic limit generalization. Static remains the default.

C1 median engine control intervals are 152.27 ms for external 32-token epochs
and 32.71 ms for conditional maintenance. Maintenance checks with no movement
have a 23.23 ms median. The remaining costs include host history updates and
draining an already-submitted iteration; the latter is useful execution, not
entirely maintenance overhead. End-to-end throughput includes all costs.

Healthy-check backoff is **research-only**. With a 32–256-token interval, C1
produces 60.19 tokens/s overall, 89.84 during stable traffic and 45.27 after the
transition. C8 produces 106.15 overall, 187.49 stable and 74.06 after transition.
Both retain matched token IDs. Stable penalties shrink to about 2%, but delayed
reaction gives up some transition benefit. C4 fails the matched-output gate:
four stable requests diverge before any promotion. Generation-zero placement
rules out a preceding expert fill; request batching is a possible cause, not an
established explanation. The receipt is retained and excluded from matched-output
speedup claims. This result does not justify a default adaptive cadence.

A separate complete source-built vLLM wheel passes ten engine/loader tests,
2,048 adaptive serving tokens with 448 promotions, and an ordinary non-cache
smoke. Its loaded native-library hashes match the built wheel; adaptive token
IDs match the static reference. This closes the focused source-build smoke gate,
not a full source-built C1/C4/C8 performance matrix or independent PR CI.

Physical results and rejected runs are recorded in the
[engineering ledger](expert-residency-ledger.md). B300 qualification still starts
with all-HBM, all-Grace and mixed native correctness, Grace-backed TMA, graph
replay and fixed-address updates before measuring any control-loop benefit.
