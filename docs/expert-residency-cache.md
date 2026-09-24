# Grace-served misses and quiescent promotion

Status: **implemented experimental control-plane policy**. The policy composes
existing prepared routing counters and journaled slot exchange.
Its host contracts and decisions live in the shared
[`b12x.moe.residency` subsystem](expert-residency-subsystem.md); the
`fused_moe.ResidencyCacheController` entry point supplies the SM103 adapter. Portable tests
prove observation, decision, exchange and reuse of the same captured graph,
and the native SM103 variant passes on a physical GB300. Adaptive performance
remains **unqualified**. This is not an installed vLLM cache or a concurrent replacement
protocol.

A cold selection executes from its existing Grace row during the observation
window. At an engine-owned pause, the policy can promote a repeatedly selected
cold expert and evict a less selected hot expert. Later executions resolve the
same canonical ID through the updated map and use HBM. Execution never waits for
a promotion before servicing the observed miss. Graphs resume only after the
ordinary [slot transaction](expert-residency-slots.md) has completed.

## Observations without a kernel change

Prepared routing counters already count canonical expert selections independently
of storage rows. For a window in which the placement generation is constant,
subtract the preceding cumulative snapshot, then classify each expert using that
generation's map. This produces exact **observed** cold selection counts and the
set of cold experts with nonzero counts. Duplicate routes count independently.
Invalid IDs remain ignored by the existing counter kernel.

The window must end before an exchange. Counting across changing generations and
classifying the entire history with the final map would give incorrect miss
statistics. `ResidencyCacheController` rejects a placement change between its
baseline and observation, a counter reset, decreasing cumulative counters, a
foreign preparation, wrong rank or missing layer/phase geometry. The engine must
recreate it after counter repreparation: `RoutingSnapshot` carries a reset epoch,
not a globally unique counter-preparation identity. Direct private map writes
remain outside the supported slot contract.

Counters contain no per-step co-occurrence information. Ten selections of an
expert could occur in one iteration or ten iterations. A nonzero window count
establishes one window touch, not a measured per-iteration touch probability or
time to first reuse. Rich traces remain an explicit research option for that
question. No miss bitset, per-tier atomic, routing-kernel specialization or extra
observer launch is added. The existing optional counter launch still has cost.

Sampling retains its existing meaning: counts cover sampled calls, are not
extrapolated and may be biased by periodic traffic. Diagnostics report
`sample_every`, calls and sampled calls. With `sample_every=1` and complete phase
coverage they describe all observed selections; otherwise cold fractions and
post-promotion hits describe only the sample. Thresholds are in observed counts.

## Policy and configuration

`ResidencyCacheConfig` requires four explicit experimental controls:

| Field | Meaning |
| --- | --- |
| `max_pairs` | Maximum disjoint exchanges proposed per layer/window; must not exceed the plan's declared update capacity |
| `minimum_cold_selections` | Minimum selections of a cold expert in this window before admission |
| `minimum_score_gain` | Minimum positive difference between candidate and victim counts |
| `minimum_residency_windows` | Minimum observed-window age of a hot expert before eviction |
| `phase` | One declared phase, default `decode`; draft/verify/prefill remain separate |

There are no measured production defaults for the four thresholds. Initial hot
experts enter at window zero; promoted experts enter at their commit window.
Windows containing no sampled valid selections, including repeated empty polls,
do not advance age. An age of two allows eviction after two subsequent nonempty
observation windows. Window duration is an engine choice, so window age is not
wall time or request age. Within-window duplicate selections can satisfy the
selection threshold; they do not prove reuse in separate iterations.

The policy uses **recent-window frequency**, independently per layer:

1. Rank eligible cold candidates by descending window count, then canonical ID.
2. Rank unprotected hot victims by ascending window count, then canonical ID.
3. Pair candidates with victims while the score gain meets the explicit threshold
   and the batch limit is not reached.

Expert payload size is uniform within an existing layer's slot contract. The
policy therefore changes membership without changing hot-row count, memory
capacity or the numerical recipe. It does not redistribute HBM across layers,
fit a transfer-cost model, predict future traffic or blend traffic classes.

The learned workload profile supplies initial placement and remains unchanged.
Runtime generations do not overwrite that profile. Discarding the controller
stops decisions; restoring the exact saved placement still requires controlled
exchanges or reprepare/restart. The automatic static-profile controller and
monitor mode do not enable this policy or silently change their activation rules.

## Engine control-plane composition

Declare `updates=ResidencyUpdateCapacity(max_pairs=...)` on the ordinary expert
plan and budget its journal. Declare one ordinary `RoutingProfileQuery` covering
the observed layers, capacities, phase and authoritative TP rank; prepare its
counter Plan in the same `PreparationSession`. Capture the optional counter
binding alongside routing/expert execution. Static mode declares no observer.
No new component or preparation variant is required.

After warmup/capture, pause submissions, reset counters and construct each
layer's controller from the same model-wide snapshot:

```python
from b12x.moe import fused_moe as moe

controls = moe.routing_profile_state(counter_plan)
controls.reset(quiescent=True)
baseline = controls.snapshot(quiescent=True)
policy = moe.ResidencyCacheController(
    spec=layer_spec,  # ResidencyLayerSpec from the canonical weight plan
    config=moe.ResidencyCacheConfig(
        max_pairs=exchange_limit,
        minimum_cold_selections=minimum_observed_count,
        minimum_score_gain=required_count_margin,
        minimum_residency_windows=hold_windows,
        phase="decode",
    ),
    counter_query=counter_plan.query,
    slots=moe.residency_slot_snapshot(expert_plan),
    baseline=baseline,
)
# Engine resumes requests. Cold experts execute directly from Grace.
```

At a later boundary, the engine pauses every producer using these slabs and
keeps the pause through acknowledgement:

```python
observed = controls.snapshot(quiescent=True)  # Once for all observed layers.
slots = moe.residency_slot_snapshot(expert_plan)
decision = policy.observe(observed, slots=slots)
if decision.pairs:
    try:
        slots = moe.exchange_expert_slots(
            expert_plan, decision.pairs,
            expected=decision.expected, quiescent=True,
        )
    except moe.ResidencyUpdateError as error:
        if not error.resumable:
            raise  # Engine MUST remain paused and recover/reload the worker.
        slots = moe.residency_slot_snapshot(expert_plan)  # Restored generation.
outcome = policy.finish(decision, slots=slots)
# Successful finish permits the engine to resume its unchanged captured graphs.
```

`observe` performs no CUDA operation or storage mutation. `finish` accepts only
its pending decision and either the unchanged snapshot or the exact expected
pair permutation at generation+1. A decline or successful rollback consumes the
observed window without claiming a promotion. Failed acknowledgement, invalid
capacity or another exception must leave the engine paused until it resolves the
pending decision or recreates policy state from a safe baseline. Calls are
serialized by the engine; the policy does not acquire scheduler locks.

Each controller observes one layer in one execution lane. Do not combine counters
from independently placed lanes: their selections do not share one map generation.

Snapshot identity checks detect completed outside exchanges; they cannot stop
another thread from submitting a graph or changing placement during observation.
The engine pause is mandatory. Preserve the slot contract's owners, device drain,
payload completion, map publication and failed-rollback rules.

For replicated TP, only the counter owner runs this policy. Broadcast canonical
pairs out of band; each rank uses its own preparation snapshot when exchanging
its shard. Acknowledge the policy only after every rank succeeds. Partial-rank or
multi-layer failure has no automatic distributed rollback; keep the lane stopped
and recover. No per-token collective or CPU interaction is introduced. EP remains
unsupported. Counter reset/enable control must have a single lifecycle owner;
a calibration worker that disables counters on completion must not independently
control the same observer during a cache experiment.

## Diagnostics and interpretation

Immutable decisions/outcomes can be logged with `dataclasses.asdict`:

- Window counts, observed cold selections and unique cold expert IDs.
- Threshold/hysteresis rejection counts, protected hot experts and unpaired
  candidates (including capacity or unavailable-victim limits).
- The window's cold fraction and **counterfactual** fraction if its same routes
  had used the proposed map. This is not a forecast or throughput estimate.
- Committed pairs, generation, hot membership and cumulative promotions. Each
  promotion has one eviction in the exclusive-tier contract.
- Observed HBM hits after promotion, including hits recorded before an expert is
  evicted. Hits exclude the window that motivated its promotion. They establish
  reuse, not causal gains over every possible static profile.
- Committed payload/map copy volume supplied by the backend exchange contract.
  For the SM103 adapter, payload bytes are four times expert size per pair; maps
  include read and publication. These are successful API-copy
  bytes, not measured C2C traffic, and exclude failed attempts/rollback traffic.

Copy volume describes the policy owner's local expert shard; aggregate rank-local
measurements out of band for TP traffic accounting.

The engine can time the pause and exchange around these calls with a monotonic
host clock. The policy supplies no fabricated duration. Report observed hits per
promotion alongside churn, copy volume and total pause; do not interpret a high
counterfactual gain as an adaptive speedup.

## Backing store and concurrent designs

The [exclusive-tier backing model](expert-residency-slots.md#payload-layout-backing-storage-and-accounting)
remains authoritative. Grace stores cold experts; eviction replaces the promoted
expert's cold row. Full canonical Grace backing would avoid eviction write-back
but add approximately 206.612 GiB for the documented qualification placement.
That cost is not reserved implicitly.

An HBM spare slot can prevent overwriting a live hot destination, but it does not
protect the cold Grace row from concurrent readers when that row is reused for
the victim. A concurrent design also needs safe cold-row retirement/backing,
publication ordering, admitted spare capacity and physical-row capacity distinct
from logical hot membership. The existing complete row permutation contract does
not implement those states. One spare is not sufficient justification to remove
the pause. No asynchronous promotion or same-iteration copy race is introduced.

Router weights/biases remain immutable. Canonical ID -> live physical row remains
the execution contract for future grouped/persistent/fused kernels; a backend
that bakes fixed logical rows into captured descriptors must explicitly preserve
or replace that contract.

## Qualification and B300 experiment

Host tests cover deterministic decisions, per-layer independence, thresholds,
hold windows, sampling, owner/reset/generation checks, rollback acknowledgement,
no-op tiers and copy-volume accounting. Portable tests compose actual prepared
counters, the production partitioner, a test-only byte reader and existing slot
transactions. They change workloads through three promotions, retain one graph,
check every payload byte and verify stable pointers and allocator counters.
They do not execute native SM103 GEMMs or prove Grace-backed TMA legality.

On B300, establish static correctness first, then run the real policy loop:

```bash
python -m pytest tests/moe/test_sm103_residency.py -q
python -m pytest tests/moe/test_residency_cache_gpu.py -k native_sm103 -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -k native_sm103 -q
compute-sanitizer --tool synccheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -k native_sm103 -q
nsys profile --trace=cuda,nvtx,osrt --output=residency-cache-native \
  python -m pytest tests/moe/test_residency_cache_gpu.py -k native_sm103 -q
```

These native tests compare each generation with a fixed all-HBM control and
change both activations and route IDs. The Nsight command traces correctness,
including copies and synchronizations; its pytest wall time is not a performance
measurement. Record the UUID, driver/toolchain, source/checkpoint identity and
raw output using the [physical runbook](sm103-qualification.md).

A serving performance experiment requires the engine callbacks above. Use the
same checkpoint, learned initial profile, geometry, request replay and resource
budgets for three separately captured lanes: static without counters, static
with identical counters but declined decisions, and policy-enabled. This isolates
observer cost from replacement benefit. Use steady traffic, then A -> B -> A
workload shifts; preserve both warm and transient windows and balanced arm order.
Do not discard failed transactions or their pauses. Report full decode latency
and throughput, observed cold selections, per-step unique cold experts only if
separately traced, promotions, earned HBM hits, churn, measured pauses and C2C
traffic. The existing physical operator benchmark remains useful for the static
per-stage baseline; a policy-serving throughput harness is deferred until an
engine implements the pause/rank boundary. No policy benefit is claimed now.
