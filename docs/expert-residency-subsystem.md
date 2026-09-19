# Shared expert residency subsystem

Status: **implemented host contracts and policy** in `b12x.moe.residency`.
The SM103 MXFP4 HBM/Grace adapter is the implemented execution backend. Portable
tests exercise its fixed-address exchange protocol and captured metadata path;
physical B300 execution and adaptive performance remain **unqualified**.
The [SM120 experiment](expert-residency-sm120-poc.md) composes native NVFP4
W4A16 execution with mapped host slots over PCIe and this same host policy.
It is a single-layer research harness, not another public serving backend.

The shared subsystem represents canonical expert identity, physical placement,
cumulative routing observations and experimental exchange decisions. It
imports only the Python standard library. It has no checkpoint recipe, hidden
dimension constraint, CUDA counter query or device allocator dependency.

## Ownership boundaries

| Owner | Responsibilities |
| --- | --- |
| `b12x.moe.residency` | Canonical ID/row mapping, generation snapshots, observation semantics, deterministic policy, acknowledgement and diagnostics |
| Backend through ordinary plans and `PreparationSession` | Numerical recipe, platform admission, storage layout, memory accounting, counter programs, execution, synchronization, journals and transactional copies |
| Serving engine | Phase labels, authoritative rank, sampling lifecycle, scheduler pause, graph submission, out-of-band snapshots, cross-rank coordination and recovery |

For SM103, these backend responsibilities remain under `b12x.moe.fused_moe`.
`plan_execution` and `PreparationSession` remain the preparation interface.
The shared namespace is not a separately registered GPU operation and has no
alternate preparation lifecycle.

The [model-wide epoch coordinator](expert-residency-epochs.md) retains separate
layer controllers and admits proposals under a global pair/copy-byte budget.
It imports no engine or device backend. The vLLM adapter owns RPC sequencing
around an engine-owned pause; partial rank/layer failure requires reload.
Full-model adaptive serving is not qualified.

## Shared contracts

`ExpertPlacement` maps every canonical expert exactly once into one of two roles:
resident execution storage (tier 0) or backing storage (tier 1). Each tuple lists
canonical IDs in physical row order. Logical identity is independent of row
number; router weights and bias do not move.

```python
from b12x.moe import residency

placement = residency.ExpertPlacement(
    total_experts=5,
    resident_expert_ids=(2, 0),
    backing_expert_ids=(4, 1, 3),
)
assert placement.expert_map[4] == (1, 0)
```

The roles do not imply coherence, available bandwidth or independently allocatable
memory. SM103 binds them to HBM and mapped coherent Grace slabs. A platform with
one physical memory pool must not count the same capacity twice by calling its
views separate tiers. A device that cannot execute backing rows directly needs
an explicit staged-miss execution contract; that path is **unsupported** here.

`RoutingObservationSpec` supplies layer identity, expert count, phase, maximum
top-k, sampling interval and rank ownership. It describes counts already produced
by a backend, rather than compiling a router. `LayerRoutingCounts` and
`RoutingSnapshot` carry cumulative uint64-range counts, call/token totals and a
counter reset epoch. Invalid route IDs are excluded by the producer; duplicate
valid selections count independently. Replicated routing is counted once by the
designated owner. EP/global-ID aggregation requires a separate backend contract.

`ResidencySlotSnapshot` binds an expert map to its preparation identity, generation
and health. `ResidencyUpdateCapacity` declares an admitted maximum exchange batch.
`ResidencyUpdateError.resumable` reports whether a failed backend transaction
restored a usable generation. Snapshots describe state; they do not own slabs or
prevent a scheduler from submitting work.

`ResidencyExchangeSpec` supplies the backend identity, direct-backing execution
and fixed-address quiescent-exchange guarantees, and successful copy-byte
accounting. A backend supplies payload bytes per exchanged pair and map bytes per
transaction. The shared policy does not assume a weight format, four copies,
int32 maps or a particular journal layout. These counts describe successful API
copies, excluding rollback attempts; they are not measured bus traffic or latency.
The descriptor is not a hardware probe or a replacement for preparation checks.
Its `backing_mode` distinguishes exclusive-row swaps from canonical backing
where cold expert E always resolves to row E. This does not change
`ExpertPlacement`'s exclusive static profile representation. Canonical runtime
maps require a backend that verifies and retains every source expert.

## Policy composition

The shared `ResidencyCacheController` constructor accepts `config`, `observations`,
`exchange`, `slots` and `baseline`. Both required backend guarantees must hold.
Each controller covers one layer/phase/lane, with interchangeable fixed-size
rows within that layer. Payload size may differ between layers. The policy does
not repack unequal rows within one layer or change per-layer slot capacity.

The [cache policy guide](expert-residency-cache.md) specifies thresholds,
deterministic ranking, hold windows and diagnostics. At each scheduler boundary:

1. Pause every producer and drain consumers of the generation being observed.
2. Obtain cumulative counts and a slot snapshot. The map must have remained
   unchanged throughout the observation window.
3. Call `observe` to propose canonical `(promote, evict)` pairs.
4. Submit those pairs to the backend's existing quiescent transaction, or decline.
5. Call `finish` with either the exact committed generation or the unchanged
   generation after decline/successful rollback. Resume only after acknowledgement.

`observe` and `finish` perform no device operations. Cold misses execute through
the backing tier before a decision. Promotions affect subsequent executions of
the same captured graph. Static serving declares no policy or observer, and
saved workload profiles remain unchanged by runtime exchanges.

The engine must retain backend owners and prevent all concurrent producers,
including raw graph replay. Fixed addresses alone do not provide memory ordering.
The backend must complete payload copies before map publication and restore both
payload and map on resumable failure. An unsuccessful rollback keeps the lane
stopped. Cross-rank commit/recovery remains engine-owned and out of band.

## SM103 compatibility

The public `fused_moe.ResidencyCacheController` preserves its existing `spec` and
`counter_query` constructor. It is a thin adapter to the shared controller:

- `ResidencyLayerSpec` continues to validate native MXFP4/A8 geometry and recipe.
- `RoutingProfileQuery` continues to describe prepared CuTe counter programs.
- The adapter translates one declared layer/phase into `RoutingObservationSpec`.
- The exchange descriptor accounts for journaling both payloads and writing them
  into opposite rows: four expert payload copies per pair, plus one map read and
  one map publication per transaction.

Shared configuration, decision, outcome, snapshot and error types are aliases at
the original public import paths. Static `ExpertResidencyPlan.placement` exposes
the shared row contract without adding serialized fields. Schema-1 layer profiles
and schema-2 automatic profiles retain their fields, hashes, model identity,
recipe checks and hardware/memory admission. Pre-extraction artifact fixtures
test their compatibility. Automatic discovery, convergence and model-wide
HBM/Grace budgeting remain in the native adapter because their memory formulas
and preparation constraints describe that backend.

The [SM103 guide](expert-residency.md), [automatic lifecycle](expert-residency-automatic.md)
and [slot contract](expert-residency-slots.md) specify preparation and deployment.
There is no production vLLM cache switch supplied by this extraction; the engine
callbacks in the cache guide are still required.

## Extending to another platform

A backend can reuse the host policy by supplying canonical counter snapshots,
generation snapshots and an exchange descriptor. It must implement and qualify
the storage/execution contract through its own ordinary preparation components:

- Admit resident/backing storage, scratch and rollback capacity against actual
  physical pools; preserve source weight and activation numerical boundaries.
- Execute backing misses correctly and resolve canonical IDs through the live map.
- Preserve captured pointers and planned capacity, with no replay allocation,
  compilation or host synchronization.
- Complete a bounded, quiescent exchange with stale-generation rejection and
  rollback; retain owners through every binding and captured graph.
- Prove same-graph execution before/after exchanges, fault recovery and exact
  backend accounting before evaluating performance on that platform.

No generic allocator, N-tier routing, staged misses, canonical duplicate backing,
spare-slot retirement or concurrent replacement protocol is implemented. Those
require additional contracts and admission tests. Reusing the host policy does
not qualify another hardware backend or make Grace-backed SM103 TMA legal.

## Canonical backing research

The [locality and canonical-fill experiment](expert-cache-evolution.md) evaluates
real checkpoint routes and a separate SM120 fill mechanism. The shared policy
remains unchanged. Its exclusive dense-row validation does not admit canonical
backup copies for resident experts; adding that capability requires an explicit
backing/row-capacity contract. The benchmark does not weaken validation or mutate
serving profiles to accommodate its experimental storage.
