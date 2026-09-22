# Hybrid expert inference

Status: **implemented experimental contracts**. Physical qualification is
source-bound. The [Qwen3-Next report](expert-cache-next80-results.md) qualifies
the single-GPU ModelOpt NVFP4 adapter at its recorded revisions; it does not
qualify later storage, TP, or observation changes.

The system distinguishes three serving tiers:

| Tier | Placement and execution |
| --- | --- |
| Basic offload | Ordinary engine arithmetic, selective routed-parameter UVA offload, no learned placement or replacement |
| Optimized static | Checkpoint-bound learned logical experts, fixed resident slots and executable host backing |
| Adaptive | The same hierarchy plus decode observations and quiescent bounded replacement |

All-resident execution is a reference when the complete useful configuration
fits. Model CUDA prefill time, client TTFT, decode delivery gaps, and aggregate
generated-token throughput are distinct measurements. A comparison across
W4A4 and W4A16 is a deployment comparison, not isolated cache overhead.

## Storage and numerical ownership

`b12x.moe.residency.storage.ExpertStorageContract` describes one logical layer's
rank-local source, backing and resident representations. Each representation
declares its encoding, per-expert payload, shared layer bytes and alignment.
Payload arithmetic does not include slab padding, workspaces, graphs, journals,
or engine reserves; the prepared adapter admits those separately.

`ExpertShard` declares logical expert count, hidden width, local intermediate
extent and its position in the global intermediate axis. Rank is not part of
the expert ID. The existing weight planner continues to own activation
precision, source format, packing, and execution recipe selection.

| Backing capability | Meaning |
| --- | --- |
| `direct_mapped` | Backing is executable without promotion |
| `prepared_canonical` | Load-time transformation creates immutable executable backing |
| `prepare_on_promotion` | Promotion transforms backing; direct cold execution is unavailable |
| `resident_only` | No host execution representation exists |

Source transformation and promotion transformation are separate declarations.
An adapter with different backing and resident representations must declare a
promotion transform. A cold-execution backend rejects adapters without that
capability. Declaring a capability does not register an executable backend.

The NVFP4 adapter retains `ExpertWeightSource` compatibility. It validates
native packed bytes, K16 scales, positive finite global scales, BF16 whole-K
W4A16 and SiLU. Preparation swizzles scale storage one expert at a time. Both
resident and cold execution consume that canonical representation. Recovery
restores victims from immutable canonical backing. The generic replacement
controller sees IDs, membership, observations, generations and declared copy
costs; it makes no encoding decisions.

FP8, MXFP4 and transformed/compressed representations have host capability
fixtures, **not serving support**. A physical FP8 adapter needs a validated
checkpoint source, prepared executable representation, cold-execution contract,
bounded transforms and rollback, exact memory accounting, and real arithmetic
and graph-replay gates. PLE and MTP remain separate future ownership domains.

## Tensor-parallel residency

The ModelOpt adapter uses the maintained engine's loader slicing. Gate/up
matrices and their block scales shard the intermediate output rows; down
matrices and block scales shard the intermediate input columns. Scalar global
and input scales replicate. The router and logical IDs replicate. Shared
experts and the model's final reduction remain owned by the ordinary MoE runner.
No expert payload is broadcast by the residency protocol.

One engine-owned policy authority interprets rank 0's counters. Qualification
collects counters on every rank and rejects disagreement before mutation.
Profiles bind TP world size and local/global geometry; rank-local preparation
IDs and addresses may differ. Logical membership, slot rows and generations
must agree. The supported loader contract uses uniform intermediate shards;
model divisibility and adapter alignment may restrict eligible world sizes.
The coordinator itself has no two-rank specialization.

The distributed transaction validates all ranks, stages local payloads while
readers remain drained, waits for every staged acknowledgement, publishes maps,
verifies every generation, acknowledges all publications, then retires staging
ownership. Before publication, failures request rollback on all reachable
ranks. Any failed epoch leaves scheduling paused and requires coordinated
reload. Missing acknowledgements and uncertain publication are never success.
The existing single-rank transaction remains the serving path for TP1.

The pair cap counts logical promotions. The byte cap applies **per rank**;
reports retain per-rank and aggregate physical copy bytes. Host envelopes are
also per rank, with live admission checking the shared physical host pool.
GPU capacities are never combined into one address space.

## Phase diagnostics

The serving harness's `--phase-timing` option preallocates bounded CUDA event
pairs after engine preparation. Events bracket each model iteration outside
captured graphs and resolve after requests finish. Receipts classify scheduled
prompt and decode tokens, preserve prompt chunk sizes, and keep mixed iteration
durations separate. Pure-prefill prompt tokens divided by model CUDA elapsed
time is an execution rate; it excludes scheduler and delivery delay and is not
client TTFT. Overflow or incomplete timing fails instead of dropping samples.

`--phase-observations` enables explicit calibration/counter diagnostics with
separate prefill and decode rows. Scheduler-provided ranges label each token;
padding has no phase. One observer adds each valid route to exactly one phase,
including repeated IDs. Static timing allocates no observer. Health-triggered
reference adaptation keeps its existing decode-only observation contract.
Phase observations do not enable prefill promotions or runtime profile switching.

## Acceptance boundaries

Host acceptance includes non-NVFP4 storage capabilities and N=1/2/3/4 transaction
success, rank disagreement, rollback, cancellation and uncertain publication.
Portable GPU acceptance includes phase-labelled replay and staged canonical
fills with stable pointers and no replay allocation. These tests do not replace
real checkpoint TP layer arithmetic, per-rank full-model coverage, lifecycle
qualification or physical serving measurements.

Qwen3.8 remains [deferred](expert-cache-qwen38-audit.md). No additional checkpoint,
PLE integration or speculative execution is enabled by these contracts. SM103
continues to require the [physical native gates](sm103-qualification.md).
