# Hybrid expert inference

Status: **implemented experimental contracts**. Physical qualification is
source-bound. The [hybrid qualification report](hybrid-inference-results.md) records TP1/TP2
serving and phase measurements for the ModelOpt NVFP4 adapter, together with
uncompleted gates. The [single-GPU control](expert-cache-next80-results.md)
remains attached to its preceding source pair.

The [continuation report](hybrid-continuation-results.md) keeps sanitizer
isolation, matched W4A16 prompt controls and TP2 exchange-value analysis separate
from the frozen qualification. It also documents the distinction between
completed production-component checks and a zero-error whole-stack sanitizer
result.

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
or engine reserves; the prepared adapter admits those separately. This contract
uses a uniform declared row size within a layer. Variable-length compressed rows
would require a bounded allocation/cost extension; the EXL3 fixture does not
claim compatibility with an actual EXL3 checkpoint or decoder.

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
An adapter with different source and backing representations must declare a
source transform; different backing and resident representations require a
promotion transform. A resident-only adapter also declares any required source
preparation. Rollback ownership is explicit for every adapter. A cold-execution backend rejects adapters without that
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
The coordinator itself has no two-rank specialization. TP serving currently
rejects routing history and anchor recovery; their existing TP1 research paths
remain available. This qualification enables ordinary adaptation only.

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

`benchmarks/moe/summarize_hybrid_prefill.py` reports pure model execution and the
elapsed span across prompt chunks separately from client TTFT. Mixed iterations
retain their complete duration and do not enter a pure-prefill rate. TP reports
use the maximum rank duration, never the sum. These event durations include
collectives but do not isolate communication time. Use the harness's explicit
`--torch-profile` diagnostic for kernel and collective attribution; its serving
times are not headline samples.
The harness exports complete traces without constructing the optional in-worker
summary table. Aggregation belongs offline: the table's retained Python event
tree caused profiler-only teardown failures in both ordinary and cache serving.

After phase calibration, `benchmarks/moe/balance_expert_profile.py` constructs
one experimental static placement with equal normalized prefill/decode weight
per layer. Integer cross multiplication preserves exact ranking and ID tie
breaking. The artifact retains both raw count sets, combined observed counts,
the objective and the source calibration hash. Its profile hash binds all of
them. Resident counts, checkpoint, TP geometry and execution recipe remain
unchanged. Loading validates the objective and reconstructs its membership;
ordinary decode-only profiles retain their original meaning.

`benchmarks/moe/nvfp4_repeatability.py` is an isolated ordinary Qwen3-Next W4A4
diagnostic. It takes a complete local checkpoint, a validated checkpoint receipt,
verified companion artifacts and a frozen tensor fixture containing input rows,
logical top-k IDs and route weights. It compares eager, captured and perturbed
executions, hashes postprocessed parameters before and after, and optionally
replays recorded FlashInfer tactics. Its independent GEMM/reference reduction
uses the native activation quantizer. A bounded fixture result does not establish
bitwise equivalence to the b12x W4A16 recipe or qualify arbitrary model inputs.

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

## Reproduce the added gates

Use the source-matched environment and immutable checkpoint receipt from the
[reference guide](expert-cache-reference.md) and
[Qwen3-Next qualification](expert-cache-next80-results.md#reproduce-the-gates).
The [current results](hybrid-inference-results.md) identify the tested source
pair and artifact hashes. Supply explicit paths; none of these commands downloads
a checkpoint or installs a runner.

Run host acceptance and the existing portable GPU tier first. The physical TP
layer worker uses the maintained model loader and final reduction. For two
authorized local devices, run the real early, middle and late layers:

```bash
export B12X_TEST_NEXT80_CHECKPOINT="$MODEL"
export B12X_CHECKPOINT_IDENTITY="$CHECKPOINT_IDENTITY"
export B12X_ACCEPTANCE_BUILD_MANIFEST="$BUILD_MANIFEST"
export VLLM_ENABLE_PCIE_ALLREDUCE=0
for layer in 0 24 47; do
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/moe/tp_checkpoint_worker.py --checkpoint "$MODEL" \
    --layer "$layer" --output "$RESULTS/layer-$layer"
done
```

The process count above selects the physical experiment. The transaction
implementation is tested separately at N=1/2/3/4. Model geometry must satisfy its
own divisibility and alignment requirements.

For TP calibration and serving, add `--tp-size "$TP_SIZE"
--disable-custom-all-reduce` to the existing serving harness. Generate a new
TP-bound profile; a TP1 artifact cannot be relabeled TP2. Use a conservative
admitted calibration envelope, record `--resources` and `--loader-coverage`,
then reuse `expert_cache_capacity --maximum` with that calibration receipt.
Inspect every rank's observed memory and release receipt before timing. The
reported maximum envelope is specific to the recorded source, geometry and
reservations.

The reference adaptive arm adds `--mode adaptive --control health
--epoch-tokens 16 --cold-threshold .15 --health-max-tokens 1024
--epoch-pairs 32 --epoch-mib 128`. The byte envelope is per rank. History,
specialist protection and anchor recovery remain disabled. Static uses the
identical initial profile with `--mode static` and allocates no observer.

For prompt measurements, supply independently retained prompt fixtures, use
`--tokens 1 --phase-timing`, and keep the context, prepared token capacity and
admission grouping fixed. Summarize the completed receipt with:

```bash
python -m benchmarks.moe.summarize_hybrid_prefill \
  "$RESULTS/prefill.jsonl" --output "$RESULTS/prefill-summary.json"
```

Collect route coverage in a separate `--mode adaptive --control observe
--phase-observations --routing-diagnostics` run. Require generation zero and
matching output IDs against its static timing control. Subtract the retained
initial counter boundary; startup routes are not evaluation traffic.

To construct the single balanced-profile experiment, enable
`--phase-observations` during disjoint calibration, retain its complete receipt,
and run `benchmarks/moe/balance_expert_profile.py --help` for the explicit input
and output arguments. This constructs one fixed equal-phase objective. Compare
it with the decode-only placement built from the same calibration, and retain
the historical profile as a separate control.

The existing `--shutdown-case health-pending`, `--shutdown-case
maintenance-cancelled` and `--repeat-lifecycle 2` options exercise engine-owned
control completion and reconstruction. Require zero cache-owned mapped bytes,
CPU sources, graphs and pending health state on every worker after release.
