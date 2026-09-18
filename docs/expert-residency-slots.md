# Quiescent expert slot exchange on SM103

Status: **implemented opt-in control-plane prototype**. Portable SM120 tests
validate fixed-address replacement, mapped-host byte access, rollback and reuse
of the same CUDA graph. Physical SM103 expert execution and Grace-backed TMA
remain **unqualified**. No adaptive policy or serving-engine integration is
installed by this feature.

## Logical identity and physical placement

The existing `PartitionRoutes` kernel reads `mapping[expert] = (tier, row)` on
every execution. A canonical router ID therefore identifies an expert payload,
not a slab row. Exchanging payloads between two fixed rows and publishing the
corresponding map preserves that identity without changing graph structure.

A captured binding retains the HBM and Grace field pointers, device map pointer,
workspace pointers, route-ID and weight pointers, output pointer, and the bound
live dimensions. Route counts and compact indices are workspace contents
produced by partitioning on each replay. Graphs can read changed contents at
those addresses. Changing a captured live dimension still requires the engine's
ordinary prepared graph-variant selection. Exchanges change neither dimensions,
tier capacities, tensor descriptors nor pointers.

This follows the distinction between captured parameters and their pointed-to
storage in [CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
Pointer stability alone does not establish ordering. CUDA's
[mapped-memory requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#mapped-memory)
also require synchronization between readers and writers. SM120 mapped-host
loads cannot establish that SM103 TMA accesses to Grace pages are legal.

Router rows, biases, correction terms and auxiliary metadata remain unchanged.
Renaming experts through router-row swaps would require coordinating all those
structures, routing counters, distributed IDs and checkpoint interpretation.
Map indirection already serves the addressing purpose, with no such identity
changes. There is no measured reason to change the router.

## Public declaration and operation

Declare bounded rollback capacity before preparation:

```python
from b12x.moe import fused_moe as moe

plan = moe.plan_execution(
    experts=weight_plan,
    weights=cpu_checkpoint_weights,
    placement=initial_static_profile,
    capacity=execution_capacity,
    memory_budget=layer_memory_budget,
    updates=moe.ResidencyUpdateCapacity(max_pairs=2),
)
# Submit this plan's real prepare_call to PreparationSession, freeze, bind,
# and capture as in expert-residency.md.
```

Omitting `updates` keeps static behavior: no journal, no mutable generation and
no additional graph node or device instruction. `ResidencyQuery` schema 2 records
`max_swap_pairs`; kernel geometry and numerical contracts are unchanged.

At an engine-owned pause:

```python
# The engine has stopped EVERY submitter using this plan's slabs, including
# raw graph.replay(), other streams, verifier lanes and background warmup.
snapshot = moe.residency_slot_snapshot(plan)
try:
    updated = moe.exchange_expert_slots(
        plan, ((cold_expert, hot_expert),),
        expected=snapshot, quiescent=True,
    )
except moe.ResidencyUpdateError as error:
    if not error.resumable:
        # Keep submissions stopped. Release graphs and reload the worker.
        raise
    # The previous payloads, map and generation have been restored.
    # Engine policy may resume that placement or keep the lane paused.
    raise
# Successful return permits the engine to resume the SAME captured graphs.
```

The `quiescent=True` argument is an explicit caller assertion, not a scheduler
lock acquired by b12x. A device synchronization cannot stop another thread from
submitting work afterward. The engine must hold its pause until the call returns,
and must exclude graph capture and preparation release during the operation.

Pairs contain distinct canonical IDs. Cross-tier pairs promote one expert and
evict the other; same-tier pairs permute physical rows. A batch may contain up to
`max_pairs` disjoint pairs. Repeated IDs, invalid IDs, empty batches, stale
snapshots and undeclared updates fail before payload writes. Arbitrary cycles,
capacity changes and replacing expert values from a different checkpoint are
unsupported. A snapshot carries preparation identity, generation, the complete
host map and health status; it performs no D2H transfer. Preparation identity
prevents reusing a generation-zero request after releasing and repreparing.

## Transaction and synchronization contract

The mechanism is a blocking quiescent transaction on one prepared layer:

1. Drain every stream on the owning device with device synchronization.
2. Copy the device map into a prepared host buffer, synchronize and compare it
   with the authoritative host generation. Unexpected map mutation poisons the
   state; b12x cannot infer which payloads an external writer changed.
3. Copy all four fields of every touched row to prepared pinned rollback storage.
   Synchronize before overwriting any slot. The journal includes both members
   of every pair, including cold rows.
4. Exchange payloads using the completed journal. Synchronize all copies before
   publishing the map.
5. Copy the complete candidate map into the same device tensor and synchronize.
   Advance host generation only after completion. Return the committed snapshot.

Copies use `cudaMemcpyAsync` with `cudaMemcpyDefault` and the owning device's
current stream. Unified-address pointer inference selects the transfer kind;
see [CUDA memory management](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
No tensor is migrated, resized or reallocated by the transaction. Python metadata
and tensor views may be created on this out-of-band path. No transaction work
belongs in graph capture or replay.

There are no readers while payloads or the map are being written. Consequently
the two int32 values in a map row need no atomic-word encoding, selector or GPU
generation check. Host generation protects stale control-plane requests; it is
not a concurrent-reader protocol. An old map remains authoritative until commit,
but its slots may be temporarily overwritten: retaining the old map alone would
not provide rollback. The journal is what permits resumption after failure.

If a staging, replacement, completion or publication operation fails, b12x drains
pending transfers, restores every touched payload and the original map when
needed, and synchronizes before reporting `resumable=True`. Generation remains
unchanged. If rollback or synchronization fails, state becomes unhealthy and
`resumable=False` requires reload. Ordinary bind/run checks reject unhealthy or
busy state. Previously captured raw graphs bypass Python guards; the engine must
never resume them after a nonresumable failure. This is in-process recovery, not
crash consistency across process/device loss.

The exchange lock serializes update callers; it does not serialize graph
submission. Prepared state and bindings retain slab/journal owners. Destroy graphs
before releasing those owners. A public binding from a released or replaced plan
is invalid, just as for static residency. TP rank coordination is engine-owned:
pause all participating ranks, apply an agreed batch to each shard and resume
only after all acknowledge success. There is no distributed transaction or
multi-layer rollback. Any partial-rank failure keeps the entire lane paused and
requires engine recovery.

## Payload layout, backing storage and accounting

The production tensor-major layout is unchanged. Each row consists of `w13`,
`w2`, `s13` and `s2`; scales are already swizzled in prepared storage. Exchanges
copy bytes without dequantizing, requantizing or changing scale layout. Preparation
rejects a payload-field set or geometry that the journal cannot preserve. Future
bias/metadata fields must participate in both storage and rollback.

Grace remains a mutually exclusive tier. An evicted hot expert replaces the cold
row of the promoted expert; all physical tier counts remain constant. A full
canonical Grace backing would eliminate eviction copy-back, but would require
additional Grace capacity for every hot expert plus a separate backing identity
contract. Spare HBM slots would similarly require admitted capacity and reader
retirement. Neither is hidden in this implementation.

`ExpertMemoryAccounting.update_host_bytes` charges one exact-size mapped pinned
journal and two aligned map buffers. `grace_total_bytes` includes cold experts
and this journal. Declaration admission and free-host checks use that total;
HBM accounting is unchanged. Each opted-in layer owns its own journal. Model-wide
automatic profiles do not enable exchanges or reserve journals; an explicit
integration must reserve their summed host cost before declaring enabled plans.
No shared-journal or shared-workspace saving is spent in admission.

For qualification geometry H=5120 and I=2304, one expert contains:

| Field | Bytes |
| --- | ---: |
| w13 | 11,796,480 |
| w2 | 5,898,240 |
| s13 | 737,280 |
| s2 | 368,640 |
| Total | 18,800,640 (17.9296875 MiB) |

One pair stages two payloads and writes two payloads: 75,202,560 copy-payload
bytes across sixteen field copies, plus map read/publication. This is API copy
volume, not measured C2C traffic; host-to-host mapped copies and memory-system
behavior affect actual traffic. With E=384, the one-pair journal costs 37,607,424
bytes per layer, about 1.401 GiB across 40 separately prepared layers. Keeping a
canonical Grace copy of 295 hot experts in each such layer would instead add
221,847,552,000 bytes (206.612 GiB). These geometries are examples, not limits.

For measured end-to-end copy bandwidth B, the bandwidth-only estimate is
`copy_bytes / B`; launch, synchronization and host-control time must be added.
No applicable B300 copy-bandwidth measurement exists in this evidence, so no
transfer-time or pause-duration estimate is claimed.

## Static profiles, counters and future policy

Static profiles still determine initial population, reproducible deployments and
fallback. The automatic controller still requires restart/reprepare to activate
a learned profile. It does not invoke slot exchange, mutate its saved profile or
silently treat runtime generation as a new profile artifact.

Existing optional routing counters continue counting canonical IDs. A future
control-plane policy can use their out-of-band snapshots to choose disjoint
pairs. No extra per-token observation mechanism is needed. Monitor stays opt-in,
reports routing estimates and performs no migration. Its measured portable
launch overhead remains relevant; it is not free telemetry.

The map answers where each logical expert resides independently of materialized
FC1/FC2, future grouped/persistent kernels or shared-token quantization. A future
backend must continue reading that map at execution time; baking rows into a
captured descriptor would require a different update contract. Concurrent spare
slot retirement, event-coordinated copies, demand-cache policy and adaptive
benefit remain research work requiring physical evidence.

## Validation and physical acceptance

`tests/moe/test_residency_updates.py` injects host copy and completion failures,
checks rollback/poisoning, stale generations, disjoint batches and admission.
`tests/moe/test_residency_updates_gpu.py` captures the production partitioner plus
a CuTe byte reader for all fields. It tests all-HBM, all-mapped-host and mixed
slabs, int32/int64 IDs, duplicates/sentinels, repeated exchanges, stream draining,
publication rollback, unchanged pointers, frozen resolution and zero allocator
counter changes during replay. The byte reader is a test oracle, not an SM120
fallback for the SM103 expert GEMM. Source-bound results and failures are in the
[ledger](expert-residency-ledger.md#quiescent-slot-exchange-evidence).

On physical B300, first run the static baseline, then the real expert-operator
same-graph comparison against a freshly prepared equivalent static placement:

```bash
python -m pytest tests/moe/test_sm103_residency.py \
  -k public_preparation_split_parity -q
python -m pytest tests/moe/test_sm103_residency.py \
  -k quiescent_slot_exchange -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/moe/test_sm103_residency.py -q
compute-sanitizer --tool synccheck --error-exitcode 91 \
  python -m pytest tests/moe/test_sm103_residency.py -q
nsys profile --trace=cuda,nvtx,osrt --output=slot-exchange \
  python -m pytest tests/moe/test_sm103_residency.py \
  -k quiescent_slot_exchange -q
```

The native tests cover all-HBM movement, all-Grace movement, cross-tier promotion
and eviction, repeated exchanges, changed activations/routes, exact parity,
unchanged addresses and binding release. They skip without physical SM103; cold
tests also require verified Grace coherency. Record source/package hashes, UUID,
driver, CUDA/Torch/CUTLASS/Triton versions and raw receipts as in the
[SM103 runbook](sm103-qualification.md). The Nsight command is a correctness trace,
not a clean latency benchmark: it includes reference preparation and assertions.

After those gates, separately measure copy completion, map publication, total
pause, C2C traffic and decode latency with the production geometry. Compare
static learned placement against periodic exchanges on deliberately changing
workloads only after a serving engine owns the pause and rank protocol. No
adaptive throughput benefit is established by compilation or byte-replay tests.
