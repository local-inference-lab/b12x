# Model-wide residency epochs

Status: **implemented control-plane prototype with experimental SM120 serving
integration**. The shared coordinator and vLLM worker protocol support one
bounded residency epoch across multiple prepared MoE layers. Host tests cover
replicated-rank coordination and failure. Portable GPU tests cover two layers
sharing one captured graph across repeated exchanges.

The [prepared SM120 serving cache](expert-cache-serving.md) supplies a CPU-source
ModelOpt NVFP4 loader and canonical-fill backend through PreparationSession.
Its single-GPU vLLM experiments measure the complete pause/RPC path separately
from earlier contract and operator tests. SM103 serving and Grace-backed TMA
remain physically unqualified. Static preparation and its defaults are unchanged.

## Ownership and lifecycle

`b12x.moe.residency.ResidencyEpochCoordinator` owns a set of layer-scoped
`ResidencyCacheController` instances and a global promotion budget. It imports
no tensor library, backend or serving engine. The engine chooses when to run an
epoch and exclusively owns pause/resume. Backends own storage, copies, rollback
and prepared programs.

An epoch uses these boundaries:

1. Pause new graph submissions once for the model and drain producers.
2. Snapshot all layer maps and the authoritative rank's prepared counters.
3. Validate every layer before advancing any policy window.
4. Score per-layer proposals and admit one bounded model-wide subset.
5. Preflight that subset on every rank before any rank starts copying.
6. Apply each selected layer transaction using its retained backend.
7. Verify all completed rank maps, acknowledge layer policy outcomes and obtain
   every worker's acknowledgement.
8. Resume once for the model.

Transactions still synchronize inside each backend. One engine pause does not
mean one DMA transaction or one device synchronization. Layers publish their
maps sequentially while the engine remains paused; publication is not atomic
across the model. No reader may execute between those publications.

Any unknown, partial, timed-out or cancelled epoch fails closed. Even a
resumable single-layer rollback cannot prove that another layer or rank has
rolled back. The adapter never resumes in an error handler. The engine must
reload every participating rank, reconstruct graphs and establish fresh policy
baselines. If communication or pause/resume itself fails, external engine
recovery must establish the stopped state; a client-side flag cannot stop an
unreachable worker. There is no automatic distributed rollback or retry.

## Shared policy contracts

`ResidencyEpochBudget(max_pairs=..., max_copy_bytes=...)` bounds logical
promotions and successful API-copy bytes across the entire model. Bytes include
payload movement, journaling when required, and one map transaction per changed
layer, multiplied by the number of equal-cost replicas. Counter readback, RPC
serialization and rollback after failure are not part of the successful-copy
budget. The budget is not a latency prediction or a hard wall-time limit.

The coordinator ranks proposals by candidate-minus-victim score divided by
payload-copy bytes. Ties use layer name and canonical expert IDs. It charges map
bytes when the first pair from a layer is selected. Per-layer capacity,
admission thresholds and residence guards still apply. Skipped proposals are
reported. This deterministic ordering has no measured fairness or throughput
guarantee.

`ResidencyCacheConfig.scoring` supports:

| Value | Score | Status |
| --- | --- | --- |
| `recent_frequency` | Counts in the completed observation window | Existing default; experimental adaptation |
| `decayed_lfu` | `previous_score * decay + window_count` | Explicit experimental option; `decay=0.5` unless supplied |

Empty/unsampled windows do not decay history or age residency guards. Candidate
admission still requires current-window cold observations. Scores do not mix
layers, phases or independent DP traffic. Cumulative observations must belong
to one unchanged map generation. External swaps, counter resets and decreasing
counts require a fresh baseline.

True LRU remains in offline trace analysis: cumulative counters do not retain
the ordering of expert touches. Implementing “LRU” from these counts would
silently change the tested policy.

An observation window is the interval between completed control snapshots.
The adapter does not translate an offline window of four invocations into a
four-token engine pause. Nor can it reconstruct several four-step windows from
one aggregate counter delta. Shorter logical windows require an explicitly
prepared device window history or a cheaper engine boundary and separate
overhead measurement.

`ResidencyExchangeSpec.backing_mode` makes map transitions explicit:

- `exclusive`: each expert has exactly one tier row; a promotion swaps the two
  occupied physical locations.
- `canonical`: cold expert E resolves to backing row E; promotion installs E
  in the victim's resident row and maps the victim to its canonical backing row.

Resident rows remain dense and uniquely occupied. A declaration of canonical
backing does not implement its transport or establish source integrity. The
backend must retain verified canonical bytes and recover the victim after an
overwrite failure. Existing exclusive validation remains strict.

## Prepared observations

The existing CuTe routing counter plan remains the observer. Preparation owns
counter storage and retained programs. Graphs contain only device observation
work; no host policy or epoch method runs during capture/replay. Warmup/capture
observations are excluded by establishing a baseline before accepting requests.

A quiescent snapshot transfers the complete counter slab once, then decodes all
layer/phase rows on the host. Invalid IDs and duplicates retain the existing
counter semantics. The authoritative TP owner counts replicated routes once;
other ranks return no routing snapshot. Padding and decode/prefill/verify/draft
classification remain the engine's responsibility. M is not a phase label.
The SM120 V2 integration uses `RoutingProfileQuery.runtime_token_limit=True`
and a prepared device setter to exclude padding, dummy work and any batch
containing prefill. Static serving has neither the setter nor counter nodes.

Static serving must omit the observer at graph construction. Setting the
control adapter to `enabled=False` performs no engine or worker operations;
it cannot remove observer nodes that a caller already captured.

## vLLM boundary

[`residency_epoch.py`](../b12x/integration/vllm/residency_epoch.py) uses the
maintained engine's supported interfaces:

- `AsyncLLM.pause_generation(mode="keep", clear_cache=False)` preserves
  queued/in-flight requests and KV state while pausing producer work.
- `collective_rpc` invokes `begin`, `prepare`, `apply` and `acknowledge` on
  `ResidencyEpochWorkerExtension`.
- `resume_generation` runs only after successful all-rank acknowledgement.

The worker extension can be registered through vLLM's existing
`--worker-extension-cls b12x.integration.vllm.residency_epoch.ResidencyEpochWorkerExtension`.
This flag alone does **not** enable expert caching. The loader/preparation
integration must install `model_runner.b12x_residency_runtime`; a missing
runtime produces an explicit unsupported-backend error.

`ResidencyEpochRuntime` retains typed `ResidencyLayerBinding` objects. Each
binding provides a validated observation contract, exchange copy accounting,
prepared pair capacity, generation snapshot, batch apply callback, captured
addresses and owner-lifetime validation. Backend calls must commit exactly one
generation for a nonempty layer batch. Runtime checks reject stale decisions,
changed pointers, partial maps and released owners.

`bind_sm103_epoch_layer(plan, observations)` adapts an already prepared native
MXFP4 HBM/Grace plan with declared exchange journals. It does not compile,
reprepare, allocate expert storage or qualify Grace-backed TMA.
`bind_sm120_epoch_layer(plan, observations)` supplies the equivalent boundary
for a prepared canonical NVFP4/W4A16 cache. Its loader is single-rank; the
replicated-TP protocol remains contract-tested rather than serving-qualified.

The engine-side use after all workers are prepared is:

```python
from b12x.moe.residency import ResidencyCacheConfig, ResidencyEpochBudget
from b12x.integration.vllm.residency_epoch import VllmResidencyEpochs

epochs = VllmResidencyEpochs(
    async_llm,
    configs={name: ResidencyCacheConfig(
        max_pairs=layer_pair_capacity,
        minimum_cold_selections=minimum_observations,
        minimum_score_gain=score_margin,
        minimum_residency_windows=residence_windows,
        scoring="decayed_lfu",
        phase="decode",
    ) for name in participating_layers},
    budget=ResidencyEpochBudget(
        max_pairs=model_pair_limit,
        max_copy_bytes=model_copy_limit,
    ),
    tp_size=tp_size,
)
baseline_receipt = await epochs.run()  # After warmup, before admitting traffic.
# At an engine-selected control boundary, outside graph execution:
epoch_receipt = await epochs.run()
```

Configuration must fit each prepared backend's pair capacity. The adapter
supports one replicated TP group with equal shard copy accounting and DP=1.
Rank maps, geometry, checkpoint identity and initial profile identity must
agree; preparation identities remain rank-local. Nonuniform shards, EP and
multi-DP orchestration require a separate integration contract. The adapter
checks the actual engine configuration and rejects DP, PP, EP and context
parallelism as well as a mismatched TP size.
No per-token collective is introduced. One owner must serialize this adapter
with all other engine pause, sleep, weight-update and reload operations.

## Memory admission and loader gate

`ResidencyServingMemory` verifies an explicit per-rank envelope before runtime
registration. It accounts for resident expert payload, full backing storage,
dense/shared device storage, KV, private workspace, graphs, metadata, host
staging/journals, other allocations and both safety reserves. Values are
supplied by the loader/backend; this class does not infer live memory or prove
that the loader avoided a GPU-first peak. Do not combine free-memory envelopes
with reservations already charged to those envelopes.

For the inspected Qwen3.8 checkpoint, the SM120 research representation costs
2,764,808 bytes per expert. Across 48 layers and 512 experts:

| Payload allocation | Bytes | GiB |
| --- | ---: | ---: |
| Complete canonical backing | 67,947,921,408 | 63.281 |
| 256 resident experts per layer | 33,973,960,704 | 31.641 |
| 128 resident experts per layer | 16,986,980,352 | 15.820 |

These are payload arithmetic, not complete-model fit results. Safetensors
headers independently report 69,363,898,368 routed-expert bytes and
36,435,075,496 other checkpoint tensor bytes. Serialized sizes are not runtime
allocation sizes. The latter include large non-MoE storage and require the
engine's actual placement rules. Alignment, graph capacity, KV, private scratch,
source retention, pinning and TP sharding remain additional constraints.

Host envelopes must be apportioned across ranks sharing one physical host.
Passing the entire available host pool independently to each rank is invalid.
The coordinator checks supplied rank envelopes. The opt-in
[SM120 serving loader](expert-cache-serving.md) admits CPU sources and canonical
mapped backing for a single rank before cache allocation. Distributed source
ownership and physical-pool apportionment remain unsupported by that loader.
Its explicit W4A16 numerical recipe is distinct from an A4 model execution
contract. Native SM103 loading remains an independent integration requirement.

## Diagnostics and serving experiment

Each control call returns a JSON-compatible receipt with all worker snapshots,
selected/skipped pairs, copy accounting, per-layer outcomes and generations.
Measured host durations include ownership acquisition, pause/drain, snapshot
RPC, policy, preflight RPC, complete transaction RPC, acknowledgement RPC,
resume, bookkeeping and total call time. Worker receipts include snapshot and
per-layer transaction wall time. Failed receipts remain in `last_receipt`.

These aggregate durations do not separately isolate scheduler queue time,
D2H transfer, publication DMA or rank communication. Backend events and engine
timeline instrumentation remain necessary for that decomposition. In the
inspected vLLM AsyncLLM, the public pause method includes a fixed 20 ms sleep;
using this adapter includes that delay. No shorter-pause performance is claimed.

The [SM120 serving harness](../benchmarks/moe/expert_cache_serving.py) uses the
CPU loader and prepared canonical backend. Its primary arms share checkpoint,
learned profile, resident capacity, graph geometry, actual route weights,
prompts and concurrency. It compares uninstrumented learned static placement
against explicit decayed LFU, including stable traffic, workload transitions,
TTFT, delivery-gap distributions, throughput and complete pause cost. The
[serving guide](expert-cache-serving.md) documents its single-rank scope and
source-bound results. Single-layer and portable byte-reader evidence remain
separate from those serving measurements.

Physical SM103 acceptance remains ordered: native all-HBM, all-Grace, mixed
correctness and Grace TMA legality; then same-graph exchange, sanitizers, miss
cost and model-wide serving. No asynchronous replacement, spare-slot scheme,
new production policy default or shared scratch arena is implemented here.

## Validation commands

Host policy and worker protocol:

```bash
python -m pytest -q tests/moe/test_residency_epoch.py \
  tests/moe/test_vllm_residency_epoch.py tests/moe/test_shared_residency.py \
  tests/moe/test_residency_cache.py tests/moe/test_residency_replay.py \
  tests/moe/test_automatic_residency.py tests/moe/test_residency_updates.py \
  tests/moe/test_expert_residency.py
```

Portable Blackwell graph and counter checks:

```bash
python -m pytest -q tests/moe/test_residency_epoch_gpu.py \
  tests/moe/test_residency_cache_gpu.py tests/moe/test_residency_kernels.py \
  tests/moe/test_routing_profile_gpu.py
```

The two-layer epoch test reads actual tier payloads using portable byte-reader
kernels. Full cases check every payload byte over 16 observation windows;
bounded cases check scale bytes over two windows for sanitizer practicality.
Both retain the same captured graph, real partition/counter programs, real
mapped-host transactions and exact pointer/allocation assertions. Neither
executes the SM103 expert GEMMs or a vLLM model.

```bash
timeout 120s compute-sanitizer --tool memcheck --error-exitcode 86 \
  python -m pytest -q tests/moe/test_residency_epoch_gpu.py -k bounded-dtype1
timeout 120s compute-sanitizer --tool synccheck --error-exitcode 86 \
  python -m pytest -q tests/moe/test_residency_epoch_gpu.py -k bounded-dtype1
python scripts/compile_routing_profile.py --experts 512 --capacity 128 \
  --top-k 10 --sample-every 1 128 --output-dir /fresh/evidence/sm103-counters
```

These explicit geometries select qualification cases; no public epoch or policy
contract requires those expert/token counts. Full sanitizer coverage remains a
separate gate; bounded success does not erase a full-suite timeout.
