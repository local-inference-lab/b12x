# SM103 serving integration and workspace ownership

Status: **control-plane adapter implemented; companion serving integration unsupported**. The
b12x [automatic residency API](expert-residency-automatic.md) is implemented and
portable-counter tested. This document identifies engine changes still required
to serve a checkpoint larger than HBM. It does not qualify SM103 execution or
install a serving option.

The [model-wide epoch adapter](expert-residency-epochs.md) uses supported vLLM
worker extensions and public AsyncLLM pause/RPC methods. It coordinates all
participating layers and replicated TP ranks, admits a bounded movement batch,
and requires coordinated reload after partial failure. The maintained loader
does not register prepared residency runtimes. Backend registration and
CPU-source loading remain prerequisites; the adapter alone is not a cache
serving option.

## Companion source boundaries

The retained `feat/b12x-sm103` companion at `f6c6ac72c3` uses `Caps`, public
warmup providers and the preceding b12x execution API. Its historical checkpoint
receipts remain tied to that source.

The maintained companion branch `fix/karmic-glm-memory-port`, inspected at
[`ef1aeaf080879865febd27a92c5644233d157987`](https://github.com/local-inference-lab/vllm/tree/ef1aeaf080879865febd27a92c5644233d157987),
already has these production integration points:

| Engine responsibility | Existing implementation | Residency requirement |
| --- | --- | --- |
| Preparation ownership | `vllm/model_executor/warmup/b12x_prepare.py`: `get_b12x_session`, `begin_b12x_preparation`, weights/state/bind stages and rank coordinator | Submit ordinary static expert plans and the optional counter request to this session. Preserve tuning/cache/priming contracts. |
| Loaded expert declaration | `vllm/model_executor/layers/fused_moe/b12x.py`: `_prepare_experts`, `process_weights_after_loading`, `get_b12x_preparation_units` | Retain source-native CPU weight/scale views until model-wide residency is determined; declare the hierarchical plan instead of preparing the ordinary representation first. |
| Runtime selected routes | The same expert provider's `apply` receives original `topk_ids` and weights and binds a retained plan | Bind the optional observer at this boundary with engine-supplied phase and valid rows. Cache-hit static serving has no observer. |
| Pause and quiescence | `vllm/v1/engine/core.py`: `pause_scheduler`, `_finish_pause`, `resume_scheduler`, `collective_rpc`; pause completion synchronizes workers | Complete the pause before counter RPCs and resume afterward. Honor runner/DP/async-output semantics. An in-process engine rejects `wait` mode, so it is not a universal drain primitive. |
| Warmup and graph construction | `vllm/v1/worker/gpu_worker.py`: preparation stages followed by `compile_or_warm_up_model` | Discard primer/capture counters and enable calibration after all warmup, before workload traffic. |

The epoch adapter selects the existing `keep` pause, preserving requests and KV
state. At the inspected revision, `AsyncLLM.pause_generation` adds a fixed 20 ms
delay after core pause completion. Complete control-plane receipts must include
it. No claim of a low-cost four-token serving epoch follows from offline
four-invocation policy windows.

Both `b12x_native_supported` and the MoE backend's `_supports_current_device`
currently admit the SM120 family on this companion branch. Extending those
checks alone would expose other components without establishing their SM103
contracts. Architecture admission must accompany the component port and its
validation.

The MXFP4 quantization loader allocates expert parameters during `create_weights`
and transforms them in `process_weights_after_loading`. The b12x provider then
calls ordinary `prepare_weights`. Hierarchical preparation instead requires
contiguous CPU checkpoint views to bound staging. A serving flag inserted only
after model loading cannot solve the greater-than-HBM loading requirement.
Copying the fully loaded GPU checkpoint back to CPU would retain that startup
peak and duplicate ownership.

The clean integration base is the maintained preparation branch. Port the
required SM103 component changes onto it; do not resurrect removed b12x APIs in
order to retain the older companion's warmup architecture. No companion source
was changed by this policy review.

## Reviewable integration sequence

1. **Checkpoint source ownership.** Add an explicit native-MXFP4 loader contract
   that keeps expert payloads/scales as CPU or mapped-file views while loading
   non-expert parameters normally. Preserve gate order, TP slicing, original
   expert IDs and all numerical metadata. Validate identity from the actual
   checkpoint/revision. EP, unsupported biases/activations and implicit
   requantization fail closed. Bound CPU staging and retain source owners until
   the preparation session has materialized both tiers.
2. **Model declaration and joint admission.** Build `ResidencyLayerSpec` from
   canonical weight plans, not model-name tests. Establish an explicit KV budget
   before spending the remaining HBM on experts; ordinary post-load free-memory
   profiling otherwise creates a circular dependency. Account for outstanding
   non-MoE weights, graphs, temporary source staging and private workspaces once.
   For replicated TP, agree on a budget all participating ranks can satisfy and
   distribute one placement decision. Do not let ranks independently select
   different profiles from changing filesystem indexes.
3. **Preparation and graph ownership.** Submit actual expert-operation primers
   and `ExpertResidencyWorker.preparation_request()` to the existing session.
   Freeze resolution before capturing. Retain CPU/slab/program owners for every
   graph that uses them. An observer is declared only for calibration/monitoring.
   Target, draft and verifier lanes have explicit phase labels; mixed batches
   require separate valid slices or are excluded. Padding rows must carry
   sentinel IDs or be excluded from the bound view. M alone cannot identify a
   phase. Prepared callbacks execute real operations, including on TP non-owners.
4. **Control plane.** Begin counting after warmup, then periodically pause
   submissions, await pause completion, and poll the authoritative TP owner.
   Supply engine-owned cumulative completed-request and processed-token totals.
   Snapshot/RPC/D2H work occurs at that boundary, never during graph replay.
   Broadcast diagnostics and the saved artifact out of band; no per-token
   collective is required. Repeated terminal polls are idempotent.
5. **Activation.** A converged auto result requests restart. A bounded experiment
   is saved without activation unless `activation="best_available"` is explicit.
   Keep the running placement fixed. The first integration should retain an
   operator-controlled restart boundary: stop admissions, drain/cancel according
   to engine policy, release graphs and prepared owners, then reload. Sleep/wake
   alone is not proof that every graph and custom allocator owner was released.
6. **Static restart.** All ranks validate the same artifact and prepare its
   immutable map. Auto with a compatible accepted artifact declares no counter
   node. Validate the checkpoint with layer/operator probes and deterministic
   requests before claiming serving correctness or quality.

This sequence permits a model-generic loader and worker integration. It also
keeps placement math, artifact policy and storage admission in b12x. A CLI can
expose those typed contracts once the loader and worker path exist. An automatic
engine restart is optional; a reliable explicit boundary is sufficient.

## Optional slot exchange boundary

The [quiescent exchange API](expert-residency-slots.md) supplies an independent
mechanism for plans that declare rollback capacity. It does not install an engine
pause or change automatic profile activation. An integration must stop all slab
users, coordinate the same transaction across TP ranks, retain all owners and
keep submissions stopped through successful commit or complete rollback. A
nonresumable error requires worker recovery; raw CUDA graphs cannot enforce the
Python health check. No cross-rank or cross-layer atomic commit is supplied.
Budget each layer's journal before preparation. Reusing a paused counter-polling
boundary is possible only after proving it excludes every graph submitter.

The [recent-frequency controller](expert-residency-cache.md) adds host-only
`observe` and `finish` hooks around this boundary. The engine supplies one
cumulative counter snapshot per model window, runs per-layer policies only on
the authoritative TP rank, coordinates the proposed exchanges, and acknowledges
the actual committed or restored snapshots before resuming. It must discard
policy baselines after counter reset/repreparation or unrelated placement
changes. A static lane without this opt-in declares no observer. This hook does
not supply scheduler integration or a serving throughput benchmark implicitly.

## Workspace ownership

Status: **design constraint; no shared arena implemented**. Every layer's private
workspace remains charged in full. The calculated 40-layer qualification saving
of 2,022,209,280 bytes is not spent in placement admission.

A reusable workspace belongs to a serialized **execution lane**, not to a model
or a request implicitly. A lane is an engine scheduling guarantee that its
operations cannot use the arena concurrently. The preparation session can own
the allocation and install fixed views, but it cannot infer that guarantee from
transformer layer order.

The ownership contract must establish all of these conditions before sharing:

- Expert slabs and per-layer route maps remain immutable plan-owned storage.
  Temporary route arrays, quantization buffers and intermediates can be arena
  views whose offsets/capacity are fixed before compilation/priming/capture.
- The arena belongs to one device and explicit lane. Independent serving lanes,
  overlapping speculative branches and independently submitted streams require
  separate arenas unless captured dependencies prove serialization. TP ranks
  own separate local arenas; no cross-rank scratch alias is implied.
- A stream orders sequential launches, but arbitrary graphs on different streams
  can overlap. Their submission owner must supply stream/event dependencies or
  separate storage. No host lock, dynamic lease allocation or wait is inserted
  into replay to repair an unspecified schedule.
- The default residency output currently resides in private workspace. Sharing
  requires caller-owned outputs whose lifetime exceeds downstream consumption,
  or a separately planned output allocation. Returning an arena view that the
  next layer overwrites is unsafe even if the kernels themselves are sequential.
- Preparation trials and real primers need valid private trial storage or the
  same declared exclusion guarantee. Trial cleanup must not release a serving
  arena or borrow it while another component is being timed.
- Every captured graph retains arena/program/slab owners. Releasing a plan or
  closing a session cannot free the arena while another bound graph references
  it. A restart invalidates all bindings before replacing backing addresses.

Qualification must include two overlapping streams/lanes, multiple captured
shapes, speculative branching, caller-output lifetime, stale bindings and
release ordering, together with no-allocation replay. Admission may replace the
sum of private scratch with the sum of per-lane maximum scratch only after these
contracts are implemented and tested. Route maps and independently live outputs
remain separately charged.

## Physical evidence and optimization decisions

The [physical operator harness](../benchmarks/moe/expert_residency.py) compares
hierarchical and all-HBM b12x on equivalent source bytes and routes, with
partition, q1, hot/cold FC1, q2, hot/cold FC2 and ordered-finalization samples.
Timing repetitions execute inside a CUDA graph to avoid Python enqueue gaps;
isolated-stage sums are not a substitute for complete-operator latency.
The harness still requires physical SM103, and does not contain a FlashInfer
comparison. Add that comparison only with the identical source layout, activation
recipe, route weights and finalization boundary; a faster but differently rounded
operator is a separate numerical mode.

Physical B300 priority is all-HBM native correctness, reduced all-Grace and mixed
correctness, Grace-backed TMA legality, then graph/sanitizer checks and stage
profiles. Record registers, CTAs/SM, occupancy, tensor/TMA utilization, HBM/C2C
traffic, launch gaps and power through the documented Nsight runs. Production
geometry and checkpoint serving follow those gates.

The portable data motivates these questions, without answering them for B300:

- An optional fused counter variant could remove the tiny-M observer launch.
  Compile separate profiled and unprofiled variants; keep off free of atomics.
  A separate variant must use the same epoch, sampling, overflow and TP semantics
  as the observer and be prepared before graph capture.
- Stable parallel partitioning may reduce the measured serial scan growth.
  Preserve original route indices, duplicates, sentinel handling, compact counts
  and the ordered finalizer. Its full-operator benefit must be measured.
- Token-shared FC1 quantization can avoid repeated token activations across
  routes, but its addressing and rounding must be checked against the qualified
  baseline before changing scratch or the numerical boundary.
- Sparse cold selections may justify conditional launches or persistent work.
  Existing guards skip inactive expert bodies; empty-grid scheduling cost remains
  a physical measurement question. No overlap or count-sized-grid benefit is
  claimed.

Monitor remains explicit, static and advisory. Its cold-selection estimates do
not forecast throughput. Rich traces retain unique-expert-per-call data for
future step-touch scoring; raw selection frequency remains the production
objective. A PR test workflow is a separate repository-infrastructure change;
local receipts and the release-wheel workflow do not constitute PR acceptance.
