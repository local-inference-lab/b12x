# SM103 features and preparation contracts

Status: **implemented prototype; physical SM103 execution unqualified**.
SM103 support uses the declaration, preparation-session, binding and execution
contracts described in [GPU preparation](gpu-profiles.md). Core compute uses
CuTe DSL. Supporting packing and metadata kernels may use Triton.
The working branch is based on master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. The
[readiness report](sm103-readiness-report.md) separates source-bound validation,
pending PR CI and physical-target qualification.
The [cadence qualification](expert-cache-cadence.md) adds bounded execution
tracing, controlled-admission comparisons, an isolated router replay and
checkpoint-byte whole-K shape checks. It explains the C4 generation-zero
divergence without changing cache arithmetic, policy defaults or the companion
engine. SM103 preparation remains the same 86-declaration/244-program inventory
at its retained compiler source; physical B300 gates remain open.
Its controlled serving matrix retains exact output equality at C1/C4/C8 with
the complete source-built engine. Healthy backoff trades fewer stable checks
for delayed transition response; it remains experimental.
The [SM120 cache proof of concept](expert-residency-sm120-poc.md) reuses the
shared host policy and exchange transaction with native NVFP4 W4A16 operations.
It adds no SM103 kernel, component registration or production serving lane.
The [SM120 cost diagnosis](expert-residency-sm120-costs.md) restricts shared
W4A16 lookup staging to Trellis payloads and adds an explicit cacheable option
to mapped-host allocation. Existing SM103 storage defaults remain unchanged.

The [routing-locality and canonical-fill experiment](expert-cache-evolution.md)
adds held-out real-route analysis and a recoverable SM120 fill prototype at
benchmark scope. It leaves shared policy and static SM103 preparation unchanged.
Cheaper PCIe fills do not establish profitable adaptation or Grace-backed TMA
legality; physical B300 qualification remains required.

The [held-out policy evaluation](expert-cache-policy-evaluation.md) compares
offline replacement policies and native SM120 replay beyond layer zero. It adds
benchmark selection and validation only; public residency contracts, serving
defaults and SM103 program counts remain unchanged.

The [prepared SM120 serving cache](expert-cache-serving.md) adds CPU checkpoint
source ownership, a registered canonical-host NVFP4/W4A16 preparation backend,
and an opt-in maintained-vLLM loader. All MoE layers share one bounded engine
epoch; static serving has no observer. SM103 storage, numerical contracts and
physical gates remain distinct. The complete source-bound SM103 compiler corpus
contains 86 declarations and 244 programs, including 238 native CuTe exports.
The [ledger](expert-residency-ledger.md) binds those counts to the tested source.

The [scheduler-maintenance control](expert-cache-maintenance.md) adds an opt-in
single-rank engine boundary, worker-local model-wide policy and explicit cold
pressure gating. It preserves the scheduler/device barrier without the public
administrative pause's output-settling delay. Static serving declares no
observer or maintenance task. The same source fixes repeated expert selections
in the native one-token packed W4A16 path; only top-1 can assume one row per
expert block. The SM103 corpus remains 86 declarations/244 programs at package
`667fad34f8bb0ba5ad048154e2cfd6281ebe206a19b1b8e32a69fd34e89e0d6b`.

## Implemented features

| Feature | Implementation |
| --- | --- |
| Architecture admission and compilation | [Architecture descriptors](../b12x/_lib/architecture.py), component capability metadata and [compiler](../b12x/_lib/compiler.py) admit SM103 and retain architecture-specific artifact identity. |
| Quantized projections | [SM103 dense lowering](../b12x/gemm/_sm103_preparation.py) supplies NVFP4, MXFP4, MXFP8, both MXFP6 formats, W6A8 and ordinary tensor/block FP8 programs to preparation. [Packed linear adapters](../b12x/gemm/blockscaled/) preserve inline weight dequantization and bounded workspace. |
| Native MoE | [NVFP4](../b12x/moe/fused_moe/_sm103.py) and [Trellis](../b12x/moe/fused_moe/_sm103_trellis.py) retain CuTe routing, tcgen05/TMEM projections and weighted reduction. Canonical Trellis weights cover uniform, coupled, mixed-rate and grouped-atom representations. `BtxSource` and `BtxWeights` expose paired BTX records through the same public weight and execution plans. |
| Hierarchical MXFP4 expert residency | [Placement contracts](../b12x/moe/fused_moe/residency.py) and [preparation](../b12x/moe/fused_moe/_residency_preparation.py) add per-layer HBM/Grace profiles, checkpoint identity, memory budgets and owned slabs through the public `fused_moe` API. [Native CuTe kernels](../b12x/moe/_shared/kernels/sm103/residency.py) provide MXFP8/MXFP4 tcgen05 projections, compact tier-local routing and one ordered FP32 FMA finalizer over unweighted BF16 expert outputs. Checkpoint weights are not requantized. Physical SM103 execution and Grace-backed TMA remain unqualified. |
| Quiescent fixed-slot exchange | [Journaled exchange](../b12x/moe/fused_moe/_residency_updates.py) preserves canonical expert IDs and captured slab/map addresses. Preparation reserves bounded rollback storage; the engine owns the pause. Batch commit, rollback, poisoning and stale-generation guards leave static defaults unchanged. The [slot guide](expert-residency-slots.md) gives the synchronization contract and physical gates. |
| Automatic residency orchestration | [Typed controller and profile store](../b12x/moe/fused_moe/automatic.py) validate workload/checkpoint identity, budget the model once, derive per-layer hot membership with joint HBM/Grace admission, balance bootstrap coverage, detect convergence/drift and separate saved experiments from accepted restart candidates. Source geometry and numerical contracts remain authoritative. |
| Grace-served cache policy | The [host controller](../b12x/moe/fused_moe/residency_cache.py) classifies counter deltas against one placement generation, proposes frequency-ranked pairs with explicit safeguards and acknowledges committed/restored snapshots. Existing Grace execution serves observed misses; exchange changes later accesses. The [policy guide](expert-residency-cache.md) defines integration and measurement limits. |
| Prepared routing counters | [CuTe counters](../b12x/moe/_shared/kernels/routing_profile.py) use preparation-owned uint64 storage, sampling and overflow detection. The disabled serving path has no observer. [Worker hooks](../b12x/integration/vllm/expert_residency.py) expose lifecycle operations without patching vLLM. |
| Attention and indexing | [Sparse MLA](../b12x/attention/sparse_mla/_sm103.py), [compressed MLA](../b12x/attention/compressed_sparse_mla/_warp.py), dense MLA and DSA preserve their distinct cache layouts and fixed launch schedules. |
| Model support | CuTe KDA/GDN, three MTP feedback contracts, mHC, HyperConnection, vocabulary projection, block-FP8 linear and DeepSeek WO retain prepared programs and planned storage. |
| Storage and communication | [Engram storage](../b12x/sequence/engram/_storage.py) owns device or mapped-host allocations and checks Grace capability. Disk reads use the synchronous upstream transaction contract. Experimental Grace TP2 transport remains separate from model qualification. |
| Static placement profiles and operator qualification | [Offline profiling](../scripts/build_expert_residency_profile.py) ranks expert selections independently per layer and emits versioned workload artifacts with expected cold fractions. The [residency benchmark](../benchmarks/moe/expert_residency.py) requires physical SM103 and records all-HBM parity, graph invariants, total latency and isolated per-stage samples. |
| Shared expert residency contracts | [`b12x.moe.residency`](../b12x/moe/residency/__init__.py) owns canonical placement, counter/generation snapshots and host recent-frequency policy. The existing fused-MoE constructor adapts native geometry and copy accounting; fixed-address storage/execution remains preparation-owned. |
| Model-wide residency epochs | [Bounded coordination](expert-residency-epochs.md) retains layer-scoped policies, admits global pair/copy-byte limits, supports explicit decayed LFU and checks complete generation acknowledgement. A vLLM worker-extension adapter performs one pause and all-rank preflight; the SM120 canonical backend and CPU loader register this boundary. Distributed loading and native SM103 serving qualification remain required. |
| Reproducible validation | [Preparation compiler](../scripts/compile_sm103_prepared.py), [kernel corpus compiler](../scripts/compile_sm103.py), resource auditors and the [qualification launcher](../scripts/qualify_sm103.py) preserve source and artifact identity. |

## Fixes required by preparation and execution

| Contract or failure | Correction |
| --- | --- |
| Declaration construction must not allocate or compile | Typed queries/configs and metadata-only compile factories replace component-local policy and warmup registries. Materialization owns storage and retains the exact programs declared to preparation. |
| Live requests must not create specializations | Capacity and immutable geometry determine programs; runtime counts drive grids, masks and views. Tests freeze kernel resolution while varying live counts, including zero. |
| SM103 dense plans must retain their architecture lowering | Dense, block-FP8 and WO factories explicitly carry target metadata into offline extraction. FP8 workspace queries distinguish output ownership, workspace ownership and recipe. |
| CUDA graph and full-graph tracing must preserve mutation | Sparse/compressed MLA, block-FP8 linear, workspace FP8 and concatenated/FP8 MTP execute through typed opaque boundaries. Mutable output/scratch are explicit; owned functional storage has a separate operator boundary. |
| Explicit-stream execution must include surrounding work | Block-FP8 quantization, output allocation, projection and bias execute on the requested stream. |
| Empty requests must not launch zero-sized CUDA grids | Compressed MLA and concatenated/FP8 MTP return their empty output views before launching. |
| FP16 split reductions must preserve output type | BF16 atomics are restricted to BF16; FP16 uses typed CuTe reduction of FP32 partials. |
| Attention sink must contribute to decode normalization | Sparse MLA merges the sink into both output normalization and returned LSE. |
| DSA compilation must replace deferred discovery kernels | The MXFP4 compiler uses the preparation-aware program cache so in-process compilation evicts placeholders and emits every declared native artifact. |
| DSA score-output presence is immutable | Execution validates the declared output contract; indices-only and score-producing plans retain their respective programs. |
| Mixed-rate Trellis cache hits must retain declared programs | Both mixed launch constructors and copied cache carriers attach their compiled kernels and weighted-reduction programs. |
| Packed FP16 inputs require a distinct precision contract | MXFP8 queries retain input dtype, select quantized execution and reject BF16-only A16. Output/workspace ownership remains explicit, including empty requests. |
| Full-rotation Trellis launch records carry broadcast metadata | The runtime selector unpacks and matches the retained broadcast field before launching the weighted reduction. |
| Large pools and vocabularies exceed 32-bit offsets | Scaled page, state and vocabulary-row addresses use Int64. GPU tests park live data beyond the signed 32-bit offset boundary. |
| Artifact loading can modify an ELF | Verified temporary copies protect manifest-bound cached objects while preserving raw hashes and launch-resource checks. |
| Dense MXFP4 support does not provide routed MXFP4 execution | A source-native A8/MXFP4 preparation contract and mixed FP8/FP4 tcgen05 backend supply the hierarchical routed path without changing SM120/SM121 dispatch. |
| Tier-local storage rows differ from original route ranks | One CuTe partition pass retains both local expert rows and original route indices. Projection addressing uses the original route row; finalization consumes original top-k order. Invalid int64 IDs are checked before narrowing. |
| A stable pointer does not make in-place payload replacement safe | Exchanges require paused producers, device draining, complete payload journaling and copy completion before publishing the map. Failed publication restores both payloads and map; failed rollback keeps the lane stopped. |
| Cumulative counts cannot identify misses after an unrelated placement change | Cache windows bind counter deltas to one preparation/generation; external exchanges and reset/regressing counters invalidate the baseline. No new kernel or static-path observer is added. |
| Per-layer control loops multiply engine pauses | A model coordinator selects one bounded batch. Worker preflight validates every layer before any rank writes. Partial layer/rank failure keeps the control loop failed until coordinated reload. |
| Counter snapshots perform one D2H operation per layer | The quiescent snapshot copies the prepared counter slab once and parses its layer/phase views on the host. No replay operation or compiled counter program changes. |
| Separate tier finalization changes rounding | Expert outputs remain unweighted until one explicit `fma.rn.f32` reduction, with a single final BF16 cast. Arithmetic adversaries distinguish this contract from reordered sums and separately rounded tiers. |
| Model-scale cold storage and workspace require explicit admission | HBM and exact-size mapped-host slabs include aligned weights/scales; accounting includes scratch, route maps, KV reservation and safety margins. Preparation verifies Grace coherency and matching checkpoint/layer identity. |
| Residency binding must not trigger lazy preparation or retain stale execution | Bind/run require explicit session preparation and reject released or replaced state. Retained programs and fixed workspace serve changing live M/top-k without runtime resolution. |
| Equal synthetic scores can concentrate bootstrap HBM in early layers | An observation-free bootstrap prioritizes equal resident fractions; learned selection-density placement remains unconstrained by that prior. |
| A hard calibration limit is not evidence of convergence | `activation="converged"` admits converged profiles by default. Explicit `best_available` accepts sufficiently sampled limit artifacts. Separate latest/converged cache indexes preserve an accepted profile when bounded experiments are saved. |
| Greedy density packing can strand HBM and exceed Grace | A bounded exact subset-sum fallback repairs joint byte feasibility under per-layer bounds. Search-size exhaustion reports unknown feasibility, not an impossible budget. |
| Python replay loops contaminate tiny-stage timings | Physical operator timing graphs contain repeated device operations; correctness replay remains a separate gate. |
| Master fixes 24-head sparse MLA shards | Rebase preserves the 16+8 TP3 partition and exact stable QSA winner construction; portable attention checks cover the rebased source. |
| A global HBM budget cannot be reused independently by every layer | The model planner charges KV, safety, counter storage and each private workspace once, then passes exact layer budgets to ordinary residency preparation. |
| A partial rank commit is not a safe model-wide rollback | The epoch driver checks all-rank readiness and completion, never resumes from an error handler, and requires coordinated reload after an unknown or partial update. Actual DP/PP/EP/context-parallel configurations are rejected by the prototype. |
| A profile for another geometry or traffic class is unsafe to reuse | Schema-2 artifacts bind model/recipe/capacity and workload; startup validates hash, implementation version, hardware and memory fit. Auto invalidates incompatible artifacts with a reason. |
| Cumulative history can hide a changed routing distribution | Convergence uses disjoint windows, per-layer membership and cold-rate stability, minimum coverage and agreement with the saved candidate. Monitor reports drift without modifying storage. |
| Master includes pooled selection alongside sparse attention | The sparse-MLA tuning contract preserves native pooled-selection eligibility. The cached-restart ownership test enables tuning selection, preserving the separate disabled-autotuning default contract. |

The native kernels retain explicit TMEM completion waits, X-axis routing grids,
compiled occupancy bounds, distinct Trellis input-scale halves and global
transform coordinates. W4A16 remains BF16 activations with inline FP4 weight
dequantization; it has no activation-scale multiplication.

The [residency API guide](expert-residency.md) specifies supported formats and
ownership. The hierarchical variant supports BF16 I/O, native MXFP4/E8M0 K32
weights, A8 activations and SiLU with an optional clamp. Biases, SITU, FP16 output,
router-weight-on-input, logits routing, expert parallelism and concurrent adaptation
are unsupported in that variant. Compact-count guards skip inactive expert work;
mixed placements still launch both tiers. No overlap benefit or B300 performance
is claimed.

## Validation and PR acceptance

The scheduler-maintenance source passes 86 focused host tests with six GPU-only
skips and the complete offline SM103 corpus of 86 declarations / 244 programs.
The [maintenance guide](expert-cache-maintenance.md) and
[engineering ledger](expert-residency-ledger.md#scheduler-owned-maintenance-and-serving-control-costs)
separate its physical SM120 serving evidence and engine-build qualification.

The [readiness report](sm103-readiness-report.md) also retains historical
shared-residency extraction totals: 1,052 host passes, 22 portable GPU passes and
three focused SM103 declarations covering 14 programs. The full 85-declaration,
241-program receipt remains tied to `94639562`. Physical-only skips remain separate. The [engineering ledger](expert-residency-ledger.md)
separates that evidence from the static-residency baseline at `78a8704f` and
retains failed runs. Counter overhead measurements on SM120 describe only the
partition/counter stage; they establish no B300 or whole-model throughput.

The separate repository registry suite has five MXFP6 metadata failures that
also reproduce at untouched `181e234b`; that gate remains unresolved. Shared
namespace import isolation and artifact compatibility pass.

Validation remains source-bound local evidence. The wheel-release workflow has
no pull-request trigger, so PR test CI must be enabled and pass independently.
Physical SM103 execution and checkpoint qualification remain separate gates.

## Integration compatibility

Consumers declare component queries or Caps, submit real preparation callbacks
to `PreparationSession`, then bind and execute with the same `Plan`. The removed
`b12x.policy` API, component `prewarm` calls and executable fields on public plans
are not compatibility interfaces. Materialized state belongs to preparation.

The companion vLLM branch at `f6c6ac72c3` targets the preceding b12x API. Its GLM
pooling and loader fixes remain recorded in
[historical evidence](sm103-glm-sparse-validation.json); that receipt does not
qualify this preparation port or establish companion API compatibility.
The maintained companion preparation base at `ef1aeaf080` uses
`PreparationSession`. The `codex/b12x-expert-cache` integration adds CPU expert
loading, model-wide admission, explicit decode phase and epoch registration for
single-rank SM120. Native SM103 loading remains deferred. The
[integration and workspace audit](expert-residency-integration.md) distinguishes
these transports and the older companion's API port.

The [SM103 automatic residency guide](expert-residency-automatic.md) specifies the
worker control-plane and routing hooks. Automatic SM103 calibration/activation
still requires engine integration. The opt-in SM120 companion uses explicit
calibration and pinned learned profiles instead of silently installing that
automatic lifecycle. Engine configuration, phase classification, quiescent
polling, rank coordination and restart remain integration responsibilities.

The [shared residency guide](expert-residency-subsystem.md) defines backend and
engine responsibilities. Existing fused-MoE imports and serialized placement
profiles remain compatible; a reusable host policy does not imply execution
support on another platform.
