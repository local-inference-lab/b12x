# SM103 features and preparation contracts

Status: **implemented prototype; physical SM103 execution unqualified**.
SM103 support uses the declaration, preparation-session, binding and execution
contracts described in [GPU preparation](gpu-profiles.md). Core compute uses
CuTe DSL. Supporting packing and metadata kernels may use Triton.
The working branch is based on master `8783519a3e42c22c0f395669ca4b20c69439c974`. The
[readiness report](sm103-readiness-report.md) separates source-bound validation,
pending PR CI and physical-target qualification.

## Implemented features

| Feature | Implementation |
| --- | --- |
| Architecture admission and compilation | [Architecture descriptors](../b12x/_lib/architecture.py), component capability metadata and [compiler](../b12x/_lib/compiler.py) admit SM103 and retain architecture-specific artifact identity. |
| Quantized projections | [SM103 dense lowering](../b12x/gemm/_sm103_preparation.py) supplies NVFP4, MXFP4, MXFP8, both MXFP6 formats, W6A8 and ordinary tensor/block FP8 programs to preparation. [Packed linear adapters](../b12x/gemm/blockscaled/) preserve inline weight dequantization and bounded workspace. |
| Native MoE | [NVFP4](../b12x/moe/fused_moe/_sm103.py) and [Trellis](../b12x/moe/fused_moe/_sm103_trellis.py) retain CuTe routing, tcgen05/TMEM projections and weighted reduction. Canonical Trellis weights cover uniform, coupled, mixed-rate and grouped-atom representations. `BtxSource` and `BtxWeights` expose paired BTX records through the same public weight and execution plans. |
| Hierarchical MXFP4 expert residency | [Placement contracts](../b12x/moe/fused_moe/residency.py) and [preparation](../b12x/moe/fused_moe/_residency_preparation.py) add per-layer HBM/Grace profiles, checkpoint identity, memory budgets and owned slabs through the public `fused_moe` API. [Native CuTe kernels](../b12x/moe/_shared/kernels/sm103/residency.py) provide MXFP8/MXFP4 tcgen05 projections, compact tier-local routing and one ordered FP32 FMA finalizer over unweighted BF16 expert outputs. Checkpoint weights are not requantized. Physical SM103 execution and Grace-backed TMA remain unqualified. |
| Automatic residency orchestration | [Typed controller and profile store](../b12x/moe/fused_moe/automatic.py) validate workload/checkpoint identity, budget the model once, derive per-layer hot membership, detect convergence/drift and report controlled restart. Source geometry and numerical contracts remain authoritative. |
| Prepared routing counters | [CuTe counters](../b12x/moe/_shared/kernels/routing_profile.py) use preparation-owned uint64 storage, sampling and overflow detection. The disabled serving path has no observer. [Worker hooks](../b12x/integration/vllm/expert_residency.py) expose lifecycle operations without patching vLLM. |
| Attention and indexing | [Sparse MLA](../b12x/attention/sparse_mla/_sm103.py), [compressed MLA](../b12x/attention/compressed_sparse_mla/_warp.py), dense MLA and DSA preserve their distinct cache layouts and fixed launch schedules. |
| Model support | CuTe KDA/GDN, three MTP feedback contracts, mHC, HyperConnection, vocabulary projection, block-FP8 linear and DeepSeek WO retain prepared programs and planned storage. |
| Storage and communication | [Engram storage](../b12x/sequence/engram/_storage.py) owns device or mapped-host allocations and checks Grace capability. Disk reads use the synchronous upstream transaction contract. Experimental Grace TP2 transport remains separate from model qualification. |
| Static placement profiles and operator qualification | [Offline profiling](../scripts/build_expert_residency_profile.py) ranks expert selections independently per layer and emits versioned workload artifacts with expected cold fractions. The [residency benchmark](../benchmarks/moe/expert_residency.py) requires physical SM103 and records all-HBM parity, graph invariants, total latency and isolated per-stage samples. |
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
| Separate tier finalization changes rounding | Expert outputs remain unweighted until one explicit `fma.rn.f32` reduction, with a single final BF16 cast. Arithmetic adversaries distinguish this contract from reordered sums and separately rounded tiers. |
| Model-scale cold storage and workspace require explicit admission | HBM and exact-size mapped-host slabs include aligned weights/scales; accounting includes scratch, route maps, KV reservation and safety margins. Preparation verifies Grace coherency and matching checkpoint/layer identity. |
| Residency binding must not trigger lazy preparation or retain stale execution | Bind/run require explicit session preparation and reject released or replaced state. Retained programs and fixed workspace serve changing live M/top-k without runtime resolution. |
| A global HBM budget cannot be reused independently by every layer | The model planner charges KV, safety, counter storage and each private workspace once, then passes exact layer budgets to ordinary residency preparation. |
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
router-weight-on-input, logits routing, expert parallelism and online adaptation
are unsupported in that variant. Compact-count guards skip inactive expert work;
mixed placements still launch both tiers. No overlap benefit or B300 performance
is claimed.

## Validation and PR acceptance

The [readiness report](sm103-readiness-report.md) gives the validation totals for
the automation source. The [engineering ledger](expert-residency-ledger.md)
separates that evidence from the static-residency baseline at `78a8704f` and
retains failed runs. Counter overhead measurements on SM120 describe only the
partition/counter stage; they establish no B300 or whole-model throughput.

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
The companion must adopt these preparation contracts before serving validation.

The [SM103 automatic residency guide](expert-residency-automatic.md) specifies the
worker control-plane and routing hooks. They are implemented b12x APIs, not an
installed vLLM serving feature. Engine configuration, phase classification,
quiescent polling, rank coordination and restart still belong to the integration.
The hooks do not port the older companion branch implicitly.
