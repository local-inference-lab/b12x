# SM103 features and preparation contracts

Status: **implemented prototype; physical SM103 execution unqualified**.
SM103 support uses the declaration, preparation-session, binding and execution
contracts described in [GPU preparation](gpu-profiles.md). Core compute uses
CuTe DSL. Supporting packing and metadata kernels may use Triton.

## Implemented features

| Feature | Implementation |
| --- | --- |
| Architecture admission and compilation | [Architecture descriptors](../b12x/_lib/architecture.py), component capability metadata and [compiler](../b12x/_lib/compiler.py) admit SM103 and retain architecture-specific artifact identity. |
| Quantized projections | [SM103 dense lowering](../b12x/gemm/_sm103_preparation.py) supplies NVFP4, MXFP4, MXFP8, both MXFP6 formats, W6A8 and ordinary tensor/block FP8 programs to preparation. [Packed linear adapters](../b12x/gemm/blockscaled/) preserve inline weight dequantization and bounded workspace. |
| Native MoE | [NVFP4](../b12x/moe/fused_moe/_sm103.py) and [Trellis](../b12x/moe/fused_moe/_sm103_trellis.py) retain CuTe routing, tcgen05/TMEM projections and weighted reduction. Canonical Trellis weights cover uniform, coupled, mixed-rate and grouped-atom representations. `BtxSource` and `BtxWeights` expose paired BTX records through the same public weight and execution plans. |
| Attention and indexing | [Sparse MLA](../b12x/attention/sparse_mla/_sm103.py), [compressed MLA](../b12x/attention/compressed_sparse_mla/_warp.py), dense MLA and DSA preserve their distinct cache layouts and fixed launch schedules. |
| Model support | CuTe KDA/GDN, three MTP feedback contracts, mHC, HyperConnection, vocabulary projection, block-FP8 linear and DeepSeek WO retain prepared programs and planned storage. |
| Storage and communication | [Engram storage](../b12x/sequence/engram/_storage.py) owns device or mapped-host allocations and checks Grace capability. Disk reads use the synchronous upstream transaction contract. Experimental Grace TP2 transport remains separate from model qualification. |
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

The native kernels retain explicit TMEM completion waits, X-axis routing grids,
compiled occupancy bounds, distinct Trellis input-scale halves and global
transform coordinates. W4A16 remains BF16 activations with inline FP4 weight
dequantization; it has no activation-scale multiplication.

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
