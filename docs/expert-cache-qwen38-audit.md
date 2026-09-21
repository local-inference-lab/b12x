# Qwen3.8-Flash-Next checkpoint audit

Status: **metadata audited; expert-cache serving unsupported**. Keep
[Qwen3-Next-80B](expert-cache-large-model.md) as the immediate full-model target.
Qwen3.8 is a flagship follow-on. Its main experts match the cache recipe, but
mixed-precision dispatch, PLE storage selection, combined admission and PLE
teardown need integration before a complete checkpoint download is justified.
No full checkpoint download or Qwen3.8 serving benchmark is part of this audit.

## Sources and scope

The audited NVIDIA revision is
[`fc694b54fb0174e0913e6adf86691ef85a4ead47`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47).
Configuration, generation configuration, quantization configuration, model card,
index, all eleven safetensors headers and bounded global-scale ranges are
retained. Header/index coverage includes **299,545 tensors**. This is an inventory
audit, not proof that a running loader consumes every tensor or that packed
weights pass an arithmetic oracle.

The source baseline is b12x `1c20f10bc66e65b7e5cde0124e7d7b416b0b6251` and companion
vLLM `019b9df5cbd9122d6e9670485879a8108a65ee8b`. Both working branches were clean.
Live default heads were b12x `f20ab3bad7def65f6211403fb73f9b1e8a33dfb0` and companion
`47ccf6c57d92f03630ebcbad3809450545825488`; neither was merged. Upstream vLLM was
`54020c3c3ec9de219929a5a956010a0251b87520` when inspected. No companion source changes
or wheel rebuild are required for this metadata-only extension.

The similarly named local `/models/Qwen3.8-Flash-Next-NVFP4` is a **different QAD
export**, with 36 shards and NVFP4 PLE. Its cached revision is
`629bc3218833a38b475b719f34aa571666f4a03e`. It is not the NVIDIA eleven-shard,
FP8-PLE snapshot and cannot qualify this audit's candidate. It remains untouched.

## Storage inventory

Counts below come from tensor extents, not repository size or parameter labels.
Main-model routed weights use NVFP4/K16. Shared experts, attention and residual
mixers are BF16. PLE uses FP8 with a BF16 scalar; optional MTP uses block FP8 and
BF16. Every class has required shape/dtype validation in
`b12x/integration/vllm/checkpoint_qwen4.py`.

| Class | Exact tensor bytes | GiB |
|---|---:|---:|
| Main routed packed weights | 60,397,977,600 | 56.250 |
| Routed block scales | 7,549,747,200 | 7.031 |
| Routed global and input scales | 589,824 | 0.001 |
| Shared experts and shared gates | 472,104,960 | 0.440 |
| Routers | 125,829,120 | 0.117 |
| Gated DeltaNet | 4,173,020,928 | 3.886 |
| Qwen Sparse Attention and indexers | 1,234,716,672 | 1.150 |
| Gated residual/hyper-connection weights | 1,281,249,280 | 1.193 |
| Embedding and output head | 2,542,796,800 | 2.368 |
| PLE embedding table | 51,200,245,760 | 47.684 |
| PLE projections, convolution, norms and metadata | 65,679,642 | 0.061 |
| Vision | 897,862,112 | 0.836 |
| Optional MTP, exclusive stored tensors | 2,698,026,496 | 2.513 |
| **Main model without MTP, including vision** | **129,941,819,898** | **121.018** |
| **Text-only main model without MTP/vision** | **129,043,957,786** | **120.182** |
| **Complete checkpoint tensor payload** | **132,639,846,394** | **123.530** |

The text-only total includes PLE. Vision is omitted only with the engine's
explicit `language_model_only` path, which constructs a missing vision stage and
excludes its weights. Sending text requests alone does not omit vision loading.
MTP has an independent excluded prefix in target-model loading. It is not
required for valid autoregressive target execution.

The model card's approximate 4B MTP label is not an exclusive memory inventory:
the checkpoint contains 2,607,150,848 MTP parameter elements excluding scales,
and the engine reuses target embeddings/head. Shared target weights must not be
counted twice. The 51B PLE claim matches the 51,200,245,760 table elements.

## Comparison with the 80B control

| Contract | Qwen3-Next-80B | Qwen3.8-Flash-Next |
|---|---|---|
| Pinned revision | `8fb2682f136cf94d932a498f18cb1e428832a912` | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Routed geometry | 48 layers, 512 experts, H2048/I512, top-10 | 48 layers, 512 experts, H2560/I640, top-10 |
| Packed routed bytes | 38,654,705,664 | 60,397,977,600 |
| Routed bytes including scales | 43,487,133,696 | 67,948,314,624 |
| Shared MLP/gate bytes | 85,132,416 | 472,104,960 |
| Non-routed target bytes excluding PLE table/vision | 3,934,092,384 | 9,895,397,402 |
| Attention | 36 GDN, 12 full attention | 36 GDN, 12 QSA; gated residual streams |
| PLE | None | 47.684 GiB FP8 table plus 0.061 GiB auxiliaries |
| Expert source + canonical host lower bound | 81.001 GiB | 126.563 GiB |
| Same lower bound plus one PLE table | 81.001 GiB | 174.247 GiB |
| Cache integration | CPU-source/shared path integrated; full model unqualified | Mixed dispatch and PLE composition still unsupported |
| Scientific role | Isolates expert residency beyond VRAM | Composes expert residency with a separate large lookup system |

Both packed routed payloads alone exceed the available SM120's 23.424 GiB CUDA
memory. Qwen3.8 also needs about 9.216 GiB of stored text-model non-routed,
non-table tensors before execution workspaces, KV/recurrent state and graphs.
Its useful resident capacity is therefore smaller than a payload-only estimate.
No resident geometry or complete live memory admission is claimed from headers.

## Runtime compatibility

The companion already contains the model card's prerequisite
[`d4d703caf908786416585ceb1f369e2e0363358b`](https://github.com/vllm-project/vllm/commit/d4d703caf908786416585ceb1f369e2e0363358b)
and merged [PR 55513](https://github.com/vllm-project/vllm/pull/55513), commit
`60ad959b6f1a5c8f602edbd608c8decbc0788c50`. Ancestry was checked directly.
No large upstream rebase or model reimplementation is indicated.

`vllm/models/qwen4_exp/nvidia/model.py`, its backend and V2 model-state module
already implement model registration/aliases, GDN, QSA/indexers, hyper-connections,
multimodal loading and PLE. V2 state owns persistent sparse-attention slots and
n-gram request context, including accepted-token rollback. These implementations
exist; their combination with this checkpoint and b12x residency is unqualified.

The main MoE inherits `Qwen3NextSparseMoeBlock`. Its ordinary shared MLP and
sigmoid gate remain external to the routed cache. `MoERunner` owns execution,
stream synchronization and combination order. The companion's existing shared
wrapper fix applies at that boundary; shared experts never enter routed IDs or
counters. Synthetic wrapper tests do not replace a real checkpoint-layer oracle.

All **73,728 routed global scales** were read through 589,788 bytes of bounded
ranges. They are positive and finite; all **24,576 gate/up pairs are equal**.
Packed shapes are gate/up `(640,1280)` and down `(2560,320)`, with K16 scales,
SiLU, normalized top-10 routing, BF16 activations under the explicit cache
whole-K recipe, and no routed bias. Block-scale values and full packed bytes
remain unchecked. No requantization or arithmetic change is authorized by this
metadata result. A promotion copies **2,764,808 payload bytes**; 32 pairs copy
84.375 MiB plus the existing map/publication accounting, below 128 MiB.

## PLE placement and ownership

The immutable table sits at decoder layer index 1. Order-three n-grams use eight
heads per order, sixteen lookups per token, each with 160 FP8 elements. Its
320,001,536 padded rows are stored in 128 parts of `(2500012,160)`. The selected
rows supply a 2560-wide PLE input; this is a sparse lookup subsystem, not MoE
replacement state. Sixteen rows describe logical access, not measured PCIe
transaction traffic.

The standard NVIDIA PLE implementation already supports a pinned CPU table with
a CUDA UVA view, GPU gather, FP8 conversion and the checkpoint scalar. Only
selected outputs are produced on device; no full-table GPU copy is required.
Its capture boundary and request history remain engine-owned. The b12x PLE
implementation also has prepared mapped-host FP8 lookup/storage and a disk path.
No additional PLE cache or new lookup kernel is needed for this audit.

Four composition gates prevent a ready-to-download recommendation:

1. `b12x_cache._CacheProvider` requires `modelopt_fp4`, whereas this checkpoint
   selects `modelopt_mixed`. The mixed config overrides `get_quant_method` and
   bypasses cache dispatch. A per-prefix NVFP4 handoff is needed; removing the
   provider guard alone is incorrect.
2. Selecting the b12x MoE backend also selects b12x PLE through `uses_b12x`.
   That path reads `ple_embedding_dtype`, whose config default is BF16; the
   NVIDIA config omits it. The standard PLE path derives FP8 from mixed
   quantization metadata. The b12x path must do likewise before allocation.
   Relabeling or converting the FP8 table would change this contract.
3. Expert-cache admission does not jointly charge the independently owned PLE
   table. One combined host/device reservation must cover both owners and
   loading peaks. A free-memory check alone is not that admission contract.
4. `TableStorage.close()` exists, but the companion PLE owner has no explicit
   teardown hook after graph/stream retirement. Destructor cleanup is not a
   cancellation/lifecycle qualification. This is an unqualified ownership path,
   not a measured leak claim.

## Host feasibility and download decision

Expert CPU sources require 67,948,314,624 bytes and aligned canonical mapped
backing 67,947,921,408 bytes. Adding **one** FP8 PLE table gives a lower bound of
**187,096,481,792 bytes (174.247 GiB)**. PLE is not duplicated as another expert
source. Of this, **119,148,167,168 bytes (110.965 GiB)** would be mapped/pinned
expert backing plus PLE under the host path. Loader objects, shard views,
conversion, staging, journals, reserves and other resident tensors are additional.

Ripper had 531,717,656,576 bytes available RAM and 1,073,068,572,672 bytes free on
the model volume. Arithmetic capacity is plausible. The locked-memory limit
was 65,998,848 KiB; CUDA host allocation is not proven to obey that limit in the
same way as `mlock`, so neither success nor failure is inferred from it. No large
pinning trial, NUMA placement qualification or OS-limit change was performed.
A 111-GiB pinned representation is not operationally qualified by free RAM alone.
The largest shard is 53,717,551,730 bytes: eager shard duplication could add that
much peak storage, whereas lazy file mappings have a different RSS profile.

If a later integration gate justifies download authorization, request exactly
revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`. The full repository requires
**132,734,506,847 bytes**, including **132,680,249,378 safetensors bytes**. The final
single-copy disk footprint is approximately 123.62 GiB plus filesystem overhead.
Allow up to another **53,717,551,730 bytes** for one staged shard (about 174 GiB
combined), or a second full copy if the download/export method duplicates the
snapshot. Use an atomic single-copy cache/export arrangement instead. These are
storage planning bounds, not authorization; existing 80B/QAD artifacts stay intact.

## Separate MTP qualification

MTP has one full-attention layer, 512 routed experts, H2560/I640, top-10 routes
and FP8 128-by-128 blocks. Expert scale grids are `(5,20)` for gate/up and
`(20,5)` for down; 307,200 bytes are block scales. Its exclusive routed tensors
occupy 2,516,889,600 bytes. Other draft modules remain BF16. The engine invokes a
separate predictor, reuses target embeddings/head and has speculative graph and
accepted-token state. Its expert selections are separate from target selections.

The single-rank dimensions are block-divisible; an eight-way tensor split of
I640 is not 128-divisible, explaining why the documented multi-GPU example needs
different sharding. This does not qualify single-SM120 MTP. Existing cache
speculation guards remain, and FP8 draft experts cannot use the main NVFP4
whole-K cache recipe. Future work needs separate draft memory/preparation,
phase/counter ownership, accepted/rejected-token accounting and cancellation
tests before comparing effective accepted tok/s with ordinary decode.

## Reproduction and decision

Use the existing header-only tool, with observed device and explicit host bounds:

```bash
python scripts/inspect_expert_cache_checkpoint.py \
  --repository nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --metadata-output "$AUDIT_DIR/metadata" --output "$AUDIT_DIR/report.json" \
  --device-bytes 25151012864 \
  --host-bytes 531717656576
```

The result deliberately says `metadata_audited_cache_integration_required`.
It requires complete inventory and matching mixed-quantization declarations,
including optional MTP shapes. It never enables cache serving. Existing 80B
preflight, CPU-source loader and numerical contracts remain intact.

| Gate | Decision |
|---|---|
| Main architecture available upstream/in companion | Pass, source inspection |
| Routed NVFP4 metadata/global-scale compatibility | Pass, bounded inspection; arithmetic still unqualified |
| Shared-wrapper composition | Existing reusable contract; real model oracle pending |
| PLE host/UVA implementation exists | Pass; NVIDIA FP8 selection and explicit teardown pending |
| Joint memory admission and useful resident capacity | Pending |
| Complete checkpoint available locally | No; local QAD snapshot is a different export |
| Main model with MTP disabled | Supported architecture decomposition; cache execution pending |
| Immediate flagship replacement/download | **Defer** |

The smallest follow-on is mixed-prefix cache dispatch plus metadata-derived FP8
PLE selection, joint admission and explicit PLE lifecycle tests. It requires no
new attention implementation, replacement policy or bounded-staging redesign.
Until those gates pass, 80B is the cleaner complete-model residency experiment.
No authorized Station endpoint was configured; physical SM103 remains deferred
under the [existing qualification runbook](sm103-qualification.md). All SM120
physical checks remain PCIe Gen4 x16 evidence, with no Grace/Gen5 extrapolation.
