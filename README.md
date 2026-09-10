# b12x

`b12x` is an SM120/SM121 CuTe DSL and Triton kernel library for local LLM inference.
It specifically targets DGX Spark, RTX Spark and the Blackwell-based RTX
cards (RTX 6000 Pro, RTX 5090).

It is *not* intended to be used in production/datacenter environments, both due to
architecture mismatches and the fast-moving pace of the library. For mission-critical
use cases please use FlashInfer, CUTLASS or TRTLLM.

## Install

```bash
pip install b12x
```

You need Python 3.10+, `torch >= 2.12`, and an SM120/SM121 GPU. The CuTe DSL
compiler and its CUDA 13 libraries come in as wheel dependencies
(`nvidia-cutlass-dsl == 4.6.2`), so there is no build step — kernels are
JIT-compiled on first use and cached.

## What's in here

Every kernel is one op at `b12x.<group>.<op>`; `list_ops()` enumerates the
complete set. The op owns its `plan`/`bind`/`run` facade in `api.py`; the
kernel guts sit in `_impl.py`/`_kernel*.py`; cross-op lowering lives in
`<group>/_shared/` and the universal compile/scratch spine in `b12x/_lib/`.

**`gemm`** — `gemm.blockscaled` is the common dense interface for raw
NVFP4/MXFP4/MXFP8/block-FP8 operands and packed MXFP8/tensor-FP8 weights; it
owns `mm`, `pack_weight`, and serving `prewarm`. The legacy
`gemm.mxfp8_linear` and `gemm.tensor_fp8_linear` imports are compatibility
aliases. `gemm.block_fp8_linear` retains a separate planned interface because
it owns caller-provided scratch and inline requantization. The fused MLA query
projection (`gemm.mla_query_projection`) and grouped WO projection
(`gemm.wo_projection`) are used around MLA attention.
`gemm.block_fp8_linear` also accepts V4.1's 32x32 E4M3/UE8M0 weight blocks
with per-32 activation quantization. `gemm.bf16_gemv.mm` handles unquantized
BF16/FP32 projections, including FP32 outputs and bias before final rounding;
live row counts share one compiled geometry/type specialization.


**`attention`** — `attention.paged` (paged-KV decode/extend, FP8 KV, MSA block
sparse, CUDA-graph-replayable), `attention.sparse_mla` and
`attention.compressed_sparse_mla` (top-k / compressed-page MLA — distinct
contracts, kept separate on purpose), `attention.dsa_indexer` (the DSA/MSA quantize →
score → select pipeline), `attention.qsa` (group-selected exact sparse GQA
decode over caller-populated, read-only main BF16 K/V), and `attention.varlen`
(contiguous batched/varlen).
The `deepseek_v41` compressed-MLA recipe uses distinct post-RoPE cache
records: 528-byte SWA rows (E4M3 plus per-32 UE8M0 scales) and 288-byte main
rows (E2M1 plus per-16 E4M3 scales, without a global scale).
`attention.mla_compress` produces normalized pre-RoPE latents for the
nonoverlapping ratio-1/ratio-2 compressor. The MXFP4 DSA recipe exposes a
score/reduce/select boundary and bounded candidate indices for hierarchical
reindexing; tensor-parallel score reduction precedes selection.


**`moe`** — `moe.fused_moe`, fused FP4 TP MoE across a micro-kernel decode
path, a unified dynamic path (persistent grid, `nvfp4`/`w4a8_mx`/`w4a8_nvfp4`),
and W4A16 (BF16 activations, inline FP4 weight dequant — no activation-scale
math), with SiLU/ReLU2/SwiGLU-OAI activations; plus `moe.ep_moe` (expert
parallel).

**the rest** — `norm.mhc` (fused RMSNorm + hyper-connection residual),
`norm.hyperconnection` (learned multi-stream residual primitives),
`sequence.{ple_hash,ple_embedding,ple}` (prime-hashed embedding IDs, fused
quantized lookup, and short-convolution state), `sequence.gdn_decode` (packed
recurrent decode), `sequence.gdn_prefill` (research-only scalar-gated prefill;
see [GDN prefill](docs/gdn-prefill.md)), `sequence.kda_prefill` (KDA prefill),
`sequence.mtp_feedback` (MTP token/multi-stream feedback fusion),
`quantization.{nvfp4,mxfp8}` (row quantizers), `comm.roce` (RoCEnante: one-shot
RDMA all-reduce/all-gather for multi-node DGX Spark TP, see `docs/rocenante.md`),
and `comm.pcie` (IPC-backed PCIe
collectives). The Qwen3.8-Flash-Next QSA, HyperConnection, PLE, GDN decode,
and MTP feedback Triton implementations are correctness references and are
not throughput-qualified production kernels.
`sequence.engram` implements V4.1's compressed-token, DEAD-bounded n-gram
hashing and row-sharded FP8/group-32 lookup. It is not an alias for Qwen PLE:
their tokenizer domains, boundary rules, scale layouts, and convolution state
differ. Engram's direct residual injection uses
`norm.hyperconnection.run_engram_mix`; ordinary affine RMSNorm is selected
with `zero_centered=False`. `norm.mhc.run_pre` and `run_post_pre` accept
incoming `pre_mix` and caller-owned `pre_out` for lagged V4.1 mixing, while
`run_collapse` supports the final weighted collapse and uniform stream mean.

PLE and Engram share the bounded io_uring row cache in
`sequence._shared.disk_table`: registered file regions, aligned O_DIRECT reads,
block deduplication/coalescing, mapped-host staging, and stream-safe reuse.
Their hashing and decoding stay separate. Engram's `DiskTable` reads the
checkpoint's FP8 weight rows and separate E8M0 scale rows without conversion or
full-table allocation; `run_lookup(..., token_count=...)` prepares fixed GPU
outputs outside `torch.compile` and CUDA graph capture. The V4.1 vLLM adapter
selects this path with `--engram-config '{"table_memory":"disk"}'` and prepares
both Engram layers before each target forward, including speculative
verification. DSpark's own draft/Markov graph has no Engram disk reads.

`sequence.embedding` provides exact unquantized BF16/FP32 row lookup into
caller-owned output, with Int32/Int64 IDs and Int64 table offsets. Its
`precompile` entrypoint warms width/type specializations before capture;
live row counts and table extents do not select compiler-cache entries.

The V4.1 serving adapter retains checkpoint-native BF16 weight storage and
BF16 activation/collective outputs instead of promoting them solely to match
reference rounding. Native accumulation, normalization, routing scores and
ratio-two softmax-pooling state remain FP32 where required. Speculative
rejection preserves per-token compressor partials in request-owned bounded
rings rather than overwriting one terminal carry state.

The adapter binds mHC and sparse-attention/indexer plan scratch to vLLM's shared
workspace. Outputs remain separate, live-row-sized allocations rather than
per-layer scheduler-capacity buffers reserved during model construction.
The [startup verification record](validation/deepseek_v41/startup_workspace_fix.json)
covers the constructor-memory repair. The subsequent
[native KV allocation repair](validation/deepseek_v41/native_kv_allocation_fix.json)
aligns main KV and index K on the same logical token blocks, preserves their
packed 890-byte-per-token global footprint, and restores model-derived cache
grouping instead of the fork's bounded GLM grouping path. TP4 SSD serving now
initializes the full 1M context at a 4096-token batch capacity and 0.95 memory
utilization; the record distinguishes capacity accounting from exercised
prompt lengths and includes graph replay, rejection, and prefix-reuse checks.

`comm.pcie.PCIeDmaAllReduce.prepare_eager_replay(dtype)` prepares a lossless,
capacity-sized CUDA graph for large eager collectives. All ranks prepare it
before serving. Live prefixes reuse that graph through fixed staging buffers;
returned outputs remain independent across calls. Compressed wire modes and
outer graph captures keep their existing execution paths.

`b12x` owns planning, scratch layout, and policy, so serving stacks only supply
metadata and capacity limits.

## Using it

Every stateful kernel lives at `b12x.<group>.<op>` and shares the **same
shape** — `plan` the work, size scratch from the plan, `bind` your tensors as
views, `run`. The module path carries the context, so the verbs and role
classes (`Caps`/`Plan`/`Binding`) are uniform across families:

```python
# norm — fused RMSNorm + hyper-connection residual mixing
from b12x.norm import mhc

plan    = mhc.plan(mhc.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = mhc.bind(plan, scratch=scratch, ...)
residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
```

```python
# moe — fused tensor-parallel routed-expert FFN (weights prepped once per model)
from b12x.moe import fused_moe

wplan   = fused_moe.plan_weights(quant_modes="nvfp4",
                                 source_format="modelopt_nvfp4", ...)
experts = fused_moe.prepare_weights(plan=wplan, ...)
plan    = fused_moe.plan(fused_moe.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                         topk_weights=tw, topk_ids=ti)
out     = fused_moe.run(binding=binding)
```

```python
# attention — sparse MLA from compressed KV pages (DeepSeek V4)
from b12x.attention import compressed_sparse_mla

plan    = compressed_sparse_mla.plan(compressed_sparse_mla.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = compressed_sparse_mla.bind(plan, scratch=scratch, q=q,
                              swa_indices=idx, swa_lengths=lens, ...)
out = compressed_sparse_mla.run(swa_k_cache=swa, binding=binding, sm_scale=scale, ...)
```

`plan` is host-side and may allocate; `bind` only narrows/views (never
allocates), which is what makes captured graphs safe; `run*` executes and is
CUDA-graph-capture safe. One-shot ops (`gemm.blockscaled.mm`,
`quantization.mxfp8.quantize_rows`) are plain functions; `comm.pcie`
collectives are stateful classes. `b12x.list_ops()` enumerates the full
set; every op exports `is_supported()`. Underneath, kernels register as torch
custom ops in the private `b12x::` namespace (torch.compile / CUDA-graph
integration) — prefer the Python API.

## PCIe DMA wire modes

`PCIeDmaAllReduce` can compress eligible BF16 all-reduces. Configure it with
`B12X_PCIE_DMA_FP8`, or pass the same value as the `fp8=` constructor
argument. Integrations such as vLLM can forward their own launch setting to
that constructor.

| Mode | Reduce-scatter | All-gather | When to use it |
|---|---|---|---|
| `0` | BF16 ring | BF16 ring | Unquantized baseline |
| `ag` | BF16 ring | block E4M3 ring | Limit E4M3 quantization to the final broadcast |
| `ring` | block E4M3 ring, requantized per hop | block E4M3 ring | Compress both phases with the neighbor ring |
| `a2a` | block E4M3 scatter with FP32 accumulation | block E4M3 broadcast | Quantize each input once and overlap direct peer transfers |
| `i8` | BF16 ring | block INT8 ring | Limit INT8 quantization to the final broadcast |
| `i8_ring` | block INT8 ring, requantized per hop | block INT8 ring | Compress both phases with the INT8 codec |
| `i8_a2a` | block INT8 scatter with FP32 accumulation | block INT8 broadcast | Use the quantize-once all-to-all topology with INT8 |
| `mx` | BF16 ring | MXFP8 ring | Limit MXFP8 quantization to the final broadcast |
| `mx_ring` | MXFP8 ring, requantized per hop | MXFP8 ring | Compress both phases with standard E4M3/E8M0 MXFP8 |
| `mx_a2a` | MXFP8 scatter with FP32 accumulation | MXFP8 broadcast | Use the quantize-once all-to-all topology with MXFP8 |

Every compressed mode uses 132 bytes per 128 values instead of 256 bytes for
BF16, a 48.4% wire-byte reduction. E4M3 and INT8 store one FP32 scale per 128
values; MXFP8 stores four E8M0 scales, one per 32 values. These modes are most
useful for large prefill collectives on PCIe-only multi-GPU systems where peer
transport is the bottleneck; they do not change the KV-cache format and usually
do not affect small decode collectives. Choose a codec by model quality gates,
then benchmark the ring and all-to-all variants on the target PCIe topology.

Compressed transport requires BF16 input and a per-rank shard divisible by
128 elements; other shapes use the BF16 path:

```bash
B12X_PCIE_DMA_FP8=i8_ring python -m your_server
```

Specializations follow static geometry, dtype, planned capacity and
device/toolchain identity, not live request counts. Before serving, precompile
the selected specializations and exercise their runtime paths, then freeze:

```python
import b12x

# ... preplan capacities and warm every selected specialization ...
b12x.freeze_kernel_resolution("serving")
```

After the freeze, any request that would trigger a new kernel compile raises
`KernelResolutionFrozenError` instead of stalling a live request (or worse,
compiling inside CUDA graph capture).

Set `B12X_PRINT_COMPILE_PROGRESS=1` to log each compiler invocation with its
cache-key parameters and duration — useful for figuring out what warmup
actually covered. `B12X_TIMING=1` enables per-kernel timing logs.

## Where to look next

- `tests/` is the executable spec — per-group API and numerical-reference
  tests showing exact tensor layouts and `plan`/`bind`/`run` call sequences.
  (`tests/_legacy/` holds the pre-namespace flat-API suite, being migrated.)
- `benchmarks/` has tuned invocations per kernel family (and `probe_*` scripts
  from tile-sweep experiments).
- `docs/` has design notes: the MoE execution model, the eager-plan-bind
  architecture, and an SM120 MLA postmortem.
- `validation/deepseek_v41/` retains the original qualification artifacts from
  the `Initial DSV41 bringup` backport (`674cacac5226baf8143327560040ce1e2e67fea5`).
  Its JSON/CSV records preserve migration-era paths and hashes as provenance;
  they do not qualify subsequent changes to this standalone tree.

Failing that, ask your friendly neighborhood AI agent — it does fine here.
