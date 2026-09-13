# b12x

`b12x` is a CuTe DSL and Triton kernel library for local LLM inference on
Blackwell GPUs. SM120/SM121 paths target DGX Spark and Blackwell RTX cards.
The SM103/B300 prototype implements NVFP4 and Trellis MoE, CuTe KDA/GDN
recurrence, dense/GLM/compressed DeepSeek MLA, DSA indexing, mHC residual
mixing, and unquantized or FP4/FP6/FP8 projections behind the existing public
APIs. It also provides Engram memory placement and an explicitly enabled
Grace TP2 transport.
SM103 kernels are cross-compiled; B300 execution and model serving remain
unqualified. See the [SM103 qualification guide](docs/sm103-qualification.md)
for supported contracts, deferred operations, and bring-up commands.

The library is experimental. Architecture recognition does not establish
operator support or production readiness; consult each operator's capability
query and planning contract.

## Install

```bash
pip install b12x
```

You need Python 3.10+, `torch >= 2.12`, and a GPU supported by the selected
operator. The CuTe DSL
compiler and its CUDA 13 libraries come in as wheel dependencies
(`nvidia-cutlass-dsl == 4.6.2`), so there is no separate build step. A
`PreparationSession` compiles missing kernels before publishing execution.

The optional vLLM [checkpoint loader](docs/checkpoint-loading.md) uses
`--load-format b12x`: coherent managed-memory direct I/O on Spark, and shared
GPUDirect Storage into device memory on discrete GPUs. Enable its
`b12x_loader` vLLM plugin; the device selects the transport automatically.

## What's in here

Every kernel is one op at `b12x.<group>.<op>`; `list_ops()` enumerates the
complete set. The op owns its `plan`/`bind`/`run` facade in `api.py`; the
kernel guts sit in `_impl.py`/`_kernel*.py`; cross-op lowering lives in
`<group>/_shared/` and the universal compile/scratch spine in `b12x/_lib/`.

**`gemm`** — `gemm.blockscaled` is the common dense interface for raw
NVFP4/MXFP4/MXFP8/block-FP8 operands and packed MXFP8/tensor-FP8 weights; it
owns declarative planning, `mm`, and `pack_weight`. The fixed
`gemm.mxfp8_linear` and `gemm.tensor_fp8_linear` interfaces also require prepared
execution. `gemm.block_fp8_linear` retains a separate interface because
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
reindexing. The current V4.1 serving adapter replicates all 32 index heads and
selects locally; integrations that shard index heads must reduce scores first.
Its native CuTe scorers dequantize Q/K to BF16 inside the paged kernels.
Prefill uses BF16 tensor-core dots and decode uses SIMT, both with FP32
accumulation. Dot results, weighted
products, and the final head sum retain their separate BF16 rounding points.


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

PLE and Engram share the bounded disk row cache in
`sequence._shared.disk_table`: registered file regions, aligned O_DIRECT reads,
block deduplication/coalescing, and stream-safe reuse. The default io_uring
transport stages rows in mapped host memory. `B12X_DISK_BACKEND=gds` selects
optional cuFile batch reads and GPU row gathering with device staging;
see [disk transport requirements and qualification](docs/disk-embedding-backends.md).
Their hashing and decoding stay separate. Engram's `DiskTable` reads the
checkpoint's FP8 weight rows and separate E8M0 scale rows without conversion or
full-table allocation; `run_lookup(..., token_count=...)` prepares fixed GPU
outputs outside `torch.compile` and CUDA graph capture. The V4.1 vLLM adapter
selects this path with `--engram-config '{"table_memory":"disk"}'` and prepares
both Engram layers before each target forward, including speculative
verification. DSpark's own draft/Markov graph has no Engram disk reads.
The reader registers immutable file descriptors, retires completions in batches,
and radix-sorts large requests using its existing job allocation as scratch.
Small requests retain the in-place sorter. Engram defaults to a fixed queue
depth of 128 (8 MB of native I/O buffers per table); PLE retains 64. No table
payload is cached between transactions, and live counts do not resize storage.

`benchmarks/benchmark_ngram_ssd.py --models engram --capacities 4096
--tokens 4096 --seqs 1 --max-seqs 8 --engram-token-bound` measures checkpoint
prefill transactions; use `--tokens 8 --seqs 8` for batch-eight decode.
`--reader-library PATH` selects an exact cached native object for diagnostic A/B
comparisons, independently recording its hash and the current source hashes.
These are hash/D2H/disk/GPU-decode timings, not full-model or TP-collective timings.

`sequence.embedding` provides exact unquantized BF16/FP32 row lookup into
caller-owned output, with Int32/Int64 IDs and Int64 table offsets. Its
`EmbeddingQuery` declares capacity, geometry and pointer dtypes. A
`PreparationSession` prepares the plan before binding or capture; live row
counts and table extents within that capacity remain runtime values.

The V4.1 serving adapter retains checkpoint-native BF16 weights and activations.
Internal accumulation, normalization, routing scores, and ratio-two
softmax-pooling state remain FP32 where used by the common implementations.
MoE output, shared-expert combination, and TP reductions use the same BF16
boundary as V4.0. Speculative rejection preserves per-token compressor partials
in request-owned bounded rings rather than overwriting one terminal carry state.

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

The [structural optimization qualification](validation/deepseek_v41/structural_optimization.json)
records the subsequent prefill/decode work. Attention consumes a whole planned
query batch after bounded indexer chunks, rather than launching attention for
every 64 rows. Eager indexing bounds score/collective width by known visibility;
captured indexing retains fixed capacity with tiled clearing of inactive columns.
Lagged mHC uses native post-pre fusion except across Engram mutation. Dense
execution regimes are prewarmed, and DSpark context preparation uses bounded
graphs and KV-only checkpoint projections. SSD Engram uses prefix-bound hashing
and lookup, with initialization and retired-row clearing owned by its staging
buffer. The benchmarks retain distinct Engram/PLE quantization and hash contracts.

The [CED acceptance record](validation/deepseek_v41/ced_prefill.json) covers
full-row encoder execution and decoder global-KV preparation followed by
bounded decoder replay. Each long request chunk keeps its trailing 128 decoder
rows; short chunks continue the request's private SWA state. Decoder and draft
SWA are not published to prefix caching, and encoder/global prefix hits leave
at least 128 tokens to regenerate decoder state. Prompt-logprob requests retain
full decoder rows. Sampling keeps its original row ABI; DSpark projects only
the selected context rows. Replay is intentionally approximate, as described
in the model report, rather than identical to full-decoder prefill.

The [profile-guided optimization record](validation/deepseek_v41/profile_optimization.json)
separates full-serving latency from native-kernel diagnostics. Its original
BF16-path changes—paired-lane PV dequantization, packed SWA pair conversion,
and native shared-byte loads—preserve that path's arithmetic. The current
default FP8 arithmetic and its independent oracle are described in the
attention benchmark section below. Split-merge cache identity also distinguishes dynamic layout ABIs when
one active split occupies workspaces with different planned capacities.

V4.1 drafting retains its own checkpoint, three draft layers, cache layout and
mHC collapse, while using the shared DSpark heads and vocabulary dispatch used
by V4.0. Query preparation is bounded by the draft query capacity; metadata
refresh covers the selected graph's padded token domain without reallocating
its buffers. Adaptive-verification confidences are broadcast from TP rank zero
before both CPU budgeting and GPU compaction, so all ranks choose compatible
graph sizes and per-request token boundaries.

Engram also supports `ENGRAM_TABLE_MEMORY=ram` in the V4.1 launcher.
Its packed E4M3 weights and E8M0 scales live in b12x CUDA-mapped host RAM;
checkpoint loading writes only each TP shard directly into its final CPU
aliases. Native GPU lookup reads those allocations over PCIe, without SSD
transactions during generation. Both table owners remain alive through graph
replay. The full checkpoint uses 188.83 GiB of pinned host RAM across TP4.
The [RAM investigation record](validation/deepseek_v41/ram_engram.json) retains
successful shard/replay smoke checks but is marked unsafe after a later host OOM.
Use SSD storage while full-RAM peak memory remains unqualified.

`comm.pcie.PCIeDmaAllReduce.prepare_eager_replay(dtype, max_elements=...)`
prepares a lossless graph for a planned element bound. A larger FP32 workspace
does not inflate the BF16 replay size. Exact-capacity eager inputs use fixed
staging buffers; other supported sizes use raw DMA instead of transferring
padding. Outputs remain independent. All ranks prepare the same dtype bounds
before capture; compressed wire modes and outer captures keep their existing paths.

`b12x` owns planning, scratch layout, and startup selection, so serving stacks only supply
metadata and capacity limits.

## Using it

Native execution follows one lifecycle: a plan declares the work, a session
prepares it, and the family binds and runs from the prepared plan. A plan
allocates no CUDA storage and compiles nothing. The component owns its legal
configurations, default, compiler extraction and memory formulas.

```python
import torch
from b12x.gemm import bf16_gemv
from b12x.preparation import PreparationSession, PreparedCall

def main():
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    plan = bf16_gemv.plan(bf16_gemv.query_from_call(x, weight))
    request = plan.request(
        name="small_linear",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(x, weight)),
    )
    with PreparationSession(device=x.device) as session:
        session.prepare((request,))
        y = bf16_gemv.mm(x, weight, plan=plan)
        torch.cuda.synchronize()
        print(y.shape)  # torch.Size([1, 8])

if __name__ == "__main__":
    main()
```

The callback receives the selected component's private materialized state and
returns a `PreparedCall` that runs the real operation with actual parameters.
Multi-candidate search needs a representative activation producer; stateful
calls restore touched state after trials. Public `bind` and functional entry
points take the plan.

Cache misses autotune by default. A valid explicit config pin wins; completed
cached choices are reused. Disabled or cancelled search still prepares the
validated default. Fixed operations prime without a race. Families that opt
into sharing prepare one state for equal declarations.

An unprepared plan is materialized with its default configuration on first
use, with a warning; after `session.freeze()` or during a CUDA graph capture
that is an error. Exact-M paths prepare every planned count.
Capture under `session.capture()`, keep the plans alive for the graph lifetime,
and destroy graphs before `session.release(plan)` or closing the session.

See [GPU preparation and startup autotuning](docs/gpu-profiles.md) for the
selection, memory, cooperative startup and vLLM integration contracts. Prefer
the Python component API over private `b12x::` custom ops.

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
device/toolchain identity. Prepare the declared operations with real inputs,
then freeze the session before serving:

```python
from b12x.preparation import PreparationSession

with PreparationSession(device=device) as session:
    session.prepare(requests)
    session.freeze()
    # Capture and serve using the prepared plans.
    # Destroy their CUDA graphs before leaving the session.
```

After the freeze, any request that would trigger a new kernel compile raises
`KernelResolutionFrozenError` instead of stalling a live request (or worse,
compiling inside CUDA graph capture).

Set `B12X_PRINT_COMPILE_PROGRESS=1` to log each compiler invocation with its
cache-key parameters and duration — useful for figuring out what warmup
actually covered. `B12X_TIMING=1` enables per-kernel timing logs.

## DeepSeek V4.1 Flash MoE benchmark

The `deepseek-v4.1-flash` checkpoint profile defaults to W4A8 (`w4a8_mx`)
and TP4. It is distinct from the V4.0 `deepseek-v4-flash` profile:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_moe.py \
  --model-profile deepseek-v4.1-flash \
  --batch-sizes 1 2 4 8 --reference none --cuda-graph --graph-only
```

Use `--model-path` to select a local V4.1 checkpoint and `--tp-rank` to select
the rank slice (default 0). V4.1 uses tensor parallelism over each expert's
intermediate channels: TP4 retains all 384 target experts on every rank, with
hidden size 5120 and logical/physical intermediate width 576. DSpark similarly
retains all 128 draft experts per rank.
Text routing uses the checkpoint gate and selection-only bias; expert IDs stay
global without rank-local masking or renormalization. The timed output is the
BF16 rank-local routed result, excluding the shared expert and TP all-reduce.

Serving reuses DSV4's existing MoE module and the shared vLLM b12x backend,
including TP-sharded shared experts. Earlier whole-expert-sharding measurements
are historical evidence, not measurements of this TP-only path.
The vLLM integration reserves shared and routed scratch together with one
`get_simultaneous` call, then supplies disjoint views while preserving overlap.

The profile uses the same MoE arithmetic, routing-weight placement, kernel
selection, and BF16 output boundary as V4.0. Checkpoint-specific dimensions,
TP slicing, and scale layouts remain distinct. It reuses one fixed-capacity
scratch plan across the requested token counts. Graph replay is checked
against eager output; the default timing mode flushes L2 outside timed events.
The compact N64-tail weight layout uses the common split-materialized M16
pipeline rather than the padded N256/K128 tiny-decode layout.
Startup selection measures eligible configurations for the declared TP4
geometry. Completed selections are cached for the device and toolchain.

## DeepSeek mHC benchmark

`benchmark_residual.py` has checkpoint-backed `deepseek-v4-flash` and
`deepseek-v4.1-flash` profiles. They invoke the actual mHC adapters from the
requested vLLM checkout, including fused RMSNorm and V4.1's incoming/predicted
mix handoff:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_residual.py \
  --model-profile deepseek-v4.1-flash \
  --vllm-path ~/projects/vllm-hh-rebase \
  --tokens 8 --l2-flush --output /tmp/mhc-v41.json
```

Switch to `--model-profile deepseek-v4-flash` for V4.0. `--model-path` selects
an explicit checkpoint; otherwise the loader uses its Hugging Face cache.
The default is layer 3's attention-post/FFN-pre boundary, with native checkpoint
parameters and synthetic BF16 activations. Timings cover CUDA graph replay on
one rank, not a model forward or collective. Hidden width is not TP-sharded
inside mHC. The current adapters are called directly rather than forced through
a separate Dynamo compilation mode.

Each named profile checks residual, activation, and mixing outputs against its
oracle before timing and after replay, freezes kernel resolution, and locks the
vLLM workspace against growth. JSON output retains raw samples, GPU operating
state, checkpoint weight hashes, and b12x/vLLM source provenance. The existing
synthetic microbenchmark remains available with `--model-profile custom`.
Lagged decode prepares the BF16-rounded collapse and its squared-sum partials
alongside the residual/projection pass, then normalizes independent hidden
tiles. It reuses existing scratch capacity and output storage; prefill keeps
its established rounding path. Compile identities include the actual static
block geometry and prepared-lagged mode, never the live token/CTA count.
Unbound lagged calls retain one native producer/finalizer mode across live
token counts as well.

## DeepSeek indexer and sparse-MLA benchmarks

`benchmark_dsa_indexer_profiles.py` and `benchmark_sparse_mla_profiles.py`
load model geometry from the checkpoint config and match the native prepared-Q/K
contracts in `vllm-hh-rebase`. These are CUDA-graph kernel benchmarks, not
checkpoint-projection or whole-vLLM-forward timings. Both default to TP4:
16 local attention heads, but 32 replicated V4.1 index heads (64 for V4.0).

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_dsa_indexer_profiles.py \
  --model-profile deepseek-v4.1-flash \
  --max-model-len 32768 --contexts 16384,32768 --rows 1,2,4,8 \
  --output /tmp/dsa-v41.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sparse_mla_profiles.py \
  --model-profile deepseek-v4.1-flash --batch-sizes 1,2,4,8 \
  --output /tmp/mla-v41.json
```

The position-sort stage uses warp-register compare/exchange for local stages,
CTA synchronization for cross-warp stages, and parallel candidate-length
reduction. It preserves index/value associations, newest-block inclusion,
partial-block clipping, and output padding without changing scoring math.

Indexer forms are `c2-dense`, `c1-source` (including publication of up to
2048 block-8 candidates), and `c1-reindex` over that source-owned candidate
set. Reuse layers do not run an indexer. Context arguments count original
tokens, not compressed states; captured V4.1 scoring retains the planned
capacity even when visibility is smaller. The 32K example exercises real
candidate pruning beyond the 16K budget.

MLA forms are `swa`, `c2`, `c1`, and `draft-swa`. C1 remains indexed attention,
not SWA-only. Drafting uses seven query rows per request by default and a
192-column padded noncausal SWA region. For target prefill, use
`--modes extend --query-tokens 128 --batch-sizes 1,2,4,8`; `--query-rows`
instead selects controlled kernel row counts. `--forms` selects a subset.
V4.1 reuses fixed decode/extend capacities; the V4.0 comparison preserves its
integration's per-capture-shape planning.

V4.1 decode uses the native FP8 implementation by default, sharing the
V4-style H8/H16, swapped-QK and FP8-PV machinery. H16 is selected for compatible
head geometry; narrow or remainder shards retain H8. V4.1 prefill uses BF16
QK and FP8 P×V, including the 128-token sliding window plus indexed-cache union.
Precision and decode head grouping are selected once by the plan's typed
`SparseMlaConfig`; bind/run and graph replay do not consult environment switches
or resolve policy again. An explicit `plan(..., override=...)` can select the
BF16 reference/debug path. Benchmark separate plans with
`--compute-modes bf16,fp8` for either decode or extend.
The serving cache stays in its native 528/288-byte formats. The I/O producer
prepares scale ratios in kernel-local shared memory; math warps perform packed
half2-to-FP8 conversion before tensor-core work. Prefill shares that canonical
KV tile between BF16 QK and FP8 P×V.

The FP8 implementation has an independent FP8 arithmetic oracle. Benchmark
cosine/relative-L2 comparisons against BF16 measure approximation error, not
bitwise equivalence or model-level quality. `--query-rms 1` provides a stronger
query stress case. `--relative-error-limit` optionally applies a BF16-comparison
budget; correctness is always checked against the selected arithmetic's oracle.
MLA timings capture repeated native invocations in one graph to exclude gaps
from a Python loop submitting tiny graphs.

Switch either command to `--model-profile deepseek-v4-flash` for the native
FP8 C4 indexer or V4 SWA/C4/C128 MLA forms. Their cache formats and arithmetic
are different workloads, not interchangeable implementations. `--model-path`
and `--vllm-path` select explicit config and integration checkouts.

Each profile checks its quantization-aware oracle, graph mutations, frozen
kernel resolution, fixed storage, and replay allocations before reporting raw
samples. High physical page IDs beyond a 2-GiB byte offset are enabled by
default. `--page-stride` supplies an allocator stride; otherwise native payload
alignment is used. The indexer workload shares a physical prefix between
queries; MLA gives each request separate cache pages. JSON records those
contracts, active/planned sizes, source hashes, GPU state, and timing scope.
The older synthetic indexer and V4.1 MLA diagnostic scripts remain available.

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
