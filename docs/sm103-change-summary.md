# SM103 features and required fixes

Status: **implemented prototype; physical SM103 execution unqualified**.
SM103/B300 support uses the existing b12x planning, binding and execution APIs.
Core compute uses CuTe DSL. Supporting packing and metadata kernels may use
Triton. The [readiness report](sm103-readiness-report.md) records qualification
limits, and the [implementation log](sm103-implementation-log.md) preserves
source-specific decisions and validation.

The b12x implementation checkpoint is `2958ad63`. Companion vLLM checkpoint
`f6c6ac72c3` includes the integration and pooling fixes; its runtime source is
`ac96719947`. Companion changes reside in that repository, with their evidence
bound by the [GLM receipt](sm103-glm-sparse-validation.json).

## Features added

| Feature | Implementation entry points |
| --- | --- |
| SM103 identity, capabilities, architecture target and typed planning | [Architecture descriptor](../b12x/_lib/architecture.py), [policy contracts](../b12x/policy/) and component capability metadata. Unsupported selections fail closed. |
| Native NVFP4 fused MoE | [SM103 MoE planning](../b12x/moe/fused_moe/_sm103.py) and [CuTe kernels](../b12x/moe/_shared/kernels/sm103/): routing, TMA/tcgen05/TMEM projections, activation quantization and weighted reduction. |
| Trellis expert execution | [Trellis planner](../b12x/moe/fused_moe/_sm103_trellis.py) and [inline projections](../b12x/moe/_shared/kernels/sm103/trellis_gemm.py): uniform, mixed-rate, grouped atom and BTX paired weights, including coupled transforms and codebook formats. |
| Quantized and unquantized projections | [Block-scaled GEMM](../b12x/gemm/blockscaled/), [SM103 projection engine](../b12x/gemm/_shared/sm103_blockscaled.py) and [DeepSeek WO](../b12x/gemm/wo_projection/): FP4/FP6/FP8, inline dequantization and caller-owned workspace. |
| GLM sparse attention and DeepSeek attention/indexing | [GLM sparse MLA](../b12x/attention/sparse_mla/_sm103.py), [compressed MLA](../b12x/attention/compressed_sparse_mla/), [dense MLA](../b12x/attention/dense_mla/) and [DSA indexer](../b12x/attention/dsa_indexer/). |
| Recurrent and model-supporting operators | [KDA/GDN decode](../b12x/sequence/gdn_decode/), [shared recurrent prefill](../b12x/sequence/_shared/delta_prefill/), [MTP feedback](../b12x/sequence/mtp_feedback/) and [mHC](../b12x/norm/mhc/), plus V4.1 compression, HyperConnection and embedding support. |
| Station memory and communication | [Engram storage](../b12x/sequence/engram/_storage.py) supports explicit placement, resident scales and bounded disk prefetch. [RoCE transport](../b12x/comm/roce/_transport.py) implements explicit experimental Grace TP2 selection and its graph lifecycle. |
| Reproducible bring-up tooling | [Offline compiler](../scripts/compile_sm103.py), [qualification launcher](../scripts/qualify_sm103.py), [resource auditor](../scripts/audit_sm103_resources.py), [common-PTXAS diagnostic](../scripts/audit_sm103_packing_ptxas.py), tests and [Station runbook](sm103-qualification.md). |

## Required b12x fixes

These changes address correctness, legal launches or serving invariants. Their
commit references identify the implementation and associated regression tests.
Paths in this table are relative to the `b12x/` package.

| Problem | Fix and source |
| --- | --- |
| Warmup and binding could resolve different capacity or precision choices | Retain one plan-time policy resolution and capacity lowering through warmup, prewarm and binding in `moe/fused_moe/_sm103.py` (`652621da`). |
| TMEM reads require explicit completion before consumers use the results | Add completion waits in `gemm/_shared/sm103_blockscaled.py` and `moe/_shared/kernels/sm103/trellis_gemm.py`; inspect generated artifacts for the waits (`7e743dc7`). |
| Routed prefill exceeded the 65,535 limit of the CUDA Y grid dimension | Launch routed GEMM, quantization and reduction along the X axis in the SM103 MoE launch and pointwise kernels (`725afdae`). |
| Cooperative MoE grids could exceed actual compiled residency | Bound launch grids using function occupancy and actual resources in `_lib/cooperative.py` and `moe/fused_moe/_impl.py`; verify cached resource metadata (`172438b8`). |
| FP16 split outputs could use BF16 atomic accumulation | Restrict BF16 atomics to BF16 outputs and reduce FP16 partials with CuTe in `_lib/dense_gemm.py`; retain FP8 serving buffers in `gemm/blockscaled/_fp8_workspace.py` (`1b675b5c`). |
| Long-K SM120 MXFP4 scale fragments did not match the mainloop's expected modes | Normalize trailing scale-fragment modes in `_lib/dense_gemm.py` while preserving their order and quantization math (`070681c5`). |
| Weight preparation and tensor-parallel extents could disagree with Trellis execution | Preserve global draw coordinates and distinct input-scale halves; keep BTX records whole at rank boundaries; select compatible tiles for non-256-divisible extents (`3e1e77db`, `b7d2a04b`, `fdfe284e`, `2e6c166e`). |
| Clamped Trellis execution could lose transform precision or resolve another callable during serving | Preserve FP16 transform boundaries and retain launches by immutable transform layout in the W4A16 kernels and fused MoE implementation (`32531ecd`). |
| Native sparse MLA split selection depended on live row counts | Resolve and serialize splits during planning in `attention/sparse_mla/_policy.py`; config schema 3 applies to both backends. Native profile coverage remains empty until the fixed schedule is requalified (`ebace59f`). |
| Caller-owned scratch and launch schedules were incomplete for some serving paths | Retain FP8, WO, compressed MLA and paged-indexer capacities and buffers before capture (`1b675b5c`, `381fe5e8`, `84ec51d2`, `534ba168`). |

## Required companion vLLM fixes

Paths below are relative to the companion vLLM repository. They are integration
changes required by the real GLM serving tests, rather than b12x kernel files.

| Problem | Fix and source |
| --- | --- |
| GLM partial-pool selections after interior padding were omitted from attention | Compact valid selections into a stable prefix in `vllm/v1/attention/backends/mla/sparse_utils.py` and enable it in `b12x_mla_sparse.py`. Preserve DCP ordering (`4e981f168d`, regression `fcd1b70d1a`). |
| Rejected verifier tokens overwrote the preceding partial pool | Retain three preceding rows plus the planned verifier length, using absolute-position ring indexing in `vllm/models/glm5next/nvidia/pooled_indexer.py` and `ops/glm_kpool.py`. Ordinary and speculative capacities cover 4/11/18 rows (`e1587ea558`). |
| Packed page conversion could turn an unmapped `-1` slot into a page-zero write | Test original slot validity before conversion in `vllm/models/glm5next/nvidia/ops/glm_kpool.py` (`413bc2c00c`, `ac96719947`). |

Pool-scaled addressing uses Int64. The regressions include high physical page
IDs, independent numerical oracles, frozen compilation, input mutation, stable
addresses and graph replay without allocation.

## Evidence and limits

The recorded b12x corpus compiles 1,225 callables and 1,235 CUDA entries for
SM103. The companion metadata corpus compiles 27 variants. All 27 deferred
operator suites collect, selecting 1,461 cases including overlap. Resource
increases and stack/local-memory flags remain recorded for physical profiling.

The final GLM component source passes 53 cases under Compute Sanitizer in two
complementary processes, with zero errors. A combined run has three allocation
failures after 50 passes and remains failed evidence. Real SM121 graph and
DFlash runs pass the fixed requests and serving checks, but checkpoint accuracy
and repeated generated text remain unqualified. Native V4.1 results do not
qualify converted Trellis checkpoint accuracy.

Physical B300 correctness and performance, Grace/NIC ordering and Station TP2
remain unqualified. Separate SM103 tiny-M and TMEM-pipelined MoE strategies, SM103 GDN
chunk-parallel prefill, direct HBM RDMA and frozen QSRT coupled high-rate
conversion remain unimplemented. No B300 performance claim or measured B300
profile is included.
