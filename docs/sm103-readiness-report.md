# SM103 preparation readiness

Status: **implemented prototype; physical SM103 execution unqualified**.
The SM103 implementation at `78a8704f7d571b9869b29f3f3d5a03831d160152` is
rebased onto master revision `8783519a3e42c22c0f395669ca4b20c69439c974`.
Plans declare typed geometry and capacity; `PreparationSession` compiles,
materializes and primes execution before binding or capture. Hierarchical
MXFP4 expert residency uses this same lifecycle to prepare HBM and coherent
Grace storage. There is no measured B300 performance result or B300 tuning winner.

The [feature and fix map](sm103-change-summary.md) identifies the implementation
and compatibility changes. The [qualification runbook](sm103-qualification.md)
separates offline evidence, portable GPU checks and physical-target acceptance.
The [expert residency contract](expert-residency.md) and
[engineering ledger](expert-residency-ledger.md) describe placement, numerical
boundaries, memory admission, exact qualification commands and retained failures.

| State | Evidence and limits |
| --- | --- |
| Implemented and compiled through preparation | The source-bound offline corpus exercises 83 declarations and compiles 239 distinct SM103 programs with CUDA uninitialized. It covers dense recipes, packed projections, WO, vocabulary projection, MTP, recurrent decode, attention, mHC, NVFP4 and hierarchical MXFP4 MoE, and HyperConnection. |
| Hierarchical MXFP4 expert residency implemented | Immutable per-layer profiles and memory budgets prepare HBM and exact-size mapped Grace slabs. Native MXFP8/MXFP4 projections consume compact tier-local routes and produce unfinalized expert outputs for one original-top-k-order FP32 FMA reduction. Checkpoint weight bytes remain unchanged. |
| Implemented and tested on SM120 | Residency component tests cover quantization, route compaction, ordered-FMA adversaries, invalid int64 IDs, mapped-host reads, live-count reuse and allocation-free CUDA graph replay. W4A16 and pooled-selection regression tests also pass. These tests do not execute SM103 tcgen05 kernels or establish Grace-backed TMA legality. |
| Implemented, awaiting physical SM103 qualification | Native tcgen05/TMEM dense and expert kernels, HBM/Grace complete-operator parity, Grace-backed TMA operands, architecture launch/resource behavior, complete model execution, chunk-parallel GDN prefill and experimental Station TP2 communication. |
| Independent PR CI pending | Commit `78a8704f` has no GitHub statuses or check runs. Source-bound local receipts are separate from CI acceptance. The repository's wheel-release workflow has no pull-request trigger. |
| Companion integration requires a port | The retained vLLM branch targets the preceding b12x interface. Its historical checkpoint and loader results are not evidence for the preparation API. |
| Unsupported or research-only | Separate tiny-M and pipelined-TMEM MoE strategies, online residency adaptation, measured HBM/Grace overlap, direct HBM RDMA and frozen QSRT coupled high-rate conversion. |

Trellis offline declarations exercise production compiler factories using
canonical weight metadata. They do not perform checkpoint preparation on the
CPU. Canonical GPU weight preparation and portable SM120 expert execution are
separate tests. BTX GPU preparation tests use SM103 admission metadata on SM120 solely to exercise byte preparation and the independent oracle; they execute no native SM103 expert kernel.

## Source-bound validation

The September 18 evidence binds package SHA256
`a3e65748153cb354026e3053fdb255cdd3693a6ca6d01baa16c64e0b59ef5bd9`
to implementation commit `78a8704f`. The
[engineering ledger](expert-residency-ledger.md#validation-evidence) records
commands, toolchains, device identity and receipt locations.

| Gate | Result and scope |
| --- | --- |
| Host preparation, architecture and selected MoE suites | 930 passed, 63 skipped. Skipped physical-target cases are not counted as passes. |
| Portable residency tests under Compute Sanitizer | 8 passed on SM120, zero sanitizer errors. Covers metadata, quantization, finalization, graph replay and mapped-host reads. |
| W4A16 and pooled-selection GPU regressions | 2 passed, 45 deselected on SM120. |
| Preparation compiler corpus | 83 declarations and 239 distinct programs cross-compiled without initializing CUDA. |
| Production-size hierarchical residency compiler matrix | 12 entry points compiled for H=5120, I=2304, E=384, HBM=295, capacity=128 and top-k=6. These values are qualification geometry, not API constants. |
| All-HBM and all-Grace compiler variants | 10 entry points each; preparation omits the empty tier. |

Production-size residency FC1/FC2 use 142/140 registers, 1,024 static shared-memory
bytes and 51,328 dynamic shared-memory bytes, with no stack or local-memory
spills. The artifacts contain native mixed blockscaled MMA and TMEM completion
waits. These resource results establish no measured occupancy or performance.

Raw local logs, compile caches and native artifacts remain outside the repository.
The [preparation-port receipt](sm103-preparation-validation.json),
[master-integration receipt](sm103-master-integration-validation.json),
[GLM receipt](sm103-glm-sparse-validation.json) and
[implementation log](sm103-implementation-log.md) retain evidence and failures
for their own source hashes. Their test totals, native resource census, wheel
contents and checkpoint results are not validation totals for `78a8704f`.

## Remaining acceptance gates

GitHub reports zero commit statuses and zero check runs for `78a8704f` as of
September 18, 2026. The
[wheel-release workflow](../.github/workflows/lil-cu134-wheel-release.yml)
runs on selected branch/tag pushes or manual dispatch; it does not run on pull
requests. A PR test workflow must be enabled and pass for the reviewed PR source
as an independent gate. Local receipts do not satisfy that gate.

Physical SM103 tests must establish complete expert execution, HBM/Grace parity,
TMA legality on Grace pages, graph replay and performance. A compatible companion
serving run must establish checkpoint behavior. Historical GLM repeatability and
accuracy gates remain unresolved. Neither CI nor portable tests replace physical
SM103 and serving qualification.
