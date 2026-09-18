# SM103 preparation readiness

Status: **implemented prototype; physical SM103 execution unqualified**.
The working branch is based on master revision `8783519a3e42c22c0f395669ca4b20c69439c974`.
Plans declare typed geometry and capacity; `PreparationSession` compiles,
materializes and primes execution before binding or capture. Hierarchical
MXFP4 expert residency uses this same lifecycle to prepare HBM and coherent
Grace storage. Opt-in calibration adds model-wide budgeting, routing counters,
validated workload profiles and controlled restart signaling. There is no
measured B300 performance result or B300 tuning winner.

The [feature and fix map](sm103-change-summary.md) identifies the implementation
and compatibility changes. The [qualification runbook](sm103-qualification.md)
separates offline evidence, portable GPU checks and physical-target acceptance.
The [expert residency contract](expert-residency.md) and
[engineering ledger](expert-residency-ledger.md) describe placement, numerical
boundaries, memory admission, exact qualification commands and retained failures.
The [automatic SM103 guide](expert-residency-automatic.md) documents the lifecycle,
configuration and explicit engine hooks.

| State | Evidence and limits |
| --- | --- |
| Implemented and compiled through preparation | The source-bound offline corpus exercises 84 declarations and compiles 241 distinct SM103 programs with CUDA uninitialized. It covers dense recipes, packed projections, WO, vocabulary projection, MTP, recurrent decode, attention, mHC, NVFP4 and hierarchical MXFP4 MoE, and HyperConnection. |
| Hierarchical MXFP4 expert residency implemented | Immutable per-layer profiles and memory budgets prepare HBM and exact-size mapped Grace slabs. Native MXFP8/MXFP4 projections consume compact tier-local routes and produce unfinalized expert outputs for one original-top-k-order FP32 FMA reduction. Checkpoint weight bytes remain unchanged. |
| Automatic residency implemented | Typed off/profile/auto/monitor modes, model-wide byte allocation, checkpoint/workload/recipe validation, atomic profiles, windowed convergence and drift diagnostics. Serving placement stays static; activation requires an engine-controlled restart. |
| Prepared counter profiling implemented | CuTe uint64 counters use retained programs and stable storage. Off has no counter node. External/native routing integration uses explicit worker hooks; the companion vLLM port remains required. |
| Implemented and tested on SM120 | Residency component tests cover quantization, route compaction, ordered-FMA adversaries, invalid int64 IDs, mapped-host reads, live-count reuse and allocation-free CUDA graph replay. Prepared counter tests add sampling, TP ownership, overflow and lifecycle coverage. These tests do not execute SM103 tcgen05 kernels or establish Grace-backed TMA legality. |
| Implemented, awaiting physical SM103 qualification | Native tcgen05/TMEM dense and expert kernels, HBM/Grace complete-operator parity, Grace-backed TMA operands, architecture launch/resource behavior, complete model execution, chunk-parallel GDN prefill and experimental Station TP2 communication. |
| Independent PR CI pending | Source-bound local receipts are separate from CI acceptance. The repository's wheel-release workflow has no pull-request trigger. |
| Companion integration requires a port | The retained vLLM branch targets the preceding b12x interface. Its historical checkpoint and loader results are not evidence for the preparation API. |
| Unsupported or research-only | Separate tiny-M and pipelined-TMEM MoE strategies, online residency adaptation, measured HBM/Grace overlap, direct HBM RDMA and frozen QSRT coupled high-rate conversion. |

Trellis offline declarations exercise production compiler factories using
canonical weight metadata. They do not perform checkpoint preparation on the
CPU. Canonical GPU weight preparation and portable SM120 expert execution are
separate tests. BTX GPU preparation tests use SM103 admission metadata on SM120 solely to exercise byte preparation and the independent oracle; they execute no native SM103 expert kernel.

## Source-bound validation

The automatic-residency evidence binds package SHA256
`aac5c587557754e1d3f4f59da6941f15bdb7cf41e715a05205288543f08c77c7`.
The [engineering ledger](expert-residency-ledger.md#automatic-residency-evidence) records
commands, toolchains, device identity and receipt locations.

| Gate | Result and scope |
| --- | --- |
| Host preparation, architecture and selected MoE suites | 977 passed, 63 skipped. Skipped physical-target cases are not counted as passes. |
| Prepared counter and worker lifecycle GPU gates | 5 passed under memcheck and 5 under synccheck on SM120, zero errors; exact counts, replay, allocator, ownership, overflow and restart/profile-reuse checks. |
| Portable profiler diagnostic | Off/on/sampled partition stages measured for M=1 through 128 and top-k=1/6/8, plus all-to-one contention. Separate launch overhead is material at tiny M; these are not B300 or full-MoE timings. |
| Preparation compiler corpus | 84 declarations and 241 distinct programs cross-compiled without initializing CUDA. |
| Production-size hierarchical residency compiler matrix | 12 entry points compiled for H=5120, I=2304, E=384, HBM=295, capacity=128 and top-k=6. These values are qualification geometry, not API constants. |
| Counter compiler variants | Both ID widths compile for SM103 with uint64 atomics and sticky overflow detection; live M/top-k remain launch arguments. |

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
contents and checkpoint results apply to those historical sources. The ledger
also preserves the static-residency baseline at `78a8704f`, including all-hot and
all-cold compiler variants, portable arithmetic and W4A16/pooled-selection tests.

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
