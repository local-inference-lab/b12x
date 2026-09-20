# SM103 preparation readiness

Status: **implemented prototype; physical SM103 execution unqualified**.
The working branch is based on master revision `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`.
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
configuration and explicit engine hooks. The [slot exchange contract](expert-residency-slots.md)
specifies opt-in fixed-address replacement at a scheduler pause. The
[shared residency subsystem](expert-residency-subsystem.md) separates host
contracts/policy from the SM103 execution adapter.
The [SM120 NVFP4 cache experiment](expert-residency-sm120-poc.md) composes another
numerical recipe and PCIe storage topology at benchmark scope. Its native SM120
results do not qualify SM103 kernels or Grace-backed TMA.
Its [PCIe cost diagnosis](expert-residency-sm120-costs.md) corrects unused lookup
staging in the shared W4A16 kernel and compares host allocation modes. SM103
residency preparation retains its existing memory policy and physical gates;
the SM120 measurements are not attributed to Grace.

The [routing-locality and canonical-fill experiment](expert-cache-evolution.md)
adds held-out real-route analysis and a recoverable SM120 fill prototype at
benchmark scope. It leaves shared policy and static SM103 preparation unchanged.
Cheaper PCIe fills do not establish profitable adaptation or Grace-backed TMA
legality; physical B300 qualification remains required.

The [held-out policy evaluation](expert-cache-policy-evaluation.md) retains
longer routing traces and additional-layer SM120 comparisons separately from
the compiler and SM103 qualification receipts below. It adds no native kernel
or preparation contract and does not qualify Grace-backed execution.

The [model-wide epoch prototype](expert-residency-epochs.md) adds bounded
cross-layer decisions, experimental decayed LFU and a fail-closed vLLM
pause/worker-RPC adapter. The [prepared SM120 serving cache](expert-cache-serving.md)
adds an opt-in CPU-source NVFP4 loader, canonical host backing, fixed VRAM slots
and real single-rank vLLM graph execution. Static and adaptive experiments start
from the same learned profile. Their serving receipts include scheduler pauses;
prior operator timings remain tied to their original source.

The [scheduler-maintenance experiment](expert-cache-maintenance.md) adds an
explicit single-rank worker-local control boundary, routing-pressure gating and
separate counters-only measurements. It retains scheduler/device drain and
fail-closed reload semantics. Native SM103 execution, distributed serving and
production adaptive defaults remain unqualified.

The [cadence investigation](expert-cache-cadence.md) isolates the generation-zero
C4 mismatch to batch-dependent BF16 router reduction using static and ordinary
non-cache controls. Controlled admission retains an exact output gate for
source-built serving comparisons. This adds diagnostic tooling, not an SM103
kernel, preparation contract or B300 qualification. The compiler inventory below
remains tied to its recorded source; these diagnostic changes require no new
program declaration.
The source-built controlled-admission matrix passes exact paired outputs across
24,576 tokens and 2,016 promotions at C1/C4/C8. Backoff reduces measured stable
overhead to roughly 1–2%, but delays pressure detection by about five seconds
relative to the workload transition. The cadence report retains the complete
interval tables, 64 host / 21 GPU / 10 companion test passes and all failures.

The SM103 preparation corpus at package SHA256
`667fad34f8bb0ba5ad048154e2cfd6281ebe206a19b1b8e32a69fd34e89e0d6b`
contains **86 declarations and 244 distinct programs**, including 238 native CuTe
exports and six supporting Triton programs. The additional declaration exercises
an optional prepared counter extent/phase setter; it does not add an SM103 cache
transport. The [ledger](expert-residency-ledger.md) retains source archives,
compiler manifests, test results and failed attempts. This is compilation
evidence, not physical B300 evidence.

| State | Evidence and limits |
| --- | --- |
| Implemented and compiled through preparation | The serving-cache source corpus covers 86 declarations and 244 distinct SM103 programs with CUDA uninitialized; 238 native CuTe exports are retained. The historical `94639562` corpus remains 85 declarations/241 programs at its own source. It covers dense recipes, packed projections, WO, vocabulary projection, MTP, recurrent decode, attention, mHC, NVFP4 and hierarchical MXFP4 MoE, and HyperConnection. |
| Hierarchical MXFP4 expert residency implemented | Immutable per-layer profiles and memory budgets prepare HBM and exact-size mapped Grace slabs. Native MXFP8/MXFP4 projections consume compact tier-local routes and produce unfinalized expert outputs for one original-top-k-order FP32 FMA reduction. Checkpoint weight bytes remain unchanged. |
| Quiescent slot exchange implemented | Declared rollback journals, disjoint canonical-ID pairs, synchronized payload/map commit, stale-generation rejection and fail-closed rollback preserve captured addresses. Portable byte-replay tests pass; SM103 TMA and native-operator exchange parity remain physical gates. |
| Shared residency subsystem implemented | Standard-library-only placement, observation and generation contracts plus host cache policy; the SM103 adapter supplies numerical admission and copy accounting. Existing imports and schema-1/schema-2 profile hashes are preserved. An opt-in canonical-host W4A16 backend has separate SM120 serving evidence; it does not qualify native SM103 execution. |
| Experimental cache policy implemented | Recent-frequency remains the default; explicit decayed LFU retains window history. Canonical counts identify cold selections for one unchanged generation. Model-wide pair/byte budgets select subsets across layers. Static profiles/defaults remain unchanged. SM120 serving benefit is workload-dependent; physical SM103 benefit remains unqualified. |
| Automatic residency implemented | Typed off/profile/auto/monitor modes, balanced cold-start placement, joint HBM/Grace admission, conservative activation, checkpoint/workload/recipe validation, atomic profiles, windowed convergence and drift diagnostics. Serving placement stays static; activation requires an engine-controlled restart. |
| Prepared counter profiling implemented | CuTe uint64 counters use retained programs and stable storage. Off has no counter node. The SM120 V2 companion supplies explicit decode phase and valid rows through a retained device setter; native SM103 serving wiring remains required. |
| Implemented and tested on SM120 | Residency component tests cover quantization, route compaction, ordered-FMA adversaries, invalid int64 IDs, mapped-host reads, live-count reuse and allocation-free CUDA graph replay. Prepared counter tests add sampling, TP ownership, overflow and lifecycle coverage. These tests do not execute SM103 tcgen05 kernels or establish Grace-backed TMA legality. |
| Implemented, awaiting physical SM103 qualification | Native tcgen05/TMEM dense and expert kernels, HBM/Grace complete-operator parity, Grace-backed TMA operands, architecture launch/resource behavior, complete model execution, chunk-parallel GDN prefill and experimental Station TP2 communication. |
| Independent PR CI pending | Source-bound local receipts are separate from CI acceptance. The repository's wheel-release workflow has no pull-request trigger. |
| Companion integration | The maintained PreparationSession base has an opt-in SM120 CPU-source loader and prepared canonical cache in `codex/b12x-expert-cache`. Native SM103 loading/serving remains deferred. The [serving guide](expert-cache-serving.md) separates supported scope, configuration and measured evidence. |
| Unsupported or research-only | Separate tiny-M and pipelined-TMEM MoE strategies, concurrent residency adaptation, distributed cache-loader serving, measured HBM/Grace overlap, direct HBM RDMA and frozen QSRT coupled high-rate conversion. |

Trellis offline declarations exercise production compiler factories using
canonical weight metadata. They do not perform checkpoint preparation on the
CPU. Canonical GPU weight preparation and portable SM120 expert execution are
separate tests. BTX GPU preparation tests use SM103 admission metadata on SM120 solely to exercise byte preparation and the independent oracle; they execute no native SM103 expert kernel.

## Source-bound validation

Scheduler-maintenance validation binds the package hash above. The focused host
suite passes **86 tests**, with six GPU-only skips; the complete offline SM103
preparation corpus passes **86 declarations / 244 programs**. The
[maintenance guide](expert-cache-maintenance.md) and
[serving-control ledger](expert-residency-ledger.md#scheduler-owned-maintenance-and-serving-control-costs)
record the separate physical SM120 serving, kernel and engine-build gates.
The focused physical suite passes 25 cases and targeted memcheck/synccheck;
the source-built companion passes ten engine/loader cases plus adaptive and
ordinary graph-serving smoke tests. The five MXFP6 registry failures reproduce
on untouched `b067db4` and remain a separate unresolved gate.
The extraction results below remain historical evidence for their own source.

The shared-residency extraction evidence binds package SHA256
`2400c738ec87ea2ac71e21a426b1de4f315c66e6e05a642578c1828f677e60d1`.
The [engineering ledger](expert-residency-ledger.md#shared-residency-extraction-evidence)
records commands, source manifests, toolchains and receipts. Validation used
edits atop `181e234b`; the package hash identifies the tested implementation
independently of documentation and the subsequent commit.

| Gate | Result and scope |
| --- | --- |
| Host preparation, architecture and selected MoE suites | 1,052 passed, 66 skipped, including 15 shared-contract/import/artifact tests, 16 recent-frequency policy tests and exchange/automatic-residency regressions. |
| Repository-wide registry check | 4 passed, 5 failed; the same five MXFP6 metadata/registry failures reproduce on untouched `181e234b`. This separate gate is unresolved. |
| Portable residency, counters, exchange and policy loop | 22 passed, 11 physical-SM103 skips on SM120. The policy tests run existing counters, partitioner and slot copies through changing workloads, preserving payload bytes, graph identity, addresses and allocator counters. |
| Policy-loop sanitizers | 2 passed, 2 SM103-only skips under each of memcheck and synccheck, zero errors. Portable byte probes do not execute native SM103 expert MMA. |
| Fresh focused SM103 compilation | Static residency, update-enabled residency and routing profiling: 3 declarations, 14 distinct programs/native CuTe exports, CUDA uninitialized. Extraction adds no kernel and changes no preparation query. |
| Historical full compiler/resource census | Package `dbc81a14…` at `94639562` retains 85 declarations/241 programs/235 native exports and 12 production-geometry residency entries. FC1/FC2 used 142/140 registers, 1,024 static plus 51,328 dynamic SMEM bytes and no stack/local memory. This full census was not repeated for the shared host contracts/policy. |
| Historical policy/attention validation | Package `6b1c98b0…` retains its separate host/attention and counter sanitizer receipts. These runs are not attributed to the shared-residency source. |
| Historical portable profiler diagnostic | Package `aac5c587…` measured off/on/sampled partition stages for M=1 through 128 and top-k=1/6/8. These measurements motivate explicit profiling; they are not B300/full-MoE or adaptive-cache timings. |

The [cache guide](expert-residency-cache.md) separates cold observations,
counterfactual placement scores, observed post-promotion hits and unmeasured
performance. Native SM103 policy-loop tests and sanitizer commands are ready;
physical B300 execution, Grace TMA and cache benefit remain deferred.

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

GitHub reports zero commit statuses and zero check runs for the reviewed starting
source `b067db404e8b2dd22855482eaa11bff68c631342` and companion
`1d1f870bd617a4905637a30fcb552859b9fb2ded` as of September 19, 2026. The
[wheel-release workflow](../.github/workflows/lil-cu134-wheel-release.yml)
runs on selected branch/tag pushes or manual dispatch; it does not run on pull
requests. A PR test workflow must be enabled and pass for the reviewed PR source
as an independent gate. Local receipts do not satisfy that gate.

Physical SM103 tests must establish complete expert execution, HBM/Grace parity,
TMA legality on Grace pages, graph replay and performance. A compatible companion
serving run must establish checkpoint behavior. Historical GLM repeatability and
accuracy gates remain unresolved. Neither CI nor portable tests replace physical
SM103 and serving qualification.
