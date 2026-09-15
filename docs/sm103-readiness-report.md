# SM103 preparation readiness

Status: **implemented prototype; physical SM103 execution unqualified**.
The SM103 branch uses the preparation API based on master revision
`213fc1b204b306bdbaa7d40d2a27529658128bf7`. Plans declare typed geometry and
capacity; preparation compiles, materializes and primes execution before binding
or capture. There is no measured B300 performance result or B300 tuning winner.

The [feature and fix map](sm103-change-summary.md) identifies the implementation
and compatibility changes. The [qualification runbook](sm103-qualification.md)
separates offline evidence, portable GPU checks and physical-target acceptance.

| State | Evidence and limits |
| --- | --- |
| Implemented and compiled through preparation | The offline corpus exercises 82 declarations and exports 221 distinct SM103 CuTe PTX/cubin objects with CUDA uninitialized. It covers dense recipes, packed projections, WO, vocabulary projection, MTP, recurrent decode, attention, mHC, MoE and HyperConnection. |
| Implemented and tested on SM120 | Portable component tests cover independent numerical oracles, quantization, high pool offsets, live-count reuse, full-graph tracing and allocation-free CUDA graph replay. These tests do not execute SM103 tcgen05 kernels. |
| Implemented, awaiting physical SM103 qualification | Native tcgen05/TMEM dense and expert kernels, architecture launch/resource behavior, complete model execution, chunk-parallel GDN prefill, Grace visibility and experimental Station TP2 communication. |
| Companion integration requires a port | The retained vLLM branch targets the preceding b12x interface. Its historical checkpoint and loader results are not evidence for the preparation API. |
| Unsupported or research-only | Separate tiny-M and pipelined-TMEM MoE strategies, direct HBM RDMA and frozen QSRT coupled high-rate conversion. |

Trellis offline declarations exercise production compiler factories using
canonical weight metadata. They do not perform checkpoint preparation on the
CPU. Canonical GPU weight preparation and portable SM120 expert execution are
separate tests. BTX GPU preparation tests use SM103 admission metadata on SM120 solely to exercise byte preparation and the independent oracle; they execute no native SM103 expert kernel.

Host checks report 877 passes, 53 skips and one baseline failure. The SM120
batches report 1,093 passes, 110 skips and three baseline failures. A pristine
checkout of the stated master revision reproduces the device-reclaim cache
expectation, mHC program-sharing expectation and first-use warning-key failures.
No failing or skipped case is counted as a pass. The preparation-specific architecture tests
exercise all corpus declarations without initializing CUDA.

The native resource corpus compiles 1,163 callables with 1,165 CUDA entry points.
All 163 TMEM readers include completion waits. The audit retains 42 callables
with stack or local-memory use for physical-target qualification; it has no
matched pre-port baseline and establishes no performance or regression result.

The source-bound preparation validation record is
[sm103-preparation-validation.json](sm103-preparation-validation.json).
Raw local logs, compile caches and native artifacts remain outside the repository.

The [GLM receipt](sm103-glm-sparse-validation.json) and
[implementation log](sm103-implementation-log.md) describe the predecessor
implementation and its source hashes. Their 1,225-callable compilation,
SM120 sanitizer and SM121 checkpoint results must not be attributed to this
source. Historical GLM checkpoint repeatability and accuracy gates remain
unresolved. Physical SM103 testing and a compatible companion serving run are
required before a release can be described as qualified.
