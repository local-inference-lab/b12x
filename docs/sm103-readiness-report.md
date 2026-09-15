# SM103 implementation readiness

Status: **implemented prototype; physical SM103 execution unqualified**.
The prototype uses the existing b12x planning, binding and execution APIs, with
companion vLLM integration. It is ready for B300 bring-up. GLM checkpoint accuracy
remains unresolved on the available SM121 systems, so the branch is not a
qualified serving release. No B300 performance result or measured B300 policy
profile is claimed.

The [qualification runbook](sm103-qualification.md) contains exact bring-up
commands and acceptance conditions. The [GLM validation receipt](sm103-glm-sparse-validation.json)
binds component, checkpoint and compilation results to their respective sources;
the [implementation log](sm103-implementation-log.md) preserves development history.

| State | Evidence and limits |
| --- | --- |
| Implemented and validated | Host policy/layout checks, SM120/SM121 component oracles and graph tests, and bounded native V4.1 checkpoint checks. GLM has passing component regressions and fixed-request checks, with unresolved checkpoint accuracy gates. |
| Implemented and cross-compiled | 1,225 b12x callables with 1,235 CUDA entries targeting SM103, plus 27 companion GLM metadata variants. Five ARM64 core and five external libraries compile and load with CUDA uninitialized. |
| Implemented, awaiting B300 execution | Native tcgen05/TMEM MoE and quantized projections, Trellis expert execution, portable attention/recurrent paths, Grace Engram placement and experimental Station TP2 communication. |
| Designed, not implemented | Separate tiny-M and TMEM-pipelined MoE strategies, GDN chunk-parallel prefill, direct HBM RDMA transport and frozen QSRT coupled high-rate conversion. Unsupported selections fail closed. |
| Hardware or checkpoint inputs required | B300 numeric/resource/performance qualification, Station memory/NIC visibility and ordering, and full-model accuracy for a provenance-bound V4.1 Trellis checkpoint. |

1. **SM103 architecture added.** Architecture recognition, capability checks,
   code generation, typed policy resolution and scratch planning select SM103
   through ordinary public APIs. Architecture-specific cubins target
   `sm_103a`; SM12x retains its instruction paths.
2. **Code implemented.** Native NVFP4 and Trellis MoE, quantized and unquantized
   projections, sparse and compressed MLA, DSA, KDA/GDN, mHC, MTP feedback,
   supporting V4.1 operators, Engram placement and experimental Grace TP2
   collectives are integrated. Companion vLLM supplies metadata and capacities;
   b12x owns policy. GLM fixes preserve partial-pool selections and speculative
   history, reject unmapped cache writes and fix sparse MLA splits at plan time.
3. **Tests passing.** The final GLM pooling source passes 53 suite cases under
   Compute Sanitizer in complementary eight- and 45-case processes, with zero
   errors. A combined run has three allocation failures after 50 passes and
   remains failed evidence. Separate attention/remapper tests cover native and
   portable paths, high page IDs, live counts 1/3/26, frozen resolution, stable
   addresses and allocation-free replay. Policy/companion host checks total
   115/87 passed. The wheel passes 88 host and four GPU checks; two generator
   checks additionally require the checkout's benchmark drivers.
4. **SM103 compilation status.** All 1,225 corpus callables compile. All 27
   deferred suites collect, selecting 1,461 cases including suite overlap.
   Inspection retains 44 stack/local flags and eight packing register increases;
   a common-PTXAS diagnostic preserves the packing differences with unchanged
   quantizer source. The 27 companion metadata variants add one separate flag:
   packed decode uses 56 rather than 48 registers after the slot-validity fix,
   with no stack/local memory. Physical profiling must assess these flags.
5. **GLM-5.3 Flash readiness.** The real 45-layer NVFP4 checkpoint passes 11
   fixed requests and produces the same 39 tokens in eager, graph and DFlash
   modes on four SM121 workers. GSM8K scores are 29/32, 30/32 and 25/32; DFlash
   fails the declared 29/32 accuracy floor. A history-only graph control scores
   27/32 and also fails that floor. On the final pooling source, repeated serial
   graph runs score 26/32 and 28/32, and C4 scores 27/32. DFlash scores 26/32,
   28/32 and 28/32 in the same sequence. Serial repeated-text matches are only
   4/32 for graphs and 3/32 for DFlash, with prefix caching enabled. This study
   does not isolate a DFlash-specific regression or waive earlier failed gates.
   Frozen resolution, graph replay and accepted draft proposals are observed;
   checkpoint accuracy remains unqualified.
6. **Trellis/V4.1 readiness.** Native V4.1 checkpoint execution passes bounded
   SM121 checks. Uniform, mixed-rate, grouped and BTX paired Trellis expert paths
   are implemented and cross-compiled, with separate reconstruction/component
   evidence. Native-weight results do not establish converted Trellis model
   accuracy or single-Station residency.
7. **RoCEnante Station readiness.** Platform probing, explicit experimental
   Grace TP2 selection, peer protocol and graph lifecycle are implemented.
   Registration, visibility, failure stress and NCCL comparison require the
   Station. The GLM checks use NCCL. Direct HBM transport remains unsupported.
8. **First tests to run on the DGX Station.** Verify GPU UUID, capability,
   opt-in limits and toolchain; execute the prepared operator oracles and
   sanitizer/graph gates; then load GLM for target-only and DFlash C1/C4 checks.
   Follow with Trellis real-weight numerics, Grace Engram lookup and TP2
   collective qualification before measuring serving latency.
9. **Highest-risk unvalidated assumptions.** Single-row TMA bounds, TMEM
   synchronization/lifetime, native quantization tolerances, Trellis calibration,
   Grace allocation visibility and NIC-to-GPU ordering need physical evidence.
   GLM accuracy variation requires diagnosis beyond the reproduced pooling bugs.
   The 32-question sample is a bring-up check, not a model-quality evaluation.
   Qwen DFlash exact token parity remains unresolved separately. The checkpoint
   trials use a compatibility native-library image; source-matched ARM64 library
   loading checks do not qualify those libraries on the GPU.
10. **Next three concrete engineering tasks.** Diagnose GLM accuracy variation
    and execute the physical SM103 operator corpus; qualify GLM and a
    provenance-bound V4.1 Trellis checkpoint with graphs and speculation;
    qualify Grace/RoCEnante TP2, then profile the real paths and generate
    measured policies.
