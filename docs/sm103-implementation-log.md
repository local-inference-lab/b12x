# SM103 implementation log

This branch implements a bounded SM103 prototype without access to a B300 or
GB300. CPU tests, SM120 execution, offline compilation, and B300 qualification
are separate evidence categories. No B300 latency or throughput is measured.

## Source and design decisions

- Base: upstream `master`, `9043b448622764a598969518d413b3fd8b3c0c07`.
  Work is isolated in `b12x-sm103` on `feat/sm103-backend`.
- Inspected MoE history: `06b4de7c` (native FC1/shared scratch), `56b13c09`
  (M1 FC2 operand loads), `8648fae3` (materialized prefill). Inspected recurrent
  scheduling history: `64b2b0f3` (independent KDA loads before recurrence).
- `b12x/_lib/architecture.py` separates architecture recognition, instruction
  family, operator admission, and platform coherency. SM100 is recognized but
  has no enabled implementation. Existing synthetic-device policy tests retain
  their heuristic behavior. SM103 plans and CuTe entry points fail closed.
- `moe/fused_moe/_backends.py` selects internal execution through the existing
  canonical and compatibility APIs. `_sm103.py` owns policy validation, fixed
  capacity scratch, binding, and retained launches. Public expert formats remain
  unchanged. Every live count stays outside compile keys.
- `moe/_shared/kernels/sm103/nvfp4_gemm.py` adapts NVIDIA's
  [NVFP4 CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py),
  retaining its BSD notice. CUTLASS supplies block-scale layouts, TMA, pipelines,
  TMEM allocation and UMMA. b12x supplies routing, quantization, activation,
  reduction and lifecycle. One CTA uses one live M row of a 128-row tile. This
  schedule provides a bring-up baseline and wastes tensor-core work at tiny M.
- The materialized strategy is implemented. Monolithic and TMEM-pipelined
  strategies have identifiers but no admissible policy configuration.
- `sm103/trellis.py` reconstructs SQG E4M3 t256 tiles in the quantizer basis.
  `TrellisPipeline` consumes the normal weight plan and rejects full execution
  until scales, rotations, mixed-rate routing and UMMA projections exist.
  K2/K3/K4 use the existing codebook and packed representation.
- MoE config schema advances from 3 to 4 and generator candidate contract from
  17 to 18. Embedded SM12x planner values and measurements are unchanged.
  JSON compression is canonicalized. Missing catalog registrations for existing
  GDN/KDA prefill policies are restored, with empty uncovered profile entries.
  The existing V4.1 benchmark preset is added to model-policy inspection; its
  inspection scope is MoE only.
- `sequence/engram/_storage.py` owns device or mapped-host table allocations and
  loading views. Grace selection requires live coherent-memory capabilities.
  Lookup bindings retain ownership. No row cache, prefetcher or claimed C2C
  bandwidth is added.
- `comm/roce/_transport.py` preserves Spark selection and adds explicit
  experimental Grace TP2 selection. The implementation reuses the pinned-region
  peer protocol and its epoch/timeout behavior. HBM GDR registration is unsupported.
  Factory methods forward selection and stats report its qualification status.

## Validation record

The final command/result ledger and artifact identities are recorded alongside
the qualification guide. Source code and evidence are frozen before the final
offline compile. Raw artifacts are outside the repository at
`../b12x-sm103-evidence/`; temporary development receipts also exist under `/tmp`.

- Host suite: architecture, registry, policy/profile serialization, canonical and
  compatibility planning, scratch, format, compiler runtime and packaging tests.
  An initial broad collection failed because upstream removed
  `validation/cutlass_migration/` in `070af610` while a compiler-cache test still
  imported two modules from it. Four exact historical dependency files from
  `070af610^` were supplied through an isolated `/tmp` PYTHONPATH to run that
  unchanged test. No test is disabled or replaced with a stub.
- Physical regression host: two RTX PRO 4000 Blackwell GPUs, CC 12.0, 24 GB each,
  driver 580.173.02. Tests use the isolated
  `/home/jasonc/b12x-sm103-regression-20260912` checkout on `ripper`.
  The occupied SM121 serving host is left untouched.
- MoE/shared-scale/GDN/resident Engram regression initially passed 83 tests;
  eight disk Engram tests could not build without `liburing` headers. Ubuntu
  development packages were extracted into the isolated test directory, and all
  15 Engram tests then passed. No system package installation was required.
- Portable SM103 quantizers pass four numerical tests on SM120, including
  SwiGLU, gate ordering, invalid IDs and 64-bit IDs outside the expert range.
  Trellis reconstruction passes three exact GPU tests, including short live
  extents under the same capacity specialization. The development failures
  identified signed ring-word extension and an FP8 helper that did not handle
  subnormals; the decoder uses unsigned words and the existing native conversion.
- Additional SM120 checks cover micro Trellis, sparse MLA, and selected KDA
  prefill contracts. Long-sequence KDA stress cases are outside this targeted run.
- The Engram placement benchmark passes exact lookup checks with changing row IDs,
  poisoned outputs and allocation-free graph replay on mapped host memory on
  SM120. Its small-table timings are harness diagnostics, not Station evidence.

Final results:

| Level | Command or suite | Result |
| --- | --- | --- |
| 1 | Combined host command below, including unchanged compiler-cache tests | 466 passed, 8 skipped |
| 1 / deferred 4 | `pytest tests/architecture/test_sm103.py tests/moe/test_sm103_nvfp4.py -q` after adding both gate orders | 26 passed, 4 skipped; all B300 execution cases skipped |
| 2 | MoE, shared-input scales, GDN decode and Engram, after supplying liburing | 91 tests passed across the suite and dependency rerun |
| 2 | Micro Trellis, sparse MLA and selected KDA prefill | 131 passed, 26 deselected |
| 2 | SM103 portable quantization and tile reconstruction on SM120 | 7 passed |
| 2 | Mapped-host Engram benchmark, base table 101, M8, five changing traces | Exact output and allocation-free graph replay passed |
| 3 | `scripts/compile_sm103.py --component all` with disassembly/resource tools | 20 PTX/cubin/MLIR/SASS/resource sets produced |
| 1 / 3 | C proxy build with isolated libibverbs headers/libraries | ABI 3; TP2 layout with two slots, 128-byte flag stride |
| 1 | `uv build --out-dir ../b12x-sm103-evidence/dist` | Wheel and source distribution built; wheel contains SM103 kernels, platform modules and retained BSD notice |

The combined host command is:

```bash
PYTHONPATH=/tmp/b12x-sm103-historical-tools:. .venv/bin/python -m pytest \
  tests/_lib tests/architecture tests/policy tests/test_registry.py \
  tests/moe/test_fused_moe_planning.py tests/moe/test_tp_moe_scratch_bindings.py \
  tests/moe/test_trellis_config.py tests/moe/test_tp_moe_w13_layout.py \
  tests/test_packaging.py -q
```

That historical run used these exact paths from `070af610^` under
`/tmp/b12x-sm103-historical-tools/`:

```text
validation/cutlass_migration/evidence/kernel_resources.py
validation/cutlass_migration/acceptance/corpus/ptx_capture.py
validation/cutlass_migration/core/comparison_identity.py
validation/cutlass_migration/paths.py
```

The retained [artifact integrity tooling](../validation/cutlass_migration/README.md)
supersedes this temporary setup. Clean-checkout commands are recorded below.

The three GPU suite commands, with `CUTE_DSL_ARCH=sm_120a`, are:

```bash
python -m pytest tests/moe/test_fused_moe.py \
  tests/moe/test_nvfp4_shared_input_scales.py tests/sequence/test_gdn_decode.py \
  tests/sequence/test_engram.py -q
python -m pytest tests/moe/test_micro_trellis.py tests/attention/test_sparse_mla.py \
  tests/sequence/test_kda_prefill.py \
  -k 'not long_sequence and not long_memory and not high_state' -q
python -m pytest tests/moe/test_sm103_pointwise.py tests/moe/test_sm103_trellis.py -q
```

Logs and the mapped-memory receipt are retained in `../b12x-sm103-evidence/logs/`.
The compact [compile receipt](sm103-validation.json) records the code revision,
source hash, toolchain, cubin hashes and resource summaries. Its source hash
matches the committed Python source. Projection kernels use 138/140 registers
for int32/int64 IDs; resource reports show zero stack and local memory for all
twenty kernels. Reported static SMEM excludes dynamic launch SMEM, so these
numbers do not establish occupancy. The full manifest and generated artifacts
are identified by the receipt's `artifact_root`. The initial collection remains
archived in `../b12x-sm103-evidence/final-sm103/`.

## Policy and qualification fixes, 2026-09-13

Status: implemented; host contracts and selected SM120 regressions pass.
B300 runtime qualification remains deferred.

Revision `652621da` retains one SM103 capacity lowering and one policy resolution,
including AUTO precision and override provenance. Warmup counts share the
lowering, and prewarm compiles it without repeating policy selection. The SM12x
planning path retains its existing behavior.

The SM103 generator, benchmark and correctness tests share a packaged Torch
oracle for the materialized BF16-stage contract. Its tests cover FP4 ties,
E4M3 saturation/underflow, zero inputs, and analytical FC2/router rounding.
Generator candidate contract version 19 invalidates earlier qualification
checkpoints. The GPU test matrix includes AUTO with up/gate weights.

Revision `1a114fcb` restores the independent manifest validators and their
dependencies. PTX capture uses `b12x._lib.compiler`; raw retention controls,
semantic hashes and object hashes remain intact. Eight offline integrity cases
check current/historical manifest schemas, tamper rejection, separate comparison
identities and hook installation.

The combined host command requires no temporary import path:

```bash
.venv/bin/python -m pytest tests/_lib tests/architecture tests/policy \
  tests/test_registry.py tests/moe/test_fused_moe_planning.py \
  tests/moe/test_tp_moe_scratch_bindings.py tests/moe/test_trellis_config.py \
  tests/moe/test_tp_moe_w13_layout.py tests/test_packaging.py \
  validation/cutlass_migration/integrity_checks tests/moe/test_sm103_nvfp4.py -q
```

Result: **487 passed, 14 skipped**. Six skips require physical B300 execution.
The independent auditor's `--help` command also succeeds from the checkout.

Live inspection found both RTX PRO 4000 Blackwell GPUs idle on `ripper`. The
isolated checkout `/home/jasonc/b12x-sm103-regression-20260912` ran:

```bash
CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ARCH=sm_120a .venv/bin/python -m pytest \
  tests/moe/test_sm103_pointwise.py tests/moe/test_sm103_trellis.py \
  tests/moe/test_nvfp4_auto.py tests/moe/test_fused_moe.py -q
```

Result: **29 passed**. The quantizer cases include zero and saturated inputs.
The physical GPU UUID was `GPU-47363510-b87a-13a5-4824-2542e97df76c`, CC 12.0.
These checks provide no SM103 runtime or performance evidence. Logs use the
`review-fixes-` prefix under `../b12x-sm103-evidence/logs/`.

Offline compilation at clean revision `1a114fcb` produced all twenty
PTX/cubin/MLIR/SASS/resource sets under
`../b12x-sm103-evidence/review-fixes-sm103/`. All cubin hashes match the initial
collection; the fixes do not change device code. The compact compile receipt
binds the collection to package source SHA-256
`e0184490812929bd5687f28f2dd9bb97dc0e4a7aa182c230105cee4add4e80de`.

`uv build --out-dir ../b12x-sm103-evidence/review-fixes-dist` produced a wheel
and source distribution. Inspection verified that the wheel contains the
packaged qualification oracle with identical source bytes.

## Portable operator admission

Status: **implemented, unqualified on SM103**. Revisions `c878e33a` and
`02f40a30` add CuTe KDA decode and admit portable Qwen GDN decode, sequential
KDA/GDN prefill, dense compressed MLA, and unquantized projections. The CuTe
KDA path uses per-coordinate lower-bounded decay and FP32 gated RMSNorm, with
Triton restricted to metadata validation. SM12x keeps its existing KDA default.
GDN config schema is 4; embedded measured configs retain their values. The
GDN and prefill candidate contracts are 3 and 6 respectively. SM103 rejects
Triton recurrent compute and the unqualified chunk-parallel GDN algorithm.

The package source SHA-256 is
`2c787cb1ca067a3468fc4d9a5e988429986f97aea2ead79d5c738d4182c63106`.
The isolated SM120 checkout reports the same hash. Host validation produced
**503 passed, 58 skipped** in `expanded-host.log`; GPU-dependent checks account
for most skips. Final focused architecture/qualification checks produced
**43 passed**. These counts overlap and are not additive.

On physical RTX PRO 4000 Blackwell GPU
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, the recurrent regression suite
produced **308 passed** through the public APIs. Dense MLA/projection
produced **69 passed, 2 skipped**; the skips require an installed combined
FlashInfer/vLLM plugin environment. The KDA suite independently produced
**24 passed**, including smaller bound capacities, frozen kernel resolution,
zero/live counts, BF16/FP32 parameters, strided beta, accepted draft checkpoints,
graph replay, and a state slot beyond the signed 32-bit element-offset boundary.

CUDA 13.4.57 Compute Sanitizer completed the same 24 KDA tests under both
memcheck and synccheck with **zero errors and exit status 0**. The tool was
extracted inside the isolated checkout's `.deps/`; system CUDA was unchanged.
The archive is available from NVIDIA's
[CUDA redistribution index](https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/).
Its SHA-256 is `443465d4dcb77f45a2eb2c9adf2d1e0d7b4d0966ce84acf6d89b4c555bd55f5b`.
One CUDA 12.8 synccheck invocation ended with status 255 after passing its test
assertions and is excluded from qualification; both a repeat and the CUDA 13.4
run completed cleanly. No kernel failure was reported in that interrupted log.

Offline compilation at `02f40a30` produced **83 callables / 91 CUDA entry
points** under `../b12x-sm103-evidence/portable-operators-sm103/`. The corpus
contains 36 recurrent, 17 dense MLA, ten unquantized projection, and the
existing twenty MoE/communication/reconstruction callables. The existing twenty
cubin hashes are unchanged. Dense window cases use the public planner's
single-query tile; unreachable multi-query window combinations are excluded.

KDA recurrence uses 80–92 allocated registers in the sampled variants; its
norm uses 24. Both have zero reported stack/local storage. Four reachable
1088-wide dense MLA variants report 8–192-byte stack frames. These resource
flags remain visible in `docs/sm103-validation.json` and require SM103 profiling.
The reports do not measure dynamic launch SMEM, occupancy, or performance.

`scripts/qualify_sm103.py` prepares a manifest without a CUDA context. Execution
requires `--execute --device-uuid`, verifies physical SM103 identity, rejects
skipped required tests, and records source, tool, GPU, and test identities.
Supplying an offline compile manifest verifies all retained artifact hashes.
Preparation succeeded against the complete compile bundle under
`../b12x-sm103-evidence/portable-operators-qualification/`; no GB300 execution
was requested or performed. Wheel and source distribution artifacts are under
`../b12x-sm103-evidence/portable-operators-dist/`. The wheel's portable operator
sources match the checkout byte for byte. Qualification scripts and tests run
from a full checkout, not the installable wheel.

## Compiler attempts and limitations

Python 3.12, PyTorch 2.14.0+cu130, CUTLASS DSL and library wheels 4.6.2, and
cuda-python 13.4.1 are installed in the isolated virtual environment. CPU-only
`cute.compile(..., no_jit_engine=True, options="--gpu-arch=sm_103a")` produces
actual PTX, cubins and host-side MLIR without a CUDA driver context.

The native MoE/transport/reconstruction subset contains nine MoE launchers,
eight TP2 RoCE launchers and three Trellis reconstruction launchers.
NVIDIA `nvdisasm` 13.4.49 can inspect
the cubins. Projection code contains `tcgen05.mma` in PTX and
`UTCOMMA.BLOCK16`/TMEM operations in SASS. Static register/resource reports
describe generated code, not achieved occupancy or runtime correctness.

An attempted CPU-only C host-object export failed resolving CUDA runtime
symbols. The supported offline workflow retains PTX/cubin/MLIR instead. Runtime
JIT host launch construction, TMA descriptors and CUDA graph execution still
require B300 qualification. No nvcc installation or guessed device profile is
required for this offline workflow.

## Remaining implementation work

See the source-level work table and qualification order in
[SM103 qualification](sm103-qualification.md). Complete GLM, DFlash2, V4.1 and
Station TP2 serving remain unsupported. A model-wide vLLM capability gate is
not enabled by this prototype; the b12x operation API remains the integration
boundary. Serving changes depend on qualifying the required individual ops.
## Source archive qualification

The compile and qualification tools accept exported source trees without Git
metadata. An exported `.git_archival.txt` records the base revision; package
hashes identify the actual files under test. Source roots nested in unrelated
checkouts do not inherit the enclosing repository's revision. Offline tests
cover archive preparation, archived revision handling, and execution from a
different directory.

## GLM sparse attention and shared-prefill synchronization

The public sparse-MLA plan now selects ordinary FP8/BF16 warp MMA on SM103
for the GLM NSA and GLM Next packed-cache contracts. Decode resolves its split
count from capacity, and prefill reuses the shared multigroup pipeline. FP8
group scaling follows ordinary FP8 QK accumulation; NVFP4 retains inline BF16
dequantization. CUDA graph replay uses caller-owned scratch and cached kernels.

The test corpus exposed a dynamic DLPack extent overflow for a packed pool
larger than 2 GiB. A one-byte base-pointer view plus an Int64 physical page
stride preserves the existing storage and large-offset addressing. Tests cover
high recycled page IDs, selected-length mutation, eight-head tails, head-major
output, empty bindings, and live row counts under frozen resolution.

Compute Sanitizer synchronization checking also found a shared-prefill barrier
error in both the portable and existing SM120 paths. Producer and consumer
branches reached separate `bar.sync` instructions with an aligned control-flow
promise. Both sites now emit `barrier.cta.sync`. Focused reproductions and the
shared-prefill corpus pass synchronization checking after that change.

Validation uses package hash
`d5dbd8cab6dd6004f00be4ba1c421acc37506bec04ff3f16d273b7e3e85d895c` in
`/home/jasonc/b12x-sm103` and the identical package on ripper at
`/home/jasonc/b12x-sm103-regression-20260912`. The base Git revision is
`6e2d2ca10289923b523f56cd09f00dbc29f7b623`; the compile manifest records the
modified source state. The physical regression device is RTX PRO 4000
Blackwell, CC 12.0, UUID `GPU-47363510-b87a-13a5-4824-2542e97df76c`.

| Check | Result |
| --- | --- |
| Host library, architecture, policy, sparse-binding and portable sparse suites | 413 passed, 27 skipped |
| Portable GLM sparse suite, Compute Sanitizer 13.4.57 memcheck | 50 passed, zero sanitizer errors |
| Portable GLM sparse and shared-prefill corpus, synccheck | 69 passed, six decode cases deselected, zero sanitizer errors |
| All-component offline SM103 compilation | 128 callables, 136 CUDA entries; all retained artifact hashes verified |

The offline corpus includes 45 sparse-attention and cache-writer callables.
Six sparse-prefill cases report stack frames of 8–136 bytes with zero reported
local storage. Dynamic launch SMEM and target occupancy remain unmeasured.
The source-bound [validation receipt](sm103-sparse-validation.json) retains
resource flags and raw log identities under `../b12x-sm103-evidence/`.
No SM103 execution or target performance measurement was performed.

## FP8 and MXFP4 DSA indexing

The public DSA planner selects the portable warp backend on SM103. Paged,
fused, and contiguous FP8 scorers use ordinary E4M3 MMA with their existing
external scales. MXFP4 decode and prefill use CuTe inline BF16 dequantization,
retaining the published dot/product/head-sum rounding boundaries and explicit
TP reduction before selection. The policy, generator, and embedded profiles
use DSA config schema 2; the merge race uses candidate contract version 2.

Frozen-resolution tests found physical pool extents in the paged/fused compile
identity. Those extents are now dynamic. The public fused, tiled, and prefill
paths reuse compiled callables across live row counts and pool views, including
records beyond the 2 GiB byte-offset boundary. Tests mutate visible lengths
during graph replay and check exact top-k sets, stable storage, and allocation
counters.

The pre-port cooperative merge could announce CTA arrival before other warps
finished publishing histogram entries. Its grid barrier now waits at a CTA
barrier before thread zero's release operation. A delayed-warp test against
revision `6e472940` observes missing-publication counts `[3,2,1,0]`; the fixed
kernel observes zero. A broader baseline memcheck run also failed a correctness
case and later stalled; its owned process was terminated after more than two
minutes. That interrupted run is diagnostic evidence only.

The final package hash is
`75fdf3cac5d9891c5974cba6940d10bf52a502e3d98ff636ab3f9592f43242dc`.
The local checkout and ripper regression checkout have identical package
sources. Validation runs on physical RTX PRO 4000 Blackwell, CC 12.0, UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, in Default compute mode.

| Check | Result |
| --- | --- |
| Host library, architecture, policy, indexer bindings and fused-indexer suites | 462 passed, 25 skipped |
| Portable indexer, fused-indexer and paged integration suites, memcheck | 174 passed, zero sanitizer errors |
| Identical GPU suites, synccheck | 174 passed, zero sanitizer errors |
| All-component SM103 compilation | 186 callables, 194 CUDA entries; artifact hashes verified |
| Wheel contents | All 443 package Python sources and embedded profiles match the checkout |

The indexer contributes 58 compile cases. Twenty-one have nonzero stack frames
of 8–312 bytes, with zero reported local storage. The
[indexer receipt](sm103-indexer-validation.json) records source identity,
resource flags, raw-log hashes, artifact paths, and package verification.
No SM103 runtime qualification or performance claim is included.

## Dense quantized projections and shared tcgen05 compute

Status: **implemented, unqualified on SM103**. The public block-scaled GEMM
API selects dense NVFP4, MXFP4, and MXFP8 TMA/tcgen05/TMEM kernels on SM103.
W4A16 and W8A16 use the existing inline BF16 warp-MMA engine through admitted
entry types. The general SM12x dense compiler entry remains gated. Packed
weights, prewarm, precision resolution, and caller-owned output/workspace
remain shared; SM103 precision defaults are unmeasured heuristics.

`gemm/_shared/sm103_blockscaled.py` contains the shared dense/routed compute
pipeline. Dense tiles use runtime row counts and group strides without route
metadata. The routed NVFP4 wrapper retains its public launch ABI. Static
inspection found a 32-bit output-row product; Int64 output strides produce
64-bit PTX address products. A deferred test crosses 2^31 output elements with
an approximately 4 GiB allocation. Calls without alpha use a constant-one
epilogue and allocate no scalar tensor.

The precision generator and benchmark use an independent NVFP4 activation
rounding/GEMM oracle. Their previous oracle reused the GEMM under test.
Generator candidate contract 4 includes SM103 and excludes its unimplemented
fused activation-quantization candidate. SM103 timing qualification requires
P0, zero throttling, stable memory clocks, and at most 30 MHz SM-clock drift.

Evidence uses package SHA-256
`4245f86a2f8aaf041f000cf5572ad6f83ea95e5c6753208014a5433303582608`
over base revision `057e4e9752054aa25860538c693dbaeae429e06f` plus the recorded
working-tree changes. The same package ran in the isolated SM120 checkout on
ripper; the physical GPU was `GPU-47363510-b87a-13a5-4824-2542e97df76c` in
Default compute mode.

| Validation | Result |
| --- | --- |
| Architecture, policy, and selected linear tests without CUDA | 386 passed, 224 skipped |
| Portable A16 and packed quantized tests on SM120 | 72 passed under memcheck; zero errors |
| Same SM120 suite under synccheck | 72 passed; zero errors |
| All-component SM103 compilation | 222 callables, 230 CUDA entries; retained artifact hashes verified |
| Native SM103 execution cases | 22 deferred; not executed on SM120 |
| Wheel/source distribution | Built; all 446 Python files and three embedded profiles match source |

The 36 linear callables report no stack or local memory. Shared MoE projections
increase allocated GPRs by four for Int32 route IDs and two for Int64 route IDs;
these deltas require target profiling. Existing stack-resource flags remain
visible in the [source-bound receipt](sm103-blockscaled-validation.json).
Full Trellis experts, compressed DeepSeek sparse attention, tensor/block FP8,
MXFP6, DFlash2/full serving, and HBM GDR remain implementation gaps.

## Tensor-scaled and compact K128 FP8 projections

`gemm/blockscaled/_fp8_cute.py` provides a CuTe entry for ordinary E4M3 warp
MMA with the shared TMA pipeline. The existing tensor-FP8 packed API and
compact K128 block-FP8 API select it on SM103. Static weight geometry and
output dtype select a conservative persistent tile; live row counts and
group strides remain runtime arguments. Output and scale addressing use
Int64. No measured SM103 profile is added.

Tensor-scaled FP8 uses the output multiplier exclusively. The shared core
uses the ordinary FP8 MMA descriptor and omits unused scale layouts,
transfers, and fragments. SM103 packed calls do not create a unit-scale cache
entry for each live count. Omitted alpha compiles out its load. Compact FP32
scales are applied to each K128 partial sum before final accumulation.

The FP32 regression exposed misplaced output columns in the shared tiled
epilogue. Unsplit plain FP8 now stores FP32 accumulators directly; split-K
retains its existing partial-output path. Transposed MMA also bypasses the
unused output TMA descriptor, preserving arbitrary one-group output widths.
Independent NVFP4 TMA/cp.async tests cover that shared store-path change.

The unchanged parent revision reproduces the legacy tensor-FP8 test failure
on the 70-SM RTX PRO 4000. That test assumed the <=64-SM compact-block policy.
Its assertions now check whether the actual device selects compact scales or
tensor scaling. Five host-side NVFP4 tile assertions also fail on the unchanged
parent: they apply small-SM probe expectations to a 188-SM device. The test
now covers both 48-SM and 188-SM policy contracts. The implementation's SM12x
policy remains unchanged.

The full SM103 compile corpus contains 241 callables and 249 CUDA entry
points. Its 19 FP8 callables use 55–128 allocated GPRs, with zero stack and
local memory. Existing callables have no positive register, stack, static-SMEM,
or local-memory deltas against the native block-scaled projection receipt.
The prior 31 stack flags remain visible; resource reports do not establish
dynamic-SMEM occupancy or performance.

SM120 regression on GPU `GPU-47363510-b87a-13a5-4824-2542e97df76c`, in Default
compute mode, passes 190 tests under both memcheck and synccheck with zero
errors. Fourteen device-specific legacy cases skip. Six additional serialized
API and NVFP4 epilogue tests pass both sanitizers with zero errors. Host tests
pass 690 cases and skip 66 GPU cases. The FP8 suite
includes independent numerical oracles, BF16/FP16/FP32 output, grouped capacity
strides, frozen resolution over multiple live counts, scale/input mutation,
graph replay, allocation checks, and output addresses beyond 2^31 elements.
Packaging verification covered byte-identical package Python files in the wheel
and sdist. It did not compare the compressed embedded profiles.

[FP8 validation receipt](sm103-fp8-validation.json) binds compilation,
resource accounting, SM120 logs, and packaging to the package source hash.
No physical SM103 execution or performance qualification occurred. Planned
BF16 block-FP8 linear, MXFP6, full Trellis experts, compressed DeepSeek sparse
attention, mHC/MTP, full serving, and direct HBM transport remain implementation
work.

## Planned block-FP8 projections

Status: **implemented; cross-compiled; unqualified on physical SM103**.
The public block-FP8 planner selects the native tcgen05/TMEM dense GEMM on
SM103. BF16/FP16 activations use the shared CuTe K32 quantizer. The 128x128
checkpoint recipe preserves its unfloored activation scales, and the 32x32
V4.1 recipe applies the 1e-4 amax floor. Logical K pads to 128; N is divisible
by eight. SM12x retains its existing backend and tile selection.

Quantizer cache identity includes device and architecture. SM103 uses one
lane layout across live row counts, including calls without an expected row
bound. Int64 source, payload, row-scale, and MMA-scale offsets cover large
buffers. Explicit streams cover the full quantize/GEMM/bias sequence.
The policy and generator use config schema 3; existing embedded profiles
change only that schema value. Candidate contract 2 invalidates incompatible
sweep checkpoints. An independent numerical oracle rejects incorrect candidates
before timing.

The full compile corpus contains 269 callables and 277 CUDA entry points.
The 28 added quantizer callables use 29–95 allocated GPRs, with zero stack
and local memory. Existing callables have no positive register, stack,
static-SMEM, or local-memory deltas against the tensor/compact-FP8 receipt.
The preceding 31 stack flags remain visible. Static resources establish
neither dynamic-SMEM occupancy nor performance.

The affected SM120 suite passes 137 tests under memcheck with zero errors.
Two additional generator tests pass memcheck; all 139 pass synccheck.
The tests cover independent arithmetic and scale-layout oracles, both half
types, zero/tiny/extreme groups, K padding, frozen resolution, graph replay,
stable addresses, allocation checks, stream ordering, public prewarm,
`torch.compile`, and source rows beyond 2^31 elements. Host tests pass
669 cases and skip 29 GPU-dependent cases; supplemental packaging and
qualification checks pass 17 cases.

An NVFP4 migration-corpus reference failed exact equality on the unchanged
parent commit `ca2fe8f8`. It scaled each input separately, while the operation
applies the supplied FP32 alpha after accumulation. The corrected reference
preserves the kernel's scaling contract and exact-equality assertion.
The pre-existing undefined Trellis rank-LUT decoder in `_lib/intrinsics.py`
remains outside this change and remains visible as a lint failure. All other
modified Python files pass Ruff.

Wheel and sdist verification compares all 447 package Python files and all
three compressed embedded profiles byte for byte. The
[planned block-FP8 receipt](sm103-planned-fp8-validation.json) records source,
compile artifacts, resources, regression logs, and package hashes. No SM103
execution or performance claim is made. MXFP6, complete Trellis expert
execution, compressed DeepSeek sparse attention, mHC/MTP/DFlash2 and complete
serving, and direct HBM transport remain implementation work.
## Native MXFP6 projections and fixed-capacity activation quantization

The existing dense FP6 interfaces dispatch to `gemm/blockscaled/_fp6.py` on
SM103. The shared TMA/tcgen05/TMEM engine supports independent E2M3, E3M2, and
E4M3 operand formats, packed FP6 global storage, byte-container compatibility,
grouped output, FP32 alpha, and BF16 row correction. TMA inserts FP6 shared
padding; barrier counts use packed global transfer bytes. Explicit byte
containers are packed in shared memory before MMA.

`FP6LinearWorkspace` and `allocate_fp6_linear_workspace` provide caller-owned
quantization storage through the existing FP6 namespace. Runtime row counts
drive both CuTe quantization kernels and GEMM. Shape, dtype, alignment,
capacity, device, and writable overlap checks precede launch. Scale tile tails
are overwritten. The SM120 default path retains its existing implementation;
the explicit workspace path is exercised on SM120 for regression evidence.

Independent tests exposed two boundary issues. The shared FP6 exponent helper
rounded positive subnormal ratios below its scale floor upward; it now clamps
them to byte zero. The numerical oracle's logarithm could round an exact power
of two slightly upward; the oracle uses exponent decomposition to avoid that
false failure. Exact subnormal-boundary and existing quantization tests pass.

Validation is recorded in [the FP6 receipt](sm103-fp6-validation.json):

- Architecture/policy and existing binding contracts: 678 passed, one skipped.
- Deferred FP6 and qualification-host collection: eight passed, 24 physical
  SM103 tests skipped on the GPU-less host.
- Physical RTX PRO 4000 Blackwell SM120 regression: 122 passed. Kernel memcheck
  and synccheck each pass all 122 tests with zero kernel reports, including
  source element and packed-output byte offsets beyond 2^31.
- The unfiltered sanitizer run emits 82 CUDA API reports. A minimal CUDA
  initialization and `cuDeviceGetCount` reproducer emits the same reports from
  CUDA Python's loader probing newer APIs against the CUDA 13.0 driver. The
  kernel-instrumentation runs explicitly use `--report-api-errors no`; raw
  reports remain available. Driver API compatibility is not qualified.
- A fresh complete SM103 build contains 327 callables and 335 CUDA entries.
  All 58 added FP6 callables compile, use 16–150 allocated GPRs, and report no
  stack or local memory. Existing allocated-register, stack, static-SMEM, and
  local-memory counts show no positive deltas. Every retained artifact hash
  verifies against the manifest.
- Wheel and sdist contents match all 450 package Python files and three
  embedded profiles. The operator qualification manifest prepares successfully
  against that identical package source.

`scripts/qualify_sm103.py --component fp6` includes native GEMM, quantization,
graph, stream, opaque serving-op, and large-offset tests.
`benchmarks/benchmark_fp6_linear.py` retains independently gated, paired
warm/cold graph measurements against an original-weight BF16 projection.
Neither command has executed on physical SM103.

Full Trellis expert execution, compressed DeepSeek sparse attention,
mHC/MTP/DFlash2 and complete serving, and remaining Station transport work
remain implementation gaps. The existing undefined Trellis rank-LUT decoder
identified in the planned FP8 log also remains unresolved.

## Inline Trellis projections and TMEM load completion

The internal SM103 Trellis projection consumes the existing compressed t256
representation and decodes weights directly into a 128x64 shared-memory tile.
FP16 operands feed tcgen05 with FP32 accumulation and FP16 output. A CTA owns
one route and 128 output columns. Invalid expert IDs and K/N padding produce
zero; all global row, expert, and tile products use Int64. The schedule uses
one operand stage and waits for MMA completion before reusing it.

`sm103/trellis_decode.py` shares native FP16 reconstruction between diagnostic
tiles and the projection. It handles the three-word circular spans at higher
rates and preserves MCG K2–K6, SQG E4M3 K2–K4, and SQG FP16 K5/K6 codebooks.
No model-sized FP16/BF16 repack enters the projection. The source weight-plan
contract constructs gate, up, and down primitives without adding a public API.
Full MoE planning remains unsupported until rotations, coupled transforms,
mixed-rate dispatch, activation, and reduction are integrated.

Compiler verification isolated an invalid TMEM readout fragment: sizing
registers from the broadcast source partition produced a 4096-element load.
The destination partition defines the per-thread register shape and emits a
valid 128-element load. Synchronization review also identified absent explicit
TMEM load-completion waits. Both the Trellis epilogue and the shared native
block-scaled epilogue now wait before consuming registers or releasing TMEM.
The compile auditor rejects TMEM loads without an emitted completion wait.
No SM103 runtime failure or performance improvement is claimed from these
source-level corrections.

The [Trellis validation receipt](sm103-trellis-validation.json) binds the final
source, artifact identities, resource comparison, and test logs. Host checks
pass 422 tests, with 35 hardware/environment skips. Physical RTX PRO 4000
Blackwell SM120 execution passes 34 reconstruction/staging tests, including
graph mutation, live-count reuse, invalid IDs, and source/weight offsets past
2^31. Kernel memcheck and synccheck pass the same corpus with API reporting
disabled for the previously isolated CUDA Python/driver probe incompatibility.
This qualifies portable kernel behavior, not driver API compatibility or SM103
MMA execution.

The expanded offline corpus contains 372 callables and 380 CUDA entries:
20 Trellis reconstruction and 28 inline projection callables. The projection
corpus includes E=384, gate/down V4.1 dimensions, K/N tails, and both route ID
widths. All artifacts are verified against their manifest. Static resource
reports do not establish occupancy or latency; the 32 KiB operand allocation
is dynamic launch SMEM and is not the report's static-SMEM field.

`tests/moe/test_sm103_trellis_gemm.py` adds 23 physical-SM103 cases covering
independent FP32 oracles, changing route counts with frozen compilation,
graphs, input/weight/route mutation, strided output, and high weight/output
offsets. The qualification runner includes these with the portable staging
suite. `benchmarks/benchmark_sm103_trellis_projection.py` prepares gated,
paired warm/cold graph measurements against per-route Torch FP16 projections.
Only its command-line help has run; no SM103 timing is supplied.

Complete Trellis experts, compressed DeepSeek sparse attention, mHC/MTP and
DFlash2 model integration, complete GLM/V4.1 serving, and remaining Station
transport work remain implementation gaps. These are not resolved by the
compiled projection or by portable reconstruction tests.

## Uniform Trellis expert execution

The `tcgen05_trellis` backend implements uniform-rate materialized experts
through the existing MoE plan/bind/run API. It retains compressed t256 weights,
decodes into shared memory for FP16 tcgen05 projections, and implements ordinary
scaled H128 or coupled H512/H128 expert transforms in CuTe. Routing maps,
SiLU/SiTU, FP32 weighted reduction and caller FP16/BF16 output are supported.
Capacity plans own stable scratch and precompiled callables; live row counts
remain runtime launch arguments. All pool-scaled offsets use Int64.

Canonical SQG E4M3 preparation accepts K2/K3/K4, records the loaded rate on the
prepared weight plan and retains FP16 internal projection buffers for either
public input dtype. The SM120 end-to-end diagnostic caught the prior BF16
internal-buffer mismatch; all six rate/transform combinations pass independent
numeric and graph checks after correction. The policy config schema is 5 and
the MoE candidate contract is 20. Existing embedded profile measurements are
unchanged; the SM103 generator qualifies the actual uniform SQG backend with
an independent decoded-weight/transform oracle before timing.

Validation: 501 host tests pass, with 42 hardware-dependent skips. SM120 passes
61 tests under memcheck and synccheck, with 13 native SM103 cases skipped and
zero reported kernel errors. The large-offset transform case indexes scale
rows past 2^31 elements. CUDA API reporting is disabled only for the documented
CUDA Python/driver probing mismatch on that regression host.

The full offline corpus contains 436 callables and 444 CUDA entry points.
The 64 added Trellis callables use 12–140 allocated GPRs, no stack or local
memory, and all TMEM readers have explicit completion waits. Existing kernels
have no positive register, stack, shared-memory or local-memory deltas. The
31 existing stack-flagged callables remain recorded. Wheel and sdist builds
contain byte-identical copies of 455 Python files and three embedded profiles.
Exact commands and hashes are in `sm103-trellis-moe-validation.json`.

Native SM103 correctness, graph replay and performance remain unqualified.
The deferred suite includes V4.1 geometry with E=384, H=5120, I=2304 and top-k=6.
Canonical MCG, mixed/paired rates and the mixed-rate descriptor limit of 256
experts remain implementation work. Full V4.1/DFlash2 serving, compressed sparse
MLA, mHC/MTP and Station direct-HBM transport remain separate gaps.
