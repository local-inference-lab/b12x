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

## Projection-tiered MCG expert execution

The native `tcgen05_trellis` backend consumes canonical MCG K3/K4/K5 weights
with independent rates for gate, up and down projections. Each projection
selects its compressed record from the existing coalesced payload and decodes
it into shared memory before FP16 tcgen05 MMA. The uniform and mixed kernels
share TMEM allocation, completion waits and the FP16 epilogue. Ordinary H128
transforms support SiLU/SiTU and FP16/BF16 public I/O. Paired/grouped records
and coupled mixed-rate transforms remain unsupported.

Canonical preparation uses 24 local-index bits above 256 experts, covering
all 384 experts while retaining Int32 descriptors. SM12x retains its eight-bit
format. Bind validates coalesced payload ownership and projection counts, and
passes tier offsets, populated counts and payload lengths as runtime scalars.
A fixed set of 15 callables serves all rate distributions and live counts.
Nine launches include route mapping, prepared namespace composition, transforms,
three projections and weighted reduction. The public A16 binder's unit-scale
flag is accepted without activation-scale arithmetic; rejecting that flag had
prevented canonical native Trellis binding, including uniform-rate execution.

Validation: 515 host tests pass with 59 hardware-dependent skips. SM120 passes
71 tests normally and under memcheck and synccheck, with 20 native SM103 cases
skipped and zero reported kernel errors. Exact operand checks cover all three
rates, projection rows, descriptor widths, route-ID widths, invalid metadata,
K/N tails and graph mutation. A mostly uninitialized 8.64 GB payload places
local expert record 205 beyond 2^31 Int32 words and validates its first operand
tiles. Preparation preserves source records for mixed and uniform populations
at E=5 and E=384. Existing SM120 uniform and mixed expert execution agrees with
the independent decoded-weight/transform oracle.

The offline corpus contains 470 callables and 478 CUDA entry points. The 34
added callables use 12–140 allocated GPRs with no stack or local memory.
Existing artifacts have no positive resource or R/UR/P/UP register-count
deltas and no instruction-count changes. All 104 TMEM readers have explicit
completion waits. The 31 existing stack-flagged callables remain recorded.
Wheel and sdist contain identical copies of 456 Python files and three profiles.
The MoE config schema is 6 and candidate contract is 21; embedded measurements
are unchanged. `sm103-trellis-mixed-validation.json` records commands and hashes.

Native mixed expert numerics and graphs remain unqualified on SM103. The
`trellis_mixed` qualification suite includes seven native cases with 384-expert
and V4.1 geometry, live counts 1/3/4/8, frozen resolution, descriptor/payload
mutation and allocation checks. Compressed DeepSeek sparse MLA, checkpoint
formats requiring paired/grouped or coupled mixed rates, mHC/MTP and DFlash2
integration, complete GLM/V4.1 serving and Station direct-HBM transport remain
implementation work.

## Planned DeepSeek compressed MLA on SM103

Status: implemented and cross-compiled; unqualified on physical SM103.
The public compressed-MLA plan selects ordinary warp MMA over native V4 or
V4.1 SWA and indexed records. Decode reserves fixed splits; extend uses the
shared multigroup pipeline. Selection metadata, lengths, pool sizes and strides
remain runtime arguments. Aligned plan-owned scratch includes padded selections
and a zeroed record for masked reads from an empty pool. Int32 metadata slices
need only four-byte alignment. Vectorized data accesses validate their alignment.

The shared epilogue masks partial head fragments before output/LSE stores and
sink reads. The compressed reference masks holes throughout a selection instead
of stopping at its first invalid slot. The scratch layout accounts for alignment
between both selection arrays and their length vectors. V4.1 FP8 decode uses
canonical FP8 operands with ordinary MMA; the SM12x native schedules remain
unchanged. Compressed LSE includes the sink, while GLM retains its selected-token
convention. Config schema 3 and candidate contract 3 select the actual backend
and fixed split geometry; embedded native measurements are unchanged.

The frozen source passes 492 host tests with 40 hardware-dependent skips.
SM120 passes 201 attention regression tests, 100 compressed-MLA/cache-writer
cases under memcheck, and 201 cases under synccheck. Both sanitizer runs report
zero kernel errors, with API reporting disabled under the previously documented
CUDA Python/driver probe mismatch. Coverage includes head tails, all-invalid
selections, empty pools, poisoned inactive pages, offsets beyond 2 GiB, live
counts under frozen resolution, full-graph Torch tracing, stable pointers and
allocation-free CUDA graph replay. V4.1 graph replay writes the SWA cache before
attention consumes it. Native writer bytes match independent format oracles.

The offline corpus contains 617 callables and 625 CUDA entry points, including
147 compressed attention/cache-writer callables. Added callables use 12–168
allocated GPRs. Four prefill callables report eight-byte stack frames with local
loads/stores and remain flagged for B300 profiling. Existing resource, exact
R/UR/P/UP register-count and instruction-count deltas are zero. All 104 TMEM
readers retain completion waits. Wheel and sdist match all 457 package Python
files and three profiles; an isolated wheel import selects the SM103 warp
heuristic. `sm103-compressed-mla-validation.json` records commands and hashes.

Physical SM103 correctness and performance remain deferred. Checkpoint-specific
paired/grouped or coupled mixed-rate Trellis, mHC/MTP/DFlash2 and complete
GLM/V4.1 serving integration, and Station direct-HBM transport remain
implementation work.


## Planned DeepSeek mHC execution on SM103

Status: implemented and cross-compiled; unqualified on physical SM103.
The existing CuTe pre/post/post-pre kernels support four residual streams at
H=4096/5120/7168 through normal planning and binding. Plans retain native
partial/reduction schedules and prefill choices. Live counts change grids and
masks while callable selection uses planned capacity. Native unbound execution
uses a fixed default schedule. Invalid decode splits are rejected before
launch when they cannot fill complete thread blocks.

SM103 planned prefill explicitly converts TF32 operands and decomposes FP32
projection weights into high/low terms. The independent FP32 oracle exposed
precision loss in the single-term projection; the policy selects the two-term
path instead of relaxing the oracle tolerance. SM12x non-lagged precision is
unchanged. Legacy unbound TF32 helpers retain their precision configuration;
SM103 prefill uses the planned contract. The mHC config schema and generator
candidate contract are both 3. Embedded profiles change only their mHC schema
version, preserving all recorded measurements. Generator candidates must pass
correctness and replay-allocation checks before timing.

Validation: 550 host tests pass with 121 hardware-dependent skips. All 82 mHC
GPU tests pass normally, under memcheck and under synccheck on SM120. Both
sanitizers report zero kernel errors; API reporting is disabled for the
previously diagnosed CUDA Python/driver probe mismatch. Local and remote
package sources have identical hashes. Graph tests mutate inputs after capture,
poison scratch and inactive tails, freeze kernel/policy resolution, and require
stable addresses with no replay allocation. Caller-owned Dynamo tracing remains
unsupported; functional tracing and caller-owned CUDA graphs are separate
contracts.

The SM103 corpus contains 748 callables and 756 CUDA entry points, including
131 mHC callables. Offline CuTe compilation supplies the documented 228 KiB
SM103 per-SM shared-memory capacity only for preferred-carveout calculation;
all other device-attribute requests fail. The 617 existing callables have no
positive resource or exact register-count deltas and no instruction-count
changes. All 104 TMEM readers retain completion waits. Added mHC callables use
17–255 allocated GPRs. Three high/low TF32 projections have 16-byte stack frames;
two H=5120 block-prefill producers have 408/496-byte frames. These five variants
remain flagged for B300 profiling alongside the 35 retained flags. Wheel and
sdist match all 457 package Python files and three embedded profiles. An
isolated wheel import resolves the SM103 precision policy. Exact commands and
hashes are recorded in `sm103-mhc-validation.json`.

Source inspection of LIL vLLM's GLM integration revision
`05c8e2750794ae79a221c0d94f4dfe304e610e99` confirms SM120-family gates in
MXFP8 linears and fused MoE. GLM mHC and DeepSeek attention also need capability
routing and plan/warmup review. The b12x MTP feedback operator implements a
Qwen tensor contract; GLM and DeepSeek feedback require their own contract
validation. These are source-level integration tasks. Paired/grouped or coupled
mixed-rate Trellis, complete MTP/DFlash2 and GLM/V4.1 serving, and Station
HBM registration/transport remain implementation work.


## Native MoE prefill route capacity

Status: implemented and cross-compiled; native execution remains unqualified
on physical SM103. Routed projection, input quantization and output reduction
place the route or token count on the CUDA X grid axis. The public planner
admits 8,192 tokens with eight selected experts and rejects Int32 count
overflow or excessive projection-column grids. Compile contract 2 and MoE
candidate contract 22 invalidate incompatible cached artifacts and sweeps.
The public architecture-capability helper is exported, and the MXFP8
compatibility API advertises its implemented SM103 backend.

Validation: 345 host tests pass with 21 hardware-dependent skips. All 13
portable pointwise tests pass under SM120 memcheck and synccheck with zero
kernel errors. The large case executes 131,074 routes and captures reduction
for 65,537 tokens with input mutation and no replay allocation. Its dyadic
operands permit an exact FP32 reference despite fused arithmetic. The native
MoE suite adds M1/M8192/M8193 replay under frozen kernel resolution with one
capacity plan; those cases remain deferred to B300.

The source-bound corpus contains 748 callables and 756 CUDA entries; an
additional nine-callable compilation covers capacity 8193. Input quantizers
gain three allocated GPRs. Exact R/UR register-count increases remain recorded,
with no additional stack or local-memory flags. All 104 TMEM readers retain
completion waits. Wheel and sdist match 457 package Python files and three
profiles, and an isolated wheel import resolves the 65,544-route plan. The
[route-grid receipt](sm103-route-grid-validation.json) binds commands, sources
and artifacts. Companion vLLM integration is separate work in
`feat/b12x-sm103` at `/home/jasonc/vllm-sm103`.

## MXFP8 serving capacities and workspace

Status: implemented and cross-compiled; physical SM103 execution remains
unqualified. Packed MXFP8 calls accept caller-owned output and scratch for
FP16 activations, using the existing activation quantizer and CuTe GEMM.
BF16 activations retain b12x precision selection. Prewarm covers all selected
precision configurations inside configured capacity buckets and preserves
endpoint calls without a capacity hint. The companion vLLM adapter retains
configured capacities, borrows shared workspace and uses an opaque Torch
operator to select a covering capacity without tracing live counts into kernel
resolution. Capture retains the borrowed scratch buffer.

Validation: 51 b12x host tests pass, with 255 hardware skips. SM120 runs pass
215 b12x tests, with 36 skips, and six focused b12x cases under both sanitizers.
The companion passes 53 host tests and twelve BF16/FP16 eager/Inductor MXFP8
serving cases under memcheck and synccheck. Sanitizers report zero kernel
errors. Graph checks freeze kernel resolution, mutate inputs, poison scratch
and output, compare independent quantization oracles, and require stable
addresses with no replay allocation. These are operator checks, not model evals.

Targeted SM103 compilation covers 36 dense callables and ten supporting
activation-packing callables. Dense resources, exact register counts and
instruction counts match the prior route-grid artifacts; all 22 TMEM readers
retain completion waits. Four NVFP4 packing variants have eight-byte stack
frames and FP32 division slow-path calls, with no SASS local loads/stores.
Their math is unchanged; retain these flags for B300 profiling. MXFP8 packing
has no stack/local-memory flags. Wheel and sdist match all 457 package Python
files and three embedded profiles. The [validation receipt](sm103-linear-workspace-validation.json)
binds commands, source identities, artifacts and limitations.

NVFP4, MXFP4 and tensor/block FP8 serving adapters still need configured
capacity and workspace integration. DeepSeek output projection, remaining
Trellis formats, GLM/DeepSeek feedback, complete model evaluation and Station
HBM transport remain implementation or integration work.

## FP4 serving capacities and activation buffers

Status: implemented and cross-compiled; physical SM103 execution remains
unqualified. The companion NVFP4 and MXFP4 adapters retain configured capacities
and reserve packed activation values and scales before capture. A shared
execution owner also preserves MXFP8 workspace handling and retains borrowed
buffers in captured graphs. NVFP4 uses the native out-quantizer, clears its
unwritten scale padding, and preserves the original alpha tensor. Strided
activations are made contiguous before quantization.

The public `gemm.blockscaled.quantize_mxfp4` helper writes caller-owned E2M1
values and UE8M0 F8_128x4 scale storage, including padding. Its supporting
Triton kernel preserves signed zero and uses runtime row counts; core GEMM
remains CuTe DSL. On SM120, MXFP4 scale fragments with several trailing K modes
are grouped into the mainloop's three-mode contract. This preserves math,
fragment order and launch policy. The long-K regression uses an exact FP32
oracle because Torch BF16 GEMM can reduce partial sums to BF16.

Validation: 52 b12x host tests and eight qualification-tool tests pass; 231
SM120 operator tests pass with 40 hardware/dependency skips. Ten MXFP4 cases
pass both memcheck and synccheck. The companion passes 53 host tests and 38
GPU cases under both sanitizers: twenty FP4 serving cases, twelve MXFP8
regressions and six MXFP4 byte comparisons against FlashInfer. Sanitizers
report zero kernel errors. Tests cover BF16/FP16, eager/Inductor, exact packed
bytes and scale padding, finite extremes/subnormals, input mutation, frozen
kernel resolution, poisoned buffers, stable addresses and no replay allocation.
These are operator tests; complete model evaluation remains outstanding.

SM103 compilation covers 36 dense and sixteen supporting packing callables.
Existing resources, exact register counts and instruction counts match the
previous targeted artifacts. All 22 dense TMEM readers retain completion
waits. The six added MXFP4 packers have no stack/local-memory flags. Four
NVFP4 packing variants retain eight-byte stack frames with FP32 division
slow-path calls and no SASS local loads/stores. Wheel and sdist match 458
package Python files and three profiles. The
[FP4 serving receipt](sm103-fp4-serving-validation.json) records exact sources,
commands, artifacts and limits. This is a targeted compile, not a rebuild of
the complete SM103 corpus.

Tensor/block FP8 serving capacity and workspace, DeepSeek output projection,
remaining Trellis formats, GLM/DeepSeek speculative feedback, complete model
execution and Station direct-HBM transport remain implementation or integration
work. SM121 regression and physical SM103 qualification remain outstanding.

## Tensor and compact block-FP8 serving workspace

Status: implemented and cross-compiled; physical SM103 execution remains
unqualified. Tensor-FP8 and compact K128 block-FP8 calls accept caller-owned
output and scratch. Scratch covers input padding, unit activation scales and
split-K partials. Tensor-FP8 prewarm prepares declared capacities without caching
unit-scale tensors by live row count. Internally owned allocations use the
requested launch stream so scratch cannot be recycled before that stream retires.
The companion adapters retain configured capacities and native quantization
buffers in shared workspace, including tensor FP8 with different BF16/FP16 input
and output dtypes.

Long-K FP16 regression exposed a shared SM12x policy error: FP16 output could
select a BF16 atomic split-K epilogue. The policy now restricts BF16 atomics to
BF16 output. FP16 retains its split count and uses FP32 partials plus a typed
CuTe reduction. The reduction's compile contract is 2. Existing BF16 atomic
accumulation and its rounding behavior remain unchanged.

The native group quantizer and Torch division can select adjacent FP8 values at
rounding midpoints because their FP32 scales differ by approximately one ULP.
The independent oracle checks scales within relative 2^-22 and admits only
adjacent codes within relative 2^-21 of the midpoint. It then checks GEMM using
independently decoded quantized operands. BF16 relative L2 must remain below
0.005, FP16 below 0.001, and cosine similarity above 0.99999. FP16 and physical
SM103 replay require exact eager equality; SM12x BF16 retains numeric checks for
unordered atomic sums.

Validation: 519 b12x host tests pass with 173 hardware skips, and 193 SM120
operator tests pass with 18 skips. Ten focused workspace tests pass both
memcheck and synccheck with zero kernel errors. The companion passes 54 host
tests and 58 GPU cases under both sanitizers with zero kernel errors, including
all 38 FP4/MXFP8 regressions. API reporting is disabled only for the recorded
CUDA-Python/SM120 driver probe mismatch. The
[FP8 serving receipt](sm103-fp8-serving-validation.json) binds sources,
commands and artifacts. The targeted SM103
corpus covers 19 FP8 and 38 dense/reduction callables. Existing resources, exact
register counts and instruction counts have no increases against the recorded
baselines. All 22 dense TMEM readers retain completion waits. Two added FP16
reductions have no stack/local-memory flags. Wheel and sdist match 459 package
Python files and three embedded profiles. These are operator checks; complete
model evaluation remains outstanding.

DeepSeek output projection, paired/grouped or coupled mixed-rate Trellis,
GLM/DeepSeek speculative feedback, complete model execution, and Station HBM
transport remain implementation or integration work. SM121 regression and
physical SM103 qualification remain outstanding. The companion still uses the
recorded precompiled native libraries rather than a build of its Python revision.

## Planned DeepSeek WO projection

Status: implemented and cross-compiled; physical SM103 execution remains
unqualified. The WO component resolves a typed `mxfp8_tcgen05` configuration
on SM103 and retains it on plans and bindings. Its native bound execution uses
caller-owned activation, intermediate and output storage around two MXFP8
GEMMs. Inverse RoPE executes in FP32 inside the first quantizer. SM103 bindings
retain capacity and fixed scratch offsets without initializing device memory
at bind time. The native path does not use the legacy allocating inverse-RoPE
operator or SM12x fused-GEMM overrides.

The CuTe quantizer uses Int64 row/group/cache offsets, runtime row grids,
device/architecture cache identity and a prewarm guard. Each call writes all
logical and padded scale bytes. Negative or out-of-range cosine positions raise
a device error. BF16 and FP16 inputs, BF16/FP32 cosine caches and Int32/Int64
positions have byte-level oracle coverage. Int32 positions may have four-byte
alignment; multiplying a position by its cache stride still uses Int64.

Validation: 523 host checks pass with 23 skips. The SM120 WO suite passes 44
checks with six native-SM103 skips. Thirteen quantizer cases pass memcheck and
synccheck with zero kernel errors, including a mostly uninitialized 4 GiB cosine
cache addressed at its tail. Tests cover M1/M3/M8/M9/M16/M127/M128/M129 under
frozen resolution, poisoned padding, input mutation, explicit streams, stable
addresses and allocation-free replay. Invalid-position tests use isolated CUDA
processes. The four-case production profile probe passes SM120 graph checks;
its diagnostic samples do not establish a performance claim.

The SM103 compilation contains 48 quantizers and eight native GEMMs. All eight
GEMMs emit MXFP8 UMMA and retain TMEM load completion waits. Static inspection
finds no stack or local-memory flags. Quantizers use 28–40 allocated registers;
GEMMs use 134 registers, 1,024 bytes of static shared memory and 67,712 bytes of
dynamic shared memory. Wheel and sdist match 460 package Python files and three
embedded profiles. The [WO receipt](sm103-wo-validation.json) binds commands,
sources and artifacts. The complete SM103 corpus has not been rebuilt.

Companion vLLM WO plan retention, shared scratch, warmup and architecture
admission remain integration work. SM12x retains its legacy projection paths;
its allocating inverse-RoPE behavior requires separate lifecycle work. Remaining
Trellis formats, GLM/DeepSeek feedback, complete model execution and Station HBM
transport remain open, along with SM121 regression and physical SM103
qualification.

## WO output ownership and strided attention input

Status: implemented; physical SM103 execution remains unqualified.
WO bindings accept an aligned caller-owned BF16 output that survives reuse of
the intermediate workspace. SM12x bound inverse-RoPE execution consumes the
binding's scratch and output through an opaque mutating operation. It shares
the functional entry's launch implementation and preserves compatible decode
epilogues. A prefill capacity cannot select an incompatible quantized decode
epilogue. Binding constructs views without device writes; scratch regions have
fixed capacity offsets on every supported architecture.

The SM103 quantizer accepts input rows sliced from padded attention buffers.
Its Int64 row stride is a runtime argument and does not enter the compile key.
Compile contract 4 records the added argument. The public `prewarm_inv_rope`
helper resolves the retained plan before capture and chooses its warmup counts
inside b12x. It rejects capture and frozen resolution.

Validation covers caller-owned output, poisoned workspace and output, padded
heads, input mutation, explicit streams, fixed capacities and allocation-free
replay. Large-offset cases address source rows and cosine-cache rows beyond
2^31 elements. The [WO serving receipt](sm103-wo-serving-validation.json)
records host and SM120 results, sanitizer checks, package identities and 56
SM103 compiled callables. The native GEMM cubins match the WO backend baseline.
Forty quantizers have positive R, UR or UP count deltas; six add two or four
allocated GPRs. No stack or local-memory flags appear. Those deltas remain
visible for physical B300 profiling, without a performance claim.

Companion vLLM still needs a WO owner that retains plans and workspace, invokes
prewarm before capture, and allocates output with the proper lifetime. Its
checkpoint scale metadata and TP reduction must remain correct. The attention
planning audit, target/draft model execution, remaining Trellis contracts and
Station HBM transport remain open. SM121 runtime regression is outstanding;
the inspected GB10 host was running a vLLM worker and was left undisturbed.

## Companion WO serving ownership and architecture admission

Status: implemented; host and SM120 operator checks pass. Physical SM103 and
complete model execution remain unqualified. Companion vLLM revision
`1441ad1edc0018c87bdd3db77c5c3d444965b113` retains configured WO capacity plans,
reserves shared workspace, prewarms both position dtypes through public b12x,
and binds a separate BF16 output through an opaque Torch operator. Capture
retains its binding. Every attention owner prepares state before matching
owners deduplicate compilation. Checkpoint metadata selects 128x128 or 32x32
weight blocks, and tensor-parallel reduction remains outside the projection.

The companion host suites pass 72 cases with four GPU skips. Four real WO
serving cases pass both SM120 memcheck and synccheck with zero kernel errors.
They exercise eager/Inductor execution, independent quantization and inverse
RoPE references, arbitrary positive FP32 128x128 scales, 32x32 UE8M0 scales,
padded input rows, changing live counts, frozen resolution, graph mutation,
poisoned output/workspace, stable addresses and unchanged cumulative allocation
counters. Companion `docs/design/b12x_wo_serving_validation.json` binds the
evidence to 2,307 Python/test source hashes and records the separately versioned
native libraries. Pre-commit checks pass.

Source review identifies live-row and page-table planning in the companion
compressed attention and indexer adapters. Compressed attention also omits
decode/extend mode from its plan and does not retain capture bindings. Those
contracts still require implementation. Aggregate DeepSeek selection now honors
the adapter architecture gate, preventing individual SM103 operation support
from admitting the incomplete model route. Remaining checkpoint formats,
GLM/DeepSeek speculative feedback, full model evaluation, Station direct-HBM
transport, matching native/ARM64 builds, SM121 regression and the complete
representative SM103 rebuild remain outstanding.


## Compressed attention capacity ownership and public prewarm

Status: implemented; physical SM103 and complete model execution remain
unqualified. Compressed MLA exposes public capacity prewarm for optional
indexed caches, sink, native V4.1 page mapping and both LSE scales. Native
bound execution stages selections into fixed scratch and chooses split/head
schedules from planned rows. SM121 dispatch uses semantic widths; optional
SWA-only prefill has reserved container storage even when the indexed route
uses decode. Binding still constructs views without device writes.

The companion owner retains configured decode/extend plans, derives separate
cache widths and page sizes, and reserves query/metadata/component scratch
before capture. Each owner prepares state before warmup deduplication. Stride
normalization handles singleton views that report contiguous storage with a
noncanonical row stride. Graphs retain the binding and staging buffers; output
belongs to the caller. The existing model attention boundary remains opaque to
Torch compilation.

The [compressed serving receipt](sm103-compressed-serving-validation.json)
binds source, packaging, compile and runtime checks. Host coverage includes
capacity preparation, rejected capture/frozen warmup, per-owner deduplication
and optional Spark staging. GPU checks exercise independent reference math,
poisoned scratch/output, input mutation, changing live rows/widths, fixed
addresses, allocation counters and high page IDs. V4.1 freezes kernel resolution
before the first real invocation after public prewarm. All 147 SM103 cubins
remain byte-identical to the inspected component baseline, including four
prefill stack/local-access flags.

The indexer audit identifies both removed legacy DSA exports and live row/page
width planning in the companion adapters. Those interfaces require public
plan/bind/run migration and retained ownership before aggregate DeepSeek SM103
admission. Model evaluation, checkpoint/speculative contracts, matching native
and ARM64 packaging, final full-corpus compilation and Station HBM transport
remain open. The inspected GB10 host has an active vLLM worker; SM121 GPU
regression remains deferred without disrupting that service.


## FP8 indexer serving ownership

Status: implemented; physical SM103 and complete model execution remain
unqualified. Public FP8 prewarm uses planned capacity with one live row and
preserves physical page pitch. Bound prefill and indices-only fused selection
use reserved storage for unused outputs and local scores.

Both companion FP8 adapters use public plan/bind/run calls with retained
capacity plans and staging. Metadata budgets come from the public plan.
GLM context auto-fit invalidates plans before warmup. Captured bindings own
query, metadata and scratch views. DCP reserves its packing and NCCL receive
buffers with the scorer and reuses the existing CuTe global selector.

The [indexer serving receipt](sm103-indexer-serving-validation.json) records
599 host tests, 199 SM120 component tests, standalone serving and GLM packed-tail
checks, and two-rank scorer/NCCL graph checks. Tests cover large recycled page
IDs, input mutation, poisoned scratch, exact index/score association, frozen
kernel resolution and cumulative allocation counters. All 58 SM103 indexer
cubins match the inspected baseline, including 21 retained stack flags.
The companion tool cross-compiles three TP2 selectors and six supporting
packing callables. Neither compilation nor SM120 execution qualifies B300.

Complete checkpoint/speculative contracts, model evaluation, matching native
and ARM64 packages, the final full-source corpus, SM121 regression and Station
HBM transport remain implementation or validation work.


## Coupled mixed-rate Trellis preparation and binding

Status: implemented; complete SM103 expert execution remains unqualified.
Canonical MCG K3/K4/K5 preparation accepts coupled H512/H128 transforms with
all-zero expert draws. Prepared tiers retain the transform flag, six-I
intermediate rows and shared gate/up input scale storage. The public SM103
backend uses eight materialized launches and retains 15 compiled callables
per static transform configuration across live counts and rate distributions.
SM120/SM121 preserve their rejection of coupled projection-tiered execution.

The [coupled mixed-rate receipt](sm103-trellis-coupled-mixed-validation.json)
records 539 host tests, 46 SM120 GPU tests, and 13 targeted cases passing both
memcheck and synccheck with zero errors. Coverage includes canonical record
preservation, five/384-expert descriptors, broadcast/per-expert scale aliasing,
exact all-K3 agreement with the uniform oracle, frozen host binding, prepared
transform graph mutation, stable addresses and cumulative allocation counts.
Ordinary MCG and uniform SQG execution regressions pass on SM120. Twenty-seven
complete-expert tests require physical SM103, including seven added coupled
mixed-rate cases.

The source-bound Trellis compile contains 176 callables, including 30 coupled
mixed-rate variants. Added variants use 12–140 allocated GPRs. The component
has no stack/local-memory flags or positive existing resource deltas, and all
56 TMEM readers retain completion waits. All 146 existing PTX artifacts match
the recorded mixed-rate baseline. Of their cubins, 125 match byte-for-byte;
21 differ in SASS register operands with identical resource counts, register
sets and instruction counts. Raw artifact identities remain distinct. The
wheel and sdist match 460 package Python files and three embedded profiles;
the extracted wheel constructs the public coupled MCG weight plan outside
the checkout.

Paired/grouped records, nonzero transform draws, complete checkpoint and
speculative execution, model evaluation, matching native/ARM64 builds,
SM121 regression, the final full-source corpus and Station HBM transport
remain open. No B300 runtime or performance result is claimed.


## Global coordinates for coupled expert draws

Status: implemented; complete SM103 expert execution remains unqualified.
Canonical `TrellisWeights` carries the global intermediate width and rank
channel offset. Nonzero expert draws require those coordinates. Preparation
slices the global preactivation/postactivation sign sequences, validates draw
IDs and aligned extents, and selects the shared input scale for extents inside
either global FC1 half. Residual draws remain zero, matching the frozen encoder
contract documented in [MoE execution](moe-execution-model.md).

The [draw-extent receipt](sm103-trellis-draw-extents-validation.json) records
641 host tests, 55 SM120 GPU tests, and 20 targeted cases passing both memcheck
and synccheck with zero errors. Canonical SQG and MCG extents match BTX packed
words, transform tables and reference outputs in both halves, with broadcast
and per-expert scale tables. All eight draws match frozen encoder bytes under
Torch 2.13 and 2.14. Uniform SQG execution and portable transform replay use
nonzero prepared signs; 27 complete-expert SM103 cases remain deferred.

The Trellis compile contains 176 callables with byte-identical PTX relative to
the coupled mixed-rate receipt. Of their cubins, 147 are byte-identical. The
29 changed cubins retain identical allocated resources, exact register sets
and instruction counts: 28 differ only in register operands, while the private
SQG FP16 K5 FC1 variant also differs in instruction order and operand reuse.
Raw identities remain distinct. The component has no stack/local-memory flags
or positive resource deltas, and all 56 TMEM readers retain completion waits.
The wheel and sdist match 460 package Python files and three embedded profiles;
the extracted wheel exposes the extent fields outside the checkout.

Paired/grouped records, coupled extents crossing distinct input-scale halves,
full checkpoint/speculative execution, model evaluation, matching native/ARM64
builds, SM121 regression, the final full-source corpus and Station HBM transport
remain open.

## Coupled input-scale halves

Status: implemented and cross-compiled; physical SM103 expert execution remains
unqualified.

Canonical SQG and projection-tiered MCG preparation retain both input-scale
vectors when a rank extent crosses the global FC1 half boundary. Uniform BTX
preparation carries the same local column split and preserves manifest extent
barriers. The synthetic BTX writer supports uniform extents ending within an
eight-slot group. Canonical and BTX records agree for whole 384-channel layers
and 256-channel subextents, including splits at columns 192 and 64.

The native FC1 implementation stages two activation rows in its existing
128-by-64 operand tile and selects the result per output column. Both physical
FC1 slots use the same split. Capacity plans prewarm both variants and reserve
both input buffers. Shared-input bindings retain their existing callable.
SM12x rejects the distinct-half contract before canonical prewarm and at legacy
bind. The public plan/bind/run interface and policy configuration are unchanged.

The [validation receipt](sm103-trellis-input-halves-validation.json) records
651 host tests, a 121-test SM120 suite, and 29 targeted cases passing memcheck
and synccheck with zero errors. The companion Torch 2.13 environment passes
51 host tests. Portable probes exercise the production operand staging and
epilogue predicate, including invalid routes, tails, frozen resolution,
mutated graph inputs and cumulative allocation checks. The native suite has
32 deferred SM103 tests. These results do not qualify complete SM103 experts.

The frozen Trellis corpus contains 196 callables and CUDA entry points. All
176 existing PTX objects are byte-identical; 149 existing cubins are identical.
The remaining 27 have identical allocated resources, exact register sets and
instruction counts. Twenty-five differ only in register operands; SQG FP16
FC1/FC2 K5 also differ in instruction ordering or reuse flags. Raw identities
remain distinct. The six dual-input variants use 167 allocated registers versus
140 in their single-input counterparts. This increase requires B300 occupancy
and performance profiling. No stack or local-memory traffic is emitted, and
all 64 TMEM readers retain load-completion waits.

The wheel and source distribution match all 460 Python package files and three
embedded profiles. An extracted-wheel import exercises the split metadata and
kernel constructor outside the checkout. The companion vLLM source remains
unchanged at `a72cfdcf7b484ab395fdd4d2ee0843877a919f39`.

Paired/grouped records, full-model speculative feedback and evaluation, a
matching native vLLM build, ARM64 dependencies, SM121 regression, direct-HBM
Station transport and the final full-project compile remain outstanding.

## Canonical grouped atom rates

Canonical MCG K2–K6 and SQG E4M3 K2–K4 preparation accepts grouped rates and
independent low/high plane rates. The native atom layout retains the original
compressed rows and uses group/expert/projection offset and rate tables.
FC1 and FC2 select and decode each native tile into shared memory before
tcgen05 MMA. Rates and offsets remain runtime operands; group size and planned
capacity are static. Ordinary and coupled transforms, nonzero draw extents
and distinct input-scale halves use the existing public MoE lifecycle.
SM12x rejects the atom layout before execution. Per-expert input-scale vectors
also retain their declared axes when the local expert count is one, and
broadcast two-element input gains preserve both scale planes.

The [validation receipt](sm103-trellis-atoms-validation.json) records 674 host
tests, 136 tests in the SM120 regression suite, and 74 host tests under the
companion Torch 2.13 environment. Fifteen atom preparation and operand tests
pass memcheck and synccheck with zero errors. Exact decoding checks include
group boundaries, unequal plane rates, malformed metadata, poisoned shared
memory, graph mutation and a mostly uninitialized 16 GiB pool whose live rows
start beyond 2^31 Uint32 words. Graph replay retains addresses and cumulative
allocation counts. Six complete atom-MoE tests await physical SM103; these
portable tests do not qualify native SM103 expert execution.

The frozen Trellis compile contains 254 callables and CUDA entry points,
including 58 atom-layout callables. All 196 existing PTX files are identical;
166 existing cubins are identical. Thirty changed raw cubins retain identical
allocated resources, exact register sets and instruction counts. No existing
resource count increases. The atom projections allocate 140 registers with
one input and 167 with two inputs, use 32,896 bytes of dynamic shared memory
plus 1,024 static bytes, and emit no stack or local-memory traffic. All 74 TMEM
readers retain load-completion waits. Grouped dispatch and dual-input resource
costs require B300 occupancy and performance qualification.

`scripts/audit_sm103_resources.py` verifies manifest-bound artifacts and
compares allocated resources, register sets, instruction counts and local
traffic. `scripts/qualify_sm103.py --component trellis_atoms` prepares or runs
the complete atom suite with explicit physical GPU selection. The wheel and
source distribution match all 462 Python files, three embedded profiles and
six native C sources; extracted-wheel imports pass outside the checkout.
The companion vLLM source remains unchanged at
`a72cfdcf7b484ab395fdd4d2ee0843877a919f39`.

Legacy BTX paired records, canonical SQG FP16 preparation, full-model
speculative feedback and evaluation, a matching native vLLM build, ARM64
dependencies, SM121 regression, Station direct-HBM transport and the final
full-project compile remain outstanding.

## Canonical SQG FP16 and bounded epilogue predicates

Canonical SQG FP16 preparation accepts uniform K5/K6, grouped rates and
independent low/high plane rates. It uses the existing 416-byte D3L descriptor
and FP16 reconstruction law. Ordinary and coupled transforms retain global
draw extents and distinct input-scale halves. Uniform native plans prewarm
16 ordinary or 18 coupled callables; grouped plans prewarm 14 or 15. Rates,
offsets and live counts retain their runtime roles. Canonical uniform SQG
FP16 also passes complete SM120 expert execution through the existing backend.

The shared native epilogue computes one tile-relative cutoff in Int64, clamps
it to 0–128, then uses boolean predicates to select the contributing MMA row.
This preserves column selection while shortening live ranges after TMEM loads.
The compiler initially emitted an eight-byte stack frame and one local load
and store for each uniform SQG FP16 dual-input projection. The final code
uses 138 allocated registers and no stack or local traffic for all eleven
dual-input variants. The eight previously implemented dual-input variants
decrease from 167 to 138 registers. The rewrite preserves quantization,
synchronization, launch geometry and planner policy; it is not a measured
performance result.

The [validation receipt](sm103-trellis-fp16-validation.json) records 710 host
tests, 147 SM120 regressions and 105 host tests under companion Torch 2.13.
Twelve targeted cases pass memcheck and synccheck with zero errors. Portable
checks cover exact K5/K6 plane reconstruction, invalid rate boundaries,
mutated graph metadata and epilogue origins above 2^31 and at 2^40. The complete
native suite has nine additional SM103 tests, including grouped, uniform and
coupled cross-half execution. Physical SM103 execution remains unqualified.

The final Trellis corpus contains 317 callables and CUDA entry points,
including 63 canonical SQG FP16 callables. Of the 254 existing entries,
246 retain identical PTX and 220 retain identical cubins. Eight changed
entries contain the epilogue rewrite; 26 other changed cubins retain identical
PTX, allocated resources, register sets and instruction counts. No existing
resource count increases, no stack or local traffic remains, and all 89 TMEM
readers retain completion waits. The before/after manifests and resource
comparison preserve the initial spill evidence.

The wheel and source distribution match all 462 Python files, three embedded
profiles and six C sources. Extracted-wheel planning and constructor checks
pass outside the checkout. Companion vLLM remains unchanged at
`a72cfdcf7b484ab395fdd4d2ee0843877a919f39`.

Legacy BTX paired records, complete speculative model integration and
evaluation, a matching native vLLM build, ARM64 dependencies, SM121 regression,
Station direct-HBM transport and the final full-project compile remain open.

## Consolidated source compilation and ARM64 dependency resolution

The [source-readiness receipt](sm103-source-readiness.json) binds the complete
representative compile command to package source revision `770f397d` and source
hash `61001cf402dd6a917a21410f095090700547766bb59cba31c0dba5efba00d0dc`.
All sixteen component groups compile: 977 CuTe callables and sixteen supporting
activation-packing callables contain 1,001 CUDA entry points. Separate native
MoE builds cover capacity 128 with gate-first weights and capacity 8193 with
65,544 routes; each contains nine callables and emits no stack or local traffic.

All 748 callables from the preceding full corpus retain identical PTX; 723
retain identical cubins. The other 25 cubins retain identical allocated
resources, exact register sets and instruction counts. No existing resource
metric increases. All 317 Trellis callables from the canonical FP16 milestone
also retain identical PTX and resource metrics. All 149 TMEM readers retain completion waits.
The census preserves forty compute stack/local-traffic flags and four supporting
NVFP4 packing frames. The latter have no explicit local loads or stores and
match the preceding packing component evidence. Hardware resource and latency
qualification remains required.

The resource auditor recognizes the FP8/FP6 `UTCQMMA` opcode alongside
`UTCHMMA` and `UTCOMMA`, only at instruction positions. Its host suite passes
15 tests, including predicated instructions and rejection of ordinary warp
MMA or misleading branch labels. The initial checker failure remains in the
evidence directory. Kernel sources and compile artifacts remain unchanged.
All 21 deferred operator suites collect successfully, and the qualification
launcher prepares their commands against the verified consolidated manifest.

Binary package resolution succeeds for standalone b12x on Python 3.12, ARM64,
CUDA 13.0 and glibc 2.28, selecting 66 packages. The combined b12x and companion
vLLM CUDA/build requirements resolve to 204 packages at glibc 2.34, including
Torch 2.13.0+cu130, Triton 3.7.1, CUTLASS DSL 4.6.2 and FlashInfer 0.6.17.
The glibc 2.28 companion attempt fails because the pinned TileLang ARM64 wheel
requires glibc 2.34. Commands, input hashes, package hashes and both resolver
results are retained. This checks dependency availability, not ARM64 installation
or native execution.

The qualification and companion integration documents describe canonical
grouped/SQG FP16 Trellis, WO ownership and indexer warmup as implemented.
Legacy BTX paired records, complete GLM/DeepSeek speculative integration and
model evaluation, matching native vLLM builds, ARM64 installation/native helpers,
available-host SM121 regression and Station direct-HBM transport remain open.
Physical SM103 execution remains unqualified.

## BTX paired-record execution

Declared BTX P22, P33, P24, P43 and P44 records use the native SM103 atom
backend through the existing compatibility preparation and plan/bind/run APIs.
MCG and SQG E4M3 support ordinary or coupled transforms and multiple complete
256-channel pairs per rank. Preparation retains compressed rows, normalizes
rate nibbles, restores record-major scale order and retains global coupled
draw/input-scale-half coordinates. Runtime metadata guards reject unknown pair
codes. SM12x keeps its coalesced single-pair implementation and existing limits.

The [BTX validation receipt](sm103-btx-pairs-validation.json) binds package
source `eacb5a7d5d2e6192c0a997769eff80ea68b6a47328252dc70cfb0d3ffeb5f411`
to 375 SM103 Trellis callables, including 58 paired-record specializations.
All 317 preceding callables retain identical PTX; 287 retain identical cubins,
and the remaining 30 retain identical resource metrics. No existing resource
metric increases, no stack/local-traffic flags appear, and all 99 TMEM readers
retain completion waits. Paired single-input projections use 140 allocated
registers; the two dual-input variants use 138. These are static measurements.

Validation passes 734 host tests, 158 SM120 tests and 122 host tests with the
companion Torch 2.13 environment. Memcheck and synccheck each pass eight
portable staging tests with zero kernel errors, including the atom-row offset
case beyond 2^31 words. The documented SM120 CUDA API reporting exception
remains explicit. Six complete BTX MoE cases require physical SM103; the atom
qualification suite collects 74 cases. Wheel and sdist contents match all 471
package files, and extracted-wheel imports pass outside the checkout.

An initial compile attempt was rejected because package source changed during
compilation. A subsequent build lacked the explicit runtime pair-vocabulary
guard and is retained as diagnostic evidence. The release build uses frozen
source and fresh caches. Frozen QSRT coupled high-rate conversion, complete
GLM/DeepSeek speculative integration and model evaluation, matching native
vLLM/ARM64 builds, available-host SM121 regression and Station direct-HBM
transport remain open. Physical SM103 execution remains unqualified.


## GLM RMS-concat MTP feedback

The existing MTP API represents GLM feedback with an explicit `rms_concat`
contract. A CuTe kernel masks zero-position embeddings and applies independent
ordinary RMS normalization; the retained CuTe BF16 projection consumes fixed
concatenation scratch. The Qwen flattened Gemma contract and its optimized
projections remain separate. Policy query schema 2 separates the contracts;
Qwen profile entries preserve measured coverage and GLM uses AUTO heuristics.
The generator races both normalization configurations through production plans.

The companion GLM MTP layer uses retained scheduler-capacity storage and an
opaque output-mutating Torch operator. Its final residual norm and normalized
recycled state are preserved. Validation passes 622 host tests, 44 SM120 MTP
and benchmark tests, 40 companion host tests and two companion eager/Inductor
call-site tests with graph mutation. Memcheck and synccheck each pass five
RMS-concat tests with zero kernel errors under the documented SM120 CUDA API
reporting exception. The compile corpus produces 30 SM103 callables/entries,
with no stack/local-memory flags. The [receipt](sm103-mtp-validation.json)
records artifact identities and validation limits.

Complete GLM checkpoint/model evaluation, DeepSeek per-stream FP8 feedback,
DFlash2 execution, frozen QSRT coupled high-rate conversion, matching native
vLLM/ARM64 builds, SM121 regression and Station HBM transport remain open.
Physical SM103 execution remains unqualified.


## DeepSeek per-stream FP8 MTP operator

The public MTP API implements `rms_streams_fp8`: zero-position embedding
masking, ordinary RMS per hidden stream, K128 E4M3 activation quantization,
separate block-FP8 projections with FP32 scales, and BF16 broadcast addition.
The plan owns fixed scratch for `max_tokens * streams` hidden rows, retains
both position-dtype normalization callables and uses runtime live-row grids.
The shared FP8 warp-MMA factory serves both compact GEMM and MTP feedback.
Policy query schema 2 preserves existing contract meanings; generator candidate
contract version 3 includes all three MTP contracts.

The [FP8 MTP receipt](sm103-mtp-fp8-validation.json) binds frozen package source
to 51 SM103 MTP callables and 19 shared FP8 GEMM callables. All 30 preceding MTP
artifacts and all 19 shared FP8 artifacts retain identical PTX and cubins.
The 21 added MTP entries have no stack/local-memory flags. Validation passes
642 host tests, 92 SM120 regressions and six FP8 MTP cases in the companion
environment, including exact activation-byte and scale-byte parity with the
native vLLM quantizer. Memcheck and synccheck each pass five tests with zero
kernel errors under the documented SM120 CUDA API reporting exception.
Graph tests exercise input/scale mutation, scratch poisoning, fixed output
addresses, unchanged tails and frozen kernel resolution. The quantizer also
passes a 65,537-row boundary. Both distributions match all 474 package files.

Initial quantization comparisons exposed an oracle error: division by the
Python scalar 448 used reciprocal multiplication and shifted E4M3 rounding
ties. The corrected oracle and a deterministic tie regression agree with the
native quantizer. Compile attempts spanning a final formatting edit were
rejected by source-integrity checks; release artifacts use frozen source.

DeepSeek companion MTP integration, complete model/speculative evaluation,
frozen QSRT coupled high-rate conversion, matching native vLLM/ARM64 builds,
SM121 regression, Station direct-HBM transport and final consolidated evidence
remain open. Physical SM103 runtime execution remains unqualified.

## DeepSeek MTP feedback integration

Companion revision `33e4f799dd916d3b302ed52de35ab54db7d3df06` installs the
per-stream FP8 feedback owner after both linear providers finalize their
weights. GLM and DeepSeek share plan/storage ownership and the output-mutating
Torch operator. The hidden projection reserves scheduler capacity times stream
count; the model retains the flat pre-head residual and existing padding
metadata. Sequence parallelism shards raw inputs and positions together.

The [serving receipt](sm103-mtp-fp8-serving-validation.json) records 63 host
passes, 15 SM120 call-site/post-load passes and 13 related provider passes.
The DeepSeek matrix covers H256/H5120, FP32/UE8M0 checkpoint scales, both local
shard ranks, eager/Inductor execution, live counts 1/4/17 and graph mutation.
Memcheck and synccheck each pass six cases with zero kernel errors under the
existing API-reporting exception. The b12x package source remains identical to
the FP8 MTP operator receipt, preserving its SM103 compilation evidence.

The test decoder and gather fixtures isolate feedback and local sharding;
actual collectives, head collapse and full-model decoding remain unqualified.
The native vLLM library is still precompiled from a different source revision.
No GLM or DeepSeek V4 checkpoint was found in the inspected GPU-host model
locations. Qwen3.8 and DFlash2 checkpoints are available for a subsequent
full-model regression. Other open work includes frozen QSRT coupled high-rate
conversion, native/ARM64 builds, SM121 regression, Station direct-HBM transport
and final consolidated evidence. Physical SM103 execution remains unqualified.

## Checkpoint loading and full-model regression

Companion revision `7d1e2df742bbf165c9335794cfc19be40b655d36` recognizes
FlashInfer's CUDA compiler under `CUDA_HOME`. Revision
`5e040862e127518c1cf5248c8f5113ab6d2e0985` restores scoped checkpoint
allocation/copy hooks, immutable safetensors file ranges, source filtering,
owned numerical-transform inputs and post-load completion. Indexed shard
selection remains authoritative. b12x owns NVFP4 W4A16 expert row preparation.
Host loader tests pass; SM120 validation passes 81 model/post-load tests and
13 MoE numerical/graph tests. All 20 b12x direct-loader tests collect. Their
execution requires host page-table access, which the inspected RTX GPUs lack.
The occupied SM121 inference hosts are not used for GPU tests.

`benchmarks.validate_vllm_generation` records real target/draft routes,
checkpoint/source/native-library hashes, exact generated tokens, prefix reuse,
speculative counters and successful graph replays. The
[generation receipt](sm103-generation-validation.json) binds seven terminal
TP2 cases on two RTX PRO 4000 Blackwell GPUs. The checkpoint directory named
`Qwen3.8-27B-NVFP4` implements the Qwen3.5 H5120 hybrid attention/GDN contract.
Target-only b12x graph output exactly matches eager output in all six requests
of the 2,482-token corpus. Each rank records 67 target graph replays.
FlashInfer versus b12x eager output differs on two distinct short prompts.

DFlash2 executes target/draft graphs, proposes tokens and accepts proposals.
A 3,882-token prompt permits 1,648-token cache reuse under the speculative
last-block reservation. That run records 182 proposals, 85 accepted tokens,
19 target graph replays and 63 draft query graph replays per rank. Its receipt
remains failed: two prompts differ from target-only eager output. The cause
of the reference mismatches remains unresolved. Inference-time JIT warnings
also remain. These runs use precompiled native vLLM libraries from a different
source revision and do not qualify full model accuracy, complete warmup,
allocation-free replay or physical SM103 execution.

Failed toolkit setup, repeated autotuning, RPC serialization and a graph
inspection wrapper that discarded its return value remain in the raw evidence.
The durable harness uses a named worker extension and preserves graph results;
15 host tests pass, including the replay-wrapper regression and existing
serving-evidence tests. Kernel package source remains unchanged.

## ARM64 installation and offline compilation

The [ARM64 receipt](sm103-arm64-validation.json) binds the hashed combined
dependency lock, 205 installed distributions, matching b12x wheel files and
the AArch64 native loader helper. Torch imports and the helper builds/loads
with CUDA hidden and uninitialized. Nine SM103 MoE callables compile on
ARM64 and have byte-identical PTX and cubins to an x86-64 build of the same
frozen source and toolchain versions. All copied artifact hashes verify.

The NVIDIA cuSPARSELt 0.8.1 library is AArch64 and loads without CUDA
initialization. Its wheel's internal `manylinux2014_sbsa` tag still causes
`uv pip check` to fail; the wheel metadata is unmodified. The existing
inference service remains running during CPU-only validation. Matching native
vLLM builds, available-SM121 GPU regression, complete GLM/DeepSeek checkpoints,
frozen QSRT coupled high-rate conversion, the SM103 vocabulary projection
fast path, Station HBM transport and final consolidated evidence remain open.

## NVFP4 projection diagnosis

The generation harness can compare selected eager NVFP4 calls with FlashInfer
CUTLASS using identical loaded weights and live inputs. It retains the original
b12x result and records the first five distinct row counts for each layer.
Graph configurations are rejected. Six selected projections in Qwen layers
0, 31 and 63 pass 60 exact comparisons across both ranks, covering 77,721,600
BF16 output elements and row counts 1, 32, 52, 234 and 256. Generated tokens
remain identical to the uninstrumented b12x reference. Sixteen host tests pass.

A separate FlashInfer eager run on the 3,882-token corpus matches all six b12x
eager request outputs. The short-prompt mismatch and speculative reference
mismatches remain unresolved; these sampled projection results do not qualify
all model layers or decoding modes.

The FlashInfer DFlash2 control also fails exact equality with its target-only
reference, on one prompt. It reuses 1,648 cached tokens, proposes 147 tokens,
accepts 77 and records 16 target and 60 draft query graph replays per rank.
Both speculative backends therefore have a failed exact-reference check;
their speculative outputs differ from each other as well. The control narrows
attribution but leaves the correctness investigation open.

## Source-matched companion core libraries

The [native-build receipt](sm103-native-build-validation.json) binds the
companion source at `5e040862e127518c1cf5248c8f5113ab6d2e0985`, 3,309
verified source/build files, pinned external dependency revisions, compiler
commands and five x86-64 core library hashes. The build uses CUDA 13.0.88,
Torch 2.13.0+cu130, GCC 13.3 and the `10.0f;12.0f` architecture list. The
selected commands contain family targets and generic legacy targets, with
no `sm_100a` target. All five core libraries load with CUDA hidden and
uninitialized. External native modules retain their precompiled identities.

The source-matched core libraries pass 185 SM120 tests covering NVFP4
quantization, b12x MoE and GLM/DeepSeek/DSpark integration. Qwen eager output
matches all six prior eager outputs on the 3,882-token corpus. Target graphs
execute 69 replays per rank but differ from eager output on one prompt.
DFlash2 exactly reproduces the six outputs of the precompiled-core run,
including its failed comparison with target-only eager output. Both DFlash2
runs record 182 proposals and 85 accepted tokens. Rebuilding the core libraries
does not resolve the speculative parity failure.

Companion revision `1e6b809f00b5235d7b1d6711070d3eae8d5c8a18` permits
`head_dtype` overrides for `UnquantizedLinearMethod`, which ModelOpt uses for
excluded output heads. The quantized-head rejection remains. Twelve focused
CPU/device tests pass on SM120, and all required hooks pass. Qwen FP32-head
eager generation completes six requests with prefix reuse and repeat equality.
Its graph run executes 67 replays per rank but differs on two prompts. The
override fixes initialization and does not resolve exact-reference parity.
Native source remains identical to the core build revision.

The compiled FP32-head control disables CUDA graphs while retaining
`torch.compile`, matching source/native identities and the request corpus.
All six requests complete, but one prompt differs from the compiled graph
run. Compilation alone does not explain the full mismatch. Padding, launch
shape and numerical-path effects remain possible; the control does not
establish a CUDA graph replay defect.

The ARM64 core build uses the same verified source with CUDA hidden. Two
compiler attempts hit their 3 GiB and 6 GiB cgroup limits. The kernel OOM
records are retained outside the repository. The bounded continuation retains
a 6 GiB resident-memory cap, allows up to 8 GiB swap and runs one compiler job.
The build completes all five core targets. All 3,309 source/build files
match the declared revision; the libraries have AArch64 ELF identities and
load with CUDA hidden and uninitialized. The command audit retains the
`sm_100f` family target. The existing inference service remains running and
returns HTTP 200 on its health endpoint. This qualifies compilation and CPU
loading; GPU execution remains unqualified.

## Planned CuTe vocabulary projection and companion warmup

The vocabulary component implements an SM103 CuTe backend through its existing
public plan/bind/run API. It reuses the BF16 GEMV reduction, compiles static
geometry before capture and owns a fixed output buffer. Live row counts reuse
the compiled callable; caller-owned output is supported with overlap guards.
Both legacy Triton variants also use Int64 matrix-row offsets. The embedded
SM120/SM121 configs retain their values; only the vocabulary config schema
version changes. The generator qualifies SM103's actual CuTe plan and rejects
nonfinite, zero, low-cosine or top-1-mismatched output before timing.

The [vocabulary receipt](sm103-vocab-validation.json) binds the reviewed source,
16 SM103 projection callables and six vocabulary specializations. The reviewed
PTX and cubins are byte-identical to the pre-formatting source. Vocabulary
specializations use 40 or 48 allocated registers and 1,024 bytes of shared
memory, with no stack/local memory. Thirteen component tests pass on physical
SM120, including frozen resolution across rows 1/4/8/9/17, graph mutation and
poisoning, stable output storage, Inductor and matrix offsets above Int32.
Seven selected memcheck tests report zero memory errors; the existing API
reporting exception is explicit in the receipt. The selected host suites pass
352 tests after rerunning 21 /tmp-quota failures on the evidence filesystem.

Companion model constructors pass actual output heads to the logits processor.
Its vocabulary warmup provider runs the selected plan before capture and
preserves explicit output-head precision overrides. Ninety-four selected
companion tests pass, including CuTe logits-processor capture/replay. This
integration applies the vocabulary plan to one live row. Full-model accuracy,
speculative parity and SM103 execution remain unqualified.

Companion revision `5c0857f9cd` records the constructor and warmup integration.
Qwen eager, graph and DFlash2 runs each preserve all six prior outputs. Eager
generation passes its required gates. Graph generation retains one prompt
mismatch against eager; DFlash2 retains two. Graph counters record 69 target
replays per rank without speculation and 19 target/63 draft replays with
speculation. DFlash2 accepts 85 of 182 proposed tokens. Raw failed receipts
remain unchanged; regression equality does not qualify speculative accuracy
or complete serving warmup.

The frozen vocabulary source also compiles all 16 projection callables on
ARM64. Every PTX file and cubin matches the x86-64 build byte-for-byte, with
CUDA uninitialized before and after compilation. The existing inference
service remains healthy. The local consolidated-corpus compiler is killed
by global host OOM at 4,659,580 KiB anonymous RSS; the incomplete artifacts
and kernel OOM record remain outside the repository. A separate build of the
same source runs on the larger host and must pass before its corpus is accepted.

## SM121 component regression and consolidated vocabulary source

The [SM121 receipt](sm103-sm121-validation.json) records 126 passing tests,
four explicit skips and two RMS-concat oracle failures on GB10. FP64 dot
products establish that both disputed values round to BF16 1.0, as the CuTe
kernel returns. The BF16 GEMM oracle changes this tie through reduced-precision
partial reductions. The reference projection now accumulates in FP64; all
five affected cases pass on both SM121 and SM120 with unchanged tolerances
and exact top-1 assertions. Runtime kernels are unchanged.

The suite includes 13 vocabulary tests, 48 BF16 GEMV tests, four NVFP4 loader
tests and 34 direct-loader tests. Seven vocabulary memcheck cases report zero
memory errors, with the API reporting exception retained explicitly. The
direct-loader cases exercise high file offsets, TP slices, registered storage
and write-combining storage. The four original vLLM containers are restored
with their original image/container identities and configuration; the service
returns HTTP 200.

The [consolidated receipt](sm103-consolidated-validation.json) verifies all
1,108 callables and 1,116 CUDA entries compiled from frozen source `7b697152`.
The corpus adds 58 BTX paired-record, 51 MTP and six vocabulary callables to
the preceding 993-callable manifest. It retains 976 identical PTX files and
936 identical cubins for existing callables, with no removals. The nine
positive register deltas add no stack/local traffic or static shared memory.
All 44 existing stack/local flags remain visible. Source and Torch/Triton
versions differ, so this is not an isolated compiler comparison. Twenty-three
physical-SM103 suites are prepared from the same frozen source; none executes.
The later oracle correction is not substituted into these raw source hashes.

The [primary checkpoint inventory](sm103-primary-models.json) resolves the
earlier checkpoint-location gap: both target families and the GLM DFlash2
draft are on maxwell. It also identifies a separate implementation gap:
companion `5c0857f9cd` does not register the actual V4.1 architecture. The
V4.1 model, CED/cache/Engram integration, required SM103 component admissions
and native MXFP4 expert representation need implementation and qualification.
The existing V4 integration does not satisfy that model contract. Full-model
accuracy, speculative parity, complete serving warmup, Station HBM transport
and frozen QSRT coupled high-rate conversion remain open.

## V4.1 supporting operators on SM103

MLA compression, HyperConnection and embedding now admit SM103 through their
existing metadata, policy and compiler gates. The HyperConnection policy
selects a complete CuTe backend on SM103; SM120/SM121 keep their selected
normalization and gate-reduction routes. The backend is retained on the plan
and does not require runtime device selection. Config and candidate contract
versions advance to 2. Only the HyperConnection config schema changes in the
three embedded profiles; measured configs remain unchanged.

The [V4.1 component receipt](sm103-v41-components.json) binds 72 supporting
callables and 74 CUDA entries to the tested source. None emits stack/local
memory traffic. The actual N129280/K5120 vocabulary shape also compiles, with
48 allocated GPRs and 1,024 shared bytes. All 16 existing projection PTX files
and cubins remain byte-identical. The compiler and qualification launchers
include the three component groups.

Both SM120 and SM121 pass 55 tests with one explicit multiple-GPU skip.
Each also passes 12 selected memcheck cases with zero memory errors, retaining
the API-reporting exception. These include high state/row offsets beyond
Int32, live counts and mutations during graph replay, and the complete CuTe
HyperConnection chain selected through a policy override. Physical SM103
execution and complete model serving remain unqualified. The distinct V4.1
companion port is prepared in an isolated worktree; its cache/encoder/Engram
integration and supported expert checkpoint representation remain separate work.
The SM121 test window restores all four original inference containers and
images, preserves their configuration hashes, and ends with HTTP 200 health.

## Engram resident scales and bounded disk prefetch

Engram disk tables accept owned E8M0 scale storage and one outstanding prefetch
per table. The read retains its transaction and preparing thread/stream until
consumption or cancellation. Existing Grace and mapped storage owners remain
part of the lookup binding. Tests vary scales by source row, so global scale
addressing cannot accidentally match compact staging addressing.

The [Engram receipt](sm103-engram-validation.json) records 24 passing SM120
tests and 24 passing memcheck cases with zero kernel memory errors; API-error
reporting is explicitly disabled. Twenty-five host checks pass. The compiler
adds 16 hash/gather callables covering primary table geometry and TP2 shards;
none uses stack/local memory. The qualification launcher includes Engram.
The test host uses task-local liburing development headers and its existing
liburing runtime. Earlier missing-library and incomplete-port failures remain
in external evidence. Physical SM103 execution and SM121 coverage of these
disk additions remain unqualified.

## Cooperative MoE residency and planned capacity

Compact repacked W4A8 launches cap their grid using the compiled CUDA
function's occupancy, actual block thread count and launch shared memory.
The bound is cached before capture. Specializations that support two resident
CTAs retain that grid capacity. Disk-cache loads verify object and launch
metadata integrity before loading a temporary object copy.

Shared-input and intermediate-materialization choices use planned capacity.
The [cooperative MoE receipt](moe-cooperative-validation.json) records four
SM120 cases with capacities 8 and 16, multiple live counts, independent
numerical oracles, poisoned outputs and frozen kernel resolution. Those cases
also pass memcheck; ten compatibility cases and 116 host cases pass.

SM121 passes all four cooperative cases and 24 Engram cases. Twelve selected
cases pass memcheck with zero memory errors and API-error reporting explicitly
disabled. The Engram coverage includes disk prefetch, resident scales, fresh
graph replay and high global row offsets. These results extend the Engram
receipt's SM120 coverage without implying physical SM103 qualification.

A reduced V4.1 model produces identical eager and graph output tokens across
six synthetic sequences. The graph arm records 21 replays; both arms freeze
b12x kernel resolution after warmup and use JIT monitor error mode. Synthetic
weights do not establish checkpoint accuracy.

## Trellis SiLU input clamps

Uncoupled SM103 Trellis execution accepts a finite positive SiLU input clamp.
The intermediate transform clamps gate values above the limit and up values
to its signed range before SiLU, retaining FP16 transform rounding. Coupled
SiTU execution rejects this option. The activation limit is static compile
metadata; live counts remain runtime arguments.

The [SiLU clamp receipt](sm103-trellis-silu-clamp-validation.json) records
119 passing host checks and one physical-SM103 skip, 30 passing portable
SM120 transform checks, and five passing memcheck cases with zero kernel
memory errors. API-error reporting is explicitly disabled. The checks include
large activation values, frozen compilation, graph replay and high offsets.
The compiler emits 28 callables for V4.1 E384/H5120/I2304, top-6, K3 experts
with limit 10 at TP1 and TP2. No stack/local traffic is emitted. Complete
SM103 expert execution and V4.1 Trellis checkpoint loading remain unqualified.

## Clamped Trellis capacity binding and V4.1 checkpoint qualification

Status: implemented; SM120 component regression and native V4.1 SM121
checkpoint qualification recorded; physical SM103 execution unqualified.

SM12x uncoupled Trellis execution preserves FP16 boundaries around clamped
SiLU preactivation, the activation product and the FC2 input transform. Plans
retain capacity launches for shared and per-expert transform tables. Binding
selects immutable layout metadata and rejects a conflicting math setting;
live row counts do not resolve a different compiled callable. The component
policy continues to select packed routing.

The [Trellis capacity receipt](moe-trellis-capacity-validation.json) records
131 passing host checks, 23 GPU-only skips and 75 passing SM120 tests. A
clamped BTX fixture exercises live counts 1/3/7/8 under one compiled callable,
with finite/nonzero output, an independent oracle and frozen resolution.
Mapped-output cosine checks accumulate both operands in FP32, and cache-reuse
checks follow the registered packed-routing policy.

The [primary checkpoint inventory](sm103-primary-models.json) identifies the
implemented V4.1 companion port and its native SM121 TP4 checkpoint receipt.
All four workers freeze b12x kernel resolution after warmup. Short prompts
and 4K-prefix reuse match the reference, and the bounded GSM8K check records
28/32 versus 27/32 without invalid answers. The native-library compatibility
image, source hashes and original-service restoration are recorded separately
from BTX expert accuracy and SM103 hardware qualification.

## Uniform BTX tensor-parallel projection tiles

Status: implemented; synthetic SM120 TP2 eager/graph parity qualified;
checkpoint accuracy and physical SM103 execution unqualified.

Trellis weight planning, scratch planning and uniform BTX preparation share
one geometry-derived tile default. When either projection width requires
128-column tiles, both fused projections use K64/N128 to preserve projection
boundaries and equal CTA thread counts. Explicit overrides remain available.
V4.1's uniform TP2 partition has 1152 intermediate channels per rank.

The [TP2 validation receipt](moe-trellis-tp2-validation.json) records 240 passing
host checks, four passing SM120 preparation/numerical checks and one memcheck
case with zero kernel memory errors. Six synthetic model sequences produce
the same 48 tokens in eager and graph execution. Each rank completes 21 graph
replays with frozen resolution and shared-expert overlap enabled.

SM12x paired-rate execution remains limited to one 256-channel pair per rank.
V4.1's wider paired extents use the separate SM103 atom implementation; the
uniform SM120 result does not qualify that implementation or BTX model accuracy.

## Consolidated Trellis TP2 source and package artifacts

Status: cross-compiled and packaged; physical SM103 execution unqualified.

The [consolidated receipt](sm103-consolidated-validation.json) binds revision
`2e6c166e` to 1,225 callables, 1,235 CUDA entry points, and their PTX, cubin,
SASS and resource hashes. The corpus includes clamped TP1/TP2 Trellis,
MLA compression, HyperConnection, embedding and Engram. Qualification
preparation verifies the artifacts against the clean source and emits all
27 operator-suite commands. All 27 suites collect successfully without
executing on a GPU.

Compared with source `7b697152`, 117 callables are added and none removed.
All 17 positive register-set deltas remain recorded. Two supporting MXFP4
packers add two allocated GPRs each; no existing callable changes stack,
local memory, static shared memory or local load/store counts. The 44
stack/local flags are unchanged and no added callable has a flag. Source
and Torch/Triton differences prevent attributing these deltas to a compiler
migration; occupancy and latency remain unqualified.

The matching wheel includes all Python sources and three embedded profiles.
All 476 packaged files match the checkout, and 56 host checks pass with imports
resolved from the extracted wheel. The frozen Git archive, complete compile
corpus, wheel and qualification commands remain outside the repository with
their identities recorded in the compact receipt.

## GLM planned KDA and pooling integration

Status: implemented in companion revision `5d58e328c1`; SM120 component and
synthetic model checks pass; checkpoint accuracy and SM103 execution unqualified.

The companion uses public KDA prefill plans for mixed batches, retains separate
speculative output ranges, and binds the public sparse MLA API. Cache negotiation
preserves complete C4 pages and excludes padding beyond planned context. Pooling
helpers keep live counts out of Triton specialization and use Int64 scaled
offsets.

The [primary model inventory](sm103-primary-models.json) records 153 passing host
checks, 31 SM120 pooling checks and three KDA checks. Six synthetic sequences
produce identical 48-token outputs in eager and graph execution, with 21 graph
replays and frozen resolution. Eleven supporting pooling variants have inspected
SM103 PTX, cubins, SASS and resources, with zero stack or local memory. Full GLM
checkpoint qualification remains separate from these results.

## GLM speculative configuration and external native artifacts

Status: synthetic SM120 DFlash2 execution qualified; external native
compilation and CPU loading qualified; checkpoint and SM103 execution
remain unqualified.

The GLM DFlash2 configuration selects the draft cache dtype independently of
the target's compressed MLA layout and reserves fixed capacity for both models.
Six requests on the four-layer dummy target and two-layer dummy draft produce
the same 48 tokens as target-only execution. The worker records 54 total graph
replays with frozen b12x resolution and no inference-time compilation.

The companion native build produces FlashMLA, its extension, FlashKDA,
QuTLASS and DeepGEMM libraries. All five load with CUDA hidden and uninitialized;
311 CUDA entry records have saved resource reports. The build uses CUDA 13.0.88,
Torch 2.13.0 and verified dependency revisions. The split toolkit requires the
installed CUDA-library header path and an isolated NVRTC linker name. The
recorded core library hashes are unchanged. These external outputs have no GPU
qualification and do not replace the libraries used by the GLM trials.

## KDA state continuity and GLM checkpoint correctness

Status: public KDA prefill-to-decode continuity qualified on SM120; real GLM
checkpoint correctness failed on SM121; physical SM103 execution unqualified.

The public sequence test uses 16 local heads, a 26-token prefix, and live
decode counts 1/4/1. Both AUTO and CuTe decode agree with the independent FP32
recurrent oracle. Live state slots begin beyond the Int32 element-offset
boundary. Frozen resolution, poisoned null state, stable addresses and
allocation-free graph replay pass.

The GLM NVFP4 checkpoint loads all 45 layers on four SM121 workers using the
companion source recorded in the [primary model inventory](sm103-primary-models.json).
All workers freeze b12x resolution after warmup and record no inference-time
compilation. The arithmetic prompt returns `38` for `19 + 23`; the capital
and expression prompts return `Canberra` and `x + 1`. This fails the checkpoint
correctness gate. Prefix reuse, GSM8K, graphs and DFlash remain unexecuted in
that trial. The original four containers, images and configurations are
restored with a healthy endpoint. Component and dummy-model results do not
qualify this checkpoint.

The controlled KDA reference diagnostic retains the same source, weights,
configuration and other b12x components. Reference KDA prefill and decode
return `42`; the capital and expression token IDs match the b12x trial.
Prefix C1/C4/repeat outputs also match each other. The bounded GSM8K check
passes 29/32 questions with zero invalid answers. All four workers freeze
b12x resolution; each records two inference-time compilations in the
reference KDA path. Original-service restoration passes again. This
associates the arithmetic discrepancy with the KDA configuration. It does
not isolate its cause across separately loaded model instances.

A subsequent run uses reference prefill and b12x decode and still returns
`38`, tied with `42` in the reported decode logits. Its first-token logits
already differ from the reference run, before decode begins. Captures from
all 136 layer/rank pairs preserve the prefill state exactly into decode.
SM120 AUTO replay reproduces the SM121 b12x outputs and states exactly;
CuTe and reference decode closely match the independent FP32 oracle on
those same inputs. B12x prefill output/state relative RMS errors stay below
0.007/0.004. These captures do not establish a decoder defect or resolve
the model discrepancy. The original service is restored with HTTP 200.

## ARM64 external native build

Status: compilation and CPU loading qualified; GPU execution unqualified.

The isolated ARM64 build now produces FlashMLA, its extension, FlashKDA,
QuTLASS and DeepGEMM. All ten core and external libraries load together
with CUDA hidden and uninitialized. The 376 checked source/build files and
nine dependency revisions match the verified x86 build, and the original
five core library hashes remain unchanged.

The external artifacts contain 195 SM100-family CUDA entry records. Resource
reports retain all 51 entries with positive stack usage; none reports local
memory. Host compilation uses no native CPU architecture flags. Commands,
binaries, resource reports and hashes are preserved outside the repository.
These results do not qualify GPU execution, DeepGEMM runtime JIT,
FlashAttention, the Rust parser or a complete companion native build.

## GLM sparse selection and fixed decode splits

Status: component correctness qualified on SM120; SM103 runtime unqualified.

A resident-checkpoint diagnostic records 1,632 KDA input/output captures across
six requests. Repeated requests diverge with both reference-only and b12x-only
KDA. The first three KDA layers remain identical on every rank; the following
KDA layer receives different inputs after sparse attention layer 3. The bounded
GSM8K check returns 27/32 correct with no invalid answers. These observations do
not isolate a KDA arithmetic defect.

GLM C4 selections store partial-pool tails after interior padding. The sparse
adapter now compacts valid selections into a stable prefix before supplying its
length to attention. The regression fails numerically before the fix and passes
through both native and portable decode/extend paths afterward, including live
cache pages beyond the Int32 byte-offset boundary. DCP preserves the selected
order on both ranks.

Native sparse MLA also consumes a fixed split count from its typed plan. Config
schema 3 serializes that count for both backends; native AUTO uses one split.
The previous runtime-planner profile entries have empty coverage until the fixed
schedule is requalified. No performance improvement is claimed.

The component evidence includes 37 GLM GPU tests, 17 remapping GPU tests, four
Compute Sanitizer cases with zero errors, frozen kernel resolution across live
row counts 1/3/26, stable addresses and allocation-free graph replay. All 19
supporting metadata variants cross-compile for SM103 with no stack or local
memory. The rebuilt wheel also passes the four attention regressions. Its raw
package fingerprint remains distinct from the checkout because four repository
Markdown files are excluded from packaging.

Commands, identities, limitations and checkpoint qualification state are recorded
in [the sparse selection receipt](sm103-glm-sparse-validation.json).

## GLM speculative pooling and unmapped slots

Status: component regressions qualified on SM120; checkpoint accuracy and SM103
execution unqualified.

Companion commits `e1587ea558` and `ac96719947` preserve the preceding partial
pool across rejected verifier rows and reject unmapped slots before packed page
conversion. Independent GPU regressions reproduce both defects on the preceding
sources. The fixed history ring uses planned capacities 4/11/18 and absolute
token positions; all scaled offsets remain Int64. The final source passes
53 cases under Compute Sanitizer in complementary eight- and 45-case processes,
with zero errors. A combined process retains 22.23 GiB and fails three allocations
after 50 passes; its failure remains in the evidence.

The final 27 metadata variants compile for SM103, producing 108 verified
artifacts without stack or local-memory use. The unmapped-slot guard adds eight
registers to one packed decode variant, from 48 to 56, with unchanged 1,024-byte
shared memory. Physical profiling must assess the retained resource flag.

The selection-fix checkpoint trial passes 11 fixed requests and produces the
same 39 tokens across eager, graphs and DFlash on four SM121 workers. GSM8K scores
are 29/32, 30/32 and 25/32 with zero invalid answers. DFlash fails the comparative
accuracy gate despite accepted draft proposals and clean frozen graph execution.
The history-only graph control scores 27/32, also failing the declared 29/32
floor, so that controller stops before DFlash. Each trial restores the original
four containers and HTTP 200 health. These failures remain separate from the
independently reproduced component defects and their passing regressions.

The [GLM receipt](sm103-glm-sparse-validation.json) binds source revisions,
commands, raw logs, offline artifacts and per-trial service restoration.

The final-source repeatability diagnostic uses identical companion
`ac96719947` source in graph and DFlash modes. Serial first/repeat and C4 scores
are 26/32, 28/32 and 27/32 for graphs, versus 26/32, 28/32 and 28/32 for DFlash,
with no invalid answers. Exact serial text matches are 4/32 and 3/32 with prefix
caching enabled. These observations do not isolate a DFlash-specific regression
or waive the earlier failed gates. Both modes pass the same 11 fixed requests
and 39 output tokens; all four workers freeze resolution and replay graphs
without inference JIT. Completed DFlash log intervals record 7,427 accepted
proposals from 13,811 drafted tokens, without a performance claim.

The original four-container service is restored with the same IDs, images and
configuration hashes. HTTP health returns 200, and an operational generation
check returns 42. Source archives and every mounted package file on all four
hosts verify after execution. The immutable diagnostic manifest is bound by the
GLM receipt; runtime sources remain unchanged by the documentation commits.
