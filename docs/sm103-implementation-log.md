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
  imports two modules from it. Four exact historical dependency files from
  `070af610^` are supplied through an isolated `/tmp` PYTHONPATH to run that
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

To reproduce the isolated historical dependencies, copy these exact paths from
`git show 070af610^:<path>` under `/tmp/b12x-sm103-historical-tools/`:

```text
validation/cutlass_migration/evidence/kernel_resources.py
validation/cutlass_migration/acceptance/corpus/ptx_capture.py
validation/cutlass_migration/core/comparison_identity.py
validation/cutlass_migration/paths.py
```

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
are in `../b12x-sm103-evidence/final-sm103/`.

## Compiler attempts and limitations

Python 3.12, PyTorch 2.14.0+cu130, CUTLASS DSL and library wheels 4.6.2, and
cuda-python 13.4.1 are installed in the isolated virtual environment. CPU-only
`cute.compile(..., no_jit_engine=True, options="--gpu-arch=sm_103a")` produces
actual PTX, cubins and host-side MLIR without a CUDA driver context.

The compile corpus contains nine MoE launchers, eight TP2 RoCE launchers and
three Trellis reconstruction launchers. NVIDIA `nvdisasm` 13.4.49 can inspect
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
