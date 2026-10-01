# CUTLASS DSL 4.7.1 qualification

All five CUTLASS DSL packages are pinned to 4.7.1. The GPU 0 compiler
comparison passes its selected correctness corpus on both 4.6.2 and 4.7.1.
Production specialization coverage and targeted timing of resource increases
remain outstanding.

## Release scope

The [4.7.1 release notes](https://github.com/NVIDIA/cutlass/releases/tag/v4.7.1)
describe fixes for warp-specialized `setmaxnreg` compilation, decorator leaks,
TVM-FFI tensor byte offsets and streams inside tuples, import-time
`CUTE_DSL_LIBS` mutation, and exported C headers for 64-bit shapes. These touch
b12x compilation, executable lifetime, tensor views, and runtime integration.
They do not establish correctness or performance for b12x kernels.

The [tagged changelog](https://github.com/NVIDIA/cutlass/blob/v4.7.1/CHANGELOG.md)
also records 4.7.0's experimental Primitives API, task scheduling framework,
and diagnostics for register spills and local memory. The compiler comparison does not
adopt those APIs or change kernel math, scheduling policy, or launch geometry.

## Source and environment

Evidence was collected on 2026-09-29 from base revision
`a489f972e0dde54fedd5f83bf73a7d3754fc60d6`, with the vLLM compatibility changes
present in both compiler arms. The compiler-comparison b12x package fingerprint is
`fa71aece409411908aafa0fb057689ce01fba88a69cb2fc8d7016336fdc7f5f6`.
This comparison isolates compiler changes; it is not a comparison against
pristine b12x source.

| Coordinate | Value |
| --- | --- |
| Worktree | `/home/luke/projects/b12x-cutlass-4.7.1` |
| Baseline Python | `/home/luke/projects/vllm/.venv/bin/python` |
| Candidate Python | Worktree `.venv/bin/python` |
| Python / PyTorch | 3.12.12 / 2.13.0+cu130 |
| Physical GPU | 0, RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| GPU UUID | `GPU-a0816187-68b2-b679-587f-0e56bac804f5` |
| Cubin PTXAS metadata, both arms | 13.3.27, `-O3 -arch sm_120a` |
| Evidence directory | `/tmp/b12x-cutlass471-evidence` |

The shared GPU had approximately 90 GiB allocated to an existing serving
process. No serving process, clock setting, or GPU assignment was changed.
No compiler A/B latency or throughput measurements were collected.

## Correctness and serving checks

The selection in `validation/cutlass_migration/corpus.txt` exercises dense
blockscaled GEMM, fused activation quantization, tensor FP8, W4A16 and W4A8 MoE,
paged and contiguous attention, and prepared graph replay. It includes live
input mutation, output poisoning, capacity and callable reuse, stable replay
allocations, paged offsets beyond 2 GiB, and replay after compiler-cache eviction
where supported by the selected tests.

Both compiler arms produce **113 passed, 4 deselected**. The four
FlashInfer/cuDNN comparison cases are excluded with
`-k 'not matches_flashinfer_cudnn'`. They are not acceptance evidence.

The grouped MXFP8 test uses an FP64 oracle with an unchanged bitwise assertion.
The MoE capacity test supplies an undersized route-output view to a deterministic
binding and requires rejection. Both corrections are applied to both compiler
arms. The passing logs are `/tmp/b12x-cutlass462-corpus-fixed-tests.log` and
`/tmp/b12x-cutlass471-corpus-fixed-tests.log`; their summary is
`/tmp/b12x-cutlass471-evidence/test-fixes-result.json`.

The migration test helpers forward `device_ordinal` keyword arguments and use
the public canonical MoE weight preparation API required by the binding helper.
The MoE test registers binding cleanup as a pytest finalizer so an assertion
failure cannot defer stream synchronization into another test's graph capture.

Four offline artifact-integrity tests, targeted Ruff checks, `git diff --check`,
and candidate `uv pip check` pass.

## Resource comparison

The sampled caches contain **88 manifest-bound objects and 96 CUDA entries per
compiler**. Every sampled entry has an exact-symbol match. The sample includes
14 kernel identifiers, one of which represents direct test bindings; it is not
a census of every production kernel and specialization.

`validation/cutlass_migration/resource_census.py` verifies manifest, semantic,
object, and artifact-evidence hashes; extracts cubins to separate files; records
exact R/UR/P/UP sets and launch resources; and compares matching entries. Raw
hashes remain unchanged. Cross-compiler identity removes only the disabled
`enable-pyir=false` option, its nested compile-option equivalent, and the
package-owned runtime library path injected into `CUTE_DSL_LIBS` by 4.6.2.
The nested keyword hash is recomputed for comparison only. Other options,
custom library paths, source fingerprints, non-CUTLASS toolchain fields, and
CUDA symbols must match.

Six entries have positive deltas. Comparison-key prefixes identify exact rows
in `final-resource-delta.json`; full semantic payloads are in each arm's
`final-resources-*/resources.json`.

| Kernel / comparison key | GPR allocation | Other positive deltas | Disposition |
| --- | --- | --- | --- |
| Dense MXFP8 BF16 / `86d41b7b4bcf` | 56 → 56 | +256 code bytes, +16 instructions | Small code increase; timing not collected. |
| Grouped fused MXFP8 / `84909fa4d18c` | 55 → 55 | +3 R, +1 UR, +128 bytes, +8 instructions | Register-use increase retained for timing review. |
| Fused MXFP8 quantization / `578dd24ce303` | 40 → 48 | +6 R | Prioritize targeted timing. |
| Dense MXFP4 BF16 / `1eae71642316` | 49 → 51 | +4 R, +128 bytes, +8 instructions | Targeted timing pending. |
| Dense MXFP8 FP16 / `9ca9b526cfee` | 56 → 56 | +256 code bytes, +16 instructions | Small code increase; timing not collected. |
| Grouped fused MXFP8 / `f39ee13bf2e5` | 40 → 48 | +8 R | Prioritize targeted timing. |

Driver occupancy is unchanged for all 96 paired entries, using the extracted
cubins, exact required thread dimensions, and manifest-bound dynamic SMEM.
No paired entry increases frame size, minimum stack, local load/store count,
static or dynamic SMEM, P/UP usage, or changes thread count or `SETMAXNREG`.
The original cache objects and manifests remain hash-identical after the
combined correctness runs. These results do not establish runtime performance.

## Reproduction

Run from the migration worktree. Use separate, initially empty cache directories
for a fresh comparison. Both interpreters must resolve the same b12x source.
The baseline interpreter is deliberately supplied through `PYTHONPATH` because
its editable installation points at the main checkout.

```bash
mapfile -t cases < validation/cutlass_migration/corpus.txt
CUDA_VISIBLE_DEVICES=0 \
  B12X_COMPILE_CACHE_DIR=/tmp/b12x-cutlass471-evidence/cache-462 \
  PYTHONPATH="$PWD" /home/luke/projects/vllm/.venv/bin/python -m pytest \
  "${cases[@]}" -k 'not matches_flashinfer_cudnn' -q --tb=short
CUDA_VISIBLE_DEVICES=0 \
  B12X_COMPILE_CACHE_DIR=/tmp/b12x-cutlass471-evidence/cache-471 \
  .venv/bin/python -m pytest \
  "${cases[@]}" -k 'not matches_flashinfer_cudnn' -q --tb=short
```

The resource collector requires a nonexistent output directory. Preserve raw
artifacts by choosing another output name when repeating collection.

```bash
for arm in 462 471; do
  .venv/bin/python validation/cutlass_migration/resource_census.py collect \
    "/tmp/b12x-cutlass471-evidence/cache-$arm" \
    "/tmp/b12x-cutlass471-evidence/resources-repeat-$arm" \
    --nvdisasm /opt/cuda/bin/nvdisasm
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    validation/cutlass_migration/resource_census.py occupancy \
    "/tmp/b12x-cutlass471-evidence/resources-repeat-$arm/resources.json" \
    "/tmp/b12x-cutlass471-evidence/occupancy-repeat-$arm.json"
done
.venv/bin/python validation/cutlass_migration/resource_census.py compare \
  /tmp/b12x-cutlass471-evidence/resources-repeat-462/resources.json \
  /tmp/b12x-cutlass471-evidence/resources-repeat-471/resources.json \
  /tmp/b12x-cutlass471-evidence/resource-repeat-delta.json
.venv/bin/python -m pytest validation/cutlass_migration/test_resource_census.py -q
```

## Acceptance gaps

The repository's migration gate still requires a closed production
specialization census, disposition of positive resource deltas through targeted
runtime evidence where warranted, and correctness/serving qualification for
uncovered paths. This sample does not cover all MLA/DSA, quantization,
communication, or attention variants. Separate vLLM integration checks under 4.7.1 exercised
`nvidia/NVIDIA_Super3_5_VL_IQ2XXS-Packed` on GPU 0 with BF16 activations, TP1,
b12x linear and MoE backends, and full/piecewise CUDA graphs. Six request checks,
including an image request, passed; an eight-question GSM8K smoke evaluation
scored 7/8 with no invalid outputs. These checks use the integration and heuristic
planner changes and are not compiler A/B evidence. Irregular-prefill route-pack
JIT warnings remain. Logs and request artifacts are retained locally under
`/tmp/vllm-b12x-super3-run/`.
