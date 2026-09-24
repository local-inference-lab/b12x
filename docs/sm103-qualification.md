# SM103 support and qualification

SM103 (B300/GB300) operators use the declaration, preparation-session,
binding and execution contracts in [GPU preparation](gpu-profiles.md). Core
compute is CuTe DSL with tcgen05/TMEM; supporting packing and metadata kernels
may use Triton. Architecture recognition alone does not admit an operator:
each component's capability metadata and planning contract decide support.

## Operators

| Area | Implementation |
| --- | --- |
| Quantized projections | [SM103 dense lowering](../b12x/gemm/_sm103_preparation.py) supplies NVFP4, MXFP4, MXFP8, both MXFP6 formats, W6A8 and tensor/block FP8 programs. [Packed linear adapters](../b12x/gemm/blockscaled/) keep inline weight dequantization and bounded workspace. |
| Native MoE | [NVFP4](../b12x/moe/fused_moe/_sm103.py) and [Trellis](../b12x/moe/fused_moe/_sm103_trellis.py) MoE keep CuTe routing, tcgen05 projections and weighted reduction. Trellis covers uniform, intermediate-Hadamard, mixed-rate and grouped representations, including paired EXL3 records through `Exl3Source` and `Exl3Weights`. |
| Attention and indexing | [Sparse MLA](../b12x/attention/sparse_mla/_sm103.py), compressed MLA, dense MLA and DSA indexing keep their cache layouts and fixed launch schedules. |
| Model components | CuTe KDA/GDN, MTP feedback, mHC, HyperConnection, vocabulary projection, block-FP8 linear and DeepSeek WO. |
| Storage | Engram owns device or mapped-host allocations and checks Grace capability. Disk lookups are synchronous. |

Pools, vocabularies and paged state that can exceed 32-bit scaled offsets use
Int64 addressing; GPU tests place live data past that boundary.

## Checks without a B300

```bash
python -m pytest tests/architecture tests/preparation -q
python scripts/compile_sm103_prepared.py --workers 4 --output-dir "$OUT/prepared"
python scripts/compile_sm103.py --component all --output-dir "$OUT/native"
```

`compile_sm103_prepared.py` compiles the production program list of each typed
declaration through the offline worker contract without initializing CUDA.
`compile_sm103.py` compiles the wider kernel corpus; add `--nvdisasm` and
`--cuobjdump` with absolute paths to keep SASS and resource reports.
Compilation cannot establish correctness, graph behavior or speed.
`qualify_sm103.py` without `--execute` lists the GPU tests it would run.

Portable regressions can run on an SM120/SM121 GPU compiled for that
architecture. Native SM103 tests skip elsewhere, and a skip is not a pass.

## Physical qualification

Select the device by UUID with no other compute process on it. The launcher
refuses non-SM103 devices, checks the source and artifact hashes against the
compile manifest, and fails on any test failure or skip.

```bash
python scripts/qualify_sm103.py --execute --device-uuid "$UUID" \
  --compile-manifest "$OUT/native/manifest.json" --output-dir "$OUT/runtime"
python scripts/qualify_sm103.py --execute --device-uuid "$UUID" \
  --compile-manifest "$OUT/native/manifest.json" --sanitizer "$COMPUTE_SANITIZER" \
  --sanitizer-tool memcheck --sanitizer-exclude-kernel nvjet --output-dir "$OUT/memcheck"
```

Repeat the sanitizer run with `synccheck`, and with `racecheck` for components
whose shared-memory ownership requires it (`--component blockscaled`).
Sanitizer runs deselect `device_trap` tests, which pass only when a kernel
traps on invalid input. `--sanitizer-exclude-kernel nvjet` leaves the cuBLAS
kernels behind Torch reference matmuls uninstrumented, because synccheck
cannot launch them; every b12x kernel stays instrumented and the receipt
records the exclusion.

Triton-built support kernels need Python headers, and Engram disk lookup needs
`liburing-dev` and `pkg-config`.

## GB300 results

One NVIDIA GB300 (compute capability 10.3, 152 SMs, verified coherent Grace
memory) with driver 595.91.07, CUDA 13.2, Compute Sanitizer 2026.1, Python
3.12, Torch 2.13.0+cu130, CUTLASS DSL 4.6.2 and Triton 3.7.1. All stages ran
from one frozen snapshot of `f22d325e` with a clean worktree.

| Stage | Result |
| --- | --- |
| `compile_sm103.py --component all` | 1,163 callables cross-compiled |
| `compile_sm103_prepared.py` | 83 declarations, 230 distinct programs (223 CuTe, 7 Triton), no failures |
| Host tests, `tests/architecture tests/preparation` | 941 passed, 43 skipped; one checkout-identity test needs a `.git` directory and passes in a checkout |
| `qualify_sm103.py --execute` | 27 components, 1,262 tests, no failures, errors or skips |
| Memcheck, every component | 1,256 tests; every summary reports 0 errors |
| Synccheck, every component | 1,256 tests; every summary reports 0 errors |
| Racecheck, `blockscaled` | 104 tests, 0 hazards |

Sanitizer runs exclude the six `device_trap` tests and leave cuBLAS `nvjet`
kernels uninstrumented, as described above.

Physical execution found these defects, fixed here:

- An MXFP4 dense GEMM with K=128 had one scale-factor K atom, smaller than
  its TMA box, and trapped. The K tile is clamped to the reduction extent.
- A composite MoE plan dispatching a smaller prepared capacity variant
  rejected an output sized for the plan capacity; the variant now receives
  the live rows.
- The prepared FP6 serving op lowered only the SM12x GEMM. It now prepares
  the native SM103 quantizer and GEMM; output matches the eager SM103 path
  bit for bit.
- `blockscaled.quantize_mxfp4` was imported but not exported.

The block-scaled GEMM keeps four TMA stages in flight. Against two stages, the
dense 4096³ GEMM falls from 202.9 to 179.6 μs (NVFP4), 199.3 to 177.2 μs
(MXFP4) and 285.7 to 206.0 μs (MXFP8), with bitwise-identical outputs.

## Limits

Operator qualification covers one GB300. Complete-model serving, B300
performance and the vLLM integration are not qualified here. IQ2_XS linear
execution has no SM103 kernel and is rejected at planning; IQ2_XS MoE is
rejected by the SM103 MoE backend. Intermediate-Hadamard Trellis extents that
cross the FC1 halves execute only on SM103; SM12x rejects them.
