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
