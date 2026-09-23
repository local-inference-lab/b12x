# GB300 physical SM103 qualification

Status: **implemented operator suite qualified on one physical GB300**, with
the fixes below. This is the first physical SM103 execution of the branch; all
earlier SM103 evidence was offline compilation or SM120 execution. It is
operator qualification only: complete-model serving, performance, Station RDMA
and the vLLM plugin remain outside its scope, as the launcher receipt records.

## Sources and environment

The base is `work/sm103-hybrid-continuation` at
`2e3115ae56153336ba1b1a7e2f23b8b91f7acd99`. The fixes are commits `780532c8`
through `552eb925` on `work/gb300-qualification`. Every physical receipt
executes a frozen archive with package hash
`977eadadf855f676d61bc81eaa2e63a0d655a0719df9029f0799e18a990f3a7b` and one test
hash, both reproduced by the committed tree. The synccheck and racecheck
receipts use the committed launcher; the operator and memcheck receipts used
its predecessor, which lacked only the recorded kernel exclusion. The compile
manifest is shared.

The host `gracie` is aarch64 Ubuntu 24.04 with Grace memory and two GPUs. The
qualified device is an NVIDIA GB300, UUID
`GPU-c146511a-0326-7ddc-4346-998d61a64b34`, compute capability 10.3, 152 SMs,
267,898,585,088 bytes, ATS addressing and verified Grace coherency (host-native
atomics, pageable access through host page tables). The second GPU, an SM120
RTX PRO 6000 Blackwell Max-Q (`GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7`), is
used only as a control to classify pre-existing portable-test failures.

Driver 595.91.07, CUDA toolkit 13.2 and Compute Sanitizer 2026.1.0. Python
3.12.14 (uv-managed, with headers), Torch 2.13.0+cu130, cuda-python 13.4.1,
CUTLASS DSL 4.6.2, Triton 3.7.1. The offline compile target describes a 148-SM
B300; this part has 152 SMs. Raw evidence is outside the repository in
`/home/jasonc/b12x-sm103-qualification-20260923` on gracie.

## Results

| Stage | Result |
| --- | --- |
| Native cross-compilation, `compile_sm103.py --component all` | 1,175 callables, `cross-compiled` |
| Prepared declarations, `compile_sm103_prepared.py` | 88 cases, 240 CuTe programs, 0 failures |
| Production residency geometry (H5120/I2304/E384/295 hot/top-6) | 12 callables, `cross-compiled` |
| Host tests, `tests/architecture tests/preparation` | 962 passed, 43 skipped from the checkout |
| Physical operator suite, `qualify_sm103.py --execute` | `operator-qualification-passed`: 32/32 components, 1,249 tests, 0 failures/errors/skips |
| Memcheck over every component | Passed: 32/32 components, 1,243 tests, 0 errors in every summary |
| Synccheck over every component, cuBLAS `nvjet` kernels uninstrumented | Passed: 32/32 components, 1,243 tests, 0 errors in every summary |
| Racecheck, residency and blockscaled components | Passed: 113 tests, 0 hazards |
| Residency production-geometry benchmark | Bitwise all-HBM parity, mutation replay and no allocator events at 1–128 tokens |

The formal host run from the archive has one additional failure,
`test_checkout_identity_uses_explicit_source_root`, which requires a `.git`
checkout; it passes from the worktree. The first physical run of the unchanged
branch failed 23 of 32 components: 1,251 tests with 155 failures, 10 errors
and 52 skips.

## Defects found on hardware

**Residency FP8×FP4 GEMM operand layout.** tcgen05 `kind::mxf8f6f4` reads E2M1
with byte addressing: each group of sixteen codes occupies its own sixteen-byte
shared-memory slot, eight code bytes then eight ignored bytes. The `w4a8_mx`
routed GEMM staged packed checkpoint rows through the packed SW64 layout. A
one-hot probe showed K indices 16–31 of every MMA reading codes 32–47, and the
last instruction of each K tile reading past the stage. FC1 errors reached 80%
of the output magnitude in every placement, and the first run was
nondeterministic. TMA now stages packed rows unchanged and the MMA warp copies
each group into the byte-addressed SW128 operand. Checkpoint storage and tier
footprints are unchanged. All HBM, Grace and mixed placements, quiescent slot
exchange and native cache control now pass exact parity and the independent
oracle.

**MXFP4 dense GEMM with K=128.** A 256-wide FP4 K tile over a 128-element
reduction leaves MXFP4 one scale-factor K atom, smaller than its TMA box, and
the kernel faults with an illegal instruction. The fault poisoned the context
for the rest of the `blockscaled` suite. The K tile is now clamped to K. Partial
final tiles (K=384, 1152) and NVFP4/MXFP8 were already correct.

**Composite MoE outputs.** Plans with warm-up capacity variants dispatch an
exact smaller variant, whose own capacity rejected an output sized for the plan
capacity. The composite state now passes that variant the live rows of the
caller's buffer. All 188 Trellis MoE, atom and mixed tests pass, including NaN
preservation beyond live rows and graph replay.

**Prepared FP6 serving on SM103.** The `quantization.mxfp6` plan behind
`b12x::fp6_dense_linear` only lowered the SM12x dense GEMM, so it could not
prepare on SM103. The plan now materializes the native SM103 activation
quantizer and FP6 GEMM, with a fixed-capacity workspace and retained programs.
Output equals the eager SM103 linear bit-for-bit across E2M3/E3M2/W6A8,
expanded and packed weights and 1/3/129 rows. The SM120 route is unchanged.

**`blockscaled.quantize_mxfp4` export.** The caller-owned MXFP4 packing added in
`ad6d59c6` was imported by `api.py` but never listed as a lazy entry point.

## Test and environment corrections

Several SM103 tests had never run on SM103 hardware and still used interfaces
removed by the preparation-session port: unprepared `moe.decode` plans,
`prepare_weights(btx_layer=...)`, `b12x.policy`, `Caps(metadata_validation=...)`,
static WO token counts, `wo.run` without `plan=`, the positional FP6 op
signature, and a two-field DSA config. Tests for SM103-admitted operators that
still gated on SM12x (delta prefill, MLA compression, Engram, dense MLA window,
mHC, MXFP4 packing and the V4.1 KV writer) now use `require_sm103_or_sm12x`.
The mHC projection test follows SM103's required high/low TF32 split.

Four failures reproduce identically on the SM120 control and predate this work:

| Test | Cause and correction |
| --- | --- |
| `test_bf16_multi_request_decode_matches_reference` | A second prepare of the same plan skips its callback; reuse the prepared scratch contract |
| `test_current_mix_reuses_capacity_and_mutable_graph_inputs[True-post_pre-17-7168]` | One of 114,688 BF16 outputs rounds one ULP (0.03125) across a boundary; BF16 outputs allow one ULP or the prior 0.016 floor, FP32 outputs keep their tolerances |
| `test_mhc_prepared_projection_splits_share_projection_...` | The projection specializes on K-split count since `35065066`; assert one shared program per count |
| `test_mhc_lagged_parallel.py` | Called an undefined `require_blackwell`, hidden by an SM12x fixture skip |

Two suite selections change. `vocabulary_projection` again excludes the
Triton-pinned graph test, renamed from `planned_*` to `prepared_*`; SM103
rejects Triton by design, and the CuTe cases cover this target.
`mtp_feedback` excludes the optional byte comparison against vLLM's native
FP8 quantization, since the launcher scope excludes vLLM installation. The
b12x FP8 path retains its independent reference tests.

Sanitizer runs deselect the new `device_trap` marker. Its two tests replay an
invalid embedding ID or WO cosine position in a child process and pass only if
the kernel traps. The first formal memcheck counted those intended traps as 390
errors, one per trapping thread, and stopped after 18 components. The plain
operator run still executes all six parametrizations.

Synccheck 2026.1 cannot launch the cuBLAS `nvjet_sm103_*` kernels behind Torch
reference matmuls: it reported 26 internal sanitizer errors ("One or more of the
parameters is invalid") in `mtp_feedback` while all 26 tests passed, and
memcheck instrumented the same launches cleanly. The launcher's new
`--sanitizer-exclude-kernel` option leaves those third-party kernels
uninstrumented and records the exclusion in the receipt; every b12x kernel
remains instrumented. The failed receipt is retained.

The first environment used system Python without `Python.h`, so every
Triton-built supporting kernel failed to compile: 35 GDN decode, 2 KDA decode,
14 MTP feedback and 8 sparse-MLA failures. In the Trellis suites it failed
before the composite-output defect above and masked it. Engram disk lookup
requires `liburing-dev` and `pkg-config` for the native loader.

## Production-geometry residency

`benchmarks/moe/expert_residency.py` uses H5120, I2304, 384 experts with 295
resident, top-6 and SwiGLU limit 10. Each row is the median of ten graph samples;
the ratio is tiered latency divided by all-HBM latency. Correctness precedes
timing and passes at every count.

| Tokens | Actual cold fraction | Tiered, μs | All-HBM, μs | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 680.4 | 678.2 | 1.0032 |
| 2 | 0 | 1,109.6 | 1,105.8 | 1.0034 |
| 4 | 0 | 2,089.8 | 2,082.5 | 1.0035 |
| 8 | 0 | 4,043.9 | 4,031.3 | 1.0031 |
| 16 | 0.0104 | 8,122.6 | 7,804.9 | 1.0407 |
| 32 | 0.0052 | 15,854.4 | 15,441.3 | 1.0268 |
| 64 | 0.0208 | 31,969.2 | 30,636.8 | 1.0435 |
| 128 | 0.0195 | 63,317.3 | 61,116.7 | 1.0360 |

Grace-served experts cost at most 4.4% here. Both placements scale linearly at
about 113 μs per route. This is a correctness baseline, not a serving-speed
result. The benchmark's isolated stage graphs are empty (Torch warns during
capture), so its per-stage medians are not interpretable.

The byte-wise E2M1 expansion dominates that time. A timing-only control that
removes the copy loop (and therefore computes wrong values) measures the
all-HBM graph at 131.7 μs for one token and 630.9 μs for eight, against 681.0
and 4,037.8 μs with the expansion. The routed GEMM also launches one 128-row M
tile per route and does not overlap TMA with MMA.

## Remaining limits

The E2M1 expansion is the first performance target: it copies one byte at a
time on the MMA warp while three warps idle. Eight-byte copies across all four
warps preserve the operand layout; any change requires rerunning these gates.
Complete-model SM103 serving still needs the native MXFP4/MXFP8 checkpoint
adapter. Station RDMA, the vLLM plugin and B300 performance are not
qualified. The optional vLLM FP8 comparison remains unexecuted. Results are
from one GB300; the 148-SM offline target and this 152-SM part share programs.

## Reproduction

Use the header-equipped environment above, with the GB300 selected by UUID and
no other compute process on it.

```bash
python -m pytest tests/architecture tests/preparation -q
python scripts/compile_sm103.py --component all --output-dir "$OUT/native" \
  --nvdisasm "$NVDISASM" --cuobjdump "$CUOBJDUMP"
python scripts/qualify_sm103.py --execute --device-uuid "$GB300_UUID" \
  --compile-manifest "$OUT/native/manifest.json" --output-dir "$OUT/runtime"
python scripts/qualify_sm103.py --execute --device-uuid "$GB300_UUID" \
  --compile-manifest "$OUT/native/manifest.json" --sanitizer "$COMPUTE_SANITIZER" \
  --sanitizer-tool memcheck --output-dir "$OUT/memcheck"
python scripts/qualify_sm103.py --execute --device-uuid "$GB300_UUID" \
  --compile-manifest "$OUT/native/manifest.json" --sanitizer "$COMPUTE_SANITIZER" \
  --sanitizer-tool synccheck --sanitizer-exclude-kernel nvjet --output-dir "$OUT/synccheck"
python scripts/qualify_sm103.py --execute --device-uuid "$GB300_UUID" --component residency \
  --compile-manifest "$OUT/native/manifest.json" --sanitizer "$COMPUTE_SANITIZER" \
  --sanitizer-tool racecheck --sanitizer-exclude-kernel nvjet --output-dir "$OUT/racecheck"
python benchmarks/moe/expert_residency.py --output "$OUT/residency.json" \
  --hidden 5120 --intermediate 2304 --experts 384 --hot-experts 295 \
  --top-k 6 --swiglu-limit 10 --tokens 1 2 4 8 16 32 64 128 --cold-fraction 0.0155
```
