# MXFP8 (W8A8) MoE experts

Native MXFP8 routed-expert serving for SM120/SM121: E4M3 weight codes with
UE8M0 group-of-32 block scales, against activations quantized at runtime to
E4M3 with UE8M0 group-of-32 block scales. Both operands are real FP8, so the
contraction runs on the block-scaled `mxf8f6f4` `m16n8k32` MMA emitted by
`cute.nvgpu.warp.MmaMXF8Op` — the same instruction family the MX-FP6 path
drives, but with no 3:4-packed operand and no in-shared-memory expansion.

This is the MoE counterpart of the dense MXFP8 paths
(`gemm/mxfp8_linear`, `gemm/blockscaled` MXFP8 packing). The FP6 recipe is
documented in `docs/mxfp6-w6a8.md`.

## Format

- **Weights on disk**: MXFP8 E4M3 codes, `uint8` or `float8_e4m3fn`, one code
  per element: `[E, 2*I, K]` for the gated FC1 and `[E, K, I]` for FC2. The
  bytes remain E4M3 throughout preparation and execution. Gate-first FC1
  projections are reordered without requantization.
- **Block scales on disk**: UE8M0, `uint8` or `float8_e8m0fnu`,
  `[E, rows, K/32]` (FC1) and `[E, K, I/32]` (FC2), **unswizzled**. Preparation
  validates them and swizzles them into the MMA layout with
  `b12x._lib.intrinsics.swizzle_block_scale` — the one canonical scale
  swizzle in the tree, shared with the MX-FP6 recipe.
- **Invalid scales are rejected, never clamped**: `0xFF` is UE8M0 NaN. Any
  block scale byte equal to `0xFF` raises during preparation; the finite range
  `0x00..0xFE` is preserved bit-for-bit.
- **Per-expert globals**: the checkpoint's `*_weight_scale_2` combine with the
  reciprocal activation global scales into runtime alphas,
  `alpha = weight_scale_2 / a_gscale`, exactly as the other MX recipes do. A
  non-finite alpha (a zero `a_gscale`) is a preparation error.
- **Activations**: quantized inside the fused kernel at both GEMM boundaries —
  the routed input row before FC1 and the SwiGLU intermediate before FC2 — to
  E4M3 with a UE8M0 K/32 scale that is the exact-IEEE-bit power-of-two ceiling
  of `amax * global_scale / 448` (`fp6_block_ue8m0_exact`, not
  `ceil(log2(...))`), with the calibrated per-expert global scale folded in at
  quantize time.
- **Geometry**: `hidden_size % 128 == 0` and `intermediate_size % 32 == 0`,
  SiLU only, `sf_vec_size == 32`, and the production `(128, 128)` MMA tile.
  Hidden (the FC1 K extent) is streamed as 128-element tiles of one byte per
  element. See "Non-128 intermediate" below for the intermediate rule.

## MoE (`moe.fused_moe`, quant mode `w8a8_mx`)

`quant_mode="w8a8_mx"` with `source_format="mxfp8_e8m0_k32"`, reached through
the canonical planned lifecycle:

Call `plan_weights` with `PackedSourceFormat.MXFP8_E8M0_K32`, the checkpoint's
`W13Layout`, `ActivationMode.A8`, SiLU, BF16 I/O and `MoEGeometry`.
Pass the serialized payloads and unswizzled scales as `PackedWeights` to
`prepare_weights`. Declare `plan_execution` with `ExecutionCapacity` for the
largest serving batch and top-k. Prepare every declaration in a
`PreparationSession` before binding caller inputs, routes and output storage.
Run the binding with `fused_moe.run`. CUDA graph capture uses that same prepared
binding under frozen kernel resolution.

`prepare_weights` returns `PreparedExperts` whose representation value is
`b12x.moe._shared.kernels.w8a8.PreparedW8A8MXFP8Weights`:
`w13_values`/`w2_values` (byte-exact `[E, N, K]` uint8),
`w13_sf_swizzled`/`w2_sf_swizzled` (`[E, pad128(N), pad4(K/32)]` uint8; FC1
is `[2, E, pad128(I), pad4(K/32)]` when `I % 128 != 0`), and
`w13_alpha`/`w2_alpha` (`[E]` f32), with `w13`/`w2`/`w13_scale`/`w2_scale`/
`w13_global_scale`/`w2_global_scale` aliases for the generic owner plumbing.
Callers hand **unswizzled** scale grids; B12X swizzles them.

A gate-first (`w31`) FC1 is rotated to the kernel-native `[up; gate]` order
during preparation, while the payload and its scale grid are both still
unswizzled, so the row flip and the MMA swizzle cannot disagree about which
row is which.

Everything else is the existing unified dynamic kernel: route/pack, the
materialized persistent work queue, SwiGLU, the atomic scatter (or the
deterministic route-buffer top-k sum), and the per-expert alpha plumbing are
unchanged from `w6a8_mx`. The recipe differences are confined to operand
transport:

| | `w6a8_mx` | `w8a8_mx` |
|---|---|---|
| B gmem K extent | `3K/4` packed bytes | `K` bytes |
| B smem staging | packed TMA tile aliased into sB, expanded in place | direct TMA into the swizzled sB |
| MMA emission | inline `mxf8f6f4` asm, `e4m3 × e2m3` | `MmaMXF8Op`, `e4m3 × e4m3` |
| weight bytes vs checkpoint | 3:4-packed, source-native | identical |

Scheduling is the unified dynamic backend; `dynamic` is the only legal backend
and `(128, 128)` the only legal tile, both enforced during tuning-contract
validation rather than discovered at launch.

## Non-128 intermediate

The intermediate size needs `I % 32 == 0`, not `I % 128 == 0`. The
`mxf8f6f4` MMA is `m16n8k32` with one UE8M0 scale per 32 K
(`.scale_vec::1X`), and the activation and intermediate quantizers work in
32-element blocks. The stacked `[up; gate]` FC1 descriptor addresses the gate
tile as `slice + ceil(I/128)`, and scale factors are TMA'd in 128-row atoms,
so with one stacked descriptor a gate half starting at row `I` inside an atom
has no valid tile or atom coordinate.

For `I % 128 != 0` (and `I % 32 == 0`) the kernel uses the existing
`separate_w13_halves` specialization: independent up and gate descriptors
over `[I, K]` views of the same payload, and preparation swizzles the two
halves' scale grids independently. The last intermediate slice's tail rows
(FC1 N) and columns (FC2 K) are TMA zero-fill, so no padded weight bytes are
read. The MMAs still run over the full 128-wide tail slice.

## Shared byte-container geometry

`MoEDynamicKernelBackend` splits the FP6-specific concerns from the genuinely
shared ones so the two recipes cannot drift:

- `is_w6a8` — 3:4-packed B staging, the in-place expansion, and the inline FP6
  MMA dispatch;
- `is_w8a8` — the new recipe;
- `is_mxf8` (`is_w6a8 or is_w8a8`) — `tile_k = sf_vec_size * 4`, the
  `MmaMXF8Op` tiled MMA and its SF atom geometry, the one-byte-per-element
  activation scratch, and the FC2-intermediate container requant.

## Tests

- `tests/moe/test_mxfp8_w8a8.py` — CPU: the planner/lowering vocabulary, the
  lossless weight contract checked byte-for-byte against the canonical scale
  layout, UE8M0 `0xFF` rejection, alpha math,
  malformed-input rejection, and a check that the oracle quantizer reproduces
  the device PTX scale-byte rule at power-of-two boundaries. GPU: numerical
  execution against `moe_reference_w8a8_mx` on the model geometry
  (K=2560, top-k=10, I=640/320/96) with 1/5/16 tokens, experts with no
  routes, and CUDA-graph replay under frozen kernel resolution at I=640/320.
- `tests/moe/test_moe_execution_model.py` — the recipe-independent lowering
  invariants.

## Verified Execution

The reduced expert-count lane uses E=16, K=2560 and top-k=10 on RTX PRO 6000
Blackwell Max-Q (SM120), at I=640, the TP=2 shard I=320 and I=96. The full
native suite passes 29 tests, one process per test. Compute Sanitizer
memcheck reports no access errors on the I=320, I=96 and I=640 oracle cases
and the I=320 graph replay; its only report is a host-side "selective device
code recompilation" API warning from the Triton loader.

At I=640 the native lane covers numerical output at M=1/5/16, empty experts,
atomic graph replay against the quantized reference and deterministic
bit-exact graph replay; Compute Sanitizer memcheck reports zero errors on its
six execution/replay cases. The NVFP4 preparation/decode/replay lane passes
three tests with a 42,538,496-byte peak allocation. NVFP4 keeps its
`separate_w13_halves` selection and gate offset of `k // 2` bytes per FP4 row.

The emitted device objects record `-arch sm_120f`. Their disassembly contains
`QMMA.SF.16832.F32.E4M3.E4M3.E8`. This proves native block-scaled MXFP8
execution on SM120 and compilation for the SM120/SM121 family. It does not
prove SM121 execution.

Qwen3.8-Flash-Next-NVFP4 QAD step 5500 serves on two GB10 (SM121) TP=2 groups
with its MXFP8 MTP draft experts on this path at I=320 per rank and its NVFP4
target experts on B12X; the engine log selects `B12X_MXFP8` for the draft and
no expert runs on Marlin. MTP acceptance rate against Marlin and per-kernel
SM121 timing remain unmeasured. The native recipe is not a serving-throughput
improvement claim.

## Related

- `docs/mxfp6-w6a8.md` — the MX-FP6 recipe that shares this kernel.
- `docs/FP6_in_B12X.md` — why `mxf8f6f4` accepts mixed 8/6-bit operands and
  what each side costs.
- `docs/moe-execution-model.md` — the axis vocabulary and the one-owner
  preparation contract.
