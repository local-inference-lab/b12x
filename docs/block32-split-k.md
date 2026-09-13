# FP32 partial reduction for block-32 decode projections

Status: implemented; private/public-plan GPU correctness and latency are
qualified. Serving qualification is required before a release claim.

The block-FP8 linear component can split the input-feature reduction across
two or four independent CTAs per output tile. Each CTA writes FP32 partials;
one reduction converts the sum to BF16. The activation quantizer retains its
block-32 amax floor of 1e-4 and the packed weights retain their original scale
layout. No BF16 atomic accumulation is used by these configurations.

The typed backends are `mxfp8_split2_fp32` and `mxfp8_split4_fp32`. They require
BF16 output, block-32 weights, planned capacity 2–8, a 16-row MMA tile, and an
input-feature count divisible by 256 times the split count. The component owns
selection and records it in `policy_resolution`; integration code supplies
model geometry and capacity only.

The four-partial reducer requires caller-owned, contiguous, 16-byte-aligned
BF16 output on the input GPU. It does not enter the allocating functional
GEMM interface, whose reduction contract remains two partials.

On SM120 devices with 188 SMs, the heuristic selects four FP32 partials and a
16×64 tile for output/input widths 1,792/5,120 and 1,152/5,120. These shapes
have too few output tiles to occupy the GPU without splitting the reduction.
Other devices and shapes retain their existing selection. Profile and caller
overrides retain precedence over the heuristic.

`plan.scratch_specs()` includes aligned caller-owned partial storage.
`plan.bind()` only creates views and does not allocate or initialize storage.
The live row count changes launch grids and masks, not compilation keys.
Four-partial storage at capacity eight costs 224 KiB for width 1,792 and
144 KiB for width 1,152. Two-partial storage uses half those amounts.

`tests/gemm/test_fp32_four_partials.py` checks live rows 1, capacity−1, and
capacity under frozen kernel resolution. It verifies the quantized FP64
oracle, poisoned partial overwrite, untouched unused capacity, fixed tensor
addresses, and allocation-free graph replay. The public-plan counterpart is
`tests/gemm/test_block_fp8_splitk.py`.

Component configuration schema 4 and generator candidate contract 3 cover
both FP32 backends. Embedded profiles retain every existing selection; only
the component schema identity changes. Offline generation races supported
public configurations and does not interpolate eligibility between row counts.
