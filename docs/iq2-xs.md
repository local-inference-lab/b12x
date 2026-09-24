# IQ2_XS routed experts

Status: implemented on SM120 and SM121. [IQ2_XXS](iq2-xxs.md) shares the
same engine with compile-time codec specialization. Standalone IQ2 FC2 is unsupported.

`moe.fused_moe` accepts `PackedSource(format="iq2_xs")` with
`IQ2XSWeights(w13, w2)`. Inputs are safetensors payload views of dtype uint8
and shape `[E, N, K/256, 74]`. There is no GGUF container loader. BF16 A16,
SiLU and ReLU² are supported; hidden and local intermediate sizes must be
positive multiples of 256. W13 denotes up/gate order, and W31 denotes
gate/up order. Preparation normalizes SiLU projections to gate/up order.

Each block holds a little-endian FP16 base, 32 uint16 descriptors, and eight
bytes of paired four-bit subscales. A descriptor selects one of 512 vectors
of eight magnitudes and one of 128 sign patterns. Reconstruction is
`signed_magnitude * (FP32(base) * (subscale + 0.5) * 0.25)`, followed by one
BF16 rounding before MMA. Activation scales do not enter A16 math.
Non-finite bases are rejected during preparation.

Preparation losslessly rearranges each matrix into:

| Plane | Logical dtype and shape |
| --- | --- |
| Descriptors | uint16 `[E, N/64, K/128, 8, 4, 8, 2, 2]` |
| Bases | FP16 `[E, K/256, N/16, 8, 2]` |
| Paired subscales | uint8 `[E, N/64, K/128, 4, 4, 8, 2]` |

Each descriptor tile contains eight K16 groups and four N16 groups. The
row-pair axes join output channels r and r+8 within each N16 group. A
K128/N64 tile occupies 2 KiB and is contiguous in global memory. Descriptors
are exposed as int32 words. Each matching subscale tile occupies 256
contiguous bytes. The two metadata planes share one allocation. Their combined payload stays at 74 bytes per 256 weights
(2.3125 bits per weight); there is no resident expanded weight or scale
matrix. Preparation copies at most 256 output rows of one expert per chunk.
A process-lifetime 4 KiB lookup table stores byte selectors for the three
exact magnitudes, 8, 25, and 43, on each device. Preparation initializes it
before binding or replay.

The W4A16 kernel asynchronously stages compact descriptors, FP16 bases and
paired subscales in shared memory. It copies the selector table once per CTA
and loads descriptors and metadata at fragment use. Each row scale evaluates
and rounds the three magnitudes once; byte permutation selects each BF16 pair.
Descriptor sign bits are
applied to packed BF16 pairs after rounding, preserving exact signed results.
Four `cp.async` stages overlap copies with decoding; M8/K128/N64 uses three
stages and stores eight activation rows. The stage handoff
follows the final descriptor and metadata reads before any warp can reuse
that stage. IQ2_XS rolls the pipeline stage loop to bound decoder code size.
Microbatch execution uses up to four resident 128-thread CTAs for K128/N64,
or two CTAs for other tiles. Direct-route projections with at least one
full grid of output tiles use whole-K waves. Smaller direct projections and
packed routes use split-K scheduling: CTAs publish partials in parallel,
and one reducer consumes them in a fixed order. Their scratch reserves one M8
partial per possible resident CTA before binding or capture.
Codec-specific packing, staging, and register decoding
are separate from routing, BF16 MMA, activation, reduction and workspace planning.
Additional IQ codecs require their own exact layout and decoding contracts.
IQ2_XS uses `W4A16FusedMoeKernel`, including its M8 specialization. The separate
`MoEMicroKernel` and FP8/FP4 activation paths do not support IQ2_XS.

Packed routing supports planned capacities with runtime live token counts.
Direct tensor-core routing with fused top-k accumulation is also available
for nondeterministic SiLU or ReLU² capacities up to eight, without input router-weight
application or activation-amax collection. Preparation races eligible routes;
the heuristic uses packed routing. All reachable launch variants are prepared
before capture. Global descriptor and metadata offsets use 64-bit arithmetic.

Direct IQ2_XS execution stores weighted FC2 route outputs in the existing FC1
buffer after the activation handoff. A grid barrier publishes those outputs
before an FP32 sum in route order and one final BF16 conversion. This avoids
order-dependent BF16 atomic accumulation across experts. Unowned mapped
experts contribute zero. Reduction stays inside the fused kernel and uses
the preplanned workspace without an additional host launch.

## Checkpoint and benchmark

`benchmarks/iq2_xs_checkpoint.py` reads routed expert weights from local
safetensors checkpoints with Qwen-style gated experts or Puzzle 3 ReLU²
experts. It validates IQ2_XS metadata and tensor shapes, applies aligned
TP slices, and preserves explicit local expert ordering. Attention and
shared-expert tensors remain outside this loader.

The canonical benchmark uses the same prepared public API and timing path as
other MoE formats:

```bash
python benchmarks/benchmark_moe.py \
  --device 0 --model-profile puzzle3-iq2-xs \
  --model-path /path/to/checkpoint \
  --batch-size-profile micro --scale-contract per-expert \
  --validate oracle --quant-mode w4a16
```

Select the assigned device and provide a local checkpoint through
`--model-path` or `B12X_MODEL_PATH`. Use `--model-profile qwen36-35b-iq2-xs`
for Qwen checkpoints. `--tp-size 2 --tp-rank 1` selects the second
block-aligned TP2 slice.
Oracle decoding is outside timing. CPU source blocks may be retained for
validation; only compact prepared weights remain on the GPU.

`python -m benchmarks.benchmark_iq2_xs_moe --help` describes the broader
checkpoint qualification harness. It records source/weight hashes and raw
graph samples, checks cosine >= 0.999 and relative L2 <= 0.01, mutates inputs
and routes after capture, poisons scratch/output, checks fixed addresses and
replay allocation counts, and freezes compilation and kernel resolution.
The harness collects garbage before capture and defers collection until capture
ends so deferred CUDA library destructors cannot invalidate the capture.
Its single-sample qualification timings are not autobench evidence.
