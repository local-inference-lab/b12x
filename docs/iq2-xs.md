# IQ2_XS routed experts

Status: implemented and numerically qualified on SM120 for the cases below.
SM121 is unqualified. Other IQ encodings and standalone IQ2_XS FC2 are unsupported.

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
| Descriptors | uint16 `[E, K/16, N/16, 16, 2]` |
| Bases | FP16 `[E, K/256, N/16, 16]` |
| Paired subscales | uint8 `[E, K/256, N/16, 8, 16]` |

Descriptors are exposed as int32 words. The two metadata planes share one
allocation. Their combined payload stays at 74 bytes per 256 weights
(2.3125 bits per weight); there is no resident expanded weight or scale
matrix. Preparation copies at most 256 output rows of one expert per chunk.
A process-lifetime 512 KiB descriptor lookup table is shared on each device
and initialized before binding or replay.

The W4A16 kernel stages compact descriptors and expands metadata only in
shared memory. Codec-specific packing, staging, and register decoding are
separate from routing, BF16 MMA, activation, reduction and workspace planning.
Additional IQ codecs require their own exact layout and decoding contracts.

Packed routing supports planned capacities with runtime live token counts.
Direct tensor-core routing with fused top-k accumulation is also available
for nondeterministic SiLU capacities up to eight, without input router-weight
application or activation-amax collection. Preparation races eligible routes;
the heuristic uses packed routing. All reachable launch variants are prepared
before capture. Global descriptor and metadata offsets use 64-bit arithmetic.

## Checkpoint and benchmark

`benchmarks/iq2_xs_checkpoint.py` reads routed expert weights from the local
`nvidia/Qwen3.6-35B-A3B-IQ2_XS-NVFP4` safetensors snapshot, validates its
IQ2_XS metadata and tensor shapes, applies aligned TP slices, and preserves
explicit local expert ordering. The model has H=2048, I=512, 256 routed
experts and top-k=8. Its attention/shared-expert NVFP4 tensors remain outside
this loader.

The canonical benchmark uses the same prepared public API and timing path as
other MoE formats:

```bash
python benchmarks/benchmark_moe.py \
  --device 11 --model-profile qwen36-35b-iq2-xs \
  --batch-size-profile micro --scale-contract per-expert \
  --validate oracle --quant-mode w4a16
```

Select the assigned device; `--model-path` can pin a specific local snapshot,
and `--tp-size 2 --tp-rank 1` selects the second block-aligned TP2 slice.
Oracle decoding is outside timing. CPU source blocks may be retained for
validation; only compact prepared weights remain on the GPU.

`python -m benchmarks.benchmark_iq2_xs_moe --help` describes the broader
checkpoint qualification harness. It records source/weight hashes and raw
graph samples, checks cosine >= 0.999 and relative L2 <= 0.01, mutates inputs
and routes after capture, poisons scratch/output, checks fixed addresses and
replay allocation counts, and freezes compilation and kernel resolution.
Its single-sample qualification timings are not autobench evidence.

## Qualification evidence

On an RTX PRO 6000 Blackwell Max-Q, CUDA 13.0 / Torch 2.12.0 and CUTLASS DSL
4.6.2, checkpoint revision `b5a12f1d999b5d8e1850ecd2f5f4c5b0ce2d16a2`
passed 540 cases: layers 0/20/39, TP1 and both TP2 ranks, all 256 experts
resident, top-k=8, live counts 1/2/4/8/16/32/128/512, and balanced, hot and
imbalanced routes. Packed execution also covered deterministic output.
Minimum cosine was 0.9999838; maximum relative L2 was 0.005711 eagerly and
0.006236 after changed-input replay. Every case passed zero-contribution,
poison, address-stability and replay-allocation checks.

The independent codebook fixture comes from llama.cpp `b2899`; all 65,536
descriptors, 16 subscale values and nine representative finite FP16 bases
passed the production decoder with bit-exact BF16 results on the GPU.
Separate prepared-plan tests cover SiLU/ReLU², reordered local experts,
nonlocal routes and multiple live counts within one capacity.
