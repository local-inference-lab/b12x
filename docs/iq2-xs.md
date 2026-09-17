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
| Descriptors | uint16 `[E, K/16, N/16, 8, 2, 2]` |
| Bases | FP16 `[E, K/256, N/16, 8, 2]` |
| Paired subscales | uint8 `[E, K/256, N/16, 8, 8, 2]` |

The row-pair axes join output channels r and r+8 within each N16 tile.
Descriptors are exposed as int32 words. The two metadata planes share one
allocation. Their combined payload stays at 74 bytes per 256 weights
(2.3125 bits per weight); there is no resident expanded weight or scale
matrix. Preparation copies at most 256 output rows of one expert per chunk.
A process-lifetime 8 KiB lookup table stores exact BF16 magnitude pairs on
each device and is initialized before binding or replay.

The W4A16 kernel asynchronously stages compact descriptors, FP16 bases and
paired subscales in shared memory. It copies the magnitude table once per CTA
and loads descriptors and metadata at fragment use. Descriptor sign bits are
applied to packed BF16 pairs after rounding, preserving exact signed results.
Shared word loads preserve
pipeline ordering. IQ2_XS rolls the pipeline stage loop to bound decoder code
size; microbatch execution uses split-K scheduling with up to two resident
256-thread CTAs per SM. Microbatch CTAs publish split-K partials in parallel;
one reducer consumes them in a fixed order. Their scratch reserves one M8
partial per possible resident CTA before binding or capture (at least
2.9375 MiB per projection on the 188-SM qualification device).
Codec-specific packing, staging, and register decoding
are separate from routing, BF16 MMA, activation, reduction and workspace planning.
Additional IQ codecs require their own exact layout and decoding contracts.
IQ2_XS uses `W4A16FusedMoeKernel`, including its M8 specialization. The separate
`MoEMicroKernel` and FP8/FP4 activation paths do not support IQ2_XS.

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
The harness collects garbage before capture and defers collection until capture
ends so deferred CUDA library destructors cannot invalidate the capture.
Its single-sample qualification timings are not autobench evidence.

## Qualification evidence

On an RTX PRO 6000 Blackwell Max-Q, CUDA 13.0 / Torch 2.12.0 and CUTLASS DSL
4.6.2, code revision `57e971b5` and checkpoint revision `b5a12f1d999b5d8e1850ecd2f5f4c5b0ce2d16a2`
passed 540 cases: layers 0/20/39, TP1 and both TP2 ranks, all 256 experts
resident, top-k=8, live counts 1/2/4/8/16/32/128/512, and balanced, hot and
imbalanced routes. Packed execution also covered deterministic output.
Minimum cosine was 0.99998307 eagerly and 0.99997914 after changed-input
replay; maximum relative L2 was 0.00582191 eagerly and 0.00649461 after replay.
Every case passed zero-contribution,
poison, address-stability and replay-allocation checks.

Code revision `57e971b5` also qualified 99 checkpoint cases using layer 0,
TP1 and both TP2 ranks, live counts
1/3/8/16 and reordered local expert IDs
`[255, 0, 17, 5, 127, 7, 253, 64, 128, 1]`. They passed the same checks,
including nonlocal routes and replay after the expert map became all nonlocal.
Minimum eager/replay cosine was 0.99997830/0.99998736; maximum relative L2
was 0.00658902/0.00575520.

The independent codebook fixture comes from llama.cpp `b2899`; all 65,536
descriptors, 16 subscale values and nine representative finite FP16 bases
passed both shared and global lookup variants of the production decoder with
bit-exact BF16 results on the GPU. Ten prepared-plan tests cover H/I geometries
256/256 and 2048/512, SiLU/ReLU², reordered local experts, nonlocal routes and
multiple live counts within one capacity.

The canonical microbenchmark on physical GPU 11
(`GPU-c7dc46e0-30bb-08e8-2ebb-f164ec57ce31`) used layer 0, TP1,
warmup=10, iterations=20, repeats=5, CUDA graphs, a 256 MiB L2 flush and
fast math. Code revision `57e971b5` measured the IQ2_XS checkpoint. The comparison at
code revision `57e971b5` measured `nvidia/Qwen3.6-35B-A3B-NVFP4` at snapshot
`491c2f1ea524c639598bf8fa787a93fed5a6fbce`, using BF16 activations through
W4A16 and identical geometry and benchmark settings.

| Live tokens | IQ2_XS graph µs | NVFP4 W4A16 graph µs |
| --- | ---: | ---: |
| 1 | 30.7 | 20.5 |
| 2 | 34.8 | 26.6 |
| 4 | 51.2 | 43.0 |
| 8 | 75.8 | 77.8 |
| Geometric mean | 45.1 | 36.8 |

Lower is better: IQ2_XS latency is 1.23 times NVFP4 W4A16 latency. All four
oracle checks passed for each checkpoint. Use
`--model-profile qwen36-35b-nvfp4 --quant-mode w4a16` for the NVFP4 comparison.
Resident IQ2_XS payload is 222 MiB per TP1 layer, or 8.671875 GiB across all
40 layers. These are routed-expert timings; they do not measure model quality
or whole-model serving throughput. Clocks are automatic and power settings
are unchanged.

The canonical IQ2_XS microbenchmark variants use 116–123 registers per thread
without stack or local memory. At code revision `172c1fd2`, the mapped direct-route
H=256/I=256 case passed Compute Sanitizer with zero errors, including poisoned
scratch and graph replay.
The decoder and prepared-execution GPU suites pass all 28 cases; shared
FP4/E8M0/direct/mapped W4A16 regressions pass 23 cases, and four mixed K3/K4
execution/capture cases pass. The expanded planning, packing and execution gate
passes 163 cases. The required
reference/sparse-routing/scratch guardrails retain the same 44 failing test
identities as base revision `a83336581`: stale API calls and one FlashInfer
backend without SM120 cubins. No additional failures were introduced.

The FP4-activation NVFP4 backend is unqualified for this checkpoint matrix:
M=8 cosine was 0.999797 against its required 0.9999. It is not the W4A16
performance comparison.
