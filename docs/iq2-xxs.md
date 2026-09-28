# IQ2_XXS packed A16 weights

IQ2_XXS uses the existing IQ2_XS dense GEMM, CuTe GEMV, and fused MoE
engines on SM120/SM121. Codec-dependent loads, metadata transfers and lookup
table sizes specialize at compilation through `cutlass.const_expr`. Codec
identity participates in compilation and table-cache identity; live row
counts remain launch arguments.

Dense weights use `blockscaled.pack_weight(blocks, recipe="iq2_xxs")`, where
`blocks` is CUDA uint8 `[N,K/256,66]`, K is divisible by 256, and N by 8.
The result is `BlockQuantLinearWeight`. Use the same `PreparationSession`
and `blockscaled.mm` lifecycle as [IQ2_XS dense GEMM](iq2-xs-dense.md).

Fused MoE uses `PackedSource(format="iq2_xxs")` and
`BlockQuantWeights(w13, w2, codec="iq2_xxs")`, with raw uint8
`[E,N,K/256,66]` tensors. BF16 activations, SiLU/ReLU², projection order,
TP alignment, mapped experts, and direct-route restrictions follow the
[IQ2_XS MoE contract](iq2-xs.md). Standalone FC2 is unsupported.
`IQ2XSWeights` and `IQ2XSLinearWeight` remain compatibility aliases with
IQ2_XS as their default codec.

Each block contains a two-byte FP16 base and eight K32 records of eight bytes.
A record stores four eight-bit magnitude-grid indices, four seven-bit sign
codes, and one four-bit subscale. Reconstruction uses
`signed_magnitude * ((FP32(base) * (subscale + 0.5)) * 0.25)`, rounded once
to BF16 before compute. Nonfinite bases are rejected during packing.

Preparation losslessly applies the existing 64-byte IQ2 payload swizzle and
stores a separate FP16 base plane. There is no separate subscale allocation.
Dense bases are K256-major `[K/256,N,2]` bytes without N-tail padding; MoE
bases use the existing row-pair layout. Owned weight planes total exactly
66 bytes per 256 weights. The fragment loader extracts descriptor pairs and
subscales in registers, then calls the common BF16 conversion helpers.

XXS has its own 256-entry magnitude grid. The shared device table cache owns
a 4 KiB magnitude table for dense or a 2 KiB selector table for MoE, prepared
before capture. These fixed tables and execution scratch are accounted
separately from packed weight bytes. Pipeline staging, M8 execution, split-K,
and ordered route summation share the XS implementations, with codec-specific
shared-memory accounting.

The existing checkpoint tools recognize `quant_algo="IQ2_XXS"`,
`group_size=256`, `block_payload_bytes=66`, and `packing="ggml"`.
`benchmark_dense_gemm.py --checkpoint-recipe iq2_xxs` selects dense weights;
the canonical MoE benchmark provides `puzzle3-iq2-xxs` and
`qwen36-35b-iq2-xxs` model profiles. Checkpoint readers validate that every
selected expert projection has the same codec. There is no GGUF loader.

The shared tests cover independent raw-block decoding, exact BF16 conversion,
byte-preserving retile, dense/MoE execution, capacity reuse, and graph replay.
Performance choices inherited from XS are starting configurations, not XXS
benchmark results. [Q8_0](q8-0.md) uses the same engines with a scalar INT8 decoder.

MoE preparation races direct/packed routing, route-block M8/16/32/48/64,
FC1 and FC2 K/N tile pairs, and two through five pipeline stages. The fused
phases may choose different tiles with matching thread counts. Geometry and
shared-memory limits filter candidates, including Q8_0's larger staging
footprint. These are compile-time choices; live rows remain runtime arguments.
The config/candidate versions invalidate selections from the routing-only search.
Shared-memory sizing includes the strided FP32 reduction scratch as well as the
pipeline and codec table. Cooperative launches request the full shared-memory
carveout used by occupancy planning, including for compact Q8_0 stages.

The following races the production preparation path on checkpoint weights:

```sh
.venv/bin/python -m benchmarks.benchmark_iq2_xs_moe --tune \
  --snapshot PATH --device cuda:0 --layers 1 --tp-sizes 1 \
  --counts 1 8 16 256 --output /tmp/super3-moe.jsonl
```

Each candidate passes the independent oracle before timing; selected plans then undergo frozen
resolution, poisoned scratch, input/route mutation and allocation-free graph
replay checks. The reader supports Super3's nested language-model configuration.
