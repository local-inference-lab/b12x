# A4 prefill over W4A16 packed weights

Status: opt-in, component-qualified on GLM-5.3-Flash TP2 geometry (288
experts, hidden 4096, intermediate 1024 per rank, top-8) on RTX PRO 6000
Max-Q. The default W4A16 behavior is unchanged.

## What it does

A W4A16 MoE keeps BF16 activations and dequantizes NVFP4 weights inline. That
is exact but expensive for large prefill steps. With
`B12X_W4A16_A4_PREFILL_MIN_TOKENS=N` set when weights are prepared and the
execution plan is declared, W4A16 bindings of at least `N` tokens run an
NVFP4-activation pipeline over the same packed W4A16 weights instead:

1. route pack into 128-route blocks by expert;
2. per-token NVFP4 quantization of the input with the shared input global
   scale (`quantize_block_fp4` semantics);
3. FC1 on the SM120 block-scaled QMMA (`kind::mxf4nvf4.scale_vec::4X`
   m16n8k64), SiLU(gate) * up, and NVFP4 requantization of the BF16-rounded
   intermediate with the expert's intermediate global scale;
4. FC2 on the same QMMA into unweighted per-route BF16 rows;
5. the FP32-weighted top-k sum.

Smaller calls (decode) keep the exact W4A16 path. No second weight copy is
made: the kernels read the W4A16 packed layout directly.

`B12X_W4A16_A4_PREFILL_TERMS=2` quantizes the input and the intermediate as
an NVFP4 value plus an NVFP4 residual (`x ~ q1 + q2`, same global scale) and
contracts both planes into the same accumulators, at twice the QMMA work. On
synthetic activations its error is a tenth of one plane's, but in GLM serving
it did not reduce needle-checksum near misses (see Measured): both planes
share the per-16 block scale, so small values next to an outlier are lost in
both. It stays an experiment.

`B12X_W4A16_A4_PREFILL_WARPS` selects the GEMM CTA: 8 warps with 64-row warp
tiles (default; 210 to 230 registers per thread, no spills) or 16 warps with
32-row tiles (128 registers per thread, which spills).

## Operand mapping

The W4A16 packed layout stores each (K16, N64) tile as 32 lane words per
column group: the word of BF16 lane `(tc_col, r)` holds nibbles
`[c1 k0, c1 k8, c2 k0, c2 k8, c1 k1, c1 k9, c2 k1, c2 k9]` of rows
`2r + {0, 1, 8, 9}`. Activations are stored with the K order inside every
16-group permuted by `pi(8h + i) = 4h + [0, 8, 1, 9, 2, 10, 3, 11][i]`, so
each QMMA B register is one `PRMT` of two packed words (bytes `{0, 2}` for
the first column of the pair, `{1, 3}` for the second). The permutation
stays inside each 16-group, so block scales are unchanged.

W4A16 scales are lifted E4M3 bytes (`s * f * 2**7` as FP16 bits 14..7, i.e.
`lifted = e4m3(s * f) + 120` for every kept scale, 0 for flushed ones). The
kernels restore `e4m3(s * f)` bytewise and fold `1 / f` back through the
packed global scale (`g * 2**119 / f`) into the per-expert alpha. Compressed
(NVFP4-CSF) experts expand their scales into the W4A16 scale scratch for this
path, like every W4A16 call above the stage-read limit.

## Contract

- Weights must be prepared with calibrated activation scales while the option
  is set (`PackedWeights.input_scale` / `intermediate_scale` as global scales,
  finite and positive). The shared input scale is the smallest global scale
  across experts (the widest calibrated range); intermediate scales stay per
  expert. Weights without valid scales stay W4A16 only.
- The path admits int32 or int64 route ids, no expert maps, no activation-max
  collection, and no router weights on the input.
- All buffers are views of caller scratch (`intermediate_cache2` holds the
  NVFP4 activations and the 128-route pack; `intermediate_cache13` the
  per-route FC2 rows). Bind admits the path only when they fit.
- One compiled pipeline per model geometry serves every live token count;
  token counts are runtime launch arguments.

## Measured

GLM-5.3-Flash NVFP4 QAD (stored FP4-CSF checkpoint) in vLLM, TP2/DCP2, MTP3,
eight request slots, two RTX PRO 6000 Max-Q (325 W), threshold 1536; vLLM
keeps the decode rows of mixed steps on W4A16. Prefill is tok/s for one 8K or
32K prompt; decode is aggregate output tok/s of two runs at 1 and 8 requests.

| Activations of calls >= 1536 tokens | Prefill 8K | Prefill 32K | Decode 1 | Decode 8 |
| --- | ---: | ---: | ---: | ---: |
| BF16 (W4A16) | 7,490 | 7,728 | 174, 168 | 529, 543 |
| NVFP4, 8-warp GEMMs (default) | 9,528 | 9,784 | 174, 163 | 523, 539 |
| NVFP4, 16-warp GEMMs | 8,965 | 9,194 | 175, 176 | 527, 538 |
| NVFP4 value + residual, 8 warps | 8,482 | 8,731 | 175, 173 | 529, 530 |

Needle checksum (500 probes at eight requests, four runs): 0.53% near misses
with BF16 activations, 1.75% with NVFP4 and 1.85% with value + residual. The
warp count does not change the per-element accumulation order.

One MoE layer of that geometry (uniform routing, microseconds; W4A16 from
separate runs of the same benchmark):

| Tokens | NVFP4, 8 warps | NVFP4, 16 warps | Value + residual, 8 warps | W4A16 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,340 | 1,545 | 1,453 | 1,435 |
| 3,072 | 1,684 | 1,930 | 2,371 | 4,114 |
| 8,192 | 3,201 | 4,494 | 5,577 | 8,735 |

## Validation

```bash
.venv/bin/python -m pytest tests/moe/test_w4a16_a4_prefill.py
```

covers the kernels against a float64 torch emulation of the quantization
contract (one and two activation planes), threshold and scale gating, and CUDA
graph replay across live token counts with the same compiled callables.
