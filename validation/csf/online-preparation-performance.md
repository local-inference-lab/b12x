# Parallel preparation of lossless FP4 scale data

## Result and compatibility

**Implemented:** NVFP4 preparation builds the existing exception-word index on
CUDA instead of copying the scale planes to NumPy and iterating over experts
on the CPU. MXFP4 rows containing at most eight scales use groups of 32 rows
per CUDA block for palette selection and packing. Both paths preserve the
encoded bytes, deterministic palette tie breaking, and reconstruction format.
Checkpoint schemas, FP4 weight values, expert activation precision, execution
kernels, and the `scale_compression="csf"` API are unchanged.

The NVFP4 builder computes exception-word masks, prefix offsets, complete
replacement words, and byte patches on the source device. One scalar host
synchronization determines the compact payload allocation. Each expert and
128-row partition can progress independently. Buffer offsets use 64-bit
arithmetic; a test places the final exception beyond 2 GiB.

For short MXFP4 rows, an optimal two-byte interval starts at an observed byte
or one below it. Comparing those candidates avoids a 256-bin histogram for
each six-byte row in Kimi TP16 down projections. The lowest base still wins
ties. The general encoder remains in use for wider MXFP4 rows and NVFP4.

**Qualified:** the byte and component contracts in the correctness table.
**Research-only:** all timings below. The hosts report NVML event mask `0x400`;
SM clocks are not locked. **Unsupported:** encoding during CUDA graph replay,
overlapping model executions sharing reconstruction scratch, and the precision
or parallel-layout combinations excluded by [the API guide](../../docs/csf-scale-compression.md).

## Real-checkpoint measurements

Each component sample measures both expert projections of layer 3 on one TP
rank, starting with source scale bytes resident on its GPU. Eight cycles use
baseline/candidate/candidate/baseline order after compilation. Ratios divide
baseline time by candidate time. These measurements exclude checkpoint I/O,
FP4 weight preparation, and whole-model warmup.

| Source and condition | Baseline | GPU preparation | Speedup |
|---|---:|---:|---:|
| Kimi-K3, TP16 rank 0, 896 experts, H=3584, N=192; scale encoding | 4.000 ms | 0.823 ms | 4.86x |
| GLM-5.3-Flash NVFP4, TP2 rank 0, 288 experts, H=4096, N=1024; encoding plus exception-word index | 134.921 ms | 2.969 ms | 45.45x |

A separate GLM timing split measured encoding at 2.61 ms and the CPU index at
132.69 ms. The GPU index took 0.47 ms. Removing the CPU index accounts for the
large improvement; the general NVFP4 palette encoder is unchanged.

Kimi used Frank2 GPU `GPU-2faf5385-78f7-dab0-5528-dfaca9cc8eb8`, RTX PRO 6000
Blackwell, 600 W, memory clock 13,365 MHz. GLM component measurements used
Frank1 GPU `GPU-cd323562-fdc3-78c3-012e-86e281433050`, the same GPU model,
600 W, memory clock 16,365 MHz. The two model rows are independent comparisons,
not a cross-host performance comparison. Hardware snapshots and individual
samples are in [the evidence JSON](online-preparation-performance-evidence.json).

### Additional concurrency

Each TP rank already compresses its own shard on its own GPU. Grouped rows
and CUDA exception indexing add parallelism within each rank without
retaining additional layers in VRAM. Encoding Kimi's two projections on two
Python threads and two CUDA streams measured 0.944 ms, versus 0.877 ms for
serial dispatch in the paired experiment. That extra concurrency was not
adopted. The compact-allocation host synchronization and dispatch costs limit
its usefulness for these short operations.

## Full native GLM loading

**Qualified for startup and eight generation requests per launch:** native
NVFP4 safetensors, TP2, no speculation, FP8 KV fixed at 3 GiB per GPU,
max context 8192, sequence capacity 32, batch-token limit 1024, aligned
recurrent state. GPUs 12/13 on Frank1 used the same source/image configuration.
All runs loaded 85.71 GiB per rank.

| Measurement per rank | CPU index, first control | CPU index, repeated control | GPU index |
|---|---:|---:|---:|
| Sum of preparation for 42 MoE layers, including first compilation | 11.62–11.70 s | 8.74–8.82 s | 2.38–2.40 s |
| Median preparation of layers after the first | 178–184 ms | 143–148 ms | 9 ms |
| Model-loading log duration | 86.06–86.64 s | 34.65–34.81 s | 26.92–27.35 s |
| Completed generation requests | 8/8 | 8/8 | 8/8 |

Disk caches and compilation history affect the loading durations. The
86-to-27-second difference must not be attributed entirely to the encoder.
The repeated control and the per-layer measurements separate that limitation
from the measured preparation improvement.

The complete model is not a bitwise-logit oracle: even the two CPU-index
control launches produced different responses/logprobs for the eight seeded
requests. Exactness is established at the encoded-scale and expert-component
boundaries below. Full-model KLD and full-vocabulary logit parity are unmeasured.
These startup measurements do not replace the decode-throughput results in
[the serving report](online-scales.md#native-versus-stored-csf-serving).

## Full native Kimi loading

**Qualified for startup, eight generation requests, and a 10,401-token
prefill:** native MXFP4 safetensors, TP16/DCP16, no speculation, FP8 KV fixed
at 3 GiB per GPU, max context 32,768, sequence capacity 8, batch-token limit
4096, and selected KDA projections in MXFP8. Expert activations remain BF16;
dense-MLA partial accumulators remain FP32. The run used Frank2 GPUs
0–13, 15, and 16 at 600 W with a +6000 memory-clock offset.

| Measurement | Result |
|---|---:|
| Model memory per rank | 86.23 GiB |
| MoE layers compressed per rank | 92 |
| Sum of online preparation per rank, including first compilation | 4.35–5.34 s |
| Median preparation of layers after the first | 6–8 ms |
| Model-loading log duration | 1117.61–1117.79 s |
| Short generation requests with finite token logprobs | 8/8 |
| Long-input request | 10,401 input tokens, 64 generated tokens |

The native checkpoint contains approximately 1.45 TiB of files. Loading and
whole-model kernel warmup dominate startup; online scale preparation accounts
for only a few seconds per rank. The requests verify execution after loading,
not model-quality parity: six short requests reached the 128-token output cap.
Full-model KLD, full-vocabulary logit parity, concurrency-8 saturation, and
1M-token context are unmeasured by this validation. Source checkpoints are
read-only and no converted checkpoint is produced.

## Correctness evidence

| Conditions and measurement | Result | Conclusion |
|---|---|---|
| NVFP4/MXFP4 codecs, inline index, and MoE component suites | 166 GPU cases passed | Byte reconstruction, native expert outputs, graph replay, source ownership, and supported geometry contracts pass |
| NVFP4 empty, sparse, and dense exceptions over four shapes | 12 complete prepared buffers exactly equal the retained CPU implementation | Header, fixed stream, metadata, payload, and padding remain byte-identical |
| Real GLM layer 3, all 288 TP2 rank-0 experts, both projections | Complete prepared buffers exactly equal CPU output | The index comparison includes actual checkpoint distributions |
| Real Kimi layer 3, all 896 TP16 rank-0 experts | All source bytes reconstruct exactly; encoded planes equal the baseline encoder | Grouped rows preserve values and encoding, including rotated exception partitions |
| Final exception stored beyond 2 GiB | Exact payload and tile metadata | Signed 32-bit byte-offset overflow is covered |
| Dense/empty/tail index readers under compute-sanitizer memcheck | 24 cases passed, zero errors | No reported invalid memory access in the tested cases |
| vLLM compressed-tensors scale placement and online ownership | 6 focused GPU cases passed; pre-commit passed | Online staging uses CPU scales and GPU weights; default placement remains unchanged |
| Kimi FP32 dense-MLA partials on the serving source composition | 4 GPU cases passed | The encoder build preserves the attention precision selected by the integration |
| Kimi configuration reduced to five layers and 16 experts, dummy weights, TP1 | API health passed after warmup and graph capture | Runtime API compatibility only; synthetic weights do not qualify model quality |

The compressed-tensors staging extension covers that loader directly. Kimi's
model configuration normalizes its native compressed-tensors metadata to the
existing `mxfp4` handler, which already stages scales for online compression.
Kimi launches therefore use `--quantization mxfp4`, not an explicit
`--quantization compressed-tensors` override.

Run the component suite in a CUDA-enabled B12X environment:

```bash
python -m pytest tests/quantization/test_nvfp4_csf_inline.py \
  tests/quantization/test_x4t_scales.py tests/quantization/test_nvfp4_csf.py \
  tests/moe/test_nvfp4_csf.py tests/moe/test_mxfp4_csf.py -q
```

With the corresponding native snapshot in the default HF cache, the real-scale
probe accepts `--model kimi` or `--model glm`:

```bash
python validation/csf/online-evidence/measure_scale_bytes.py \
  --model glm --output /results/glm-scale-preparation.json
```

## Identities and raw receipts

B12X baseline: `c41b25529b2c4153f58aeab81d91bb662f1a3e39`.
B12X preparation implementation: `654f7a6205b0a7d907113094a473d5d8cfc3ba38`.
vLLM staging extension: `d92d1fa6efe8aa77d66da49e5c7d182a5017bd39`
on online feature revision `324917a26d0e43ad319d0e5d68aae2f8a62e5131`.
Serving uses the frozen compatible beta overlay recorded by each source
manifest, rather than equating that overlay with a bare repository revision.

Kimi serving uses B12X composition
`0125f956a8a0fe84f1401be27c8dfe2b74b4773c`: the preparation implementation
above plus `d887c66fb2fd7d0f5ea7e8ee91eb3382c604e22f` from merged
[B12X #441](https://github.com/local-inference-lab/b12x/pull/441).
That attention change provides the FP32 partial-accumulator API required by
the vLLM overlay. It is a runtime dependency, not an encoder change; the
Kimi validation retains FP32 attention partials and BF16 expert activations.

Local image: `sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`,
CUDA 13.4 / CUTLASS DSL 4.7.1. Frank2's imported image has engine identifier
`sha256:a853f9275c13eb871747d30d62f44bd12930d0f1f892198460d94527d0763104`;
Config, RootFS layers, architecture, and OS compare exactly.

| Native source | HF snapshot |
|---|---|
| `moonshotai/Kimi-K3` | `2496450e92e425c886db095102a52a6682ca3970` |
| `local-inference-lab/GLM-5.3-Flash-NVFP4` | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` |

Raw commands, source manifests, samples, tests, and serving logs are retained
under `/data/trellis-quant/csf-online-performance-20261002` on Frank1. Frank2
runtime and source probes are under `/data/ssd2/csf-online-performance-20261002`.
The evidence JSON records content hashes for the verification receipts.
