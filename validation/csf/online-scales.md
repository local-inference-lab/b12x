# Lossless FP4 scale preparation and TP serving measurements

## Behavior and compatibility

**Implemented:** `WeightPlanConstraints(scale_compression="csf")` compresses
native MXFP4 or ModelOpt NVFP4 expert scale bytes during ordinary B12X weight
preparation. It retains the existing FP4 weight representation, activation
precision, global factors, and CSF execution kernels. The default remains
uncompressed preparation. Checkpoint formats are unchanged.

The GPU encoder selects the lowest row base maximizing coverage by a two-byte
MXFP4 interval or a sixteen-byte NVFP4 interval. Outliers retain the original
byte in the existing exception format. Encoding synchronizes to allocate the
compact exception stream and occurs before graph capture. Reconstruction
scratch is supplied by the integration and may be shared only by serialized
layers. The integration must release original scale parameters after preparation.
See [the preparation guide](../../docs/csf-scale-compression.md).

The paired vLLM implementation uses `VLLM_B12X_MOE_FP4_CSF=1`, default `0`,
with the standard safetensors loader. It allocates original expert scales on
the CPU, stages one layer through the ordinary weight post-processing context,
and releases the source parameters. Scratch is model-scoped, weakly indexed,
and retained by prepared layers. B12X performs no checkpoint discovery or TP
slicing. Dense-layer quantization and expert activation selection are unchanged.

**Unsupported:** NVFP4 A8, automatic precision switching, an A16 token cutoff,
ungated experts, overlapping execution using the same scratch, and MXFP4
nonunit global weight factors or A8 activation factors. The vLLM integration
rejects PP greater than one, DP greater than one, EP, microbatch overlap, and
EPLB. Existing CSF geometry restrictions apply. Dispersed source bytes can
produce a larger representation; compression does not imply a fixed saving.

## Correctness and ownership

**Qualified** on RTX PRO 6000 Blackwell with CUDA 13.4 / CUTLASS DSL 4.7.1:

| Conditions and measurement | Result | Conclusion |
|---|---|---|
| MXFP4/NVFP4 codec and MoE suites, including zero/dense exceptions, column tails, interval ties, gate/up order, and graph replay | 129 tests passed | Existing and online representations satisfy the tested byte and expert-output contracts |
| Source-scale alias, source-weight alias, and nonunit MXFP4 factor rejection before mutation | 3 focused cases passed | Invalid source/scratch ownership fails before preparation changes weights |
| vLLM model-scoped scratch, lifetime, and source release | 5 tests passed | Independent models do not share ownership; original scale tensors become collectable |
| Original DS4.1 layer 3, TP4 rank 0, all 384 experts, H=5120, N=576; 1/17/33/128 tokens | Prepared weight/scale bytes and native expert outputs exactly equal | MXFP8 arithmetic remains native on the tested real layer |
| NVFP4 A4/A16 and MXFP4 A8/A16, frozen graph resolution, scratch poisoned before replay | Exact native outputs, no replay allocation | Encoder is not a replay operation; reconstruction restores required scale bytes |

Native and compressed expert preparations receive independent source clones:
ordinary preparation can normalize weights in place. The MXFP4 A16 cases use
the supported Kimi H=3584 geometries. They qualify the component, not a complete
Kimi online serving deployment. Full-model KLD and complete logit parity were
not measured for the online loading feature.

Reproduce the component gates in a B12X CUDA environment:

```bash
python -m pytest tests/moe/test_nvfp4_csf.py tests/moe/test_mxfp4_csf.py \
  tests/quantization/test_nvfp4_csf.py tests/quantization/test_x4t_scales.py -q
```

The reported 129-test sweep precedes addition of the source-weight alias case;
the focused three-case rerun covers the final rejection guard. The vLLM command
is `python -m pytest --confcutdir=tests/kernels/moe tests/kernels/moe/test_b12x.py
-k 'online_csf or source_release' -xq`. Repository pre-commit hooks pass.
B12X changed-code lint passes except for three unchanged `B008` diagnostics
on existing `FrozenMapping()` defaults in `api.py`; baseline comparison is
retained with the local receipts.

## Encoder cost on original scale bytes

**Research-only timing:** six sequential synchronized encodes of both expert
projections on GPU 14, 600 W. Inputs are already CUDA-resident. Exact
reconstruction by the existing decoder is required before reporting results.
The first sample includes compilation and allocation; all six samples are
in [the evidence JSON](online-scales-evidence.json). Metadata and shared decode
scratch are excluded from the compressed byte counts below.

| Source layer and shard | Original scale bytes | Fixed stream + exceptions | First encode | Final four encodes |
|---|---:|---:|---:|---:|
| DS4.1 layer 3, TP4 rank 0 | 106,168,320 | 17,299,132 | 1.381 s | 2.878–2.919 ms |
| Qwen3.8-Flash-Next layer 3, TP1 | 157,286,400 | 82,924,412 | 0.267 s | 2.471–2.555 ms |

The second samples are 29.35 and 54.52 ms respectively. These are encoder
measurements, not whole-model loading times. Weight normalization, CPU staging,
validation, and preparation add cost. Per-iteration clocks were not captured;
the host reports NVML throttle mask `0x400`, so these timings do not qualify
a clean-clock release performance claim.

The byte measurement and real DS4.1 expert probe are retained in
[`online-evidence/`](online-evidence/). Mount the native HF cache at
`/root/.cache/huggingface/hub` and an output directory at `/results`, then run
`python validation/csf/online-evidence/measure_scale_bytes.py` or
`python validation/csf/online-evidence/probe_ds41_experts.py` with a selected GPU.

## Complete native-checkpoint loading

**Qualified for startup and the listed smoke requests:** the ordinary
safetensors loader read native checkpoints with online CSF enabled; each
server answered eight short generation requests. No converted checkpoint was
written. These smoke requests do not establish full-model accuracy parity.

| Model/configuration | Loaded model memory | Online preparation logged by layers | Result |
|---|---:|---:|---|
| Qwen NVFP4, TP1, MTP3, PLE host memory 26.82 GiB, FP8 KV | 71.13 GiB/GPU | 49 layers; 13.29 s total; 0.135 s median | 8/8 requests completed; KV 14.94 GiB / 810,925 tokens |
| DS4.1 MXFP4, TP4, no speculation, Engram in host memory, FP8 KV fixed at 2 GiB/GPU | 68.72 GiB/GPU | 40 layers/rank; 5.82–8.90 s per rank | 8/8 requests completed; native `B12X_MXFP4_MXFP8` experts |

Qwen model loading took 299.73 s; DS4.1 rank loading took 185.09–196.44 s.
These are loader log durations, not additional encoder overhead. A separate
Qwen native TP1 control loaded 73.86 GiB/GPU and exposed 12.20 GiB KV. The
online run used GPU 11 and that control used GPU 10: the memory observations
show source-scale release, but the startup durations are not a paired speed
comparison. Full online Kimi and GLM serving are unmeasured.

## Native versus stored-CSF serving

The following comparisons use **precompressed checkpoints** and the repaired
decoder from the FP4-CSF serving work. They do not measure online encoding.
Both arms use the same physical GPUs, native expert activation mode, FP8 KV,
and a 30-second sustained decode window at prompt context 0. Only fully
admitted, unqueued cells are included. All throughput figures are
**research-only** because NVML reports `0x400`; raw clocks, physical UUIDs,
source-manifest hashes, and per-pass samples are in the evidence JSON.

| Model/configuration | Concurrency | Native tok/s | CSF tok/s | CSF/native − 1 | Native steps/s | CSF steps/s |
|---|---:|---:|---:|---:|---:|---:|
| GLM TP2, no speculation | 1 | 126.82 | 117.22 | −7.57% | — | — |
| GLM TP2, no speculation | 8 | 517.72 | 504.28 | −2.60% | — | — |
| Qwen TP1, MTP3, PLE in RAM | 1 | 200.60 | 194.84 | −2.87% | 86.03 | 84.59 |
| Qwen TP1, MTP3, PLE in RAM | 8 | 816.58 | 799.82 | −2.05% | 351.33 | 352.19 |

GLM uses GPUs 12/13 at loaded memory clock 16,365 MHz, 600 W, KV 3 GiB/GPU,
max length 8192, batch-token limit 1024, sequence capacity 32, prefix caching
with aligned recurrent state, and at most 6144 output tokens per request.
Both arms have two passes. CSF C8 ranges from 494.12 to 514.44 tok/s; the
spread is material. The TP2 native target occupies 89.12 GiB/GPU; adding the
MTP draft did not fit in the tested configuration. GLM C1 remains a measured
regression and is not concealed by the C8 mean.

Qwen uses GPU 10 at loaded memory clock 13,365 MHz, 600 W, memory utilization
0.96, max length 262144, batch-token limit 4096, sequence capacity 16, and at
most 8192 output tokens. Native has one valid C1/C8 pass and CSF has two.
At C8, output throughput is 2.05% lower while the acceptance-normalized step
rate is 0.24% higher; MTP acceptance therefore matters to the output comparison.
At C1 the step rate is 1.67% lower. Native C16 admitted at most 14 requests and
is capacity-limited, so its throughput is excluded. CSF sustained all 16 at
1146.61 tok/s on average; that is a capacity observation, not a C16 speedup
claim. These measurements do not resolve the separate Qwen TP2 regression.

## Artifact identities

Image: `sha256:5501c32fa048b25d53004ea9cd391477467eae9b4e9d9c2dcccc851086dac46b`.
Online source bases are B12X `bb5f4c4a43b420a1451f38c8aca0b0d9e6385662` and
vLLM `a5430f5f25b681cf0b3f35bd339e1d1f21d47fb1`. Serving uses a compatible beta
overlay; the receipts identify all source bytes rather than equating the
overlay with a bare repository revision. Final source hashes, raw artifact
hashes, launcher options, and model-load observations are in
[`online-scales-evidence.json`](online-scales-evidence.json).

| Checkpoint | HF revision |
|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `b797d2e1160b9596b2570e56c1d3590faa09d4ed` |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF` | `4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` |
| `local-inference-lab/GLM-5.3-Flash-NVFP4` | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-CSF` | `20f4777422f833c48b67bb0e554e1bc61dc55ca1` |

Full local receipts are under `/data/trellis-quant/csf-online-20261001`;
each serving arm has `launch.json`, `source-manifest.json`, `server.log`,
`telemetry.csv`, and request/benchmark outputs. Test receipts are
`final-gpu-tests-1.log`, `online-guards-final-correct-image.log`, and
`vllm-online-tests-final.log`. The implementation does not claim online load
qualification on other GPUs, compiler versions, or overlapping execution modes.
