# W4A16 static expert LoRA — SM120 validation receipt

This receipt records the post-review performance and graph-replay check for
the static rank-4 expert-LoRA path. It is evidence for source commit
`6d39c0b98e94f4f12966bb26d2830afc255169d2`.

## Environment

- GPU: NVIDIA GeForce RTX 5090, SM120, 170 SMs, physical UUID
  `GPU-03f754c5-967b-4338-4f2b-affeedf61251`
- Driver: 595.84
- Container:
  `voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18`
- Python: 3.12.3
- PyTorch: 2.13.0, CUDA 13.3
- CUTLASS DSL: 4.6.2
- Triton: 3.7.1+gitf797708c.nv26.7
- Git status during the run: clean worktree, one local commit ahead of the
  then-published PR branch

## Shape and method

- 256 experts
- hidden size 4096
- full intermediate size 2048, TP4 shard size 512
- top-k 6
- SwiGLU/SiLU with limit 10.0
- BF16 activations
- native ModelOpt FP4 weights
- synthetic rank-4 adapter values with separate, contiguous gate/up B storage
  matching vLLM's runtime layout
- CUDA-graph replay, 100 iterations per sample, seven samples per variant
- baseline and LoRA samples interleaved in alternating A/B then B/A order

The synthetic values preserve the production launch geometry without
embedding or disclosing a private adapter. Correctness is covered separately
by the oracle and live CUDA-graph tests listed below.

Command:

```text
/opt/venv/bin/python benchmarks/benchmark_static_expert_lora.py \
  --tokens 1 2 4 8 \
  --iterations 100 \
  --repeats 7 \
  --split-w13-b \
  --output /results/pr240-6d39c0b-synthetic-split-gpu4.json
```

## Results

Times are microseconds per graph replay. `slowdown_x` is LoRA divided by the
unmodified base dispatcher for the same token count.

| Tokens | LoRA path | Base median | LoRA median | slowdown_x | Replay allocation delta |
|---:|---|---:|---:|---:|---:|
| 1 | direct augmented | 32.7933 | 38.9770 | 1.1886 | 0 bytes |
| 2 | staged | 41.6454 | 47.1258 | 1.1316 | 0 bytes |
| 4 | staged | 75.8266 | 65.6470 | 0.8658 | 0 bytes |
| 8 | staged | 151.1171 | 149.8672 | 0.9917 | 0 bytes |

The negative apparent overhead at M=4 and M=8 does not mean that LoRA makes a
fixed kernel faster. Static LoRA intentionally selects the staged tensor-core
path at M>=2, while the base dispatcher can select a different small-M path.
These rows therefore validate the production dispatch decision as a whole.
The single-token row is the direct, like-for-like decode cost most relevant to
steady-state autoregressive serving: 18.86% in this run.

Raw samples:

```text
M=1 base=[33.082559, 32.794240, 32.793281, 32.805121, 32.785921, 32.792001, 32.786880]
M=1 lora=[39.101441, 39.044480, 39.018559, 38.968639, 38.928959, 38.976960, 38.930881]

M=2 base=[41.939840, 41.645441, 41.551361, 41.649599, 41.467199, 41.693439, 41.610241]
M=2 lora=[47.175999, 47.136960, 47.105598, 47.106881, 47.133121, 47.125759, 47.118082]

M=4 base=[76.031680, 75.598402, 75.793920, 75.865278, 75.826559, 75.793281, 75.908160]
M=4 lora=[65.748801, 65.575042, 65.666561, 65.593600, 65.920639, 65.574079, 65.647039]

M=8 base=[150.759363, 150.807362, 150.858879, 151.555204, 151.319361, 151.177921, 151.117125]
M=8 lora=[145.148478, 145.150080, 148.053436, 149.867201, 150.569916, 150.142078, 150.950403]
```

Every row produced finite base and LoRA output, a non-zero adapter delta, and
zero change in PyTorch allocated bytes across the measured replay interval.

## Correctness and lifecycle checks

The committed source was also exercised on the same physical GPU with:

- 16 direct/staged cases spanning M=1/M=17, native/packed FP4,
  split/fused W13 B storage, int32/int64 mappings, eager execution, and live
  CUDA-graph replay: all passed.
- Seven static-LoRA primitive and immediate-after-bind graph-freeze cases:
  all passed. The binding test freezes kernel resolution before the first
  execution, proving that the complete staged CuTe launch set was resolved at
  bind time.
- Seven low-level and public W4A8 Trellis cases covering the corrected optional
  micro-kernel ABI: all passed.

Pytest's only warnings were expected cache-write warnings caused by mounting
the source tree read-only inside the validation container.
