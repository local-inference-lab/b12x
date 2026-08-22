# W4A16 static expert LoRA — SM120 validation receipt

This receipt records the post-review performance and graph-replay check for
the static rank-4 expert-LoRA path. It is evidence for source commit
`bc11e00018caa2fc4e632479bd6bf43c1e2cef13` (source tree
`fd6a0cc009a493e2bf6316860e03a9f925a98cbc`) against comparison revision
`36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` (`origin/master` merge base).

## Environment

- GPU: NVIDIA GeForce RTX 5090, SM120, 170 SMs, physical UUID
  `GPU-03f754c5-967b-4338-4f2b-affeedf61251`
- Driver: 595.84
- Container:
  `voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18`
  (`sha256:414ec7d0d28358cfd8af0697f330f5c8acbb80e4dc4e5ba69c9fd5b5855ea804`)
- Python: 3.12.3
- PyTorch: 2.13.0, CUDA 13.3
- CUTLASS DSL: 4.6.2
- Triton: 3.7.1+gitf797708c.nv26.7
- PTXAS: CUDA 13.3, V13.3.73
- Worktree: `/root/b12x-lora-pr`
- GPU mode: P1, default compute mode, recovery action `None`; only the
  benchmark process was resident on the selected physical GPU
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
embedding or disclosing a private adapter. Before timing each variant, the
benchmark compares both base and LoRA outputs with the FP32 W4A16 oracle and
requires mean row cosine similarity >= 0.9975. A failure aborts before timing.

Command:

```text
/opt/venv/bin/python /root/b12x-lora-pr/benchmarks/benchmark_static_expert_lora.py \
  --tokens 1 2 4 8 \
  --iterations 100 \
  --repeats 7 \
  --split-w13-b \
  --output /results/pr240-bc11e00-synthetic-split-gpu4.json
```

## Results

Times are microseconds per graph replay. `slowdown_x` is LoRA divided by the
unmodified base dispatcher for the same token count.

| Tokens | LoRA path | Base median | LoRA median | slowdown_x | Replay allocation delta |
|---:|---|---:|---:|---:|---:|
| 1 | direct augmented | 32.7923 | 38.9424 | 1.1875 | 0 bytes |
| 2 | staged | 41.4307 | 47.1194 | 1.1373 | 0 bytes |
| 4 | staged | 75.6445 | 65.6413 | 0.8678 | 0 bytes |
| 8 | staged | 150.1123 | 148.1398 | 0.9869 | 0 bytes |

The negative apparent overhead at M=4 and M=8 does not mean that LoRA makes a
fixed kernel faster. Static LoRA intentionally selects the staged tensor-core
path at M>=2, while the base dispatcher can select a different small-M path.
These rows therefore validate the production dispatch decision as a whole.
The single-token row is the direct, like-for-like decode cost most relevant to
steady-state autoregressive serving: 18.75% in this isolated kernel run.

Raw samples:

```text
M=1 base=[33.100801, 32.788160, 32.792959, 32.839680, 32.790401, 32.785280, 32.792320]
M=1 lora=[39.044480, 38.942399, 38.957441, 38.934400, 38.960640, 38.936000, 38.935680]

M=2 base=[41.628799, 41.189122, 41.167998, 41.330881, 41.456962, 41.430721, 41.494079]
M=2 lora=[47.206721, 47.116799, 47.120638, 47.136960, 47.118402, 47.117119, 47.119360]

M=4 base=[76.058879, 75.680642, 75.604801, 75.659838, 75.638399, 75.644479, 75.630078]
M=4 lora=[65.800319, 65.578880, 65.641279, 65.581441, 65.644479, 65.577922, 65.646720]

M=8 base=[150.115843, 150.112324, 149.854403, 150.040636, 150.042562, 150.343361, 150.277119]
M=8 lora=[144.769278, 144.844484, 148.139839, 147.871037, 148.690557, 148.420801, 150.321283]
```

Every row produced finite base and LoRA output, a non-zero adapter delta, zero
change in PyTorch allocated bytes across the measured replay interval, and a
passing FP32 oracle comparison:

| Tokens | Base cosine | LoRA cosine | Base RMSE | LoRA RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.99999726 | 0.99999589 | 0.00019891 | 0.00024451 |
| 2 | 0.99999750 | 0.99998534 | 0.00028487 | 0.00072082 |
| 4 | 0.99999714 | 0.99998188 | 0.00036669 | 0.00095414 |
| 8 | 0.99999720 | 0.99998260 | 0.00035941 | 0.00093315 |

## Correctness and lifecycle checks

The committed source was also exercised on the same physical GPU with:

- 16 direct/staged cases spanning M=1/M=17, native/packed FP4,
  split/fused W13 B storage, int32/int64 mappings, eager execution, and live
  CUDA-graph replay: all passed.
- Two public immediate-after-bind cases (M=1 with int64 mapping and M=7 with
  int32 mapping) plus the 16 direct/staged cases: 18 passed. Binding validates
  the adapter and prewarms the typed CuTe direct ABI and every Triton
  shrink/expand specialization before the first execution and graph capture.
- Seven low-level and public W4A8 Trellis cases covering the corrected optional
  micro-kernel ABI: all passed.

Focused command: `python -m pytest tests/moe/test_w4a16_e2e.py -k
'static_lora and (oracle_and_graph or scratch_plan_binding)' -x -q`.

## Compiler artifacts

The machine-readable companion receipt
`validation/performance/w4a16_static_expert_lora_sm120.json` contains the full
compile specs, exact command, environment snapshots, and every hash. The fresh
cache produced nine CuTe manifests and five Triton PTX/CUBIN pairs.

| CuTe kernel | Cache key | Object SHA-256 | Evidence SHA-256 |
|---|---|---|---|
| small_m_direct | `03937d5d...d6f70` | `ccea5165...47eeb` | `a1219de0...cbc4f` |
| topk_sum | `123757d2...a1c6e` | `01b951e6...7d6c2` | `ee09ad0f...7308a` |
| small_m_direct | `3fb65001...b22969` | `4b6bf258...cc282` | `f3a549bc...8299` |
| activation | `4c50d389...907eb` | `34043b59...261ec` | `7d74c23d...2cf65` |
| small_m_direct | `4d641ac8...8886c` | `4d383bf4...ea74d` | `5f08e456...e37f3` |
| gemm | `81e7eae3...7835b` | `23757009...0717` | `01d4702c...0c33` |
| gemm | `8680faaa...f30e` | `11059dbc...dc50` | `0c553c47...b3b2` |
| small_m_direct | `b69c060a...43cc6` | `676cf3b8...4359` | `04d5f595...f2da` |
| small_m_direct | `d693eb30...cd575` | `4b8084df...c4cf` | `96380e5d...95d6` |

| Triton specialization | CUBIN SHA-256 | PTX SHA-256 |
|---|---|---|
| expand_pair_add | `0689999c...e63f5` | `bd1d44cb...255e6` |
| shrink (W13) | `7604c47e...c3111` | `9a79eb5d...c2eda` |
| expand_token_sum | `e67268d8...7e45` | `38134032...4a9b` |
| expand_add | `9b9a3e7d...20f0d` | `ea566953...07301` |
| shrink (W2) | `2ad16d6f...cc3c` | `65363f30...ad0b4` |
