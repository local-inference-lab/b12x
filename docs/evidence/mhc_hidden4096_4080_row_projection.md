# Hidden-size-4096 multi-head connection projection at 4080 live rows

Status: **qualified** for CUDA graph replay on SM120 with the software and
hardware conditions recorded below. The result qualifies the multi-head
connection (mHC) projection operation; it does not by itself predict
end-to-end serving throughput.

## Operation and revisions

The operation is the fused multi-head connection post/pre transform with BF16
inputs, a 24-by-16384 FP32 projection, TF32 tensor-core math, fused RMSNorm,
4096 hidden columns, four residual streams, and 4080 live rows. A scheduler
batch with capacity 4096 can reserve 16 rows and present this live-row count to
the kernel.

- Comparison revision: `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`
- Qualified revision: `86a840624a0e9fc0a8077c2d43184456c8c66cf9`
- Qualified worktree:
  `/root/vllm/worktrees/b12x-glm53-mhc-prefill-2fcf23a0-20260828`
- Comparison kernel SHA-256:
  `decba110f9e817d2f5016f7255a01d4349381a435380eba559f3a50490954859`
- Qualified kernel SHA-256:
  `14ecec5e95491f5c8285f50c77089d3ff1446fbff3e1cfd13edc05abfaf86aea`
- Benchmark SHA-256:
  `8a29604868889d5f0f5368ce1d5f49077068f1c02cab9d5c919608945c9fc102`
- Container image:
  `voipmonitor/vllm:glm53-flash-nvfp4-luke-clean-vllme75bcfd-b12x58a046f-fi1ac6942-cu133-torch213-20260828-r1`
- Container image ID:
  `sha256:ecc2b7a12cc369d1f04fc896be29170a8fbb1e027f1dd9f58f202f23a54a38ea`

The comparison arm used the B12X tree embedded in the container. The qualified
arm mounted `b12x/norm/mhc/_kernels.py` from the qualified revision. Both arms
mounted the benchmark from the qualified revision so that threshold selection
and raw sample reporting were identical.

## Hardware and execution mode

- Physical GPU index: 7
- GPU UUID: `GPU-0027fc86-3322-ce2a-856c-f49eb61eb63e`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Compute architecture: SM120, compiled with `CUTE_DSL_ARCH=sm_120a`
- Persistence mode: enabled
- Compute mode: default
- Initial state: P8, 180 MHz SM, 405 MHz memory, throttle mask `0x0`, 29 C
- Final state: P1, 2610 MHz SM, 13365 MHz memory, throttle mask `0x0`, 31 C
- CUDA visibility inside the container: physical GPU 7 mapped to logical GPU 0
- Timing mode: captured CUDA graph, 20 warmup replays, 30 measured replays per
  arm, no requested L2 flush
- Arm order: comparison A, qualified A, qualified B, comparison B
- Compile state: a fresh B12X and CuTe DSL cache directory for every arm

Clocks were not administratively locked. The balanced arm order and identical
medians in both comparison arms limit drift in this qualification, but the
recorded result is not a fixed-clock release certification.

## Command

Each arm ran the following command from `/opt/glm53-flash/b12x` inside the
container. The qualified arm additionally mounted the qualified
`b12x/norm/mhc/_kernels.py` at the corresponding container path.

```bash
/opt/venv/bin/python -B benchmarks/benchmark_residual.py \
  --tokens 4080 \
  --expected-m 4096 \
  --hidden-size 4096 \
  --split-k 64 \
  --block-k 256 \
  --block-h 512 \
  --fuse-rmsnorm \
  --prefill-tf32-mma \
  --no-prefill-block-m \
  --warmup 20 \
  --iters 30 \
  --print-samples
```

The container environment set `CUDA_VISIBLE_DEVICES=0`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUTE_DSL_ARCH=sm_120a`,
`PYTHONDONTWRITEBYTECODE=1`, and arm-local `B12X_COMPILE_CACHE_DIR` and
`CUTE_DSL_CACHE_DIR` values.

## Correctness and graph replay

`tests/gemm/test_launch_custom_ops.py` launches the production projection on
both sides of the dispatch boundary, captures it in a CUDA graph, mutates the
stable input allocation, poisons the output workspace with NaNs, and replays
the graph. The reconstructed projection must be finite, nonzero, and within
`rtol=0.02, atol=0.0625` of `torch.nn.functional.linear`.

The targeted command passed on the recorded GPU:

```bash
/opt/venv/bin/python -B -m pytest -q \
  tests/gemm/test_launch_custom_ops.py \
  -k "mhc_prefill_chunk_geometry or mhc_prefill_chunk_boundary"
```

Result: `3 passed, 11 deselected`.

For the 4080-row benchmark, the comparison arm reported output RMSE
`4.76e-06`, projection maximum error `0.0625`, and projection RMSE `0.000875`.
The qualified arm reported output RMSE `4.76e-06`, projection maximum error
`0.0625`, and projection RMSE `0.000861`.

## Timing result

| Arm | Selected geometry | Median latency | Minimum latency |
|---|---|---:|---:|
| Comparison A | `m16n8k256s1wm1wn1`, one K split | 381.60 us | 378.53 us |
| Qualified A | `m192n24k64s2wm12wn1`, eight K splits | 296.10 us | 292.51 us |
| Qualified B | `m192n24k64s2wm12wn1`, eight K splits | 296.59 us | 292.54 us |
| Comparison B | `m16n8k256s1wm1wn1`, one K split | 381.60 us | 378.53 us |

The pooled comparison median is 381.60 us and the pooled qualified median is
296.58 us. The qualified geometry reduces latency by 22.28%. Expressed in the
opposite direction, fixed-work throughput is 1.2867 times the comparison arm,
or 28.67% higher.

### Raw replay samples in microseconds

Comparison A:

```text
382.66,378.72,380.58,380.58,382.62,382.62,380.58,380.58,380.58,384.67,378.53,381.60,380.54,380.58,382.62,378.53,383.65,382.62,380.58,380.58,382.62,382.62,381.60,378.53,382.62,382.62,382.62,381.60,382.62,382.59
```

Qualified A:

```text
298.59,294.82,296.61,296.61,294.53,296.64,294.56,298.62,294.59,296.61,294.56,298.66,294.53,292.51,295.62,296.61,298.66,294.56,294.56,294.53,296.64,297.60,292.51,296.61,294.59,294.53,296.64,298.66,295.58,296.58
```

Qualified B:

```text
299.55,294.91,296.61,294.53,296.64,296.61,294.56,297.63,294.56,294.56,296.61,296.61,296.61,294.56,295.58,294.56,296.58,292.54,296.61,294.59,296.61,295.58,296.61,296.61,298.66,296.61,294.56,294.56,295.58,296.61
```

Comparison B:

```text
382.50,380.83,384.67,380.58,382.62,382.62,380.58,382.62,380.58,380.58,381.63,382.56,378.53,384.67,378.53,382.62,381.60,380.58,380.58,380.54,382.62,378.53,383.65,380.58,382.62,382.62,380.58,381.60,382.59,380.58
```

## Compatibility

The 4080-row threshold applies only to hidden size 4096. Hidden size 7168 and
other hidden sizes retain the 4096-row threshold. The global environment
override continues to set every hidden size unless
`B12X_MHC_PREFILL_TF32_TMA_CHUNK_4096_MIN_TOKENS` explicitly overrides hidden
size 4096. Live row counts remain runtime launch inputs and are not part of the
kernel compile key.
