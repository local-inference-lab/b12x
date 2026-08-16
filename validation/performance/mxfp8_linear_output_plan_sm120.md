# MXFP8 Linear Output-Storage Plan Qualification

Status: **qualified**

The `mxfp8_linear.mm` output-storage policy selects the swapped TMA
`(64, 32)` MMA tile for the Kimi-K3 tensor-parallel projection shape
`M=3, N=132, K=7168`. The policy preserves the tuned default tile for other
shapes before adapting it to swapped operand order.

## Conditions

- B12X source revision: `8cb3f16adc6499fd8d87b82a42348490973e10db`
- Container:
  `voipmonitor/vllm:kimi-k3-tp16-vllm2ddc210-b12x3bce5d8-cu133-torch213-20260816-r1`
- Source mount: `/src/b12x`, selected through `PYTHONPATH=/src/b12x`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
  UUID `GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`, PCI bus
  `00000000:03:00.0`
- GPU state at receipt collection: P1, 2,610 MHz SM clock, 13,365 MHz memory
  clock, 600 W power limit, default compute mode, active throttle mask
  `0x0000000000000000`
- Driver: 610.57.04
- CUDA runtime: 13.3
- PyTorch: 2.13.0
- Operation: complete `mxfp8_linear.mm` CUDA-graph replay, including BF16
  activation quantization and the dense MXFP8 GEMM
- Correctness reference: dequantized MXFP8 activation and weight matrix
  multiplication converted to BF16
- Timing: 100 warmup pairs followed by 1,000 recorded sample pairs; arm order
  alternates AB then BA
- Compile cache: empty at process start; each plan is mapped to the cache
  objects created by its graph-capture setup
- PTXAS: CUDA 13.3 V13.3.73 at `/usr/local/cuda-13.3/bin/ptxas`, SHA-256
  `afd8d1e1fa6e310f7faee44f6621e4c1315fb7fd6da7d4d87414358e12a651dc`

## Result

| Swapped TMA MMA tile | First replay | Warm median | Warm minimum | Warm maximum | Reference mismatches |
|---|---:|---:|---:|---:|---:|
| `(64, 32)` | 33.312 us | 18.080 us | 17.056 us | 19.104 us | 0 |
| `(64, 64)` | 36.448 us | 22.176 us | 20.128 us | 22.464 us | 0 |

The `(64, 32)` plan required 81.53% of the `(64, 64)` median time. Equivalently,
the `(64, 64)` plan was 1.2265 times slower. Both plans produced finite output
with no element outside `rtol=0.01, atol=0.02`.

Both comparison arms use source revision
`8cb3f16adc6499fd8d87b82a42348490973e10db`; the result compares output plans,
not code revisions. The recorded ratio is the `(64, 64)` median divided by the
`(64, 32)` median, so a value above one means `(64, 64)` is slower.

## CUTLASS and PTXAS artifact map

The empty B12X compile cache produced three hash-verified CUTLASS objects:

| Plan setup | Kernel | Cache key | Object SHA-256 |
|---|---|---|---|
| `(64, 32)` | MXFP8 activation quantization | `20a391e33aa371c2dc2ebfb3c674160985cfc0ccec82f8858e1b44bef1e8a6ca` | `9226fea97a5e103c55b242d5f646939edf18c3cc4b57621608de6f1e9fc6335a` |
| `(64, 32)` | dense GEMM | `e956a4f7d65fa58ce9823d8ea6a392126be71108fbabcaad0f40681a83514274` | `4758ba3386853fe22f06145b1789a1caa3fd5fd2676c67a40ded1731de14b64f` |
| `(64, 64)` | dense GEMM | `b8120d52d28a1e28cc192b33fbd789bd674839e8797547385a256b121360d7d6` | `3d2afcb84ffd1358719848e34dcbd7c8da15147c4c9138fb9677b009237c32fd` |

Each raw receipt entry also records the manifest hash, compile-spec hash,
CUTLASS DSL 4.6.2 toolchain identity, compile options, compile environment,
target identity, and launch metadata. The PTXAS executable identity above is
shared by all three objects.

The compressed raw receipt, including all timing samples, source hashes,
invocation, and hardware metadata, is
[`mxfp8_linear_output_plan_sm120.json.gz`](mxfp8_linear_output_plan_sm120.json.gz).
Its SHA-256 digest is
`5d6532a9df5228e2f0fae3e964011158736f2078035fb35222ec8f6061e3fed2`.
The decompressed JSON SHA-256 digest is
`342143dd87a9ed5b934be646ff1a09110bc6198fa94002700b59b7c3066b361f`.

## Reproduction

Run the benchmark from a clean checkout of the recorded B12X revision. Mount
that checkout at `/src/b12x` so the container imports the recorded source:

```bash
benchmark_cache=$(mktemp -d)
docker run --rm --gpus '"device=0"' \
  --entrypoint /bin/bash \
  -v "$PWD:/src/b12x:ro" \
  -v "$benchmark_cache:/cache" \
  -e PYTHONPATH=/src/b12x \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e B12X_COMPILE_CACHE_DIR=/cache/b12x-compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/cute \
  -e CUTE_DSL_CACHE_DIR=/cache/cute-dsl \
  -e CUDA_CACHE_PATH=/cache/cuda \
  -e XDG_CACHE_HOME=/cache/xdg \
  -e B12X_BENCHMARK_SOURCE_REVISION=8cb3f16adc6499fd8d87b82a42348490973e10db \
  -e B12X_BENCHMARK_WORKTREE_STATE=clean \
  -e B12X_BENCHMARK_CONTAINER_IMAGE=voipmonitor/vllm:kimi-k3-tp16-vllm2ddc210-b12x3bce5d8-cu133-torch213-20260816-r1 \
  voipmonitor/vllm:kimi-k3-tp16-vllm2ddc210-b12x3bce5d8-cu133-torch213-20260816-r1 \
  -lc 'unset NCCL_GRAPH_FILE; cd /src/b12x && python benchmarks/benchmark_mxfp8_linear_output_plan.py --iterations 1000 --measurement-warmup 100'
```
