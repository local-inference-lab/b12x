# Native W4A16 inactive-route validation on SM120

Status: **qualified** for native ModelOpt W4A16 fused small-M execution and
FC2-only runtime-M execution on NVIDIA RTX PRO 6000 Blackwell GPUs.

## Operation contract

An expert identifier outside `[0, resident_expert_count)` is an inactive route.
The operation must not read weights or scales through that identifier, and its
contribution must be exactly zero. Valid routes retain their identifiers and
weights. Caller-owned route tensors remain unchanged in eager execution and
CUDA Graph replay.

Fixed-M fused launches sanitize their compile-time-bounded route table in
shared memory. FC2-only launches accept runtime M, so they validate each route
inline against the runtime resident-expert count. The FC2 implementation must
not index shared route storage whose extent was compiled for a smaller M.

## Source and runtime identity

- Repository: `local-inference-lab/b12x`
- Runtime implementation revision: `af354efdbadb8b722da7c696e41d7e35b849b1ec`
- Runtime implementation tree: `dc1d9340849219f99b5ba93e915b0fbd10967028`
- Test-complete revision: `6a41770fb1514c4db03b4ff552380ec6821a3ae9`
- Test-complete tree: `5ea752169a9c4a2863f51f16ed77cc85371f1353`
- Comparison revision for valid FC2 routes:
  `debaafe156c9824396178d53e01e5f15d2a2a04a`
- Comparison tree: `ec6edd9da4687f83519fd37bd7322ea0800f0ace`
- Implementation worktree:
  `/root/vllm/worktrees/b12x-ii-w4a16-inactive-routes-v2-20260817`
- Comparison worktree:
  `/mnt/luke/kimi-k3-runs/pr227-fc2-review-20260817/b12x-debaafe`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120
- Driver: 610.57.04
- Runtime: CUDA 13.3, PyTorch 2.13.0, CUTLASS DSL 4.6.2
- Test image:
  `voipmonitor/vllm@sha256:ffb25774eaa90850b4cacfb88ed9e55072818e99bad977f1315c7118e7a730b2`

## Correctness and memory safety

The targeted test command selected the staging invariant, an established valid
FC2 case, both narrow and wide invalid-route M=3 cases, and the eager plus CUDA
Graph M=7 case:

```bash
docker run --rm --gpus device=0 --ipc=host \
  --entrypoint /opt/venv/bin/python3 \
  -e PYTHONPATH=/workspace \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e B12X_COMPILE_CACHE_DIR=/cache/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/cute \
  -e CUDA_CACHE_PATH=/cache/cuda \
  -e CUTE_DSL_CACHE_DIR=/cache/cute-dsl \
  -v /root/vllm/worktrees/b12x-ii-w4a16-inactive-routes-v2-20260817:/workspace:ro \
  -v /mnt/luke/kimi-k3-cache/pr227-fc2-review-20260817:/cache:rw \
  voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllmdf13924-b12xec6edd9-cu133-torch213-20260817-r6 \
  -m pytest -q -s \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_runtime_m_does_not_use_fixed_route_staging \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_only_consumes_contiguous_bf16_and_native_mxfp4 \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_only_zeroes_invalid_routes_at_runtime_m3 \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_only_is_cuda_graph_safe_with_preallocated_output
```

Result: five parametrized tests passed. The invalid identifiers included `-1`,
`-9`, and the first upper-bound identifier. The M=3 test exercised 256- and
512-element intermediate widths. Assertions covered finite output, nonzero
valid-route output, exact-zero inactive rows, an independent constant oracle,
stable graph replay, and immutable route inputs.

The same invalid-route cases were executed under Compute Sanitizer:

```bash
docker run --rm --gpus device=0 --ipc=host \
  --entrypoint /usr/local/cuda/bin/compute-sanitizer \
  -e PYTHONPATH=/workspace \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e B12X_COMPILE_CACHE_DIR=/cache/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/cute \
  -e CUDA_CACHE_PATH=/cache/cuda \
  -e CUTE_DSL_CACHE_DIR=/cache/cute-dsl \
  -v /root/vllm/worktrees/b12x-ii-w4a16-inactive-routes-v2-20260817:/workspace:ro \
  -v /mnt/luke/kimi-k3-cache/pr227-fc2-review-20260817:/cache:rw \
  voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllmdf13924-b12xec6edd9-cu133-torch213-20260817-r6 \
  --tool memcheck --error-exitcode=99 \
  /opt/venv/bin/python3 -m pytest -q -s \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_only_zeroes_invalid_routes_at_runtime_m3 \
  /workspace/tests/moe/test_w4a16_e2e.py::test_w4a16_fc2_only_is_cuda_graph_safe_with_preallocated_output
```

Result: three parametrized tests passed and Compute Sanitizer reported
`ERROR SUMMARY: 0 errors`.

An extended W4A16 and micro-Trellis selector produced 67 passes, 16 skips, and
two failures. Both failures require a caller-owned CUDA Graph output buffer and
reproduce unchanged at comparison revision `debaafe156c9824396178d53e01e5f15d2a2a04a`;
they are not introduced by inactive-route handling.

## FC2 valid-route latency diagnostic

The diagnostic used valid routes, a caller-owned output, 200 graph warmups,
nine samples, and 5,000 CUDA Graph replays per sample. Lower latency is better;
the reported ratio is implementation latency divided by comparison latency.

| Source | M | Raw microseconds per replay | Median |
|---|---:|---|---:|
| comparison `debaafe1` | 2 | 2.226042, 2.201734, 2.150394, 2.139565, 2.154854, 2.202118, 2.181280, 2.181818, 2.180480 | 2.181280 |
| implementation `af354efd` | 2 | 2.170336, 2.137523, 2.150829, 2.141190, 2.157146, 2.165818, 2.146022, 2.139597, 2.221146 | 2.150829 |
| implementation `af354efd` | 7 | 4.098701, 4.098419, 4.098074, 4.097894, 4.098157, 4.097914, 4.098016, 4.097914, 4.098009 | 4.098016 |

The M=2 median ratio is 0.9860. This diagnostic establishes that runtime route
validation did not produce a measurable valid-route FC2 regression. It is not
a speedup claim because the arms were measured sequentially without a locked
clock.

## Full-model integration

The candidate image copied the implementation-revision `b12x` package over the
published integration image without changing vLLM, LMCache, model weights, or
launch arguments:

- Candidate image ID:
  `sha256:d2060dab541504dfec43e352a817e353e9672eec34231fab734db340b69c3154`
- Base image digest:
  `sha256:ffb25774eaa90850b4cacfb88ed9e55072818e99bad977f1315c7118e7a730b2`
- Candidate Dockerfile SHA-256:
  `0707481e9ea583cf8ac13524c730e6d1913a60d53aa5409d5c7b0227fc2d3549`

Conditions were the official Kimi-K3 MXFP4 target, the Inferact seven-token
DSpark draft, TP16/DCP16, FP8 target KV cache, a 1,000,000-token model limit,
native vision, LMCache, and CUDA Graph capture size eight. InstantTensor loaded
90.48 GiB per GPU. Physical target KV capacity was 1,033,126 tokens. CUDA Graph
capture completed and the API became healthy.

The benchmark program is
[`benchmark-kimi-k3-dspark-decode.py`](https://github.com/local-inference-lab/rtx6kpro/blob/a82029c0ffa9c1cccfa9215e927c3db0ae2aeb57/models/kimi-k3/tools/benchmark-kimi-k3-dspark-decode.py),
SHA-256 `b465fb785fc11b5b2941510eca1cfbdd159a55d8258c827a3db110309768023b`.
The command used 256 stored prompt tokens, 1,024 generated tokens, temperature
zero, seed one, two warmups, and eight measured runs:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-dspark-decode.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-MXFP4-DSpark7-DCP16-1M \
  --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
  --prompt-tokens 256 --max-tokens 1024 --warmups 2 --runs 8 \
  --output-dir full-model-dspark-normalized-256x1024
```

Target-cycle rate is the acceptance-independent performance metric. Higher is
better; the reported ratio is candidate median divided by base-image median.

| Arm | Raw target cycles/s | Median |
|---|---|---:|
| base image | 31.384470, 31.589480, 31.633320, 31.563746, 31.237360, 31.340825, 31.474961, 31.232879 | 31.429716 |
| candidate image | 31.391607, 31.526917, 31.527495, 31.454522, 31.589554, 31.332777, 31.679354, 31.512322 | 31.519620 |

The candidate/base median ratio is 1.0029. Emitted throughput was 130.446
tokens/s for the candidate and 118.773 tokens/s for the base image, but that
difference is not attributed to the kernel because median draft acceptance was
0.4476 and 0.3973 respectively.

During a separate sustained 4,096-token candidate run, all 16 GPUs remained in
P1 at 100% utilization, memory clocks were 13,365 MHz, SM clocks ranged from
2,670 to 2,865 MHz, and the active clock-event mask was zero on every GPU. The
run measured 30.918 target cycles/s. GPU 0 was
`GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`.

Conclusion: the implementation satisfies inactive-route address safety and
zero-contribution semantics for fused fixed-M and FC2 runtime-M execution. The
qualified full-model profile loads, captures, and decodes without a target-cycle
regression.
