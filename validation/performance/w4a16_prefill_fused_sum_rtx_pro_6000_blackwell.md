# W4A16 bounded prefill reduction qualification

Status: **qualified** for the full-model serving hardware, source composition,
and launch contract recorded below. The isolated kernel latency comparison is
**diagnostic** because its A-B-B-A run did not retain a contemporaneous GPU-mode
sample.

## Purpose

This record validates the opt-in W4A16 BF16 prefill path selected by
`B12X_W4A16_PREFILL_FUSED_SUM=1`. The path reduces routed FC2 results directly
into one FP32 row per input token. Its purpose is to bound scratch memory so a
4,096-token Kimi-K3 scheduler chunk fits beside the model and a physical
1,057,049-token KV cache.

The machine-readable receipt
`w4a16_prefill_fused_sum_rtx_pro_6000_blackwell.json.gz` contains every
CUDA-event timing sample from the A-B-B-A kernel comparison. Its SHA-256 is
`65db1e3220c45246e0eede79da4d9ebb448f27c631bfcf23194a5248efdeb551`.

## Source and hardware

- Repository: `local-inference-lab/b12x`
- Implementation revision: `0c3be37138f74a6d0213c10202e0077c2d2a44da`
- Implementation tree: `4794737159345008fa897579d6d3b67ed671a151`
- Materialized-path base revision:
  `c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5`
- Measured worktree:
  `/root/vllm/worktrees/b12x-k3-prefill-fused-reduce-20260819`
- Container:
  `voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa`
- CUDA runtime: 13.3
- PyTorch: `2.13.0a0+9186a08`
- Driver: `610.57.04`
- Microbenchmark GPU: physical GPU 0,
  `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Microbenchmark GPU UUID:
  `GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`
- Full-model hardware: 16 GPUs of the same type

The two persistent MoE artifacts use CUTLASS DSL 4.6.2 and PTXAS
`Cuda compilation tools, release 13.3, V13.3.27`:

| Arm | Compile-cache key | Object SHA-256 |
|---|---|---|
| Materialized BF16 routes | `d44aed42740f285a69b9bd972756111de92eb4ec8f7a21e683be3750a58eeeb6` | `34c5396f3d9f140fad4fed016d11516f19e8e063bff3729764811abf85e0f96a` |
| FP32 bounded reduction | `19a9b91e405c43b3eea9751b6ac02bf85c94918e254bfe005a064cf37eb67579` | `fb313da8431a46cc86ab1a2e6972dcf0f10af76554ea7a71cc666a617aa494b3` |

## Scratch contract

The measured rank shape has 4,096 tokens, hidden width 7,168, tensor-parallel
intermediate width 192, 896 router experts, top-k 16, BF16 activations, and the
`situ` activation.

| Caller-owned W4A16 scratch | Bytes per rank | MiB per rank |
|---|---:|---:|
| Materialized route output | 1,063,787,080 | 1,014.51 |
| FP32 fused reduction | 292,035,144 | 278.51 |
| Memory released | 771,751,936 | 736.00 |

The fused total includes a 48 MiB FC1 cache, a 112 MiB FP32 per-token
accumulator, a 24 MiB activation cache, GEMM accumulation scratch, and route
metadata. The materialized total stores one 7,168-element BF16 output for each
of the 65,536 token-route pairs.

## Kernel timing

Arm A sets `B12X_W4A16_PREFILL_FUSED_SUM=0` and materializes BF16 route output.
Arm B sets the flag to `1` and performs direct FP32 per-token reduction. Both
arms use identical synthetic weights, activations, routing, compiled source,
L2 flushing, and CUDA-graph timing.

Each A1-B1-B2-A2 process records ten repeats of 100 CUDA-event samples after
20 warmup iterations per repeat. The aggregate statistic is the median of the
20 per-repeat medians for each arm.

| Arm | Aggregate median | Samples |
|---|---:|---:|
| Materialized BF16 routes | 5,214.17 us | 2,000 |
| FP32 fused reduction | 5,304.32 us | 2,000 |

The ratio is fused latency divided by materialized latency: `1.01729`. The
bounded reduction is 1.73% slower as an isolated MoE launch. It is not claimed
as a kernel-latency optimization. Its serving gain comes from replacing four
1,024-token MoE launches with one 4,096-token launch under the same physical KV
allocation.

The A-B-B-A processes did not retain P-state or throttle-mask snapshots. Their
latency ratio is diagnostic rather than hardware-qualified. The full-model
measurement below retained 7,072 GPU-mode samples and is the authoritative
serving-performance result.

Reproduce either arm with this command and set the feature flag to `0` or `1`:

```bash
B12X_W4A16_PREFILL_FUSED_SUM=1 \
python benchmarks/benchmark_moe.py \
  --model-profile kimi-k3-mxfp4-shape \
  --batch-sizes 4096 \
  --quant-mode w4a16 \
  --validate none \
  --reference none \
  --compare-prefill-fused-sum \
  --graph-only \
  --warmup 20 \
  --iters 100 \
  --repeats 10 \
  --timing-json /tmp/w4a16-prefill-arm.json
```

## Full-model result

The serving test uses the official `moonshotai/Kimi-K3` MXFP4 target, the
`Inferact/Kimi-K3-DSpark` draft, TP16/DCP16, one active sequence, Triton KDA,
B12X MLA, disabled prefix caching, and a 1,325,000,000-byte FP8 KV allocation
per rank. `MAX_NUM_BATCHED_TOKENS=4102` reserves six DSpark slots and leaves an
exact 4,096-token prefill scheduler limit.

The complete launch command, source-composition procedure, benchmark command,
and machine-readable serving receipt are published in the
[Kimi-K3 full-MXFP4 4096-token prefill profile](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/full-mxfp4-p4096-prefill.md).
The compressed receipt beside this document also contains every measured
request's time to first token and effective prefill throughput.

| Prompt tokens | Four 1,024-token MoE launches | One 4,096-token MoE launch | Change |
|---:|---:|---:|---:|
| 8,192 | 2,723.1 tok/s | 3,861.7 tok/s | +41.81% |
| 32,768 | 2,897.0 tok/s | 3,732.5 tok/s | +28.84% |
| 65,535 | 2,839.7 tok/s | 3,554.4 tok/s | +25.17% |

The 8,192-token result is the median of six requests. The other results are
medians of three requests. Every request supplies token IDs directly, emits
one token, measures streamed time to first token, and uses a unique cache salt.
The physical KV capacity reported by vLLM is 1,057,049 tokens.

During the full-model measurements, every GPU reached 100% utilization, every
active sample was in P1, active SM clocks ranged from 2,595 to 2,880 MHz, and
the active clock median was 2,790 MHz. All 7,072 recorded GPU-mode samples had
an active throttle mask of zero.

## Correctness and regression coverage

- The default W4A16 GPU suite passed 292 tests and skipped 16 unsupported
  cases.
- Five focused planner, arena, activation-amax, CUDA Graph, and fused-output
  tests passed with the opt-in path selected.
- Fused output had cosine similarity 1.0 and maximum BF16 absolute difference
  0.015625 against materialized reduction.
- CUDA Graph replay retained scratch addresses, performed no replay-time
  allocation, consumed the FP32 accumulator, and produced finite nonzero
  output.
- A DSpark decode sanity test measured 31.486 normalized target cycles/s.

## Limitations

FP32 global additions use relaxed ordering. Bitwise equality with materialized
route reduction and bitwise repeat determinism are unsupported. The isolated
kernel measurement uses synthetic Kimi-K3-shaped inputs; the serving result is
the end-to-end evidence for scheduler throughput. Qualification does not cover
FP16, full-rotation Trellis, activation-amax capture, other GPU types, or other
topologies.
