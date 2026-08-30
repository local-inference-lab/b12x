# Hidden-size-4096 medium-row mHC projection qualification

## Status and operation contract

Status: **qualified** for the multi-head connection (mHC) TF32 projection on
SM120 when the hidden size is 4096 and the live row count is 2,304 through
3,583.

The dispatch policy selects an M64/N24/K64 projection with eight K splits. It
uses three pipeline stages for 2,304 through 3,071 rows and two stages for
3,072 through 3,583 rows. At 3,584 rows the existing wide-CTA chunk geometry
takes over. Hidden size 7168, decode dispatch, scratch ownership, output
layout, and public operation signatures are unchanged.

## Source and hardware

- Comparison revision: `fc1d4b68f7a5b0cfdb88bf06abccd869f5c589d5`,
  tree `7fcb6afb69b04b6981060012e943e125e58168b4`.
- Qualified implementation revision:
  `f90d16ee6fadc0f56488df06a0e2fd9bbccdc265`, tree
  `25f5d5c38d092931bd8e6df95eb6c01fea3c4680`.
- Runtime image:
  `local/vllm:glm53-flash-nvfp4-dcp4-dflash2-flashkda-bt4096-20260830-r3`.
- Physical GPU 0: `GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, driver 610.57.04, PCIe Gen5 x16.
- CUDA graph replay was used. Each process performed 20 warmups and 30 timed
  iterations after kernel compilation.

## Correctness

The focused test command was:

```bash
CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B -m pytest -q \
  tests/gemm/test_launch_custom_ops.py \
  -k "mhc_prefill_tf32_projection_geometry_boundaries or mhc_prefill_tf32_optimized_geometry_matches_reference_under_graph_replay or mhc_launch_ops_have_fake_dispatch"
```

Result: `5 passed, 10 deselected`.

The graph-replay cases cover 2,304, 3,072, and 3,583 rows. They mutate the
input and poison the reduction workspace before replay, then compare the
projection with `torch.nn.functional.linear`. The maximum absolute tolerance
is 0.0625 and the relative tolerance is 0.02.

## Performance

Both revisions were measured with this command at the 3,072-row policy
boundary:

```bash
/opt/venv/bin/python -B benchmarks/benchmark_residual.py \
  --tokens 3072 \
  --expected-m 4096 \
  --hidden-size 4096 \
  --split-k 64 \
  --block-k 256 \
  --block-h 512 \
  --fuse-rmsnorm \
  --prefill-tf32-mma \
  --no-prefill-block-m \
  --warmup 20 \
  --iters 30
```

Five independent process-level medians were collected for each revision:

| Revision | Geometry | Median samples, microseconds | Median |
| --- | --- | --- | ---: |
| Comparison | M16/N8/K256, one K split | 261.25, 263.09, 261.25, 260.02, 259.71 | 261.25 |
| Qualified | M64/N24/K64, eight K splits, two stages | 198.27, 197.97, 196.74, 198.27, 198.27 | 198.27 |

The fixed-work latency reduction is 24.11%. The equivalent throughput ratio is
`261.25 / 198.27 = 1.317648`, or 31.76% higher projection throughput. The
qualified output reported maximum projection error 0.0625 and projection RMSE
0.000887; the comparison reported 0.0625 and 0.000902.

This result qualifies the projection operation. End-to-end serving throughput
depends on attention, index selection, communication, mixture-of-experts, and
scheduler work and is therefore not inferred from this operation benchmark.
