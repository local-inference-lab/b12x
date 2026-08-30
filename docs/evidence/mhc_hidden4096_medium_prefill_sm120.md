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
- Raw-sample benchmark instrumentation revision:
  `168ca6bf712a8ef45f0225dfa6dfe5e79731e5bb`.
- Comparison worktree:
  `/root/vllm/worktrees/b12x-master-fc1d4b68-20260830`.
- Qualified worktree: `/root/vllm/tmp/b12x-pr-mhc-medium-20260830`.
- Runtime foundation image:
  `local/vllm:glm53-flash-nvfp4-dcp4-dflash2-flashkda-bt4096-20260830-r4`,
  image ID
  `sha256:0029fb596153966c997e1f09737e3ea7a547bf1c7a251f356e01c681c9f917d6`.
- Physical GPU 0: `GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, driver 610.57.04, PCIe Gen5 x16.
- GPU-mode snapshots before and after collection were P1, persistence enabled,
  default compute mode, 2,692 MHz SM clock, 13,365 MHz memory clock, active
  throttle mask `0x0`, 37 C, and 84.67/84.81 W against a 600 W limit.
- Toolchain map: `nvidia-cutlass-dsl==4.6.2` from
  `/opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl`, PyTorch 2.13.0,
  and `/usr/local/cuda/bin/ptxas` build
  `cuda_13.3.r13.3/compiler.38244171_0`; `CUTE_DSL_ARCH=sm_120a`.
- Source-specific compile-cache manifests were fixed for every timed process.
  The comparison cache contained 12 files with aggregate manifest SHA-256
  `226560cd2787c5ce64420e8971e958d06ad4c7c44e9bda8492a8b60abce974c3`;
  the qualified cache contained six files with aggregate manifest SHA-256
  `f30d8076e47305c7a192bfbfb6141ad6e63e1fd09942a573f55e5938668f7687`.

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
CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ARCH=sm_120a \
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
  --iters 30 \
  --raw-samples
```

Each arm first ran one zero-warmup replay and one 20-warmup preconditioning
process. Five process-level medians per arm were then collected in the balanced
order comparison, qualified, qualified, comparison, comparison, qualified,
qualified, comparison, comparison, qualified. Every timed process loaded its
arm's fixed cache objects before graph capture.

| Revision | Geometry | Median samples, microseconds | Median |
| --- | --- | --- | ---: |
| Comparison | M16/N8/K256, one K split | 261.76, 261.07, 259.71, 259.73, 261.25 | 261.07 |
| Qualified | M64/N24/K64, eight K splits, two stages | 198.27, 198.27, 198.27, 198.27, 198.27 | 198.27 |

The zero-warmup replay was 261.60 us for the comparison and 201.70 us for the
qualified implementation. The preconditioning process medians were 260.53 us
and 197.44 us, respectively.

Representative 30-replay samples from the fixed cached objects, microseconds:

```text
comparison: 268.96,257.95,259.71,260.26,257.15,261.76,265.89,259.71,261.76,261.79,257.66,259.71,259.74,259.71,261.79,261.76,262.78,259.71,261.76,259.74,265.86,259.71,263.81,257.66,258.69,263.84,257.70,259.71,257.66,257.70
qualified: 201.38,198.91,196.22,197.02,195.42,198.27,198.27,198.27,194.18,196.22,196.22,196.22,196.22,196.22,196.22,198.27,194.18,198.27,200.32,200.32,198.27,201.34,196.22,198.27,198.27,198.27,196.22,196.22,198.27,196.22
```

The fixed-work latency reduction is 24.05%. The ratio direction is comparison
latency divided by qualified latency: `261.07 / 198.27 = 1.316740`, or 31.67%
higher projection throughput. The qualified output reported maximum projection
error 0.0625 and projection RMSE 0.000887; the comparison reported 0.0625 and
0.000902.

This result qualifies the projection operation. End-to-end serving throughput
depends on attention, index selection, communication, mixture-of-experts, and
scheduler work and is therefore not inferred from this operation benchmark.
