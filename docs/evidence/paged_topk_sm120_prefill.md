# SM120 paged top-k prefill qualification

## Status and operation contract

Status: **qualified** for the paged SM120 top-k selector used by GLM-5.3 C4
prefill.

The selector admits the largest shared-memory candidate capacity supported by
the device, omits the final score write when the caller requests indices only,
and initializes padding candidates before an exact overflow rescan. The public
indexer interfaces, logical-index output contract, page layout, top-k value,
and supported input layouts are unchanged.

For a row with fewer live candidates than `topk`, the result contains every
live candidate at most once and fills remaining entries with `-1`. CUDA graph
replay may change live lengths and logits without retaining a stale padded
winner.

## Source and hardware

- Comparison revision: `fc1d4b68f7a5b0cfdb88bf06abccd869f5c589d5`,
  tree `7fcb6afb69b04b6981060012e943e125e58168b4`.
- Qualified implementation revision:
  `ee611d20258b00a976123d4fead0a218de04a709`, tree
  `20f47518383665eeab46bfca5a6ceda7d117063d`.
- Comparison worktree:
  `/root/vllm/worktrees/b12x-master-fc1d4b68-20260830`.
- Qualified worktree: `/root/vllm/tmp/b12x-pr-paged-topk-20260830`.
- Runtime foundation image:
  `local/vllm:glm53-flash-nvfp4-dcp4-dflash2-flashkda-bt4096-20260830-r4`,
  image ID
  `sha256:0029fb596153966c997e1f09737e3ea7a547bf1c7a251f356e01c681c9f917d6`.
- Physical GPU 1: `GPU-3747d42a-c416-459c-cf88-bc2e84f471b1`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, driver 610.57.04, PCIe Gen5 x16.
- GPU-mode snapshots before and after collection were P1, persistence enabled,
  default compute mode, 2,692 MHz SM clock, 13,365 MHz memory clock, active
  throttle mask `0x0`, 38 C, and 77.88/78.84 W against a 600 W limit.
- Toolchain map: `nvidia-cutlass-dsl==4.6.2` from
  `/opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl`, PyTorch 2.13.0,
  and `/usr/local/cuda/bin/ptxas` build
  `cuda_13.3.r13.3/compiler.38244171_0`; `CUTE_DSL_ARCH=sm_120a`.
- Source-specific compile-cache manifests were fixed for every timed process.
  The comparison cache contained four files with aggregate manifest SHA-256
  `ad17d606754fdf21f11bc4e32be598ea49eb0bcde8b8964e8ee0b228b1716a3d`;
  the qualified cache contained six files with aggregate manifest SHA-256
  `dab33ecfbeb248a2b90b2b6768ddb969acc089a842165bf792dabd267afc8abd`.

## Correctness

The focused suite was:

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B -m pytest -q \
  tests/attention/test_attention_dsa_indexer_api.py \
  tests/attention/test_paged_prefill_topk_long_context.py \
  -k "row_topk or paged_prefill_topk"
```

Result: `15 passed, 30 deselected`. The capacity-cache unit selection produced
`7 passed, 29 deselected` and verifies one capability query for each immutable
device/top-k pair.

A production-geometry analytic run used 4,080 query rows, 64 replicated
indexer heads, an 8,192-token visible cache, top-k 512, and a shared paged
table. It reported `analytic_pass(max_abs=0.000488)`. The indices-only contract
is covered by the GPU suite above because analytic score comparison requires
score materialization.

## Performance

The comparison revision always materializes final scores. The qualified
revision used the serving contract in which sparse attention consumes only
selected indices:

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B benchmarks/benchmark_paged_indexer.py \
  --rows 4080 \
  --global-heads 64 \
  --tp-size 4 \
  --page-table-width 128 \
  --seq-len 8192 \
  --mode supertile-topk \
  --route paged-tiled \
  --topk 512 \
  --indices-only \
  --warmup 10 \
  --iters 30
```

The comparison command omitted `--indices-only` because that revision does not
implement the indices-only contract. Each arm first ran one zero-warmup replay
and one 10-warmup preconditioning process. Five process-level medians per arm
were then collected in the balanced order comparison, qualified, qualified,
comparison, comparison, qualified, qualified, comparison, comparison,
qualified. Every timed process loaded its arm's fixed cache objects before
graph capture.

| Revision | Process medians, microseconds | Median |
| --- | --- | ---: |
| Comparison | 1,074.18, 1,074.18, 1,073.15, 1,073.20, 1,073.86 | 1,073.86 us |
| Qualified | 1,052.50, 1,052.67, 1,052.67, 1,053.57, 1,052.67 | 1,052.67 us |

The zero-warmup replay was 1,073.15 us for the comparison and 1,050.62 us for
the qualified implementation. The preconditioning process medians were
1,074.18 us and 1,052.67 us, respectively.

The fixed-work latency reduction is 1.97%. The ratio direction is comparison
latency divided by qualified latency: `1073.86 / 1052.67 = 1.020130`, or 2.01%
higher selector throughput.

Comparison raw samples, microseconds:

```text
1074.66, 1072.77, 1071.68, 1072.70, 1073.73, 1073.73, 1072.10,
1073.15, 1072.00, 1071.10, 1075.10, 1074.82, 1073.09, 1074.18,
1077.09, 1074.43, 1074.18, 1071.10, 1074.18, 1073.15, 1072.13,
1074.18, 1075.20, 1074.18, 1075.20, 1074.18, 1074.18, 1075.20,
1074.18, 1073.15
```

Qualified raw samples, microseconds:

```text
1050.62, 1050.46, 1050.53, 1050.21, 1050.69, 1052.22, 1051.81,
1052.67, 1051.55, 1051.26, 1051.36, 1050.24, 1050.56, 1051.65,
1052.42, 1054.34, 1052.67, 1054.72, 1053.70, 1052.67, 1053.70,
1052.67, 1053.70, 1053.70, 1052.67, 1053.70, 1052.67, 1053.70,
1054.72, 1052.67
```

The operation result does not by itself establish an end-to-end serving gain.
The release-level benchmark must include page-table preparation, C4 scoring,
communication, attention, and all remaining model work.
