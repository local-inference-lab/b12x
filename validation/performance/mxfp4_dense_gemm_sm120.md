# MXFP4 Dense GEMM Enablement Qualification

Status: **qualified**

The MXFP4 enablement in `DenseGemmKernel` (MMA op dispatch on `sf_dtype` plus
scale-factor fragment K-mode grouping) makes the native
`Float4E2M1FN` + `Float8E8M0FNU` + `sf_vec_size=32` path functional and does
not change the NVF4 or MXFP8 paths: both regression arms produce
**bit-identical outputs** across the two revisions and their medians agree
within noise at locked clocks.

## Conditions

- Before revision: `e68f812f15e6b06420cc649eb9caccfa42d1b9c4` (master),
  worktree `/opt/models/qwen38-nvfp4-workspace/b12x-before`, mounted in the
  container at `/src/b12x-before`; measured `b12x/_lib/dense_gemm.py`
  SHA-256
  `f3ea177103652486c82225dc9b91a8caa718f9345e9289c3bfc9e19e1bf42203`.
- After revision: `d891b26422f7e1777e4cfcf5a3344f5e3cfd54a0`
  (`fix/mxfp4-mma-op-dispatch`), worktree
  `/opt/models/qwen38-nvfp4-workspace/b12x-after`, mounted at
  `/src/b12x-after`; measured `b12x/_lib/dense_gemm.py` SHA-256
  `0d90040f8165de89c8e326b12e59811fb1e0cae489c0ad00ebed70f0d065466d`.
- Benchmark: `benchmarks/benchmark_mxfp4_dense_gemm.py` (added by this
  record's commit on `fix/mxfp4-mma-op-dispatch`), run from the branch
  checkout mounted at `/bench`; the measured source tree is selected with
  `--source` and its `dense_gemm.py` SHA-256 is recorded in the receipt.
- Container: local image `qwen38-quant:latest` (image ID `1eca39d593a2`),
  entrypoint overridden to `/bin/sh`, GPU pinned with
  `--gpus '"device=0"'`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (SM120),
  UUID `GPU-88c8cc9c-c760-f70b-ea74-4598a39f5296`, PCI bus
  `00000000:01:00.0`, 300 W power limit, default compute mode, persistence
  enabled.
- GPU mode during measurement: SM clocks locked with
  `nvidia-smi -i 0 -lgc 2100,2100` for both arms (released with `-rgc`
  afterwards); post-track queries show P1 at 2092 MHz (before arm, throttle
  mask `0x0`) and 2055 MHz (after arm, transient SW power-cap bit `0x4`).
  The receipt stores the full pre- and post-track GPU state per arm.
- Driver: 595.58.03. CUDA runtime: 13.3. PyTorch: 2.13.0.
- Operation: `gemm.blockscaled.mm` captured into a CUDA graph; the first
  replay is recorded separately, then 100 warmup replays, then 1,000
  recorded replays timed individually with CUDA events.
- Operands are built with fixed per-track torch seeds, so both arms consume
  identical inputs and output digests are directly comparable.

## Tracks and shapes

| Track | ab_dtype | sf_dtype | sf_vec_size | Shape (M, N, K) | Role |
|---|---|---|---|---|---|
| `nvfp4` | `float4_e2m1fn` | `float8_e4m3fn` | 16 | 2048, 4096, 4096 | regression arm |
| `mxfp8` | `float8_e4m3fn` | `float8_e8m0fnu` | 32 | 512, 1024, 512 (L=4) | regression arm |
| `mxfp4` | `float4_e2m1fn` | `float8_e8m0fnu` | 32 | 2048, 4096, 4096 | enabled capability |

## Result

| Track | Arm | First replay | Warm median | Warm min | Warm max | Warm p99 | Correctness |
|---|---|---:|---:|---:|---:|---:|---|
| `nvfp4` | before | 83.104 us | 71.008 us | 69.792 us | 73.888 us | 73.024 us | graph == eager bit-exact |
| `nvfp4` | after | 84.864 us | 70.976 us | 69.216 us | 75.840 us | 72.928 us | graph == eager bit-exact |
| `mxfp8` | before | 26.336 us | 12.800 us | 12.480 us | 16.832 us | 13.824 us | graph == eager bit-exact |
| `mxfp8` | after | 27.264 us | 12.832 us | 12.512 us | 17.664 us | 13.888 us | graph == eager bit-exact |
| `mxfp4` | before | — | — | — | — | — | `OpError: expects the 'sf_type' Op parameter to be Float8E4M3FN` |
| `mxfp4` | after | 114.048 us | 98.880 us | 96.736 us | 104.864 us | 101.888 us | 0 mismatches vs dequantized fp32 einsum reference (eager and graph), bit-exact |

Cross-revision output digests (SHA-256 of the raw output tensor bytes):

| Track | Digest (identical in both arms) |
|---|---|
| `nvfp4` | `cbf5f84af84a2659263e625345f91ff574ade59b3a4198ceb576ce956aa17414` |
| `mxfp8` | `4b57140fd9393916ebc24b40db4a6a1a794bc1955ed8bec953cd6c03b8f2256f` |

**Ratio definition**: for each regression track, the recorded ratio is the
*after* warm median divided by the *before* warm median, so a value above one
means the after revision is slower.

| Track | after median / before median |
|---|---:|
| `nvfp4` | 0.9995 |
| `mxfp8` | 1.0025 |

Both ratios are within run-to-run noise at locked clocks; combined with the
bit-identical outputs, the NVF4 and MXFP8 paths are unchanged. The `mxfp4`
track has no before/after ratio: the before revision fails at MMA op
construction (the `OpError` above, raised by
`MmaSM120BlockScaledOp.__post_init__` because `MmaMXF4NVF4Op` hardcodes
`sf_vec_size=16`), so the after arm records the enabled path's absolute
performance and bit-exact correctness instead.

The raw receipts for both arms, including all 1,000 timing samples per track,
full command lines, source-file digests, and pre/post-track GPU state, are in
[`mxfp4_dense_gemm_sm120.json.gz`](mxfp4_dense_gemm_sm120.json.gz).
Its SHA-256 digest is
`a5ede2bf95ecbe70dbe72e5f597040ec5196d0a317a0b69f1d6f00f4ab1289a9`.
The decompressed JSON SHA-256 digest is
`187b87a887b22b6366c44c328243593907a41837e5a2131a639c1fa70aac99a8`.

## Reproduction

Create clean worktrees of both revisions, lock the GPU clocks, and run each
arm from the branch checkout:

```bash
git worktree add /path/b12x-before e68f812f15e6b06420cc649eb9caccfa42d1b9c4
git worktree add /path/b12x-after  d891b26422f7e1777e4cfcf5a3344f5e3cfd54a0
nvidia-smi -i 0 -lgc 2100,2100

docker run --rm --gpus '"device=0"' --entrypoint /bin/sh \
  -v /path/b12x-after:/bench -v /path/b12x-before:/src/b12x-before -v /out:/out \
  <image> -c 'python3 /bench/benchmarks/benchmark_mxfp4_dense_gemm.py \
    --source /src/b12x-before --revision e68f812f15e6b06420cc649eb9caccfa42d1b9c4 \
    --label before --tracks nvfp4,mxfp8,mxfp4 --json-out /out/receipt-before.json'

docker run --rm --gpus '"device=0"' --entrypoint /bin/sh \
  -v /path/b12x-after:/bench -v /path/b12x-after:/src/b12x-after -v /out:/out \
  <image> -c 'python3 /bench/benchmarks/benchmark_mxfp4_dense_gemm.py \
    --source /src/b12x-after --revision d891b26422f7e1777e4cfcf5a3344f5e3cfd54a0 \
    --label after --tracks nvfp4,mxfp8,mxfp4 --json-out /out/receipt-after.json'

nvidia-smi -i 0 -rgc
```
