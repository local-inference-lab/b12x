# Qwen3.8 Flash Next paired MXFP8 projection qualification

Status: qualified for NVIDIA RTX PRO 6000 Blackwell Workstation Edition decode
shapes.

The Qwen3.8 Flash Next Gated DeltaNet input path applies two independent MXFP8
linear projections to the same BF16 hidden state. The wide projection produces
16,384 QKVZ values per row and the narrow projection produces 96 BA values per
row; both consume 2,560 input values. `b12x.gemm.mxfp8_linear.mm_pair` overlaps
the narrow projection with the wide projection and joins the streams before
returning either output.

## Qualification contract

- Serial implementation revision: `9ae41c5cb9935d740456479954b0089f80bd2ef2`.
- Paired implementation revision: `f6cf4e953679a84b0d999515821c5b03f162320d`.
- Benchmark: `benchmarks/benchmark_mxfp8_linear_pair.py`.
- Physical GPU: index 2, UUID
  `GPU-167fbc3f-fd06-7f08-9e06-ee02946d041c`, PCI address
  `00000000:23:00.0`.
- Operating mode: stock automatic clocks, without an application clock lock or
  memory overclock.
- Software: CUDA 13.3 and PyTorch 2.13.0.
- Measurement: 20 warmup replays and 100 measured CUDA-graph replays per arm.
  Serial-first and paired-first order alternates between samples.
- Correctness gate: both output tensors must be bit-identical before timing.
- Ratio: serial median milliseconds divided by paired median milliseconds. A
  ratio above 1 means the paired operation is faster.

The command executed inside the container was:

```bash
python benchmarks/benchmark_mxfp8_linear_pair.py \
  --tokens 1 4 \
  --warmups 20 \
  --samples 100 \
  --physical-gpu-index 2 \
  --operating-mode "stock automatic clocks; no application clock lock or memory overclock" \
  --baseline-revision 9ae41c5cb9935d740456479954b0089f80bd2ef2 \
  --output docs/evidence/qwen38_flash_next_mxfp8_projection_pair_rtx6000_20260904.json
```

## Results

| Rows | Serial median | Paired median | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.027424 ms | 0.020752 ms | 1.322x | 24.33% |
| 4 | 0.027680 ms | 0.016720 ms | 1.656x | 39.60% |

The paired operation preserves exact outputs and reduces the complete
projection-pair GPU interval for both Qwen decode row counts. The committed JSON
artifact
`docs/evidence/qwen38_flash_next_mxfp8_projection_pair_rtx6000_20260904.json`
contains all 100 raw timings for both arms and both row counts.

This qualification covers CUDA-graph replay with the stated Qwen dimensions.
It does not claim a benefit for bandwidth-saturating prefill shapes; callers
must select a bounded row policy and retain serial execution above that bound.
