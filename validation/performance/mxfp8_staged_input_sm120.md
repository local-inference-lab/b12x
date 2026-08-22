# Staged MXFP8 Linear Input on SM120

Status: qualified.

The benchmark at `benchmarks/benchmark_mxfp8_staged_input.py` compares two
ways to construct the input for one MXFP8 linear operation. The concatenated
arm retains six BF16 feature tensors and joins them before quantization. The
staged arm quantizes each feature tensor into caller-owned MXFP8 storage and
writes the GEMM result directly into caller-owned output storage.

The measurement used B12X revision
`a2d6f82bedb5a7b8b935823c6cdfb6acce09caf5` from a clean worktree, PyTorch
2.13.0 with CUDA 13.3, NVIDIA driver 610.57.04, and one NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPU. The operation shape was M=4096, K=43008,
and N=7168, with K supplied as six 7168-column slices.

Both arms produced bitwise-identical BF16 output. Peak allocated memory fell
from 1,395,524,096 bytes to 808,321,536 bytes, a reduction of 587,202,560 bytes
or 42.08%. Median elapsed time across nine measured executions was 5.926 ms for
concatenation and 5.934 ms for staging, a 0.14% difference. Raw samples and
runtime metadata are stored in `mxfp8_staged_input_sm120.json`.
