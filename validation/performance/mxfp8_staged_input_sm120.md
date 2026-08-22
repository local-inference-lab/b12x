# Staged MXFP8 Linear Input on SM120

Status: qualified.

The benchmark at `benchmarks/benchmark_mxfp8_staged_input.py` compares two
ways to construct the input for one MXFP8 linear operation. The concatenated
arm retains six BF16 feature tensors and joins them before quantization. The
staged arm quantizes each feature tensor into caller-owned MXFP8 storage and
writes the GEMM result directly into caller-owned output storage.

The measurement used B12X revision
`4b8662a67afe0c42597e38b53d6d35cdcb14c205` from a clean worktree, PyTorch
2.13.0 with CUDA 13.3, NVIDIA driver 610.57.04, and one NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPU identified as
`GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3`. The operation shape was M=4096,
K=43008, and N=7168, with K supplied as six 7168-column slices. Nine timing
pairs were interleaved, and both arms reused the same prebuilt BF16 inputs;
source generation was excluded from elapsed time.

Both arms produced bitwise-identical BF16 output. Peak allocated memory fell
from 1,278,083,584 bytes to 663,355,904 bytes, a reduction of 614,727,680 bytes
or 48.10%. Median elapsed time was 5.930 ms for concatenation and 5.844 ms for
staging, a 1.45% reduction. Raw samples, physical-GPU operating state, and
runtime metadata are stored in `mxfp8_staged_input_sm120.json`.
