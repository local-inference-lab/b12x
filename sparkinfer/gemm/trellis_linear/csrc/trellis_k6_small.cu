// SPDX-License-Identifier: MIT
//
// Narrow K6/MCG dense launcher adapted from ExLlamaV3. The vendored kernel
// headers retain their MIT license in vendor/LICENSE.exllamav3. SparkInfer
// owns the tensor validation, workspace, dispatch, and CUDA-stream contract.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

#include "vendor/util.h"
#include "vendor/util.cuh"

namespace cg = cooperative_groups;

#include "vendor/quant/exl3_gemm_kernel.cuh"

namespace {

constexpr int kBits = 6;
constexpr int kCodebookMcg = 1;
constexpr int kTileM = 16;
constexpr int kTileK = 32;
constexpr int kTileN = 128;
constexpr int kSharedStages = 4;
constexpr int kFragmentStages = 3;
constexpr int kThreads = 512;
constexpr int kDynamicSmem = 90 * 1024;

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

void launch_k6_mcg(
    const torch::Tensor& input,
    const torch::Tensor& trellis,
    torch::Tensor& output,
    const torch::Tensor& suh,
    torch::Tensor& rotated_input,
    const torch::Tensor& svh,
    torch::Tensor& locks,
    int64_t requested_sms) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  TORCH_CHECK(input.is_cuda() && trellis.is_cuda() && output.is_cuda(),
              "Trellis K6 tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf,
              "Trellis K6 small-M path requires FP16 input and output");
  TORCH_CHECK(trellis.scalar_type() == at::kShort,
              "Trellis K6 payload must be viewed as int16");
  TORCH_CHECK(suh.scalar_type() == at::kHalf && svh.scalar_type() == at::kHalf,
              "Trellis K6 rotation scales must be FP16");
  TORCH_CHECK(rotated_input.scalar_type() == at::kHalf,
              "Trellis K6 rotated input must be FP16");
  TORCH_CHECK(locks.scalar_type() == at::kInt,
              "Trellis K6 lock workspace must be int32");
  TORCH_CHECK(input.is_contiguous() && trellis.is_contiguous() &&
                  output.is_contiguous() && suh.is_contiguous() &&
                  rotated_input.is_contiguous() && svh.is_contiguous() &&
                  locks.is_contiguous(),
              "Trellis K6 small-M tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && trellis.dim() == 3,
              "Trellis K6 expects A[M,K], B[K/16,N/16,96], C[M,N]");

  int size_m = static_cast<int>(input.size(0));
  int size_k = static_cast<int>(input.size(1));
  int size_n = static_cast<int>(output.size(1));
  TORCH_CHECK(size_m >= 1 && size_m <= 128,
              "Trellis K6 small-M path supports 1..128 rows, got ", size_m);
  TORCH_CHECK(output.size(0) == size_m && rotated_input.sizes() == input.sizes(),
              "Trellis K6 output/scratch shapes do not match input");
  TORCH_CHECK(trellis.size(0) * 16 == size_k &&
                  trellis.size(1) * 16 == size_n && trellis.size(2) == 96,
              "Trellis K6 native payload shape does not match M/K/N");
  TORCH_CHECK(suh.numel() == size_k && svh.numel() == size_n,
              "Trellis K6 rotation scale width mismatch");
  TORCH_CHECK(size_k % kTileK == 0 && size_n % kTileN == 0,
              "Trellis K6 K/N must be divisible by 32/128");
  TORCH_CHECK(locks.numel() >= size_n / 16,
              "Trellis K6 lock workspace is too small");

  int device = input.get_device();
  int available_sms = 0;
  check_cuda(cudaDeviceGetAttribute(&available_sms, cudaDevAttrMultiProcessorCount,
                                    device),
             "cudaDeviceGetAttribute");
  const int tiles = (size_k / kTileK) * (size_n / kTileN);
  int num_sms = requested_sms > 0 ? static_cast<int>(requested_sms) : available_sms;
  num_sms = std::max(1, std::min(num_sms, std::min(available_sms, tiles)));

  auto kernel = exl3_gemm_kernel<kBits, false, kCodebookMcg, kTileM, kTileK,
                                 kTileN, kSharedStages, kFragmentStages>;
  check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmem),
             "cudaFuncSetAttribute");

  const half* a_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const uint16_t* b_ptr =
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>());
  half* c_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  int* lock_ptr = locks.data_ptr<int>();
  const half* suh_ptr = reinterpret_cast<const half*>(suh.data_ptr<at::Half>());
  half* rotated_ptr =
      reinterpret_cast<half*>(rotated_input.data_ptr<at::Half>());
  const half* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  void* args[] = {&a_ptr,       &b_ptr,      &c_ptr,  &size_m, &size_k,
                  &size_n,     &lock_ptr,   &suh_ptr, &rotated_ptr, &svh_ptr};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  check_cuda(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), num_sms,
                                         kThreads, args, kDynamicSmem, stream),
             "cudaLaunchCooperativeKernel");
  check_cuda(cudaPeekAtLastError(), "Trellis K6 kernel launch");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("launch_k6_mcg", &launch_k6_mcg,
             "SparkInfer K6/MCG small-M dense GEMM");
}
