// Bounded-degree TP16 fused BF16 add + global greedy argmax.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES
#define B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES 24
#endif

static_assert(B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES >= 0);

namespace pcie_vocab_argmax {

constexpr int kWorldSize = 16;
constexpr int kIslandSize = 4;
constexpr int kNumIslands = 4;
constexpr int kSlots = 2;
constexpr int kMaxBatch = 8;
constexpr int kThreads = 512;

struct Pair {
  float value;
  int32_t index;
};

struct alignas(128) Flag {
  uint32_t value;
  uint32_t padding[31];
};

struct alignas(256) Header {
  uint32_t generation[kMaxBatch];
  uint32_t generation_padding[64 - kMaxBatch];
  Flag local_ready[kSlots][kMaxBatch][kIslandSize];
  Flag island_ready[kSlots][kMaxBatch][kNumIslands];
  Pair local_pair[kSlots][kMaxBatch];
  Pair island_pair[kSlots][kMaxBatch];
};

static_assert(sizeof(Flag) == 128);
static_assert(sizeof(Header) % 256 == 0);

struct Runtime {
  Header* slabs[kWorldSize] = {};
  int rank;
  int64_t local_elements;
  int max_batch;
};

__device__ __forceinline__ uint32_t load_acquire(const uint32_t* ptr) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(ptr));
  return value;
}

__device__ __forceinline__ void store_release(
    uint32_t* ptr, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(value)
               : "memory");
}

__device__ __forceinline__ void wait_for(
    const uint32_t* ptr, uint32_t generation) {
  while (load_acquire(ptr) < generation) {
#if __CUDA_ARCH__ >= 700 && \
    B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES > 0
    __nanosleep(B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES);
#endif
  }
}

__device__ __forceinline__ Pair better(Pair lhs, Pair rhs) {
  const bool lhs_nan = isnan(lhs.value);
  const bool rhs_nan = isnan(rhs.value);
  if (lhs_nan != rhs_nan) {
    return rhs_nan ? rhs : lhs;
  }
  if (rhs.value > lhs.value ||
      (rhs.value == lhs.value && rhs.index < lhs.index)) {
    return rhs;
  }
  return lhs;
}

__device__ __forceinline__ Pair warp_reduce(Pair value) {
  constexpr unsigned mask = 0xffffffffU;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Pair other{
        __shfl_down_sync(mask, value.value, offset),
        __shfl_down_sync(mask, value.index, offset),
    };
    if ((threadIdx.x & 31) + offset < 32) {
      value = better(value, other);
    }
  }
  return value;
}

__global__ void fused_add_argmax_tp16(
    Runtime runtime,
    const nv_bfloat16* base,
    const nv_bfloat16* bias,
    int64_t base_row_stride,
    int64_t bias_row_stride,
    int64_t* output) {
  __shared__ Pair warp_pairs[32];
  __shared__ Pair block_pair;
  __shared__ uint32_t generation;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const int64_t base_row_offset =
      static_cast<int64_t>(row) * base_row_stride;
  const int64_t bias_row_offset =
      static_cast<int64_t>(row) * bias_row_stride;

  Pair best{-CUDART_INF_F, INT_MAX};
  for (int64_t index = tid; index < runtime.local_elements;
       index += blockDim.x) {
    const float sum = __bfloat162float(base[base_row_offset + index]) +
                      __bfloat162float(bias[bias_row_offset + index]);
    const nv_bfloat16 rounded = __float2bfloat16_rn(sum);
    const Pair candidate{
        __bfloat162float(rounded),
        static_cast<int32_t>(runtime.rank * runtime.local_elements + index),
    };
    best = better(best, candidate);
  }
  best = warp_reduce(best);
  if (lane == 0) {
    warp_pairs[warp] = best;
  }
  __syncthreads();

  if (warp == 0) {
    Pair warp_best = lane < warps
                         ? warp_pairs[lane]
                         : Pair{-CUDART_INF_F, INT_MAX};
    warp_best = warp_reduce(warp_best);
    if (lane == 0) {
      block_pair = warp_best;
    }
  }
  __syncthreads();
  if (tid != 0) {
    return;
  }

  Header* self = runtime.slabs[runtime.rank];
  generation = atomicAdd(&self->generation[row], 1U) + 1U;
  const int slot = static_cast<int>(generation & 1U);
  const int island = runtime.rank / kIslandSize;
  const int local_rank = runtime.rank % kIslandSize;
  const int island_base = island * kIslandSize;

  self->local_pair[slot][row] = block_pair;
  __threadfence_system();
#pragma unroll
  for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
    if (peer_local == local_rank) {
      continue;
    }
    Header* peer = runtime.slabs[island_base + peer_local];
    store_release(
        &peer->local_ready[slot][row][local_rank].value,
        generation);
  }
#pragma unroll
  for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
    if (peer_local == local_rank) {
      continue;
    }
    wait_for(
        &self->local_ready[slot][row][peer_local].value,
        generation);
  }

  Pair island_best{-CUDART_INF_F, INT_MAX};
#pragma unroll
  for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
    island_best = better(
        island_best,
        runtime.slabs[island_base + peer_local]->local_pair[slot][row]);
  }
  self->island_pair[slot][row] = island_best;
  __threadfence_system();

#pragma unroll
  for (int peer_island = 0; peer_island < kNumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    Header* peer =
        runtime.slabs[peer_island * kIslandSize + local_rank];
    store_release(
        &peer->island_ready[slot][row][island].value,
        generation);
  }
#pragma unroll
  for (int peer_island = 0; peer_island < kNumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    wait_for(
        &self->island_ready[slot][row][peer_island].value,
        generation);
  }

  Pair global_best{-CUDART_INF_F, INT_MAX};
#pragma unroll
  for (int peer_island = 0; peer_island < kNumIslands; ++peer_island) {
    global_best = better(
        global_best,
        runtime.slabs[peer_island * kIslandSize + local_rank]
            ->island_pair[slot][row]);
  }
  output[row] = static_cast<int64_t>(global_best.index);
}

}  // namespace pcie_vocab_argmax

using Runtime = pcie_vocab_argmax::Runtime;

static int64_t slab_bytes() {
  return static_cast<int64_t>(sizeof(pcie_vocab_argmax::Header));
}

static int64_t init_runtime(
    const std::vector<int64_t>& slab_ptrs,
    int64_t rank,
    int64_t local_elements,
    int64_t max_batch) {
  if (slab_ptrs.size() != pcie_vocab_argmax::kWorldSize) {
    throw std::invalid_argument("vocabulary argmax requires TP16");
  }
  if (rank < 0 || rank >= pcie_vocab_argmax::kWorldSize ||
      local_elements <= 0 ||
      local_elements * pcie_vocab_argmax::kWorldSize >= (int64_t(1) << 31) ||
      max_batch <= 0 || max_batch > pcie_vocab_argmax::kMaxBatch) {
    throw std::invalid_argument("invalid vocabulary argmax geometry");
  }
  auto runtime = std::make_unique<Runtime>();
  runtime->rank = static_cast<int>(rank);
  runtime->local_elements = local_elements;
  runtime->max_batch = static_cast<int>(max_batch);
  for (int index = 0; index < pcie_vocab_argmax::kWorldSize; ++index) {
    runtime->slabs[index] =
        reinterpret_cast<pcie_vocab_argmax::Header*>(slab_ptrs[index]);
  }

  const int island = runtime->rank / pcie_vocab_argmax::kIslandSize;
  const int local_rank = runtime->rank % pcie_vocab_argmax::kIslandSize;
  for (int peer_local = 0; peer_local < pcie_vocab_argmax::kIslandSize;
       ++peer_local) {
    const int peer_rank = island * pcie_vocab_argmax::kIslandSize + peer_local;
    if (runtime->slabs[peer_rank] == nullptr) {
      throw std::invalid_argument("missing local-island slab");
    }
  }
  for (int peer_island = 0; peer_island < pcie_vocab_argmax::kNumIslands;
       ++peer_island) {
    const int peer_rank =
        peer_island * pcie_vocab_argmax::kIslandSize + local_rank;
    if (runtime->slabs[peer_rank] == nullptr) {
      throw std::invalid_argument("missing cross-island lane slab");
    }
  }
  return reinterpret_cast<int64_t>(runtime.release());
}

static void fused_add_argmax(
    int64_t runtime_handle,
    torch::Tensor base,
    torch::Tensor bias,
    torch::Tensor output) {
  auto* runtime = reinterpret_cast<Runtime*>(runtime_handle);
  TORCH_CHECK(runtime != nullptr, "vocabulary argmax runtime is closed");
  TORCH_CHECK(base.is_cuda() && bias.is_cuda() && output.is_cuda(),
              "tensors must be CUDA");
  TORCH_CHECK(base.scalar_type() == torch::kBFloat16 &&
                  bias.scalar_type() == torch::kBFloat16,
              "base and bias must be BF16");
  TORCH_CHECK(output.scalar_type() == torch::kInt64,
              "output must be int64");
  TORCH_CHECK(base.device() == bias.device() &&
                  base.device() == output.device(),
              "tensors must be on the same CUDA device");
  TORCH_CHECK(base.dim() == 2 && bias.dim() == 2,
              "base and bias must be two-dimensional");
  TORCH_CHECK(base.stride(1) == 1 && bias.stride(1) == 1,
              "input last dimensions must be contiguous");
  TORCH_CHECK(base.stride(0) > 0 && bias.stride(0) > 0,
              "input row strides must be positive");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(base.sizes() == bias.sizes(),
              "base and bias shapes must match");
  TORCH_CHECK(base.dim() == 2 && base.size(1) == runtime->local_elements,
              "inputs must be [batch, local_vocab]");
  TORCH_CHECK(base.size(0) > 0 && base.size(0) <= runtime->max_batch,
              "batch exceeds vocabulary argmax capacity");
  TORCH_CHECK(output.dim() == 1 && output.size(0) == base.size(0),
              "output must be [batch]");

  const c10::cuda::CUDAGuard guard(base.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  pcie_vocab_argmax::fused_add_argmax_tp16
      <<<static_cast<int>(base.size(0)), pcie_vocab_argmax::kThreads, 0, stream>>>(
          *runtime,
          reinterpret_cast<const nv_bfloat16*>(base.data_ptr()),
          reinterpret_cast<const nv_bfloat16*>(bias.data_ptr()),
          base.stride(0),
          bias.stride(0),
          reinterpret_cast<int64_t*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void destroy_runtime(int64_t runtime_handle) {
  delete reinterpret_cast<Runtime*>(runtime_handle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("slab_bytes", &slab_bytes);
  module.def("init_runtime", &init_runtime);
  module.def("fused_add_argmax", &fused_add_argmax);
  module.def("destroy_runtime", &destroy_runtime);
}
