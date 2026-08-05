// Model-free TP12/TP16 lane-distributed hierarchical BF16 all-reduce POC.
//
// Every rank maps its three local-island peers plus the same local lane in
// every other island (five peers at TP12, six at TP16).  Block ownership is
// striped over the four local lanes.  The owner performs the established
// rank-ordered four-way island sum, combines corresponding island partials in
// island order, then publishes its final block to the other local lanes.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace lane_allreduce_poc {

constexpr int kMaxWorldSize = 16;
constexpr int kIslandSize = 4;
constexpr int kMaxNumIslands = 4;
constexpr int kMaxBlocks = 32;
constexpr int kThreads = 256;
constexpr size_t kAlignment = 256;

struct alignas(128) Flag {
  uint32_t value;
  uint32_t padding[31];
};

struct alignas(256) Header {
  uint32_t generation[kMaxBlocks];
  uint32_t collective_generation;
  uint32_t generation_padding[63 - kMaxBlocks];
  Flag stage_ready[kMaxBlocks][kIslandSize];
  Flag partial_ready[kMaxBlocks][kMaxNumIslands];
  Flag final_ready[kMaxBlocks];
  Flag consumed[kMaxWorldSize];
};

static_assert(sizeof(Flag) == 128);
static_assert(sizeof(Header) % kAlignment == 0);

constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct Layout {
  size_t stage;
  size_t partial;
  size_t final;
  size_t bytes;
};

static Layout make_layout(int64_t max_elements) {
  if (max_elements <= 0) {
    throw std::invalid_argument("max_elements must be positive");
  }
  const size_t elements = static_cast<size_t>(max_elements);
  Layout layout;
  layout.stage = align_up(sizeof(Header), kAlignment);
  layout.partial = align_up(
      layout.stage + elements * sizeof(nv_bfloat16), kAlignment);
  layout.final = align_up(
      layout.partial + elements * sizeof(float), kAlignment);
  layout.bytes = align_up(
      layout.final + elements * sizeof(nv_bfloat16), kAlignment);
  return layout;
}

struct Runtime {
  void* slabs[kMaxWorldSize] = {};
  int rank;
  int world_size;
  int64_t max_elements;
  Layout layout;
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
#if __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
  }
}

__device__ __forceinline__ Header* header(void* slab) {
  return reinterpret_cast<Header*>(slab);
}

__device__ __forceinline__ nv_bfloat16* stage(
    void* slab, const Layout& layout) {
  return reinterpret_cast<nv_bfloat16*>(
      reinterpret_cast<char*>(slab) + layout.stage);
}

__device__ __forceinline__ float* partial(
    void* slab, const Layout& layout) {
  return reinterpret_cast<float*>(
      reinterpret_cast<char*>(slab) + layout.partial);
}

__device__ __forceinline__ nv_bfloat16* final_buffer(
    void* slab, const Layout& layout) {
  return reinterpret_cast<nv_bfloat16*>(
      reinterpret_cast<char*>(slab) + layout.final);
}

template <int NumIslands>
__global__ void consumption_preamble(Runtime runtime) {
  static_assert(NumIslands == 3 || NumIslands == 4);
  if (threadIdx.x != 0) {
    return;
  }
  const int rank = runtime.rank;
  const int island = rank / kIslandSize;
  const int local_rank = rank % kIslandSize;
  Header* self_header = header(runtime.slabs[rank]);
  const uint32_t generation =
      atomicAdd(&self_header->collective_generation, 1U) + 1U;

#pragma unroll
  for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
    if (peer_local == local_rank) {
      continue;
    }
    const int peer_rank = island * kIslandSize + peer_local;
    Header* peer_header = header(runtime.slabs[peer_rank]);
    store_release(&peer_header->consumed[rank].value, generation);
  }
#pragma unroll
  for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    const int peer_rank = peer_island * kIslandSize + local_rank;
    Header* peer_header = header(runtime.slabs[peer_rank]);
    store_release(&peer_header->consumed[rank].value, generation);
  }

#pragma unroll
  for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
    if (peer_local == local_rank) {
      continue;
    }
    const int peer_rank = island * kIslandSize + peer_local;
    wait_for(&self_header->consumed[peer_rank].value, generation);
  }
#pragma unroll
  for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    const int peer_rank = peer_island * kIslandSize + local_rank;
    wait_for(&self_header->consumed[peer_rank].value, generation);
  }
}

template <int NumIslands>
__global__ void lane_all_reduce_bf16(
    Runtime runtime,
    const nv_bfloat16* input,
    nv_bfloat16* output,
    int64_t elements) {
  static_assert(NumIslands == 3 || NumIslands == 4);
  const int rank = runtime.rank;
  const int island = rank / kIslandSize;
  const int local_rank = rank % kIslandSize;
  const int island_base = island * kIslandSize;
  const int block = blockIdx.x;
  const int owner_lane = block & (kIslandSize - 1);
  const bool is_owner = local_rank == owner_lane;
  void* self_slab = runtime.slabs[rank];
  Header* self_header = header(self_slab);

  __shared__ uint32_t generation;
  if (threadIdx.x == 0) {
    generation = atomicAdd(&self_header->generation[block], 1U) + 1U;
  }
  __syncthreads();

  nv_bfloat16* self_stage = stage(self_slab, runtime.layout);
  for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                       threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    self_stage[index] = input[index];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();
#pragma unroll
    for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
      if (peer_local == local_rank) {
        continue;
      }
      Header* peer_header =
          header(runtime.slabs[island_base + peer_local]);
      store_release(
          &peer_header->stage_ready[block][local_rank].value,
          generation);
    }
  }

  if (!is_owner) {
    if (threadIdx.x == 0) {
      wait_for(&self_header->final_ready[block].value, generation);
    }
    __syncthreads();
    const int owner_rank = island_base + owner_lane;
    const nv_bfloat16* owner_final =
        final_buffer(runtime.slabs[owner_rank], runtime.layout);
    for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                         threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      output[index] = owner_final[index];
    }
    return;
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
      if (peer_local == local_rank) {
        continue;
      }
      wait_for(
          &self_header->stage_ready[block][peer_local].value,
          generation);
    }
  }
  __syncthreads();

  float* self_partial = partial(self_slab, runtime.layout);
  for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                       threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float sum = 0.0F;
#pragma unroll
    for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
      const int peer_rank = island_base + peer_local;
      sum += __bfloat162float(
          stage(runtime.slabs[peer_rank], runtime.layout)[index]);
    }
    self_partial[index] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      if (peer_island == island) {
        continue;
      }
      const int peer_rank = peer_island * kIslandSize + local_rank;
      Header* peer_header = header(runtime.slabs[peer_rank]);
      store_release(
          &peer_header->partial_ready[block][island].value,
          generation);
    }
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      if (peer_island == island) {
        continue;
      }
      wait_for(
          &self_header->partial_ready[block][peer_island].value,
          generation);
    }
  }
  __syncthreads();

  nv_bfloat16* self_final = final_buffer(self_slab, runtime.layout);
  for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                       threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float sum = 0.0F;
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      const int peer_rank = peer_island * kIslandSize + local_rank;
      sum += partial(runtime.slabs[peer_rank], runtime.layout)[index];
    }
    const nv_bfloat16 value = __float2bfloat16_rn(sum);
    self_final[index] = value;
    output[index] = value;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();
#pragma unroll
    for (int peer_local = 0; peer_local < kIslandSize; ++peer_local) {
      if (peer_local == local_rank) {
        continue;
      }
      Header* peer_header =
          header(runtime.slabs[island_base + peer_local]);
      store_release(&peer_header->final_ready[block].value, generation);
    }
  }
}

}  // namespace lane_allreduce_poc

using Runtime = lane_allreduce_poc::Runtime;

static int64_t slab_bytes(int64_t max_elements) {
  return static_cast<int64_t>(
      lane_allreduce_poc::make_layout(max_elements).bytes);
}

static int64_t init_runtime(
    const std::vector<int64_t>& slab_ptrs,
    int64_t rank,
    int64_t max_elements) {
  const int world_size = static_cast<int>(slab_ptrs.size());
  if (world_size != 12 && world_size != 16) {
    throw std::invalid_argument("lane POC requires TP12 or TP16");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::invalid_argument("invalid lane POC rank");
  }
  auto runtime = std::make_unique<Runtime>();
  runtime->rank = static_cast<int>(rank);
  runtime->world_size = world_size;
  runtime->max_elements = max_elements;
  runtime->layout = lane_allreduce_poc::make_layout(max_elements);
  for (int index = 0; index < world_size; ++index) {
    runtime->slabs[index] = reinterpret_cast<void*>(slab_ptrs[index]);
  }

  const int island = runtime->rank / lane_allreduce_poc::kIslandSize;
  const int local_rank = runtime->rank % lane_allreduce_poc::kIslandSize;
  const int num_islands = world_size / lane_allreduce_poc::kIslandSize;
  if (runtime->slabs[runtime->rank] == nullptr) {
    throw std::invalid_argument("lane POC self slab is missing");
  }
  for (int peer_local = 0; peer_local < lane_allreduce_poc::kIslandSize;
       ++peer_local) {
    const int peer_rank = island * lane_allreduce_poc::kIslandSize + peer_local;
    if (runtime->slabs[peer_rank] == nullptr) {
      throw std::invalid_argument("lane POC local peer slab is missing");
    }
  }
  for (int peer_island = 0; peer_island < num_islands; ++peer_island) {
    const int peer_rank =
        peer_island * lane_allreduce_poc::kIslandSize + local_rank;
    if (runtime->slabs[peer_rank] == nullptr) {
      throw std::invalid_argument("lane POC cross-island peer slab is missing");
    }
  }
  return reinterpret_cast<int64_t>(runtime.release());
}

static void all_reduce_bf16(
    int64_t runtime_handle,
    torch::Tensor input,
    torch::Tensor output,
    int64_t blocks) {
  auto* runtime = reinterpret_cast<Runtime*>(runtime_handle);
  TORCH_CHECK(runtime != nullptr, "lane POC runtime is closed");
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          output.scalar_type() == torch::kBFloat16,
      "lane POC supports BF16 only");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(input.sizes() == output.sizes(),
              "input and output shapes must match");
  TORCH_CHECK(input.numel() > 0 && input.numel() <= runtime->max_elements,
              "input exceeds lane POC capacity");
  TORCH_CHECK(blocks == 16 || blocks == 32,
              "lane POC supports the K3 16/32-block grids only");

  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* input_ptr =
      reinterpret_cast<const nv_bfloat16*>(input.data_ptr());
  auto* output_ptr = reinterpret_cast<nv_bfloat16*>(output.data_ptr());
  if (runtime->world_size == 12) {
    lane_allreduce_poc::consumption_preamble<3>
        <<<1, 32, 0, stream>>>(*runtime);
    lane_allreduce_poc::lane_all_reduce_bf16<3>
        <<<static_cast<int>(blocks), lane_allreduce_poc::kThreads, 0, stream>>>(
            *runtime, input_ptr, output_ptr, input.numel());
  } else {
    lane_allreduce_poc::consumption_preamble<4>
        <<<1, 32, 0, stream>>>(*runtime);
    lane_allreduce_poc::lane_all_reduce_bf16<4>
        <<<static_cast<int>(blocks), lane_allreduce_poc::kThreads, 0, stream>>>(
            *runtime, input_ptr, output_ptr, input.numel());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void destroy_runtime(int64_t runtime_handle) {
  delete reinterpret_cast<Runtime*>(runtime_handle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("slab_bytes", &slab_bytes);
  module.def("init_runtime", &init_runtime);
  module.def("all_reduce_bf16", &all_reduce_bf16);
  module.def("destroy_runtime", &destroy_runtime);
}
