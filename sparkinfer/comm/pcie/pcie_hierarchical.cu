// Bounded-degree TP12/TP16 all-reduce for four-GPU PCIe islands.
//
// Physical/logical ranks are expected to form three or four contiguous
// islands: 0..3, 4..7, 8..11, and optionally 12..15.
//
// Each non-leader maps only its island leader.  Each leader maps its three
// local peers and the other leaders, so no CUDA context needs more than six
// peer connections.  One kernel performs:
//   local stage -> four-rank reduction -> leader reduction ->
//   local broadcast.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pcie_hierarchical {

constexpr int kMaxWorldSize = 16;
constexpr int kIslandSize = 4;
constexpr int kMaxNumIslands = kMaxWorldSize / kIslandSize;
constexpr int kMaxBlocks = 32;
constexpr int kThreads = 256;
constexpr size_t kAlignment = 256;

struct alignas(128) Flag {
  uint32_t value;
  uint32_t padding[31];
};

struct alignas(256) Header {
  uint32_t generation[kMaxBlocks];
  uint32_t generation_padding[64 - kMaxBlocks];
  Flag local_arrived[kMaxBlocks][kIslandSize];
  Flag leader_ready[kMaxBlocks][kMaxNumIslands];
  Flag leader_consumed[kMaxBlocks][kMaxNumIslands];
  Flag final_ready[kMaxBlocks];
  Flag local_consumed[kMaxBlocks][kIslandSize];
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
  layout.partial =
      align_up(layout.stage + elements * sizeof(nv_bfloat16), kAlignment);
  layout.final =
      align_up(layout.partial + elements * sizeof(float), kAlignment);
  layout.bytes =
      align_up(layout.final + elements * sizeof(nv_bfloat16), kAlignment);
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
__global__ void hierarchical_all_reduce_bf16(
    Runtime runtime,
    const nv_bfloat16* input,
    nv_bfloat16* output,
    int64_t elements) {
  static_assert(NumIslands == 3 || NumIslands == 4);
  const int rank = runtime.rank;
  const int island = rank / kIslandSize;
  const int local_rank = rank % kIslandSize;
  const int leader_rank = island * kIslandSize;
  const bool is_leader = local_rank == 0;
  const int block = blockIdx.x;
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
    if (!is_leader) {
      Header* leader_header = header(runtime.slabs[leader_rank]);
      store_release(
          &leader_header->local_arrived[block][local_rank].value,
          generation);
    }
  }

  if (!is_leader) {
    if (threadIdx.x == 0) {
      wait_for(&self_header->final_ready[block].value, generation);
    }
    __syncthreads();

    const nv_bfloat16* leader_final =
        final_buffer(runtime.slabs[leader_rank], runtime.layout);
    for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                         threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      output[index] = leader_final[index];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      __threadfence_system();
      Header* leader_header = header(runtime.slabs[leader_rank]);
      store_release(
          &leader_header->local_consumed[block][local_rank].value,
          generation);
    }
    return;
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int peer = 1; peer < kIslandSize; ++peer) {
      wait_for(
          &self_header->local_arrived[block][peer].value,
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
    for (int peer = 0; peer < kIslandSize; ++peer) {
      const int peer_rank = leader_rank + peer;
      sum += __bfloat162float(stage(runtime.slabs[peer_rank], runtime.layout)[index]);
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
      const int peer_leader = peer_island * kIslandSize;
      Header* peer_header = header(runtime.slabs[peer_leader]);
      store_release(
          &peer_header->leader_ready[block][island].value,
          generation);
    }
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      if (peer_island == island) {
        continue;
      }
      wait_for(
          &self_header->leader_ready[block][peer_island].value,
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
      const int peer_leader = peer_island * kIslandSize;
      sum += partial(runtime.slabs[peer_leader], runtime.layout)[index];
    }
    const nv_bfloat16 value = __float2bfloat16_rn(sum);
    self_final[index] = value;
    output[index] = value;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();

    // Every leader has consumed every partial before publishing this.
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      if (peer_island == island) {
        continue;
      }
      const int peer_leader = peer_island * kIslandSize;
      Header* peer_header = header(runtime.slabs[peer_leader]);
      store_release(
          &peer_header->leader_consumed[block][island].value,
          generation);
    }

    // Publish the final value to the three local consumers.
#pragma unroll
    for (int peer = 1; peer < kIslandSize; ++peer) {
      Header* peer_header = header(runtime.slabs[leader_rank + peer]);
      store_release(&peer_header->final_ready[block].value, generation);
    }

    // Do not let this leader overwrite either its partial or final staging
    // until all remote consumers from this generation are done.
#pragma unroll
    for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
      if (peer_island == island) {
        continue;
      }
      wait_for(
          &self_header->leader_consumed[block][peer_island].value,
          generation);
    }
#pragma unroll
    for (int peer = 1; peer < kIslandSize; ++peer) {
      wait_for(
          &self_header->local_consumed[block][peer].value,
          generation);
    }
  }
}

}  // namespace pcie_hierarchical

using Runtime = pcie_hierarchical::Runtime;

static int64_t slab_bytes(int64_t max_elements) {
  return static_cast<int64_t>(
      pcie_hierarchical::make_layout(max_elements).bytes);
}

static int64_t init_runtime(
    const std::vector<int64_t>& slab_ptrs,
    int64_t rank,
    int64_t max_elements) {
  const int world_size = static_cast<int>(slab_ptrs.size());
  if (world_size != 12 && world_size != 16) {
    throw std::invalid_argument(
        "hierarchical runtime requires exactly 12 or 16 slab pointers");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::invalid_argument("invalid hierarchical all-reduce rank");
  }
  auto* runtime = new Runtime;
  runtime->rank = static_cast<int>(rank);
  runtime->world_size = world_size;
  runtime->max_elements = max_elements;
  runtime->layout = pcie_hierarchical::make_layout(max_elements);
  for (int index = 0; index < world_size; ++index) {
    runtime->slabs[index] = reinterpret_cast<void*>(slab_ptrs[index]);
  }

  const int island = runtime->rank / pcie_hierarchical::kIslandSize;
  const int local_rank = runtime->rank % pcie_hierarchical::kIslandSize;
  const int leader = island * pcie_hierarchical::kIslandSize;
  if (runtime->slabs[runtime->rank] == nullptr ||
      runtime->slabs[leader] == nullptr) {
    delete runtime;
    throw std::invalid_argument("self and island-leader slabs must be mapped");
  }
  if (local_rank == 0) {
    for (int peer = 0; peer < pcie_hierarchical::kIslandSize; ++peer) {
      if (runtime->slabs[leader + peer] == nullptr) {
        delete runtime;
        throw std::invalid_argument("leader is missing a local peer slab");
      }
    }
    const int num_islands = world_size / pcie_hierarchical::kIslandSize;
    for (int peer_island = 0; peer_island < num_islands; ++peer_island) {
      if (runtime->slabs[peer_island * pcie_hierarchical::kIslandSize] ==
          nullptr) {
        delete runtime;
        throw std::invalid_argument("leader is missing a peer-leader slab");
      }
    }
  }
  return reinterpret_cast<int64_t>(runtime);
}

static void all_reduce_bf16(
    int64_t runtime_handle,
    torch::Tensor input,
    torch::Tensor output,
    int64_t blocks) {
  auto* runtime = reinterpret_cast<Runtime*>(runtime_handle);
  TORCH_CHECK(runtime != nullptr, "runtime is closed");
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          output.scalar_type() == torch::kBFloat16,
      "hierarchical all-reduce currently supports BF16 only");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(input.sizes() == output.sizes(),
              "input and output shapes must match");
  TORCH_CHECK(input.numel() > 0 && input.numel() <= runtime->max_elements,
              "input exceeds runtime capacity");
  TORCH_CHECK(
      blocks == 1 || blocks == 2 || blocks == 4 || blocks == 8 ||
          blocks == 16 || blocks == 32,
      "blocks must be one of 1, 2, 4, 8, 16, 32");

  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* input_ptr =
      reinterpret_cast<const nv_bfloat16*>(input.data_ptr());
  auto* output_ptr = reinterpret_cast<nv_bfloat16*>(output.data_ptr());
  if (runtime->world_size == 12) {
    pcie_hierarchical::hierarchical_all_reduce_bf16<3>
        <<<static_cast<int>(blocks), pcie_hierarchical::kThreads, 0, stream>>>(
            *runtime, input_ptr, output_ptr, input.numel());
  } else if (runtime->world_size == 16) {
    pcie_hierarchical::hierarchical_all_reduce_bf16<4>
        <<<static_cast<int>(blocks), pcie_hierarchical::kThreads, 0, stream>>>(
            *runtime, input_ptr, output_ptr, input.numel());
  } else {
    TORCH_CHECK(false, "unsupported hierarchical all-reduce world size");
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
