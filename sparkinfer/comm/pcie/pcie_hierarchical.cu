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

#ifndef SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES
#define SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES 24
#endif

#ifndef SPARKINFER_PCIE_HIERARCHICAL_THREADS
#define SPARKINFER_PCIE_HIERARCHICAL_THREADS 224
#endif

#ifndef SPARKINFER_PCIE_HIERARCHICAL_BF16X2
#define SPARKINFER_PCIE_HIERARCHICAL_BF16X2 1
#endif

#ifndef SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS
#define SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS 7168
#endif

static_assert(SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES >= 0);
static_assert(SPARKINFER_PCIE_HIERARCHICAL_THREADS >= 32);
static_assert(SPARKINFER_PCIE_HIERARCHICAL_THREADS <= 1024);
static_assert(SPARKINFER_PCIE_HIERARCHICAL_THREADS % 32 == 0);
static_assert(
    SPARKINFER_PCIE_HIERARCHICAL_BF16X2 == 0 ||
    SPARKINFER_PCIE_HIERARCHICAL_BF16X2 == 1);
static_assert(SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS >= 0);

namespace pcie_hierarchical {

constexpr int kMaxWorldSize = 16;
constexpr int kIslandSize = 4;
constexpr int kMaxNumIslands = kMaxWorldSize / kIslandSize;
constexpr int kMaxBlocks = 32;
constexpr bool kEnableVectorizedBF16x2 =
    SPARKINFER_PCIE_HIERARCHICAL_BF16X2 != 0;
constexpr int64_t kVectorizedBF16x2MaxElements =
    SPARKINFER_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS;
constexpr int kVectorThreads = 112;
constexpr int kScalarThreads = SPARKINFER_PCIE_HIERARCHICAL_THREADS;
constexpr size_t kAlignment = 256;

struct alignas(128) Flag {
  uint32_t value;
  uint32_t padding[31];
};

struct alignas(256) Header {
  uint32_t generation[kMaxBlocks];
  uint32_t collective_generation;
  uint32_t generation_padding[63 - kMaxBlocks];
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
  size_t stage[2];
  size_t partial[2];
  size_t final[2];
  size_t bytes;
};

static Layout make_layout(int64_t max_elements) {
  if (max_elements <= 0) {
    throw std::invalid_argument("max_elements must be positive");
  }
  const size_t elements = static_cast<size_t>(max_elements);
  Layout layout;
  size_t offset = align_up(sizeof(Header), kAlignment);
  for (int slot = 0; slot < 2; ++slot) {
    layout.stage[slot] = offset;
    layout.partial[slot] = align_up(
        layout.stage[slot] + elements * sizeof(nv_bfloat16), kAlignment);
    layout.final[slot] = align_up(
        layout.partial[slot] + elements * sizeof(float), kAlignment);
    offset = align_up(
        layout.final[slot] + elements * sizeof(nv_bfloat16), kAlignment);
  }
  layout.bytes = offset;
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
#if __CUDA_ARCH__ >= 700 && \
    SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES > 0
    __nanosleep(SPARKINFER_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES);
#endif
  }
}

__device__ __forceinline__ Header* header(void* slab) {
  return reinterpret_cast<Header*>(slab);
}

__device__ __forceinline__ nv_bfloat16* stage(
    void* slab, const Layout& layout, int slot) {
  return reinterpret_cast<nv_bfloat16*>(
      reinterpret_cast<char*>(slab) + layout.stage[slot]);
}

__device__ __forceinline__ float* partial(
    void* slab, const Layout& layout, int slot) {
  return reinterpret_cast<float*>(
      reinterpret_cast<char*>(slab) + layout.partial[slot]);
}

__device__ __forceinline__ nv_bfloat16* final_buffer(
    void* slab, const Layout& layout, int slot) {
  return reinterpret_cast<nv_bfloat16*>(
      reinterpret_cast<char*>(slab) + layout.final[slot]);
}

template <int NumIslands>
__global__ void deferred_consumption_preamble(Runtime runtime) {
  static_assert(NumIslands == 3 || NumIslands == 4);
  const int rank = runtime.rank;
  const int island = rank / kIslandSize;
  const int local_rank = rank % kIslandSize;
  const int leader_rank = island * kIslandSize;
  const bool is_leader = local_rank == 0;
  Header* self_header = header(runtime.slabs[rank]);

  // The preamble is ordered after the preceding all-reduce on this rank's
  // stream. Reaching it therefore credits every read of the preceding
  // collective, independent of that collective's grid size. Keeping this
  // handshake outside the data kernel is important for K3, whose decode
  // sequence alternates 32-block 7168-element and 16-block 3584-element
  // collectives: a per-block credit does not describe the same byte ranges
  // when the grid changes.
  if (threadIdx.x != 0) {
    return;
  }
  const uint32_t generation =
      atomicAdd(&self_header->collective_generation, 1U) + 1U;
  if (!is_leader) {
    Header* leader_header = header(runtime.slabs[leader_rank]);
    store_release(
        &leader_header->local_consumed[0][local_rank].value,
        generation);
    return;
  }

#pragma unroll
  for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    const int peer_leader = peer_island * kIslandSize;
    Header* peer_header = header(runtime.slabs[peer_leader]);
    store_release(
        &peer_header->leader_consumed[0][island].value,
        generation);
  }
#pragma unroll
  for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
    if (peer_island == island) {
      continue;
    }
    wait_for(
        &self_header->leader_consumed[0][peer_island].value,
        generation);
  }
#pragma unroll
  for (int peer = 1; peer < kIslandSize; ++peer) {
    wait_for(
        &self_header->local_consumed[0][peer].value,
        generation);
  }
}

template <
    int NumIslands,
    bool DoubleBuffered,
    bool DeferredConsumption,
    bool VectorizedBF16x2>
__global__ void hierarchical_all_reduce_bf16(
    Runtime runtime,
    const nv_bfloat16* input,
    nv_bfloat16* output,
    int64_t elements) {
  static_assert(NumIslands == 3 || NumIslands == 4);
  static_assert(!(DoubleBuffered && DeferredConsumption));
  const int rank = runtime.rank;
  const int island = rank / kIslandSize;
  const int local_rank = rank % kIslandSize;
  const int leader_rank = island * kIslandSize;
  const bool is_leader = local_rank == 0;
  const int block = blockIdx.x;
  void* self_slab = runtime.slabs[rank];
  Header* self_header = header(self_slab);

  uint32_t generation = 0;
  __shared__ uint32_t shared_generation;
  if (threadIdx.x == 0) {
    generation = atomicAdd(&self_header->generation[block], 1U) + 1U;
    if constexpr (DoubleBuffered) {
      shared_generation = generation;
    }
  }
  if constexpr (DoubleBuffered) {
    __syncthreads();
    generation = shared_generation;
  }
  const int slot = DoubleBuffered ? int(generation & 1U) : 0;

  nv_bfloat16* self_stage = stage(self_slab, runtime.layout, slot);
  if constexpr (VectorizedBF16x2) {
    const int64_t pairs = elements / 2;
    const auto* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
    auto* self_stage2 = reinterpret_cast<__nv_bfloat162*>(self_stage);
    for (int64_t pair = static_cast<int64_t>(block) * blockDim.x +
                        threadIdx.x;
         pair < pairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      self_stage2[pair] = input2[pair];
    }
    if ((elements & 1) != 0 && block == 0 && threadIdx.x == 0) {
      self_stage[elements - 1] = input[elements - 1];
    }
  } else {
    for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                         threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      self_stage[index] = input[index];
    }
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
        final_buffer(runtime.slabs[leader_rank], runtime.layout, slot);
    if constexpr (VectorizedBF16x2) {
      const int64_t pairs = elements / 2;
      const auto* leader_final2 =
          reinterpret_cast<const __nv_bfloat162*>(leader_final);
      auto* output2 = reinterpret_cast<__nv_bfloat162*>(output);
      for (int64_t pair = static_cast<int64_t>(block) * blockDim.x +
                          threadIdx.x;
           pair < pairs;
           pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        output2[pair] = leader_final2[pair];
      }
      if ((elements & 1) != 0 && block == 0 && threadIdx.x == 0) {
        output[elements - 1] = leader_final[elements - 1];
      }
    } else {
      for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                           threadIdx.x;
           index < elements;
           index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        output[index] = leader_final[index];
      }
    }
    if constexpr (!DoubleBuffered && !DeferredConsumption) {
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence_system();
        Header* leader_header = header(runtime.slabs[leader_rank]);
        store_release(
            &leader_header->local_consumed[block][local_rank].value,
            generation);
      }
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

  float* self_partial = partial(self_slab, runtime.layout, slot);
  if constexpr (VectorizedBF16x2) {
    const int64_t pairs = elements / 2;
    auto* self_partial2 = reinterpret_cast<float2*>(self_partial);
    for (int64_t pair = static_cast<int64_t>(block) * blockDim.x +
                        threadIdx.x;
         pair < pairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      float2 sum = make_float2(0.0F, 0.0F);
#pragma unroll
      for (int peer = 0; peer < kIslandSize; ++peer) {
        const int peer_rank = leader_rank + peer;
        const auto* peer_stage2 = reinterpret_cast<const __nv_bfloat162*>(
            stage(runtime.slabs[peer_rank], runtime.layout, slot));
        const float2 value = __bfloat1622float2(peer_stage2[pair]);
        sum.x += value.x;
        sum.y += value.y;
      }
      self_partial2[pair] = sum;
    }
    if ((elements & 1) != 0 && block == 0 && threadIdx.x == 0) {
      float sum = 0.0F;
#pragma unroll
      for (int peer = 0; peer < kIslandSize; ++peer) {
        const int peer_rank = leader_rank + peer;
        sum += __bfloat162float(
            stage(runtime.slabs[peer_rank], runtime.layout, slot)[elements - 1]);
      }
      self_partial[elements - 1] = sum;
    }
  } else {
    for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                         threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      float sum = 0.0F;
#pragma unroll
      for (int peer = 0; peer < kIslandSize; ++peer) {
        const int peer_rank = leader_rank + peer;
        sum += __bfloat162float(
            stage(runtime.slabs[peer_rank], runtime.layout, slot)[index]);
      }
      self_partial[index] = sum;
    }
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

  nv_bfloat16* self_final = final_buffer(self_slab, runtime.layout, slot);
  if constexpr (VectorizedBF16x2) {
    const int64_t pairs = elements / 2;
    auto* self_final2 = reinterpret_cast<__nv_bfloat162*>(self_final);
    auto* output2 = reinterpret_cast<__nv_bfloat162*>(output);
    for (int64_t pair = static_cast<int64_t>(block) * blockDim.x +
                        threadIdx.x;
         pair < pairs;
         pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      float2 sum = make_float2(0.0F, 0.0F);
#pragma unroll
      for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
        const int peer_leader = peer_island * kIslandSize;
        const auto* peer_partial2 = reinterpret_cast<const float2*>(
            partial(runtime.slabs[peer_leader], runtime.layout, slot));
        const float2 value = peer_partial2[pair];
        sum.x += value.x;
        sum.y += value.y;
      }
      const __nv_bfloat162 value = __float22bfloat162_rn(sum);
      self_final2[pair] = value;
      output2[pair] = value;
    }
    if ((elements & 1) != 0 && block == 0 && threadIdx.x == 0) {
      float sum = 0.0F;
#pragma unroll
      for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
        const int peer_leader = peer_island * kIslandSize;
        sum += partial(runtime.slabs[peer_leader], runtime.layout, slot)[
            elements - 1];
      }
      const nv_bfloat16 value = __float2bfloat16_rn(sum);
      self_final[elements - 1] = value;
      output[elements - 1] = value;
    }
  } else {
    for (int64_t index = static_cast<int64_t>(block) * blockDim.x +
                         threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      float sum = 0.0F;
#pragma unroll
      for (int peer_island = 0; peer_island < NumIslands; ++peer_island) {
        const int peer_leader = peer_island * kIslandSize;
        sum += partial(runtime.slabs[peer_leader], runtime.layout, slot)[index];
      }
      const nv_bfloat16 value = __float2bfloat16_rn(sum);
      self_final[index] = value;
      output[index] = value;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();

    // The legacy single-buffer mode needs explicit consumption handshakes.
    // With two slots, the next collective writes the other slot. A rank cannot
    // advance to the following generation until all ranks have entered the
    // intervening collective's local/leader barriers, so a two-generation
    // overwrite is impossible and these end waits are redundant.
    if constexpr (!DoubleBuffered && !DeferredConsumption) {
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
    }

    // Publish the final value to the three local consumers.
#pragma unroll
    for (int peer = 1; peer < kIslandSize; ++peer) {
      Header* peer_header = header(runtime.slabs[leader_rank + peer]);
      store_release(&peer_header->final_ready[block].value, generation);
    }

    if constexpr (!DoubleBuffered && !DeferredConsumption) {
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
}

template <int NumIslands, bool DoubleBuffered, bool DeferredConsumption>
static void launch_all_reduce_bf16(
    cudaStream_t stream,
    int blocks,
    const Runtime& runtime,
    const nv_bfloat16* input,
    nv_bfloat16* output,
    int64_t elements,
    bool vectorized_bf16x2) {
  if (vectorized_bf16x2) {
    hierarchical_all_reduce_bf16<
        NumIslands, DoubleBuffered, DeferredConsumption, true>
        <<<blocks, kVectorThreads, 0, stream>>>(
            runtime, input, output, elements);
  } else {
    hierarchical_all_reduce_bf16<
        NumIslands, DoubleBuffered, DeferredConsumption, false>
        <<<blocks, kScalarThreads, 0, stream>>>(
            runtime, input, output, elements);
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
    int64_t blocks,
    bool double_buffered,
    bool deferred_consumption) {
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
  TORCH_CHECK(
      !(double_buffered && deferred_consumption),
      "double buffering and deferred consumption are mutually exclusive");
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* input_ptr =
      reinterpret_cast<const nv_bfloat16*>(input.data_ptr());
  auto* output_ptr = reinterpret_cast<nv_bfloat16*>(output.data_ptr());
  const bool vectorized_bf16x2 =
      pcie_hierarchical::kEnableVectorizedBF16x2 &&
      input.numel() <= pcie_hierarchical::kVectorizedBF16x2MaxElements &&
      reinterpret_cast<std::uintptr_t>(input_ptr) % alignof(__nv_bfloat162) ==
          0 &&
      reinterpret_cast<std::uintptr_t>(output_ptr) % alignof(__nv_bfloat162) ==
          0;
  if (runtime->world_size == 12) {
    if (deferred_consumption) {
      pcie_hierarchical::deferred_consumption_preamble<3>
          <<<1, 32, 0, stream>>>(*runtime);
      pcie_hierarchical::launch_all_reduce_bf16<3, false, true>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    } else if (double_buffered) {
      pcie_hierarchical::launch_all_reduce_bf16<3, true, false>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    } else {
      pcie_hierarchical::launch_all_reduce_bf16<3, false, false>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    }
  } else if (runtime->world_size == 16) {
    if (deferred_consumption) {
      pcie_hierarchical::deferred_consumption_preamble<4>
          <<<1, 32, 0, stream>>>(*runtime);
      pcie_hierarchical::launch_all_reduce_bf16<4, false, true>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    } else if (double_buffered) {
      pcie_hierarchical::launch_all_reduce_bf16<4, true, false>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    } else {
      pcie_hierarchical::launch_all_reduce_bf16<4, false, false>(
          stream, static_cast<int>(blocks), *runtime, input_ptr, output_ptr,
          input.numel(), vectorized_bf16x2);
    }
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
