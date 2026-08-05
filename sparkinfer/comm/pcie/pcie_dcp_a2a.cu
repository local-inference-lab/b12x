// PCIe one-shot exchange for DCP attention output and LSE reduction.
//
// Each rank first copies its full partial output and FP32 LSE into one of two
// IPC-visible staging slots. A single system-scope barrier makes those copies
// visible, after which every rank pulls only its destination head shard from
// all peers and performs the stable LSE-weighted reduction while storing the
// final output. This deliberately follows the low-latency, one-barrier design
// of pcie_oneshot.cu rather than implementing a generic NCCL-style all-to-all.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                                \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::stringstream _message;                                              \
      _message << cudaGetErrorString(e) << "\n"                                \
               << __FILE__ << ':' << __LINE__;                                 \
      throw std::runtime_error(_message.str());                                \
    }                                                                          \
  } while (0)

namespace pcie_dcp_a2a {

namespace cg = cooperative_groups;

constexpr int kMaxBlocks = 64;
constexpr int kMaxRanks = 16;
constexpr int kFlagStride = 32;
using FlagType = uint32_t;

static int env_int(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::atoi(raw);
}

// Launch-shape overrides for latency sweeps. Every rank must use identical
// values: warp geometry decides which rows each block stages, and peer
// block b only reads what writer block b staged.
static int dcp_threads_override() {
  static const int value = env_int("SPARKINFER_PCIE_DCP_THREADS", 0);
  return value;
}

static int dcp_block_limit_override() {
  static const int value = env_int("SPARKINFER_PCIE_DCP_BLOCK_LIMIT", 0);
  return value;
}

static int dcp_test_delay_rank() {
  static const int value = env_int("SPARKINFER_PCIE_DCP_TEST_DELAY_RANK", -1);
  return value;
}

static uint64_t dcp_test_delay_cycles() {
  static const uint64_t value = static_cast<uint64_t>(std::max(
      0, env_int("SPARKINFER_PCIE_DCP_TEST_POST_BARRIER_DELAY_CYCLES", 0)));
  return value;
}

struct Signal {
  alignas(128) FlagType staging_generation;
  FlagType active_staging_slot;
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
};

struct RankSignals {
  Signal *signals[kMaxRanks];
};

struct RankStaging {
  void *ptrs[kMaxRanks];
};

struct DoubleStaging {
  RankStaging slots[2];
};

template <typename T> struct __align__(16) Pack {
  // Every transaction is exactly 16 bytes. Half/BF16 therefore carry eight
  // elements while the raw E4M3 query-gather path carries sixteen bytes.
  T values[16 / sizeof(T)];
};

#define DINLINE __device__ __forceinline__

static DINLINE void store_flag(FlagType *address, FlagType value) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(value),
               "l"(address));
}

static DINLINE FlagType load_flag(FlagType *address) {
  FlagType value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address));
  return value;
}

static DINLINE FlagType load_flag_gpu(FlagType *address) {
  FlagType value;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__global__ void advance_staging_slot_kernel(Signal *self) {
  if (threadIdx.x == 0) {
    const FlagType generation = self->staging_generation;
    self->active_staging_slot = generation & FlagType{1};
    self->staging_generation = generation + FlagType{1};
  }
}

template <int world_size>
DINLINE void select_staging(RankStaging &staging,
                            const DoubleStaging &staging_options,
                            Signal *self) {
  if (threadIdx.x == 0) {
    // The one-CTA control node runs earlier on this stream.  Every worker CTA
    // therefore observes one operation-wide slot regardless of worker grid
    // size or rank launch skew.
    const int slot = int(load_flag_gpu(&self->active_staging_slot) & FlagType{1});
#pragma unroll
    for (int peer = 0; peer < world_size; ++peer) {
      staging.ptrs[peer] = staging_options.slots[slot].ptrs[peer];
    }
  }
  __syncthreads();
}

// pre_sync orders in-kernel staging stores (from every warp of the block)
// before the flag post; without staging the flags can go out immediately.
template <int world_size, bool pre_sync>
DINLINE void start_barrier(const RankSignals &signals, Signal *self, int rank) {
  if constexpr (pre_sync) __syncthreads();
  if (threadIdx.x < world_size) {
    __threadfence_system();
    const auto value = self->self_counter[blockIdx.x][threadIdx.x] +=
        FlagType{1};
    auto *peer = &signals.signals[threadIdx.x]
                      ->peer_counter[value % 2][blockIdx.x][rank * kFlagStride];
    auto *mine =
        &self->peer_counter[value % 2][blockIdx.x][threadIdx.x * kFlagStride];
    store_flag(peer, value);
    while (load_flag(mine) != value) {
    }
  }
  __syncthreads();
}

DINLINE void test_post_barrier_delay(int rank, int delayed_rank,
                                     uint64_t delay_cycles) {
  if (delay_cycles != 0 && rank == delayed_rank && threadIdx.x == 0) {
    const uint64_t start = clock64();
    while (clock64() - start < delay_cycles) {
    }
  }
  __syncthreads();
}

DINLINE float to_float(half value) { return __half2float(value); }
DINLINE float to_float(nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> DINLINE T from_float(float value);

template <> DINLINE half from_float<half>(float value) {
  return __float2half(value);
}

template <> DINLINE nv_bfloat16 from_float<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

DINLINE float sanitize_lse(float value) {
  return isfinite(value) ? value : -CUDART_INF_F;
}

// Warp-per-row LSE-weighted reduction with in-kernel staging.
//
// Reader rows are the (batch, local_head) pairs of this rank's head shard,
// and every rank strides that identical row space with identical warp
// geometry. The warp that owns reader row (b, h) also stages this rank's
// contributions for (b, h) across all destination shards (packs and LSE),
// so the block-pairwise barrier covers exactly what the matching reader
// blocks pull. Each warp reads the LSE values once per row (one lane per
// source, packs_per_head times fewer remote scalar reads than per-pack
// loads), builds the softmax weights via shuffles in rotated source order,
// and streams the row's packs with lane-parallel pulls.
template <typename T, int world_size>
__global__ void __launch_bounds__(512, 1)
    dcp_lse_reduce_kernel(const T *__restrict__ local_output,
                          const float *__restrict__ local_lse,
                          DoubleStaging staging_options, int64_t lse_offset,
                          RankSignals signals, Signal *self,
                          T *__restrict__ output, int rank, int batch,
                          int total_heads, int head_dim,
                          int64_t input_stride_batch,
                          int64_t input_stride_head,
                          int64_t output_stride_batch,
                          int64_t output_stride_head, bool natural_log,
                          int delayed_rank, uint64_t delay_cycles) {
  constexpr int kPackElems = 8;
  const int heads_per_rank = total_heads / world_size;
  const int packs_per_head = head_dim / kPackElems;
  const int rows = batch * heads_per_rank;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_first = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * warps_per_block;

  const auto *local_packs = reinterpret_cast<const Pack<T> *>(local_output);
  auto *output_packs = reinterpret_cast<Pack<T> *>(output);

  __shared__ RankStaging staging;
  select_staging<world_size>(staging, staging_options, self);

  auto *staging_out = reinterpret_cast<Pack<T> *>(staging.ptrs[rank]);
  auto *staging_lse = reinterpret_cast<float *>(
      reinterpret_cast<char *>(staging.ptrs[rank]) + lse_offset);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / heads_per_rank;
    const int local_head = row - batch_index * heads_per_rank;
#pragma unroll
    for (int dest = 0; dest < world_size; ++dest) {
      const int64_t source_row = int64_t(batch_index) * total_heads +
                                 dest * heads_per_rank + local_head;
      const int64_t input_base =
          int64_t(batch_index) * input_stride_batch +
          int64_t(dest * heads_per_rank + local_head) * input_stride_head;
      const int64_t staging_base = source_row * packs_per_head;
      for (int pack = lane; pack < packs_per_head; pack += warpSize) {
        staging_out[staging_base + pack] = local_packs[input_base + pack];
      }
      if (lane == 0) {
        staging_lse[source_row] = local_lse[source_row];
      }
    }
  }
  start_barrier<world_size, true>(signals, self, rank);
  test_post_barrier_delay(rank, delayed_rank, delay_cycles);

  // Rotated source pointers so every later access uses a compile-time
  // index; the self source reads the local tensors directly (still hot in
  // L2 from staging).
  const Pack<T> *rot_packs[world_size];
#pragma unroll
  for (int i = 0; i < world_size; ++i) {
    const int src = (rank + i) % world_size;
    rot_packs[i] = src == rank
                       ? local_packs
                       : reinterpret_cast<const Pack<T> *>(staging.ptrs[src]);
  }

  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / heads_per_rank;
    const int local_head = row - batch_index * heads_per_rank;
    const int global_head = rank * heads_per_rank + local_head;
    const int64_t source_row = int64_t(batch_index) * total_heads + global_head;

    // Lane i pulls rotated source i's LSE; shuffles then give every lane
    // all sources in rotated order. The pointer is derived per lane from
    // the kernel param block to avoid a runtime-indexed local array.
    float lane_lse = -CUDART_INF_F;
    if (lane < world_size) {
      const int src = (rank + lane) % world_size;
      const float *lse_ptr =
          src == rank ? local_lse
                      : reinterpret_cast<const float *>(
                            reinterpret_cast<const char *>(staging.ptrs[src]) +
                            lse_offset);
      lane_lse = sanitize_lse(lse_ptr[source_row]);
    }
    float weights[world_size];
    float max_lse = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < world_size; ++i) {
      const float value = __shfl_sync(0xffffffff, lane_lse, i);
      weights[i] = value;
      max_lse = fmaxf(max_lse, value);
    }
    if (!isfinite(max_lse)) {
      max_lse = 0.0f;
    }
    float weight_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < world_size; ++i) {
      const float delta = weights[i] - max_lse;
      const float weight = isfinite(weights[i])
                               ? (natural_log ? expf(delta) : exp2f(delta))
                               : 0.0f;
      weights[i] = weight;
      weight_sum += weight;
    }
    const float inv_weight_sum = 1.0f / fmaxf(weight_sum, 1.0e-10f);

    const int64_t staging_base = source_row * packs_per_head;
    const int64_t local_base =
        int64_t(batch_index) * input_stride_batch +
        int64_t(global_head) * input_stride_head;
    const int64_t output_base =
        int64_t(batch_index) * output_stride_batch +
        int64_t(local_head) * output_stride_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      float accum[kPackElems] = {};
#pragma unroll
      for (int i = 0; i < world_size; ++i) {
        const int src = (rank + i) % world_size;
        const int64_t source_base = src == rank ? local_base : staging_base;
        const Pack<T> values = rot_packs[i][source_base + pack];
        const float weight = weights[i] * inv_weight_sum;
        // Empty DCP shards legitimately return LSE=-inf and leave their
        // partial output undefined. IEEE 0*NaN is NaN, so do not read or
        // accumulate an invalid contributor merely because its weight is 0.
        if (weight == 0.0f) {
          continue;
        }
#pragma unroll
        for (int element = 0; element < kPackElems; ++element) {
          accum[element] += weight * to_float(values.values[element]);
        }
      }
      Pack<T> result;
#pragma unroll
      for (int element = 0; element < kPackElems; ++element) {
        result.values[element] = from_float<T>(accum[element]);
      }
      output_packs[output_base + pack] = result;
    }
  }
}

// Warp-per-output-row head gather with in-kernel staging. Every rank
// strides the identical output row space with identical warp geometry, so
// the writer warp that owns row (batch, global_head) with source == this
// rank stages exactly the packs the matching reader blocks pull. Row-major
// warps also hoist the head/source arithmetic out of the pack loop (the
// pack-strided version paid six integer divisions per 16B pack).
template <typename T, int world_size>
__global__ void __launch_bounds__(512, 1)
    all_gather_heads_kernel(const T *__restrict__ local_input,
                            DoubleStaging staging_options, RankSignals signals,
                            Signal *self, T *__restrict__ output, int rank,
                            int batch, int local_heads, int head_dim,
                            int delayed_rank, uint64_t delay_cycles) {
  constexpr int kPackElems = 16 / sizeof(T);
  const int packs_per_head = head_dim / kPackElems;
  const int total_heads = local_heads * world_size;
  const int rows = batch * total_heads;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_first = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * warps_per_block;
  const auto *local_packs = reinterpret_cast<const Pack<T> *>(local_input);

  __shared__ RankStaging staging;
  select_staging<world_size>(staging, staging_options, self);

  auto *staging_out = reinterpret_cast<Pack<T> *>(staging.ptrs[rank]);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / total_heads;
    const int global_head = row - batch_index * total_heads;
    const int source_rank = global_head / local_heads;
    if (source_rank != rank) continue;
    const int local_head = global_head - source_rank * local_heads;
    const int64_t base =
        (int64_t(batch_index) * local_heads + local_head) * packs_per_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      staging_out[base + pack] = local_packs[base + pack];
    }
  }
  start_barrier<world_size, true>(signals, self, rank);
  test_post_barrier_delay(rank, delayed_rank, delay_cycles);

  auto *output_packs = reinterpret_cast<Pack<T> *>(output);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / total_heads;
    const int global_head = row - batch_index * total_heads;
    const int source_rank = global_head / local_heads;
    const int local_head = global_head - source_rank * local_heads;
    const auto *source_packs =
        source_rank == rank
            ? local_packs
            : reinterpret_cast<const Pack<T> *>(staging.ptrs[source_rank]);
    const int64_t source_base =
        (int64_t(batch_index) * local_heads + local_head) * packs_per_head;
    const int64_t output_base = int64_t(row) * packs_per_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      output_packs[output_base + pack] = source_packs[source_base + pack];
    }
  }
}

// Gather two independently typed, contiguous rank-local rows behind one IPC
// barrier. The payload is copied as raw 16-byte records, so BF16 projection
// values and FP32 router logits retain every bit while landing in separate,
// globally rank-major output tensors. One CTA is intentional: Kimi-K3's pair
// is only 672 bytes per rank and the ordinary gather also selects one CTA for
// this geometry.
template <int world_size>
__global__ void __launch_bounds__(512, 1)
    all_gather_pair_kernel(
        const uint8_t *__restrict__ local_first,
        const uint8_t *__restrict__ local_second,
        DoubleStaging staging_options, RankSignals signals, Signal *self,
        uint8_t *__restrict__ output_first,
        uint8_t *__restrict__ output_second, int rank, int batch,
        int first_row_bytes, int second_row_bytes, int delayed_rank,
        uint64_t delay_cycles) {
  using BytePack = Pack<uint8_t>;
  constexpr int kPackBytes = sizeof(BytePack);
  const int64_t first_packs = first_row_bytes / kPackBytes;
  const int64_t second_packs = second_row_bytes / kPackBytes;
  const int64_t combined_packs = first_packs + second_packs;
  const auto *first = reinterpret_cast<const BytePack *>(local_first);
  const auto *second = reinterpret_cast<const BytePack *>(local_second);

  __shared__ RankStaging staging;
  select_staging<world_size>(staging, staging_options, self);

  auto *staging_out = reinterpret_cast<BytePack *>(staging.ptrs[rank]);
  for (int64_t linear = threadIdx.x; linear < batch * first_packs;
       linear += blockDim.x) {
    const int64_t batch_index = linear / first_packs;
    const int64_t pack = linear - batch_index * first_packs;
    staging_out[batch_index * combined_packs + pack] = first[linear];
  }
  for (int64_t linear = threadIdx.x; linear < batch * second_packs;
       linear += blockDim.x) {
    const int64_t batch_index = linear / second_packs;
    const int64_t pack = linear - batch_index * second_packs;
    staging_out[batch_index * combined_packs + first_packs + pack] =
        second[linear];
  }
  start_barrier<world_size, true>(signals, self, rank);
  test_post_barrier_delay(rank, delayed_rank, delay_cycles);

  auto *first_out = reinterpret_cast<BytePack *>(output_first);
  auto *second_out = reinterpret_cast<BytePack *>(output_second);
  const int64_t first_output_packs = batch * world_size * first_packs;
  for (int64_t linear = threadIdx.x; linear < first_output_packs;
       linear += blockDim.x) {
    const int64_t batch_index = linear / (world_size * first_packs);
    const int64_t row_pack = linear - batch_index * world_size * first_packs;
    const int source_rank = static_cast<int>(row_pack / first_packs);
    const int64_t pack = row_pack - source_rank * first_packs;
    const auto *source =
        source_rank == rank
            ? first
            : reinterpret_cast<const BytePack *>(staging.ptrs[source_rank]);
    const int64_t source_base = source_rank == rank
                                    ? batch_index * first_packs
                                    : batch_index * combined_packs;
    first_out[linear] = source[source_base + pack];
  }
  const int64_t second_output_packs = batch * world_size * second_packs;
  for (int64_t linear = threadIdx.x; linear < second_output_packs;
       linear += blockDim.x) {
    const int64_t batch_index = linear / (world_size * second_packs);
    const int64_t row_pack = linear - batch_index * world_size * second_packs;
    const int source_rank = static_cast<int>(row_pack / second_packs);
    const int64_t pack = row_pack - source_rank * second_packs;
    const auto *source =
        source_rank == rank
            ? second
            : reinterpret_cast<const BytePack *>(staging.ptrs[source_rank]);
    const int64_t source_base =
        source_rank == rank ? batch_index * second_packs
                            : batch_index * combined_packs + first_packs;
    second_out[linear] = source[source_base + pack];
  }
}

// TP16 Kimi-K3 decode specialization.  The ordinary paired path materializes
// all 896 FP32 router logits and then launches a separate top-k kernel.  This
// path reads the rank-local router shards directly from the same IPC staging
// used by the BF16 down-projection gather and emits only the 16 selected
// experts.  Selection uses the same float ordering and lower-expert-id tie
// break as vLLM's native one-group sigmoid+bias top-k implementation.
__device__ __forceinline__ uint32_t kimi_float_order_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t mask = (bits & 0x80000000U) != 0U ? 0xffffffffU : 0x80000000U;
  return bits ^ mask;
}

__device__ __forceinline__ uint64_t kimi_topk_key(float score, int expert) {
  return (uint64_t{kimi_float_order_key(score)} << 32) |
         uint64_t{0xffff - expert};
}

__device__ __forceinline__ int kimi_topk_expert(uint64_t key) {
  return 0xffff - static_cast<int>(key & uint64_t{0xffff});
}

__device__ __forceinline__ uint64_t kimi_warp_max_key(uint64_t key) {
  const uint32_t hi = static_cast<uint32_t>(key >> 32);
  const uint32_t lo = static_cast<uint32_t>(key);
  uint32_t max_hi;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
               : "=r"(max_hi)
               : "r"(hi));
  const uint32_t lo_contribution = hi == max_hi ? lo : 0U;
  uint32_t max_lo;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
               : "=r"(max_lo)
               : "r"(lo_contribution));
  return (uint64_t{max_hi} << 32) | uint64_t{max_lo};
}

template <int count>
__device__ __forceinline__ void kimi_sort_keys_descending(uint64_t (&keys)[count]) {
  static_assert(count == 2 || count == 8);
#pragma unroll
  for (int width = 2; width <= count; width *= 2) {
#pragma unroll
    for (int stride = width / 2; stride > 0; stride /= 2) {
#pragma unroll
      for (int index = 0; index < count; ++index) {
        const int peer = index ^ stride;
        if (peer > index) {
          const bool descending = (index & width) == 0;
          const bool swap = descending ? keys[index] < keys[peer]
                                       : keys[index] > keys[peer];
          if (swap) {
            const uint64_t temporary = keys[index];
            keys[index] = keys[peer];
            keys[peer] = temporary;
          }
        }
      }
    }
  }
}

template <int count>
__device__ __forceinline__ uint64_t kimi_lane_owned_top16(
    uint64_t (&keys)[count], int lane) {
  kimi_sort_keys_descending(keys);
  uint64_t previous = 0;
  uint64_t lane_key = 0;
#pragma unroll
  for (int selected = 0; selected < 16; ++selected) {
    const bool remove_head = selected > 0 && previous == keys[0];
    const int tail_expert = kimi_topk_expert(keys[count - 1]);
#pragma unroll
    for (int item = 0; item < count; ++item) {
      keys[item] = remove_head && item == count - 1
                       ? kimi_topk_key(-CUDART_INF_F, tail_expert)
                   : remove_head ? keys[item + 1]
                                 : keys[item];
    }
    previous = kimi_warp_max_key(keys[0]);
    if (lane == selected) {
      lane_key = previous;
    }
  }
  return lane_key;
}

template <int world_size>
__global__ void __launch_bounds__(512, 1)
    all_gather_pair_kimi_topk_kernel(
        const __nv_bfloat16 *__restrict__ local_down,
        const float *__restrict__ local_router, DoubleStaging staging_options,
        RankSignals signals, Signal *self,
        const float *__restrict__ correction_bias,
        __nv_bfloat16 *__restrict__ output_down,
        float *__restrict__ topk_weights, int32_t *__restrict__ topk_ids,
        int rank, int delayed_rank, uint64_t delay_cycles) {
  static_assert(world_size == 16);
  constexpr int kThreads = 512;
  constexpr int kTopkThreads = 256;
  constexpr int kRouterWarps = 4;
  constexpr int kPreprocessItems = 4;
  constexpr int kLocalDown = 224;
  constexpr int kLocalExperts = 56;
  constexpr int kNumExperts = world_size * kLocalExperts;
  constexpr int kTopK = 16;
  constexpr int kDownBytes = kLocalDown * sizeof(__nv_bfloat16);
  constexpr int kRouterBytes = kLocalExperts * sizeof(float);
  constexpr int kDownPacks = kDownBytes / sizeof(Pack<uint8_t>);
  constexpr int kRouterPacks = kRouterBytes / sizeof(Pack<uint8_t>);
  constexpr int kCombinedPacks = kDownPacks + kRouterPacks;
  using BytePack = Pack<uint8_t>;
  __shared__ RankStaging staging;
  __shared__ __align__(16) float selection_scores[kNumExperts];
  __shared__ __align__(16) float unbiased_scores[kNumExperts];
  __shared__ uint64_t intermediate_keys[4 * kTopK];
  select_staging<world_size>(staging, staging_options, self);

  auto *staging_out = reinterpret_cast<BytePack *>(staging.ptrs[rank]);
  const auto *down_packs = reinterpret_cast<const BytePack *>(local_down);
  const auto *router_packs = reinterpret_cast<const BytePack *>(local_router);
  for (int pack = threadIdx.x; pack < kDownPacks; pack += blockDim.x) {
    staging_out[pack] = down_packs[pack];
  }
  for (int pack = threadIdx.x; pack < kRouterPacks; pack += blockDim.x) {
    staging_out[kDownPacks + pack] = router_packs[pack];
  }
  start_barrier<world_size, true>(signals, self, rank);
  test_post_barrier_delay(rank, delayed_rank, delay_cycles);

  const int lane = int(threadIdx.x) & 31;
  const int warp_id = int(threadIdx.x) >> 5;
  // Pull both payloads with the ordinary pair kernel's 512-thread copy
  // geometry, but land router bytes in shared memory instead of a global
  // 896-float intermediate. The following CTA barrier is the handoff to the
  // native-shaped top-k phase.
  auto *down_out_packs = reinterpret_cast<BytePack *>(output_down);
  auto *router_shared_packs =
      reinterpret_cast<BytePack *>(selection_scores);
  for (int linear = int(threadIdx.x);
       linear < world_size * (kDownPacks + kRouterPacks);
       linear += kThreads) {
    if (linear < world_size * kDownPacks) {
      const int source_rank = linear / kDownPacks;
      const int pack = linear - source_rank * kDownPacks;
      const auto *source =
          source_rank == rank
              ? down_packs
              : reinterpret_cast<const BytePack *>(staging.ptrs[source_rank]);
      down_out_packs[linear] = source[pack];
    } else {
      const int router_linear = linear - world_size * kDownPacks;
      const int source_rank = router_linear / kRouterPacks;
      const int pack = router_linear - source_rank * kRouterPacks;
      const auto *source =
          source_rank == rank
              ? router_packs
              : reinterpret_cast<const BytePack *>(staging.ptrs[source_rank]) +
                    kDownPacks;
      router_shared_packs[router_linear] = source[pack];
    }
  }
  __syncthreads();

  if (threadIdx.x < kTopkThreads) {
#pragma unroll
    for (int item = 0; item < kPreprocessItems; ++item) {
      const int expert = int(threadIdx.x) + item * kTopkThreads;
      if (expert < kNumExperts) {
        float unbiased = 0.0F;
        float selection = -CUDART_INF_F;
        const float input = selection_scores[expert];
        const float bias = correction_bias[expert];
        if (isfinite(input) && isfinite(bias)) {
          unbiased = 0.5F * tanhf(0.5F * input) + 0.5F;
          const float biased = unbiased + bias;
          selection = biased == 0.0F ? 0.0F : biased;
        }
        unbiased_scores[expert] = unbiased;
        selection_scores[expert] = selection;
      }
    }
  }
  __syncthreads();

  if (warp_id < kRouterWarps) {
    uint64_t worker_keys[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int expert = warp_id * 256 + item * 32 + lane;
      const float score =
          expert < kNumExperts ? selection_scores[expert] : -CUDART_INF_F;
      worker_keys[item] = kimi_topk_key(score, expert);
    }
    const uint64_t selected = kimi_lane_owned_top16(worker_keys, lane);
    if (lane < kTopK) {
      intermediate_keys[warp_id * kTopK + lane] = selected;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    uint64_t merge_keys[2];
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      merge_keys[item] = intermediate_keys[item * 32 + lane];
    }
    const uint64_t selected = kimi_lane_owned_top16(merge_keys, lane);
    const int expert = lane < kTopK ? kimi_topk_expert(selected) : -1;
    const bool finite = lane < kTopK && expert >= 0 && expert < kNumExperts;
    const float unbiased = finite ? unbiased_scores[expert] : 0.0F;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const float sum = cg::reduce(warp, unbiased, cg::plus<float>{});
    if (lane < kTopK) {
      float scale = 1.0F;
      scale /= sum + 1e-20F;
      topk_weights[lane] = finite ? unbiased * scale : 0.0F;
      topk_ids[lane] = expert;
    }
  }
}

class PCIeDCPA2A {
public:
  int rank_;
  int world_size_;
  RankSignals signals_{};
  Signal *self_signal_;
  DoubleStaging staging_{};
  int64_t output_capacity_elems_;
  int64_t lse_offset_;
  int64_t lse_capacity_;


  PCIeDCPA2A(Signal **signals,
             const std::vector<std::array<void *, 2>> &staging,
             int64_t output_capacity_elems, int64_t lse_offset,
             int64_t lse_capacity, int rank, int world_size)
      : rank_(rank), world_size_(world_size), self_signal_(signals[rank]),
        output_capacity_elems_(output_capacity_elems), lse_offset_(lse_offset),
        lse_capacity_(lse_capacity) {
    for (int peer = 0; peer < world_size_; ++peer) {
      signals_.signals[peer] = signals[peer];
      staging_.slots[0].ptrs[peer] = staging[peer][0];
      staging_.slots[1].ptrs[peer] = staging[peer][1];
    }
  }

  template <typename T>
  void run(cudaStream_t stream, const T *partial_output,
           const float *partial_lse, T *output, int batch, int total_heads,
           int head_dim, int64_t input_stride_batch,
           int64_t input_stride_head, int64_t output_stride_batch,
           int64_t output_stride_head, bool natural_log, int threads,
           int block_limit) {
    const int64_t output_elems = int64_t(batch) * total_heads * head_dim;
    const int64_t lse_elems = int64_t(batch) * total_heads;
    if (output_elems > output_capacity_elems_ || lse_elems > lse_capacity_) {
      throw std::runtime_error("PCIe DCP A2A staging capacity exceeded");
    }
    if (head_dim % 8 != 0) {
      throw std::runtime_error("head_dim must be a multiple of 8");
    }
    if (total_heads % world_size_ != 0) {
      throw std::runtime_error("total_heads must be divisible by world size");
    }
    if (threads < world_size_ || threads > 1024) {
      throw std::runtime_error("invalid thread count");
    }
    if (block_limit <= 0 || block_limit > kMaxBlocks) {
      throw std::runtime_error("invalid block limit");
    }

    if (const int env_threads = dcp_threads_override(); env_threads > 0) {
      threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    }
    if (const int env_blocks = dcp_block_limit_override(); env_blocks > 0) {
      block_limit = std::min(env_blocks, kMaxBlocks);
    }
    if (threads % 32 != 0) {
      throw std::runtime_error("threads must be a multiple of 32");
    }

    // Select one staging slot per execution in a graph-capturable device node.
    // Worker CTA count may change large -> small -> large without leaving
    // dormant per-block parity counters behind.
    const int heads_per_rank = total_heads / world_size_;
    const int rows = batch * heads_per_rank;
    const int warps_per_block = threads / 32;
    const int blocks = std::max(
        1, std::min(block_limit, (rows + warps_per_block - 1) / warps_per_block));
    advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_signal_);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
    const int delayed_rank = dcp_test_delay_rank();
    const uint64_t delay_cycles = dcp_test_delay_cycles();

#define LAUNCH(world)                                                          \
  dcp_lse_reduce_kernel<T, world><<<blocks, threads, 0, stream>>>(             \
      partial_output, partial_lse, staging_, lse_offset_, signals_,           \
      self_signal_, output, rank_, batch, total_heads, head_dim,              \
      input_stride_batch, input_stride_head, output_stride_batch,              \
      output_stride_head, natural_log, delayed_rank, delay_cycles)
    switch (world_size_) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      throw std::runtime_error("PCIe DCP A2A supports 2, 4, 8, or 16 ranks");
    }
#undef LAUNCH
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }

  template <typename T>
  void all_gather_heads(cudaStream_t stream, const T *local_input, T *output,
                        int batch, int local_heads, int head_dim, int threads,
                        int block_limit) {
    constexpr int kPackElems = 16 / sizeof(T);
    const int total_heads = local_heads * world_size_;
    const int64_t output_elems = int64_t(batch) * total_heads * head_dim;
    if (output_elems > output_capacity_elems_) {
      throw std::runtime_error("PCIe DCP all-gather staging capacity exceeded");
    }
    if (head_dim % kPackElems != 0) {
      throw std::runtime_error(
          "head_dim must align to a 16-byte gather transaction");
    }
    if (threads < world_size_ || threads > 1024) {
      throw std::runtime_error("invalid thread count");
    }
    if (block_limit <= 0 || block_limit > kMaxBlocks) {
      throw std::runtime_error("invalid block limit");
    }

    if (const int env_threads = dcp_threads_override(); env_threads > 0) {
      threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    }
    if (const int env_blocks = dcp_block_limit_override(); env_blocks > 0) {
      block_limit = std::min(env_blocks, kMaxBlocks);
    }
    if (threads % 32 != 0) {
      throw std::runtime_error("threads must be a multiple of 32");
    }

    // Slot ownership is published by a device control node, including for
    // back-to-back eager launches and every CUDA graph replay.
    const int rows = batch * total_heads;
    const int warps_per_block = threads / 32;
    const int blocks = std::max(
        1, std::min(block_limit, (rows + warps_per_block - 1) / warps_per_block));
    advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_signal_);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
    const int delayed_rank = dcp_test_delay_rank();
    const uint64_t delay_cycles = dcp_test_delay_cycles();

#define LAUNCH(world)                                                          \
  all_gather_heads_kernel<T, world><<<blocks, threads, 0, stream>>>(           \
      local_input, staging_, signals_, self_signal_, output, rank_, batch,    \
      local_heads, head_dim, delayed_rank, delay_cycles)
    switch (world_size_) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    case 16:
      LAUNCH(16);
      break;
    default:
      throw std::runtime_error(
          "PCIe DCP all-gather supports 2, 4, 8, or 16 ranks");
    }
#undef LAUNCH
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }

  void all_gather_pair(cudaStream_t stream, const void *local_first,
                       const void *local_second, void *output_first,
                       void *output_second, int batch, int first_row_bytes,
                       int second_row_bytes, int threads) {
    constexpr int kPackBytes = sizeof(Pack<uint8_t>);
    const int64_t output_bytes =
        int64_t(batch) * world_size_ * (first_row_bytes + second_row_bytes);
    // Staging slots are allocated in two-byte capacity units because the
    // attention path is BF16. Raw paired payloads may use all those bytes.
    if (output_bytes > output_capacity_elems_ * 2) {
      throw std::runtime_error("PCIe DCP pair staging capacity exceeded");
    }
    if (first_row_bytes <= 0 || second_row_bytes <= 0 ||
        first_row_bytes % kPackBytes != 0 ||
        second_row_bytes % kPackBytes != 0) {
      throw std::runtime_error(
          "paired rows must be positive and 16-byte aligned");
    }
    if (threads < world_size_ || threads > 512 || threads % 32 != 0) {
      throw std::runtime_error("invalid paired-gather thread count");
    }
    if (const int env_threads = dcp_threads_override(); env_threads > 0) {
      threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    }

    advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_signal_);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
    const int delayed_rank = dcp_test_delay_rank();
    const uint64_t delay_cycles = dcp_test_delay_cycles();

#define LAUNCH_PAIR(world)                                                     \
  all_gather_pair_kernel<world><<<1, threads, 0, stream>>>(                   \
      reinterpret_cast<const uint8_t *>(local_first),                         \
      reinterpret_cast<const uint8_t *>(local_second), staging_, signals_,   \
      self_signal_, reinterpret_cast<uint8_t *>(output_first),                \
      reinterpret_cast<uint8_t *>(output_second), rank_, batch,              \
      first_row_bytes, second_row_bytes, delayed_rank, delay_cycles)
    switch (world_size_) {
    case 2:
      LAUNCH_PAIR(2);
      break;
    case 4:
      LAUNCH_PAIR(4);
      break;
    case 8:
      LAUNCH_PAIR(8);
      break;
    case 16:
      LAUNCH_PAIR(16);
      break;
    default:
      throw std::runtime_error("PCIe DCP supports 2, 4, 8, or 16 ranks");
    }
#undef LAUNCH_PAIR
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }

  void all_gather_pair_kimi_topk(
      cudaStream_t stream, const __nv_bfloat16 *local_down,
      const float *local_router, const float *correction_bias,
      __nv_bfloat16 *output_down, float *topk_weights, int32_t *topk_ids) {
    if (world_size_ != 16) {
      throw std::runtime_error(
          "Kimi paired gather+top-k requires exactly 16 ranks");
    }
    constexpr int kLocalPayloadBytes =
        224 * sizeof(__nv_bfloat16) + 56 * sizeof(float);
    if (int64_t(world_size_) * kLocalPayloadBytes >
        output_capacity_elems_ * 2) {
      throw std::runtime_error("PCIe DCP Kimi staging capacity exceeded");
    }

    advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_signal_);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
    const int delayed_rank = dcp_test_delay_rank();
    const uint64_t delay_cycles = dcp_test_delay_cycles();
    all_gather_pair_kimi_topk_kernel<16><<<1, 512, 0, stream>>>(
        local_down, local_router, staging_, signals_, self_signal_,
        correction_bias, output_down, topk_weights, topk_ids, rank_,
        delayed_rank, delay_cycles);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }
};

} // namespace pcie_dcp_a2a

using fptr_t = int64_t;

static fptr_t init_dcp_a2a(const std::vector<fptr_t> &signal_ptrs,
                           const std::vector<fptr_t> &staging0_ptrs,
                           const std::vector<fptr_t> &staging1_ptrs,
                           int64_t output_capacity_elems, int64_t lse_offset,
                           int64_t lse_capacity, int64_t rank) {
  const int world_size = signal_ptrs.size();
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8 ||
              world_size == 16);
  TORCH_CHECK_EQ(staging0_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK_EQ(staging1_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  pcie_dcp_a2a::Signal *signals[pcie_dcp_a2a::kMaxRanks];
  std::vector<std::array<void *, 2>> staging(world_size);
  for (int peer = 0; peer < world_size; ++peer) {
    signals[peer] = reinterpret_cast<pcie_dcp_a2a::Signal *>(signal_ptrs[peer]);
    staging[peer] = {reinterpret_cast<void *>(staging0_ptrs[peer]),
                     reinterpret_cast<void *>(staging1_ptrs[peer])};
  }
  return reinterpret_cast<fptr_t>(
      new pcie_dcp_a2a::PCIeDCPA2A(signals, staging, output_capacity_elems,
                                   lse_offset, lse_capacity, rank, world_size));
}

static void lse_reduce_scatter(fptr_t pointer, torch::Tensor &partial_output,
                               torch::Tensor &partial_lse,
                               torch::Tensor &output, bool natural_log,
                               int64_t threads, int64_t block_limit) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(partial_output));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(partial_output.is_cuda() && partial_lse.is_cuda() &&
              output.is_cuda());
  TORCH_CHECK(partial_lse.is_contiguous());
  TORCH_CHECK_EQ(partial_output.dim(), 3);
  TORCH_CHECK_EQ(partial_lse.dim(), 2);
  TORCH_CHECK_EQ(output.dim(), 3);
  TORCH_CHECK_EQ(partial_lse.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(partial_output.scalar_type(), output.scalar_type());

  const int64_t batch = partial_output.size(0);
  const int64_t total_heads = partial_output.size(1);
  const int64_t head_dim = partial_output.size(2);
  TORCH_CHECK_GT(batch, 0);
  TORCH_CHECK_EQ(partial_lse.size(0), batch);
  TORCH_CHECK_EQ(partial_lse.size(1), total_heads);
  TORCH_CHECK_EQ(total_heads % runtime->world_size_, 0);
  TORCH_CHECK_EQ(output.size(0), batch);
  TORCH_CHECK_EQ(output.size(1), total_heads / runtime->world_size_);
  TORCH_CHECK_EQ(output.size(2), head_dim);
  TORCH_CHECK_EQ(partial_output.stride(2), 1);
  TORCH_CHECK_EQ(output.stride(2), 1);

  const auto input_heads = partial_output.size(1);
  const auto output_heads = output.size(1);
  const bool input_token_major =
      partial_output.stride(0) == input_heads * head_dim &&
      partial_output.stride(1) == head_dim;
  const bool input_head_major =
      partial_output.stride(0) == head_dim &&
      partial_output.stride(1) >= batch * head_dim;
  const bool output_token_major =
      output.stride(0) == output_heads * head_dim &&
      output.stride(1) == head_dim;
  const bool output_head_major =
      output.stride(0) == head_dim && output.stride(1) >= batch * head_dim;
  TORCH_CHECK(input_token_major || input_head_major,
              "partial_output must be packed token-major or head-major");
  TORCH_CHECK(output_token_major || output_head_major,
              "output must be packed token-major or head-major");
  TORCH_CHECK_EQ(partial_output.stride(0) % 8, 0);
  TORCH_CHECK_EQ(partial_output.stride(1) % 8, 0);
  TORCH_CHECK_EQ(output.stride(0) % 8, 0);
  TORCH_CHECK_EQ(output.stride(1) % 8, 0);

  const int64_t input_stride_batch = partial_output.stride(0) / 8;
  const int64_t input_stride_head = partial_output.stride(1) / 8;
  const int64_t output_stride_batch = output.stride(0) / 8;
  const int64_t output_stride_head = output.stride(1) / 8;

  switch (partial_output.scalar_type()) {
  case at::ScalarType::Half:
    runtime->run(stream,
                 reinterpret_cast<const half *>(partial_output.data_ptr()),
                 reinterpret_cast<const float *>(partial_lse.data_ptr()),
                 reinterpret_cast<half *>(output.data_ptr()), int(batch),
                 int(total_heads), int(head_dim), input_stride_batch,
                 input_stride_head, output_stride_batch, output_stride_head,
                 natural_log, int(threads), int(block_limit));
    break;
  case at::ScalarType::BFloat16:
    runtime->run(
        stream,
        reinterpret_cast<const nv_bfloat16 *>(partial_output.data_ptr()),
        reinterpret_cast<const float *>(partial_lse.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(output.data_ptr()), int(batch),
        int(total_heads), int(head_dim), input_stride_batch,
        input_stride_head, output_stride_batch, output_stride_head,
        natural_log, int(threads), int(block_limit));
    break;
  default:
    TORCH_CHECK(false, "partial_output must be float16 or bfloat16");
  }
}

static void all_gather_heads(fptr_t pointer, torch::Tensor &local_input,
                             torch::Tensor &output, int64_t threads,
                             int64_t block_limit) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_input));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(local_input.is_cuda() && output.is_cuda());
  TORCH_CHECK(local_input.is_contiguous() && output.is_contiguous());
  TORCH_CHECK_EQ(local_input.dim(), 3);
  TORCH_CHECK_EQ(output.dim(), 3);
  TORCH_CHECK_EQ(local_input.scalar_type(), output.scalar_type());

  const int64_t batch = local_input.size(0);
  const int64_t local_heads = local_input.size(1);
  const int64_t head_dim = local_input.size(2);
  TORCH_CHECK_GT(batch, 0);
  TORCH_CHECK_GT(local_heads, 0);
  TORCH_CHECK_EQ(output.size(0), batch);
  TORCH_CHECK_EQ(output.size(1), local_heads * runtime->world_size_);
  TORCH_CHECK_EQ(output.size(2), head_dim);

  switch (local_input.scalar_type()) {
  case at::ScalarType::Half:
    runtime->all_gather_heads(
        stream, reinterpret_cast<const half *>(local_input.data_ptr()),
        reinterpret_cast<half *>(output.data_ptr()), int(batch),
        int(local_heads), int(head_dim), int(threads), int(block_limit));
    break;
  case at::ScalarType::BFloat16:
    runtime->all_gather_heads(
        stream,
        reinterpret_cast<const nv_bfloat16 *>(local_input.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(output.data_ptr()), int(batch),
        int(local_heads), int(head_dim), int(threads), int(block_limit));
    break;
  case at::ScalarType::Float8_e4m3fn:
    // The gather is a bitwise exchange. Treat E4M3 records as bytes; no
    // arithmetic or conversion occurs in this path.
    runtime->all_gather_heads(
        stream, reinterpret_cast<const uint8_t *>(local_input.data_ptr()),
        reinterpret_cast<uint8_t *>(output.data_ptr()), int(batch),
        int(local_heads), int(head_dim), int(threads), int(block_limit));
    break;
  default:
    TORCH_CHECK(false,
                "local_input must be float16, bfloat16, or float8_e4m3fn");
  }
}

static bool is_supported_pair_dtype(at::ScalarType dtype) {
  return dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
         dtype == at::ScalarType::Float ||
         dtype == at::ScalarType::Float8_e4m3fn;
}

static void all_gather_pair(fptr_t pointer, torch::Tensor &local_first,
                            torch::Tensor &local_second,
                            torch::Tensor &output_first,
                            torch::Tensor &output_second, int64_t threads) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_first));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(local_first.is_cuda() && local_second.is_cuda() &&
              output_first.is_cuda() && output_second.is_cuda());
  TORCH_CHECK(local_first.device() == local_second.device() &&
              local_first.device() == output_first.device() &&
              local_first.device() == output_second.device());
  TORCH_CHECK(local_first.is_contiguous() && local_second.is_contiguous() &&
              output_first.is_contiguous() && output_second.is_contiguous());
  TORCH_CHECK_EQ(local_first.dim(), 2);
  TORCH_CHECK_EQ(local_second.dim(), 2);
  TORCH_CHECK_EQ(output_first.dim(), 2);
  TORCH_CHECK_EQ(output_second.dim(), 2);
  TORCH_CHECK_EQ(local_first.scalar_type(), output_first.scalar_type());
  TORCH_CHECK_EQ(local_second.scalar_type(), output_second.scalar_type());
  TORCH_CHECK(is_supported_pair_dtype(local_first.scalar_type()) &&
              is_supported_pair_dtype(local_second.scalar_type()),
              "paired inputs must be float16, bfloat16, float32, or "
              "float8_e4m3fn");

  const int64_t batch = local_first.size(0);
  TORCH_CHECK_GT(batch, 0);
  TORCH_CHECK_EQ(local_second.size(0), batch);
  TORCH_CHECK_EQ(output_first.size(0), batch);
  TORCH_CHECK_EQ(output_second.size(0), batch);
  TORCH_CHECK_EQ(output_first.size(1),
                 local_first.size(1) * runtime->world_size_);
  TORCH_CHECK_EQ(output_second.size(1),
                 local_second.size(1) * runtime->world_size_);
  const int64_t first_row_bytes =
      local_first.size(1) * local_first.element_size();
  const int64_t second_row_bytes =
      local_second.size(1) * local_second.element_size();
  TORCH_CHECK_LE(first_row_bytes, std::numeric_limits<int>::max());
  TORCH_CHECK_LE(second_row_bytes, std::numeric_limits<int>::max());

  runtime->all_gather_pair(
      stream, local_first.data_ptr(), local_second.data_ptr(),
      output_first.data_ptr(), output_second.data_ptr(), int(batch),
      int(first_row_bytes), int(second_row_bytes), int(threads));
}

static void all_gather_pair_kimi_topk(
    fptr_t pointer, torch::Tensor &local_down, torch::Tensor &local_router,
    torch::Tensor &correction_bias, torch::Tensor &output_down,
    torch::Tensor &topk_weights, torch::Tensor &topk_ids) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_down));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(runtime->world_size_, 16);
  TORCH_CHECK(local_down.is_cuda() && local_router.is_cuda() &&
              correction_bias.is_cuda() && output_down.is_cuda() &&
              topk_weights.is_cuda() && topk_ids.is_cuda());
  TORCH_CHECK(local_down.device() == local_router.device() &&
              local_down.device() == correction_bias.device() &&
              local_down.device() == output_down.device() &&
              local_down.device() == topk_weights.device() &&
              local_down.device() == topk_ids.device());
  TORCH_CHECK(local_down.is_contiguous() && local_router.is_contiguous() &&
              correction_bias.is_contiguous() && output_down.is_contiguous() &&
              topk_weights.is_contiguous() && topk_ids.is_contiguous());
  TORCH_CHECK_EQ(local_down.scalar_type(), at::ScalarType::BFloat16);
  TORCH_CHECK_EQ(local_router.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(correction_bias.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(output_down.scalar_type(), at::ScalarType::BFloat16);
  TORCH_CHECK_EQ(topk_weights.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(topk_ids.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK(local_down.sizes() == at::IntArrayRef({1, 224}));
  TORCH_CHECK(local_router.sizes() == at::IntArrayRef({1, 56}));
  TORCH_CHECK(correction_bias.sizes() == at::IntArrayRef({896}));
  TORCH_CHECK(output_down.sizes() == at::IntArrayRef({1, 3584}));
  TORCH_CHECK(topk_weights.sizes() == at::IntArrayRef({1, 16}));
  TORCH_CHECK(topk_ids.sizes() == at::IntArrayRef({1, 16}));

  runtime->all_gather_pair_kimi_topk(
      stream,
      reinterpret_cast<const __nv_bfloat16 *>(local_down.data_ptr()),
      reinterpret_cast<const float *>(local_router.data_ptr()),
      reinterpret_cast<const float *>(correction_bias.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(output_down.data_ptr()),
      reinterpret_cast<float *>(topk_weights.data_ptr()),
      reinterpret_cast<int32_t *>(topk_ids.data_ptr()));
}

static void dispose(fptr_t pointer) {
  delete reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
}

static int64_t meta_size() { return sizeof(pcie_dcp_a2a::Signal); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("init_dcp_a2a", &init_dcp_a2a, "initialize PCIe DCP A2A");
  module.def("lse_reduce_scatter", &lse_reduce_scatter,
             "fused PCIe DCP LSE reduce-scatter");
  module.def("all_gather_heads", &all_gather_heads,
             "PCIe DCP head-dimension all-gather");
  module.def("all_gather_pair", &all_gather_pair,
             "PCIe DCP raw paired all-gather");
  module.def("all_gather_pair_kimi_topk", &all_gather_pair_kimi_topk,
             "TP16 Kimi paired projection gather and sigmoid top-k");
  module.def("dispose", &dispose, "dispose PCIe DCP A2A");
  module.def("meta_size", &meta_size, "signal metadata size");
}
