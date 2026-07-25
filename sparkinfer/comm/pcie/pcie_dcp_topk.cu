// Exact owner-sharded transport for DCP sparse top-k.
//
// Candidate exchange runs on the DCP group. Each source writes directly into
// the destination owner's row-major CUDA-IPC slab, which the exact row-top-k
// kernel can consume without a pack, NCCL all-to-all, or unpack.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                                \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::stringstream message;                                               \
      message << cudaGetErrorString(e) << "\\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (0)

namespace pcie_dcp_topk {

constexpr int kMaxBlocks = 128;
constexpr int kMaxRanks = 8;
constexpr int kFlagStride = 32;
using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128)
      FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
};

struct RankSignals {
  Signal *signals[kMaxRanks];
};

struct RankStaging {
  void *ptrs[kMaxRanks];
};

#define DINLINE __device__ __forceinline__

static DINLINE void store_flag(FlagType *address, FlagType value) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" : : "r"(value),
               "l"(address));
}
static DINLINE FlagType load_flag(FlagType *address) {
  FlagType value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address));
  return value;
}

template <int world_size>
DINLINE void block_pair_barrier(const RankSignals &signals, Signal *self,
                                int rank) {
  __syncthreads();
  if (threadIdx.x < world_size) {
    __threadfence_system();
    const auto value =
        self->self_counter[blockIdx.x][threadIdx.x] += FlagType{1};
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

DINLINE void block_range(int64_t total, int64_t &begin, int64_t &end) {
  const int64_t chunk = (total + gridDim.x - 1) / gridDim.x;
  begin = int64_t(blockIdx.x) * chunk;
  end = min(begin + chunk, total);
}

// Destination layout is two independent row-major planes:
//
//   indices[owner_row, source_rank, topk]
//   scores [owner_row, source_rank, topk]
//
// Every source writes its disjoint source-rank column directly. The owner sees
// the exact rank-major table expected by the NCCL oracle.
template <int world_size>
__global__ void __launch_bounds__(512, 1) stage_owner_candidates_kernel(
    const int4 *__restrict__ local_indices,
    const int4 *__restrict__ local_scores, RankStaging staging,
    RankSignals signals, Signal *self, int rank, int rows, int topk,
    int64_t candidate_plane_packs, bool wait_for_prior_consumer) {
  const int owner_rows = rows / world_size;
  const int packs_per_row = topk / 4;
  const int64_t owner_packs = int64_t(owner_rows) * packs_per_row;
  const int64_t output_row_packs = int64_t(world_size) * packs_per_row;
  int64_t begin, end;
  block_range(owner_packs, begin, end);

  // A captured graph reuses one fixed staging address on every replay. Each
  // rank reaches this kernel only after its previous same-stream owner
  // consumer, so this group barrier prevents a faster peer from overwriting a
  // slower owner's slab before that consumer retires.
  if (wait_for_prior_consumer) {
    block_pair_barrier<world_size>(signals, self, rank);
  }

#pragma unroll 1
  for (int step = 0; step < world_size; ++step) {
    const int destination = (rank + step) % world_size;
    auto *destination_indices =
        reinterpret_cast<int4 *>(staging.ptrs[destination]);
    auto *destination_scores = destination_indices + candidate_plane_packs;
    const int64_t input_offset = int64_t(destination) * owner_packs;
    for (int64_t pack = begin + threadIdx.x; pack < end;
         pack += blockDim.x) {
      const int64_t owner_row = pack / packs_per_row;
      const int64_t column_pack = pack - owner_row * packs_per_row;
      const int64_t output_offset =
          owner_row * output_row_packs + int64_t(rank) * packs_per_row +
          column_pack;
      destination_indices[output_offset] = local_indices[input_offset + pack];
      destination_scores[output_offset] = local_scores[input_offset + pack];
    }
  }

  block_pair_barrier<world_size>(signals, self, rank);
}

static void validate_launch(int world_size, int threads, int block_limit) {
  if (threads < world_size || threads > 512 || threads % 32 != 0) {
    throw std::runtime_error(
        "threads must be a multiple of 32 in [32, 512]");
  }
  if (block_limit <= 0 || block_limit > kMaxBlocks) {
    throw std::runtime_error("invalid block limit");
  }
}

class PCIeDCPTopKOwnerExchange {
 public:
  int rank_;
  int world_size_;
  int max_rows_;
  int max_owner_rows_;
  int topk_;
  int64_t candidate_plane_elems_;
  RankSignals signals_{};
  Signal *self_signal_;
  RankStaging candidates_[2]{};
  uint32_t next_slot_ = 0;

  PCIeDCPTopKOwnerExchange(
      Signal **signals,
      const std::vector<std::array<void *, 2>> &candidate_staging,
      int max_rows, int topk, int rank, int world_size)
      : rank_(rank), world_size_(world_size), max_rows_(max_rows),
        max_owner_rows_(max_rows / world_size), topk_(topk),
        candidate_plane_elems_(int64_t(max_owner_rows_) * world_size * topk),
        self_signal_(signals[rank]) {
    for (int peer = 0; peer < world_size_; ++peer) {
      signals_.signals[peer] = signals[peer];
      candidates_[0].ptrs[peer] = candidate_staging[peer][0];
      candidates_[1].ptrs[peer] = candidate_staging[peer][1];
    }
  }

  int stage(cudaStream_t stream, const int *local_indices,
            const float *local_scores, int rows, int threads,
            int block_limit) {
    if (rows <= 0 || rows > max_rows_ || rows % world_size_ != 0) {
      throw std::runtime_error(
          "rows must fit capacity and be divisible by world size");
    }
    validate_launch(world_size_, threads, block_limit);
    const int64_t owner_packs =
        int64_t(rows / world_size_) * (topk_ / 4);
    const int blocks = int(std::max<int64_t>(
        1, std::min<int64_t>(block_limit,
                             (owner_packs + threads - 1) / threads)));
    // Host-side slot selection executes once during CUDA graph capture. Graph
    // replay intentionally reuses that capture-stable address: the channel is
    // stream-affine and the owner consumer is ordered after this kernel in the
    // same graph/stream. Eager calls toggle slabs without an overflowing
    // counter so one call cannot overwrite the immediately preceding result.
    const int slot = static_cast<int>(next_slot_);
    next_slot_ ^= uint32_t{1};
    const int64_t candidate_plane_packs = candidate_plane_elems_ / 4;
    cudaStreamCaptureStatus capture_status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &capture_status));
    const bool wait_for_prior_consumer =
        capture_status != cudaStreamCaptureStatusNone;

#define LAUNCH(world)                                                        \
  stage_owner_candidates_kernel<world><<<blocks, threads, 0, stream>>>(     \
      reinterpret_cast<const int4 *>(local_indices),                        \
      reinterpret_cast<const int4 *>(local_scores), candidates_[slot],      \
      signals_, self_signal_, rank_, rows, topk_, candidate_plane_packs,     \
      wait_for_prior_consumer)
    switch (world_size_) {
      case 2:
        LAUNCH(2);
        break;
      case 3:
        LAUNCH(3);
        break;
      case 4:
        LAUNCH(4);
        break;
      case 6:
        LAUNCH(6);
        break;
      case 8:
        LAUNCH(8);
        break;
      default:
        throw std::runtime_error("unsupported DCP top-k world size");
    }
#undef LAUNCH
    CHECK_CUDA_SUCCESS(cudaGetLastError());
    return slot;
  }

  void *local_candidate_indices(int slot) const {
    return candidates_[slot].ptrs[rank_];
  }

  void *local_candidate_scores(int slot) const {
    return static_cast<void *>(
        static_cast<uint8_t *>(candidates_[slot].ptrs[rank_]) +
        candidate_plane_elems_ * sizeof(int32_t));
  }
};

}  // namespace pcie_dcp_topk

using fptr_t = int64_t;

static fptr_t init_dcp_topk_owner_exchange(
    const std::vector<fptr_t> &signal_ptrs,
    const std::vector<fptr_t> &candidate0_ptrs,
    const std::vector<fptr_t> &candidate1_ptrs, int64_t max_rows,
    int64_t topk, int64_t rank) {
  const int world_size = int(signal_ptrs.size());
  TORCH_CHECK(world_size == 2 || world_size == 3 || world_size == 4 ||
              world_size == 6 || world_size == 8);
  TORCH_CHECK_EQ(candidate0_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK_EQ(candidate1_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK(rank >= 0 && rank < world_size);
  TORCH_CHECK(max_rows > 0 && max_rows % world_size == 0);
  TORCH_CHECK(topk > 0 && topk % 4 == 0);

  pcie_dcp_topk::Signal *signals[pcie_dcp_topk::kMaxRanks];
  std::vector<std::array<void *, 2>> candidates(world_size);
  for (int peer = 0; peer < world_size; ++peer) {
    signals[peer] =
        reinterpret_cast<pcie_dcp_topk::Signal *>(signal_ptrs[peer]);
    candidates[peer] = {reinterpret_cast<void *>(candidate0_ptrs[peer]),
                        reinterpret_cast<void *>(candidate1_ptrs[peer])};
  }
  return reinterpret_cast<fptr_t>(
      new pcie_dcp_topk::PCIeDCPTopKOwnerExchange(
          signals, candidates, int(max_rows), int(topk), int(rank),
          world_size));
}

static std::vector<torch::Tensor> stage_owner_candidates(
    fptr_t pointer, torch::Tensor &local_indices, torch::Tensor &local_scores,
    int64_t threads, int64_t block_limit) {
  auto *runtime =
      reinterpret_cast<pcie_dcp_topk::PCIeDCPTopKOwnerExchange *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_indices));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(local_indices.is_cuda() && local_scores.is_cuda());
  TORCH_CHECK(local_scores.device() == local_indices.device());
  TORCH_CHECK(local_indices.is_contiguous() && local_scores.is_contiguous());
  TORCH_CHECK_EQ(local_indices.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(local_scores.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(local_indices.dim(), 2);
  TORCH_CHECK_EQ(local_scores.sizes(), local_indices.sizes());
  TORCH_CHECK_EQ(local_indices.size(1), runtime->topk_);
  const int64_t rows = local_indices.size(0);
  TORCH_CHECK_GT(rows, 0);
  TORCH_CHECK_EQ(rows % runtime->world_size_, 0);

  const int slot = runtime->stage(
      stream, reinterpret_cast<const int *>(local_indices.data_ptr()),
      reinterpret_cast<const float *>(local_scores.data_ptr()), int(rows),
      int(threads), int(block_limit));
  const int64_t owner_rows = rows / runtime->world_size_;
  const int64_t candidate_width = runtime->world_size_ * runtime->topk_;
  auto no_delete = [](void *) {};
  auto indices = torch::from_blob(
      runtime->local_candidate_indices(slot), {owner_rows, candidate_width},
      no_delete, local_indices.options());
  auto scores = torch::from_blob(
      runtime->local_candidate_scores(slot), {owner_rows, candidate_width},
      no_delete, local_scores.options());
  return {indices, scores};
}

static void dispose_owner_exchange(fptr_t pointer) {
  delete reinterpret_cast<pcie_dcp_topk::PCIeDCPTopKOwnerExchange *>(pointer);
}

static int64_t meta_size() { return sizeof(pcie_dcp_topk::Signal); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("init_dcp_topk_owner_exchange", &init_dcp_topk_owner_exchange,
             "initialize exact DCP top-k owner exchange");
  module.def("stage_owner_candidates", &stage_owner_candidates,
             "stage exact rank-major candidates on each row owner");
  module.def("dispose_owner_exchange", &dispose_owner_exchange,
             "dispose exact DCP top-k owner exchange");
  module.def("meta_size", &meta_size, "signal metadata size");
}
