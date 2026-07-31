#pragma once

#include <algorithm>
#include <limits>

#ifdef __CUDACC__
#include <array>
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace pcie {

// A device-side grid rendezvous can only make forward progress when every CTA
// in the grid can be resident at once. Keep the arithmetic independent from
// CUDA so its overflow and boundary behavior can be unit-tested on CPU hosts.
constexpr int resident_grid_capacity(int active_blocks_per_sm, int sm_count) {
  if (active_blocks_per_sm <= 0 || sm_count <= 0) return 0;
  if (active_blocks_per_sm > std::numeric_limits<int>::max() / sm_count)
    return std::numeric_limits<int>::max();
  return active_blocks_per_sm * sm_count;
}

constexpr int clamp_resident_grid_size(
    int requested_blocks, int active_blocks_per_sm, int sm_count) {
  const int capacity = resident_grid_capacity(active_blocks_per_sm, sm_count);
  if (requested_blocks <= 0 || capacity <= 0) return 0;
  return std::min(requested_blocks, capacity);
}

#ifdef __CUDACC__
// A cooperative launch makes the occupancy bound an admission contract rather
// than a best-effort calculation. Since CUDA 11.2, multiple cooperative grids
// may overlap, but the scheduler admits them without partially resident grids;
// that is the property required by the in-kernel staging rendezvous.
template <typename Kernel, typename... Args>
cudaError_t launch_cooperative(
    Kernel kernel,
    int blocks,
    int threads,
    cudaStream_t stream,
    Args... args) {
  std::array<void*, sizeof...(Args)> kernel_args{
      static_cast<void*>(&args)...};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      kernel_args.data(),
      0,
      stream);
}
#endif

}  // namespace pcie
}  // namespace sparkinfer
