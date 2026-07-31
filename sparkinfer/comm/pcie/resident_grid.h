#pragma once

#include <algorithm>
#include <limits>

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

}  // namespace pcie
}  // namespace sparkinfer
