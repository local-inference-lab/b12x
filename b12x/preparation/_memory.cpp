#include <pybind11/pybind11.h>
#include <c10/cuda/CUDACachingAllocator.h>

int64_t allocated_bytes(int device) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    constexpr auto aggregate = static_cast<size_t>(
        c10::CachingAllocator::StatType::AGGREGATE);
    return stats.allocated_bytes[aggregate].current;
}

void release_graph_pool_cache(uint64_t first, uint64_t second) {
    if (first == 0 && second == 0) {
        throw std::invalid_argument("graph pool cleanup requires a private pool");
    }
    c10::cuda::CUDACachingAllocator::emptyCache({first, second});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("allocated_bytes", &allocated_bytes);
    module.def("release_graph_pool_cache", &release_graph_pool_cache);
}
