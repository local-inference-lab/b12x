#pragma once

#include <map>
#include <utility>

namespace sparkinfer::pcie {

// Owns imported IPC handles whose close routine reports an error code instead
// of throwing. Successful closes clear their entries so repeated cleanup is
// idempotent; failed entries remain available to an explicit retry.
template <typename Key, typename Handle, typename Error, Error Success>
class IpcHandleRegistry {
 public:
  using CloseFn = Error (*)(Handle) noexcept;
  using Map = std::map<Key, Handle>;
  using iterator = typename Map::iterator;

  explicit IpcHandleRegistry(CloseFn close) noexcept : close_(close) {}

  IpcHandleRegistry(const IpcHandleRegistry&) = delete;
  IpcHandleRegistry& operator=(const IpcHandleRegistry&) = delete;
  IpcHandleRegistry(IpcHandleRegistry&&) = delete;
  IpcHandleRegistry& operator=(IpcHandleRegistry&&) = delete;

  ~IpcHandleRegistry() noexcept {
    (void)close_all_noexcept();
  }

  iterator find(const Key& key) {
    return handles_.find(key);
  }

  iterator end() {
    return handles_.end();
  }

  std::pair<iterator, bool> emplace(const Key& key, Handle handle) {
    return handles_.emplace(key, handle);
  }

  Error close_all_noexcept() noexcept {
    Error first_error = Success;
    for (auto& entry : handles_) {
      Handle& handle = entry.second;
      if (handle == Handle{}) {
        continue;
      }
      const Error error = close_(handle);
      if (error == Success) {
        handle = Handle{};
      } else if (first_error == Success) {
        first_error = error;
      }
    }
    return first_error;
  }

 private:
  Map handles_;
  CloseFn close_;
};

}  // namespace sparkinfer::pcie
