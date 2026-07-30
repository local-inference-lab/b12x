#pragma once

#include <cstdint>

namespace sparkinfer::pcie {

// A double-buffer selector with an explicitly finite state space. Keeping the
// phase as an enum makes signed overflow and out-of-range remainders impossible.
class TwoSlotSelector {
 public:
  constexpr int take() noexcept {
    if (next_ == Next::kFirst) {
      next_ = Next::kSecond;
      return 0;
    }
    next_ = Next::kFirst;
    return 1;
  }

 private:
  enum class Next : std::uint32_t { kFirst, kSecond };
  Next next_ = Next::kFirst;
};

namespace detail {
constexpr bool two_slot_sequence_is_preserved() {
  TwoSlotSelector selector;
  return selector.take() == 0 && selector.take() == 1 &&
         selector.take() == 0 && selector.take() == 1;
}
}  // namespace detail

static_assert(detail::two_slot_sequence_is_preserved(),
              "two-slot selection must start at zero and alternate");
static_assert(sizeof(TwoSlotSelector) == sizeof(std::uint32_t),
              "two-slot selector storage must remain 32-bit");

}  // namespace sparkinfer::pcie
