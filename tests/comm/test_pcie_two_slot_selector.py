from __future__ import annotations

import subprocess
from pathlib import Path


def test_two_slot_selector_preserves_phase_for_long_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "two_slot_selector_test.cpp"
    binary = tmp_path / "two_slot_selector_test"
    source.write_text(
        r"""
#include <cstdint>
#include "sparkinfer/comm/pcie/two_slot_selector.h"

int main() {
  sparkinfer::pcie::TwoSlotSelector selector;
  constexpr std::uint64_t kTransitions = 10'000'003;
  for (std::uint64_t call = 0; call < kTransitions; ++call) {
    if (selector.take() != static_cast<int>(call & 1u)) return 1;
  }
  return selector.take() == static_cast<int>(kTransitions & 1u) ? 0 : 2;
}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(repo_root),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True)
