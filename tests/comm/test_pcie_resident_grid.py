from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PCIE = ROOT / "sparkinfer" / "comm" / "pcie"


def test_resident_grid_arithmetic_compiles_and_handles_boundaries(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the header-only contract test")

    source = tmp_path / "resident_grid_test.cc"
    source.write_text(
        """
#include <limits>
#include "sparkinfer/comm/pcie/resident_grid.h"

using sparkinfer::pcie::clamp_resident_grid_size;
using sparkinfer::pcie::resident_grid_capacity;

static_assert(resident_grid_capacity(2, 14) == 28);
static_assert(resident_grid_capacity(0, 14) == 0);
static_assert(resident_grid_capacity(2, 0) == 0);
static_assert(
    resident_grid_capacity(std::numeric_limits<int>::max(), 2) ==
    std::numeric_limits<int>::max());
static_assert(clamp_resident_grid_size(64, 2, 14) == 28);
static_assert(clamp_resident_grid_size(8, 2, 14) == 8);
static_assert(clamp_resident_grid_size(0, 2, 14) == 0);

int main() { return 0; }
""",
        encoding="utf-8",
    )
    binary = tmp_path / "resident_grid_test"
    subprocess.run(
        [compiler, "-std=c++17", f"-I{ROOT}", str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize(
    ("source_name", "kernels", "guarded_launches"),
    (
        (
            "pcie_oneshot.cu",
            ("pcie_allreduce_kernel", "fused_resident_grid_capacity"),
            2,
        ),
        ("pcie_twoshot.cu", ("rs_fp8_kernel", "ag_fp8_kernel"), 2),
    ),
)
def test_grid_rendezvous_launches_have_occupancy_guards(
    source_name: str, kernels: tuple[str, ...], guarded_launches: int
) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    assert "cudaDevAttrMultiProcessorCount" in source
    for kernel in kernels:
        assert kernel in source
    assert source.count("<<<resident_blocks") == guarded_launches
    assert "resident grid capacity" in source

    if source_name == "pcie_oneshot.cu":
        assert (
            "ctas_per_row = std::min(ctas_per_row, resident_capacity / rows);" in source
        )
        assert "rows > resident_capacity" in source
