from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_PACKAGE = "sparkinfer.comm.pcie"
TRELLIS_PACKAGE = "sparkinfer.gemm.trellis_linear"
TRELLIS_SOURCE_DIR = ROOT / "sparkinfer" / "gemm" / "trellis_linear" / "csrc"
RUNTIME_CUDA_SOURCES = {
    "pcie_dcp_a2a.cu",
    "pcie_dcp_topk.cu",
    "pcie_dma.cu",
    "pcie_hierarchical.cu",
    "pcie_oneshot.cu",
    "pcie_twoshot.cu",
}


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert package_data[PCIE_PACKAGE] == ["*.cu"]
    packaged_sources = {
        path.name for path in (ROOT / "sparkinfer" / "comm" / "pcie").glob("*.cu")
    }
    assert packaged_sources >= RUNTIME_CUDA_SOURCES


def test_trellis_runtime_sources_and_license_are_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"][TRELLIS_PACKAGE]
    required = [
        TRELLIS_SOURCE_DIR / "trellis_k6_small.cu",
        TRELLIS_SOURCE_DIR / "vendor" / "LICENSE.exllamav3",
        TRELLIS_SOURCE_DIR / "vendor" / "quant" / "exl3_gemm_kernel.cuh",
    ]

    for path in required:
        assert path.is_file()
        relative = path.relative_to(TRELLIS_SOURCE_DIR.parent)
        assert any(relative.match(pattern) for pattern in patterns), (
            f"{relative} is required by the Trellis JIT but is not package data"
        )
