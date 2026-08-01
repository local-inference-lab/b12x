from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_PACKAGE = "sparkinfer.comm.pcie"
PCIE_SOURCE_DIR = ROOT / "sparkinfer" / "comm" / "pcie"
PCIE_PACKAGE_PATTERNS = {"*.cu", "*.h"}
RUNTIME_CUDA_SOURCES = {
    "pcie_dcp_a2a.cu",
    "pcie_dcp_topk.cu",
    "pcie_dma.cu",
    "pcie_oneshot.cu",
    "pcie_twoshot.cu",
}
LOCAL_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert set(package_data[PCIE_PACKAGE]) == PCIE_PACKAGE_PATTERNS
    assert {
        path.name for path in PCIE_SOURCE_DIR.glob("*.cu")
    } == RUNTIME_CUDA_SOURCES


def test_runtime_cuda_local_includes_are_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_patterns = config["tool"]["setuptools"]["package-data"][PCIE_PACKAGE]

    for source in PCIE_SOURCE_DIR.glob("*.cu"):
        for include in LOCAL_INCLUDE_RE.findall(source.read_text(encoding="utf-8")):
            dependency = source.parent / include
            assert dependency.is_file(), f"{source.name} includes missing {include}"
            assert any(dependency.match(pattern) for pattern in package_patterns), (
                f"{source.name} includes {include}, but {include} is not package data"
            )
