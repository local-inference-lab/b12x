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


def _packaged_paths(root: Path, patterns: list[str]) -> set[Path]:
    return {
        candidate.resolve()
        for pattern in patterns
        for candidate in root.glob(pattern)
    }


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert set(package_data[PCIE_PACKAGE]) == PCIE_PACKAGE_PATTERNS
    assert {
        path.name for path in PCIE_SOURCE_DIR.glob("*.cu")
    } >= RUNTIME_CUDA_SOURCES


def test_runtime_cuda_local_includes_are_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_patterns = config["tool"]["setuptools"]["package-data"][PCIE_PACKAGE]
    packaged_paths = _packaged_paths(PCIE_SOURCE_DIR, package_patterns)

    for source in PCIE_SOURCE_DIR.glob("*.cu"):
        for include in LOCAL_INCLUDE_RE.findall(source.read_text(encoding="utf-8")):
            dependency = source.parent / include
            assert dependency.is_file(), f"{source.name} includes missing {include}"
            assert dependency.resolve() in packaged_paths, (
                f"{source.name} includes {include}, but {include} is not package data"
            )


def test_package_root_glob_does_not_hide_nested_header_omissions(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pcie"
    nested_header = package_root / "nested" / "runtime.h"
    nested_header.parent.mkdir(parents=True)
    nested_header.write_text("#pragma once\n", encoding="utf-8")

    assert nested_header.resolve() not in _packaged_paths(package_root, ["*.h"])
    assert nested_header.resolve() in _packaged_paths(package_root, ["**/*.h"])
