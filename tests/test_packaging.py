from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_SOURCE = ROOT / "b12x" / "comm" / "pcie"
CUTLASS_DSL_VERSION = "4.6.2"


def test_cutlass_dsl_packages_use_one_qualified_version() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])
    packages = (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    )

    for package in packages:
        assert f"{package}=={CUTLASS_DSL_VERSION}" in dependencies


def test_pcie_collectives_are_python_only() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"].get("package-data", {})

    assert "b12x.comm.pcie" not in package_data
    assert not list(PCIE_SOURCE.glob("*.cu"))
    assert not list(PCIE_SOURCE.glob("*.cuh"))
    assert not list(PCIE_SOURCE.glob("*.cpp"))
    for source in PCIE_SOURCE.glob("*.py"):
        text = source.read_text()
        assert "torch.utils.cpp_extension" not in text
        assert "cpp_extension.load" not in text
        assert "load_inline(" not in text
