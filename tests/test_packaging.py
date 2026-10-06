from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_SOURCE = ROOT / "b12x" / "comm" / "pcie"


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


def test_wheel_runtime_locks_match_required_cutlass_family() -> None:
    """The bundle manifest must declare the kernel runtime required by its wheel."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    versions = {
        dependency.split("==", 1)[1]
        for dependency in config["project"]["dependencies"]
        if dependency.startswith("nvidia-cutlass-dsl")
    }
    assert len(versions) == 1
    for path in (ROOT / "ci" / "lil_wheels").rglob("runtime.lock"):
        contract = dict(line.split("=", 1) for line in path.read_text().splitlines())
        assert contract["cutlass-dsl.version"] in versions, path
