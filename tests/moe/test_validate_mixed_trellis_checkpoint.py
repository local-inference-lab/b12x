from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")


def _load_validator():
    path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "validate_mixed_trellis_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_mixed_trellis_checkpoint", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tier(*, shared: bool, offset: float = 0.0):
    h_rows = 1 if shared else 2
    return SimpleNamespace(
        gate_suh=torch.full((h_rows, 8), 1.0 + offset, dtype=torch.float16),
        up_suh=torch.full((h_rows, 8), 2.0 + offset, dtype=torch.float16),
        down_svh=torch.full((h_rows, 8), 3.0 + offset, dtype=torch.float16),
        intermediate_rotations=torch.full((2, 12), 4.0 + offset),
    )


def test_combine_rotations_preserves_one_physical_h_row() -> None:
    validator = _load_validator()
    rotations, broadcast_suh, broadcast_svh = validator._combine_rotations(
        _tier(shared=True), _tier(shared=True)
    )
    assert broadcast_suh is True
    assert broadcast_svh is True
    assert rotations.gate_suh.shape == (1, 8)
    assert rotations.up_suh.shape == (1, 8)
    assert rotations.down_svh.shape == (1, 8)
    assert rotations.intermediate.shape == (4, 12)


def test_combine_rotations_rejects_different_shared_rows() -> None:
    validator = _load_validator()
    with pytest.raises(ValueError, match="shared SUH rows differ"):
        validator._combine_rotations(_tier(shared=True), _tier(shared=True, offset=1.0))
