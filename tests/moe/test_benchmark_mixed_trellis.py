from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")


def _load_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "benchmark_mixed_trellis.py"
    spec = importlib.util.spec_from_file_location("benchmark_mixed_trellis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Tier:
    num_experts: int
    trellis_bits: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


def test_benchmark_materializes_full_abi5_tier_storage(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "HIDDEN", 32)
    monkeypatch.setattr(benchmark, "INTERMEDIATE", 16)

    materialized = 2
    compiled = 3
    bits = 3
    w13 = torch.arange(2 * materialized * 2 * 1 * 8 * bits, dtype=torch.int32)
    w2 = torch.arange(materialized * 1 * 2 * 8 * bits, dtype=torch.int32)
    tier = _Tier(
        num_experts=materialized,
        trellis_bits=bits,
        w13=w13,
        w2=w2,
        w13_global_scale=torch.tensor([2.0, 3.0]),
        w2_global_scale=torch.tensor([4.0, 5.0]),
    )

    padded = benchmark._pad_exact_tier_storage(tier, compiled)

    assert padded.num_experts == compiled
    padded_w13 = padded.w13.view(2, compiled, 2, 1, 8 * bits)
    assert torch.equal(
        padded_w13[:, :materialized], w13.view(2, materialized, 2, 1, 8 * bits)
    )
    assert torch.count_nonzero(padded_w13[:, materialized:]) == 0
    padded_w2 = padded.w2.view(compiled, 1, 2, 8 * bits)
    assert torch.equal(padded_w2[:materialized], w2.view(materialized, 1, 2, 8 * bits))
    assert torch.count_nonzero(padded_w2[materialized:]) == 0
    assert torch.equal(padded.w13_global_scale, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(padded.w2_global_scale, torch.tensor([4.0, 5.0, 1.0]))
