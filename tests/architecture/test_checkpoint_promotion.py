"""A checkpoint graph promotion must move an expert selected by that graph."""

import importlib.util
from pathlib import Path

import pytest


def test_real_route_promotion_selects_observed_cold_expert():
    path = Path(__file__).parents[1] / "moe/test_next80_checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    choose = oracle.observed_cold_candidate
    assert choose([0, 2, 2, 9], 8) == 9
    assert choose([0, 2, 2, 9], 1) == 9
    for ids in ([], [0, 0], [1, 7]):
        with pytest.raises(ValueError, match="do not exercise"):
            choose(ids, 8)
