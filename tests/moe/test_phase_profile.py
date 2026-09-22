"""Balanced placement preserves raw observations and rejects undeclared weights."""

from copy import deepcopy

import pytest

from b12x.integration.vllm.expert_cache import digest
from b12x.integration.vllm.phase_profile import validate
from b12x.moe.fused_moe.residency import balanced_phase_profile
from benchmarks.moe.balance_expert_profile import build
from tests.moe.test_expert_cache_capacity import fixture


def test_normalization_does_not_let_long_prefill_dominate():
    value = balanced_phase_profile(
        prefill_counts=[900, 100, 0],
        decode_counts=[0, 1, 9],
        hot_count=2,
        layer="a",
        model_fingerprint="b",
        workload="calibration",
        provenance="test",
    )
    assert value.hbm_expert_ids == (0, 2)
    assert value.selection_counts == (900, 101, 9)
    assert value.phase == "all"
    tied = balanced_phase_profile(
        prefill_counts=[1, 0],
        decode_counts=[0, 1],
        hot_count=1,
        layer="a",
        model_fingerprint="b",
        workload="c",
        provenance="test",
    )
    assert tied.hbm_expert_ids == (0,)


@pytest.mark.parametrize(
    "prefill,decode", [([0, 0], [1, 2]), ([1], [1, 2]), ([-1, 2], [1, 2])]
)
def test_incomplete_or_invalid_phase_counts_rejected(prefill, decode):
    with pytest.raises(ValueError):
        balanced_phase_profile(
            prefill_counts=prefill,
            decode_counts=decode,
            hot_count=1,
            layer="a",
            model_fingerprint="b",
            workload="c",
            provenance="test",
        )


def test_artifact_is_receipt_bound_and_objective_hashed():
    profile, receipt, _ = fixture()
    rows = receipt[0]["result"][0]["snapshot"]["layers"]
    rows[0]["phase"] = "decode"
    rows.append(dict(layer="layer", phase="prefill", counts=[100, 10, 0, 10]))
    profile["phase_counts"] = deepcopy(rows)
    profile["hash"] = digest({k: v for k, v in profile.items() if k != "hash"})
    receipt[0]["result"][0]["hash"] = profile["hash"]
    original = deepcopy(profile)
    result = build(profile, receipt)
    assert profile == original
    assert result["hash"] != profile["hash"]
    assert result["identity"] == profile["identity"]
    assert len(result["placements"]["layer"]["hbm_expert_ids"]) == 2
    validate(result)
    result["placement_objective"]["prefill_weight"] = [3, 4]
    with pytest.raises(ValueError, match="objective"):
        validate(result)
    profile["phase_counts"][1]["counts"][0] += 1
    profile["hash"] = digest({k: v for k, v in profile.items() if k != "hash"})
    receipt[0]["result"][0]["hash"] = profile["hash"]
    with pytest.raises(ValueError, match="differ"):
        build(profile, receipt)
