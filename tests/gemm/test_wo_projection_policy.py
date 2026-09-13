from __future__ import annotations

from dataclasses import replace

import pytest

from b12x.gemm._shared.wo_mxfp8 import _should_use_exact_b16_wo
from b12x.gemm.wo_projection._policy import (
    WO_PROJECTION_POLICY,
    WoProjectionConfig,
    WoProjectionQuery,
)
from b12x.policy import DeviceIdentity, FrozenMapping, PolicyContext, PolicySource


_DECODE = WoProjectionQuery(
    dtype="bfloat16", max_tokens=8, groups=2, group_width=4096, rank=1024, hidden=5120
)


def test_exact_b16_wo_is_spark_only() -> None:
    assert _should_use_exact_b16_wo(tokens=16, sm_count=20)
    assert not _should_use_exact_b16_wo(tokens=16, sm_count=188)
    assert not _should_use_exact_b16_wo(tokens=8, sm_count=20)


@pytest.mark.parametrize("tile", (0, 64, 128))
def test_decode_tile_roundtrip_and_explicit_override(tile):
    config = WoProjectionConfig(decode_tile_n=tile)
    assert WoProjectionConfig.from_profile(FrozenMapping(config.to_dict())) == config
    resolved = (
        PolicyContext(device=None)
        .with_override(WO_PROJECTION_POLICY.component_id, config)
        .resolve(WO_PROJECTION_POLICY, _DECODE)
    )
    assert resolved.config == config


@pytest.mark.parametrize(
    "payload",
    (
        {"backend": "mxfp8"},
        {"backend": "mxfp8", "decode_tile_n": True},
        {"backend": "mxfp8", "decode_tile_n": "64"},
        {"backend": "mxfp8", "decode_tile_n": 64, "extra": 1},
    ),
)
def test_malformed_decode_profile_fails_closed(payload):
    with pytest.raises((TypeError, ValueError)):
        WoProjectionConfig.from_profile(FrozenMapping(payload))


@pytest.mark.parametrize(
    "change",
    (
        {"max_tokens": 9},
        {"groups": 4},
        {"hidden": 4096},
        {"dtype": "float16"},
    ),
)
def test_decode_tile_cannot_escape_qualified_geometry(change):
    policy = PolicyContext(device=None).with_override(
        WO_PROJECTION_POLICY.component_id, WoProjectionConfig(decode_tile_n=64)
    )
    with pytest.raises(ValueError, match="decode domain"):
        policy.resolve(WO_PROJECTION_POLICY, replace(_DECODE, **change))


def test_embedded_and_unknown_device_resolution_preserves_policy_modes():
    known = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    )
    result = PolicyContext(device=known).resolve(WO_PROJECTION_POLICY, _DECODE)
    assert result.source == PolicySource.PREPLANNED
    unknown = replace(known, product_name="Synthetic unmeasured GPU")
    result = PolicyContext(device=unknown).resolve(WO_PROJECTION_POLICY, _DECODE)
    assert result.source == PolicySource.HEURISTIC
    assert result.config.decode_tile_n == 0


def test_profile_race_covers_each_live_decode_row_and_scale_layout():
    from b12x.policy.generation.providers.wo_projection import _Session, _cases

    cases = [case for case in _cases() if case.query["group_width"] == 4096]
    assert {
        (case.metadata["tokens"], case.metadata["block_size"]) for case in cases
    } == {(rows, block) for rows in range(1, 9) for block in (32, 128)}
    assert {case.query["max_tokens"] for case in cases} == {8}
    for case in cases:
        assert {c.config["decode_tile_n"] for c in _Session(None).candidates(case)} == {
            0,
            64,
            128,
        }
