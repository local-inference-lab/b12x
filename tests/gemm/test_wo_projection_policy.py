from __future__ import annotations

import pytest

from b12x.gemm._shared.wo_mxfp8 import _should_use_exact_b16_wo, _wo_launch_codes
from b12x.gemm.wo_projection._policy import (
    WO_PROJECTION_POLICY,
    WoProjectionConfig,
    WoProjectionQuery,
    _heuristic,
    _validate,
)
from b12x.policy import DeviceIdentity
from b12x.policy.generation.providers.gemm import _wo_projection_candidate_configs

_GB10 = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="NVIDIA GB10",
)
_BIG = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 0),
    sm_count=188,
    product_name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)


def _query(max_tokens: int, *, groups: int = 4) -> WoProjectionQuery:
    # DSV4-Flash TP2: 4 local groups of 8 heads x 512 = group_width 4096.
    return WoProjectionQuery(
        dtype="bfloat16",
        max_tokens=max_tokens,
        groups=groups,
        group_width=4096,
        rank=1024,
        hidden=4096,
    )


def _config(**overrides: object) -> WoProjectionConfig:
    base: dict[str, object] = {
        "backend": "mxfp8",
        "wo_a_tile_m": 0,
        "wo_a_tile_n": 0,
        "wo_b_tile_m": 0,
        "wo_b_tile_n": 0,
        "wo_b_fused_quant": False,
        "quantized_intermediate": False,
    }
    base.update(overrides)
    return WoProjectionConfig(**base)  # type: ignore[arg-type]


def test_exact_b16_wo_is_spark_only() -> None:
    assert _should_use_exact_b16_wo(tokens=16, sm_count=20)
    assert not _should_use_exact_b16_wo(tokens=16, sm_count=188)
    assert not _should_use_exact_b16_wo(tokens=8, sm_count=20)


def test_heuristic_reproduces_runtime_chain_selection() -> None:
    """The heuristic must name the same WO-A/WO-B chain the fused op picks."""
    decode = _heuristic(_query(4), _GB10)
    assert decode.wo_a_tile == (16, 64)
    assert decode.wo_b_fused_quant and not decode.quantized_intermediate
    assert decode.wo_b_tile in ((16, 64), (16, 128))

    exact = _heuristic(_query(16), _GB10)
    assert exact.quantized_intermediate and not exact.wo_b_fused_quant
    assert exact.wo_a_tile == (32, 64) and exact.wo_b_tile == (32, 64)

    mid = _heuristic(_query(64), _GB10)
    assert mid == _config()
    assert not _heuristic(_query(16), _BIG).quantized_intermediate


@pytest.mark.parametrize(
    ("max_tokens", "overrides", "message"),
    [
        (64, {"wo_b_fused_quant": True}, "fused"),
        (32, {"quantized_intermediate": True}, "quantized"),
        (4, {"wo_b_fused_quant": True, "quantized_intermediate": True}, "exclusive"),
        (4, {"wo_a_tile_m": 48, "wo_a_tile_n": 64}, "tile"),
    ],
)
def test_validate_rejects_chains_the_runtime_cannot_launch(
    max_tokens: int, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_query(max_tokens), _config(**overrides), _GB10)


@pytest.mark.parametrize("max_tokens", [1, 8, 12, 16, 64, 2048])
def test_generator_candidates_are_all_valid(max_tokens: int) -> None:
    candidates = _wo_projection_candidate_configs(max_tokens)
    assert candidates
    for candidate in candidates:
        _validate(_query(max_tokens), WoProjectionConfig.from_profile(candidate), _GB10)


def test_launch_codes_zero_means_builtin_rules() -> None:
    assert _wo_launch_codes(None) == (0, 0, 0, 0, 0, 0)
    assert _wo_launch_codes(_config()) == (0, 0, 0, 0, 1, 1)
    codes = _wo_launch_codes(
        _config(
            wo_a_tile_m=16,
            wo_a_tile_n=64,
            wo_b_tile_m=16,
            wo_b_tile_n=128,
            wo_b_fused_quant=True,
        )
    )
    assert codes == (16, 64, 16, 128, 2, 1)
    with pytest.raises(TypeError):
        _wo_launch_codes(object())


def test_profile_round_trip_and_schema() -> None:
    config = _config(wo_a_tile_m=32, wo_a_tile_n=64, quantized_intermediate=True)
    assert WoProjectionConfig.from_profile(config.to_dict()) == config
    assert WO_PROJECTION_POLICY.config_schema_version == 2
