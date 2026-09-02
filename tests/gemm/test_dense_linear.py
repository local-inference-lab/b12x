"""Host-side contracts for the profiled dense linear launch policy."""

from __future__ import annotations

import pytest

from b12x._lib.dense_gemm import _select_default_dense_gemm_plan
from b12x.gemm.blockscaled._linear import _dense_launch_kwargs
from b12x.gemm.dense_linear._policy import (
    DENSE_LINEAR_POLICY,
    DENSE_LINEAR_RECIPES,
    DenseLinearConfig,
    DenseLinearQuery,
    _heuristic,
    _validate,
    check_tile,
    recipe_planner_flags,
)
from b12x.policy import DeviceIdentity
from b12x.policy.generation.providers.gemm import _dense_linear_candidate_configs

_GB10 = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="NVIDIA GB10",
)


def _query(
    recipe: str, *, k: int = 4096, n: int = 4096, m: int = 4
) -> DenseLinearQuery:
    return DenseLinearQuery(
        recipe=recipe,
        in_features=k,
        out_features=n,
        max_tokens=m,
        output_dtype="bfloat16",
    )


def _config(**overrides: object) -> DenseLinearConfig:
    base: dict[str, object] = {
        "backend": "dense",
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 0,
        "load_path": "tma",
        "swap_ab": False,
        "split_k": 0,
    }
    base.update(overrides)
    return DenseLinearConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("recipe", DENSE_LINEAR_RECIPES)
@pytest.mark.parametrize("max_tokens", [1, 8, 64, 2048])
def test_heuristic_reproduces_builtin_planner(recipe: str, max_tokens: int) -> None:
    """An uncovered query must launch exactly what the runtime launches today."""
    query = _query(recipe, m=max_tokens)
    config = _heuristic(query, _GB10)
    plan = _select_default_dense_gemm_plan(
        max_tokens,
        query.out_features,
        query.in_features,
        _GB10.sm_count,
        expected_m=max_tokens,
        **recipe_planner_flags(recipe),
    )
    assert config.mma_tiler_mn == plan.mma_tiler_mn
    assert config.load_path == plan.load_path
    assert config.swap_ab == plan.swap_ab
    assert config.tile_k == 0 and config.split_k == 0
    _validate(query, config, _GB10)


@pytest.mark.parametrize(
    ("recipe", "overrides", "message"),
    [
        ("mxfp8", {"load_path": "cpasync"}, "cp.async"),
        ("mxfp4", {"tile_m": 64, "tile_n": 64, "swap_ab": True}, "swapped"),
        ("nvfp4", {"tile_m": 64, "tile_n": 16}, "MMA tile"),
        ("nvfp4", {"tile_m": 16, "tile_n": 64}, "MMA tile"),
        ("block_fp8", {"tile_k": 128}, "K tile"),
        ("nvfp4", {"split_k": 2}, "split-K"),
        ("mxfp8", {"tile_k": 256}, "divide"),
        ("nvfp4", {"load_path": "cpasync", "tile_m": 64, "tile_n": 64}, "K <="),
    ],
)
def test_validate_rejects_unsupported_launches(
    recipe: str, overrides: dict[str, object], message: str
) -> None:
    """Profile entries the kernel cannot implement fail closed at resolution."""
    k = 4096
    if "tile_k" in overrides and recipe == "mxfp8":
        k = 4096 + 128
    if overrides.get("load_path") == "cpasync" and recipe == "nvfp4":
        k = 16384
    query = _query(recipe, k=k)
    with pytest.raises(ValueError, match=message):
        _validate(query, _config(**overrides), _GB10)


def test_swapped_tiles_follow_the_kernel_convention() -> None:
    """Swapped plans are written transposed, exactly as the planner emits them."""
    check_tile("mxfp8", (64, 16), True)
    check_tile("nvfp4", (128, 32), True)
    check_tile("mxfp8", (16, 64), False)
    with pytest.raises(ValueError):
        check_tile("mxfp8", (16, 64), True)
    with pytest.raises(ValueError):
        check_tile("block_fp8", (64, 16), True)


@pytest.mark.parametrize(
    "recipe", ["nvfp4", "mxfp4", "mxfp8", "tensor_fp8", "block_fp8"]
)
@pytest.mark.parametrize("max_tokens", [1, 16, 256, 2048])
def test_generator_candidates_are_all_valid(recipe: str, max_tokens: int) -> None:
    """Every raced candidate must be a config the policy would accept back."""
    query = _query(recipe, m=max_tokens)
    candidates = _dense_linear_candidate_configs(
        recipe=recipe, in_features=query.in_features, max_tokens=max_tokens
    )
    assert candidates
    assert len({tuple(sorted(c.items())) for c in candidates}) == len(candidates)
    for candidate in candidates:
        _validate(query, DenseLinearConfig.from_profile(candidate), _GB10)


def test_launch_kwargs_round_trip_through_codes() -> None:
    """Custom-op integer codes must recreate the dense_gemm keyword overrides."""
    config = _config(
        tile_m=64, tile_n=32, tile_k=256, load_path="cpasync", swap_ab=True, split_k=0
    )
    kwargs = _dense_launch_kwargs(
        config.tile_m, config.tile_n, config.tile_k, config.split_k, 2, 2
    )
    assert kwargs == {
        "mma_tiler_mn": (64, 32),
        "_tile_k_override": 256,
        "load_path": "cpasync",
        "swap_ab": True,
    }
    assert _dense_launch_kwargs(0, 0, 0, 0, 0, 0) == {}


def test_profile_round_trip_preserves_every_field() -> None:
    config = _config(tile_m=16, tile_n=128, split_k=4)
    assert DenseLinearConfig.from_profile(config.to_dict()) == config
    assert DENSE_LINEAR_POLICY.config_schema_version == 1
