"""Typed launch policy for the packed dense linear recipes.

The dense GEMM behind ``blockscaled.mm_nvfp4`` / ``mm_mxfp4`` / ``mm_block_fp8``,
``mxfp8_linear.mm`` and ``tensor_fp8_linear.mm`` owns several launch knobs
(MMA tile, K tile, load path, operand swap, split-K). This policy exposes
them as one profiled component so serving integrations resolve a measured
plan per (recipe, geometry, capacity) instead of the built-in heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    DENSE_LINEAR,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)

DENSE_LINEAR_BACKEND = "dense"
DENSE_LINEAR_RECIPES = ("nvfp4", "mxfp4", "mxfp8", "tensor_fp8", "block_fp8", "mxfp6")
# Tiles are expressed in the kernel's own convention: with ``swap_ab`` the
# planner transposes the tuned tile, so a swapped decode plan reads (64, 16).
FP8_TILES = (
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)
FP8_SWAPPED_TILES = ((64, 16), (64, 32), (128, 16), (128, 32))
FP4_TILES = ((64, 64), (64, 128), (128, 64), (128, 128))
FP4_NARROW_TILES = ((64, 16), (64, 32), (128, 16), (128, 32))
DENSE_LINEAR_TILES = tuple(
    dict.fromkeys(FP8_TILES + FP8_SWAPPED_TILES + FP4_TILES + FP4_NARROW_TILES)
)
DENSE_LINEAR_LOAD_PATHS = ("tma", "cpasync")
DENSE_LINEAR_TILE_K = (0, 128, 256, 512)
DENSE_LINEAR_SPLIT_K = (0, 1, 2, 4)
# Per-recipe kernel support (mirrors ``DenseGemmKernel.can_implement``).
_FP4_RECIPES = frozenset({"nvfp4", "mxfp4"})
_SWAP_RECIPES = frozenset({"nvfp4", "mxfp8"})
_CPASYNC_RECIPES = frozenset({"nvfp4"})
_TILE_K_RECIPES = frozenset({"nvfp4", "mxfp8"})
_SPLIT_K_RECIPES = frozenset({"mxfp8", "block_fp8"})
_DEFAULT_SM_COUNT = 48


@dataclass(frozen=True, kw_only=True)
class DenseLinearQuery:
    recipe: str
    in_features: int
    out_features: int
    max_tokens: int
    output_dtype: str


@dataclass(frozen=True, kw_only=True)
class DenseLinearConfig:
    backend: str
    tile_m: int
    tile_n: int
    tile_k: int
    load_path: str
    swap_ab: bool
    split_k: int

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "DenseLinearConfig":
        expected = {
            "backend",
            "tile_m",
            "tile_n",
            "tile_k",
            "load_path",
            "swap_ab",
            "split_k",
        }
        if set(payload) != expected:
            raise ValueError(
                "dense linear configs require exactly " + ", ".join(sorted(expected))
            )
        if not isinstance(payload["swap_ab"], bool):
            raise TypeError("dense linear swap_ab must be a boolean")
        return cls(
            backend=str(payload["backend"]),
            tile_m=int(payload["tile_m"]),
            tile_n=int(payload["tile_n"]),
            tile_k=int(payload["tile_k"]),
            load_path=str(payload["load_path"]),
            swap_ab=bool(payload["swap_ab"]),
            split_k=int(payload["split_k"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "load_path": self.load_path,
            "swap_ab": self.swap_ab,
            "split_k": self.split_k,
        }

    @property
    def mma_tiler_mn(self) -> tuple[int, int]:
        return (self.tile_m, self.tile_n)


def _encode(query: DenseLinearQuery) -> dict[str, object]:
    return {
        name: getattr(query, name) for name in DenseLinearQuery.__dataclass_fields__
    }


def recipe_planner_flags(recipe: str) -> dict[str, bool]:
    """Map a recipe onto the dense planner's format flags."""
    if recipe not in DENSE_LINEAR_RECIPES:
        raise ValueError(f"unsupported dense linear recipe {recipe!r}")
    return {
        "is_mxfp8": recipe in ("mxfp8", "tensor_fp8", "block_fp8"),
        "is_mxfp6": recipe == "mxfp6",
        "block_fp8": recipe == "block_fp8",
    }


def _fallback_tile(query: DenseLinearQuery) -> tuple[int, int]:
    """Host-only approximation of the dense planner for environments where the
    kernel module cannot be imported."""
    if query.max_tokens == 1:
        return (16, 64)
    if query.max_tokens <= 8:
        return (16, 128)
    if query.max_tokens <= 128:
        return (32, 128) if query.out_features > 1_536 else (64, 64)
    return (64, 128)


def _heuristic(
    query: DenseLinearQuery,
    device: DeviceIdentity | None,
) -> DenseLinearConfig:
    """Reproduce the built-in dense planner so an uncovered query behaves
    exactly as an unplanned launch does today."""
    sm_count = device.sm_count if device is not None else _DEFAULT_SM_COUNT
    flags = recipe_planner_flags(query.recipe)
    try:
        from b12x._lib.dense_gemm import _select_default_dense_gemm_plan
    except Exception:  # pragma: no cover - kernel module unavailable host-side
        tile = _fallback_tile(query)
        load_path, swap_ab = "tma", False
    else:
        plan = _select_default_dense_gemm_plan(
            query.max_tokens,
            query.out_features,
            query.in_features,
            sm_count,
            is_mxfp8=flags["is_mxfp8"],
            is_mxfp6=flags["is_mxfp6"],
            block_fp8=flags["block_fp8"],
            expected_m=query.max_tokens,
        )
        tile = tuple(plan.mma_tiler_mn)
        load_path = str(plan.load_path)
        swap_ab = bool(plan.swap_ab)
    return DenseLinearConfig(
        backend=DENSE_LINEAR_BACKEND,
        tile_m=int(tile[0]),
        tile_n=int(tile[1]),
        tile_k=0,
        load_path=load_path,
        swap_ab=swap_ab,
        split_k=0,
    )


def _validate(
    query: DenseLinearQuery,
    config: DenseLinearConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.recipe not in DENSE_LINEAR_RECIPES:
        raise ValueError(f"unsupported dense linear recipe {query.recipe!r}")
    if query.output_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported output dtype {query.output_dtype!r}")
    if query.max_tokens <= 0 or query.in_features <= 0 or query.out_features <= 0:
        raise ValueError("dense linear geometry must be positive")
    k_alignment = 128 if query.recipe == "block_fp8" else 32
    if query.in_features % k_alignment:
        raise ValueError(
            f"{query.recipe} in_features must be a multiple of {k_alignment}"
        )
    if config.backend != DENSE_LINEAR_BACKEND:
        raise ValueError(f"unsupported dense linear backend {config.backend!r}")
    check_tile(query.recipe, (config.tile_m, config.tile_n), config.swap_ab)
    if config.load_path not in DENSE_LINEAR_LOAD_PATHS:
        raise ValueError(f"unsupported dense linear load path {config.load_path!r}")
    if config.load_path == "cpasync" and query.recipe not in _CPASYNC_RECIPES:
        raise ValueError(f"{query.recipe} does not accept the cp.async load path")
    if config.tile_k not in DENSE_LINEAR_TILE_K:
        raise ValueError("unsupported dense linear K tile")
    if config.tile_k and query.recipe not in _TILE_K_RECIPES:
        raise ValueError(f"{query.recipe} does not accept an explicit K tile")
    if config.tile_k and query.in_features % config.tile_k:
        raise ValueError("dense linear K tile must divide in_features")
    if config.split_k not in DENSE_LINEAR_SPLIT_K:
        raise ValueError("unsupported dense linear split-K")
    if config.split_k and query.recipe not in _SPLIT_K_RECIPES:
        raise ValueError(f"{query.recipe} does not accept an explicit split-K")


def check_tile(recipe: str, tile: tuple[int, int], swap_ab: bool) -> None:
    """Reject (tile, swap) pairs the dense kernel cannot implement for a recipe."""
    if swap_ab and recipe not in _SWAP_RECIPES:
        raise ValueError(f"{recipe} does not accept swapped operands")
    if recipe in _FP4_RECIPES:
        if tile in FP4_TILES:
            return
        if tile in FP4_NARROW_TILES and swap_ab and recipe == "nvfp4":
            return
        raise ValueError(f"unsupported {recipe} MMA tile {tile}")
    if swap_ab:
        if tile in FP8_SWAPPED_TILES or tile in FP4_TILES:
            return
        raise ValueError(f"unsupported swapped {recipe} MMA tile {tile}")
    if tile not in FP8_TILES:
        raise ValueError(f"unsupported {recipe} MMA tile {tile}")


DENSE_LINEAR_POLICY = ComponentPolicy(
    component_id=DENSE_LINEAR,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(DenseLinearQuery.__dataclass_fields__),
    config_fields=frozenset(DenseLinearConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=DenseLinearConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "FP4_NARROW_TILES",
    "FP4_TILES",
    "FP8_SWAPPED_TILES",
    "FP8_TILES",
    "check_tile",
    "DENSE_LINEAR_BACKEND",
    "DENSE_LINEAR_LOAD_PATHS",
    "DENSE_LINEAR_POLICY",
    "DENSE_LINEAR_RECIPES",
    "DENSE_LINEAR_SPLIT_K",
    "DENSE_LINEAR_TILES",
    "DENSE_LINEAR_TILE_K",
    "DenseLinearConfig",
    "DenseLinearQuery",
    "recipe_planner_flags",
]
