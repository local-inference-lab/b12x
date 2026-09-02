"""Typed component policy for W_o projection planning.

The fused WO-A / WO-B decode chain has three launch regimes (fused-quant
WO-B at small M, the quantized-intermediate exact path around M=16, and the
standalone-quant path beyond) plus one MMA tile per GEMM. Config schema 2
exposes those choices so the profile generator can race them per geometry
and capacity instead of trusting hand-tuned constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    WO_PROJECTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)

WO_PROJECTION_BACKEND = "mxfp8"
WO_PROJECTION_TILES = (
    (0, 0),  # leave the built-in dense planner in charge
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)
WO_B_FUSED_TILED_PLANS = ((16, 64), (16, 128))
_DEFAULT_SM_COUNT = 48
_DEFAULT_LOW_SM_MAX = 48


@dataclass(frozen=True, kw_only=True)
class WoProjectionQuery:
    dtype: str
    max_tokens: int
    groups: int
    group_width: int
    rank: int
    hidden: int


@dataclass(frozen=True, kw_only=True)
class WoProjectionConfig:
    backend: str
    wo_a_tile_m: int
    wo_a_tile_n: int
    wo_b_tile_m: int
    wo_b_tile_n: int
    wo_b_fused_quant: bool
    quantized_intermediate: bool

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "WoProjectionConfig":
        expected = {
            "backend",
            "wo_a_tile_m",
            "wo_a_tile_n",
            "wo_b_tile_m",
            "wo_b_tile_n",
            "wo_b_fused_quant",
            "quantized_intermediate",
        }
        if set(payload) != expected:
            raise ValueError(
                "WO projection configs require exactly " + ", ".join(sorted(expected))
            )
        for flag in ("wo_b_fused_quant", "quantized_intermediate"):
            if not isinstance(payload[flag], bool):
                raise TypeError(f"WO projection {flag} must be a boolean")
        return cls(
            backend=str(payload["backend"]),
            wo_a_tile_m=int(payload["wo_a_tile_m"]),
            wo_a_tile_n=int(payload["wo_a_tile_n"]),
            wo_b_tile_m=int(payload["wo_b_tile_m"]),
            wo_b_tile_n=int(payload["wo_b_tile_n"]),
            wo_b_fused_quant=bool(payload["wo_b_fused_quant"]),
            quantized_intermediate=bool(payload["quantized_intermediate"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "wo_a_tile_m": self.wo_a_tile_m,
            "wo_a_tile_n": self.wo_a_tile_n,
            "wo_b_tile_m": self.wo_b_tile_m,
            "wo_b_tile_n": self.wo_b_tile_n,
            "wo_b_fused_quant": self.wo_b_fused_quant,
            "quantized_intermediate": self.quantized_intermediate,
        }

    @property
    def wo_a_tile(self) -> tuple[int, int] | None:
        if self.wo_a_tile_m and self.wo_a_tile_n:
            return (self.wo_a_tile_m, self.wo_a_tile_n)
        return None

    @property
    def wo_b_tile(self) -> tuple[int, int] | None:
        if self.wo_b_tile_m and self.wo_b_tile_n:
            return (self.wo_b_tile_m, self.wo_b_tile_n)
        return None

    def launch_codes(self) -> tuple[int, int, int, int, int, int]:
        """Integer encoding consumed by the fused WO custom ops."""
        return (
            int(self.wo_a_tile_m),
            int(self.wo_a_tile_n),
            int(self.wo_b_tile_m),
            int(self.wo_b_tile_n),
            2 if self.wo_b_fused_quant else 1,
            2 if self.quantized_intermediate else 1,
        )


def _encode(query: WoProjectionQuery) -> dict[str, object]:
    return {name: getattr(query, name) for name in WoProjectionQuery.__dataclass_fields__}


def _low_sm_max() -> int:
    try:
        from b12x._lib.dense_gemm import _WO_SPARK_MAX_SMS
    except Exception:  # pragma: no cover - kernel module unavailable host-side
        return _DEFAULT_LOW_SM_MAX
    return int(_WO_SPARK_MAX_SMS)


def _fused_wo_b_tile(
    query: WoProjectionQuery, sm_count: int
) -> tuple[int, int]:
    """The tile the fused-quant WO-B launch pins today (see
    ``wo_mxfp8._wo_b_fused_tiled_plan``)."""
    try:
        from b12x._lib.dense_gemm import _select_default_dense_gemm_plan
    except Exception:  # pragma: no cover - kernel module unavailable host-side
        return (16, 128)
    plan = _select_default_dense_gemm_plan(
        query.max_tokens,
        query.hidden,
        query.rank * query.groups,
        sm_count,
        is_mxfp8=True,
        expected_m=query.max_tokens,
    )
    tile = tuple(plan.mma_tiler_mn)
    return tile if tile in WO_B_FUSED_TILED_PLANS else (16, 128)


def _heuristic(
    query: WoProjectionQuery,
    device: DeviceIdentity | None,
) -> WoProjectionConfig:
    """Reproduce the launch chain the runtime selects without a profile."""
    sm_count = device.sm_count if device is not None else _DEFAULT_SM_COUNT
    spark = sm_count <= _low_sm_max()
    m = query.max_tokens
    dsv4_tp2 = (
        query.groups == 4
        and query.group_width == 512
        and query.rank == 1024
        and query.hidden == 4096
    )
    quantized_intermediate = spark and (m == 16 or (9 <= m <= 15 and dsv4_tp2))
    small_rank = query.rank <= 1536
    if quantized_intermediate:
        wo_a = (32, 64) if small_rank else (0, 0)
        wo_b = (32, 64) if (m == 16 or query.rank * query.groups == 4096) else (0, 0)
        fused = False
    elif 1 <= m <= 8:
        wo_a = (16, 64) if small_rank else (0, 0)
        wo_b = _fused_wo_b_tile(query, sm_count)
        fused = True
    else:
        wo_a = (0, 0)
        wo_b = (0, 0)
        fused = False
    return WoProjectionConfig(
        backend=WO_PROJECTION_BACKEND,
        wo_a_tile_m=wo_a[0],
        wo_a_tile_n=wo_a[1],
        wo_b_tile_m=wo_b[0],
        wo_b_tile_n=wo_b[1],
        wo_b_fused_quant=fused,
        quantized_intermediate=quantized_intermediate,
    )


def _validate(
    query: WoProjectionQuery,
    config: WoProjectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != WO_PROJECTION_BACKEND:
        raise ValueError(f"unsupported WO projection backend {config.backend!r}")
    if query.dtype != "bfloat16":
        raise ValueError(f"unsupported WO projection dtype {query.dtype!r}")
    if min(query.max_tokens, query.groups, query.group_width, query.rank, query.hidden) <= 0:
        raise ValueError("WO projection geometry must be positive")
    if query.group_width % 32 or (query.rank * query.groups) % 32:
        raise ValueError("WO projection K extents must be multiples of 32")
    for name, tile in (("WO-A", (config.wo_a_tile_m, config.wo_a_tile_n)), ("WO-B", (config.wo_b_tile_m, config.wo_b_tile_n))):
        if tile not in WO_PROJECTION_TILES:
            raise ValueError(f"unsupported {name} MMA tile {tile}")
    if config.wo_b_fused_quant and config.quantized_intermediate:
        raise ValueError("fused-quant WO-B and the quantized intermediate are exclusive")
    if config.wo_b_fused_quant and query.max_tokens > 8:
        raise ValueError("fused-quant WO-B requires max_tokens <= 8")
    if config.quantized_intermediate and query.max_tokens > 16:
        raise ValueError("the quantized WO intermediate requires max_tokens <= 16")


WO_PROJECTION_POLICY = ComponentPolicy(
    component_id=WO_PROJECTION,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(WoProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(WoProjectionConfig.__dataclass_fields__),
    encode_query=_encode,
    decode_profile=WoProjectionConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)


__all__ = [
    "WO_B_FUSED_TILED_PLANS",
    "WO_PROJECTION_BACKEND",
    "WO_PROJECTION_POLICY",
    "WO_PROJECTION_TILES",
    "WoProjectionConfig",
    "WoProjectionQuery",
]
