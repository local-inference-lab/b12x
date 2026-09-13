"""Typed component policy for W_o projection planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import WO_PROJECTION, ComponentPolicy, DeviceIdentity, FrozenMapping


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
    backend: str = "mxfp8"
    decode_tile_n: int = 0

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "WoProjectionConfig":
        if set(payload) != {"backend", "decode_tile_n"}:
            raise ValueError("WO configs require backend and decode_tile_n")
        if not isinstance(payload["backend"], str):
            raise TypeError("WO backend must be a string")
        if type(payload["decode_tile_n"]) is not int:
            raise TypeError("WO decode_tile_n must be an integer")
        return cls(backend=payload["backend"], decode_tile_n=payload["decode_tile_n"])

    def to_dict(self) -> dict[str, object]:
        return {"backend": self.backend, "decode_tile_n": self.decode_tile_n}


def _encode(query: WoProjectionQuery) -> dict[str, object]:
    if not isinstance(query, WoProjectionQuery):
        raise TypeError("query must be WoProjectionQuery")
    return {name: getattr(query, name) for name in query.__dataclass_fields__}


def _heuristic(
    _query: WoProjectionQuery,
    _device: DeviceIdentity | None,
) -> WoProjectionConfig:
    # Unmeasured devices and geometries retain dense GEMM's existing selection.
    return WoProjectionConfig()


def _validate(query, config, _device) -> None:
    if not isinstance(config, WoProjectionConfig):
        raise TypeError("config must be WoProjectionConfig")
    if config.backend != "mxfp8":
        raise ValueError(f"unsupported WO backend {config.backend!r}")
    if type(config.decode_tile_n) is not int or config.decode_tile_n not in (
        0,
        64,
        128,
    ):
        raise ValueError("WO decode_tile_n must be 0, 64 or 128")
    if config.decode_tile_n and not (
        query.dtype == "bfloat16"
        and 1 <= query.max_tokens <= 8
        and (query.groups, query.group_width, query.rank, query.hidden)
        == (2, 4096, 1024, 5120)
    ):
        raise ValueError(
            "WO decode tile override requires the BF16 2x4096/1024/5120 decode domain"
        )


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


__all__ = ["WO_PROJECTION_POLICY", "WoProjectionConfig", "WoProjectionQuery"]
