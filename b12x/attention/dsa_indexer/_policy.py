"""Typed component policy for DSA indexer planning.

The config carries the launch knobs the paged indexer exposes: the route
(fused score+top-k, tiled supertile scoring, or the packed prefill scorer),
the tiled supertile width, the fused kernel's CTAs per row group and its
cross-CTA merge switch, and the tiled route's two-level fold. Every field has
an ``auto`` value that defers to the runtime's own resolution, so the all-auto
config is exactly today's heuristic behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import DSA_INDEXER, ComponentPolicy, DeviceIdentity, FrozenMapping

INDEXER_ROUTE_AUTO = "auto"
INDEXER_ROUTES = frozenset(
    {INDEXER_ROUTE_AUTO, "paged_fused", "paged_tiled", "packed_contiguous"}
)
INDEXER_FOLD_MODES = frozenset({"auto", "on", "off"})
# Tiled supertiles are measured in K rows and must align to the tile block.
INDEXER_SUPERTILE_ALIGN = 512
# The fused merge scratch is reserved for this many SM waves of CTAs, so a
# profile may oversubscribe the auto wave (num_sms // rows) up to this factor.
INDEXER_FUSED_CTAS_WAVES = 2


@dataclass(frozen=True, kw_only=True)
class DsaIndexerQuery:
    source_layout: str
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    num_idx_heads: int
    max_q_rows: int
    max_k_rows: int
    max_page_table_width: int
    top_k: int
    page_size: int
    score_mode: str
    shared_page_table: bool

    def profile_fields(self) -> dict[str, object]:
        return {
            "source_layout": self.source_layout,
            "mode": self.mode,
            "dtype": self.dtype,
            "kv_dtype": self.kv_dtype,
            "num_q_heads": self.num_q_heads,
            "num_idx_heads": self.num_idx_heads,
            "max_q_rows": self.max_q_rows,
            "max_k_rows": self.max_k_rows,
            "max_page_table_width": self.max_page_table_width,
            "top_k": self.top_k,
            "page_size": self.page_size,
            "score_mode": self.score_mode,
            "shared_page_table": self.shared_page_table,
        }


@dataclass(frozen=True, kw_only=True)
class DsaIndexerConfig:
    """Paged indexer launch knobs; ``auto``/0/-1 defer to the runtime."""

    route: str = INDEXER_ROUTE_AUTO
    supertile_k: int = 0
    fused_ctas_per_group: int = 0
    fused_merge_threshold: int = -1
    two_level_fold: str = "auto"

    _FIELDS = (
        "route",
        "supertile_k",
        "fused_ctas_per_group",
        "fused_merge_threshold",
        "two_level_fold",
    )

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "DsaIndexerConfig":
        if set(payload) != set(cls._FIELDS):
            raise ValueError(
                "DSA indexer profiles require exactly "
                f"{sorted(cls._FIELDS)}, got {sorted(payload)}"
            )
        for name in ("supertile_k", "fused_ctas_per_group", "fused_merge_threshold"):
            value = payload[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"DSA indexer {name} must be an integer")
        for name in ("route", "two_level_fold"):
            if not isinstance(payload[name], str):
                raise TypeError(f"DSA indexer {name} must be a string")
        return cls(**{name: payload[name] for name in cls._FIELDS})

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._FIELDS}


def _heuristic(
    _query: DsaIndexerQuery,
    _device: DeviceIdentity | None,
) -> DsaIndexerConfig:
    return DsaIndexerConfig()


def _validate(
    query: DsaIndexerQuery,
    config: DsaIndexerConfig,
    device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    if config.route not in INDEXER_ROUTES:
        raise ValueError(f"unsupported DSA indexer route {config.route!r}")
    if (
        config.route in ("paged_fused", "paged_tiled")
        and query.source_layout != "paged"
    ):
        raise ValueError(f"route {config.route!r} requires a paged source layout")
    if config.route == "paged_fused" and (
        query.mode != "decode" or query.shared_page_table
    ):
        raise ValueError("the fused paged indexer route is decode-only")
    if config.supertile_k < 0 or config.supertile_k % INDEXER_SUPERTILE_ALIGN:
        raise ValueError(
            f"DSA indexer supertile_k must be 0 or a multiple of "
            f"{INDEXER_SUPERTILE_ALIGN}, got {config.supertile_k}"
        )
    if config.fused_ctas_per_group < 0:
        raise ValueError("DSA indexer fused_ctas_per_group must be non-negative")
    if config.fused_ctas_per_group and device is not None:
        budget = INDEXER_FUSED_CTAS_WAVES * int(device.sm_count)
        if max(1, int(query.max_q_rows)) * config.fused_ctas_per_group > budget:
            raise ValueError(
                "DSA indexer fused_ctas_per_group oversubscribes the reserved "
                f"merge scratch: rows={query.max_q_rows} x "
                f"{config.fused_ctas_per_group} > {budget}"
            )
    if config.fused_merge_threshold < -1:
        raise ValueError("DSA indexer fused_merge_threshold must be >= -1")
    if config.two_level_fold not in INDEXER_FOLD_MODES:
        raise ValueError(
            f"DSA indexer two_level_fold must be one of "
            f"{sorted(INDEXER_FOLD_MODES)}, got {config.two_level_fold!r}"
        )


DSA_INDEXER_POLICY = ComponentPolicy(
    component_id=DSA_INDEXER,
    query_schema_version=2,
    config_schema_version=2,
    query_fields=frozenset(
        {
            "source_layout",
            "mode",
            "dtype",
            "kv_dtype",
            "num_q_heads",
            "num_idx_heads",
            "max_q_rows",
            "max_k_rows",
            "max_page_table_width",
            "top_k",
            "page_size",
            "score_mode",
            "shared_page_table",
        }
    ),
    config_fields=frozenset(DsaIndexerConfig._FIELDS),
    encode_query=DsaIndexerQuery.profile_fields,
    decode_profile=DsaIndexerConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
    fallback_warning_fields=frozenset(
        {"source_layout", "mode", "num_q_heads", "top_k", "score_mode"}
    ),
)

__all__ = [
    "DSA_INDEXER_POLICY",
    "INDEXER_FOLD_MODES",
    "INDEXER_FUSED_CTAS_WAVES",
    "INDEXER_ROUTES",
    "INDEXER_ROUTE_AUTO",
    "INDEXER_SUPERTILE_ALIGN",
    "DsaIndexerConfig",
    "DsaIndexerQuery",
]
