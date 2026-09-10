"""Policy contract for chunked GDN prefill: query, config, heuristic."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy
from b12x.policy.components import GDN_PREFILL
from b12x.policy.types import FrozenMapping
from .._shared.delta_prefill.policy import PrefillConfig

from .._shared.delta_prefill.workspace import (
    BACKEND, CHUNK_TOKENS, K_SPLIT_CHOICES, STAGE_CHOICES, V_SPLIT_CHOICES,
    WINDOW_BYTES_BUDGET, WORKSPACE_RECORD_BYTES, WorkspaceRecord,
    default_window_tiles, tiles_capacity, validate_shared_memory,
)


@dataclass(frozen=True, kw_only=True)
class GdnPrefillQuery:
    """Immutable geometry and planned capacity of one GDN prefill plan."""

    key_heads: int
    value_heads: int
    head_dim: int
    model_dtype: str
    state_dtype: str
    qk_l2norm: bool
    checkpoint_export: bool
    max_tokens: int
    max_seqs: int

    @property
    def heads(self) -> int:
        return self.value_heads

    def profile_fields(self) -> dict[str, object]:
        return {
            "key_heads": int(self.key_heads),
            "value_heads": int(self.value_heads),
            "head_dim": int(self.head_dim),
            "model_dtype": str(self.model_dtype),
            "state_dtype": str(self.state_dtype),
            "qk_l2norm": bool(self.qk_l2norm),
            "checkpoint_export": bool(self.checkpoint_export),
            "max_tokens": int(self.max_tokens),
            "max_seqs": int(self.max_seqs),
        }


@dataclass(frozen=True)
class GdnPrefillConfig(PrefillConfig):
    """Launch configuration for scalar-gated GDN prefill."""

    algorithm: str = "sequential"
    segment_tokens: int = 256

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "GdnPrefillConfig":
        base = PrefillConfig.from_profile(FrozenMapping({
            key: value for key, value in payload.items()
            if key not in ("algorithm", "segment_tokens")
        }))
        algorithm = payload.get("algorithm", "sequential")
        segment_tokens = payload.get("segment_tokens", 256)
        if not isinstance(algorithm, str):
            raise TypeError("algorithm must be a string")
        if isinstance(segment_tokens, bool) or not isinstance(segment_tokens, int):
            raise TypeError("segment_tokens must be an integer")
        return cls(**base.to_dict(), algorithm=algorithm, segment_tokens=segment_tokens)

    def to_dict(self) -> dict[str, object]:
        return {**super().to_dict(), "algorithm": self.algorithm,
                "segment_tokens": self.segment_tokens}


def _heuristic(query: GdnPrefillQuery, device) -> GdnPrefillConfig:
    del device
    return GdnPrefillConfig(
        backend=BACKEND,
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    )


def _validate(query: GdnPrefillQuery, config: GdnPrefillConfig, device) -> None:
    del device
    if query.key_heads <= 0 or query.value_heads != 3 * query.key_heads:
        raise ValueError("GDN prefill requires three value heads per key head")
    if config.backend != BACKEND:
        raise ValueError(f"unsupported {GDN_PREFILL} backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {GDN_PREFILL} v_split {config.v_split!r}; expected one "
            f"of {V_SPLIT_CHOICES}"
        )
    if config.k_split not in K_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {GDN_PREFILL} k_split {config.k_split!r}; expected one "
            f"of {K_SPLIT_CHOICES}"
        )
    if config.stages not in STAGE_CHOICES:
        raise ValueError(
            f"unsupported {GDN_PREFILL} stages {config.stages!r}; expected one of "
            f"{STAGE_CHOICES}"
        )
    if 2 * config.v_split * config.k_split + 32 > 1024:
        raise ValueError(f"{GDN_PREFILL} v_split x k_split exceeds the thread limit")
    if isinstance(config.window_tiles, bool) or int(config.window_tiles) < 1:
        raise ValueError(f"{GDN_PREFILL} window_tiles must be a positive integer")
    if config.algorithm not in ("sequential", "chunk_parallel"):
        raise ValueError(f"unsupported GDN prefill algorithm {config.algorithm!r}")
    if config.segment_tokens not in (128, 256, 512, 1024):
        raise ValueError("segment_tokens must be 128, 256, 512, or 1024")
    if config.algorithm == "chunk_parallel":
        segments = (query.max_tokens + config.segment_tokens - 1) // config.segment_tokens + query.max_seqs
        if segments > 4096:
            raise ValueError("chunk-parallel GDN supports at most 4096 planned segments")
        window = tiles_capacity(query.max_tokens, segments)
        if config.window_tiles != window:
            raise ValueError(f"chunk-parallel window_tiles must equal planned tile capacity {window}")
    validate_shared_memory(
        config.v_split, config.k_split, config.stages, config.window_tiles,
        max_sequence_tiles=config.segment_tokens // 16 if config.algorithm == "chunk_parallel" else 0,
        summary_mode=5 if config.algorithm == "chunk_parallel" and config.k_split == 1 else 0,
    )
    if query.head_dim != 128:
        raise ValueError(f"{GDN_PREFILL} requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            f"{GDN_PREFILL} requires bfloat16 activations and float32 state, got "
            f"{query.model_dtype}/{query.state_dtype}"
        )


GDN_PREFILL_POLICY = ComponentPolicy(
    component_id=GDN_PREFILL,
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(
        {
            "key_heads",
            "value_heads",
            "head_dim",
            "model_dtype",
            "state_dtype",
            "qk_l2norm",
            "checkpoint_export",
            "max_tokens",
            "max_seqs",
        }
    ),
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles", "algorithm", "segment_tokens"}),
    encode_query=GdnPrefillQuery.profile_fields,
    decode_profile=GdnPrefillConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)

__all__ = [
    "BACKEND",
    "CHUNK_TOKENS",
    "K_SPLIT_CHOICES",
    "GDN_PREFILL_POLICY",
    "GdnPrefillConfig",
    "GdnPrefillQuery",
    "STAGE_CHOICES",
    "V_SPLIT_CHOICES",
    "WINDOW_BYTES_BUDGET",
    "WORKSPACE_RECORD_BYTES",
    "WorkspaceRecord",
    "default_window_tiles",
    "tiles_capacity",
]
