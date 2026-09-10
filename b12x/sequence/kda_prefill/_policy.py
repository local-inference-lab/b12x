"""Policy contract for chunked KDA prefill: query, config, heuristic."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy
from b12x.policy.components import KDA_PREFILL
from .._shared.delta_prefill.policy import PrefillConfig

from .._shared.delta_prefill.workspace import (
    BACKEND, CHUNK_TOKENS, K_SPLIT_CHOICES, STAGE_CHOICES, V_SPLIT_CHOICES,
    WINDOW_BYTES_BUDGET, WORKSPACE_RECORD_BYTES, WorkspaceRecord,
    default_window_tiles, tiles_capacity, validate_shared_memory,
)


@dataclass(frozen=True, kw_only=True)
class KdaPrefillQuery:
    """Immutable geometry and planned capacity of one KDA prefill plan."""

    heads: int
    head_dim: int
    model_dtype: str
    state_dtype: str
    qk_l2norm: bool
    checkpoint_export: bool
    max_tokens: int
    max_seqs: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "model_dtype": str(self.model_dtype),
            "state_dtype": str(self.state_dtype),
            "qk_l2norm": bool(self.qk_l2norm),
            "checkpoint_export": bool(self.checkpoint_export),
            "max_tokens": int(self.max_tokens),
            "max_seqs": int(self.max_seqs),
        }


class KdaPrefillConfig(PrefillConfig):
    """Launch configuration for lower-bounded KDA prefill."""


def _heuristic(query: KdaPrefillQuery, device) -> KdaPrefillConfig:
    del device
    return KdaPrefillConfig(
        backend=BACKEND,
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    )


def _validate(query: KdaPrefillQuery, config: KdaPrefillConfig, device) -> None:
    del device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported {KDA_PREFILL} backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} v_split {config.v_split!r}; expected one "
            f"of {V_SPLIT_CHOICES}"
        )
    if config.k_split not in K_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} k_split {config.k_split!r}; expected one "
            f"of {K_SPLIT_CHOICES}"
        )
    if config.stages not in STAGE_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} stages {config.stages!r}; expected one of "
            f"{STAGE_CHOICES}"
        )
    if 2 * config.v_split * config.k_split + 32 > 1024:
        raise ValueError(f"{KDA_PREFILL} v_split x k_split exceeds the thread limit")
    if isinstance(config.window_tiles, bool) or int(config.window_tiles) < 1:
        raise ValueError(f"{KDA_PREFILL} window_tiles must be a positive integer")
    validate_shared_memory(config.v_split, config.k_split, config.stages, config.window_tiles,
                           reuse_value_buffer=False)
    if query.head_dim != 128:
        raise ValueError(f"{KDA_PREFILL} requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            f"{KDA_PREFILL} requires bfloat16 activations and float32 state, got "
            f"{query.model_dtype}/{query.state_dtype}"
        )


KDA_PREFILL_POLICY = ComponentPolicy(
    component_id=KDA_PREFILL,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "heads",
            "head_dim",
            "model_dtype",
            "state_dtype",
            "qk_l2norm",
            "checkpoint_export",
            "max_tokens",
            "max_seqs",
        }
    ),
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles"}),
    encode_query=KdaPrefillQuery.profile_fields,
    decode_profile=KdaPrefillConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)

__all__ = [
    "BACKEND",
    "CHUNK_TOKENS",
    "K_SPLIT_CHOICES",
    "KDA_PREFILL_POLICY",
    "KdaPrefillConfig",
    "KdaPrefillQuery",
    "STAGE_CHOICES",
    "V_SPLIT_CHOICES",
    "WINDOW_BYTES_BUDGET",
    "WORKSPACE_RECORD_BYTES",
    "WorkspaceRecord",
    "default_window_tiles",
    "tiles_capacity",
]
