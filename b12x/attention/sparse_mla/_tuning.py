"""Fixed configuration contract for sparse MLA planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import DeviceIdentity, FrozenMapping, Knob, TuningContract


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    max_q_rows: int
    max_width: int
    page_size: int
    model_type: int | None
    head_major_output: bool
    scale_format: int
    cache_record_bytes: int
    fp8_rope: bool
    latent_scale_per_token: bool
    has_attention_sink: bool
    cache_layout: str
    operation: str
    slot_dtype: str | None
    prefill_mg_enabled: bool
    max_batch: int = 0
    max_kv_rows: int = 0
    max_page_table_width: int = 0
    max_chunks_per_row: int = 0
    max_q_chunks: int = 0
    physical_block_size: int = 0
    physical_record_width: int = 0
    num_cache_blocks: int = 0
    max_physical_records: int = 0
    tp_size: int = 0
    use_cuda_graph: bool = False
    budget_max_splits: int | None = None
    budget_max_partial_rows: int | None = None
    pool_size: int = 0
    pool_topk: int = 0


@dataclass(frozen=True, kw_only=True)
class SparseMlaConfig:
    backend: str
    num_splits: int = 1

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "SparseMlaConfig":
        if set(payload) != {"backend", "num_splits"}:
            raise ValueError(
                "sparse MLA config requires backend and num_splits"
            )
        return cls(**dict(payload))

    def to_dict(self):
        return {"backend": self.backend, "num_splits": self.num_splits}


def _default_config(query: SparseMlaQuery, device: DeviceIdentity | None):
    if query.operation not in ("cache_writer", "pooled_selection") and device is not None and device.compute_capability == (10, 3):
        # Capacity alone determines the split schedule; live rows and selected
        # lengths never select another compiled callable during replay.
        splits = min(4, query.max_chunks_per_row, (query.max_width + 63) // 64)
        return SparseMlaConfig(
            backend="warp", num_splits=splits if query.mode == "decode" else 1
        )
    # A fixed conservative schedule also keeps native decode independent of
    # live rows. Measured overrides may select another planned split count.
    return SparseMlaConfig(backend="native", num_splits=1)


def _validate(
    query: SparseMlaQuery, config: SparseMlaConfig, device: DeviceIdentity | None
):
    if not isinstance(config, SparseMlaConfig):
        raise TypeError("config must be SparseMlaConfig")
    if query.operation in ("cache_writer", "pooled_selection"):
        if config != SparseMlaConfig(backend="native"):
            raise ValueError("cache writers and pooled selection require the native backend without splits")
        return
    if config.backend not in ("native", "warp"):
        raise ValueError(f"unsupported sparse MLA backend {config.backend!r}")
    if (
        device is not None
        and device.compute_capability == (10, 3)
        and config.backend != "warp"
    ):
        raise ValueError("SM103 sparse MLA requires the warp backend")
    if type(config.num_splits) is not int or config.num_splits <= 0:
        raise ValueError("num_splits must be a positive integer")
    limit = min(query.max_chunks_per_row, max(1, (query.max_width + 63) // 64))
    if config.num_splits > limit or (query.mode != "decode" and config.num_splits != 1):
        raise ValueError("num_splits exceeds the planned sparse MLA capacity")
    if config.backend == "native":
        return
    from .._shared.mla.traits import ModelType

    if query.model_type not in (None, ModelType.GLM_NSA, ModelType.GLM_NEXT):
        raise ValueError("warp sparse MLA implements GLM NSA and GLM Next")
    expected_dim = 512 if query.model_type == ModelType.GLM_NEXT else 576
    if query.qk_head_dim != expected_dim or query.v_head_dim != 512:
        raise ValueError("warp sparse MLA requires the selected GLM head dimensions")
    if query.dtype != "bfloat16" or query.kv_dtype != "uint8":
        raise ValueError("warp sparse MLA requires BF16 queries and packed byte KV")
    if query.num_q_heads <= 0 or query.num_q_heads % 8:
        raise ValueError("warp sparse MLA requires query heads divisible by eight")
    if query.mode not in ("decode", "extend", "verify", "draft_extend"):
        raise ValueError("unsupported sparse MLA mode")


def _parameters(query, device):
    config = _default_config(query, device)
    return {"backend": (config.backend,), "num_splits": (config.num_splits,)}


_KEY_FIELDS = frozenset(SparseMlaQuery.__dataclass_fields__) - {
    "max_page_table_width", "num_cache_blocks", "max_physical_records",
}


def _validate_query(query, device):
    if not isinstance(query, SparseMlaQuery):
        raise TypeError("query must be SparseMlaQuery")
    if query.operation not in ("attention", "strided_attention", "cache_writer", "pooled_selection"):
        raise ValueError("unknown sparse MLA operation")


TUNING = TuningContract(
    component_id="attention.sparse_mla", query_schema_version=4,
    config_schema_version=2, candidate_contract_version=5,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset(SparseMlaConfig.__dataclass_fields__),
    encode_query=lambda query: {name: getattr(query, name) for name in _KEY_FIELDS},
    encode_config=SparseMlaConfig.to_dict, decode_config=SparseMlaConfig.from_config,
    validate_query=_validate_query, validate_config=_validate,
    default_config=_default_config, parameters=_parameters,
    knobs=(Knob(name="backend", values=None), Knob(name="num_splits", values=None)),
)
