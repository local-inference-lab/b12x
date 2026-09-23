"""Prepared canonical-host cache contract for native SM120 W4A16."""

from dataclasses import asdict, dataclass

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract


@dataclass(frozen=True, kw_only=True)
class ExpertCacheQuery:
    experts: int
    resident: int
    hidden: int
    intermediate: int
    max_tokens: int
    top_k: int
    w13_layout: str
    checkpoint_fingerprint: str
    profile_hash: str
    max_pairs: int = 0
    numerical_recipe: str = "nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum"


@dataclass(frozen=True, kw_only=True)
class ExpertCacheConfig:
    backend: str = "sm120_w4a16_canonical_mapped"
    cold_prefill: str = "fused"


def validate(query, device):
    if not isinstance(query, ExpertCacheQuery):
        raise TypeError("canonical expert cache requires ExpertCacheQuery")
    for name in (
        "experts",
        "resident",
        "hidden",
        "intermediate",
        "max_tokens",
        "top_k",
    ):
        if type(getattr(query, name)) is not int or getattr(query, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if query.resident > query.experts or query.top_k > query.experts:
        raise ValueError("resident count or top-k exceeds expert count")
    if query.hidden % 128 or query.intermediate % 128:
        raise ValueError("canonical cache requires H and I divisible by 128")
    if query.max_tokens * query.top_k >= 2**31 or query.experts >= 2**31:
        raise ValueError("expert and route capacities must fit int32")
    if type(query.max_pairs) is not int or not 0 <= query.max_pairs <= min(
        query.resident, query.experts - query.resident
    ):
        raise ValueError("fill capacity exceeds the resident/backing split")
    if query.w13_layout not in ("w13", "w31"):
        raise ValueError("unknown gate/up ordering")
    for value in (query.checkpoint_fingerprint, query.profile_hash):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("cache source and placement require SHA256 identities")
    if (
        query.numerical_recipe
        != ExpertCacheQuery.__dataclass_fields__["numerical_recipe"].default
    ):
        raise ValueError("unsupported expert cache numerical recipe")
    if device is not None and device.compute_capability != (12, 0):
        raise ValueError(
            "canonical mapped W4A16 expert cache currently requires physical SM120"
        )


def validate_config(query, config, device):
    validate(query, device)
    if not isinstance(config, ExpertCacheConfig) or config.backend != ExpertCacheConfig().backend:
        raise ValueError("unsupported canonical cache backend")
    if config.cold_prefill not in ("fused", "two_cta", "two_cta_pipeline3"):
        raise ValueError("unsupported cold-prefill variant")
    if config.cold_prefill != "fused" and query.max_tokens < 16:
        raise ValueError("cold-prefill variants require capacity of at least 16 tokens")


TUNING = TuningContract(
    component_id="moe.expert_cache",
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(ExpertCacheQuery.__dataclass_fields__),
    config_fields=frozenset(ExpertCacheConfig.__dataclass_fields__),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: ExpertCacheConfig(**dict(payload)),
    validate_query=validate,
    validate_config=validate_config,
    default_config=lambda query, device: ExpertCacheConfig(),
    knobs=(
        Knob(
            name="backend",
            values=(ExpertCacheConfig().backend,),
            binding=ParameterBinding.COMPILE,
        ),
        # Experimental variants require an explicit pin; automatic races retain
        # the qualified fused implementation.
        Knob(name="cold_prefill", values=("fused",), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=3,
    parameters=lambda query, device: ParameterSpace.create(TUNING.knobs),
    materialize=lambda query, device, choice: ExpertCacheConfig(**dict(choice)),
)
