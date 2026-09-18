"""Preparation selection for native hierarchical expert execution."""
from dataclasses import asdict, dataclass
import math

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract


@dataclass(frozen=True, kw_only=True)
class ResidencyQuery:
    hidden: int
    intermediate: int
    experts: int
    hot_experts: int
    max_tokens: int
    max_top_k: int
    profile_hash: str
    model_fingerprint: str
    gate_first: bool
    swiglu_limit: float | None = None
    numerical_mode: str = "mxfp8_fp32_activation_bf16_expert_ordered_fma"


@dataclass(frozen=True, kw_only=True)
class ResidencyConfig:
    backend: str = "sm103_mxfp8_mxfp4"


def _validate_query(query, device):
    if not isinstance(query, ResidencyQuery):
        raise TypeError("expert residency requires ResidencyQuery")
    for name in ("hidden", "intermediate", "experts", "max_tokens", "max_top_k"):
        if type(getattr(query, name)) is not int or getattr(query, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if query.hidden % 128 or query.intermediate % 128:
        raise ValueError("native MXFP4 residency requires H and I divisible by 128")
    if type(query.hot_experts) is not int or not 0 <= query.hot_experts <= query.experts:
        raise ValueError("HBM expert count is outside the layer geometry")
    if query.max_tokens * query.max_top_k >= 2**31 or query.experts >= 2**31:
        raise ValueError("route capacity and expert count must fit signed int32")
    if query.hidden // 128 > 65535 or 2 * query.intermediate // 128 > 65535:
        raise ValueError("projection width exceeds the CUDA grid limit")
    if query.swiglu_limit is not None and (not math.isfinite(query.swiglu_limit) or query.swiglu_limit <= 0):
        raise ValueError("SwiGLU limit must be finite and positive")
    if type(query.gate_first) is not bool or query.numerical_mode != ResidencyQuery.__dataclass_fields__["numerical_mode"].default:
        raise ValueError("unsupported expert residency numerical contract")
    if len(query.profile_hash) != 64 or not query.model_fingerprint:
        raise ValueError("expert residency requires profile and model identity")
    if device is not None and device.compute_capability != (10, 3):
        raise ValueError("native hierarchical MXFP4 MoE requires SM103")


def _validate(query, config, device):
    _validate_query(query, device)
    if config != ResidencyConfig():
        raise ValueError("unsupported expert residency configuration")


TUNING = TuningContract(
    component_id="moe.expert_residency", query_schema_version=1, config_schema_version=1,
    query_fields=frozenset(ResidencyQuery.__dataclass_fields__),
    config_fields=frozenset(ResidencyConfig.__dataclass_fields__),
    encode_query=asdict, encode_config=asdict,
    decode_config=lambda payload: ResidencyConfig(**dict(payload)),
    validate_query=_validate_query, validate_config=_validate,
    default_config=lambda query, device: ResidencyConfig(),
    knobs=(Knob(name="backend", values=("sm103_mxfp8_mxfp4",), binding=ParameterBinding.COMPILE),),
    candidate_contract_version=1,
    parameters=lambda query, device: ParameterSpace.create(TUNING.knobs),
    materialize=lambda query, device, choice: ResidencyConfig(**dict(choice)),
)
