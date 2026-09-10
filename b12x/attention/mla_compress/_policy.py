"""Plan-time policy for projected CSA1/CSA2 compression."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from b12x.policy import ComponentPolicy, FrozenMapping


@dataclass(frozen=True, kw_only=True)
class MlaCompressQuery:
    ratio: int
    max_tokens: int
    max_requests: int
    max_states: int
    head_dim: int = 512


@dataclass(frozen=True, kw_only=True)
class MlaCompressConfig:
    backend: str = "cute"
    threads: int = 128

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> MlaCompressConfig:
        if frozenset(payload) != frozenset(cls.__dataclass_fields__):
            raise ValueError("invalid MLA compression config fields")
        if not isinstance(payload["threads"], int) or isinstance(payload["threads"], bool):
            raise TypeError("threads must be an integer")
        return cls(backend=str(payload["backend"]), threads=payload["threads"])


def _validate(query, config, _device):
    if query.ratio not in (1, 2) or query.head_dim != 512:
        raise ValueError("CSA compression supports ratio 1/2 and head_dim 512")
    if min(query.max_tokens, query.max_requests, query.max_states) <= 0:
        raise ValueError("compression capacities must be positive")
    if max(query.max_tokens, query.max_requests) >= 2**31:
        raise ValueError("token/request capacities must fit int32")
    if query.max_states > (2**63 - 1) // 512:
        raise ValueError("state extent must fit int64")
    if config.backend != "cute" or config.threads != 128:
        raise ValueError("compression uses the CuTe 128-thread kernel")


MLA_COMPRESS_POLICY = ComponentPolicy(
    component_id="attention.mla_compress",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(MlaCompressQuery.__dataclass_fields__),
    config_fields=frozenset(MlaCompressConfig.__dataclass_fields__),
    encode_query=asdict,
    decode_profile=MlaCompressConfig.from_profile,
    heuristic=lambda _query, _device: MlaCompressConfig(),
    validate_config=_validate,
)
