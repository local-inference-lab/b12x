"""Frozen grouped FP8 GEMM query/config: static geometry + planned capacity.

The query never carries live request counts: ``masked_m`` values, labels,
and live ``m_total`` are runtime scalars (device tensors at call time), per
the b12x no-live-counts rule. ``m_capacity`` is the planned allocation bound
(masked ``m_cap`` / contiguous max live ``m_total``); ``expected_m`` is a
scheduling regime hint, matching the dense/blockscaled convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from b12x.preparation import FrozenMapping, TuningContract
from b12x.preparation.tuning import Knob, ParameterBinding

MODES = ("masked", "contiguous")


@dataclass(frozen=True, kw_only=True)
class MGroupFP8GemmQuery:
    mode: str  # "masked" | "contiguous"
    num_groups: int
    n: int
    k: int
    m_capacity: int  # masked: per-group m_cap; contiguous: max live m_total
    a_sf_gran: int  # 128 for masked, 32 for contiguous
    b_sf_gran: int = 128
    c_dtype: str = "bfloat16"
    expected_m: int | None = None  # scheduling regime hint, never a live count


@dataclass(frozen=True, kw_only=True)
class MGroupFP8GemmConfig:
    backend: str
    tile_m: int
    tile_n: int
    tile_k: int
    implementation: str = "single"

    @classmethod
    def from_config(cls, payload):
        if set(payload) != {"backend", "tile_m", "tile_n", "tile_k", "implementation"}:
            raise ValueError("grouped FP8 config requires implementation, backend and tiles")
        return cls(**dict(payload))

    def to_dict(self):
        return asdict(self)


def validate_query(query):
    if not isinstance(query, MGroupFP8GemmQuery):
        raise TypeError("query must be MGroupFP8GemmQuery")
    if query.mode not in MODES:
        raise ValueError(f"grouped FP8 mode must be one of {MODES}")
    if any(type(value) is not int or value <= 0 for value in (
        query.num_groups, query.n, query.k, query.m_capacity,
    )):
        raise ValueError("grouped FP8 geometry must be positive integers")
    expected_a_gran = 128 if query.mode == "masked" else 32
    if query.a_sf_gran != expected_a_gran:
        raise ValueError(
            f"{query.mode} mode requires gran-{expected_a_gran} A scales, "
            f"got {query.a_sf_gran}"
        )
    if query.b_sf_gran != 128:
        raise ValueError("grouped FP8 requires gran-128 B scales")
    if query.k % 128:
        raise ValueError("grouped FP8 requires K divisible by 128")
    if query.k % query.a_sf_gran:
        raise ValueError("grouped FP8 K must divide its A scale granularity")
    if query.c_dtype != "bfloat16":
        raise ValueError(f"grouped FP8 output must be BF16, got {query.c_dtype!r}")
    if query.expected_m is not None and (
        type(query.expected_m) is not int
        or query.expected_m <= 0
        or query.expected_m > query.m_capacity
    ):
        raise ValueError("expected_m must be a positive integer within m_capacity or None")


def validate_config(query, config, device):
    if not isinstance(config, MGroupFP8GemmConfig):
        raise TypeError("config must be MGroupFP8GemmConfig")
    if config.backend != "cutedsl":
        raise ValueError("grouped FP8 kernels are CuTe DSL only")
    if config.tile_m not in (16, 32, 64, 128) or config.tile_n not in (64, 128):
        raise ValueError("grouped FP8 tile M must be 16/32/64/128 and tile N must be 64/128")
    if query.mode == "contiguous" and config.tile_m not in (64, 128):
        raise ValueError(
            "contiguous label runs are 128-aligned; tile M must divide it"
        )
    if config.tile_k not in (64, 128):
        raise ValueError("grouped FP8 tile K must be 64 or 128")
    if config.tile_k == 64 and config.tile_m != 128:
        raise ValueError("MXFP8 BK64 staging requires a 128-row tile")
    if config.implementation not in ("single", "joint_v1", "masked_compact"):
        raise ValueError("unknown grouped implementation")
    if config.implementation == "masked_compact":
        if query.mode != "masked":
            raise ValueError("masked_compact requires masked mode")
        if device is not None and device.compute_capability != (12, 0):
            raise ValueError("masked_compact is not qualified on this compute capability")
    if config.implementation == "joint_v1":
        if query.mode != "contiguous" or query.m_capacity > 131072 or (config.tile_m, config.tile_n, config.tile_k) != (128, 128, 64):
            raise ValueError("joint_v1 requires contiguous capacity <=131072 and full tile 128x128x64")
        if device is not None and device.compute_capability != (12, 0):
            raise ValueError("joint_v1 is not qualified on this compute capability")
    elif query.mode == "contiguous" and config.tile_k != 128:
        raise ValueError("the single grouped-labels kernel requires tile K 128")
    if query.k % config.tile_k:
        raise ValueError("grouped FP8 K must divide into complete staged tiles")


def _validate_query(query, device):
    validate_query(query)


def _qualified_pro(device):
    return (device is not None and device.vendor == "nvidia"
            and device.compute_capability == (12, 0) and device.sm_count == 188
            and device.product_name == "nvidia rtx pro 6000 blackwell server edition")


def joint_eligible(query, device):
    return (query.mode == "contiguous" and _qualified_pro(device)
            and ((query.m_capacity <= 131072
                  and (query.n + 127) // 128 >= 16 and query.k >= 2048)
                 or (query.num_groups == 64
                     and (query.n, query.k) in ((1024, 4096), (4096, 512))
                     and 4096 < query.m_capacity <= 65536)
                 or (query.num_groups == 96
                     and (query.n, query.k) in ((1152, 5120), (5120, 640))
                     and 4096 < query.m_capacity <= 65536)
                 or (query.num_groups == 384
                     and (query.n, query.k) == (576, 5120)
                     and 4096 < query.m_capacity <= 131072)))


def default_config(query, device):
    if joint_eligible(query, device):
        return MGroupFP8GemmConfig(backend="cutedsl", tile_m=128, tile_n=128,
                                  tile_k=64, implementation="joint_v1")
    if query.mode == "contiguous":
        # 64-row tiles halve the persistent-grid tail quantum; large runs
        # keep 128 to avoid the extra per-tile prologue/epilogue cost.
        tile_m = 64 if query.m_capacity <= 4096 else 128
    elif query.m_capacity <= 64:
        tile_m = 32
    elif query.expected_m is not None and query.expected_m <= 32:
        # Decode regime: few live rows per group; a 32-row tile quarters the
        # padded MMA work of tile_m=128.
        tile_m = 32
    else:
        tile_m = 128
    tile_n = 128 if query.n >= 128 else 64
    if query.mode == "masked" and tile_m == 32:
        # More, smaller N tiles balance the persistent grid at small live-m.
        tile_n = 64
    return MGroupFP8GemmConfig(
        backend="cutedsl", tile_m=tile_m, tile_n=tile_n, tile_k=128,
        implementation="masked_compact" if query.mode == "masked" and _qualified_pro(device) else "single",
    )


def _default_config(query, device):
    config = default_config(query, device)
    validate_config(query, config, device)
    return config


def _parameters(query, device):
    contiguous = query.mode == "contiguous"
    joint = joint_eligible(query, device)
    return dict(
        implementation=(("single", "joint_v1") if joint else
                        ("single", "masked_compact") if not contiguous and device is not None and device.compute_capability == (12, 0) else ("single",)),
        tile_m=(64, 128) if contiguous else (16, 32, 64, 128),
        tile_k=(64, 128) if joint or not contiguous else (128,),
    )


TUNING = TuningContract(
    component_id="gemm.mgroup_fp8_gemm",
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(field.name for field in fields(MGroupFP8GemmQuery)),
    config_fields=frozenset(field.name for field in fields(MGroupFP8GemmConfig)),
    encode_query=lambda query: {field.name: getattr(query, field.name) for field in fields(query)},
    encode_config=MGroupFP8GemmConfig.to_dict,
    decode_config=MGroupFP8GemmConfig.from_config,
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=_default_config,
    candidate_contract_version=8,
    semantic_version=3,
    knobs=(
        Knob(name="implementation", values=("single", "joint_v1", "masked_compact"), binding=ParameterBinding.COMPILE),
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(name="tile_m", values=(16, 32, 64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_n", values=(64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_k", values=(64, 128), binding=ParameterBinding.COMPILE),
    ),
    parameters=_parameters,
)


__all__ = [
    "MODES",
    "MGroupFP8GemmQuery",
    "MGroupFP8GemmConfig",
    "TUNING",
    "default_config",
    "validate_config",
    "validate_query",
]
