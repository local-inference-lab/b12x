"""Immutable geometry and ownership for prepared routing counters."""
from dataclasses import asdict, dataclass

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract
from .residency import _integer


@dataclass(frozen=True, kw_only=True)
class RoutingProfileQuery:
    layers: tuple[tuple[str, int], ...]
    max_tokens: int
    max_top_k: int
    phases: tuple[str, ...] = ("decode",)
    sample_every: int = 1
    rank: int = 0
    owner_rank: int = 0
    tp_size: int = 1
    expert_parallel: bool = False

    def __post_init__(self):
        from .automatic import PHASES
        object.__setattr__(self, "layers", tuple(tuple(x) for x in self.layers))
        object.__setattr__(self, "phases", tuple(self.phases))
        if not self.layers or len({n for n, _ in self.layers}) != len(self.layers):
            raise ValueError("routing profiler requires unique layers")
        for name, experts in self.layers:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("routing profiler requires layer identity")
            _integer("experts", experts, 1)
            if experts >= 2**31:
                raise ValueError("expert count exceeds supported geometry")
        if not self.phases or len(set(self.phases)) != len(self.phases) or any(p not in PHASES for p in self.phases):
            raise ValueError("routing profiler requires unique explicit phases")
        for key in ("max_tokens", "max_top_k", "sample_every", "tp_size"):
            _integer(key, getattr(self, key), 1)
        for key in ("rank", "owner_rank"):
            _integer(key, getattr(self, key))
            if getattr(self, key) >= self.tp_size:
                raise ValueError("profiling rank must belong to the TP group")
        if self.max_tokens*self.max_top_k >= 2**31 or self.sample_every >= 2**63:
            raise ValueError("routing capacity or sampling interval exceeds the counter ABI")
        if self.expert_parallel is not False:
            raise ValueError("expert-parallel profiling requires global-ID semantics and is unsupported")

    @property
    def storage_bytes(self):
        return (sum(((experts+6)//2*2)*8 for _, experts in self.layers)*len(self.phases) + 16
                if self.rank == self.owner_rank else 0)


@dataclass(frozen=True, kw_only=True)
class RoutingProfileConfig:
    backend: str = "cute_counters"


def _validate_query(query, device):
    if not isinstance(query, RoutingProfileQuery):
        raise TypeError("routing profiler requires RoutingProfileQuery")
    if device is not None and device.compute_capability not in ((10, 3), (12, 0), (12, 1)):
        raise ValueError("routing counters require SM103, SM120, or SM121")


def _validate(query, config, device):
    _validate_query(query, device)
    if config != RoutingProfileConfig():
        raise ValueError("unsupported routing counter configuration")


TUNING = TuningContract(component_id="moe.routing_profile", query_schema_version=1,
    config_schema_version=1, query_fields=frozenset(RoutingProfileQuery.__dataclass_fields__),
    config_fields=frozenset(RoutingProfileConfig.__dataclass_fields__), encode_query=asdict,
    encode_config=asdict, decode_config=lambda p: RoutingProfileConfig(**dict(p)),
    validate_query=_validate_query, validate_config=_validate,
    default_config=lambda q, d: RoutingProfileConfig(), candidate_contract_version=1,
    knobs=(Knob(name="backend", values=("cute_counters",), binding=ParameterBinding.COMPILE),),
    parameters=lambda q, d: ParameterSpace.create(TUNING.knobs),
    materialize=lambda q, d, c: RoutingProfileConfig(**dict(c)))
