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
    runtime_token_limit: bool = False
    health_summary: bool = False
    history_depth: int = 0
    anchor_summary: bool = False
    observe_all_ranks: bool = False
    runtime_phase_ranges: bool = False

    def __post_init__(self):
        from ..residency.contracts import PHASES
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
        if type(self.runtime_token_limit) is not bool:
            raise TypeError("runtime_token_limit must be boolean")
        if type(self.observe_all_ranks) is not bool:
            raise TypeError("observe_all_ranks must be boolean")
        if type(self.runtime_phase_ranges) is not bool:
            raise TypeError("runtime_phase_ranges must be boolean")
        if self.runtime_phase_ranges and (self.phases != ("decode", "prefill") or self.sample_every != 1 or self.runtime_token_limit):
            raise ValueError("phase ranges require decode/prefill, unsampled counts and no token-limit observer")

        if type(self.health_summary) is not bool:
            raise TypeError("health_summary must be boolean")
        if self.health_summary and (self.phases != ("decode",) or self.rank != self.owner_rank):
            raise ValueError("health summary requires owner-rank decode counters")
        if type(self.anchor_summary) is not bool or (self.anchor_summary and not self.health_summary):
            raise ValueError("anchor summary requires explicit health preparation")
        _integer("history_depth", self.history_depth)
        if self.history_depth and (self.phases != ("decode",) or self.rank != self.owner_rank):
            raise ValueError("routing history requires owner-rank decode counters")

    @property
    def history_bytes(self):
        """Each device and pinned-host ring owns this many payload bytes."""
        return self.history_depth * self.storage_bytes

    @property
    def health_device_bytes(self):
        return (sum(e for _, e in self.layers)*(17 if self.anchor_summary else 16)
                + len(self.layers)*(96 if self.anchor_summary else 80)
                if self.health_summary else 0)

    @property
    def health_host_bytes(self):
        return len(self.layers)*(56 if self.anchor_summary else 48) if self.health_summary else 0

    @property
    def storage_bytes(self):
        return (sum(((experts+6)//2*2)*8 for _, experts in self.layers)*len(self.phases) + 16
                + (8 * ((self.max_tokens + 1) // 2) if self.runtime_phase_ranges else 0)
                if self.rank == self.owner_rank or self.observe_all_ranks else 0)


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


TUNING = TuningContract(component_id="moe.routing_profile", query_schema_version=6,
    config_schema_version=1, query_fields=frozenset(RoutingProfileQuery.__dataclass_fields__),
    config_fields=frozenset(RoutingProfileConfig.__dataclass_fields__), encode_query=asdict,
    encode_config=asdict, decode_config=lambda p: RoutingProfileConfig(**dict(p)),
    validate_query=_validate_query, validate_config=_validate,
    default_config=lambda q, d: RoutingProfileConfig(), candidate_contract_version=1,
    knobs=(Knob(name="backend", values=("cute_counters",), binding=ParameterBinding.COMPILE),),
    parameters=lambda q, d: ParameterSpace.create(TUNING.knobs),
    materialize=lambda q, d, c: RoutingProfileConfig(**dict(c)))
