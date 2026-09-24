"""Out-of-band calibration and static expert placement for an execution lane."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ..residency.contracts import PHASES, LayerRoutingCounts, RoutingSnapshot
from .residency import ExpertMemoryBudget, ExpertResidencyPlan, _integer

PROFILE_VERSION = 2
IMPLEMENTATION_VERSION = 1
ALGORITHM = "selection_density_joint_v2"



def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _fraction(name, value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ResidencyLayerSpec:
    """Checkpoint recipe and planned capacity for one native MXFP4 layer."""
    layer: str
    experts: int
    hidden: int
    intermediate: int
    max_tokens: int
    max_top_k: int
    gate_first: bool = False
    swiglu_limit: float | None = None
    source_format: str = "fp4_e8m0_k32"
    numerical_mode: str = "mxfp8_fp32_activation_bf16_expert_ordered_fma"
    minimum_hot: int = 0
    maximum_hot: int | None = None

    def __post_init__(self):
        _text("layer", self.layer)
        from ._residency_tuning import TUNING
        TUNING.validate_query(self.query(0), None)
        if self.hidden % 256:
            raise ValueError("the public native MXFP4/A8 weight planner requires H divisible by 256")
        if self.source_format != "fp4_e8m0_k32":
            raise ValueError("automatic residency supports native MXFP4 E8M0 K32 only")
        _integer("minimum_hot", self.minimum_hot)
        if self.maximum_hot is None:
            object.__setattr__(self, "maximum_hot", self.experts)
        _integer("maximum_hot", self.maximum_hot)
        if not self.minimum_hot <= self.maximum_hot <= self.experts:
            raise ValueError("hot count bounds exceed layer geometry")

    @classmethod
    def from_weight_plan(cls, *, layer, weight_plan, capacity, **bounds):
        """Derive geometry from the loader's canonical source and activation contract."""
        import torch
        from .planning import WeightPlan, ActivationMode
        from .source import PackedSource, PackedSourceFormat
        from .weights import WeightPacking
        if (not isinstance(weight_plan, WeightPlan) or not isinstance(weight_plan.source, PackedSource)
                or weight_plan.source.format != PackedSourceFormat.MXFP4_E8M0_K32
                or weight_plan.prepared_format.packing is not WeightPacking.SOURCE_NATIVE
                or weight_plan.activation.mode != ActivationMode.A8
                or weight_plan.activation.io_dtype != torch.bfloat16
                or weight_plan.activation.nonlinearity != "silu"
                or weight_plan.activation.swiglu_alpha is not None
                or weight_plan.activation.swiglu_beta is not None):
            raise ValueError("automatic residency requires the native MXFP4/A8 BF16 SiLU recipe")
        return cls(layer=layer, experts=weight_plan.geometry.num_experts,
            hidden=weight_plan.geometry.hidden_size, intermediate=weight_plan.geometry.intermediate_size,
            max_tokens=capacity.max_tokens, max_top_k=capacity.top_k,
            gate_first=weight_plan.source.w13_layout.value == "w31",
            swiglu_limit=weight_plan.activation.swiglu_limit, **bounds)

    def query(self, hot):
        from ._residency_tuning import ResidencyQuery
        return ResidencyQuery(hidden=self.hidden, intermediate=self.intermediate,
            experts=self.experts, hot_experts=hot, max_tokens=self.max_tokens,
            max_top_k=self.max_top_k, gate_first=self.gate_first, swiglu_limit=self.swiglu_limit,
            numerical_mode=self.numerical_mode, profile_hash="0"*64, model_fingerprint="accounting")

    @property
    def expert_bytes(self):
        from ._residency_storage import tier_layout
        return tier_layout(1, self.hidden, self.intermediate)[1]

    def memory(self, hot):
        from ._residency_storage import accounting
        return accounting(self.query(hot))


@dataclass(frozen=True, kw_only=True)
class ResidencyModelSpec:
    checkpoint_fingerprint: str
    layers: tuple[ResidencyLayerSpec, ...]
    model_revision: str = "unspecified"
    tokenizer_revision: str = "unspecified"

    def __post_init__(self):
        for key in ("checkpoint_fingerprint", "model_revision", "tokenizer_revision"):
            _text(key, getattr(self, key))
        object.__setattr__(self, "layers", tuple(sorted(self.layers, key=lambda x: x.layer)))
        if not self.layers or len({s.layer for s in self.layers}) != len(self.layers):
            raise ValueError("model must declare unique MoE layers")

    def workspace_estimate(self, *, concurrent_lanes=1):
        """Compare owned scratch with a hypothetical sequential lane arena.

        Only private bytes are admitted by the allocator. The arena estimate is
        diagnostic: no storage is aliased and no saving is spent on hot rows.
        """
        _integer("concurrent_lanes", concurrent_lanes, 1)
        sizes = tuple(s.memory(0).scratch_bytes for s in self.layers)
        private = sum(sizes)*concurrent_lanes
        arena = max(sizes)*concurrent_lanes
        return {"private_bytes": private, "hypothetical_arena_bytes": arena,
                "potential_savings_bytes": private-arena, "concurrent_lanes": concurrent_lanes,
                "arena_implemented": False}

    @property
    def identity(self):
        return _digest(asdict(self))


@dataclass(frozen=True, kw_only=True)
class ModelExpertMemoryBudget:
    """One model-wide envelope; every reservation is subtracted exactly once.

    Limits describe an integration-owned capacity envelope, not the same free
    memory snapshot independently reused by each layer. Scratch and route maps
    are derived from declared layers. Private workspaces are charged in full.
    """
    hbm_bytes: int
    grace_bytes: int
    non_moe_hbm_bytes: int = 0
    other_hbm_bytes: int = 0
    kv_reserved_bytes: int = 0
    hbm_safety_bytes: int = 0
    grace_safety_bytes: int = 0
    profiling_bytes: int = 0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _integer(name, getattr(self, name))
        if self.reserved_hbm > self.hbm_bytes or self.grace_safety_bytes > self.grace_bytes:
            raise ValueError("model reservations exceed memory capacity")

    @classmethod
    def from_available(cls, *, device, grace_bytes=None, **outstanding_reservations):
        """Snapshot free memory before expert preparation, outside CUDA capture.

        Pass only allocations still to come. Already resident model/KV tensors
        have reduced free HBM and must not be subtracted a second time.
        """
        import torch
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("memory budgeting cannot execute during CUDA graph capture")
            free, _ = torch.cuda.mem_get_info(device)
        if grace_bytes is None:
            grace_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return cls(hbm_bytes=free, grace_bytes=grace_bytes, **outstanding_reservations)

    @property
    def reserved_hbm(self):
        return (self.non_moe_hbm_bytes + self.other_hbm_bytes + self.kv_reserved_bytes
                + self.hbm_safety_bytes + self.profiling_bytes)

    def expert_capacity(self, model):
        scratch = sum(s.memory(0).scratch_bytes + s.memory(0).route_map_bytes for s in model.layers)
        usable = self.hbm_bytes - self.reserved_hbm - scratch
        if usable < 0:
            raise ValueError("model scratch, route maps and reservations exceed HBM")
        return usable, self.grace_bytes - self.grace_safety_bytes


@dataclass(frozen=True, kw_only=True)
class ResidencyHardware:
    compute_capability: tuple[int, int]
    grace_coherent: bool

    def __post_init__(self):
        object.__setattr__(self, "compute_capability", tuple(self.compute_capability))
        if type(self.grace_coherent) is not bool:
            raise ValueError("Grace coherency must come from a platform capability probe")

    @classmethod
    def detect(cls, device):
        """Inspect hardware outside graph capture; SM103 alone is insufficient."""
        import torch
        from b12x._lib.platform import probe_platform
        return cls(compute_capability=torch.cuda.get_device_capability(device),
                   grace_coherent=probe_platform(device).grace_coherent)


@dataclass(frozen=True, kw_only=True)
class ResidencyCalibrationConfig:
    phase: str = "decode"
    sample_every: int = 1
    minimum_observations: int = 10000
    convergence_window: int = 2000
    stable_windows: int = 4
    hot_set_similarity: float = .99
    cold_fraction_delta: float = .002
    request_limit: int | None = None
    token_limit: int | None = None

    def __post_init__(self):
        if self.phase not in (*PHASES, "all"):
            raise ValueError("unknown calibration phase")
        for key in ("sample_every", "minimum_observations", "convergence_window", "stable_windows"):
            _integer(key, getattr(self, key), 1)
        for key in ("request_limit", "token_limit"):
            if getattr(self, key) is not None:
                _integer(key, getattr(self, key), 1)
        _fraction("hot_set_similarity", self.hot_set_similarity)
        _fraction("cold_fraction_delta", self.cold_fraction_delta)


@dataclass(frozen=True, kw_only=True)
class ResidencyMonitorConfig:
    sample_every: int = 128
    minimum_observations: int = 10000
    drift_threshold: float = .02

    def __post_init__(self):
        _integer("sample_every", self.sample_every, 1)
        _integer("minimum_observations", self.minimum_observations, 1)
        _fraction("drift_threshold", self.drift_threshold)


@dataclass(frozen=True, kw_only=True)
class AutomaticResidencyConfig:
    mode: str = "off"
    workload: str = "default"
    provenance: str = "integration routing counters"
    calibration: ResidencyCalibrationConfig = ResidencyCalibrationConfig()
    monitor: ResidencyMonitorConfig = ResidencyMonitorConfig()
    profile_path: str | None = None
    reuse_cache: bool = True
    refresh: str = "restart_required"
    activation: str = "converged"

    def __post_init__(self):
        if self.mode not in ("off", "static", "profile", "auto", "monitor"):
            raise ValueError("residency mode must be off, static, profile, auto, or monitor")
        for name in ("workload", "provenance"):
            _text(name, getattr(self, name))
        if not isinstance(self.calibration, ResidencyCalibrationConfig) or not isinstance(self.monitor, ResidencyMonitorConfig):
            raise TypeError("calibration and monitor require typed configuration")
        if type(self.reuse_cache) is not bool or self.refresh != "restart_required":
            raise ValueError("only explicit cache reuse and restart-required activation are supported")
        if self.activation not in ("converged", "best_available"):
            raise ValueError("activation must be converged or best_available")
        if self.profile_path is not None:
            _text("profile_path", self.profile_path)



def _counts(model, snapshot, phase):
    expected = {s.layer: s.experts for s in model.layers}
    values = {key: [0]*e for key, e in expected.items()}
    decode = {key: 0 for key in expected}
    for row in snapshot.layers:
        if row.layer not in expected or len(row.counts) != expected[row.layer]:
            raise ValueError("snapshot layer geometry differs from model")
        if row.phase == "decode":
            decode[row.layer] += sum(row.counts)
        if phase == "all" or row.phase == phase:
            values[row.layer] = [a+b for a, b in zip(values[row.layer], row.counts, strict=True)]
    return {key: tuple(row) for key, row in values.items()}, decode


def derive_placement(model, budget, counts, *, workload, provenance, phase):
    """Rank real observations by density and admit both memory tiers jointly."""
    if set(counts) != {s.layer for s in model.layers}:
        raise ValueError("placement counts must cover exactly the declared layers")
    for s in model.layers:
        values = counts[s.layer]
        if len(values) != s.experts:
            raise ValueError("count geometry differs from model")
        for value in values:
            _integer("selection count", value)
        if not sum(values):
            raise ValueError(f"layer {s.layer} has no routing observations")
    return _allocate_placement(model, budget, counts, workload=workload,
                               provenance=provenance, phase=phase)


def _allocate_placement(model, budget, counts, *, workload, provenance, phase):
    from ._residency_allocation import allocate_rows
    hot = allocate_rows(model.layers, *budget.expert_capacity(model), counts)
    return tuple(ExpertResidencyPlan(total_experts=s.experts,
        hbm_expert_ids=tuple(sorted(hot[s.layer])),
        grace_expert_ids=tuple(e for e in range(s.experts) if e not in hot[s.layer]),
        layer=s.layer, model_fingerprint=model.checkpoint_fingerprint, workload=workload,
        provenance=provenance, selection_counts=tuple(counts[s.layer]) if counts is not None else (),
        phase=phase if phase in ("decode", "prefill", "all") else "all") for s in model.layers)


def placement_memory(model, profiles):
    if len(profiles) != len(model.layers):
        raise ValueError("placement must cover every layer")
    memory = tuple(s.memory(len(p.hbm_expert_ids)) for s, p in zip(model.layers, profiles, strict=True))
    return dict(hbm_expert_bytes=sum(m.hbm_expert_bytes for m in memory),
                grace_expert_bytes=sum(m.grace_expert_bytes for m in memory),
                scratch_bytes=sum(m.scratch_bytes for m in memory),
                route_map_bytes=sum(m.route_map_bytes for m in memory))


def _cold(profiles, counts):
    total = sum(sum(c) for c in counts.values())
    return sum(sum(counts[p.layer][e] for e in p.grace_expert_ids) for p in profiles) / total if total else 0.0


def _similarity(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right))/len(union) if union else 1.0


@dataclass(frozen=True, kw_only=True)
class ResidencyProfile:
    model: ResidencyModelSpec
    workload: str
    phase: str
    provenance: str
    placements: tuple[ExpertResidencyPlan, ...]
    created_at: str
    converged: bool
    stable_windows: int
    request_count: int
    token_count: int
    sample_every: int
    termination: str
    schema_version: int = PROFILE_VERSION
    implementation_version: int = IMPLEMENTATION_VERSION
    algorithm: str = ALGORITHM
    statistics: str = "selection_counts"

    def __post_init__(self):
        if (type(self.schema_version) is not int or type(self.implementation_version) is not int
                or self.schema_version != PROFILE_VERSION or self.implementation_version != IMPLEMENTATION_VERSION
                or self.algorithm != ALGORITHM or self.statistics != "selection_counts"):
            raise ValueError("unsupported residency profile schema, implementation, algorithm or statistics")
        object.__setattr__(self, "placements", tuple(self.placements))
        if self.phase not in (*PHASES, "all") or type(self.converged) is not bool:
            raise ValueError("invalid profile phase or convergence state")
        if self.termination not in ("converged", "limit") or self.converged != (self.termination == "converged"):
            raise ValueError("profile convergence and termination reason disagree")
        for key in ("workload", "provenance", "created_at", "termination"):
            _text(key, getattr(self, key))
        for key in ("stable_windows", "request_count", "token_count"):
            _integer(key, getattr(self, key))
        _integer("sample_every", self.sample_every, 1)
        if len(self.placements) != len(self.model.layers):
            raise ValueError("profile does not cover model layers")
        for s, p in zip(self.model.layers, self.placements, strict=True):
            if (s.layer != p.layer or s.experts != p.total_experts
                    or p.model_fingerprint != self.model.checkpoint_fingerprint
                    or p.workload != self.workload or p.provenance != self.provenance
                    or p.phase != (self.phase if self.phase in ("all", "decode", "prefill") else "all")
                    or not s.minimum_hot <= len(p.hbm_expert_ids) <= s.maximum_hot
                    or len(p.selection_counts) != s.experts or not sum(p.selection_counts)):
                raise ValueError("profile layer identity, observations or geometry mismatch")

    @property
    def profile_hash(self):
        return _digest(asdict(self))

    @property
    def expected_cold_fraction(self):
        return _cold(self.placements, {p.layer: p.selection_counts for p in self.placements})

    @property
    def memory(self):
        return placement_memory(self.model, self.placements)

    def layer_budget(self, layer):
        """Return an apportioned per-plan budget with no repeated global reserve."""
        index = next(i for i, s in enumerate(self.model.layers) if s.layer == layer)
        m = self.model.layers[index].memory(len(self.placements[index].hbm_expert_ids))
        return ExpertMemoryBudget(hbm_bytes=m.hbm_total_bytes, grace_bytes=m.grace_expert_bytes)

    def to_dict(self):
        return {**asdict(self), "profile_hash": self.profile_hash, "memory": self.memory,
                "expected_cold_fraction": self.expected_cold_fraction}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        digest, memory, cold = (values.pop(key) for key in ("profile_hash", "memory", "expected_cold_fraction"))
        model = dict(values["model"])
        model["layers"] = tuple(ResidencyLayerSpec(**row) for row in model["layers"])
        values["model"] = ResidencyModelSpec(**model)
        values["placements"] = tuple(ExpertResidencyPlan(**row) for row in values["placements"])
        result = cls(**values)
        if digest != result.profile_hash or memory != result.memory or cold != result.expected_cold_fraction:
            raise ValueError("residency profile integrity mismatch")
        return result

    def validate_activation(self, config):
        """Admission for automatic reuse is separate from artifact validity."""
        if not self.converged and config.activation != "best_available":
            raise ValueError("profile is not converged; automatic activation requires "
                             "convergence or explicit activation='best_available'")

    def validate(self, *, model, config, budget, hardware):
        if self.model != model:
            raise ValueError("checkpoint, revision, layer geometry or numerical recipe mismatch")
        if self.workload != config.workload or self.phase != config.calibration.phase:
            raise ValueError("workload or routing phase mismatch")
        if hardware.compute_capability != (10, 3):
            raise ValueError("hierarchical residency requires physical SM103")
        if any(p.grace_expert_ids for p in self.placements) and not hardware.grace_coherent:
            raise ValueError("profile requires verified Grace coherency")
        hot, cold = budget.expert_capacity(model)
        if self.memory["hbm_expert_bytes"] > hot or self.memory["grace_expert_bytes"] > cold:
            raise ValueError("profile does not fit the current model memory budget")


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".profile-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ResidencyProfileStore:
    """Content-addressed artifacts with an atomic index per model/workload/phase."""
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path.home()/".cache"/"b12x"/"expert_residency"

    def _namespace(self, model, workload, phase):
        return self.root/_digest(model.checkpoint_fingerprint)/_digest(workload)/_digest((model.identity, phase))

    def write(self, profile):
        directory = self._namespace(profile.model, profile.workload, profile.phase)
        path = directory/f"{profile.profile_hash}.json"
        _atomic_write(path, profile.to_dict())
        index = {"schema_version": PROFILE_VERSION, "profile_hash": profile.profile_hash}
        _atomic_write(directory/"latest.json", index)
        if profile.converged:
            _atomic_write(directory/"current.json", index)
        return path

    def read(self, *, model, config):
        if config.profile_path is not None:
            path = Path(config.profile_path)
            expected = None
        else:
            directory = self._namespace(model, config.workload, config.calibration.phase)
            path = directory/("latest.json" if config.activation == "best_available" else "current.json")
            if not path.exists() and config.activation == "converged":
                # Surface the actual rejection reason when only an experiment
                # exists. A bounded experiment cannot replace a converged index.
                path = directory/"latest.json"
            index = json.loads(path.read_text())
            if set(index) != {"schema_version", "profile_hash"} or index["schema_version"] != PROFILE_VERSION:
                raise ValueError("unsupported profile store index")
            expected = index["profile_hash"]
            if not isinstance(expected, str) or len(expected) != 64 or any(x not in "0123456789abcdef" for x in expected):
                raise ValueError("invalid profile store digest")
            path = directory/f"{expected}.json"
        profile = ResidencyProfile.from_dict(json.loads(path.read_text()))
        if expected is not None and profile.profile_hash != expected:
            raise ValueError("profile store index integrity mismatch")
        return profile


@dataclass(frozen=True, kw_only=True)
class ResidencyProgress:
    state: str
    reason: str
    profile: ResidencyProfile | None = None
    profile_path: str | None = None
    stable_windows: int = 0
    similarities: tuple[tuple[str, float], ...] = ()
    cold_fractions: tuple[tuple[str, float], ...] = ()
    expected_cold_fraction: float | None = None


@dataclass(frozen=True, kw_only=True)
class ResidencyDrift:
    recommended: bool
    observations: int
    active_cold_fraction: float
    candidate_cold_fraction: float
    similarities: tuple[tuple[str, float], ...]
    membership_changed_fraction: float
    bytes_moved: int


class ResidencyController:
    """Thin engine hooks; this object never reloads or mutates a running model.

    The integration pauses graph submission before snapshot/reset. Replicated
    TP routes are owned by one designated rank; other ranks receive the artifact
    out of band. Expert-parallel local IDs are unsupported.
    """
    def __init__(self, *, config, model, budget, hardware, store=None, owner_rank=0):
        self.config, self.model, self.hardware = config, model, hardware
        # Counter storage is model-wide and never subtracted again by a layer.
        phases = len(PHASES) if config.calibration.phase == "all" else 1
        counted = config.mode not in ("off", "static")
        counters = (sum((s.experts+6)//2*2*8 for s in model.layers)*phases + 16) if counted else 0
        self.budget = replace(budget, profiling_bytes=budget.profiling_bytes+counters)
        self.store = store if store is not None else ResidencyProfileStore()
        _integer("owner_rank", owner_rank)
        self.owner_rank = owner_rank
        self.active = None
        self.progress = ResidencyProgress(state="new", reason="startup has not run")
        self._previous = None
        self._previous_placement = None
        self._previous_cold = None
        self._epoch = None
        self._stable = 0
        self._requests = self._tokens = 0
        self._last_seen = None
        self._monitor_previous = None

    def startup(self):
        if self.progress.state != "new":
            raise RuntimeError("residency startup may run only once")
        c = self.config
        if c.mode == "off":
            self.progress = ResidencyProgress(state="off", reason="automatic residency disabled")
            return self.progress
        if self.hardware.compute_capability != (10, 3):
            raise ValueError("automatic hierarchical deployment requires SM103")
        self.budget.expert_capacity(self.model)
        reason = "profile discovery disabled"
        if c.mode == "static":
            # A static lane never counts routes: it serves a validated profile
            # or the balanced bootstrap placement until the engine restarts.
            if c.reuse_cache or c.profile_path is not None:
                try:
                    profile = self.store.read(model=self.model, config=c)
                    profile.validate(model=self.model, config=c, budget=self.budget, hardware=self.hardware)
                    profile.validate_activation(c)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    if c.profile_path is not None:
                        raise ValueError(f"pinned residency profile is unusable: {error}") from error
                    reason = f"profile invalid or unavailable: {error}"
                else:
                    self.active = profile
                    self.progress = ResidencyProgress(state="ready",
                        reason="compatible static profile found; prepare before serving", profile=profile,
                        expected_cold_fraction=profile.expected_cold_fraction)
                    return self.progress
            self.progress = ResidencyProgress(state="ready", reason=f"{reason}; balanced static placement")
            return self.progress
        if c.mode in ("auto", "monitor") and (c.reuse_cache or c.profile_path is not None):
            try:
                profile = self.store.read(model=self.model, config=c)
                profile.validate(model=self.model, config=c, budget=self.budget, hardware=self.hardware)
                profile.validate_activation(c)
                self.active = profile
                self.progress = ResidencyProgress(state="monitoring" if c.mode == "monitor" else "ready",
                    reason="compatible static profile found; prepare before serving", profile=profile,
                    expected_cold_fraction=profile.expected_cold_fraction)
                return self.progress
            except (OSError, ValueError, TypeError, KeyError) as error:
                reason = f"profile invalid or unavailable: {error}"
        if c.mode == "monitor":
            raise ValueError(f"monitor requires a valid static profile: {reason}")
        self.progress = ResidencyProgress(state="calibrating", reason=reason)
        return self.progress

    def profiler_query(self, *, rank=0, tp_size=1):
        """Declare counters only for calibration/monitoring, before capture."""
        if self.progress.state not in ("calibrating", "monitoring"):
            return None
        from ._routing_profile_tuning import RoutingProfileQuery
        phases = PHASES if self.config.calibration.phase == "all" else (self.config.calibration.phase,)
        sample = self.config.monitor.sample_every if self.progress.state == "monitoring" else self.config.calibration.sample_every
        return RoutingProfileQuery(layers=tuple((s.layer, s.experts) for s in self.model.layers),
            phases=phases, max_tokens=max(s.max_tokens for s in self.model.layers),
            max_top_k=max(s.max_top_k for s in self.model.layers), sample_every=sample,
            rank=rank, owner_rank=self.owner_rank, tp_size=tp_size)

    def placements(self):
        """Use a validated profile or a bootstrap with balanced resident fractions.

        Bootstrap chooses storage rows without synthetic workload observations.
        Calibration still needs a working hierarchical operator on the target.
        """
        if self.active is not None:
            return self.active.placements
        static = self.config.mode == "static" and self.progress.state == "ready"
        if not static and self.progress.state not in ("calibrating", "profile_saved", "restart_required", "insufficient"):
            raise RuntimeError("automatic placement is not active")
        provenance = "balanced static placement" if static else "balanced fractional calibration bootstrap"
        return _allocate_placement(self.model, self.budget, None, workload=self.config.workload,
            provenance=provenance, phase=self.config.calibration.phase)

    def plan_execution(self, *, layer, weight_plan, weights):
        """Declare the existing public execution Plan using apportioned storage."""
        from .api import plan_execution
        from .execution import ExecutionCapacity
        spec = next(s for s in self.model.layers if s.layer == layer)
        capacity = ExecutionCapacity(max_tokens=spec.max_tokens, top_k=spec.max_top_k)
        actual = ResidencyLayerSpec.from_weight_plan(layer=layer, weight_plan=weight_plan,
            capacity=capacity, minimum_hot=spec.minimum_hot, maximum_hot=spec.maximum_hot)
        if actual != spec:
            raise ValueError("loader weight contract differs from the automatic model declaration")
        placement = next(p for p in self.placements() if p.layer == layer)
        if placement.grace_expert_ids and not self.hardware.grace_coherent:
            raise ValueError("calibration storage requires verified Grace coherency")
        memory = spec.memory(len(placement.hbm_expert_ids))
        return plan_execution(experts=weight_plan, weights=weights, capacity=capacity,
            placement=placement, memory_budget=ExpertMemoryBudget(hbm_bytes=memory.hbm_total_bytes,
                                                                  grace_bytes=memory.grace_expert_bytes))

    def _observe(self, snapshot):
        if snapshot.rank != self.owner_rank:
            raise ValueError("only the authoritative TP rank may submit routing statistics")
        if self._epoch is None:
            self._epoch = snapshot.epoch
        if snapshot.epoch != self._epoch:
            raise ValueError("counter epoch changed during calibration; create a new controller")
        counts, decode = _counts(self.model, snapshot, self.config.calibration.phase)
        if self._last_seen is not None and any(b < a for key in counts for a, b in zip(self._last_seen[key], counts[key], strict=True)):
            raise ValueError("cumulative routing counters decreased")
        self._last_seen = counts
        return counts, decode

    def observe(self, snapshot, *, request_count=0, token_count=0):
        """Evaluate a cumulative snapshot outside the request execution path."""
        if self.progress.state != "calibrating":
            raise RuntimeError("routing observations require active calibration")
        for name, value in (("request_count", request_count), ("token_count", token_count)):
            _integer(name, value)
        if request_count < self._requests or token_count < self._tokens:
            raise ValueError("engine observation totals decreased")
        self._requests, self._tokens = request_count, token_count
        counts, decode = self._observe(snapshot)
        cfg = self.config.calibration
        limit = ((cfg.request_limit is not None and request_count >= cfg.request_limit)
                 or (cfg.token_limit is not None and token_count >= cfg.token_limit))
        baseline = self._previous or {key: (0,)*len(value) for key, value in counts.items()}
        window = {key: tuple(b-a for a, b in zip(baseline[key], counts[key], strict=True)) for key in counts}
        sufficient = all(sum(row) >= cfg.minimum_observations for row in counts.values())
        if cfg.phase in ("decode", "all"):
            sufficient = sufficient and all(decode.values())
        if not all(sum(row) >= cfg.convergence_window for row in window.values()):
            if not limit:
                return self.progress
            if not sufficient:
                self.progress = ResidencyProgress(state="insufficient", reason="calibration limit reached without enough routes per layer")
                return self.progress
        kwargs = dict(workload=self.config.workload, provenance=self.config.provenance, phase=cfg.phase)
        candidate = derive_placement(self.model, self.budget, counts, **kwargs)
        similarities, cold_rows = (), ()
        if all(sum(row) >= cfg.convergence_window for row in window.values()):
            proposed = derive_placement(self.model, self.budget, window, **kwargs)
            cold_rows = tuple((p.layer, sum(window[p.layer][e] for e in p.grace_expert_ids)/sum(window[p.layer])) for p in proposed)
            if self._previous_placement is not None:
                similarities = tuple((p.layer, _similarity(p.hbm_expert_ids, old.hbm_expert_ids))
                    for p, old in zip(proposed, self._previous_placement, strict=True))
                stable = (sufficient and all(v >= cfg.hot_set_similarity for _, v in similarities)
                    and all(abs(value-dict(self._previous_cold)[layer]) <= cfg.cold_fraction_delta for layer, value in cold_rows)
                    and all(_similarity(p.hbm_expert_ids, cumulative.hbm_expert_ids) >= cfg.hot_set_similarity
                            for p, cumulative in zip(proposed, candidate, strict=True)))
                self._stable = self._stable+1 if stable else 0
            self._previous, self._previous_placement, self._previous_cold = counts, proposed, cold_rows
        converged = sufficient and self._stable >= cfg.stable_windows
        if limit or converged:
            if not sufficient:
                self.progress = ResidencyProgress(state="insufficient", reason="calibration limit reached without enough routes per layer")
                return self.progress
            profile = ResidencyProfile(model=self.model, workload=self.config.workload, phase=cfg.phase,
                provenance=self.config.provenance, placements=candidate,
                created_at=datetime.now(timezone.utc).isoformat(), converged=converged,
                stable_windows=self._stable, request_count=request_count, token_count=token_count,
                sample_every=cfg.sample_every, termination="converged" if converged else "limit")
            profile.validate(model=self.model, config=self.config, budget=self.budget, hardware=self.hardware)
            path = self.store.write(profile)
            activate = self.config.mode == "auto" and (converged or self.config.activation == "best_available")
            self.progress = ResidencyProgress(state="restart_required" if activate else "profile_saved",
                reason="calibration converged" if converged else "explicit calibration limit reached; convergence not established",
                profile=profile, profile_path=str(path), stable_windows=self._stable,
                similarities=similarities, cold_fractions=cold_rows, expected_cold_fraction=profile.expected_cold_fraction)
        else:
            self.progress = ResidencyProgress(state="calibrating", reason="awaiting stable routing windows",
                stable_windows=self._stable, similarities=similarities, cold_fractions=cold_rows,
                expected_cold_fraction=_cold(candidate, counts))
        return self.progress

    def monitor(self, snapshot):
        """Report potential cold-selection reduction; never activate a candidate."""
        if self.progress.state != "monitoring" or self.active is None:
            raise RuntimeError("drift detection requires an active static profile")
        cumulative, _ = self._observe(snapshot)
        baseline = self._monitor_previous or {key: (0,)*len(value) for key, value in cumulative.items()}
        counts = {key: tuple(b-a for a, b in zip(baseline[key], cumulative[key], strict=True)) for key in cumulative}
        if any(sum(row) < self.config.monitor.minimum_observations for row in counts.values()):
            return None
        self._monitor_previous = cumulative
        candidate = derive_placement(self.model, self.budget, counts, workload=self.config.workload,
            provenance=self.config.provenance, phase=self.config.calibration.phase)
        active_cold, candidate_cold = _cold(self.active.placements, counts), _cold(candidate, counts)
        pairs = tuple(zip(self.model.layers, self.active.placements, candidate, strict=True))
        moved = sum(len(set(a.hbm_expert_ids)^set(b.hbm_expert_ids))*s.expert_bytes for s, a, b in pairs)
        changed = sum(len(set(a.hbm_expert_ids)-set(b.hbm_expert_ids)) for _, a, b in pairs)
        hot = sum(len(a.hbm_expert_ids) for _, a, _ in pairs)
        return ResidencyDrift(recommended=active_cold-candidate_cold >= self.config.monitor.drift_threshold,
            observations=sum(sum(row) for row in counts.values()), active_cold_fraction=active_cold,
            candidate_cold_fraction=candidate_cold,
            similarities=tuple((s.layer, _similarity(a.hbm_expert_ids, b.hbm_expert_ids)) for s, a, b in pairs),
            membership_changed_fraction=changed/hot if hot else 0.0, bytes_moved=moved)
