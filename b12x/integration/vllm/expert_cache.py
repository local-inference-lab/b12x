"""Model admission and preparation for the experimental vLLM SM120 cache.

The engine supplies loaded CPU sources, serving capacity and iteration phase.
This module owns cache declarations, profile validation and budget arithmetic.
It neither schedules requests nor pauses the engine.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._cache_preparation import ExpertCacheQuery, memory_for
from b12x.moe.fused_moe.residency import profile_from_counts
from b12x.moe.residency import RoutingObservationSpec, ResidencyAnchor
from b12x.preparation import PreparedCall
from .residency_epoch import (
    ResidencyEpochRuntime,
    ResidencyServingMemory,
    bind_sm120_epoch_layer,
)


@dataclass(frozen=True, kw_only=True)
class ExpertCacheServingConfig:
    """Explicit experimental recipe and one worker's model-wide envelopes.

    TP/DP/PP=1 is the first loader qualification scope. Host backing is entirely
    cacheable mapped memory; source CPU tensors are charged separately. Static
    and adaptive require the same learned artifact. Profile writes an artifact
    only when requested out of band and never changes live placement.
    """

    mode: str
    activation: str
    profile_path: str
    workload: str
    expert_device_bytes: int
    host_bytes: int
    kv_reserved_bytes: int
    graph_reserved_bytes: int
    device_safety_bytes: int
    host_safety_bytes: int
    max_pairs_per_layer: int = 2
    health_probes: bool = False
    history_depth: int = 0
    anchor_health: bool = False

    def __post_init__(self):
        if (
            self.mode not in ("static", "adaptive", "profile")
            or self.activation != "w4a16"
        ):
            raise ValueError(
                "experimental SM120 cache requires static/adaptive/profile and explicit w4a16 activation"
            )
        if not self.profile_path or not self.workload:
            raise ValueError(
                "expert cache requires a profile path and workload identity"
            )
        for name in (
            "expert_device_bytes",
            "host_bytes",
            "kv_reserved_bytes",
            "graph_reserved_bytes",
            "device_safety_bytes",
            "host_safety_bytes",
            "max_pairs_per_layer",
            "history_depth",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.health_probes) is not bool or (self.health_probes and self.mode != "adaptive"):
            raise ValueError("health probes require explicitly adaptive serving")
        if type(self.anchor_health) is not bool or (self.anchor_health and not self.health_probes):
            raise ValueError("anchor health requires explicitly adaptive health probes")
        if self.history_depth and self.mode != "adaptive":
            raise ValueError("routing history requires explicitly adaptive serving")
        if self.mode == "adaptive" and not self.max_pairs_per_layer:
            raise ValueError("adaptive cache requires positive prepared fill capacity")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class ExpertCacheModel:
    """One serialized execution lane, owned alongside its PreparationSession."""

    def __init__(self, config, checkpoint_id, device):
        self.config, self.checkpoint_id, self.device = (
            config,
            checkpoint_id,
            torch.device(device),
        )
        self.sources, self.plans, self.placements = {}, {}, {}
        self.counter = self.memory = self.runtime = None
        self.anchor = None
        self.capacity = None
        self.source_reserved_bytes = 0
        self._counters = None
        self.observed_layers = ()
        self.load_device_peak_bytes = 0

    def reserve_source(self, nbytes):
        """Admit CPU source parameters before the loader allocates them."""
        if (
            self.source_reserved_bytes + nbytes + self.config.host_safety_bytes
            > self.config.host_bytes
        ):
            raise ValueError(
                "routed CPU checkpoint sources exceed the model host envelope"
            )
        self.source_reserved_bytes += nbytes

    def close(self):
        """Release cache references after readers and PreparationSession retire."""
        if any(p.prepared is not None for p in self.plans.values()) or (
            self.counter is not None and self.counter.prepared is not None
        ):
            raise RuntimeError("close the preparation session after graphs before cache owners")
        # The engine has drained every stream, including pending health copies.
        # Break counter/health cycles while their CUDA owners are still live.
        if self._counters is not None:
            self._counters.health = self._counters.history = None
        self.runtime = self._counters = self.counter = None
        self.sources.clear()
        self.plans.clear()
        self.placements.clear()
        self.source_reserved_bytes = 0

    def add_source(self, source):
        name = source.weights.layer_name
        if (
            self.plans
            or name in self.sources
            or source.weights.checkpoint_fingerprint != self.checkpoint_id
        ):
            raise ValueError("duplicate, late or foreign checkpoint source")
        self.sources[name] = source

    def _identity(self):
        return {
            "version": 1,
            "checkpoint": self.checkpoint_id,
            "top_k": None if self.capacity is None else self.capacity.top_k,
            "recipe": "nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum",
            "workload": self.config.workload,
            "layers": {
                name: {
                    **asdict(s.plan.geometry),
                    "w13_layout": s.plan.source.w13_layout.value,
                }
                for name, s in sorted(self.sources.items())
            },
        }

    def declare(self, capacity):
        if self.plans:
            if self.memory is None:
                raise RuntimeError(
                    "model cache admission failed; reload before retrying"
                )
            if self.capacity != capacity:
                raise ValueError("cache capacity changed after model-wide admission")
            return
        if not self.sources:
            raise ValueError("model has no routed expert sources")
        self.capacity = capacity
        self.load_device_peak_bytes = torch.cuda.max_memory_allocated(self.device)
        c = self.config
        profile = None
        if c.mode != "profile":
            profile = json.loads(Path(c.profile_path).read_text())
            expected = profile.pop("hash", None)
            if (
                digest(profile) != expected
                or profile.get("identity") != self._identity()
            ):
                raise ValueError(
                    "expert cache profile hash, checkpoint, workload, geometry or recipe mismatch"
                )
            self.profile_id = expected
        else:
            self.profile_id = "calibration:" + digest(self._identity())
        queries, minimum = {}, {}
        for name, s in sorted(self.sources.items()):
            g = s.plan.geometry
            q = ExpertCacheQuery(
                experts=g.num_experts,
                resident=1,
                hidden=g.hidden_size,
                intermediate=g.intermediate_size,
                max_tokens=capacity.max_tokens,
                top_k=capacity.top_k,
                w13_layout=s.plan.source.w13_layout.value,
                checkpoint_fingerprint=self.checkpoint_id,
                profile_hash="0" * 64,
            )
            queries[name] = q
            minimum[name] = memory_for(q, self.device, s.source_bytes)
        if profile is None:
            counts = dict.fromkeys(queries, 1)
            used = sum(m.hbm_total_bytes for m in minimum.values())
            used += moe.RoutingProfileQuery(
                layers=tuple((name, q.experts) for name, q in queries.items()),
                max_tokens=capacity.max_tokens,
                max_top_k=capacity.top_k,
            ).storage_bytes
            # Fair cold start; learned profiles may later allocate unevenly.
            while True:
                changed = False
                for name, q in queries.items():
                    if counts[name] == q.experts:
                        continue
                    candidate = memory_for(
                        replace(q, resident=counts[name] + 1),
                        self.device,
                        self.sources[name].source_bytes,
                    )
                    delta = candidate.hbm_total_bytes - minimum[name].hbm_total_bytes
                    if used + delta <= c.expert_device_bytes:
                        counts[name] += 1
                        used += delta
                        minimum[name] = candidate
                        changed = True
                if not changed:
                    break
            for name, q in queries.items():
                self.placements[name] = profile_from_counts(
                    counts=(0,) * q.experts,
                    hot_count=counts[name],
                    layer=name,
                    model_fingerprint=self.checkpoint_id,
                    workload=c.workload,
                    provenance="fair bootstrap for explicit calibration",
                    phase="decode",
                )
        else:
            self.placements = {
                name: moe.ExpertResidencyPlan.from_dict(value)
                for name, value in profile["placements"].items()
            }
            if set(self.placements) != set(self.sources):
                raise ValueError(
                    "profile must cover exactly the participating MoE layers"
                )
        memories = {}
        for name, q in queries.items():
            p = self.placements[name]
            if not sum(p.selection_counts) and c.mode != "profile":
                raise ValueError(
                    "static/adaptive serving requires observed learned placements"
                )
            if (
                p.layer != name
                or p.total_experts != q.experts
                or p.model_fingerprint != self.checkpoint_id
                or p.workload != c.workload
                or p.phase != "decode"
            ):
                raise ValueError(
                    "learned placement differs from checkpoint, workload, layer or decode phase"
                )
            pairs = (
                min(
                    c.max_pairs_per_layer,
                    len(p.hbm_expert_ids),
                    len(p.grace_expert_ids),
                )
                if c.mode == "adaptive"
                else 0
            )
            q = replace(
                q,
                resident=len(p.hbm_expert_ids),
                profile_hash=p.profile_hash,
                max_pairs=pairs,
            )
            memories[name] = memory_for(q, self.device, self.sources[name].source_bytes)
            m = memories[name]
            self.plans[name] = moe.plan_execution(
                experts=self.sources[name],
                capacity=capacity,
                placement=p,
                memory_budget=moe.ExpertMemoryBudget(
                    hbm_bytes=m.hbm_total_bytes, grace_bytes=m.grace_total_bytes
                ),
                updates=moe.ResidencyUpdateCapacity(max_pairs=pairs) if pairs else None,
            )
        if c.mode != "static":
            self.observed_layers = tuple(
                name
                for name, plan in self.plans.items()
                if c.mode == "profile" or plan.query.max_pairs
            )
            if not self.observed_layers:
                raise ValueError(
                    "adaptive mode requires at least one layer with nonresident experts"
                )
            self.counter = moe.plan_routing_profile(
                moe.RoutingProfileQuery(
                    layers=tuple(
                        (name, self.plans[name].query.experts)
                        for name in self.observed_layers
                    ),
                    max_tokens=capacity.max_tokens,
                    max_top_k=capacity.top_k,
                    runtime_token_limit=True,
                    health_summary=c.health_probes,
                    history_depth=c.history_depth,
                    anchor_summary=c.anchor_health,
                )
            )
        if c.anchor_health:
            self.anchor = ResidencyAnchor(
                profile_id=self.profile_id, checkpoint=self.checkpoint_id,
                recipe=self._identity()["recipe"], workload=c.workload,
                placements=tuple((n, self.placements[n].placement)
                                 for n in self.observed_layers),
            )
        counter_bytes = 0 if self.counter is None else (
            self.counter.query.storage_bytes + self.counter.query.health_device_bytes
            + self.counter.query.history_bytes)
        health_host_bytes = 0 if self.counter is None else (
            self.counter.query.health_host_bytes + self.counter.query.history_bytes)
        device_cache = sum(m.hbm_total_bytes for m in memories.values()) + counter_bytes
        if device_cache > c.expert_device_bytes:
            raise ValueError(
                "prepared cache workspace, resident rows and counters exceed the model device envelope"
            )
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        host_free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        host_new = sum(m.backing_bytes + m.update_host_bytes for m in memories.values()) + health_host_bytes
        if host_new + c.host_safety_bytes > host_free:
            raise ValueError(
                "canonical mapped backing exceeds available physical host memory"
            )
        self.memory = ResidencyServingMemory(
            device_capacity=total,
            host_capacity=c.host_bytes,
            resident_experts=sum(m.resident_bytes for m in memories.values()),
            backing_experts=sum(m.backing_bytes for m in memories.values()),
            dense_model=allocated,
            kv=c.kv_reserved_bytes,
            graphs=c.graph_reserved_bytes,
            workspace=sum(m.workspace_bytes for m in memories.values()),
            metadata=sum(m.metadata_bytes for m in memories.values()) + counter_bytes,
            host_staging=sum(m.update_host_bytes for m in memories.values()) + health_host_bytes,
            host_sources=sum(m.source_bytes for m in memories.values()),
            device_safety=c.device_safety_bytes,
            host_safety=c.host_safety_bytes,
            other_device=max(0, total - free - allocated),
        )

    def requests(self):
        if self.memory is None:
            raise RuntimeError("model-wide cache admission must precede preparation")
        requests = []
        for name, plan in self.plans.items():
            q = plan.query

            def make_call(state, q=q):
                a = (
                    torch.randn(
                        (q.max_tokens, q.hidden),
                        device=self.device,
                        dtype=torch.bfloat16,
                    )
                    * 0.125
                )
                ids = (
                    torch.arange(
                        q.max_tokens * q.top_k, device=self.device, dtype=torch.int32
                    )
                    % q.experts
                ).reshape(q.max_tokens, q.top_k)
                weights = torch.full(ids.shape, 1 / q.top_k, device=self.device)
                binding = state.bind(a=a, topk_ids=ids, topk_weights=weights)
                return PreparedCall(
                    run=binding.run,
                    output=binding.output,
                    owners=(state,),
                    close=state.close,
                )

            requests.append(
                plan.request(name="expert_cache:" + name, prepare_call=make_call)
            )
        if self.counter is not None:

            def counter_call(state):
                ids = torch.zeros(
                    (1, self.capacity.top_k), device=self.device, dtype=torch.int32
                )
                bindings = [
                    state.bind(layer=name, phase="decode", topk_ids=ids)
                    for name in self.observed_layers
                ]

                def run():
                    state.set_token_limit(self.capacity.max_tokens)
                    for binding in bindings:
                        binding.run()
                    if state.health is not None:
                        state.health.rebase(("preparation",))
                    if state.history is not None:
                        state.history.rebase(("preparation",))
                        for _ in range(state.history.depth):
                            state.history.checkpoint(("preparation",))

                return PreparedCall(run=run, owners=(state, ids, *bindings))

            requests.append(
                self.counter.request(
                    name="expert_cache:counters", prepare_call=counter_call
                )
            )
        return tuple(requests)

    def attach(self):
        """Install callbacks after preparation, before any serving graph capture."""
        if self.counter is None or self._counters is not None:
            return
        counters = moe.routing_profile_state(self.counter)
        self._counters = counters
        counters.reset(quiescent=True)
        counters.set_token_limit(0)
        if counters.history is not None:
            # Preparation's copy/event priming precedes producer ownership.
            counters.history.epoch = counters.history.stream = None
        if counters.health is not None:
            counters.health.bind_maps({n: self.plans[n].prepared.state.mapping
                                      for n in self.observed_layers})
            if self.anchor is not None:
                counters.health.bind_anchor(self.anchor)
        if self.config.mode == "adaptive":
            bindings = {
                name: bind_sm120_epoch_layer(
                    plan,
                    RoutingObservationSpec(
                        layer=name,
                        experts=plan.query.experts,
                        phase="decode",
                        max_top_k=plan.query.top_k,
                    ),
                )
                for name, plan in self.plans.items()
                if plan.query.max_pairs
            }
            self.runtime = ResidencyEpochRuntime(
                rank=0,
                owner_rank=0,
                bindings=bindings,
                snapshot_counters=counters.snapshot,
                checkpoint_id=self.checkpoint_id,
                initial_profile_id=self.profile_id,
                memory=self.memory,
                anchor=self.anchor,
            )

    def prepare_observation(self, decode_tokens):
        """Publish phase-qualified live rows with one prepared metadata launch."""
        if self._counters is not None:
            self._counters.set_token_limit(decode_tokens)

    def observe(self, name, ids):
        if name in self.observed_layers:
            moe.bind_routing_profile(
                self.counter, layer=name, phase="decode", topk_ids=ids
            ).run()

    def save_profile(self, *, quiescent=False):
        if self.config.mode != "profile":
            raise ValueError("only explicit profile mode may write a learned artifact")
        snapshot = moe.routing_profile_state(self.counter).snapshot(quiescent=quiescent)
        rows = {row.layer: row for row in snapshot.layers}
        if any(not sum(row.counts) for row in rows.values()):
            raise ValueError("every layer needs real decode observations before saving")
        placements = {
            name: profile_from_counts(
                counts=rows[name].counts,
                hot_count=len(p.hbm_expert_ids),
                layer=name,
                model_fingerprint=self.checkpoint_id,
                workload=self.config.workload,
                provenance="vLLM explicit decode calibration",
                phase="decode",
            ).to_dict()
            for name, p in self.placements.items()
        }
        payload = {
            "identity": self._identity(),
            "placements": placements,
            "termination": "explicit_calibration_boundary",
            "converged": False,
        }
        payload["hash"] = digest(payload)
        path = Path(self.config.profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "path": str(path),
            "hash": payload["hash"],
            "snapshot": asdict(snapshot),
        }
