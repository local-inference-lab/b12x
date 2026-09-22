"""Explicit vLLM control-plane adapter for prepared model-wide residency.

Uses the supported worker extension and AsyncLLM pause/RPC interfaces. It does
not install a weight loader, select a backend, or add nodes to serving graphs.
A backend must register prepared bindings before this adapter can run.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Callable
from types import MappingProxyType
from uuid import uuid4

from b12x.moe.residency import (
    LayerRoutingCounts, ResidencyCacheController, ResidencyEpochCoordinator,
    ResidencyExchangeSpec, ResidencySlotSnapshot, RoutingObservationSpec,
    RoutingSnapshot, updated_slot_map,
)
from b12x.moe.residency.contracts import _integer, _text


def _slots(value):
    return ResidencySlotSnapshot(**{**value, "expert_map": tuple(tuple(x) for x in value["expert_map"])})


def _snapshot(value):
    return RoutingSnapshot(epoch=value["epoch"], rank=value["rank"], layers=tuple(
        LayerRoutingCounts(**row) for row in value["layers"]))


@dataclass(frozen=True, kw_only=True)
class ResidencyServingMemory:
    """Per-rank model admission, including complete backing and private scratch.

    Capacity envelopes and reservations must use the same accounting basis.
    Pinned staging is additional to canonical backing. Graph/KV/model bytes are
    reserved once, not independently in every layer. No shared-arena saving is
    assumed. The loader supplies values before materializing expert storage.
    """
    device_capacity: int
    host_capacity: int
    resident_experts: int
    backing_experts: int
    dense_model: int
    kv: int
    workspace: int
    graphs: int
    metadata: int
    host_staging: int
    device_safety: int
    host_safety: int
    other_device: int = 0
    other_host: int = 0
    host_sources: int = 0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _integer(name, getattr(self, name))
        if self.device_bytes > self.device_capacity or self.host_bytes > self.host_capacity:
            raise ValueError("model-wide residency memory reservations exceed capacity")

    @property
    def device_bytes(self):
        return sum(getattr(self, n) for n in ("resident_experts", "dense_model", "kv",
            "workspace", "graphs", "metadata", "device_safety", "other_device"))

    @property
    def host_bytes(self):
        return (self.backing_experts + self.host_staging + self.host_sources
                + self.host_safety + self.other_host)


@dataclass(frozen=True, kw_only=True)
class ResidencyLayerBinding:
    """Prepared backend callbacks retained by one worker for the model lifetime.

    apply(pairs, expected=..., quiescent=True) commits one complete layer
    generation. pointers() includes every captured slab/map/workspace address.
    validate() checks preparation owners remain alive. No callback is invoked
    from graph replay. Backends retain source identity and rollback guarantees.
    """
    observations: RoutingObservationSpec
    exchange: ResidencyExchangeSpec
    max_pairs: int
    snapshot: Callable
    apply: Callable
    pointers: Callable
    validate: Callable
    timings: Callable | None = None
    stage: Callable | None = None
    publish: Callable | None = None
    rollback: Callable | None = None
    finish: Callable | None = None


class ResidencyEpochRuntime:
    """Worker-side transaction participant; registration is explicit and opt-in."""
    def __init__(self, *, rank, owner_rank, bindings, snapshot_counters,
                 checkpoint_id, initial_profile_id, memory: ResidencyServingMemory, anchor=None,
                 verify_rank_counters=False):
        _integer("rank", rank)
        _integer("owner_rank", owner_rank)
        _text("checkpoint_id", checkpoint_id)
        _text("initial_profile_id", initial_profile_id)
        if not isinstance(memory, ResidencyServingMemory):
            raise TypeError("worker requires admitted model-wide memory accounting")
        self.bindings = MappingProxyType(dict(bindings))
        if not self.bindings:
            raise ValueError("worker has no prepared residency layers")
        for name, binding in self.bindings.items():
            if (not isinstance(binding, ResidencyLayerBinding)
                    or binding.observations.layer != name
                    or binding.observations.rank != rank
                    or binding.observations.owner_rank != owner_rank):
                raise ValueError("residency binding differs from worker/layer identity")
            _integer("prepared exchange capacity", binding.max_pairs, 1)
        self.rank, self.owner_rank = rank, owner_rank
        self.snapshot_counters = snapshot_counters
        self.verify_rank_counters = verify_rank_counters
        self.checkpoint_id, self.initial_profile_id, self.memory = checkpoint_id, initial_profile_id, memory
        if anchor is not None:
            from b12x.moe.residency.anchor import ResidencyAnchor
            if (not isinstance(anchor, ResidencyAnchor) or anchor.checkpoint != checkpoint_id
                    or anchor.profile_id != initial_profile_id
                    or set(anchor.resident_ids) != set(bindings)):
                raise ValueError("anchor identity differs from prepared cache")
            for name, placement in anchor.placements:
                slots = bindings[name].snapshot()
                hot = {e for e, (t, _) in enumerate(slots.expert_map) if t == 0}
                if (placement.total_experts != len(slots.expert_map)
                        or hot != set(placement.resident_expert_ids)):
                    raise ValueError("initial cache differs from learned anchor")
        self.anchor = anchor
        self._pointers = {name: tuple(b.pointers()) for name, b in self.bindings.items()}
        self._token, self._command, self._before = None, None, None
        self._stage, self._failed = "idle", False
        self._controller_kind = None

    def _validate(self):
        if self._failed:
            raise RuntimeError("residency worker failed; reload every rank before resuming")
        for name, binding in self.bindings.items():
            binding.validate()
            if tuple(binding.pointers()) != self._pointers[name]:
                raise RuntimeError("captured residency addresses changed")

    def _require(self, token, stage):
        self._validate()
        if token != self._token or self._stage != stage:
            raise RuntimeError("stale residency epoch token or invalid transaction stage")

    def begin(self, token):
        """Called only after the engine's completed pause/drain boundary."""
        self._validate()
        if self._controller_kind not in (None, "rpc"):
            raise RuntimeError("residency controller changed; reload the lane")
        self._controller_kind = "rpc"
        _text("epoch token", token)
        if self._stage != "idle":
            raise RuntimeError("worker already has a pending residency epoch")
        self._token, self._stage = token, "observed"
        before = {name: b.snapshot() for name, b in sorted(self.bindings.items())}
        for name, slots in before.items():
            updated_slot_map(slots, (), backing_mode=self.bindings[name].exchange.backing_mode)
        start = perf_counter_ns()
        snapshot = self.snapshot_counters(quiescent=True) if self.rank == self.owner_rank or self.verify_rank_counters else None
        snapshot_ns = perf_counter_ns() - start
        self._before = before
        return {"rank": self.rank, "checkpoint_id": self.checkpoint_id,
            "initial_profile_id": self.initial_profile_id, "memory": asdict(self.memory),
            "snapshot_ns": snapshot_ns, "snapshot": asdict(snapshot) if snapshot else None,
            "layers": {name: {"slots": asdict(before[name]), "exchange": asdict(b.exchange),
                "observations": asdict(b.observations), "max_pairs": b.max_pairs}
                for name, b in sorted(self.bindings.items())}}

    def prepare(self, token, commands):
        """Validate every local layer before any rank is permitted to overwrite."""
        self._require(token, "observed")
        command = commands[str(self.rank)]
        if set(command) != set(self.bindings):
            raise ValueError("model epoch command differs from registered layers")
        for name, item in command.items():
            if len(item["pairs"]) > self.bindings[name].max_pairs:
                raise ValueError("epoch exceeds prepared layer exchange capacity")
            if _slots(item["expected"]) != self._before[name] or self.bindings[name].snapshot() != self._before[name]:
                raise ValueError("stale layer generation at epoch prepare")
            updated_slot_map(self._before[name], item["pairs"], backing_mode=self.bindings[name].exchange.backing_mode)
        # Retain plain immutable values, independent of the RPC argument lifetime.
        self._command = {name: tuple(tuple(pair) for pair in item["pairs"]) for name, item in command.items()}
        self._stage = "prepared"
        return {"rank": self.rank, "ready": True}

    def stage(self, token):
        self._require(token, "prepared")
        self._stage = "staging"
        for name, binding in sorted(self.bindings.items()):
            if not all((binding.stage, binding.publish, binding.rollback, binding.finish)):
                raise RuntimeError("backend lacks reversible distributed payload staging")
            if self._command[name]:
                binding.stage(self._command[name], expected=self._before[name], quiescent=True)
            if binding.snapshot() != self._before[name]:
                raise RuntimeError("staging published a generation before all ranks were ready")
        self._stage = "staged"
        return {"rank": self.rank, "staged": True}

    def rollback(self, token):
        if token != self._token or self._stage not in ("observed", "prepared", "staging", "staged"):
            self._failed = True
            raise RuntimeError("cannot roll back uncertain publication")
        try:
            for name, binding in sorted(self.bindings.items()):
                if binding.rollback is not None:
                    binding.rollback()
                if binding.snapshot() != self._before[name]:
                    raise RuntimeError("rollback did not restore the complete generation")
        except BaseException:
            self._failed = True
            raise
        self._stage, self._token, self._command, self._before = "idle", None, None, None
        return {"rank": self.rank, "rolled_back": True}

    def publish(self, token):
        self._require(token, "staged")
        self._stage = "publishing"
        try:
            for name, binding in sorted(self.bindings.items()):
                if self._command[name]:
                    binding.publish()
            self._stage = "applied"
            return {"rank": self.rank, "layers": {
                n: asdict(b.snapshot()) for n, b in self.bindings.items()}}
        except BaseException:
            self._failed = True
            raise

    def apply(self, token):
        self._require(token, "prepared")
        timings, after = {}, {}
        try:
            for name, binding in sorted(self.bindings.items()):
                pairs, before = self._command[name], self._before[name]
                if binding.snapshot() != before:
                    raise ValueError("layer changed after epoch prepare")
                start = perf_counter_ns()
                if pairs:
                    binding.apply(pairs, expected=before, quiescent=True)
                timings[name] = perf_counter_ns() - start
                after[name] = binding.snapshot()
                expected = ResidencySlotSnapshot(preparation_id=before.preparation_id,
                    generation=before.generation + bool(pairs), healthy=True,
                    expert_map=updated_slot_map(before, pairs, backing_mode=binding.exchange.backing_mode))
                if after[name] != expected:
                    raise RuntimeError("backend did not publish the complete selected layer generation")
            self._validate()
            self._stage = "applied"
        except BaseException:
            self._failed = True
            raise
        return {"rank": self.rank, "layers": {n: asdict(s) for n, s in after.items()},
                "transaction_ns": timings}

    def acknowledge(self, token, *, staged=False):
        self._require(token, "applied")
        # No backend may mutate after apply and before the distributed success gate.
        for name, binding in self.bindings.items():
            before, pairs = self._before[name], self._command[name]
            expected = ResidencySlotSnapshot(preparation_id=before.preparation_id,
                generation=before.generation + bool(pairs), healthy=True,
                expert_map=updated_slot_map(before, pairs, backing_mode=binding.exchange.backing_mode))
            if binding.snapshot() != expected:
                raise RuntimeError("layer changed before distributed acknowledgement")
        if staged:
            self._stage = "acknowledged"
            return {"rank": self.rank, "acknowledged": True}
        self._stage, self._token, self._command, self._before = "idle", None, None, None
        return {"rank": self.rank, "acknowledged": True}

    def finish(self, token):
        self._require(token, "acknowledged")
        for name, binding in self.bindings.items():
            if self._command[name]:
                binding.finish()
        self._stage, self._token, self._command, self._before = "idle", None, None, None
        return {"rank": self.rank, "finished": True}


class ResidencyEpochWorkerExtension:
    """Use with vLLM's supported --worker-extension-cls registration.

    The loader/preparation integration installs model_runner.b12x_residency_runtime
    only after source ownership, budget admission and graph bindings are valid.
    Missing registration fails closed; this extension cannot make a regular
    all-resident model into an expert cache.
    """
    def _b12x_epoch_runtime(self):
        runtime = getattr(self.model_runner, "b12x_residency_runtime", None)
        if not isinstance(runtime, ResidencyEpochRuntime):
            raise RuntimeError("model has no prepared residency runtime; CPU-source loader/backend integration is required")
        return runtime

    def b12x_phase_timing_start(self, capacity=4096):
        from b12x.testing.phase_timing import PhaseTiming
        if getattr(self.model_runner, "b12x_phase_timing", None) is not None:
            raise RuntimeError("phase timing is already active")
        self.model_runner.b12x_phase_timing = PhaseTiming(capacity)
        return {"capacity": capacity}

    def b12x_phase_timing_finish(self):
        timing = getattr(self.model_runner, "b12x_phase_timing", None)
        if timing is None:
            raise RuntimeError("phase timing was not started")
        result = timing.finish()
        self.model_runner.b12x_phase_timing = None
        return result

    def b12x_residency_maintenance(self, config):
        """Only the engine's completed single-rank drain may invoke this method."""
        from .residency_maintenance import LocalResidencyMaintenance
        config = dict(config)
        movement_mode = config.pop("movement_mode", "adapt")
        runtime = self._b12x_epoch_runtime()
        cache = getattr(self.model_runner, "b12x_expert_cache", None)
        health = getattr(getattr(cache, "_counters", None), "health", None)
        if health is not None and health.pending:
            raise RuntimeError("consume pending health result before maintenance")
        maintenance = getattr(runtime, "_local_maintenance", None)
        if maintenance is None:
            maintenance = LocalResidencyMaintenance(runtime, config)
            runtime._local_maintenance = maintenance
        elif maintenance.config != config:
            raise ValueError("maintenance session/config changed; reload the lane")
        result = maintenance.run(movement_mode=movement_mode)
        cache = getattr(self.model_runner, "b12x_expert_cache", None)
        health = getattr(getattr(cache, "_counters", None), "health", None)
        if health is not None:
            # Mutation completes at the existing maintenance boundary. Rebase
            # before another graph can count selections against the new map.
            health.rebase(self._b12x_health_generation())
        history = getattr(getattr(cache, "_counters", None), "history", None)
        if history is not None:
            import torch
            with torch.cuda.stream(self.model_runner.main_stream):
                history.rebase(self._b12x_health_generation())
        return result

    def _b12x_health_generation(self):
        runtime = self._b12x_epoch_runtime()
        runtime._validate()
        if runtime._stage != "idle":
            raise RuntimeError("residency transaction is pending")
        return tuple((n, b.snapshot().preparation_id, b.snapshot().generation)
                     for n, b in sorted(runtime.bindings.items()))

    def b12x_residency_health(self, operation, record_history=False):
        """Serialized single-worker utility; event polling never drains the device.

        The maintained executor submits this on the same actor as model calls.
        Only its producer stream may write the prepared routing counters. A
        pending read must be consumed before the client requests maintenance.
        """
        if type(record_history) is not bool or (record_history and operation != "start"):
            raise ValueError("history recording is a boolean start-only option")
        import torch
        cache = getattr(self.model_runner, "b12x_expert_cache", None)
        if cache is not None and cache.tp_rank != 0:
            self._b12x_health_generation()
            return {"rank": cache.tp_rank, "observer_only": True}
        health = getattr(getattr(cache, "_counters", None), "health", None)
        if health is None:
            raise RuntimeError("cache health probes were not prepared")
        started = perf_counter_ns()
        generation = self._b12x_health_generation()
        with torch.cuda.stream(self.model_runner.main_stream):
            if operation == "start":
                checkpoint = None
                if record_history:
                    history = getattr(cache._counters, "history", None)
                    if history is None:
                        raise RuntimeError("routing history was not prepared")
                    health._validate(generation)
                    checkpoint = history.checkpoint(generation)
                health.start(generation)
                return {"submitted": True, "history": checkpoint,
                        "worker_wall_ns": perf_counter_ns()-started}
            if operation == "poll":
                result = health.poll(generation)
                if result is not None:
                    result["worker_poll_ns"] = perf_counter_ns()-started
                return result
            raise ValueError("unknown residency health operation")

    def b12x_residency_checkpoint(self):
        """Optional observation cut, independent of health or policy cadence."""
        import torch
        cache = getattr(self.model_runner, "b12x_expert_cache", None)
        history = getattr(getattr(cache, "_counters", None), "history", None)
        if history is None:
            raise RuntimeError("routing history was not prepared")
        generation = self._b12x_health_generation()
        with torch.cuda.stream(self.model_runner.main_stream):
            return history.checkpoint(generation)

    def b12x_residency_begin(self, token):
        return self._b12x_epoch_runtime().begin(token)

    def b12x_residency_prepare(self, token, commands):
        return self._b12x_epoch_runtime().prepare(token, commands)

    def b12x_residency_apply(self, token):
        return self._b12x_epoch_runtime().apply(token)

    def b12x_residency_acknowledge(self, token):
        return self._b12x_epoch_runtime().acknowledge(token)

    def b12x_residency_stage(self, token):
        return self._b12x_epoch_runtime().stage(token)

    def b12x_residency_rollback(self, token):
        return self._b12x_epoch_runtime().rollback(token)

    def b12x_residency_publish(self, token):
        return self._b12x_epoch_runtime().publish(token)

    def b12x_residency_acknowledge_staged(self, token):
        return self._b12x_epoch_runtime().acknowledge(token, staged=True)

    def b12x_residency_finish(self, token):
        result = self._b12x_epoch_runtime().finish(token)
        cache = getattr(self.model_runner, "b12x_expert_cache", None)
        health = getattr(getattr(cache, "_counters", None), "health", None)
        if health is not None:
            health.rebase(self._b12x_health_generation())
        return result

    def b12x_lifecycle_resources(self, stage):
        """Explicit diagnostic; no allocation or observation node enters replay."""
        import os
        from b12x.testing.lifecycle import record_worker_resources

        path = os.environ.get("B12X_LIFECYCLE_OUTPUT")
        if not path:
            raise RuntimeError("lifecycle recording requires an explicit output path")
        return record_worker_resources(self, stage, path)

    def b12x_expert_cache_status(self):
        """Inspect ownership without adding observation work to static serving."""
        model = getattr(self.model_runner, "b12x_expert_cache", None)
        if model is None:
            raise RuntimeError("model has no prepared canonical expert cache")
        manager = getattr(self.model_runner, "cudagraph_manager", None)
        graphs = getattr(manager, "graphs", {})
        return {"mode": model.config.mode, "rank": model.tp_rank, "world_size": model.tp_size,
            "checkpoint": model.checkpoint_id,
            "profile": model.profile_id, "memory": asdict(model.memory),
            "load_device_peak_bytes": model.load_device_peak_bytes,
            "graphs": {str(descriptor): id(graph) for descriptor, graph in graphs.items()},
            "layers": {name: {"resident": plan.query.resident, "experts": plan.query.experts,
                "max_pairs": plan.query.max_pairs,
                "storage": asdict(model.sources[name].storage),
                "pointers": plan.prepared.state.pointers(),
                "generation": plan.prepared.state.updates.snapshot().generation if plan.prepared.state.updates else 0}
                for name,plan in model.plans.items()}}

    def b12x_expert_cache_start_profile(self, *, quiescent=False):
        """Exclude startup work after the caller completes engine pause/drain."""
        model = getattr(self.model_runner, "b12x_expert_cache", None)
        if model is None:
            raise RuntimeError("model has no prepared canonical expert cache")
        return model.start_profile(quiescent=quiescent)

    def b12x_expert_cache_save_profile(self, *, quiescent=False):
        """Caller must first complete the engine pause/drain protocol."""
        model = getattr(self.model_runner, "b12x_expert_cache", None)
        if model is None:
            raise RuntimeError("model has no prepared canonical expert cache")
        return model.save_profile(quiescent=quiescent)


def bind_sm120_epoch_layer(plan, observations):
    """Bind the prepared canonical cache; this installs no graph instrumentation."""
    from b12x.preparation import require_prepared
    state = require_prepared(plan, "moe.expert_cache")
    if state.updates is None:
        raise ValueError("model epochs require fill capacity declared before preparation")
    if (observations.experts != plan.query.experts
            or observations.max_top_k != plan.query.top_k):
        raise ValueError("epoch observation geometry differs from the prepared cache")

    def validate():
        if require_prepared(plan, "moe.expert_cache") is not state:
            raise RuntimeError("cache was replaced; rebuild the model epoch runtime")
        state.validate()

    payload = sum(v[0].numel() for v in state.canonical.values())
    return ResidencyLayerBinding(observations=observations,
        exchange=ResidencyExchangeSpec(backend="sm120_w4a16_canonical",
            direct_backing_execution=True, fixed_address_quiescent_exchange=True,
            payload_copy_bytes_per_pair=payload, map_copy_bytes_per_transaction=2*plan.query.experts*8,
            backing_mode="canonical"), max_pairs=plan.query.max_pairs,
        snapshot=state.updates.snapshot, apply=state.updates.apply,
        stage=state.updates.stage, publish=state.updates.publish_staged,
        rollback=state.updates.rollback_staged, finish=state.updates.finish_staged,
        pointers=state.pointers, validate=validate,
        timings=lambda: dict(getattr(state.updates, "last_timings_ns", {})))


def bind_sm103_epoch_layer(plan, observations):
    """Retain an already prepared native MXFP4 HBM/Grace exchange backend.

    This function adds no compilation, migration or graph nodes. Preparation
    has already admitted source-native storage and optional exchange journals.
    It does not confer physical SM103 qualification.
    """
    from b12x.preparation import require_prepared
    from b12x.moe.fused_moe._residency_updates import exchange_expert_slots, residency_slot_snapshot
    from b12x.moe.fused_moe._residency_storage import tier_layout

    state = require_prepared(plan, "moe.expert_residency")
    if state.updates is None:
        raise ValueError("model epochs require exchange capacity declared before preparation")
    if (not isinstance(observations, RoutingObservationSpec)
            or observations.experts != plan.query.experts
            or observations.max_top_k > plan.query.max_top_k):
        raise ValueError("epoch observations differ from prepared expert geometry")

    def validate():
        if require_prepared(plan, "moe.expert_residency") is not state:
            raise RuntimeError("residency plan was replaced; rebuild the model epoch runtime")
        state.updates.require_healthy()

    def pointers():
        return (state.mapping.data_ptr(), state.slab.data_ptr(),
                *(tier.slab.data_ptr() for tier in state.tiers if tier is not None))

    # Layout alignment is per slab, while successful exchanges copy field rows.
    fields, _ = tier_layout(1, plan.query.hidden, plan.query.intermediate)
    payload = sum(math.prod(shape) for _, _, shape in fields)
    return ResidencyLayerBinding(observations=observations,
        exchange=ResidencyExchangeSpec(backend="sm103_mxfp4_exclusive",
            direct_backing_execution=True, fixed_address_quiescent_exchange=True,
            payload_copy_bytes_per_pair=4*payload,
            map_copy_bytes_per_transaction=2*plan.query.experts*8),
        max_pairs=plan.query.max_swap_pairs, snapshot=lambda: residency_slot_snapshot(plan),
        apply=lambda pairs, **kwargs: exchange_expert_slots(plan, pairs, **kwargs),
        pointers=pointers, validate=validate)


class VllmResidencyEpochs:
    """Engine-owned, explicitly invoked epochs using AsyncLLM's public methods.

    One instance exclusively owns pause/resume for one TP group (DP=1). Failure,
    timeout or cancellation after pause initiation leaves the lane paused. All
    ranks must reload before creating a replacement instance. No background
    scheduler, per-token host call, or automatic retry is installed.
    """
    def __init__(self, engine, *, configs, budget, tp_size=1, owner_rank=0,
                 dp_size=1, enabled=True, rpc_timeout=60.0):
        _integer("tp_size", tp_size, 1)
        _integer("owner_rank", owner_rank)
        if dp_size != 1 or owner_rank >= tp_size:
            raise ValueError("residency epochs require one DP lane and a valid replicated-TP owner")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(rpc_timeout) not in (int, float) or not math.isfinite(rpc_timeout) or rpc_timeout <= 0:
            raise ValueError("RPC timeout must be finite and positive")
        self.engine, self.configs, self.budget = engine, dict(configs), budget
        self.tp_size, self.owner_rank, self.enabled = tp_size, owner_rank, enabled
        self.rpc_timeout = rpc_timeout
        self.coordinator, self.failed, self.last_receipt = None, False, None
        self._lock = asyncio.Lock()
        if enabled:
            self._validate_parallel_config()

    def _validate_parallel_config(self):
        parallel = self.engine.vllm_config.parallel_config
        if (parallel.tensor_parallel_size != self.tp_size or parallel.data_parallel_size != 1
                or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel
                or parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1):
            raise ValueError("engine configuration must match one replicated TP group without DP/PP/EP/context parallelism")

    def _ranks(self, replies):
        if len(replies) != self.tp_size or {r["rank"] for r in replies} != set(range(self.tp_size)):
            raise RuntimeError("residency RPC did not return every TP rank exactly once")
        return {r["rank"]: r for r in replies}

    def _validate_replicas(self, ranks):
        owner = ranks[self.owner_rank]
        if set(owner["layers"]) != set(self.configs):
            raise ValueError("worker layers differ from model-wide policy declaration")
        for rank, value in ranks.items():
            if (value["checkpoint_id"], value["initial_profile_id"]) != (owner["checkpoint_id"], owner["initial_profile_id"]):
                raise ValueError("ranks disagree on checkpoint or initial static profile")
            if set(value["layers"]) != set(owner["layers"]):
                raise ValueError("ranks disagree on participating MoE layers")
            if getattr(self, "verify_rank_counters", False):
                if value["snapshot"] is None or {**value["snapshot"], "rank": self.owner_rank} != owner["snapshot"]:
                    raise ValueError("TP ranks disagree on canonical routing observations")
            elif rank != self.owner_rank and value["snapshot"] is not None:
                raise ValueError("replicated TP observations must be counted on one owner")
            for name, layer in value["layers"].items():
                reference = owner["layers"][name]
                slots, expected = _slots(layer["slots"]), _slots(reference["slots"])
                spec = ResidencyExchangeSpec(**layer["exchange"])
                updated_slot_map(slots, (), backing_mode=spec.backing_mode)
                observation = RoutingObservationSpec(**layer["observations"])
                if (observation.rank != rank or observation.owner_rank != self.owner_rank
                        or observation.layer != name):
                    raise ValueError("worker observation rank or layer identity differs")
                if ({**layer["observations"], "rank": self.owner_rank} != reference["observations"]
                        or layer["exchange"] != reference["exchange"]
                        or layer["max_pairs"] != reference["max_pairs"]
                        or (slots.generation, slots.expert_map) != (expected.generation, expected.expert_map)):
                    raise ValueError("replicated TP geometry, costs or residency maps differ")
                if self.configs[name].max_pairs > layer["max_pairs"]:
                    raise ValueError("cache policy exceeds prepared exchange capacity")
        if owner["snapshot"] is None:
            raise ValueError("authoritative routing snapshot is missing")

    async def run(self):
        """First call establishes the post-warmup baseline; later calls adapt.

        Each call is one policy observation window. Aggregate counters cannot
        reconstruct four-step windows from a longer pause interval. Window
        frequency remains an explicit engine experiment.
        """
        if not self.enabled:
            return {"enabled": False}
        requested = perf_counter_ns()
        async with self._lock:
            if self.failed:
                raise RuntimeError("residency lane requires coordinated reload")
            self._validate_parallel_config()
            if await self.engine.is_paused():
                raise RuntimeError("residency controller cannot acquire an already paused engine")
            token, started = uuid4().hex, perf_counter_ns()
            receipt = {"token": token, "status": "pausing", "stages_ns": {},
                       "acquire_and_ownership_check_ns": started - requested}
            self.last_receipt = receipt

            async def timed(name, operation):
                start = perf_counter_ns()
                try:
                    return await operation
                finally:
                    receipt["stages_ns"][name] = perf_counter_ns() - start

            async def rpc(stage, *args):
                return self._ranks(await timed(stage, self.engine.collective_rpc(
                    "b12x_residency_" + stage, timeout=self.rpc_timeout, args=args)))

            try:
                await timed("pause_and_drain", self.engine.pause_generation(mode="keep", clear_cache=False))
                receipt["engine_pause_stages_ns"] = getattr(self.engine, "last_pause_timings_ns", {})
                ranks = await rpc("begin", token)
                self._validate_replicas(ranks)
                owner = ranks[self.owner_rank]
                baseline = _snapshot(owner["snapshot"])
                slots = {name: _slots(v["slots"]) for name, v in owner["layers"].items()}
                start = perf_counter_ns()
                decision = None
                if self.coordinator is None:
                    controllers = {name: ResidencyCacheController(config=self.configs[name],
                        observations=RoutingObservationSpec(**v["observations"]),
                        exchange=ResidencyExchangeSpec(**v["exchange"]), slots=slots[name], baseline=baseline)
                        for name, v in owner["layers"].items()}
                    self.coordinator = ResidencyEpochCoordinator(controllers, budget=self.budget, replicas=self.tp_size)
                    pairs = {name: () for name in slots}
                else:
                    decision = self.coordinator.observe(baseline, slots=slots)
                    pairs = {v.layer: v.pairs for v in decision.layers}
                receipt["stages_ns"]["policy"] = perf_counter_ns() - start
                commands = {str(rank): {name: {"expected": value["slots"], "pairs": pairs[name]}
                            for name, value in report["layers"].items()} for rank, report in ranks.items()}
                ready = await rpc("prepare", token, commands)
                if not all(v.get("ready") is True for v in ready.values()):
                    raise RuntimeError("a residency rank rejected epoch prepare")
                applied = await rpc("apply", token)
                for rank, report in applied.items():
                    if set(report["layers"]) != set(slots):
                        raise RuntimeError("rank completion is missing a layer")
                    for name, value in report["layers"].items():
                        before = _slots(ranks[rank]["layers"][name]["slots"])
                        expected = ResidencySlotSnapshot(preparation_id=before.preparation_id,
                            generation=before.generation + bool(pairs[name]), healthy=True,
                            expert_map=updated_slot_map(before, pairs[name],
                                backing_mode=owner["layers"][name]["exchange"]["backing_mode"]))
                        if _slots(value) != expected:
                            raise RuntimeError("rank completion differs from selected residency generation")
                outcomes = self.coordinator.finish(decision,
                    slots={n: _slots(s) for n, s in applied[self.owner_rank]["layers"].items()}) if decision else {}
                acknowledged = await rpc("acknowledge", token)
                if not all(v.get("acknowledged") is True for v in acknowledged.values()):
                    raise RuntimeError("a residency rank did not acknowledge completion")
                receipt.update(status="resuming", baseline=decision is None,
                    decision=asdict(decision) if decision else None,
                    outcomes={n: asdict(v) for n, v in outcomes.items()}, workers=ranks, completed=applied)
                await timed("resume", self.engine.resume_generation())
                receipt["status"] = "resumed"
            except BaseException as error:
                self.failed = True
                if self.coordinator is not None:
                    self.coordinator.fail()
                receipt.update(status="reload_required", error=type(error).__name__ + ": " + str(error))
                # Never resume from a finally block: another rank/layer may have
                # committed even when this RPC raised or its caller cancelled.
                raise
            finally:
                completed = perf_counter_ns()
                receipt["pause_to_resume_wall_ns"] = completed - started
                receipt["total_wall_ns"] = completed - requested
                receipt["control_bookkeeping_ns"] = completed - started - sum(receipt["stages_ns"].values())
            return receipt
