"""Worker/RPC contract tests; these do not measure a serving engine."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.integration.vllm.residency_epoch import (
    ResidencyEpochRuntime, ResidencyEpochWorkerExtension, ResidencyLayerBinding,
    ResidencyServingMemory, VllmResidencyEpochs,
)
from b12x.moe import residency as r
from tests.moe.test_residency_epoch import config, make_slots, snapshot, spec


def memory():
    return ResidencyServingMemory(device_capacity=1000, host_capacity=1000,
        resident_experts=200, backing_experts=400, dense_model=100, kv=100,
        workspace=100, graphs=50, metadata=10, host_staging=100,
        device_safety=100, host_safety=100)


class Layer:
    def __init__(self, name, rank, canonical):
        self.slots = make_slots(name, rank=rank, canonical=canonical)
        self.spec = spec(canonical)
        self.addresses = (1000, 2000, 3000)
        self.fail = False
        self.live = True

    def validate(self):
        if not self.live:
            raise RuntimeError("preparation released")

    def apply(self, pairs, *, expected, quiescent):
        assert quiescent and expected == self.slots
        if self.fail:
            raise r.ResidencyUpdateError("injected backend failure", resumable=True)
        self.slots = replace(self.slots, generation=self.slots.generation + 1,
            expert_map=r.updated_slot_map(self.slots, pairs, backing_mode=self.spec.backing_mode))


class Engine:
    def __init__(self, ranks=2, canonical=False):
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
            tensor_parallel_size=ranks, data_parallel_size=1, pipeline_parallel_size=1,
            enable_expert_parallel=False, decode_context_parallel_size=1, prefill_context_parallel_size=1))
        self.paused, self.events, self.workers, self.layers = False, [], [], []
        self.fail = None
        self.counts = snapshot()
        self.reads = [0]*ranks
        for rank in range(ranks):
            layers = {name: Layer(name, rank, canonical) for name in ("a", "b")}
            bindings = {name: ResidencyLayerBinding(observations=r.RoutingObservationSpec(
                layer=name, experts=4, phase="decode", max_top_k=2, rank=rank, owner_rank=0),
                exchange=layer.spec, max_pairs=2, snapshot=lambda layer=layer: layer.slots,
                apply=layer.apply, pointers=lambda layer=layer: layer.addresses,
                validate=layer.validate) for name, layer in layers.items()}
            def read(*, quiescent, rank=rank):
                assert self.paused and quiescent
                self.reads[rank] += 1
                return self.counts
            runtime = ResidencyEpochRuntime(rank=rank, owner_rank=0, bindings=bindings,
                snapshot_counters=read, checkpoint_id="checkpoint-sha", initial_profile_id="learned-profile-sha",
                memory=memory())
            worker = ResidencyEpochWorkerExtension()
            worker.model_runner = SimpleNamespace(b12x_residency_runtime=runtime)
            self.workers.append(worker)
            self.layers.append(layers)

    async def is_paused(self):
        return self.paused

    async def pause_generation(self, *, mode, clear_cache):
        assert mode == "keep" and clear_cache is False
        self.events.append("pause")
        self.paused = True

    async def resume_generation(self):
        self.events.append("resume")
        self.paused = False

    async def collective_rpc(self, method, *, timeout, args):
        assert self.paused
        self.events.append(method)
        replies = []
        for rank, worker in enumerate(self.workers):
            if self.fail == (method, rank):
                raise RuntimeError("injected rank failure")
            replies.append(getattr(worker, method)(*args))
        return replies


def driver(engine, **changes):
    return VllmResidencyEpochs(engine, configs={name: config(scoring="decayed_lfu") for name in ("a", "b")},
        budget=r.ResidencyEpochBudget(max_pairs=2, max_copy_bytes=528), tp_size=len(engine.workers), **changes)


@pytest.mark.parametrize("canonical", [False, True])
def test_model_epoch_pauses_once_snapshots_once_and_publishes_all_ranks(canonical):
    async def run():
        e = Engine(canonical=canonical)
        d = driver(e)
        assert (await d.run())["baseline"]
        e.counts = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 10)}, 1)
        receipt = await d.run()
        assert receipt["decision"]["selected_pairs"] == 2
        assert receipt["decision"]["copy_bytes"] == 528
        assert e.reads == [2, 0]
        assert e.events.count("pause") == e.events.count("resume") == 2
        assert all(layers["a"].slots.expert_map[2][0] == 0 for layers in e.layers)
        assert all(layers["b"].slots.expert_map[3][0] == 0 for layers in e.layers)
        assert receipt["status"] == "resumed" and not e.paused
        assert set(receipt["stages_ns"]) == {"pause_and_drain", "begin", "policy", "prepare", "apply", "acknowledge", "resume"}
        assert receipt["total_wall_ns"] >= sum(receipt["stages_ns"].values())
        return e, d
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["begin", "prepare", "apply", "acknowledge"])
def test_rank_failure_never_resumes_and_requires_reload(stage):
    async def run():
        e, d = Engine(), None
        d = driver(e)
        await d.run()
        e.counts = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 10)}, 1)
        e.fail = ("b12x_residency_" + stage, 1)
        with pytest.raises(RuntimeError, match="rank failure"):
            await d.run()
        assert e.paused and e.events.count("resume") == 1
        assert d.last_receipt["status"] == "reload_required"
        with pytest.raises(RuntimeError, match="reload"):
            await d.run()
    asyncio.run(run())


def test_partial_layer_failure_is_not_mistaken_for_model_rollback():
    async def run():
        e = Engine()
        d = driver(e)
        await d.run()
        e.counts = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 10)}, 1)
        e.layers[0]["b"].fail = True
        with pytest.raises(r.ResidencyUpdateError):
            await d.run()
        assert e.layers[0]["a"].slots.generation == 1
        assert e.layers[0]["b"].slots.generation == 0
        assert e.paused and d.failed
        # Only reloading all rank owners permits a fresh engine lifecycle.
        fresh = Engine()
        await driver(fresh).run()
        assert not fresh.paused
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["generation", "pointer", "release", "profile"])
def test_stale_maps_pointers_released_owners_and_mismatched_prior_fail_closed(mutation):
    async def run():
        e = Engine()
        d = driver(e)
        await d.run()
        layer = e.layers[1]["b"]
        if mutation == "generation":
            layer.slots = replace(layer.slots, generation=1)
        elif mutation == "pointer":
            layer.addresses = (4, 5, 6)
        elif mutation == "release":
            layer.live = False
        else:
            e.workers[1].model_runner.b12x_residency_runtime.initial_profile_id = "different"
        with pytest.raises((ValueError, RuntimeError)):
            await d.run()
        assert e.paused and all(x.slots.generation == 0 for x in e.layers[0].values())
    asyncio.run(run())


def test_disabled_performs_no_engine_or_worker_operations():
    e = Engine()
    assert asyncio.run(driver(e, enabled=False).run()) == {"enabled": False}
    assert not e.events and e.reads == [0, 0]


def local_maintenance(engine, threshold=0.2, pairs=2):
    from dataclasses import asdict
    from b12x.integration.vllm.residency_maintenance import LocalResidencyMaintenance
    engine.paused = True
    runtime = engine.workers[0].model_runner.b12x_residency_runtime
    return LocalResidencyMaintenance(runtime, {
        "session": "test", "layers": {n: asdict(config(scoring="decayed_lfu")) for n in ("a", "b")},
        "budget": {"max_pairs": pairs, "max_copy_bytes": 528},
        "cold_fraction_threshold": threshold,
    })


def test_local_health_gate_preserves_stable_generation_then_tracks_transition():
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e)
    assert m.run()["baseline"]
    for window in range(1, 4):
        e.counts = snapshot({n: (20*window, 0, window, 0) for n in ("a", "b")}, window)
        receipt = m.run()
        assert receipt["health"] == "healthy" and receipt["selected_pairs"] == 0
        assert all(l.slots.generation == 0 for l in e.layers[0].values())
    e.counts = snapshot({n: (60, 0, 43, 0) for n in ("a", "b")}, 4)
    receipt = m.run()
    assert receipt["health"] == "pressure" and receipt["selected_pairs"] == 2
    assert all(l.slots.expert_map[2][0] == 0 for l in e.layers[0].values())
    assert all(l.addresses == (1000, 2000, 3000) for l in e.layers[0].values())
    assert e.reads == [5]


@pytest.mark.parametrize("failure", ["stale", "partial", "pointer"])
def test_local_failure_poisoning_requires_reload(failure):
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e)
    m.run()
    e.counts = snapshot({n: (0, 0, 10, 0) for n in ("a", "b")}, 1)
    if failure == "stale":
        e.layers[0]["b"].slots = replace(e.layers[0]["b"].slots, generation=1)
    elif failure == "partial":
        e.layers[0]["b"].fail = True
    else:
        e.layers[0]["b"].addresses = (1, 2, 3)
    with pytest.raises((ValueError, RuntimeError, r.ResidencyUpdateError)):
        m.run()
    with pytest.raises(RuntimeError, match="reload"):
        m.run()
    if failure == "partial":
        assert e.layers[0]["a"].slots.generation == 1


def test_local_zero_budget_and_mixed_controller_rejection():
    e = Engine(ranks=1)
    m = local_maintenance(e, threshold=None, pairs=0)
    m.run()
    e.counts = snapshot({n: (0, 0, 10, 0) for n in ("a", "b")}, 1)
    receipt = m.run()
    assert receipt["proposed_pairs"] == 2 and receipt["selected_pairs"] == 0
    assert receipt["transaction_ns"] == {}
    with pytest.raises(RuntimeError, match="controller changed"):
        m.runtime.begin("foreign")


def test_policy_diagnostics_retain_scores_without_changing_decisions():
    engines = [Engine(ranks=1, canonical=True) for _ in range(2)]
    controls = [local_maintenance(e, threshold=None) for e in engines]
    controls[1].diagnostics = True
    for m in controls:
        m.run()
    for e in engines:
        e.counts = snapshot({n: (0, 0, 12, 0) for n in ("a", "b")}, 1)
    plain, recorded = [m.run() for m in controls]
    assert "policy_observations" not in plain
    assert plain["layers"] == recorded["layers"]
    assert plain["selected_pairs"] == recorded["selected_pairs"]
    observation, = recorded["policy_observations"]
    assert observation["deferred"] is False
    for row in observation["layers"].values():
        assert row["counts"] == row["scores"] == (0, 0, 12, 0)
        assert row["window"] == 1
        assert row["selected"] and row["candidates"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_maintenance_replays_deferred_cuts_then_publishes_once(monkeypatch, corrupt):
    from contextlib import nullcontext
    import torch
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e, threshold=None)
    m.diagnostics = True

    class History:
        generation = None
        reads = 0

        def rebase(self, generation):
            self.generation = generation

        def read(self, generation, *, quiescent):
            assert quiescent and self.generation == generation
            self.reads += 1
            frames = [snapshot({n: counts for n in ("a", "b")}, calls)
                      for counts, calls in (((0, 0, 80, 0), 1),
                                           ((0, 0, 80, 8), 2),
                                           ((0, 0, 80, 16), 3))]
            return tuple(frames), {"checkpoints": 3, "coalesced_checkpoints": 0}

    class Counters:
        history = History()
        health = None

        def snapshot(self, *, quiescent):
            assert quiescent and e.paused
            return e.counts

    counters = Counters()
    worker = e.workers[0]
    worker.model_runner.main_stream = None
    worker.model_runner.b12x_expert_cache = SimpleNamespace(_counters=counters)
    m.runtime.snapshot_counters = counters.snapshot
    m.runtime._local_maintenance = m
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    assert worker.b12x_residency_maintenance(m.config)["baseline"]
    e.counts = snapshot({n: (0, 0, 80, 10 if corrupt else 18) for n in ("a", "b")}, 4)
    if corrupt:
        with pytest.raises(ValueError, match="decreased"):
            worker.b12x_residency_maintenance(m.config)
        assert m.runtime._failed
        assert all(layer.slots.generation == 0 for layer in e.layers[0].values())
        return
    receipt = worker.b12x_residency_maintenance(m.config)
    assert receipt["history"]["replayed_windows"] == 2
    assert receipt["selections"] == receipt["cold_selections"] == 196
    assert counters.history.reads == 1
    assert [x["deferred"] for x in receipt["policy_observations"]] == [True, True, False]
    assert all(c._window == 3 and c._scores == (0, 0, 20, 14)
               for c in m.coordinator.controllers.values())
    assert all(layer.slots.generation == 1 and layer.slots.expert_map[3][0] == 0
               for layer in e.layers[0].values())
    assert counters.history.generation == worker._b12x_health_generation()


def test_history_does_not_redefine_the_full_interval_pressure_gate():
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e, threshold=0.15)
    names = ("a", "b")

    class History:
        def read(self, generation, *, quiescent):
            assert quiescent
            return tuple(snapshot({n: counts for n in names}, calls)
                         for calls, counts in enumerate(((100, 0, 0, 0),
                                                         (100, 0, 1, 0),
                                                         (100, 0, 1, 1)), 1)), {}

    class Counters:
        history = History()

        def snapshot(self, *, quiescent):
            assert quiescent and e.paused
            return e.counts

    m.runtime.snapshot_counters = Counters().snapshot
    assert m.run()["baseline"]
    e.counts = snapshot({n: (100, 0, 1, 9) for n in names}, 4)
    receipt = m.run()
    # The final window is entirely cold, but the complete interval is healthy.
    assert receipt["selections"] == 220 and receipt["cold_selections"] == 20
    assert receipt["health"] == "healthy" and receipt["selected_pairs"] == 0
    assert all(c._window == 3 and c._scores[3] == 9
               for c in m.coordinator.controllers.values())
    assert all(layer.slots.generation == 0 for layer in e.layers[0].values())


def test_maintenance_driver_serializes_checks_and_poisoned_outcomes():
    from b12x.integration.vllm.residency_maintenance import VllmResidencyMaintenance

    async def run():
        active, calls = 0, 0

        async def maintain(config):
            nonlocal active, calls
            active += 1
            assert active == 1
            await asyncio.sleep(0)
            active -= 1
            calls += 1
            if calls == 3:
                raise RuntimeError("unknown worker outcome")
            return {"worker": {"baseline": calls == 1}}

        driver = VllmResidencyMaintenance(
            SimpleNamespace(residency_maintenance=maintain),
            configs={n: config() for n in ("a", "b")},
            budget=r.ResidencyEpochBudget(max_pairs=2, max_copy_bytes=528),
        )
        results = await asyncio.gather(driver.run(), driver.run())
        assert calls == 2 and results[1]["lock_wait_ns"] > 0
        with pytest.raises(RuntimeError, match="unknown worker"):
            await driver.run()
        with pytest.raises(RuntimeError, match="reload"):
            await driver.run()
        assert calls == 3

    asyncio.run(run())


def test_cannot_steal_external_pause_or_merge_dp_lanes():
    e = Engine()
    e.paused = True
    with pytest.raises(RuntimeError, match="already paused"):
        asyncio.run(driver(e).run())
    with pytest.raises(ValueError, match="DP lane"):
        driver(e, dp_size=2)
    assert e.paused and not e.events


def test_missing_loader_backend_registration_is_explicit():
    worker = ResidencyEpochWorkerExtension()
    worker.model_runner = SimpleNamespace()
    with pytest.raises(RuntimeError, match="CPU-source loader/backend"):
        worker.b12x_residency_begin("token")


def test_memory_admission_counts_full_backing_staging_and_model_reservations():
    assert memory().device_bytes == 660 and memory().host_bytes == 600
    with pytest.raises(ValueError, match="capacity"):
        replace(memory(), graphs=1000)
    with pytest.raises(ValueError, match="capacity"):
        replace(memory(), backing_experts=900)


def test_worker_rejects_stale_token_and_duplicate_apply():
    e = Engine(ranks=1)
    e.paused = True
    runtime = e.workers[0].model_runner.b12x_residency_runtime
    report = runtime.begin("epoch")
    commands = {"0": {name: {"expected": row["slots"], "pairs": ()} for name, row in report["layers"].items()}}
    with pytest.raises(RuntimeError, match="stale"):
        runtime.prepare("foreign", commands)
    runtime.prepare("epoch", commands)
    runtime.apply("epoch")
    with pytest.raises(RuntimeError, match="stage"):
        runtime.apply("epoch")
    runtime.acknowledge("epoch")


def test_cancellation_after_partial_apply_keeps_lane_paused():
    async def run():
        e = Engine()
        d = driver(e)
        await d.run()
        e.counts = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 10)}, 1)
        original = e.collective_rpc
        async def cancel(method, **kwargs):
            result = await original(method, **kwargs)
            if method == "b12x_residency_apply":
                raise asyncio.CancelledError()
            return result
        e.collective_rpc = cancel
        with pytest.raises(asyncio.CancelledError):
            await d.run()
        assert e.paused and d.failed and e.events.count("resume") == 1
    asyncio.run(run())


def test_wire_roundtrip_does_not_depend_on_python_tuple_serialization():
    import json
    async def run():
        e = Engine()
        original = e.collective_rpc
        async def serialize(method, *, timeout, args):
            return json.loads(json.dumps(await original(method, timeout=timeout,
                args=tuple(json.loads(json.dumps(args))))))
        e.collective_rpc = serialize
        d = driver(e)
        await d.run()
        e.counts = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 10)}, 1)
        assert (await d.run())["decision"]["selected_pairs"] == 2
    asyncio.run(run())


def test_all_prepared_capacities_checked_before_any_write():
    async def run():
        e = Engine()
        d = driver(e)
        runtime = e.workers[1].model_runner.b12x_residency_runtime
        runtime.bindings = {name: replace(binding, max_pairs=1) for name, binding in runtime.bindings.items()}
        with pytest.raises(ValueError):
            await d.run()
        assert e.paused and "b12x_residency_apply" not in e.events
    asyncio.run(run())


@pytest.mark.parametrize("name,value", [("tensor_parallel_size", 4), ("data_parallel_size", 2),
    ("pipeline_parallel_size", 2), ("enable_expert_parallel", True),
    ("decode_context_parallel_size", 2), ("prefill_context_parallel_size", 2)])
def test_actual_engine_parallel_configuration_is_checked(name, value):
    e = Engine()
    setattr(e.vllm_config.parallel_config, name, value)
    with pytest.raises(ValueError, match="engine configuration"):
        driver(e)
    assert not e.events


def test_anchor_recovery_uses_existing_transaction_and_allows_later_adaptation(monkeypatch):
    from dataclasses import asdict
    from b12x.integration.vllm.residency_maintenance import LocalResidencyMaintenance
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e, threshold=None)
    anchor = r.ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint-sha',
        recipe='recipe', workload='general', placements=tuple((n, r.ExpertPlacement(
            total_experts=4, resident_expert_ids=(0, 1), backing_expert_ids=(2, 3))) for n in ('a', 'b')))
    m.runtime.anchor = anchor
    settings = {**m.config, 'anchor_thresholds': asdict(r.RoutingAnchorThresholds(
        advantage_fraction=.02, minimum_layer_fraction=.75))}
    m = LocalResidencyMaintenance(m.runtime, settings)
    assert m.run()['baseline']
    e.counts = snapshot({n: (0, 0, 12, 0) for n in ('a', 'b')}, 1)
    with monkeypatch.context() as patch:
        def forbidden_comparison(*args, **kwargs):
            raise AssertionError("normal adaptation must not compute anchor regret")
        patch.setattr("b12x.integration.vllm.residency_maintenance.compare_anchor",
                      forbidden_comparison)
        first = m.run()
    assert first['movement_mode'] == 'adapt' and first['selected_pairs'] == 2
    assert 'anchor' not in first
    e.counts = snapshot({n: (24, 0, 12, 0) for n in ('a', 'b')}, 2)
    returned = m.run(movement_mode="recenter")
    assert returned['movement_mode'] == 'recenter' and returned['selected_pairs'] == 2
    assert all(v['pairs'] == ((0, 2),) for v in returned['layers'].values())
    e.counts = snapshot({n: (24, 0, 60, 0) for n in ('a', 'b')}, 3)
    again = m.run()
    assert again['movement_mode'] == 'adapt' and again['selected_pairs'] == 2
    assert m.runtime.anchor is anchor and anchor.resident_ids == {'a': (0, 1), 'b': (0, 1)}


def test_runtime_rejects_anchor_identity_and_initial_geometry_mismatch():
    e = Engine(ranks=1, canonical=True)
    old = e.workers[0].model_runner.b12x_residency_runtime
    anchor = r.ResidencyAnchor(profile_id='a'*64, checkpoint=old.checkpoint_id,
        recipe='recipe', workload='general', placements=tuple((n, r.ExpertPlacement(
            total_experts=4, resident_expert_ids=(0, 1), backing_expert_ids=(2, 3))) for n in ('a', 'b')))
    kwargs = dict(rank=0, owner_rank=0, bindings=old.bindings, snapshot_counters=old.snapshot_counters,
                  checkpoint_id=old.checkpoint_id, initial_profile_id='a'*64, memory=old.memory)
    assert ResidencyEpochRuntime(**kwargs, anchor=anchor).anchor is anchor
    for bad in (replace(anchor, checkpoint='foreign'), replace(anchor, profile_id='b'*64),
                replace(anchor, placements=tuple((n, r.ExpertPlacement(total_experts=4,
                    resident_expert_ids=(0, 2), backing_expert_ids=(1, 3))) for n in ('a', 'b')))):
        with pytest.raises(ValueError, match='anchor'):
            ResidencyEpochRuntime(**kwargs, anchor=bad)


def test_declined_anchor_recheck_never_falls_through_to_normal_adaptation():
    from dataclasses import asdict
    from b12x.integration.vllm.residency_maintenance import LocalResidencyMaintenance
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e, threshold=None)
    m.runtime.anchor = r.ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint-sha',
        recipe='recipe', workload='general', placements=tuple((n, r.ExpertPlacement(
            total_experts=4, resident_expert_ids=(0, 1), backing_expert_ids=(2, 3))) for n in ('a', 'b')))
    m = LocalResidencyMaintenance(m.runtime, {**m.config, 'anchor_thresholds': asdict(
        r.RoutingAnchorThresholds(advantage_fraction=.02, minimum_layer_fraction=.75))})
    m.run()
    e.counts = snapshot({n: (0, 0, 12, 0) for n in ('a', 'b')}, 1)
    # A delayed client signal no longer supported by the full observation may
    # update history, but cannot become an unrelated adaptive transaction.
    receipt = m.run(movement_mode='recenter')
    assert receipt['movement_mode'] == 'recenter' and receipt['selected_pairs'] == 0
    assert all(layer.slots.generation == 0 for layer in e.layers[0].values())


@pytest.mark.parametrize("pairs,limit,selected", [(0, 528, 0), (4, 0, 0),
                                                 (1, 528, 1), (4, 528, 2)])
def test_independent_recovery_budget_preserves_later_normal_adaptation(pairs, limit, selected):
    from dataclasses import asdict
    from b12x.integration.vllm.residency_maintenance import LocalResidencyMaintenance
    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e, threshold=None, pairs=2)
    anchor = r.ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint-sha',
        recipe='recipe', workload='general', placements=tuple((n, r.ExpertPlacement(
            total_experts=4, resident_expert_ids=(0, 1), backing_expert_ids=(2, 3))) for n in ('a', 'b')))
    m.runtime.anchor = anchor
    m = LocalResidencyMaintenance(m.runtime, {**m.config,
        'recenter_budget': dict(max_pairs=pairs, max_copy_bytes=limit),
        'anchor_thresholds': asdict(r.RoutingAnchorThresholds(
            advantage_fraction=.02, minimum_layer_fraction=.75))})
    m.run()
    e.counts = snapshot({n: (0, 0, 12, 0) for n in ('a', 'b')}, 1)
    assert m.run()['selected_pairs'] == 2
    before = {n: b.snapshot() for n, b in m.runtime.bindings.items()}
    e.counts = snapshot({n: (24, 0, 12, 0) for n in ('a', 'b')}, 2)
    recovered = m.run(movement_mode='recenter')
    assert recovered['selected_pairs'] == selected
    assert recovered['proposal_backlog']['budget'] == dict(max_pairs=pairs, max_copy_bytes=limit)
    if not selected:
        assert before == {n: b.snapshot() for n, b in m.runtime.bindings.items()}
    e.counts = snapshot({n: (24, 0, 12, 60) for n in ('a', 'b')}, 3)
    again = m.run()
    assert again['selected_pairs'] == 2
    assert again['proposal_backlog']['budget']['max_pairs'] == 2
    assert m.runtime.anchor is anchor
