"""Arbitrary-N transaction gates with real host canonical fills, no GPU claim."""

import asyncio
from dataclasses import asdict, replace

import pytest
import torch

from b12x.integration.vllm.tp_residency import TensorParallelResidencyMaintenance
from tests.moe.test_prepared_expert_cache import updates
from tests.moe.test_vllm_residency_epoch import Engine
from tests.moe.test_residency_epoch import config, snapshot


def group(world):
    engine = Engine(ranks=world, canonical=True)
    engine.paused = True
    states = []
    for rank, worker in enumerate(engine.workers):
        runtime = worker.model_runner.b12x_residency_runtime
        runtime.verify_rank_counters = True
        runtime.snapshot_counters = lambda quiescent, rank=rank: replace(engine.counts, rank=rank)
        bindings, local = {}, {}
        for name, binding in runtime.bindings.items():
            state = updates()
            local[name] = state
            bindings[name] = replace(binding, snapshot=state.snapshot, apply=state.apply,
                stage=state.stage, publish=state.publish_staged, rollback=state.rollback_staged,
                finish=state.finish_staged)
        runtime.bindings = bindings
        states.append(local)
    control = TensorParallelResidencyMaintenance(dict(
        layers={n: asdict(config(scoring="decayed_lfu")) for n in ("a", "b")},
        budget={"max_pairs": 2, "max_copy_bytes": 264}, cold_fraction_threshold=.15,
    ), world)
    events = []

    def rpc(method, *, args, timeout):
        assert timeout > 0
        events.append(method)
        result = []
        for rank in reversed(range(world)):
            if engine.fail == (method, rank):
                raise RuntimeError("missing rank acknowledgement")
            result.append(getattr(engine.workers[rank], method)(*args))
        return result

    return engine, control, states, rpc, events


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_arbitrary_rank_order_publishes_one_map_with_per_rank_budget(world):
    e, c, states, rpc, events = group(world)
    assert c.run(rpc)["baseline"]
    e.counts = snapshot({"a": (0, 0, 10, 0), "b": (0, 0, 0, 12)}, 1)
    receipt = c.run(rpc)
    assert receipt["selected_pairs"] == 2
    assert receipt["max_rank_copy_bytes"] == 264
    assert receipt["copy_bytes"] == 264 * world
    assert events.index("b12x_residency_stage") < events.index("b12x_residency_publish")
    for local in states:
        for name, candidate in (("a", 2), ("b", 3)):
            state = local[name]
            state.require_executable()
            assert state.snapshot().generation == 1
            assert state.snapshot().expert_map[candidate][0] == 0
            slot = state.snapshot().expert_map[candidate][1]
            for field, resident in state.resident.items():
                assert torch.equal(resident[slot], state.canonical[field][candidate])


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("stage", ["prepare", "stage", "publish", "acknowledge_staged", "finish"])
def test_missing_arbitrary_rank_never_infers_distributed_success(world, stage):
    e, c, states, rpc, events = group(world)
    c.run(rpc)
    e.counts = snapshot({n: (0, 0, 10, 0) for n in ("a", "b")}, 1)
    e.fail = ("b12x_residency_" + stage, world // 2)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        c.run(rpc)
    assert c.failed and e.paused
    if stage in ("prepare", "stage"):
        for local in states:
            for state in local.values():
                assert state.snapshot().generation == 0
                state.require_executable()
                for name, resident in state.resident.items():
                    assert torch.equal(resident, state.canonical[name][:2])
    with pytest.raises(RuntimeError, match="reload"):
        c.run(rpc)


@pytest.mark.parametrize("fault", ["profile", "generation", "routing", "cancel", "rollback"])
def test_disagreement_cancellation_and_failed_rollback_remain_closed(fault):
    e, c, states, rpc, events = group(4)
    c.run(rpc)
    e.counts = snapshot({n: (0, 0, 10, 0) for n in ("a", "b")}, 1)
    runtime = e.workers[2].model_runner.b12x_residency_runtime
    if fault == "profile":
        runtime.initial_profile_id = "foreign"
    elif fault == "generation":
        states[2]["a"]._generation += 1
    elif fault == "routing":
        runtime.snapshot_counters = lambda quiescent: snapshot({"a": (1, 0, 10, 0), "b": (0, 0, 10, 0)}, 1)
    def inject(method, *, args, timeout):
        if fault == "cancel" and method.endswith("stage"):
            raise asyncio.CancelledError()
        if fault == "rollback" and method.endswith(("stage", "rollback")):
            raise RuntimeError("rank unavailable")
        return rpc(method, args=args, timeout=timeout)
    with pytest.raises((ValueError, RuntimeError, asyncio.CancelledError)):
        c.run(inject)
    assert c.failed and e.paused
    assert "b12x_residency_publish" not in events[events.index("b12x_residency_finish") + 1:]


def test_staged_payload_cannot_execute_or_publish_twice():
    state = updates()
    before = state.snapshot()
    state.stage(((2, 0),), expected=before, quiescent=True)
    assert state.snapshot() == before
    with pytest.raises(RuntimeError, match="pending"):
        state.require_executable()
    state.publish_staged()
    with pytest.raises(RuntimeError, match="staged"):
        state.publish_staged()
    state.finish_staged()
    state.require_executable()


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_delayed_participant_blocks_publication(world):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    engine, control, states, rpc, events = group(world)
    control.run(rpc)
    engine.counts = snapshot({n: (0, 0, 10, 0) for n in ("a", "b")}, 1)
    reached, release = Event(), Event()
    worker = engine.workers[world // 2]
    original = worker.b12x_residency_stage

    def delayed(*args):
        reached.set()
        if not release.wait(5):
            raise TimeoutError("test participant was not released")
        return original(*args)

    worker.b12x_residency_stage = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(control.run, rpc)
        try:
            assert reached.wait(5)
            assert not future.done()
            assert all(s.snapshot().generation == 0 for local in states for s in local.values())
        finally:
            release.set()
        assert future.result(timeout=5)["status"] == "complete"
