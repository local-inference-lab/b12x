"""Read-only health decisions and asynchronous result ownership."""

import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest
from b12x.moe.residency.health import RoutingHealthThresholds
from b12x.moe.fused_moe.routing_profile import RoutingProfileQuery, plan_routing_profile
from b12x.integration.vllm.residency_health import VllmResidencyHealth
from tests.moe.test_expert_cache_serving import config


def summary(rows):
    return dict(
        layers={
            str(i): dict(selections=n, cold_selections=c)
            for i, (n, c) in enumerate(rows)
        },
        selections=sum(n for n, c in rows),
        cold_selections=sum(c for n, c in rows),
    )


def test_global_and_breadth_health_reject_single_layer_noise_without_mutation():
    gate = RoutingHealthThresholds(cold_fraction=0.15, minimum_layer_fraction=0.5)
    a = summary([(100, 90)] + [(100, 1)] * 7)
    before = deepcopy(a)
    assert gate.assess(a)["health"] == "healthy"
    assert a == before
    assert gate.assess(summary([(100, 30)] * 8))["health"] == "pressure"
    assert gate.assess(summary([(0, 0)] * 8))["health"] == "unobserved"
    assert gate.assess(summary([(100, 90), (0, 0)]))["health"] == "unobserved"
    assert gate.assess(summary([(100, 0)] * 8))["health"] == "healthy"
    with pytest.raises(ValueError):
        gate.assess(summary([(5, 6)]))
    a["selections"] += 1
    with pytest.raises(ValueError, match="totals"):
        gate.assess(a)


def test_health_declaration_is_explicit_and_memory_is_model_wide(tmp_path):
    q = RoutingProfileQuery(layers=(("a", 4), ("b", 7)), max_tokens=128, max_top_k=8)
    assert not q.health_summary and q.health_device_bytes == q.health_host_bytes == 0
    h = replace(q, health_summary=True)
    assert h.health_device_bytes == 11 * 16 + 2 * 80 and h.health_host_bytes == 96
    assert plan_routing_profile(h).prepared is None
    with pytest.raises(ValueError):
        replace(h, phases=("decode", "verify"))
    with pytest.raises(ValueError):
        replace(h, rank=1, tp_size=2)
    with pytest.raises(ValueError):
        replace(config(tmp_path), mode="static", health_probes=True)


def test_probe_does_not_invoke_policy_and_cancellation_consumes_pending_slot():
    class Engine:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.operations = []

        async def collective_rpc(self, method, *, args):
            assert method == "b12x_residency_health"
            (operation,) = args
            self.operations.append(operation)
            if operation == "start":
                self.started.set()
                return [{"submitted": True}]
            await self.release.wait()
            return [summary([(8, 2)])]

    async def run():
        e = Engine()
        p = VllmResidencyHealth(e)
        task = asyncio.create_task(p.probe())
        await e.started.wait()
        task.cancel()
        await asyncio.sleep(0.005)
        assert not task.done()
        e.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert e.operations == ["start", "poll"]
        with pytest.raises(RuntimeError, match="uncertain"):
            await p.probe()
        e = Engine()
        e.release.set()
        result = await VllmResidencyHealth(e).probe()
        assert result["summary"]["cold_selections"] == 2
        assert e.operations == ["start", "poll"]

    asyncio.run(run())


def test_retrospective_payback_rejects_transient_leads_and_charges_prior_cost():
    from benchmarks.moe.summarize_expert_health import economics

    assert (
        economics([10, 20, 30, 40], [9, 21, 28, 35])["sustained_break_even_tokens"] == 3
    )
    assert (
        economics([10, 20, 30, 40], [9, 21, 28, 35], prior_penalty_ms=6)[
            "sustained_break_even_tokens"
        ]
        is None
    )
    assert economics([10, 20], [9, 18])["final_gain_ms"] == 2
    with pytest.raises(ValueError):
        economics([1], [1, 2])


def test_worker_health_reads_leave_policy_and_generation_untouched(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from tests.moe.test_vllm_residency_epoch import Engine, local_maintenance, snapshot

    e = Engine(ranks=1, canonical=True)
    m = local_maintenance(e)
    m.run()
    worker = e.workers[0]
    runtime = worker.model_runner.b12x_residency_runtime
    runtime._local_maintenance = m

    class Health:
        pending = False
        generation = None

        def rebase(self, g):
            self.generation = g

        def start(self, g):
            assert g == self.generation
            self.pending = True

        def poll(self, g):
            assert g == self.generation
            self.pending = False
            return summary([(8, 6), (8, 6)])

    health = Health()
    worker.model_runner.b12x_expert_cache = SimpleNamespace(
        _counters=SimpleNamespace(health=health)
    )
    worker.model_runner.main_stream = None
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    health.rebase(worker._b12x_health_generation())
    before = [
        (c._scores, c._window, c._baseline, c._slots)
        for c in m.coordinator.controllers.values()
    ]
    for _ in range(3):
        assert worker.b12x_residency_health("start")["submitted"]
        with pytest.raises(RuntimeError, match="pending health"):
            worker.b12x_residency_maintenance(m.config)
        assert worker.b12x_residency_health("poll")["cold_selections"] == 12
    assert before == [
        (c._scores, c._window, c._baseline, c._slots)
        for c in m.coordinator.controllers.values()
    ]
    assert e.reads == [1]
    e.counts = snapshot({n: (0, 0, 20, 0) for n in ("a", "b")}, 1)
    assert worker.b12x_residency_maintenance(m.config)["selected_pairs"] == 2
    assert health.generation == worker._b12x_health_generation()
    assert all(c._window == 1 for c in m.coordinator.controllers.values())
    worker.model_runner.b12x_expert_cache = None
    with pytest.raises(RuntimeError, match="not prepared"):
        worker.b12x_residency_health("start")
