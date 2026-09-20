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


def test_history_capacity_is_independent_and_fully_admitted(tmp_path):
    q = RoutingProfileQuery(layers=(("a", 4), ("b", 7)), max_tokens=128, max_top_k=8)
    assert q.history_bytes == 0
    for depth in (1, 2, 4, 8, 16):
        h = replace(q, history_depth=depth)
        assert h.history_bytes == depth * q.storage_bytes
        assert h.health_device_bytes == h.health_host_bytes == 0
        assert plan_routing_profile(h).prepared is None
    for depth in (-1, True, 1.5):
        with pytest.raises(ValueError):
            replace(q, history_depth=depth)
    with pytest.raises(ValueError):
        replace(q, history_depth=4, phases=("verify",))
    with pytest.raises(ValueError):
        replace(q, history_depth=4, rank=1, tp_size=2)
    with pytest.raises(ValueError):
        replace(config(tmp_path), mode="static", history_depth=4)


def test_history_rejects_decreasing_unused_tail_and_reset_epoch():
    from b12x.moe.residency.contracts import validate_routing_progress
    from tests.moe.test_residency_epoch import snapshot

    a = snapshot({"a": (0, 0, 8, 0), "b": (0, 0, 0, 9)}, 1)
    b = snapshot({"a": (0, 0, 12, 0), "b": (0, 0, 0, 10)}, 2)
    c = snapshot({"a": (0, 0, 10, 0), "b": (0, 0, 0, 11)}, 3)
    validate_routing_progress((a, a, b))
    with pytest.raises(ValueError, match="decreased"):
        validate_routing_progress((a, b, c))
    with pytest.raises(ValueError, match="epoch"):
        validate_routing_progress((a, replace(b, epoch=2)))
    with pytest.raises(ValueError, match="layer"):
        validate_routing_progress((a, replace(b, layers=b.layers[:1])))


def test_history_churn_distinguishes_completed_and_censored_lifetimes():
    from benchmarks.moe.summarize_expert_history import churn

    rows = []
    for token, pair, hits in (
        (32, (2, 0), ()),
        (64, (3, 2), ((2, 0),)),
        (96, (2, 3), ((3, 5),)),
    ):
        rows.append(
            dict(
                kind="maintenance",
                output_tokens_so_far=token,
                receipt=dict(
                    worker=dict(
                        baseline=False,
                        copy_bytes=100,
                        layers={"a": dict(pairs=(pair,), evicted_hits=hits)},
                    )
                ),
            )
        )
    result = churn(rows, tokens=128, initial_hot={"a": (0, 1)})
    assert result["promotions"] == 3 and result["re_promotions"] == 1
    assert result["zero_hit_completed_lifetimes"] == 1
    assert result["completed_promoted_lifetimes"] == 2
    assert result["active_right_censored"] == 1
    assert result["final_membership_change"] == {"a": 0.5}
    assert result["copy_bytes"] == 300
    assert not result["demand_observation_available"]


@pytest.mark.parametrize("record_history", [False, True])
def test_probe_does_not_invoke_policy_and_cancellation_consumes_pending_slot(
    record_history,
):
    class Engine:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.operations = []

        async def collective_rpc(self, method, *, args):
            assert method == "b12x_residency_health"
            operation = args[0]
            assert args == (
                ("start", True)
                if record_history and operation == "start"
                else (operation,)
            )
            self.operations.append(operation)
            if operation == "start":
                self.started.set()
                return [{"submitted": True}]
            await self.release.wait()
            return [summary([(8, 2)])]

    async def run():
        e = Engine()
        p = VllmResidencyHealth(e, record_history=record_history)
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
        result = await VllmResidencyHealth(e, record_history=record_history).probe()
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

        def _validate(self, g):
            assert g == self.generation

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
    with pytest.raises(RuntimeError, match="history was not prepared"):
        worker.b12x_residency_health("start", True)

    class History:
        recorded = []

        def rebase(self, g):
            self.generation = g

        def checkpoint(self, g):
            assert g == self.generation
            self.recorded.append(g)
            return {"checkpoint": len(self.recorded)}

    history = History()
    history.rebase(worker._b12x_health_generation())
    worker.model_runner.b12x_expert_cache._counters.history = history
    assert worker.b12x_residency_checkpoint()["checkpoint"] == 1
    assert worker.b12x_residency_health("start", True)["history"]["checkpoint"] == 2
    worker.b12x_residency_health("poll")
    assert before == [
        (c._scores, c._window, c._baseline, c._slots)
        for c in m.coordinator.controllers.values()
    ]
    assert e.reads == [1]
    e.counts = snapshot({n: (0, 0, 20, 0) for n in ("a", "b")}, 1)
    assert worker.b12x_residency_maintenance(m.config)["selected_pairs"] == 2
    assert health.generation == worker._b12x_health_generation()
    assert history.generation == health.generation
    assert all(c._window == 1 for c in m.coordinator.controllers.values())
    worker.model_runner.b12x_expert_cache = None
    with pytest.raises(RuntimeError, match="not prepared"):
        worker.b12x_residency_health("start")


def test_anchor_comparison_breadth_identity_and_opt_out(tmp_path):
    from dataclasses import FrozenInstanceError
    from b12x.moe.residency import ResidencyAnchor, RoutingAnchorThresholds, ExpertPlacement, compare_anchor
    placement = ExpertPlacement(total_experts=4, resident_expert_ids=(0, 1), backing_expert_ids=(2, 3))
    anchor = ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint', recipe='recipe',
                             workload='general', placements=(('a', placement), ('b', placement)))
    hot = anchor.resident_ids
    counts = {'a': (90, 0, 1, 0), 'b': (90, 0, 1, 0)}
    gate = RoutingAnchorThresholds(advantage_fraction=.02, minimum_layer_fraction=.75)
    assert not gate.assess(compare_anchor(counts, hot, hot))['anchor_better']
    changed = {'a': (1, 2), 'b': (1, 2)}
    assert gate.assess(compare_anchor(counts, changed, hot))['anchor_better']
    assert not gate.assess(compare_anchor(counts, hot, changed))['anchor_better']
    assert not gate.assess(compare_anchor(counts, {**hot, 'a': (1, 2)}, hot))['anchor_better']
    assert not gate.assess(compare_anchor({n: (0,)*4 for n in hot}, changed, hot))['anchor_better']
    with pytest.raises(FrozenInstanceError):
        anchor.profile_id = 'b'*64
    hot['a'] = (2, 3)
    assert anchor.resident_ids['a'] == (0, 1)
    with pytest.raises(ValueError, match='capacities'):
        compare_anchor(counts, {'a': (1,), 'b': (1, 2)}, anchor.resident_ids)
    q = RoutingProfileQuery(layers=(('a', 4), ('b', 7)), max_tokens=8, max_top_k=4)
    h = replace(q, health_summary=True)
    a = replace(h, anchor_summary=True)
    assert a.health_device_bytes - h.health_device_bytes == 11 + 2*16
    assert a.health_host_bytes - h.health_host_bytes == 2*8
    with pytest.raises(ValueError, match='explicit health'):
        replace(q, anchor_summary=True)
    with pytest.raises(ValueError):
        replace(config(tmp_path), mode='static', anchor_health=True)


def test_anchor_analysis_rejects_corrupt_profile_and_foreign_identity(tmp_path):
    from benchmarks.moe.analyze_residency_anchor import load_anchor
    from b12x.integration.vllm.expert_cache import digest
    from b12x.moe.fused_moe.residency import profile_from_counts
    import json
    plan = profile_from_counts(counts=(10, 5, 1, 0), hot_count=2, layer='a',
        model_fingerprint='checkpoint', workload='general', provenance='test', phase='decode')
    artifact = dict(identity={'checkpoint': 'checkpoint'}, placements={'a': plan.to_dict()})
    artifact['hash'] = digest(artifact)
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(artifact))
    prepared = dict(profile=artifact['hash'], checkpoint='checkpoint',
                    layers={'a': dict(experts=4, resident=2)})
    assert load_anchor(path, prepared)['a'] == plan
    for changes in ({'profile': 'foreign'}, {'checkpoint': 'foreign'},
                    {'layers': {'a': dict(experts=5, resident=2)}}):
        with pytest.raises(ValueError):
            load_anchor(path, {**prepared, **changes})
    artifact['identity']['checkpoint'] = 'changed'
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='hash'):
        load_anchor(path, prepared)
