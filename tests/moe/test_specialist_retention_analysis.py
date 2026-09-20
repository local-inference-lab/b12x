"""Read-only routing diagnostics must preserve the prepared policy state."""

from dataclasses import replace

import pytest
import torch

from b12x.testing.vllm_routing_snapshot import RoutingSnapshotWorker
from tests.moe.test_vllm_residency_epoch import Engine


def test_boundary_read_requires_quiescence_and_preserves_runtime(monkeypatch):
    engine = Engine(ranks=1, canonical=True)
    engine.paused = True
    worker = RoutingSnapshotWorker()
    worker.model_runner = engine.workers[0].model_runner
    runtime = worker._b12x_epoch_runtime()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    before = dict(vars(runtime))
    with pytest.raises(ValueError, match="pause"):
        worker.diagnostic_routing_snapshot()
    result = worker.diagnostic_routing_snapshot(quiescent=True)
    assert result["snapshot"]["epoch"] == engine.counts.epoch
    assert engine.reads == [1] and vars(runtime) == before
    assert all(s["generation"] == 0 for s in result["slots"].values())
    runtime._stage = "prepared"
    with pytest.raises(RuntimeError, match="idle"):
        worker.diagnostic_routing_snapshot(quiescent=True)
    runtime._stage = "idle"
    def corrupt():
        layer = engine.layers[0]["a"]
        layer.slots = replace(layer.slots, generation=1)
    monkeypatch.setattr(torch.cuda, "synchronize", corrupt)
    with pytest.raises(RuntimeError, match="changed"):
        worker.diagnostic_routing_snapshot(quiescent=True)


def test_boundary_demand_detects_reset_and_keeps_duplicate_counts():
    from benchmarks.moe.analyze_specialist_retention import boundary_demand
    from tests.moe.test_residency_epoch import snapshot
    from dataclasses import asdict
    rows = [dict(kind='request', index=0, workload='general'),
            dict(kind='routing_boundary', next_request=0, time_ns=0,
                 receipt=[dict(snapshot=asdict(snapshot()))]),
            dict(kind='routing_boundary', next_request=1, time_ns=10,
                 receipt=[dict(snapshot=asdict(snapshot({'a': (0, 5, 0, 0), 'b': (1, 1, 0, 0)}, 1)))])]
    phases, groups = boundary_demand(rows)
    assert phases['general']['a'] == [0, 5, 0, 0] and len(groups) == 1
    rows[-1]['receipt'][0]['snapshot']['epoch'] += 1
    with pytest.raises(ValueError, match='reset'):
        boundary_demand(rows)


def test_offline_oracle_and_protection_only_choose_eligible_victims():
    from benchmarks.moe.analyze_specialist_retention import choose_victim
    scores, future = (0, 2, 5), (100, 0, 40)
    assert choose_victim((0, 1, 2), scores, future) == 0
    assert choose_victim((0, 1, 2), scores, future, (0,)) == 1
    assert choose_victim((0, 1, 2), scores, future, oracle=True) == 1
    assert choose_victim((0,), scores, future, (0,)) is None
