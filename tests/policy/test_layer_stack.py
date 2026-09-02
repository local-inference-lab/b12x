"""Host-side contracts of the shared in-place layer-stack harness."""

from __future__ import annotations

from dataclasses import dataclass, field


from b12x.policy.generation.providers import layer_stack
from b12x.policy.generation.providers.layer_stack import (
    CONFIRM_ROUNDS,
    GenericLayerStack,
    _confirm_winner,
    _context_hidden,
    _ContextRace,
    _keep_race,
    layer_budget,
)
from b12x.policy.generation.sweep import SweepCandidate, SweepMeasurement


def _race(op_us: float, samples: list[float] | None = None) -> _ContextRace:
    queue = list(samples or [])

    def sample():
        value = queue.pop(0) if queue else op_us
        return value, value * 10.0

    return _ContextRace(
        op_us=op_us,
        stack_us=op_us * 10.0,
        finite=True,
        repetitions=3,
        sample=sample,
        graph=None,
    )


def _measurement(config: dict, latency: float) -> SweepMeasurement:
    return SweepMeasurement(
        candidate=SweepCandidate.create(config),
        latency_us=latency,
        correct=True,
        metrics={},
    )


def test_context_hidden_follows_the_narrower_side() -> None:
    assert _context_hidden(4096, 512) == 512
    assert _context_hidden(512, 4096) == 512
    assert _context_hidden(48, 2560) == 256
    assert _context_hidden(2560, 96) == 256


def test_keep_race_retains_only_leader_and_baseline() -> None:
    races: dict = {}
    baseline = {"tile": 1}
    cands = [SweepCandidate.create({"tile": t}) for t in (1, 2, 3)]
    _keep_race(races, cands[0], _race(30.0), baseline)  # baseline, leader
    _keep_race(races, cands[1], _race(20.0), baseline)  # new leader
    _keep_race(races, cands[2], _race(25.0), baseline)  # slower: dropped
    kept = {k for k in races if not k.startswith("__")}
    assert kept == {cands[0].candidate_id, cands[1].candidate_id}
    assert races["__leader__"][0] == cands[1].candidate_id
    assert races["__baseline__"] == cands[0].candidate_id


def test_confirm_winner_replaces_both_latencies_with_interleaved_medians() -> None:
    baseline = {"tile": 1}
    ms = [_measurement({"tile": 1}, 30.0), _measurement({"tile": 2}, 20.0)]
    races = {
        "__leader__": (ms[1].candidate.candidate_id, 20.0),
        "__baseline__": ms[0].candidate.candidate_id,
        ms[0].candidate.candidate_id: _race(30.0, [29.0] * CONFIRM_ROUNDS),
        ms[1].candidate.candidate_id: _race(20.0, [28.0] * CONFIRM_ROUNDS),
    }
    confirmed = _confirm_winner(ms, races, baseline_config=baseline)
    by_id = {m.candidate.candidate_id: m for m in confirmed}
    assert by_id[ms[0].candidate.candidate_id].latency_us == 29.0
    assert by_id[ms[1].candidate.candidate_id].latency_us == 28.0
    assert by_id[ms[1].candidate.candidate_id].metrics["first_pass_us"] == 20.0
    assert by_id[ms[1].candidate.candidate_id].metrics["confirmed"] is True


def test_confirm_winner_accepts_a_custom_pass_predicate() -> None:
    @dataclass
    class Loose:
        candidate: SweepCandidate
        latency_us: float | None
        cosine: float
        metrics: dict = field(default_factory=dict)

    a = Loose(SweepCandidate.create({"tile": 1}), 30.0, 0.999)
    b = Loose(SweepCandidate.create({"tile": 2}), 20.0, 0.999)
    races = {
        "__leader__": (b.candidate.candidate_id, 20.0),
        a.candidate.candidate_id: _race(30.0, [31.0] * CONFIRM_ROUNDS),
        b.candidate.candidate_id: _race(20.0, [21.0] * CONFIRM_ROUNDS),
    }
    confirmed = _confirm_winner(
        [a, b], races, baseline_config={"tile": 1}, passes=lambda m: m.cosine >= 0.99
    )
    assert [m.latency_us for m in confirmed] == [31.0, 21.0]


def test_layer_budget_clamps_to_free_memory(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (10 << 30, 96 << 30)
    )
    assert layer_budget("cuda:0", 1 << 30) == 3
    assert layer_budget("cuda:0", 3 << 30) == 2
    assert layer_budget("cuda:0", 50 << 30) == 1


def test_generic_stack_brackets_only_the_tested_slot() -> None:
    log: list[str] = []

    class Event:
        def __init__(self, name):
            self.name = name

        def record(self):
            log.append(self.name)

    class Stack(GenericLayerStack):
        def __init__(self):  # bypass the CUDA fillers
            self.layers = 2
            self.hidden_state = 0.0
            self.norm_weight = 1.0
            self.fillers_in = [lambda h: log.append("in") or h for _ in range(2)]
            self.fillers_out = [lambda o: log.append("out") or o for _ in range(2)]

        def produce(self, layer, activation):
            log.append("produce")

        def tested(self, layer, slot):
            log.append(f"tested:{slot}")
            return 0.0

    monkeypatch_norm = lambda x, w, eps=1e-6: x  # noqa: E731
    stack = Stack()
    original = layer_stack.rmsnorm
    layer_stack.rmsnorm = monkeypatch_norm
    try:
        stack.run(
            "plan", [(Event("start"), Event("end")), (Event("start"), Event("end"))]
        )
    finally:
        layer_stack.rmsnorm = original
    assert log == ["in", "produce", "start", "tested:plan", "end", "out"] * 2
