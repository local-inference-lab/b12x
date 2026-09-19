"""Host checks for the physical policy replay experiment."""

from dataclasses import replace

import pytest

from benchmarks.moe.sm120_trace_replay import prepare_policy
from b12x.testing.residency_replay import LayerTrace, RouteCall


def trace(test_ids):
    return LayerTrace(
        "layer.12",
        6,
        128,
        128,
        "checkpoint",
        "decode",
        (
            RouteCall("train", "code", "train", ((5, 5),)),
            RouteCall("train", "code", "train", ((4, 4),)),
            *(RouteCall("test", "code", "test", (tuple(ids),)) for ids in test_ids),
        ),
    )


@pytest.mark.parametrize("policy", ["b12x", "lru", "lfu", "decayed_lfu"])
def test_canonical_schedule_matches_physical_hit_accounting(policy):
    t = trace([(0, 0)] * 8 + [(1, 1)] * 8 + [(2, 2)] * 8)
    calls, initial, result = prepare_policy(t, hot=2, window=2, policy=policy)
    assert initial == (4, 5)  # Held-out expert zero must not leak into the prior.
    changes = {w["end"]: w["pairs"] for w in result["windows"]}
    slots = list(initial)
    cold = 0
    for i, call in enumerate(calls):
        cold += sum(e not in slots for e in call.ids[0])
        for candidate, victim in changes.get(i + 1, ()):
            assert candidate not in slots and victim in slots
            slots[slots.index(victim)] = candidate
        assert len(set(slots)) == 2
    assert result["movement_pairs"] > 0
    assert cold == result["totals"]["cold_selections"]
    assert changes[len(calls)] == ()


@pytest.mark.parametrize("policy", ["b12x", "lru", "lfu", "decayed_lfu"])
def test_future_suffix_does_not_change_causal_decisions(policy):
    prefix = [(0, 0)] * 6
    a = prepare_policy(trace(prefix + [(1, 1)] * 8), hot=2, window=2, policy=policy)
    b = prepare_policy(trace(prefix + [(3, 3)] * 8), hot=2, window=2, policy=policy)
    assert a[1] == b[1]
    assert a[2]["windows"][:3] == b[2]["windows"][:3]


def test_physical_replay_rejects_ambiguous_invocation_shapes():
    t = trace([(0, 1), (1,)])
    with pytest.raises(ValueError, match="C1 and fixed top-k"):
        prepare_policy(t, hot=2, window=2, policy="lfu")
    t = replace(t, calls=t.calls[:2])
    with pytest.raises(ValueError, match="C1 and fixed top-k"):
        prepare_policy(t, hot=2, window=2, policy="lfu")
    t = replace(t, calls=t.calls + (RouteCall("test", "code", "test", ((0,), (1,))),))
    with pytest.raises(ValueError, match="C1 and fixed top-k"):
        prepare_policy(t, hot=2, window=2, policy="lfu")
