"""Model-wide admission, policy windows and all-layer acknowledgement."""
from dataclasses import replace

import pytest

from b12x.moe import residency as r


def snapshot(counts=None, calls=0, rank=0):
    counts = counts or {"a": (0,)*4, "b": (0,)*4}
    return r.RoutingSnapshot(epoch=1, rank=rank, layers=tuple(r.LayerRoutingCounts(
        layer=name, phase="decode", counts=tuple(row), calls=calls, sampled_calls=calls,
        tokens=sum(row), sampled_tokens=sum(row)) for name, row in counts.items()))


def make_slots(name, *, rank=0, canonical=False):
    return r.ResidencySlotSnapshot(preparation_id=f"{name}:{rank}", generation=0,
        expert_map=((0, 0), (0, 1), (1, 2 if canonical else 0), (1, 3 if canonical else 1)), healthy=True)


def config(**changes):
    return r.ResidencyCacheConfig(max_pairs=2, minimum_cold_selections=1,
        minimum_score_gain=1, minimum_residency_windows=0, **changes)


def spec(canonical=False):
    return r.ResidencyExchangeSpec(backend="test", direct_backing_execution=True,
        fixed_address_quiescent_exchange=True, payload_copy_bytes_per_pair=100,
        map_copy_bytes_per_transaction=32, backing_mode="canonical" if canonical else "exclusive")


def coordinator(*, max_pairs=3, max_bytes=364, replicas=1, canonical=False, **policy):
    slots = {name: make_slots(name, canonical=canonical) for name in ("a", "b")}
    controllers = {name: r.ResidencyCacheController(config=config(**policy),
        observations=r.RoutingObservationSpec(layer=name, experts=4, phase="decode", max_top_k=2),
        exchange=spec(canonical), slots=value, baseline=snapshot()) for name, value in slots.items()}
    return r.ResidencyEpochCoordinator(controllers, budget=r.ResidencyEpochBudget(
        max_pairs=max_pairs, max_copy_bytes=max_bytes), replicas=replicas), slots


def complete(coordinator, decision, slots):
    result = {}
    for item in decision.layers:
        before = slots[item.layer]
        result[item.layer] = replace(before, generation=before.generation + bool(item.pairs),
            expert_map=r.updated_slot_map(before, item.pairs,
                backing_mode=coordinator.controllers[item.layer].exchange.backing_mode))
    return result


@pytest.mark.parametrize("canonical", [False, True])
def test_global_budget_selects_subset_and_accounts_one_map_per_layer(canonical):
    c, slots = coordinator(canonical=canonical)
    d = c.observe(snapshot({"a": (0, 0, 8, 6), "b": (0, 0, 10, 2)}, 1), slots=slots)
    assert d.selected_pairs == 3 and d.skipped_pairs == 1 and d.copy_bytes == 364
    assert {x.layer: x.pairs for x in d.layers} == {"a": ((2, 0), (3, 1)), "b": ((2, 0),)}
    result = complete(c, d, slots)
    outcomes = c.finish(d, slots=result)
    assert sum(x.promotions for x in outcomes.values()) == 3
    assert all(s.generation == 1 for s in result.values())


def test_declining_movement_keeps_decayed_history_without_changing_slots():
    c, slots = coordinator(scoring="decayed_lfu")
    first = c.observe(snapshot({n: (8, 0, 4, 0) for n in slots}, 1),
                      slots=slots, allow_movement=False)
    assert first.proposed_pairs == first.selected_pairs == 0
    c.finish(first, slots=slots)
    second = c.observe(snapshot({n: (8, 0, 14, 0) for n in slots}, 2), slots=slots)
    assert all(x.decision.scores == (4, 0, 12, 0) for x in second.layers)
    assert all(x.pairs == ((2, 1),) for x in second.layers)


@pytest.mark.parametrize("replicas,budget,pairs,used", [(1, 131, 0, 0), (1, 132, 1, 132),
    (2, 263, 0, 0), (2, 264, 1, 264), (2, 464, 2, 464)])
def test_global_budget_includes_every_replica_and_transaction_map(replicas, budget, pairs, used):
    c, slots = coordinator(max_bytes=budget, replicas=replicas)
    d = c.observe(snapshot({"a": (0, 0, 8, 6), "b": (0, 0, 1, 1)}, 1), slots=slots)
    assert (d.selected_pairs, d.copy_bytes) == (pairs, used)
    c.finish(d, slots=complete(c, d, slots))


def test_invalid_later_layer_does_not_consume_first_observation():
    c, slots = coordinator()
    observed = snapshot({"a": (0, 0, 8, 6), "b": (0, 0, 10, 2)}, 1)
    with pytest.raises(ValueError, match="placement changed"):
        c.observe(observed, slots={**slots, "b": replace(slots["b"], generation=1)})
    d = c.observe(observed, slots=slots)
    result = complete(c, d, slots)
    with pytest.raises(ValueError, match="accepted pairs"):
        c.finish(d, slots={**result, "b": slots["b"]})
    # No early layer was acknowledged by the rejected model completion.
    c.finish(d, slots=result)
    with pytest.raises(ValueError, match="stale"):
        c.finish(d, slots=result)


def test_noop_stable_workload_and_transition():
    c, slots = coordinator()
    d = c.observe(snapshot({"a": (8, 4, 0, 0), "b": (8, 4, 0, 0)}, 1), slots=slots)
    assert d.selected_pairs == d.copy_bytes == 0
    c.finish(d, slots=slots)
    d = c.observe(snapshot({"a": (8, 4, 8, 0), "b": (8, 4, 0, 8)}, 2), slots=slots)
    assert d.selected_pairs == 2
    c.finish(d, slots=complete(c, d, slots))


def test_failure_requires_new_model_baseline():
    c, slots = coordinator()
    d = c.observe(snapshot(), slots=slots)
    c.fail()
    with pytest.raises(RuntimeError, match="reload"):
        c.finish(d, slots=slots)
    with pytest.raises(RuntimeError, match="reload"):
        c.observe(snapshot(), slots=slots)


def test_decayed_scores_preserve_history_and_empty_polls_do_not_decay():
    c, slots = coordinator(scoring="decayed_lfu", max_pairs=0)
    d = c.observe(snapshot({"a": (0, 20, 0, 0), "b": (0, 20, 0, 0)}, 1), slots=slots)
    c.finish(d, slots=slots)
    for _ in range(3):
        d = c.observe(snapshot({"a": (0, 20, 0, 0), "b": (0, 20, 0, 0)}, 1), slots=slots)
        assert d.layers[0].decision.scores == (0, 20, 0, 0)
        c.finish(d, slots=slots)
    d = c.observe(snapshot({"a": (1, 20, 5, 0), "b": (1, 20, 5, 0)}, 2), slots=slots)
    assert d.layers[0].decision.scores == (1, 10, 5, 0)
    assert d.layers[0].decision.pairs == ((2, 0),)
    # Recent frequency would evict expert 1, whose current-window count is zero.
    c.finish(d, slots=slots)


@pytest.mark.parametrize("changes", [{"scoring": "lru"}, {"decay": 0}, {"decay": 1},
    {"decay": float("nan")}, {"decay": True}])
def test_invalid_policy_controls_fail_closed(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_backing_modes_and_subset_acknowledgement_are_explicit():
    slots = make_slots("a", canonical=True)
    with pytest.raises(ValueError, match="physical row"):
        r.updated_slot_map(slots, ())
    assert r.updated_slot_map(slots, ((2, 0),), backing_mode="canonical") == (
        (1, 0), (0, 1), (0, 0), (1, 3))
    c, slots = coordinator()
    d = c.observe(snapshot({"a": (0, 0, 8, 6), "b": (0, 0, 10, 2)}, 1), slots=slots)
    with pytest.raises(ValueError, match="subset"):
        c.controllers["a"].validate_completion(d.layers[0].decision, slots=slots["a"], accepted_pairs=((1, 0),))


def test_epoch_requires_matching_sampling_and_phase():
    c, slots = coordinator()
    other = r.ResidencyCacheController(config=config(), observations=r.RoutingObservationSpec(
        layer="b", experts=4, phase="decode", max_top_k=2, sample_every=2),
        exchange=spec(), slots=slots["b"], baseline=snapshot())
    with pytest.raises(ValueError, match="sampling"):
        r.ResidencyEpochCoordinator({"a": c.controllers["a"], "b": other}, budget=c.budget)
