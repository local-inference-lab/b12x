"""Recent-frequency decisions from generation-bound cumulative counters."""
from dataclasses import replace

import pytest

from b12x.moe import fused_moe as moe
from tests.moe.test_residency_updates import fixture, HostTransfer


def snapshot(counts=(0, 0, 0, 0), *, calls=0, epoch=0, rank=0, phase="decode"):
    return moe.RoutingSnapshot(epoch=epoch, rank=rank, layers=(moe.LayerRoutingCounts(
        layer="layer", phase=phase, counts=counts, calls=calls, sampled_calls=calls,
        tokens=sum(counts), sampled_tokens=sum(counts)),))


def controller(slots, *, config_changes=None, query_changes=None):
    config = moe.ResidencyCacheConfig(max_pairs=2, minimum_cold_selections=2,
        minimum_score_gain=2, minimum_residency_windows=0)
    config = replace(config, **(config_changes or {}))
    spec = moe.ResidencyLayerSpec(layer="layer", experts=4, hidden=256,
        intermediate=256, max_tokens=128, max_top_k=3)
    query = moe.RoutingProfileQuery(layers=(("layer", 4),), max_tokens=128, max_top_k=3)
    query = replace(query, **(query_changes or {}))
    return moe.ResidencyCacheController(spec=spec, config=config, counter_query=query,
        slots=slots, baseline=snapshot())


def test_batch_frequency_order_ties_diagnostics_and_promotion_reuse():
    updates = fixture()
    policy = controller(updates.snapshot())
    decision = policy.observe(snapshot((1, 7, 1, 7), calls=1), slots=updates.snapshot())
    assert decision.pairs == ((1, 0), (3, 2))
    assert decision.unique_cold_experts == (1, 3) and decision.cold_selections == 14
    assert decision.cold_fraction == 14/16 and decision.counterfactual_cold_fraction == 2/16
    result = updates.exchange(decision.pairs, expected=decision.expected, quiescent=True)
    outcome = policy.finish(decision, slots=result)
    assert outcome.hot_experts == (1, 3) and outcome.promotions == 2
    assert outcome.committed_payload_copy_bytes == 8*policy.spec.expert_bytes and outcome.committed_map_copy_bytes == 64
    # Only post-promotion selections count as earned HBM hits.
    decision = policy.observe(snapshot((10, 10, 1, 8), calls=2), slots=result)
    assert decision.observed_hits_since_promotion == ((1, 3), (3, 1))
    assert decision.pairs == ((0, 3),)
    result = updates.exchange(decision.pairs, expected=result, quiescent=True)
    outcome = policy.finish(decision, slots=result)
    assert outcome.evicted_promotion_hits == ((3, 1),)
    assert outcome.observed_hits_after_promotion == 4 and outcome.promotions == 3


def test_threshold_hysteresis_and_explicit_budget():
    u = fixture()
    p = controller(u.snapshot(), config_changes={"max_pairs": 1})
    d = p.observe(snapshot((0, 4, 0, 4), calls=1), slots=u.snapshot())
    assert d.pairs == ((1, 0),) and d.unpaired_candidates == 1
    p.finish(d, slots=u.snapshot())  # Decline without changing placement.
    d = p.observe(snapshot((2, 5, 2, 7), calls=2), slots=u.snapshot())
    assert d.below_threshold == 1 and d.below_hysteresis == 1 and not d.pairs
    assert p.finish(d, slots=u.snapshot()).promotions == 0


def test_residency_guard_and_empty_polls_cannot_accelerate_eviction():
    u = fixture()
    p = controller(u.snapshot(), config_changes={"minimum_residency_windows": 2})
    d = p.observe(snapshot((0, 5, 0, 0), calls=1), slots=u.snapshot())
    assert not d.pairs and d.protected_hot_experts == (0, 2)
    p.finish(d, slots=u.snapshot())
    for _ in range(3):
        d = p.observe(snapshot((0, 5, 0, 0), calls=1), slots=u.snapshot())
        assert d.window == 1 and not d.pairs
        p.finish(d, slots=u.snapshot())
    d = p.observe(snapshot((0, 10, 0, 0), calls=2), slots=u.snapshot())
    p.finish(d, slots=u.exchange(d.pairs, expected=d.expected, quiescent=True))
    d = p.observe(snapshot((0, 10, 8, 9), calls=3), slots=u.snapshot())
    assert d.protected_hot_experts == (1,) and not d.pairs  # Other hot victim scores 8.
    p.finish(d, slots=u.snapshot())
    d = p.observe(snapshot((0, 10, 16, 18), calls=4), slots=u.snapshot())
    assert d.pairs == ((3, 1),)


def test_rollback_decline_and_pending_acknowledgement():
    u = fixture(transfer=HostTransfer(fail=10))
    p = controller(u.snapshot())
    d = p.observe(snapshot((0, 5, 0, 0), calls=1), slots=u.snapshot())
    with pytest.raises(RuntimeError, match="pending"):
        p.observe(snapshot((0, 5, 0, 0), calls=1), slots=u.snapshot())
    with pytest.raises(moe.ResidencyUpdateError) as caught:
        u.exchange(d.pairs, expected=d.expected, quiescent=True)
    assert caught.value.resumable
    outcome = p.finish(d, slots=u.snapshot())
    assert outcome.promotions == outcome.committed_payload_copy_bytes == 0
    with pytest.raises(ValueError, match="stale"):
        p.finish(d, slots=u.snapshot())
    d = p.observe(snapshot((0, 10, 0, 0), calls=2), slots=u.snapshot())
    assert d.pairs == ((1, 0),)
    with pytest.raises(ValueError, match="healthy"):
        p.finish(d, slots=replace(u.snapshot(), healthy=False))
    with pytest.raises(ValueError, match="differs"):
        p.finish(d, slots=replace(u.snapshot(), generation=100))
    p.finish(d, slots=u.exchange(d.pairs, expected=d.expected, quiescent=True))


@pytest.mark.parametrize("bad", [snapshot((0, 1, 0, 0), epoch=1), snapshot(rank=1),
    snapshot(phase="prefill"), snapshot((0, 0)), snapshot((0, 1, 0, 0))])
def test_counter_identity_phase_geometry_and_consistency_fail_closed(bad):
    u = fixture()
    p = controller(u.snapshot())
    with pytest.raises(ValueError): p.observe(bad, slots=u.snapshot())
    assert p.observe(snapshot((0, 3, 0, 0), calls=1), slots=u.snapshot()).pairs == ((1, 0),)


def test_counter_regression_external_swap_and_foreign_preparation():
    u = fixture()
    p = controller(u.snapshot())
    d = p.observe(snapshot((0, 5, 0, 0), calls=1), slots=u.snapshot())
    p.finish(d, slots=u.snapshot())
    with pytest.raises(ValueError, match="decreased"):
        p.observe(snapshot((0, 4, 0, 0), calls=2), slots=u.snapshot())
    with pytest.raises(ValueError, match="placement changed"):
        p.observe(snapshot(), slots=fixture().snapshot())
    changed = u.exchange(((0, 1),), expected=u.snapshot(), quiescent=True)
    with pytest.raises(ValueError, match="placement changed"):
        p.observe(snapshot((0, 8, 0, 0), calls=2), slots=changed)


def test_sampling_is_reported_without_extrapolating_and_tp_counts_once():
    u = fixture()
    p = controller(u.snapshot(), query_changes={"sample_every": 4})
    row = snapshot((0, 3, 0, 0), calls=1).layers[0]
    d = p.observe(moe.RoutingSnapshot(epoch=0, rank=0, layers=(replace(row, calls=4),)), slots=u.snapshot())
    assert d.sample_every == 4 and d.calls == 4 and d.sampled_calls == 1 and d.cold_selections == 3
    with pytest.raises(ValueError, match="authoritative"):
        moe.ResidencyCacheController(spec=p.spec, config=p.config,
            counter_query=replace(p.counter_query, rank=1, tp_size=2), slots=u.snapshot(), baseline=snapshot(rank=1))


@pytest.mark.parametrize("hot", [True, False])
def test_single_tier_has_no_exchange(hot):
    u = fixture()
    slots = replace(u.snapshot(), expert_map=tuple((0 if hot else 1, e) for e in range(4)))
    p = controller(slots)
    d = p.observe(snapshot((1, 3, 5, 7), calls=1), slots=slots)
    assert not d.pairs and d.cold_selections == (0 if hot else 16)
    assert p.finish(d, slots=slots).committed_payload_copy_bytes == 0


def test_explicit_config_and_geometry_validation():
    u = fixture()
    p = controller(u.snapshot())
    for kwargs in ({"max_pairs": 0}, {"minimum_cold_selections": True}, {"minimum_score_gain": 0},
                   {"minimum_residency_windows": -1}, {"phase": "all"}):
        with pytest.raises(ValueError): replace(p.config, **kwargs)
    with pytest.raises(ValueError, match="physical row"):
        controller(replace(u.snapshot(), expert_map=((0, 0),)*4))


def test_independent_layer_windows_do_not_share_expert_scores():
    a, b = fixture(), fixture()
    first = controller(a.snapshot())
    second = moe.ResidencyCacheController(spec=replace(first.spec, layer="other", hidden=512),
        config=first.config,
        counter_query=replace(first.counter_query, layers=(("layer", 4), ("other", 4))),
        slots=b.snapshot(), baseline=moe.RoutingSnapshot(epoch=0, rank=0,
            layers=(moe.LayerRoutingCounts(layer="other", phase="decode", counts=(0,)*4),)))
    snapshot_a = snapshot((0, 5, 0, 0), calls=1)
    snapshot_b = moe.RoutingSnapshot(epoch=0, rank=0, layers=(*snapshot_a.layers,
        replace(snapshot((0, 0, 0, 7), calls=1).layers[0], layer="other")))
    da = first.observe(snapshot_b, slots=a.snapshot())
    db = second.observe(snapshot_b, slots=b.snapshot())
    assert da.pairs == ((1, 0),) and db.pairs == ((3, 0),)
    result = second.finish(db, slots=b.exchange(db.pairs, expected=db.expected, quiescent=True))
    assert result.committed_payload_copy_bytes == 4*second.spec.expert_bytes == 8*first.spec.expert_bytes


def test_wrong_or_missing_acknowledgement_cannot_advance_policy():
    a, b = fixture(), fixture()
    first, second = controller(a.snapshot()), controller(b.snapshot())
    with pytest.raises(ValueError, match="stale"):
        first.finish(None, slots=a.snapshot())
    da = first.observe(snapshot((0, 5, 0, 0), calls=1), slots=a.snapshot())
    db = second.observe(snapshot((0, 5, 0, 0), calls=1), slots=b.snapshot())
    with pytest.raises(ValueError, match="stale"):
        first.finish(db, slots=a.snapshot())
    changed = a.exchange(((2, 3),), expected=a.snapshot(), quiescent=True)
    with pytest.raises(ValueError, match="differs"):
        first.finish(da, slots=changed)
