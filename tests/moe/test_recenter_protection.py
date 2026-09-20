"""Temporary hit-based retention affects recovery victims only."""
from dataclasses import replace
import pytest
from b12x.moe import residency as r
from tests.moe.test_residency_epoch import config, make_slots, snapshot, spec


def scenario(protect=1, windows=2):
    slots = make_slots('a', canonical=True)
    controller = r.ResidencyCacheController(
        config=config(recenter_protected_experts=protect, recenter_protection_windows=windows),
        observations=r.RoutingObservationSpec(layer='a', experts=4, phase='decode', max_top_k=2),
        exchange=spec(True), slots=slots, baseline=snapshot({'a': (0,)*4}))
    counts, calls = [0]*4, 0
    def step(delta, *, recenter=False, propose=True, commit=True):
        nonlocal counts, calls, slots
        counts = [a+b for a,b in zip(counts,delta,strict=True)]
        calls += bool(sum(delta))
        decision = controller.observe(snapshot({'a': tuple(counts)}, calls), slots=slots,
                                      recenter_to=(0,1) if recenter else None, propose=propose)
        if commit and decision.pairs:
            slots = replace(slots, generation=slots.generation+1,
                expert_map=r.updated_slot_map(slots, decision.pairs, backing_mode='canonical'))
        controller.finish(decision, slots=slots)
        return decision
    assert step((0,0,10,4)).pairs == ((2,0),(3,1))
    step((0,0,40,1), propose=False)
    return controller, step


def test_recovery_protects_hits_not_low_current_score_and_expires():
    controller, step = scenario()
    first = step((100,80,0,8), recenter=True)
    assert first.recenter_protected_experts == (2,) and first.pairs == ((0,3),)
    until = first.recenter_protection_until_window
    empty = step((0,0,0,0), recenter=True)
    assert empty.window == first.window and empty.recenter_protected_experts == (2,)
    second = step((1,80,0,0), recenter=True)
    assert not second.pairs and second.recenter_protection_until_window == until
    expired = step((1,80,0,0), recenter=True)
    assert expired.recenter_protected_experts == () and expired.pairs == ((1,2),)
    later = step((1,80,0,0), recenter=True)
    assert later.recenter_protection_until_window == until  # No automatic renewal.


def test_normal_adaptation_can_evict_protected_resident_and_rearm_after_drift():
    controller, step = scenario(windows=16)
    first = step((100,80,0,8), recenter=True)
    normal = step((20,90,0,0))
    assert normal.pairs == ((1,2),) and not normal.recenter_protected_experts
    # Moving an anchor candidate does not renew the episode.
    assert controller._recenter_until == first.recenter_protection_until_window
    step((0,0,100,80))
    assert controller._recenter_until is None
    step((0,0,40,1), propose=False)
    again = step((100,80,0,8), recenter=True)
    assert again.recenter_protected_experts == (2,) and again.recenter_protection_until_window > first.recenter_protection_until_window


def test_declined_anchor_gate_never_moves_or_arms_and_default_is_unchanged():
    controller, step = scenario()
    declined = step((100,80,0,8), recenter=True, propose=False)
    assert not declined.pairs and controller._recenter_until is None
    plain, unguarded = scenario(protect=0, windows=0)
    d = unguarded((100,80,0,8), recenter=True)
    assert d.pairs == ((0,2),(1,3)) and d.recenter_protected_experts == ()


def test_stale_generation_or_changed_anchor_cannot_renew_protection():
    controller, step = scenario(windows=16)
    d = step((100,80,0,8), recenter=True)
    before = (controller._window, controller._recenter_retained, controller._recenter_until)
    with pytest.raises(ValueError, match='placement changed'):
        controller.observe(snapshot({'a': (1000,)*4}, 100), slots=replace(controller._slots,generation=99), recenter_to=(0,1))
    with pytest.raises(ValueError, match='reference changed'):
        controller.observe(snapshot({'a': (1000,)*4}, 100), slots=controller._slots, recenter_to=(0,3))
    assert before == (controller._window, controller._recenter_retained, controller._recenter_until)


@pytest.mark.parametrize('kw', [dict(recenter_protected_experts=1), dict(recenter_protection_windows=2),
                                dict(recenter_protected_experts=-1), dict(recenter_protected_experts=True)])
def test_invalid_protection_config(kw):
    with pytest.raises(ValueError):config(**kw)


def test_model_rejects_changed_later_reference_before_consuming_any_window():
    from tests.moe.test_residency_epoch import coordinator
    model, slots = coordinator(recenter_protected_experts=1, recenter_protection_windows=2)
    def anchor(second):
        return r.ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint', recipe='recipe', workload='general',
            placements=tuple((n, r.ExpertPlacement(total_experts=4, resident_expert_ids=ids,
                backing_expert_ids=tuple(e for e in range(4) if e not in ids)))
                for n,ids in [('a',(0,1)),('b',second)]))
    first = model.observe(snapshot({'a':(1,0,0,0),'b':(1,0,0,0)},1), slots=slots, recenter=anchor((0,1)))
    model.finish(first,slots=slots)
    before = {n:c._window for n,c in model.controllers.items()}
    with pytest.raises(ValueError,match='reference changed'):
        model.observe(snapshot({'a':(2,0,0,0),'b':(2,0,0,0)},2), slots=slots, recenter=anchor((2,3)))
    assert {n:c._window for n,c in model.controllers.items()} == before


def test_zero_hit_promotion_receives_no_recovery_protection():
    controller, step = scenario()
    # A normal generation replaces both residents, resetting earned-hit lifetimes.
    step((100,80,0,0))
    step((0,0,100,80))
    decision = step((100,80,0,0), recenter=True)
    assert decision.recenter_protected_experts == ()
    assert len(decision.pairs) == 2


def test_recovery_diagnostic_accounts_for_temporary_victim_guard():
    from benchmarks.moe.analyze_anchor_recovery import classify_layer
    _, step = scenario()
    decision = step((100,80,0,8), recenter=True)
    result = classify_layer(counts=decision.counts, scores=decision.scores, hot=(2,3), anchor=(0,1),
        protected=(*decision.protected_hot_experts,*decision.recenter_protected_experts),
        minimum_count=1, margin=1, layer_pairs=2, proposed=decision.pairs, selected=decision.pairs)
    assert result['reasons'] == {'selected':1, 'protected_victim':1}
