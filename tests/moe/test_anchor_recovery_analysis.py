"""Diagnostic eligibility is checked against the actual controller proposals."""

from dataclasses import replace

import pytest

from b12x.moe import residency as r
from benchmarks.moe.analyze_anchor_recovery import classify_layer
from tests.moe.test_residency_epoch import coordinator, snapshot


def test_diagnostic_distinguishes_global_cap_from_margin_without_changing_policy():
    model, slots = coordinator(max_pairs=1, max_bytes=1000)
    anchor = r.ResidencyAnchor(profile_id='a'*64, checkpoint='checkpoint',
        recipe='recipe', workload='general', placements=tuple((n, r.ExpertPlacement(
            total_experts=4, resident_expert_ids=(2, 3), backing_expert_ids=(0, 1))) for n in slots))
    decision = model.observe(snapshot({n: (1, 5, 10, 2) for n in slots}, 1),
                             slots=slots, recenter=anchor)
    results = []
    for layer in decision.layers:
        d = layer.decision
        results.append(classify_layer(counts=d.counts, scores=d.scores, hot=(0, 1),
            anchor=(2, 3), protected=d.protected_hot_experts, minimum_count=1,
            margin=1, layer_pairs=2, proposed=d.pairs, selected=layer.pairs))
    assert sum(x['reasons'].get('global_budget', 0) for x in results) == 1
    assert all(x['reasons']['score_margin'] == 1 for x in results)
    assert all(x['traffic_weighted_coverage'] == 0 for x in results)


def test_diagnostic_attributes_primary_guard_and_counts_weighted_coverage():
    common = dict(counts=(10, 0, 0, 0, 9, 1, 0, 8, 0), scores=(10, 0, 0, 0, 9, 1, 0, 8, 0),
                  hot=(0, 1, 2, 3, 8), anchor=(0, 4, 5, 6, 7), minimum_count=2,
                  margin=1, layer_pairs=1)
    result = classify_layer(**common, protected=(2, 3, 8), proposed=((4, 1),), selected=())
    assert result['reasons'] == dict(minimum_count=1, unobserved=1, protected_victim=1, global_budget=1)
    assert result['traffic_weighted_coverage'] == 10/28
    result = classify_layer(**common, protected=(), proposed=((4, 1),), selected=((4, 1),))
    assert result['reasons']['per_layer_cap'] == 1
    result = classify_layer(**common, protected=(), proposed=(), selected=(), allow=False)
    assert result['reasons']['anchor_gate'] == 2
    with pytest.raises(ValueError, match='proposals'):
        classify_layer(**common, protected=(), proposed=(), selected=())


def test_residence_guard_is_reported_when_actual_controller_protects_victims():
    model, slots = coordinator(max_pairs=1, max_bytes=1000)
    c = model.controllers['a']
    c.config = replace(c.config, minimum_residency_windows=3)
    d = c.observe(snapshot({'a': (0, 0, 12, 8), 'b': (0, 0, 0, 0)}, 1),
                  slots=slots['a'], recenter_to=(2, 3))
    result = classify_layer(counts=d.counts, scores=d.scores, hot=(0, 1), anchor=(2, 3),
        protected=d.protected_hot_experts, minimum_count=1, margin=1, layer_pairs=2,
        proposed=d.pairs, selected=())
    assert result['reasons'] == {'protected_victim': 2}
