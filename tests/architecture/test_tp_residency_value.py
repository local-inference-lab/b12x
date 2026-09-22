"""Offline transaction value requires complete, matching rank observations."""

from copy import deepcopy

import pytest

from benchmarks.moe.analyze_tp_residency_value import analyze


def records():
    rows = []
    mapping = [[0, 0], [1, 1], [0, 1], [1, 3]]
    generation = 0
    previous = [0] * 4
    for index, (counts, pairs) in enumerate(
        [
            ([0, 0, 0, 0], []),
            ([0, 5, 0, 0], [[1, 0]]),
            ([2, 12, 0, 0], [[3, 1]]),
            ([5, 13, 0, 2], []),
        ]
    ):
        delta = [b - a for a, b in zip(previous, counts, strict=True)]
        rank = dict(
            checkpoint_id="checkpoint",
            initial_profile_id="profile",
            snapshot={"layers": [dict(layer="layer", phase="decode", counts=counts)]},
            layers={
                "layer": {
                    "slots": dict(generation=generation, expert_map=deepcopy(mapping))
                }
            },
        )
        worker = dict(
            status="complete",
            baseline=index == 0,
            workers={str(r): deepcopy(rank) for r in range(3)},
            layers={}
            if index == 0
            else {"layer": dict(pairs=pairs, generation=generation + bool(pairs))},
            selected_pairs=len(pairs),
            proposed_pairs=len(pairs),
            proposal_backlog={},
            copy_bytes=3 * len(pairs),
            per_rank_copy_bytes=[len(pairs)] * 3,
            selections=sum(delta),
            cold_selections=sum(
                n for n, (tier, _) in zip(delta, mapping, strict=True) if tier
            ),
        )
        rows.append(
            dict(
                kind="maintenance",
                time_ns=index,
                receipt=dict(
                    worker=worker,
                    engine_wall_ns=1,
                    engine_stages_ns={},
                ),
            )
        )
        for candidate, victim in pairs:
            mapping[candidate] = [0, mapping[victim][1]]
            mapping[victim] = [1, victim]
        generation += bool(pairs)
        previous = counts
    return rows


def test_lifetime_stops_when_exchange_is_undone_and_keeps_eviction_demand():
    result = analyze(records())
    first, second, _ = result["transactions"]
    assert result["ranks"] == 3
    assert first["promoted_hits"] == 7
    assert first["evicted_expert_demand"] == 2
    assert first["net_avoided_cold"] == 5
    assert not first["pairs"][0]["right_censored"]
    assert second["net_avoided_cold"] == 1
    assert second["pairs"][0]["right_censored"]


def test_final_transaction_has_no_invented_future_hits():
    result = analyze(records()[:3])
    assert not result["transactions"][0]["pairs"][0]["right_censored"]
    last = result["transactions"][-1]
    assert last["unobserved_pairs"] == 1
    assert last["zero_hit_observed_pairs"] == 0
    assert last["promoted_hits"] is None
    assert last["net_avoided_cold"] is None
    assert last["pairs"][0]["right_censored"]


def test_disagreement_and_counter_reset_fail_closed():
    rows = records()
    rows[1]["receipt"]["worker"]["workers"]["2"]["snapshot"]["layers"][0]["counts"][
        0
    ] += 1
    with pytest.raises(ValueError, match="TP ranks disagree"):
        analyze(rows)
    rows = records()
    for rank in rows[2]["receipt"]["worker"]["workers"].values():
        rank["snapshot"]["layers"][0]["counts"][1] = 0
    with pytest.raises(ValueError, match="counter reset"):
        analyze(rows)


def test_unrecorded_map_change_is_rejected():
    rows = records()
    for rank in rows[2]["receipt"]["worker"]["workers"].values():
        rank["layers"]["layer"]["slots"]["generation"] += 1
    rows[2]["receipt"]["worker"]["layers"]["layer"]["generation"] += 1
    with pytest.raises(ValueError, match="placement changed"):
        analyze(rows)


def test_profile_change_between_epochs_is_rejected():
    rows = records()
    for rank in rows[2]["receipt"]["worker"]["workers"].values():
        rank["initial_profile_id"] = "another-profile"
    with pytest.raises(ValueError, match="identity or baseline changed"):
        analyze(rows)


def test_request_boundaries_remain_retrospective_timestamps():
    rows = records()
    result = analyze(rows + [dict(kind="request", index=0, workload="held-out", start_wall_ns=2)])
    assert result["transactions"][0]["completion_time_ns"] == 1
    assert result["request_admissions"] == [dict(index=0, workload="held-out", start_wall_ns=2)]
    assert result["transactions"] == analyze(rows)["transactions"]
