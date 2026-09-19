"""Independent locality and held-out replay checks."""

import json
from dataclasses import replace

import pytest

from b12x.moe.residency import ResidencyCacheConfig
from b12x.testing.residency_replay import (
    LayerTrace,
    RouteCall,
    locality,
    read_trace,
    replay,
    retrospective_cost,
)


def call(ids, *, request="r", split="test", workload="code"):
    return RouteCall(request, workload, split, tuple(tuple(row) for row in ids))


def trace(rows):
    return LayerTrace("layer", 4, 128, 128, "checkpoint", "decode", tuple(rows))


def config(**kwargs):
    return ResidencyCacheConfig(
        **(
            dict(
                max_pairs=1,
                minimum_cold_selections=2,
                minimum_score_gain=2,
                minimum_residency_windows=0,
            )
            | kwargs
        )
    )


def test_duplicate_selection_is_not_future_invocation_reuse():
    calls = [call([[0, 0, 1]]), call([[2, 0, 0]]), call([[1, 3, 3]])]
    result = locality(calls, 4, horizons=(1, 2, 4), budgets=(1, 2))
    assert result["selection_counts"] == [4, 2, 1, 2]
    assert result["unique_per_invocation"]["mean"] == 2
    assert result["interarrival_invocations"]["n"] == 2
    assert result["horizons"][1]["eligible_touches"] == 4
    assert result["horizons"][1]["reused"] == 1
    assert result["horizons"][1]["later_route_selections"] == 2
    assert result["horizons"][2]["reused"] == 2
    assert result["horizons"][4]["p_reused"] is None
    assert result["horizons"][4]["censored_touches"] == 6


def test_request_boundaries_censor_and_do_not_invent_reuse():
    rows = [call([[0]], request="a"), call([[0]], request="b")]
    assert (
        locality(rows, 4, horizons=(1,), budgets=(1,))["horizons"][1][
            "eligible_touches"
        ]
        == 0
    )
    t = trace([call([[1]], split="train", request="train"), *rows])
    r = replay(t, budget=1, window=1, policy="static")
    assert all(m["right_censored"] for m in r["misses"])


def test_held_out_initial_population_and_current_miss_stays_cold():
    t = trace(
        [
            call([[0, 0]], split="train", request="train"),
            *[call([[2, 2]]) for _ in range(4)],
        ]
    )
    r = replay(t, budget=1, window=1, config=config())
    assert r["totals"]["cold_selections"] == 2
    assert r["totals"]["static_cold_selections"] == 8
    assert r["promotions"][0]["useful_hits"] == 6
    assert r["promotions"][0]["promoted_at"] == 1
    assert r["movement_pairs"] == 1
    assert r["misses"][0]["horizons"][1] == dict(touches=1, selections=2)
    assert r["promotions"][0]["censored_at_end"]
    assert r["wasted_completed"] == 0
    cost = retrospective_cost(r, promotion_us=5, cold_saving_us_per_selection=1)
    assert cost["net_vs_static_us"] == 1
    assert cost["promotions"][0]["invocations_to_break_even"] == 3


def test_rotating_churn_and_eviction_cost_are_visible():
    t = trace(
        [
            call([[0, 0]], split="train", request="train"),
            call([[1, 1]]),
            call([[2, 2]]),
            call([[3, 3]]),
        ]
    )
    r = replay(t, budget=1, window=1, config=config())
    assert r["movement_pairs"] == 2  # Never promote after final observation.
    assert r["wasted_completed"] == 1 and r["zero_hit_censored"] == 1
    assert sum(p["useful_hits"] for p in r["promotions"]) == 0
    assert (
        retrospective_cost(r, promotion_us=5, cold_saving_us_per_selection=1)[
            "net_vs_static_us"
        ]
        == -10
    )


@pytest.mark.parametrize("policy", ["static", "b12x", "lru", "lfu", "decayed_lfu"])
@pytest.mark.parametrize("budget", [0, 1, 4])
def test_policies_are_deterministic_and_obey_capacity(policy, budget):
    t = trace(
        [
            call([[3, 3]], split="train", request="train"),
            *[call([[i % 4, (i + 1) % 4, -1]]) for i in range(9)],
        ]
    )
    a = replay(t, budget=budget, window=2, policy=policy, config=config())
    assert a == replay(t, budget=budget, window=2, policy=policy, config=config())
    assert all(
        len(w["hot_experts"]) == budget and len(w["pairs"]) <= 1 for w in a["windows"]
    )
    if budget in (0, 4):
        assert a["movement_pairs"] == 0
        assert a["totals"]["cold_selections"] == (18 if budget == 0 else 0)


def test_geometry_phase_and_absent_training_fail_closed():
    t = trace([call([[0]])])
    with pytest.raises(ValueError, match="disjoint"):
        replay(t, budget=1, window=1)
    with pytest.raises(ValueError):
        replay(t, budget=5, window=1, initial="positional")
    with pytest.raises(ValueError, match="phase"):
        replay(
            t, budget=1, window=1, initial="positional", config=config(phase="verify")
        )


def test_parser_validates_integrity_of_trace_semantics(tmp_path):
    data = dict(
        schema="b12x-routing-invocations-v1",
        checkpoint="sha256:abc",
        truncated=False,
        layers=[
            dict(
                layer="x",
                experts=4,
                hidden=128,
                intermediate=128,
                phase="decode",
                segments=[
                    dict(
                        request="r",
                        workload="code",
                        split="test",
                        calls=[dict(ids=[[0, 1, -1]])],
                    )
                ],
            )
        ],
    )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(data))
    assert read_trace(path)[0].calls[0].ids == ((0, 1, -1),)
    for bad in (2**40, 4, -2, True):
        data["layers"][0]["segments"][0]["calls"][0]["ids"] = [[bad]]
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="expert ID"):
            read_trace(path)
    data["truncated"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="untruncated"):
        read_trace(path)


def test_layers_have_independent_priors():
    a = trace([call([[0]], split="train", request="train"), call([[0]])])
    b = replace(
        a,
        layer="other",
        calls=(call([[1]], split="train", request="train"), call([[0]])),
    )
    assert (
        replay(a, budget=1, window=1, policy="static")["totals"]["cold_selections"] == 0
    )
    assert (
        replay(b, budget=1, window=1, policy="static")["totals"]["cold_selections"] == 1
    )


def test_native_export_import_excludes_prompt_and_unprocessed_tail(tmp_path):
    import base64
    import hashlib
    import io
    import numpy as np
    from scripts.import_vllm_expert_trace import convert

    array = np.array([[[0, 1]], [[1, 2]], [[2, 3]], [[3, 0]]], dtype=np.uint16)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    response = dict(
        usage=dict(prompt_tokens=2, completion_tokens=3),
        choices=[dict(routed_experts=base64.b64encode(buffer.getvalue()).decode())],
    )
    raw = json.dumps(response).encode()
    (tmp_path / "response.json").write_bytes(raw)
    manifest = dict(
        execution=dict(concurrency=1, speculative=False),
        experts=4,
        hidden=128,
        intermediate=128,
        layers=1,
        checkpoint="test",
        requests=[
            dict(
                response="response.json",
                sha256=hashlib.sha256(raw).hexdigest(),
                request="r",
                workload="code",
                split="test",
            )
        ],
    )
    result = convert(manifest, tmp_path)
    assert result["layers"][0]["segments"][0]["calls"] == [
        dict(ids=[[2, 3]]),
        dict(ids=[[3, 0]]),
    ]
    manifest["execution"]["speculative"] = True
    with pytest.raises(ValueError, match="nonspeculative"):
        convert(manifest, tmp_path)
    manifest["execution"]["speculative"] = False
    manifest["requests"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="integrity"):
        convert(manifest, tmp_path)


def test_precomputed_reuse_preserves_every_replay_diagnostic():
    from b12x.testing.residency_replay import reuse_diagnostics

    t = trace(
        [
            call([[0, 0]], split="train", request="train"),
            *[call([[i % 3, i % 3]]) for i in range(8)],
        ]
    )
    future = reuse_diagnostics([c for c in t.calls if c.split == "test"], (1, 2))
    assert replay(t, budget=1, window=1, horizons=(1, 2), config=config()) == replay(
        t, budget=1, window=1, horizons=(1, 2), config=config(), future_use=future
    )


def test_lfu_unseen_expert_has_zero_frequency_for_score_margin():
    t = trace([call([[0]], split="train", request="train"), call([[1]]), call([[2]])])
    r = replay(
        t,
        budget=1,
        window=1,
        policy="lfu",
        config=config(minimum_cold_selections=1, minimum_score_gain=2),
    )
    assert r["movement_pairs"] == 0
