"""Capacity projection preserves calibration evidence and serving reservations."""

from dataclasses import replace

import pytest

from benchmarks.moe.expert_cache_capacity import (
    allocate,
    calibration_counts,
    maximum_envelope,
    plan,
    runtime_reservations,
)
from b12x.integration.vllm.expert_cache import digest
from b12x.moe.fused_moe._cache_preparation import ExpertCacheMemory, ExpertCacheQuery
from b12x.moe.fused_moe.residency import profile_from_counts


def fixture():
    counts = {"layer": (7, 1, 9, 3)}
    placement = profile_from_counts(
        counts=counts["layer"],
        hot_count=2,
        layer="layer",
        model_fingerprint="a" * 64,
        workload="calibration",
        provenance="test",
        phase="decode",
    )
    profile = dict(
        identity=dict(
            checkpoint="a" * 64,
            workload="calibration",
            top_k=4,
            recipe="nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum",
            layers={
                "layer": dict(
                    num_experts=4,
                    hidden_size=128,
                    intermediate_size=128,
                    w13_layout="w31",
                )
            },
        ),
        placements={"layer": placement.to_dict()},
    )
    profile["hash"] = digest(profile)
    receipt = [
        dict(
            kind="profile",
            result=[
                dict(
                    hash=profile["hash"],
                    snapshot=dict(layers=[dict(layer="layer", counts=[7, 1, 9, 3])]),
                )
            ],
        ),
        dict(kind="complete"),
    ]
    return profile, receipt, counts


def test_calibration_counts_require_receipt_and_identity():
    profile, receipt, expected = fixture()
    assert calibration_counts(profile, receipt) == expected
    receipt[0]["result"][0]["snapshot"]["layers"][0]["counts"][0] += 1
    with pytest.raises(ValueError, match="counts mismatch"):
        calibration_counts(profile, receipt)
    profile["identity"]["checkpoint"] = "b" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        calibration_counts(profile, receipt)


@pytest.mark.parametrize("world", [2, 3, 4])
def test_capacity_uses_agreed_tp_calibration_without_summing_replicas(world):
    from copy import deepcopy

    profile, receipt, expected = fixture()
    profile["identity"]["tp_size"] = world
    profile["hash"] = digest({k: v for k, v in profile.items() if k != "hash"})
    owner = receipt[0]["result"][0]
    owner["hash"] = profile["hash"]
    receipt[0]["result"] = []
    for rank in reversed(range(world)):
        result = deepcopy(owner)
        result["snapshot"]["rank"] = rank
        receipt[0]["result"].append(result)
    assert calibration_counts(profile, receipt) == expected
    receipt[0]["result"][0]["snapshot"]["layers"][0]["counts"][0] += 1
    with pytest.raises(ValueError, match="disagree"):
        calibration_counts(profile, receipt)


def test_fair_capacity_includes_workspace_and_never_exceeds_experts():
    q = ExpertCacheQuery(
        experts=4,
        resident=1,
        hidden=128,
        intermediate=128,
        max_tokens=4,
        top_k=4,
        w13_layout="w31",
        checkpoint_fingerprint="a" * 64,
        profile_hash="b" * 64,
    )

    def memory(q, count):
        return ExpertCacheMemory(count * 10, q.experts * 10, 0, 100, 4, 8)

    queries = {"a": q, "b": replace(q, experts=3)}
    with pytest.raises(ValueError, match="one slot"):
        allocate(queries, 232, 5, memory)
    counts, _, used = allocate(queries, 243, 5, memory)
    assert counts == {"a": 2, "b": 1} and used == 243
    counts, _, used = allocate(queries, 1000, 5, memory)
    assert counts == {"a": 4, "b": 3} and used == 283


def test_capacity_projection_keeps_counts_and_noncache_reservations(monkeypatch):
    import benchmarks.moe.expert_cache_capacity as module

    profile, _, counts = fixture()
    monkeypatch.setattr(
        module,
        "memory_for",
        lambda q, d, s: ExpertCacheMemory(
            q.resident * 10000,
            q.experts * 10000,
            s,
            q.max_tokens * 250,
            64,
            32 if q.max_pairs else 0,
        ),
    )
    baseline = dict(
        device_capacity=1000000,
        host_capacity=1000000,
        resident_experts=20000,
        backing_experts=40000,
        dense_model=10000,
        kv=20000,
        workspace=1000,
        graphs=5000,
        metadata=64,
        host_staging=32,
        host_sources=40000,
        device_safety=7000,
        host_safety=8000,
        other_device=3000,
    )
    row, placements = plan(profile, counts, baseline, 100000, device="cpu", capacity=4)
    assert row["all_resident"] and not row["adaptive_supported"]
    assert row["resident_payload_fraction"] == 1
    assert tuple(placements["layer"]["selection_counts"]) == counts["layer"]
    for key in (
        "dense_model",
        "kv",
        "graphs",
        "host_sources",
        "device_safety",
        "other_device",
    ):
        assert row["memory"][key] == baseline[key]
    wider, wider_placements = plan(
        profile, counts, baseline, 100000, device="cpu", capacity=256
    )
    assert wider["counts"]["layer"] < row["counts"]["layer"]
    assert wider["memory"]["workspace"] > row["memory"]["workspace"]
    assert wider["memory"]["kv"] == row["memory"]["kv"]
    assert wider["memory"]["device_safety"] == row["memory"]["device_safety"]
    assert tuple(wider_placements["layer"]["selection_counts"]) == counts["layer"]
    with pytest.raises(ValueError, match="reservations exceed"):
        plan(
            profile,
            counts,
            dict(baseline, device_capacity=1),
            100000,
            device="cpu",
            capacity=4,
        )


def test_runtime_reservation_counts_only_unaccounted_storage():
    memory = dict(
        device_capacity=1000,
        host_capacity=1000,
        resident_experts=100,
        backing_experts=200,
        dense_model=100,
        kv=100,
        workspace=100,
        graphs=100,
        metadata=0,
        host_staging=0,
        device_safety=100,
        host_safety=100,
        other_device=50,
    )
    reference = [
        dict(
            kind="resources",
            result=[dict(stage="graphs_ready", device_total=1000, device_free=400)],
        )
    ]
    adjusted, accounting = runtime_reservations(memory, reference)
    assert adjusted["other_device"] == 100
    assert adjusted["device_safety"] == 100
    assert accounting["additional_engine_reservation_bytes"] == 50
    # Dense, KV, graphs, engine storage and safety remain reserved. Only the
    # expert envelope can consume the remaining bytes, including non-GiB tails.
    assert maximum_envelope(adjusted) == 500
    assert maximum_envelope(dict(adjusted, device_capacity=1001)) == 501
    assert maximum_envelope(dict(adjusted, device_safety=150)) == 450
    twice, accounting = runtime_reservations(adjusted, reference)
    assert twice == adjusted and not accounting["additional_engine_reservation_bytes"]
    with pytest.raises(ValueError, match="resource checkpoints"):
        runtime_reservations(memory, [])


def test_static_route_replay_scores_canonical_ids_and_rejects_reset_or_movement():
    from benchmarks.moe.summarize_expert_cache_capacity import replay_static_cold

    profile, _, _ = fixture()
    records = [
        dict(
            kind="configuration",
            arguments=dict(control="observe", admission="together"),
            numerical_recipe="cache_w4a16_bf16_whole_k",
        ),
        dict(kind="prepared", status=dict(checkpoint="a" * 64)),
        dict(
            kind="routing_boundary",
            next_request=0,
            receipt=[
                dict(
                    slots={"layer": dict(generation=0)},
                    snapshot=dict(layers=[dict(layer="layer", counts=[0, 0, 0, 0])]),
                )
            ],
        ),
        dict(kind="request", index=0, workload="code"),
        dict(
            kind="routing_boundary",
            next_request=1,
            receipt=[
                dict(
                    slots={"layer": dict(generation=0)},
                    snapshot=dict(layers=[dict(layer="layer", counts=[10, 5, 0, 1])]),
                )
            ],
        ),
        dict(kind="complete"),
    ]
    result = replay_static_cold(profile, records)
    assert result["selections"] == 16 and result["cold"] == 6
    assert result["windows"][0]["workloads"] == ["code"]
    records[-2]["receipt"][0]["slots"]["layer"]["generation"] = 1
    with pytest.raises(ValueError, match="placement changed"):
        replay_static_cold(profile, records)
    records[-2]["receipt"][0]["slots"]["layer"]["generation"] = 0
    records[2]["receipt"][0]["snapshot"]["layers"][0]["counts"][0] = 11
    with pytest.raises(ValueError, match="counter reset"):
        replay_static_cold(profile, records)
