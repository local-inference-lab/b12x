"""Capacity projection preserves calibration evidence and serving reservations."""

from dataclasses import replace

import pytest

from benchmarks.moe.expert_cache_capacity import allocate, calibration_counts, plan
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
            q.resident * 10000, q.experts * 10000, s, 1000, 64, 32 if q.max_pairs else 0
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
    with pytest.raises(ValueError, match="reservations exceed"):
        plan(
            profile,
            counts,
            dict(baseline, device_capacity=1),
            100000,
            device="cpu",
            capacity=4,
        )
