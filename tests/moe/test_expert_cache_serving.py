"""Model admission and checkpoint-bound learned cache artifacts."""

from dataclasses import replace
import json

import pytest
import torch

from b12x.integration.vllm.expert_cache import (
    ExpertCacheModel,
    ExpertCacheServingConfig,
    digest,
)
from b12x.moe.fused_moe._cache_preparation import ExpertCacheMemory
from b12x.moe.fused_moe.execution import ExecutionCapacity
from b12x.moe.fused_moe.residency import profile_from_counts
from tests.moe.test_prepared_expert_cache import source


def test_experimental_check_cadence_backs_off_only_on_observed_health():
    from benchmarks.moe.expert_cache_serving import maintenance_check_interval

    interval, history = 32, []
    for health in ("healthy", "healthy", "healthy", "healthy", "pressure", None):
        interval = maintenance_check_interval(
            interval, minimum=32, maximum=256, health=health
        )
        history.append(interval)
    assert history == [64, 128, 256, 256, 32, 32]
    with pytest.raises(ValueError, match="ordered"):
        maintenance_check_interval(16, minimum=32, maximum=256, health="healthy")


def config(tmp_path, **changes):
    return ExpertCacheServingConfig(
        **(
            dict(
                mode="profile",
                activation="w4a16",
                profile_path=str(tmp_path / "profile.json"),
                workload="general",
                expert_device_bytes=150000,
                host_bytes=1 << 30,
                kv_reserved_bytes=100,
                graph_reserved_bytes=100,
                device_safety_bytes=100,
                host_safety_bytes=100,
            )
            | changes
        )
    )


def model(tmp_path, monkeypatch, **changes):
    import b12x.integration.vllm.expert_cache as module

    def memory(q, device, source_bytes):
        return ExpertCacheMemory(
            q.resident * 28000,
            q.experts * 28000,
            source_bytes,
            100,
            100,
            64 if q.max_pairs else 0,
        )

    monkeypatch.setattr(module, "memory_for", memory)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (900000, 1000000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 80000)
    value = ExpertCacheModel(config(tmp_path, **changes), "a" * 64, "cuda:0")
    a = source()
    value.add_source(a)
    value.add_source(replace(a, weights=replace(a.weights, layer_name="second")))
    return value


def test_model_admission_counts_existing_device_use_once_and_bootstraps_fairly(
    tmp_path, monkeypatch
):
    value = model(tmp_path, monkeypatch)
    value.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert value.memory.dense_model == 80000 and value.memory.other_device == 20000
    counts = [len(p.hbm_expert_ids) for p in value.placements.values()]
    assert max(counts) - min(counts) <= 1
    assert value.memory.host_sources == sum(
        s.source_bytes for s in value.sources.values()
    )
    assert value.memory.device_bytes <= value.memory.device_capacity
    assert value.counter is not None


def write_profile(value):
    value.capacity = ExecutionCapacity(max_tokens=4, top_k=4)
    placements = {
        name: profile_from_counts(
            counts=(1, 2, 20, 30),
            hot_count=2,
            layer=name,
            model_fingerprint="a" * 64,
            workload="general",
            provenance="held-out calibration",
            phase="decode",
        ).to_dict()
        for name in value.sources
    }
    payload = {
        "identity": value._identity(),
        "placements": placements,
        "termination": "explicit_calibration_boundary",
        "converged": False,
    }
    payload["hash"] = digest(payload)
    from pathlib import Path

    Path(value.config.profile_path).write_text(json.dumps(payload))
    return payload


def test_static_and_adaptive_start_from_identical_learned_profile(
    tmp_path, monkeypatch
):
    static = model(tmp_path, monkeypatch, mode="static")
    write_profile(static)
    adaptive = model(tmp_path, monkeypatch, mode="adaptive")
    capacity = ExecutionCapacity(max_tokens=4, top_k=4)
    static.declare(capacity)
    adaptive.declare(capacity)
    assert static.placements == adaptive.placements
    assert static.counter is None and adaptive.counter is not None
    assert all(p.query.max_pairs == 0 for p in static.plans.values())
    assert all(p.query.max_pairs == 2 for p in adaptive.plans.values())


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint", "b" * 64), ("recipe", "a4"), ("workload", "code"), ("version", 2)],
)
def test_profile_identity_is_validated_even_with_recomputed_hash(
    tmp_path, monkeypatch, field, value
):
    m = model(tmp_path, monkeypatch, mode="static")
    payload = write_profile(m)
    payload.pop("hash")
    payload["identity"][field] = value
    payload["hash"] = digest(payload)
    from pathlib import Path

    Path(m.config.profile_path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))


def test_failed_model_admission_cannot_be_retried_as_prepared(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch, host_bytes=1)
    with pytest.raises(ValueError, match="capacity"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    with pytest.raises(RuntimeError, match="admission failed"):
        m.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    with pytest.raises(RuntimeError, match="admission"):
        m.requests()


def test_source_identity_and_preallocation_host_admission(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        m.add_source(source())
    with pytest.raises(ValueError, match="host envelope"):
        m.reserve_source(m.config.host_bytes)


def test_cpu_source_owner_bytes_include_original_scale_storage():
    s = source()
    owner = torch.empty((4, 2), dtype=torch.float32)
    retained = replace(s, owners=(owner, s.weights.w13))
    assert (
        retained.source_bytes == s.source_bytes + owner.numel() * owner.element_size()
    )


def test_routing_top_k_is_part_of_profile_identity(tmp_path, monkeypatch):
    value = model(tmp_path, monkeypatch, mode="static")
    write_profile(value)
    with pytest.raises(ValueError, match="mismatch"):
        value.declare(ExecutionCapacity(max_tokens=4, top_k=2))


def test_adaptive_omits_fully_resident_layers_from_observations(tmp_path, monkeypatch):
    from pathlib import Path

    value = model(tmp_path, monkeypatch, mode="adaptive", expert_device_bytes=300000)
    payload = write_profile(value)
    payload["placements"]["second"] = profile_from_counts(
        counts=(1, 2, 20, 30),
        hot_count=4,
        layer="second",
        model_fingerprint="a" * 64,
        workload="general",
        provenance="fully resident learned control",
        phase="decode",
    ).to_dict()
    payload.pop("hash")
    payload["hash"] = digest(payload)
    Path(value.config.profile_path).write_text(json.dumps(payload))
    value.declare(ExecutionCapacity(max_tokens=4, top_k=4))
    assert value.observed_layers == ("layer",)
    assert value.plans["second"].query.max_pairs == 0
    assert value.counter.query.layers == (("layer", 4),)
