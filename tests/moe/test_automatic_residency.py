"""Host oracles for model-wide placement, profile reuse and calibration state."""
from dataclasses import replace
import json

import pytest
import torch

from b12x.moe.fused_moe.automatic import (
    ALGORITHM, AutomaticResidencyConfig, LayerRoutingCounts, ModelExpertMemoryBudget,
    ResidencyCalibrationConfig, ResidencyController, ResidencyHardware, ResidencyLayerSpec,
    ResidencyModelSpec, ResidencyMonitorConfig, ResidencyProfile, ResidencyProfileStore,
    RoutingSnapshot, derive_placement, placement_memory,
)
from b12x.moe.fused_moe.routing_profile import RoutingProfileQuery, plan_routing_profile


def model(*, varying=False):
    return ResidencyModelSpec(checkpoint_fingerprint="sha256:model", layers=tuple(
        ResidencyLayerSpec(layer=n, experts=4, hidden=256, intermediate=512 if varying and n == "b" else 256,
            max_tokens=8, max_top_k=3) for n in ("a", "b")))


def budget(m, hot=2, **kwargs):
    overhead = sum(s.memory(0).scratch_bytes+s.memory(0).route_map_bytes for s in m.layers)
    return ModelExpertMemoryBudget(hbm_bytes=overhead+hot*m.layers[0].expert_bytes, grace_bytes=2**30, **kwargs)


def counts(a=(90, 5, 4, 1), b=(1, 4, 5, 90), *, factor=1, phase="decode", rank=0, epoch=1):
    return RoutingSnapshot(epoch=epoch, rank=rank, layers=tuple(LayerRoutingCounts(layer=layer,
        phase=phase, counts=tuple(n*factor for n in row)) for layer, row in (("a", a), ("b", b))))


def controller(tmp_path, *, mode="auto", m=None, cfg=None, **options):
    m = m or model()
    c = AutomaticResidencyConfig(mode=mode, calibration=cfg or ResidencyCalibrationConfig(
        minimum_observations=100, convergence_window=100, stable_windows=2), **options)
    return ResidencyController(config=c, model=m, budget=replace(budget(m), hbm_bytes=budget(m).hbm_bytes+1024),
        hardware=ResidencyHardware(compute_capability=(10, 3), grace_coherent=True),
        store=ResidencyProfileStore(tmp_path))


def completed(tmp_path, **kwargs):
    c = controller(tmp_path, **kwargs)
    assert c.startup().state == "calibrating"
    for n in range(1, 4):
        p = c.observe(counts(factor=n), request_count=n, token_count=10*n)
    return c, p.profile


def test_rank_per_layer_ties_and_global_allocation():
    m = model()
    p = derive_placement(m, budget(m), {"a": (10, 10, 0, 0), "b": (0, 9, 0, 0)},
        workload="code", provenance="test", phase="decode")
    assert p[0].hbm_expert_ids == (0, 1) and p[1].hbm_expert_ids == ()
    p = derive_placement(m, budget(m), {"a": (90, 5, 4, 1), "b": (1, 4, 5, 90)},
        workload="code", provenance="test", phase="decode")
    assert p[0].hbm_expert_ids == (0,) and p[1].hbm_expert_ids == (3,)
    assert sum(x.hbm_expert_bytes for s, p in zip(m.layers, p) for x in [s.memory(len(p.hbm_expert_ids))]) == 2*m.layers[0].expert_bytes


def test_varying_sizes_use_selection_density():
    m = model(varying=True)
    p = derive_placement(m, budget(m, hot=2), {"a": (10, 10, 0, 0), "b": (0, 19, 0, 0)},
        workload="code", provenance="test", phase="decode")
    assert p[0].hbm_expert_ids == (0, 1) and p[1].hbm_expert_ids == ()


def test_memory_reservations_charged_once():
    m = model()
    b = budget(m, hot=3)
    reserve = m.layers[0].expert_bytes
    b = replace(b, non_moe_hbm_bytes=reserve//4, kv_reserved_bytes=reserve//4,
        hbm_safety_bytes=reserve//4, other_hbm_bytes=reserve//8, profiling_bytes=reserve//8)
    assert b.expert_capacity(m)[0] == 2*reserve
    p = derive_placement(m, b, {"a": (9, 1, 0, 0), "b": (0, 0, 1, 9)},
        workload="code", provenance="test", phase="decode")
    assert placement_memory(m, p)["hbm_expert_bytes"] == 2*reserve


@pytest.mark.parametrize("change", ["scratch", "grace", "minimum", "zero", "geometry"])
def test_impossible_placement_fails_closed(change):
    m, values = model(), {"a": (9, 1, 0, 0), "b": (0, 0, 1, 9)}
    b = budget(m)
    if change == "scratch": b = replace(b, hbm_bytes=0)
    if change == "grace": b = replace(b, grace_bytes=0)
    if change == "minimum": m = replace(m, layers=tuple(replace(s, minimum_hot=3) for s in m.layers))
    if change == "zero": values["a"] = (0, 0, 0, 0)
    if change == "geometry": values["a"] = (0, 1)
    with pytest.raises(ValueError):
        derive_placement(m, b, values, workload="agent", provenance="test", phase="decode")


def test_converges_and_requires_restart(tmp_path):
    c, p = completed(tmp_path)
    assert c.progress.state == "restart_required" and p.converged and p.stable_windows == 2
    assert p.expected_cold_fraction == .1
    assert c.active is None
    assert ResidencyProfile.from_dict(p.to_dict()) == p
    for s, row in zip(c.model.layers, p.placements):
        p.layer_budget(s.layer).admit(s.memory(len(row.hbm_expert_ids)))
    resumed = controller(tmp_path)
    assert resumed.startup().state == "ready" and resumed.profiler_query() is None
    assert resumed.active == p


def test_profile_mode_never_activates_and_cache_disable(tmp_path):
    c, p = completed(tmp_path, mode="profile")
    assert c.progress.state == "profile_saved" and c.active is None
    assert controller(tmp_path, reuse_cache=False).startup().state == "calibrating"
    assert controller(tmp_path, mode="profile").startup().state == "calibrating"
    assert controller(tmp_path, workload="math").startup().state == "calibrating"
    pin = controller(tmp_path, profile_path=c.progress.profile_path)
    assert pin.startup().profile == p


def test_off_does_not_read_store_or_probe(monkeypatch, tmp_path):
    c = controller(tmp_path, mode="off")
    def forbidden(*args, **kwargs): pytest.fail("off mode performed discovery")
    monkeypatch.setattr(c.store, "read", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    assert c.startup().state == "off" and c.profiler_query() is None


@pytest.mark.parametrize("mutation", ["checkpoint", "geometry", "recipe", "version", "workload", "hash", "algorithm", "memory", "phase"])
def test_incompatible_or_corrupt_artifact_invalidates(tmp_path, mutation):
    c, p = completed(tmp_path)
    path = c.progress.profile_path
    raw = p.to_dict()
    if mutation == "checkpoint": raw["model"]["checkpoint_fingerprint"] = "other"
    if mutation == "geometry": raw["model"]["layers"][0]["hidden"] = 512
    if mutation == "recipe": raw["model"]["layers"][0]["source_format"] = "nvfp4"
    if mutation == "version": raw["schema_version"] += 1
    if mutation == "workload": raw["workload"] = "math"
    if mutation == "hash": raw["profile_hash"] = "0"*64
    if mutation == "algorithm": raw["algorithm"] = "online_cache"
    if mutation == "memory": raw["memory"]["hbm_expert_bytes"] = 0
    if mutation == "phase": raw["phase"] = "draft"
    from pathlib import Path
    Path(path).write_text(json.dumps(raw))
    c = controller(tmp_path)
    assert c.startup().state == "calibrating"
    assert "invalid" in c.progress.reason


def test_validation_checks_current_memory_hardware_and_identity(tmp_path):
    c, p = completed(tmp_path)
    kwargs = dict(model=c.model, config=c.config, budget=c.budget, hardware=c.hardware)
    for field, value in (("model", replace(c.model, tokenizer_revision="other")),
            ("config", replace(c.config, workload="math")),
            ("budget", budget(c.model, hot=1)),
            ("hardware", replace(c.hardware, grace_coherent=False)),
            ("hardware", replace(c.hardware, compute_capability=(12, 0)))):
        with pytest.raises(ValueError): p.validate(**(kwargs | {field: value}))


def test_changed_distribution_does_not_converge(tmp_path):
    c = controller(tmp_path)
    c.startup()
    a = b = (0, 0, 0, 0)
    for n in range(8):
        x = (90, 5, 4, 1) if n % 2 else (1, 4, 5, 90)
        a = tuple(t+v for t, v in zip(a, x))
        b = tuple(t+v for t, v in zip(b, reversed(x)))
        assert c.observe(counts(a, b)).state == "calibrating"
    assert c.progress.stable_windows == 0


def test_minor_noise_converges_but_cold_rate_changes_do_not(tmp_path):
    cfg = ResidencyCalibrationConfig(minimum_observations=100, convergence_window=100,
        stable_windows=2, cold_fraction_delta=.025)
    c = controller(tmp_path, cfg=cfg)
    c.startup()
    c.observe(counts())
    c.observe(counts((181, 9, 8, 2), (2, 8, 9, 181)))
    assert c.observe(counts(factor=3)).state == "restart_required"
    c = controller(tmp_path/"unstable", cfg=cfg)
    c.startup()
    c.observe(counts())
    c.observe(counts((141, 35, 18, 6), (6, 18, 35, 141)))
    assert c.progress.stable_windows == 0


@pytest.mark.parametrize("phase", ["decode", "all"])
def test_insufficient_routes_and_missing_decode_do_not_converge(tmp_path, phase):
    cfg = ResidencyCalibrationConfig(phase=phase, minimum_observations=100,
        convergence_window=100, stable_windows=1, request_limit=3)
    c = controller(tmp_path, cfg=cfg)
    c.startup()
    assert c.observe(counts(factor=100, phase="prefill"), request_count=3).state == "insufficient"
    assert not list(tmp_path.rglob("*.json"))


def test_limit_persists_nonconverged_with_label(tmp_path):
    c = controller(tmp_path, cfg=ResidencyCalibrationConfig(minimum_observations=100,
        convergence_window=100, stable_windows=4, token_limit=10))
    c.startup()
    p = c.observe(counts(), token_count=10)
    assert p.state == "restart_required" and not p.profile.converged and p.profile.termination == "limit"


def test_tp_owner_and_epoch_guard(tmp_path):
    c = controller(tmp_path)
    c.startup()
    q = c.profiler_query(rank=1, tp_size=2)
    assert q.storage_bytes == 0
    with pytest.raises(ValueError, match="authoritative"): c.observe(counts(rank=1))
    c.observe(counts())
    with pytest.raises(ValueError, match="epoch"): c.observe(counts(epoch=2))
    with pytest.raises(ValueError, match="decreased"): c.observe(counts(factor=0))
    with pytest.raises(ValueError, match="expert-parallel"):
        replace(q, expert_parallel=True)


def test_monitor_reports_drift_without_mutation(tmp_path):
    _, profile = completed(tmp_path)
    c = controller(tmp_path, mode="monitor", monitor=ResidencyMonitorConfig(minimum_observations=100, drift_threshold=.1))
    assert c.startup().state == "monitoring"
    d = c.monitor(counts((1, 4, 5, 90), (90, 5, 4, 1)))
    assert d.recommended and d.active_cold_fraction == .99 and d.candidate_cold_fraction == .1
    assert d.membership_changed_fraction == 1 and d.bytes_moved == 4*c.model.layers[0].expert_bytes
    assert c.active == profile and c.progress.state == "monitoring"
    with pytest.raises(ValueError, match="valid static profile"):
        controller(tmp_path/"missing", mode="monitor").startup()


def test_profile_store_separates_workloads_and_preserves_content(tmp_path):
    c, p = completed(tmp_path, workload="../agent")
    c2, p2 = completed(tmp_path, workload="math")
    assert p.profile_hash != p2.profile_hash
    assert len(list(tmp_path.rglob("current.json"))) == 2
    assert c.store.read(model=c.model, config=c.config) == p


def test_profiler_declaration_is_pure_registered_and_capacity_only(monkeypatch):
    def forbidden(*a, **k): pytest.fail("declaration touched CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    from b12x.preparation.catalog import get_tuning_contract
    q = RoutingProfileQuery(layers=(("layer", 384),), max_tokens=128, max_top_k=8)
    p = plan_routing_profile(q)
    assert p.prepared is None
    assert p.contract is get_tuning_contract("moe.fused_moe", variant="routing_profile")
    assert q.storage_bytes == 3136
    assert "live" not in p.contract.query_fields


@pytest.mark.parametrize("kwargs", [{"sample_every": 0}, {"max_tokens": 0}, {"rank": 1},
    {"phases": ("unknown",)}, {"layers": (("a", 0),)}, {"max_tokens": 2**31}])
def test_profiler_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        RoutingProfileQuery(**({"layers": (("a", 4),), "max_tokens": 128, "max_top_k": 8} | kwargs))


def test_capacity_accounting_matches_real_geometry():
    from b12x.moe.fused_moe._residency_storage import tier_layout
    for h, i, e in ((256, 256, 4), (5120, 2304, 384), (4096, 1536, 96)):
        s = ResidencyLayerSpec(layer="x", experts=e, hidden=h, intermediate=i, max_tokens=128, max_top_k=6)
        for n in (0, 1, e//2, e):
            assert s.memory(n).hbm_expert_bytes == n*s.expert_bytes == tier_layout(n, h, i)[1]


def test_auto_bootstrap_and_weight_plan_bridge_are_pure(tmp_path, monkeypatch):
    from b12x.moe import fused_moe as moe
    wp = moe.plan_weights(source=moe.PackedSource(format="fp4_e8m0_k32"),
        activation=moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=4, hidden_size=256, intermediate_size=256),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"))
    cap = moe.ExecutionCapacity(max_tokens=8, top_k=3)
    assert ResidencyLayerSpec.from_weight_plan(layer="a", weight_plan=wp, capacity=cap) == model().layers[0]
    c = controller(tmp_path)
    c.startup()
    weights = moe.PackedWeights(torch.zeros(4, 512, 128, dtype=torch.uint8), torch.zeros(4, 256, 128, dtype=torch.uint8),
        torch.zeros(4, 512, 8, dtype=torch.uint8), torch.zeros(4, 256, 8, dtype=torch.uint8), torch.ones(4), torch.ones(4),
        checkpoint_fingerprint="sha256:model", layer_name="a")
    def forbidden(*a, **k): pytest.fail("declaration touched CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    plan = c.plan_execution(layer="a", weight_plan=wp, weights=weights)
    assert plan.prepared is None and plan.component_id == "moe.expert_residency"
    assert len(c.placements()[0].hbm_expert_ids) == 2
    assert not list(tmp_path.rglob("*.json"))


def test_worker_off_never_installs_observer(tmp_path):
    from b12x.integration.vllm.expert_residency import ExpertResidencyWorker
    worker = ExpertResidencyWorker(controller(tmp_path, mode="off"))
    assert worker.startup().state == "off"
    assert worker.preparation_request() is None
    assert worker.bind_routes(layer="a", phase="decode", topk_ids=None) is None
    assert worker.poll() is None


def test_monitor_uses_disjoint_windows(tmp_path):
    completed(tmp_path)
    c = controller(tmp_path, mode="monitor", monitor=ResidencyMonitorConfig(minimum_observations=100))
    c.startup()
    assert not c.monitor(counts(factor=100)).recommended
    assert c.monitor(counts(factor=100)) is None
    d = c.monitor(counts((9001, 504, 405, 190), (190, 405, 504, 9001)))
    assert d.recommended and d.active_cold_fraction == .99


def test_atomic_store_failure_keeps_previous_index(tmp_path, monkeypatch):
    c, p = completed(tmp_path)
    import b12x.moe.fused_moe.automatic as automatic
    replace_file = automatic.os.replace
    def fail_index(source, destination):
        if destination.name == "current.json": raise OSError("injected interrupted index publication")
        return replace_file(source, destination)
    monkeypatch.setattr(automatic.os, "replace", fail_index)
    with pytest.raises(OSError):
        c.store.write(replace(p, created_at="later"))
    assert c.store.read(model=c.model, config=c.config) == p


def test_profile_store_index_cannot_escape_namespace(tmp_path):
    c, _ = completed(tmp_path)
    index = next(tmp_path.rglob("current.json"))
    index.write_text(json.dumps({"schema_version": 2, "profile_hash": "../other"}))
    assert controller(tmp_path).startup().state == "calibrating"


def test_all_phase_requires_decode_and_records_verify_separately(tmp_path):
    cfg = ResidencyCalibrationConfig(phase="all", minimum_observations=100, convergence_window=100, stable_windows=1)
    c = controller(tmp_path, cfg=cfg)
    c.startup()
    assert c.profiler_query().phases == ("decode", "prefill", "verify", "draft")
    for n in (1, 2):
        snap = counts(factor=n)
        snap = replace(snap, layers=snap.layers + counts(factor=n, phase="verify").layers)
        result = c.observe(snap)
    assert result.state == "restart_required" and result.profile.phase == "all"


def test_workspace_estimate_never_discounts_admitted_storage():
    m = model(varying=True)
    estimate = m.workspace_estimate(concurrent_lanes=2)
    assert not estimate["arena_implemented"]
    assert estimate["private_bytes"] == 2*sum(s.memory(0).scratch_bytes for s in m.layers)
    assert estimate["hypothetical_arena_bytes"] == 2*max(s.memory(0).scratch_bytes for s in m.layers)
    b = budget(m)
    assert b.expert_capacity(m)[0] == 2*m.layers[0].expert_bytes


def test_model_artifact_exports_existing_static_plans(tmp_path):
    from b12x.moe.fused_moe.residency import read_profiles
    c, profile = completed(tmp_path)
    assert read_profiles(c.progress.profile_path) == profile.placements


def test_worker_rejects_unknown_phase_and_layer(tmp_path):
    from b12x.integration.vllm.expert_residency import ExpertResidencyWorker
    worker = ExpertResidencyWorker(controller(tmp_path))
    worker.startup()
    with pytest.raises(ValueError, match="phase"):
        worker.bind_routes(layer="a", phase="unknown", topk_ids=None)
    with pytest.raises(ValueError, match="layer"):
        worker.bind_routes(layer="typo", phase="prefill", topk_ids=None)
    assert worker.bind_routes(layer="a", phase="prefill", topk_ids=None) is None
