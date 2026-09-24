"""Host gates for immutable placement, admission, and declaration purity."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import json

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe.residency import (
    ExpertMemoryAccounting, ExpertMemoryBudget, ExpertResidencyPlan,
    profile_from_counts, profiles_from_trace, read_profiles, write_profiles,
)
from b12x.moe.fused_moe._residency_storage import accounting, _swizzle_scale


def placement(hot=(0, 2), cold=(1, 3)):
    return ExpertResidencyPlan(total_experts=4, hbm_expert_ids=hot, grace_expert_ids=cold,
        layer="layer.7", model_fingerprint="sha256:test", workload="agent", provenance="trace:seed7")


def declaration(*, max_tokens=8, top_k=3, profile=None, memory_budget=None, updates=None):
    e, h, i = 4, 256, 256
    weight_plan = moe.plan_weights(source=moe.PackedSource(format="fp4_e8m0_k32"),
        activation=moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"))
    weights = moe.PackedWeights(
        torch.zeros(e, 2*i, h//2, dtype=torch.uint8), torch.zeros(e, h, i//2, dtype=torch.uint8),
        torch.full((e, 2*i, h//32), 127, dtype=torch.uint8),
        torch.full((e, h, i//32), 127, dtype=torch.uint8), torch.ones(e), torch.ones(e),
        checkpoint_fingerprint="sha256:test", layer_name="layer.7")
    plan = moe.plan_execution(experts=weight_plan, weights=weights,
        placement=profile or placement(), memory_budget=memory_budget or ExpertMemoryBudget(hbm_bytes=2**30, grace_bytes=2**30),
        capacity=moe.ExecutionCapacity(max_tokens=max_tokens, top_k=top_k), updates=updates)
    return plan, weights


def test_mapping_is_immutable_and_complete():
    p = placement(hot=[2, 0], cold=[3, 1])
    assert p.expert_map == ((0, 1), (1, 1), (0, 0), (1, 0))
    with pytest.raises(FrozenInstanceError):
        p.layer = "mutated"
    for ids in ((0, 0), (0, 1), (-1, 2), (0, 4), (False, 2)):
        with pytest.raises(ValueError):
            placement(hot=ids)


def test_profile_hash_and_round_trip(tmp_path):
    p = placement()
    assert p.profile_hash != replace(p, workload="multilingual").profile_hash
    path = tmp_path / "profile.json"
    write_profiles(path, (p,))
    assert read_profiles(path) == (p,)
    artifact = json.loads(path.read_text())
    artifact["layers"][0]["workload"] = "tampered"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_profiles(path)
    with pytest.raises(ValueError, match="duplicate"):
        write_profiles(path, (p, p))


def test_trace_profiles_are_per_layer_and_workload():
    records = [
        {"layer": "a", "phase": "decode", "expert_ids": [0, 0, 0, 2, -1]},
        {"layer": "b", "phase": "decode", "expert_ids": [1, 1, 3]},
        {"layer": "a", "phase": "prefill", "expert_ids": [3]*20},
    ]
    kwargs = dict(experts_per_layer={"a": 4, "b": 4}, hot_count=1,
                  model_fingerprint="model", workload="agent", provenance="trace")
    a, b = profiles_from_trace(records, phase="decode", **kwargs)
    assert a.hbm_expert_ids == (0,) and b.hbm_expert_ids == (1,)
    assert a.expected_cold_fraction == .25 and b.expected_cold_fraction == 1/3
    assert profiles_from_trace(records, phase="prefill", **kwargs)[0].hbm_expert_ids == (3,)
    with pytest.raises(ValueError, match="invalid expert"):
        profiles_from_trace([dict(layer="a", phase="decode", expert_ids=[4])], **kwargs)
    p = profile_from_counts(counts=(1, 1, 2, 3), hot_bytes=31, expert_bytes=16,
        layer="a", model_fingerprint="m", workload="w", provenance="t")
    assert p.hbm_expert_ids == (3,)


def test_budget_includes_scratch_map_kv_and_reserves():
    memory = ExpertMemoryAccounting(hbm_expert_bytes=100, grace_expert_bytes=200, scratch_bytes=30, route_map_bytes=10)
    ExpertMemoryBudget(hbm_bytes=200, grace_bytes=210, hbm_safety_bytes=10, kv_reserved_bytes=50, grace_safety_bytes=10).admit(memory)
    with pytest.raises(ValueError, match="HBM budget"):
        ExpertMemoryBudget(hbm_bytes=199, grace_bytes=210, hbm_safety_bytes=10, kv_reserved_bytes=50).admit(memory)
    with pytest.raises(ValueError, match="Grace budget"):
        ExpertMemoryBudget(hbm_bytes=200, grace_bytes=199).admit(memory)


def test_declaration_is_pure_and_registered(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("declaration initialized CUDA or compiled a program")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    from b12x._lib import compiler
    monkeypatch.setattr(compiler, "compile", forbidden)
    plan, _ = declaration()
    assert plan.prepared is None
    assert plan.query.max_tokens == 8 and plan.query.max_top_k == 3
    from b12x.preparation.catalog import get_tuning_contract
    assert plan.contract is get_tuning_contract("moe.fused_moe", variant="residency")
    assert accounting(plan.query).hbm_expert_bytes == accounting(plan.query).grace_expert_bytes
    with pytest.raises(RuntimeError, match="not prepared"):
        moe.bind(plan)
    with pytest.raises(ValueError, match="HBM budget"):
        declaration(memory_budget=ExpertMemoryBudget(hbm_bytes=1, grace_bytes=2**30))


def test_swizzle_preserves_every_source_scale():
    source = torch.arange(256*16).remainder(255).to(torch.uint8).view(256, 16)
    actual = _swizzle_scale(source).flatten()
    for row in range(256):
        for column in range(16):
            offset = (row//128)*128*16 + (column//4)*512 + (row%32)*16 + ((row%128)//32)*4 + column%4
            assert actual[offset] == source[row, column]


def test_all_hot_all_cold_and_capacity_accounting():
    hot, _ = declaration(profile=placement(hot=(0, 1, 2, 3), cold=()))
    cold, _ = declaration(profile=placement(hot=(), cold=(0, 1, 2, 3)))
    assert accounting(hot.query).grace_expert_bytes == 0
    assert accounting(cold.query).hbm_expert_bytes == 0
    assert accounting(hot.query).hbm_expert_bytes == accounting(cold.query).grace_expert_bytes
    small, _ = declaration(max_tokens=1, top_k=1)
    assert accounting(small.query).scratch_bytes < accounting(hot.query).scratch_bytes


def test_storage_remap_preserves_checkpoint_bytes_and_accounting():
    from b12x.moe.fused_moe._residency_storage import materialize_tier, tier_layout
    plan, weights = declaration()
    for expert in range(4):
        weights.w13[expert].fill_(expert*13+1)
        weights.w2[expert].fill_(expert*17+2)
        weights.w13_block_scales[expert].fill_(120+expert)
        weights.w2_block_scales[expert].fill_(124+expert)
    tier = materialize_tier((3, 0, 2), weights, plan.query, torch.device("cpu"), grace=False)
    layout, nbytes = tier_layout(3, 256, 256)
    assert tier.slab.numel() == nbytes
    assert all(offset % 256 == 0 for _, offset, _ in layout)
    for row, original in enumerate((3, 0, 2)):
        assert torch.equal(tier.fields["w13"][row], weights.w13[original])
        assert torch.equal(tier.fields["w2"][row], weights.w2[original])
        assert torch.equal(tier.fields["s13"][row], _swizzle_scale(weights.w13_block_scales[original]))
        assert torch.equal(tier.fields["s2"][row], _swizzle_scale(weights.w2_block_scales[original]))
    weights.w13.fill_(255)
    assert not torch.equal(tier.fields["w13"][0], weights.w13[3])


def test_capability_and_configuration_fail_closed():
    from scripts._sm103_preparation_corpus import IDENTITY
    from b12x.moe.fused_moe._residency_tuning import TUNING, ResidencyConfig
    plan, _ = declaration()
    choice = TUNING.configure(plan.query, device=IDENTITY, search=False)
    assert choice.default == ResidencyConfig()
    for capability in ((12, 0), (12, 1), (10, 0)):
        with pytest.raises(ValueError, match="SM103"):
            TUNING.configure(plan.query, device=replace(IDENTITY, compute_capability=capability), search=False)
    with pytest.raises(ValueError, match="unsupported"):
        TUNING.validate_config(plan.query, ResidencyConfig(backend="online_cache"), IDENTITY)


def test_model_identity_and_layer_are_admission_contracts():
    for changed in (replace(placement(), model_fingerprint="other checkpoint"), replace(placement(), layer="layer.8")):
        with pytest.raises(ValueError, match="fingerprint and layer"):
            declaration(profile=changed)


def test_old_binding_cannot_lazily_reprepare_a_released_plan():
    from b12x.moe.fused_moe._residency_preparation import ResidencyBinding
    plan, _ = declaration()
    binding = ResidencyBinding(a=torch.empty(1, 256), calls=(), output=torch.empty(1, 256), owners=(), plan=plan)
    with pytest.raises(RuntimeError, match="released"):
        moe.run(binding=binding)


def test_host_capacity_counts_only_cpu_numa_nodes(tmp_path):
    from b12x.moe.fused_moe._residency_storage import host_available_bytes

    def node(index, cpus, free_kb, file_kb):
        path = tmp_path / f"node{index}"
        path.mkdir()
        (path / "cpulist").write_text(cpus + "\n")
        (path / "meminfo").write_text(
            f"Node {index} MemTotal:       99999999 kB\n"
            f"Node {index} MemFree:        {free_kb} kB\n"
            f"Node {index} Active(file):   {file_kb} kB\n"
            f"Node {index} Inactive(file): {file_kb} kB\n"
            f"Node {index} SReclaimable:   0 kB\n"
        )

    node(0, "0-71", 1000, 500)
    # Coherent GPU memory is a CPU-less node and is not Grace capacity.
    node(1, "", 9_000_000, 0)
    assert host_available_bytes(tmp_path) == (1000 + 2 * 500) * 1024
