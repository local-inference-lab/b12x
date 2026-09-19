"""Shared policy independence and compatibility with SM103 profile artifacts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from b12x.moe import residency as r


def snapshot(counts=(0,)*5, calls=0):
    return r.RoutingSnapshot(epoch=7, rank=2, layers=(r.LayerRoutingCounts(
        layer="arbitrary.layer", phase="verify", counts=counts,
        calls=calls, sampled_calls=calls, tokens=sum(counts), sampled_tokens=sum(counts)),))


def policy(**exchange_changes):
    placement = r.ExpertPlacement(total_experts=5,
        resident_expert_ids=(2, 0), backing_expert_ids=(4, 1, 3))
    slots = r.ResidencySlotSnapshot(preparation_id="host-only", generation=9,
        expert_map=placement.expert_map, healthy=True)
    exchange = r.ResidencyExchangeSpec(backend="host-test-payload",
        direct_backing_execution=True, fixed_address_quiescent_exchange=True,
        payload_copy_bytes_per_pair=123, map_copy_bytes_per_transaction=17)
    controller = r.ResidencyCacheController(
        config=r.ResidencyCacheConfig(max_pairs=2, minimum_cold_selections=2,
            minimum_score_gain=1, minimum_residency_windows=0, phase="verify"),
        observations=r.RoutingObservationSpec(layer="arbitrary.layer", experts=5,
            phase="verify", max_top_k=7, rank=2, owner_rank=2),
        exchange=replace(exchange, **exchange_changes), slots=slots, baseline=snapshot())
    return controller, slots


def test_shared_policy_uses_backend_accounting_and_canonical_rows():
    controller, slots = policy()
    decision = controller.observe(snapshot((1, 8, 0, 2, 7), 1), slots=slots)
    assert decision.pairs == ((1, 2), (4, 0))
    assert decision.cold_selections == 17 and decision.counterfactual_cold_fraction == 3/18
    mapping = list(slots.expert_map)
    for resident, backing in decision.pairs:
        mapping[resident], mapping[backing] = mapping[backing], mapping[resident]
    committed = replace(slots, generation=10, expert_map=tuple(mapping))
    outcome = controller.finish(decision, slots=committed)
    assert outcome.committed_payload_copy_bytes == 246
    assert outcome.committed_map_copy_bytes == 17
    assert outcome.hot_experts == (1, 4)
    decision = controller.observe(snapshot((9, 11, 0, 2, 8), 2), slots=committed)
    assert decision.pairs == ((0, 4),)
    assert decision.observed_hits_since_promotion == ((1, 3), (4, 1))
    declined = controller.finish(decision, slots=committed)
    assert declined.promotions == 2
    assert declined.committed_payload_copy_bytes == declined.committed_map_copy_bytes == 0


@pytest.mark.parametrize("capability", ["direct_backing_execution", "fixed_address_quiescent_exchange"])
def test_cache_requires_backend_miss_and_exchange_capabilities(capability):
    with pytest.raises(ValueError, match="requires direct backing execution"):
        policy(**{capability: False})


@pytest.mark.parametrize("values", [
    {"payload_copy_bytes_per_pair": -1}, {"map_copy_bytes_per_transaction": True},
    {"direct_backing_execution": 1}, {"backend": " "},
])
def test_backend_descriptor_rejects_invalid_metadata(values):
    with pytest.raises(ValueError):
        policy(**values)


@pytest.mark.parametrize("resident,backing", [((0, 0), (1, 2)), ((0,), (1,)), ((False,), (1, 2))])
def test_placement_requires_complete_canonical_partition(resident, backing):
    with pytest.raises(ValueError, match="partition"):
        r.ExpertPlacement(total_experts=3, resident_expert_ids=resident, backing_expert_ids=backing)


def test_shared_policy_runs_without_importing_tensor_libraries_or_backends():
    # Run a real decision/acknowledgement in an isolated interpreter; merely
    # checking imports in pytest would miss libraries imported by conftest.
    program = '''
import importlib.abc
import sys
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "triton", "cutlass", "cuda"}
                or fullname.startswith(("b12x.moe.fused_moe", "b12x.preparation"))):
            raise AssertionError("shared policy imported " + fullname)
sys.meta_path.insert(0, BlockBackend())
from tests.moe.test_shared_residency import test_shared_policy_uses_backend_accounting_and_canonical_rows
test_shared_policy_uses_backend_accounting_and_canonical_rows()
'''
    subprocess.run([sys.executable, "-c", program], check=True)


def test_legacy_imports_keep_type_identity_and_controller_adapter():
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe import api
    for name in ("LayerRoutingCounts", "RoutingSnapshot", "ResidencySlotSnapshot",
                 "ResidencyUpdateCapacity", "ResidencyUpdateError", "ResidencyCacheConfig",
                 "ResidencyCacheDecision", "ResidencyCacheOutcome"):
        assert getattr(moe, name) is getattr(r, name)
        assert name in api.__all__
    assert issubclass(moe.ResidencyCacheController, r.ResidencyCacheController)


@pytest.mark.parametrize("schema", [1, 2])
def test_pre_extraction_artifact_bytes_and_hashes_are_preserved(tmp_path, schema):
    # Fixtures were emitted by 181e234b5320eae67f7e1096672129d01c14ff09
    # before extraction. Only the schema-2 timestamp was fixed for reproducibility.
    from b12x.moe.fused_moe.residency import read_profiles
    from b12x.moe.fused_moe.automatic import ResidencyProfile
    from tests.moe.test_expert_residency import placement
    from tests.moe.test_automatic_residency import completed
    path = Path(__file__).parent / "fixtures" / f"residency-schema{schema}.json"
    expected = json.loads(path.read_text())
    if schema == 1:
        actual = {"version": 1, "layers": [placement().to_dict()]}
    else:
        _, profile = completed(tmp_path)
        actual = replace(profile, created_at=expected["created_at"]).to_dict()
        assert ResidencyProfile.from_dict(expected).profile_hash == actual["profile_hash"]
    assert json.loads(json.dumps(actual)) == expected
    plans = read_profiles(path)
    assert all(p.placement.expert_map == p.expert_map for p in plans)


def test_profile_view_does_not_relax_native_recipe_or_hash_validation():
    from b12x.moe.fused_moe.automatic import ResidencyLayerSpec, ResidencyProfile
    path = Path(__file__).parent / "fixtures" / "residency-schema2.json"
    payload = json.loads(path.read_text())
    payload["profile_hash"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        ResidencyProfile.from_dict(payload)
    with pytest.raises(ValueError):
        ResidencyLayerSpec(layer="arbitrary.layer", experts=5, hidden=37,
            intermediate=113, max_tokens=7, max_top_k=3)
