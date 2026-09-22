"""Storage declarations are capabilities, not format execution qualification."""

from dataclasses import replace

import pytest

from b12x.moe.residency.storage import (
    BackingMode, ExpertRepresentation, ExpertShard, ExpertStorageContract,
    ExpertStorageSource,
)


def contract(**changes):
    fp8 = ExpertRepresentation(encoding="fp8_block128", bytes_per_expert=123456, alignment=128)
    return ExpertStorageContract(**(dict(
        adapter="fixture_fp8", checkpoint="checkpoint", layer="moe", recipe="fp8_bf16",
        shard=ExpertShard(experts=7, hidden=256, intermediate=128, global_intermediate=128),
        source=fp8, backing=fp8, resident=fp8, mode=BackingMode.DIRECT,
        direct_cold_execution=True,
        rollback="canonical_restore",
    ) | changes))


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_rank_local_storage_preserves_logical_expert_identity(world):
    for rank in range(world):
        value = contract(shard=ExpertShard(experts=7, hidden=256, intermediate=128,
            global_intermediate=128 * world, intermediate_start=128 * rank,
            rank=rank, world_size=world))
        assert value.shard.experts == 7
        assert value.source.payload_bytes(7) == 7 * 123456
        value.require_cold_execution()


def test_transformed_backing_and_promotion_are_distinct_capabilities():
    compressed = ExpertRepresentation(encoding="exl3_fixture", bytes_per_expert=111)
    executable = ExpertRepresentation(encoding="prepared_fixture", bytes_per_expert=333)
    value = contract(source=compressed, backing=compressed, resident=executable,
        mode=BackingMode.ON_PROMOTION, direct_cold_execution=False,
        promotion_transform="bounded_decode", rollback="retained_slot_journal")
    assert value.source.payload_bytes(7) != value.resident.payload_bytes(7)
    with pytest.raises(ValueError, match="direct cold"):
        value.require_cold_execution()
    with pytest.raises(ValueError, match="transform"):
        replace(value, promotion_transform=None)
    prepared = replace(value, mode=BackingMode.PREPARED, backing=executable,
        direct_cold_execution=True, source_transform="prepare_btx_or_trellis",
        promotion_transform=None)
    prepared.require_cold_execution()
    resident = replace(value, mode=BackingMode.RESIDENT_ONLY, backing=None)
    with pytest.raises(ValueError, match="direct cold"):
        resident.require_cold_execution()


def test_mxfp4_scale_geometry_is_an_adapter_property():
    row = ExpertRepresentation(encoding="mxfp4_e8m0_k32", bytes_per_expert=901)
    value = contract(source=row, backing=row, resident=row, recipe="mxfp4_mxfp8")
    assert value.resident.bytes_per_expert == 901
    with pytest.raises(ValueError, match="power of two"):
        replace(row, alignment=96)


def test_representation_changes_require_explicit_preparation_capabilities():
    value = contract()
    foreign = ExpertRepresentation(encoding="compressed_source", bytes_per_expert=111)
    with pytest.raises(ValueError, match="source transform"):
        replace(value, source=foreign)
    prepared = replace(value, source=foreign, mode=BackingMode.PREPARED,
                       source_transform="decode_at_load")
    assert prepared.direct_cold_execution
    with pytest.raises(ValueError, match="preparation transform"):
        replace(value, source=foreign, backing=None, mode=BackingMode.RESIDENT_ONLY,
                direct_cold_execution=False, rollback="slot_journal")
    resident = replace(value, source=foreign, backing=None, mode=BackingMode.RESIDENT_ONLY,
                       direct_cold_execution=False, source_transform="decode_at_load",
                       rollback="slot_journal")
    assert resident.source != resident.resident


def test_nvfp4_adapter_implements_storage_and_rejects_foreign_execution():
    from tests.moe.test_prepared_expert_cache import source, declaration

    value = source()
    assert isinstance(value, ExpertStorageSource)
    assert value.storage.source.payload_bytes(4) == value.source_bytes
    assert value.storage.mode == BackingMode.PREPARED
    assert value.storage.backing == value.storage.resident
    assert sum(v.numel() * v.element_size() for v in value.row(0).values()) == value.storage.resident.bytes_per_expert
    assert declaration(value).prepared is None
    for expert in (-1, 4, True):
        with pytest.raises(ValueError, match="expert ID"):
            value.row(expert)

    class Foreign:
        storage = contract()
        source_bytes = 123456 * 7
        def validate_values(self): pass
        def row(self, expert): return {}

    from b12x.moe.fused_moe._cache_preparation import plan
    from b12x.moe.fused_moe.residency import ExpertResidencyPlan
    placement = ExpertResidencyPlan(total_experts=7, hbm_expert_ids=(0,),
        grace_expert_ids=tuple(range(1, 7)), layer="moe", model_fingerprint="checkpoint",
        workload="test", provenance="capability fixture")
    with pytest.raises(NotImplementedError, match="execution backend"):
        plan(source=Foreign(), capacity=None, placement=placement, memory_budget=None)
