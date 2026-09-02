"""Shape corpus contracts for the GEMM profile components."""

from __future__ import annotations

from b12x.policy.generation.gemm_corpus import (
    dense_linear_shapes,
    wo_projection_geometries,
)


def test_dsv4_wo_group_width_matches_the_served_projection() -> None:
    """vLLM packs DSV4 WO-A over group_width = heads_per_group * head_dim (8 x 512)."""
    dsv4 = [g for g in wo_projection_geometries() if g.model_id.startswith("deepseek")]
    assert dsv4
    for geometry in dsv4:
        assert geometry.group_width == 8 * (448 + 64)
        assert geometry.rank == 1024 and geometry.hidden == 4096
        assert geometry.groups * geometry.tp_size == 8


def test_block_fp8_shards_keep_whole_scale_blocks() -> None:
    for shape in dense_linear_shapes():
        if shape.recipe == "block_fp8":
            assert shape.in_features % 128 == 0 and shape.out_features % 128 == 0
