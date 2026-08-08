from __future__ import annotations

import pytest

from b12x.moe._shared.w4a16_layout import plan_mixed_trellis_buffers


def _layout(*, tier0_experts: int, tier1_experts: int, max_m_blocks: int = 1024):
    return plan_mixed_trellis_buffers(
        size_m=3072,
        hidden_size=6144,
        intermediate_size=768,
        top_k=8,
        total_experts=tier0_experts + tier1_experts,
        moe_block_size=32,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=1,
        sms=188,
    )


def test_mixed_buffer_layout_is_tier_partition_agnostic() -> None:
    first = _layout(tier0_experts=206, tier1_experts=50)
    second = _layout(tier0_experts=148, tier1_experts=108)
    incompatible = _layout(tier0_experts=160, tier1_experts=64)

    assert first == second
    assert first != incompatible
    assert first.capacity_rows == 3072 * 8
    assert first.total_experts == 256


def test_mixed_buffer_layout_rejects_insufficient_route_capacity() -> None:
    with pytest.raises(ValueError, match="exceeds compiled max_m_blocks"):
        _layout(tier0_experts=206, tier1_experts=50, max_m_blocks=1)
