"""Pure host-side allocation planning for W4A16 mixed Trellis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixedTrellisBufferLayout:
    """Allocation identity for a mixed-Trellis persistent buffer set."""

    size_m: int
    hidden_size: int
    intermediate_size: int
    capacity_rows: int
    total_experts: int
    route_slots: int
    route_blocks: int
    fc1_cols: int
    fc1_scratch_elements: int
    fc2_scratch_elements: int
    workspace_elements: int


def max_packed_route_slots(numel: int, block_size: int, num_experts: int) -> int:
    max_packed_routes = int(numel) + int(num_experts) * (int(block_size) - 1)
    if int(numel) < int(num_experts):
        max_packed_routes = min(
            int(numel) * int(block_size),
            max_packed_routes,
        )
    return max_packed_routes


def packed_gemm_scratch_elements(
    *,
    size_n: int,
    route_slots: int,
    moe_block_size: int,
    sms: int,
) -> int:
    elements = min(
        int(size_n) * int(route_slots),
        int(sms) * 4 * int(moe_block_size) * 256,
    )
    if moe_block_size == 8:
        elements *= 2
    return max(elements, 1)


def plan_mixed_trellis_buffers(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    total_experts: int,
    moe_block_size: int,
    max_m_blocks: int,
    blocks_per_sm: int,
    sms: int,
) -> MixedTrellisBufferLayout:
    """Plan the complete persistent allocation without constructing tensors."""

    capacity_rows = int(size_m) * int(top_k)
    route_slots = max_packed_route_slots(
        capacity_rows,
        int(moe_block_size),
        int(total_experts),
    )
    route_blocks = (route_slots + int(moe_block_size) - 1) // int(moe_block_size)
    if route_blocks > int(max_m_blocks):
        raise ValueError(
            "mixed Trellis route capacity exceeds compiled max_m_blocks: "
            f"needed {route_blocks}, compiled {max_m_blocks}"
        )
    fc1_cols = 2 * int(intermediate_size)
    return MixedTrellisBufferLayout(
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        capacity_rows=capacity_rows,
        total_experts=int(total_experts),
        route_slots=route_slots,
        route_blocks=route_blocks,
        fc1_cols=fc1_cols,
        fc1_scratch_elements=packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=route_slots,
            moe_block_size=int(moe_block_size),
            sms=int(sms),
        ),
        fc2_scratch_elements=packed_gemm_scratch_elements(
            size_n=int(hidden_size),
            route_slots=route_slots,
            moe_block_size=int(moe_block_size),
            sms=int(sms),
        ),
        workspace_elements=max(int(sms) * 4, int(blocks_per_sm) * int(sms)) + 2,
    )
