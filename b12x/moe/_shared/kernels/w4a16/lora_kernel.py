"""Native CUDA primitives for the staged W4A16 static-expert LoRA path.

These kernels deliberately implement only the fixed rank-4 contract in
``lora.py``.  The caller owns every tensor, including the rank scratch, so a
prewarmed invocation is safe to capture and replay in a CUDA graph.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_RANK = 4
_SHRINK_BLOCK_K = 1024
_EXPAND_BLOCK_N = 256


@triton.jit
def _w4a16_static_lora_shrink_kernel(
    x,
    adapter_a,
    expert_ids,
    rank_scratch,
    NUM_ROUTES: tl.constexpr,
    WIDTH: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    INPUT_ROW_DIVISOR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(expert_ids + route).to(tl.int32)
    valid_expert = (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.maximum(0, tl.minimum(expert, NUM_EXPERTS - 1))
    input_row = route // INPUT_ROW_DIVISOR

    accumulator = 0.0
    offsets = tl.arange(0, BLOCK_K)
    for start in tl.range(0, WIDTH, BLOCK_K):
        columns = start + offsets
        mask = columns < WIDTH
        x_values = tl.load(
            x + input_row * WIDTH + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        a_values = tl.load(
            adapter_a
            + safe_expert * (RANK * WIDTH)
            + rank * WIDTH
            + columns,
            mask=mask & valid_expert,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(x_values * a_values, axis=0)

    tl.store(
        rank_scratch + route * RANK + rank,
        accumulator,
        mask=route < NUM_ROUTES,
    )


@triton.jit
def _w4a16_static_lora_expand_add_kernel(
    rank_scratch,
    adapter_b,
    expert_ids,
    route_weights,
    destination,
    SCALE: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HAS_ROUTE_WEIGHTS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    route = tl.program_id(0)
    output_block = tl.program_id(1)
    columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = columns < OUTPUT_WIDTH

    expert = tl.load(expert_ids + route).to(tl.int32)
    valid_expert = (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.maximum(0, tl.minimum(expert, NUM_EXPERTS - 1))
    b_base = safe_expert * (OUTPUT_WIDTH * RANK) + columns * RANK
    rank_base = route * RANK

    delta = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for rank in tl.static_range(0, RANK):
        rank_value = tl.load(rank_scratch + rank_base + rank).to(tl.float32)
        b_value = tl.load(
            adapter_b + b_base + rank,
            mask=column_mask & valid_expert,
            other=0.0,
        ).to(tl.float32)
        delta += rank_value * b_value

    multiplier = SCALE
    if HAS_ROUTE_WEIGHTS:
        multiplier *= tl.load(route_weights + route).to(tl.float32)
    destination_offsets = route * OUTPUT_WIDTH + columns
    previous = tl.load(
        destination + destination_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    updated = tl.where(valid_expert, previous + delta * multiplier, previous)
    tl.store(destination + destination_offsets, updated, mask=column_mask)


def _validate_projection_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    ndim: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def run_w4a16_static_lora_projection(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    expert_ids: torch.Tensor,
    destination: torch.Tensor,
    rank_scratch: torch.Tensor,
    *,
    scale: float,
    input_row_divisor: int = 1,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add one routed rank-4 projection to a caller-owned destination.

    ``input_row_divisor=top_k`` maps flattened FC1 routes back to the original
    token rows.  FC2 passes one input row per route and supplies router weights
    because the normal W4A16 FC2 path weights each route before top-k reduction.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    device = x.device
    _validate_projection_tensor(
        x,
        name="x",
        dtype=torch.bfloat16,
        device=device,
        ndim=2,
    )
    _validate_projection_tensor(
        adapter_a,
        name="adapter_a",
        dtype=torch.bfloat16,
        device=device,
        ndim=3,
    )
    _validate_projection_tensor(
        adapter_b,
        name="adapter_b",
        dtype=torch.bfloat16,
        device=device,
        ndim=3,
    )
    _validate_projection_tensor(
        expert_ids,
        name="expert_ids",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    _validate_projection_tensor(
        destination,
        name="destination",
        dtype=torch.bfloat16,
        device=device,
        ndim=2,
    )
    _validate_projection_tensor(
        rank_scratch,
        name="rank_scratch",
        dtype=torch.bfloat16,
        device=device,
        ndim=2,
    )

    num_experts, rank, width = (int(v) for v in adapter_a.shape)
    if rank != _RANK:
        raise ValueError(f"adapter_a rank must be {_RANK}, got {rank}")
    output_experts, output_width, output_rank = (
        int(v) for v in adapter_b.shape
    )
    if output_experts != num_experts or output_rank != _RANK:
        raise ValueError(
            "adapter_b must have shape "
            f"({num_experts}, output_width, {_RANK}), got {tuple(adapter_b.shape)}"
        )
    if int(x.shape[1]) != width:
        raise ValueError(f"x width must be {width}, got {int(x.shape[1])}")

    num_routes = int(expert_ids.numel())
    divisor = int(input_row_divisor)
    if divisor <= 0:
        raise ValueError(f"input_row_divisor must be positive, got {divisor}")
    if num_routes != int(x.shape[0]) * divisor:
        raise ValueError(
            "expert_ids must contain exactly input_rows * input_row_divisor "
            f"routes, got {num_routes} for {int(x.shape[0])} * {divisor}"
        )
    if tuple(destination.shape) != (num_routes, output_width):
        raise ValueError(
            f"destination must have shape {(num_routes, output_width)}, "
            f"got {tuple(destination.shape)}"
        )
    if tuple(rank_scratch.shape) != (num_routes, _RANK):
        raise ValueError(
            f"rank_scratch must have shape {(num_routes, _RANK)}, "
            f"got {tuple(rank_scratch.shape)}"
        )

    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")
    if route_weights is not None:
        _validate_projection_tensor(
            route_weights,
            name="route_weights",
            dtype=torch.float32,
            device=device,
            ndim=1,
        )
        if int(route_weights.numel()) != num_routes:
            raise ValueError(
                f"route_weights must contain {num_routes} values, "
                f"got {int(route_weights.numel())}"
            )

    _w4a16_static_lora_shrink_kernel[(num_routes, _RANK)](
        x,
        adapter_a,
        expert_ids,
        rank_scratch,
        NUM_ROUTES=num_routes,
        WIDTH=width,
        NUM_EXPERTS=num_experts,
        INPUT_ROW_DIVISOR=divisor,
        RANK=_RANK,
        BLOCK_K=_SHRINK_BLOCK_K,
        num_warps=4,
    )
    _w4a16_static_lora_expand_add_kernel[
        (num_routes, triton.cdiv(output_width, _EXPAND_BLOCK_N))
    ](
        rank_scratch,
        adapter_b,
        expert_ids,
        route_weights if route_weights is not None else destination,
        destination,
        SCALE=scale,
        NUM_ROUTES=num_routes,
        OUTPUT_WIDTH=output_width,
        NUM_EXPERTS=num_experts,
        HAS_ROUTE_WEIGHTS=route_weights is not None,
        RANK=_RANK,
        BLOCK_N=_EXPAND_BLOCK_N,
        num_warps=4,
    )
    return destination


__all__ = ["run_w4a16_static_lora_projection"]
