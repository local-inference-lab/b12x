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


def prewarm_w4a16_static_lora_triton_kernels(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    split_w13_b: bool,
    has_token_lora_mapping: bool,
    token_lora_mapping_dtype: torch.dtype,
    adapter_slot: int,
    w13_scale: float,
    w2_scale: float,
    direct: bool,
    device: torch.device,
) -> None:
    """Compile every Triton specialization used by one bound LoRA shape.

    ``JITFunction.warmup`` compiles from dtypes and constexpr metadata without
    dereferencing or mutating the caller's tensors.  Keeping this at binding
    time makes the subsequent first serving invocation safe to capture.
    """

    if token_lora_mapping_dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "token_lora_mapping_dtype must be torch.int32 or torch.int64, "
            f"got {token_lora_mapping_dtype}"
        )
    m = int(m)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_experts = int(num_experts)
    topk = int(topk)
    routes = m * topk
    mapping_dtype = (
        token_lora_mapping_dtype if has_token_lora_mapping else torch.int32
    )

    def warm_shrink(
        *, width: int, input_row_divisor: int, token_row_divisor: int
    ) -> None:
        _w4a16_static_lora_shrink_kernel.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.int32,
            mapping_dtype,
            torch.bfloat16,
            WIDTH=int(width),
            NUM_EXPERTS=num_experts,
            INPUT_ROW_DIVISOR=int(input_row_divisor),
            TOKEN_ROW_DIVISOR=int(token_row_divisor),
            ADAPTER_SLOT=int(adapter_slot),
            HAS_TOKEN_LORA_MAPPING=bool(has_token_lora_mapping),
            RANK=_RANK,
            BLOCK_K=_SHRINK_BLOCK_K,
            num_warps=4,
            grid=(routes, _RANK),
        )

    with torch.cuda.device(device):
        warm_shrink(
            width=hidden_size,
            input_row_divisor=topk,
            token_row_divisor=topk,
        )
        if not direct:
            if split_w13_b:
                _w4a16_static_lora_expand_pair_add_kernel.warmup(
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.int32,
                    torch.bfloat16,
                    SCALE=float(w13_scale),
                    OUTPUT_WIDTH=intermediate_size,
                    NUM_EXPERTS=num_experts,
                    RANK=_RANK,
                    BLOCK_N=_EXPAND_BLOCK_N,
                    num_warps=4,
                    grid=(routes, triton.cdiv(intermediate_size, _EXPAND_BLOCK_N)),
                )
            else:
                _w4a16_static_lora_expand_add_kernel.warmup(
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.int32,
                    torch.bfloat16,
                    torch.bfloat16,
                    SCALE=float(w13_scale),
                    OUTPUT_WIDTH=2 * intermediate_size,
                    NUM_EXPERTS=num_experts,
                    HAS_ROUTE_WEIGHTS=False,
                    RANK=_RANK,
                    BLOCK_N=_EXPAND_BLOCK_N,
                    num_warps=4,
                    grid=(
                        routes,
                        triton.cdiv(2 * intermediate_size, _EXPAND_BLOCK_N),
                    ),
                )

        warm_shrink(
            width=intermediate_size,
            input_row_divisor=1,
            token_row_divisor=topk,
        )
        if direct:
            _w4a16_static_lora_expand_token_sum_kernel.warmup(
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                torch.float32,
                torch.bfloat16,
                SCALE=float(w2_scale),
                NUM_TOKENS=m,
                TOPK=topk,
                OUTPUT_WIDTH=hidden_size,
                NUM_EXPERTS=num_experts,
                RANK=_RANK,
                BLOCK_N=_EXPAND_BLOCK_N,
                num_warps=4,
                grid=(m, triton.cdiv(hidden_size, _EXPAND_BLOCK_N)),
            )
        else:
            _w4a16_static_lora_expand_add_kernel.warmup(
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                torch.float32,
                torch.bfloat16,
                SCALE=float(w2_scale),
                OUTPUT_WIDTH=hidden_size,
                NUM_EXPERTS=num_experts,
                HAS_ROUTE_WEIGHTS=True,
                RANK=_RANK,
                BLOCK_N=_EXPAND_BLOCK_N,
                num_warps=4,
                grid=(routes, triton.cdiv(hidden_size, _EXPAND_BLOCK_N)),
            )


@triton.jit
def _w4a16_static_lora_shrink_kernel(
    x,
    adapter_a,
    expert_ids,
    token_lora_mapping,
    rank_scratch,
    WIDTH: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    INPUT_ROW_DIVISOR: tl.constexpr,
    TOKEN_ROW_DIVISOR: tl.constexpr,
    ADAPTER_SLOT: tl.constexpr,
    HAS_TOKEN_LORA_MAPPING: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(expert_ids + route).to(tl.int32)
    valid_expert = (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.maximum(0, tl.minimum(expert, NUM_EXPERTS - 1))
    input_row = route // INPUT_ROW_DIVISOR
    token_row = route // TOKEN_ROW_DIVISOR
    adapter_active = True
    if HAS_TOKEN_LORA_MAPPING:
        adapter_active = tl.load(token_lora_mapping + token_row) == ADAPTER_SLOT
    valid_expert = valid_expert & adapter_active

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

    tl.store(rank_scratch + route * RANK + rank, accumulator)


@triton.jit
def _w4a16_static_lora_expand_add_kernel(
    rank_scratch,
    adapter_b,
    expert_ids,
    route_weights,
    destination,
    SCALE: tl.constexpr,
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


@triton.jit
def _w4a16_static_lora_expand_pair_add_kernel(
    rank_scratch,
    gate_b,
    up_b,
    expert_ids,
    destination,
    SCALE: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Expand vLLM's separate gate/up B buffers into fused FC1 output."""

    route = tl.program_id(0)
    output_block = tl.program_id(1)
    columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = columns < OUTPUT_WIDTH

    expert = tl.load(expert_ids + route).to(tl.int32)
    valid_expert = (expert >= 0) & (expert < NUM_EXPERTS)
    safe_expert = tl.maximum(0, tl.minimum(expert, NUM_EXPERTS - 1))
    b_base = safe_expert * (OUTPUT_WIDTH * RANK) + columns * RANK
    rank_base = route * RANK

    gate_delta = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_delta = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for rank in tl.static_range(0, RANK):
        rank_value = tl.load(rank_scratch + rank_base + rank).to(tl.float32)
        gate_value = tl.load(
            gate_b + b_base + rank,
            mask=column_mask & valid_expert,
            other=0.0,
        ).to(tl.float32)
        up_value = tl.load(
            up_b + b_base + rank,
            mask=column_mask & valid_expert,
            other=0.0,
        ).to(tl.float32)
        gate_delta += rank_value * gate_value
        up_delta += rank_value * up_value

    destination_base = route * (2 * OUTPUT_WIDTH)
    gate_offsets = destination_base + columns
    up_offsets = destination_base + OUTPUT_WIDTH + columns
    old_gate = tl.load(
        destination + gate_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    old_up = tl.load(
        destination + up_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    gate_out = tl.where(valid_expert, old_gate + gate_delta * SCALE, old_gate)
    up_out = tl.where(valid_expert, old_up + up_delta * SCALE, old_up)
    tl.store(destination + gate_offsets, gate_out, mask=column_mask)
    tl.store(destination + up_offsets, up_out, mask=column_mask)


@triton.jit
def _w4a16_static_lora_expand_token_sum_kernel(
    rank_scratch,
    adapter_b,
    expert_ids,
    route_weights,
    destination,
    SCALE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Add routed LoRA FC2 deltas directly to the token-summed output."""

    token = tl.program_id(0)
    output_block = tl.program_id(1)
    columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = columns < OUTPUT_WIDTH
    delta = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for route_in_token in tl.static_range(0, TOPK):
        route = token * TOPK + route_in_token
        expert = tl.load(expert_ids + route).to(tl.int32)
        valid_expert = (expert >= 0) & (expert < NUM_EXPERTS)
        safe_expert = tl.maximum(0, tl.minimum(expert, NUM_EXPERTS - 1))
        b_base = safe_expert * (OUTPUT_WIDTH * RANK) + columns * RANK
        rank_base = route * RANK
        route_delta = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for rank in tl.static_range(0, RANK):
            rank_value = tl.load(rank_scratch + rank_base + rank).to(tl.float32)
            b_value = tl.load(
                adapter_b + b_base + rank,
                mask=column_mask & valid_expert,
                other=0.0,
            ).to(tl.float32)
            route_delta += rank_value * b_value
        route_weight = tl.load(route_weights + route).to(tl.float32)
        delta += tl.where(valid_expert, route_delta * route_weight, 0.0)

    destination_offsets = token * OUTPUT_WIDTH + columns
    previous = tl.load(
        destination + destination_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        destination + destination_offsets,
        previous + delta * SCALE,
        mask=column_mask & (token < NUM_TOKENS),
    )


def _validate_projection_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    device: torch.device,
    ndim: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}")
    allowed_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in allowed_dtypes:
        expected = " or ".join(str(value) for value in allowed_dtypes)
        raise TypeError(f"{name} must be {expected}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def run_w4a16_static_lora_shrink(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    expert_ids: torch.Tensor,
    rank_scratch: torch.Tensor,
    *,
    input_row_divisor: int = 1,
    token_row_divisor: int | None = None,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
) -> torch.Tensor:
    """Compute the routed rank-4 contraction into caller-owned scratch."""

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
        expert_ids,
        name="expert_ids",
        dtype=torch.int32,
        device=device,
        ndim=1,
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
    token_divisor = divisor if token_row_divisor is None else int(token_row_divisor)
    if token_divisor <= 0 or num_routes % token_divisor != 0:
        raise ValueError(
            "token_row_divisor must be positive and divide the route count, "
            f"got {token_divisor} for {num_routes} routes"
        )
    adapter_slot = int(adapter_slot)
    if adapter_slot < 0:
        raise ValueError("adapter_slot must be non-negative")
    if token_lora_mapping is not None:
        _validate_projection_tensor(
            token_lora_mapping,
            name="token_lora_mapping",
            dtype=(torch.int32, torch.int64),
            device=device,
            ndim=1,
        )
        required_mapping_rows = num_routes // token_divisor
        if int(token_lora_mapping.numel()) < required_mapping_rows:
            raise ValueError(
                "token_lora_mapping must contain at least "
                f"{required_mapping_rows} entries, got "
                f"{int(token_lora_mapping.numel())}"
            )
    if tuple(rank_scratch.shape) != (num_routes, _RANK):
        raise ValueError(
            f"rank_scratch must have shape {(num_routes, _RANK)}, "
            f"got {tuple(rank_scratch.shape)}"
        )

    _w4a16_static_lora_shrink_kernel[(num_routes, _RANK)](
        x,
        adapter_a,
        expert_ids,
        token_lora_mapping if token_lora_mapping is not None else expert_ids,
        rank_scratch,
        WIDTH=width,
        NUM_EXPERTS=num_experts,
        INPUT_ROW_DIVISOR=divisor,
        TOKEN_ROW_DIVISOR=token_divisor,
        ADAPTER_SLOT=adapter_slot,
        HAS_TOKEN_LORA_MAPPING=token_lora_mapping is not None,
        RANK=_RANK,
        BLOCK_K=_SHRINK_BLOCK_K,
        num_warps=4,
    )
    return rank_scratch


def run_w4a16_static_lora_output_sum(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    destination: torch.Tensor,
    rank_scratch: torch.Tensor,
    *,
    scale: float,
    topk: int,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
) -> torch.Tensor:
    """Add a routed rank-4 projection directly to token-summed output."""

    topk = int(topk)
    if topk <= 0:
        raise ValueError("topk must be positive")
    if int(expert_ids.numel()) != int(destination.shape[0]) * topk:
        raise ValueError("expert_ids must contain destination_rows * topk routes")
    run_w4a16_static_lora_shrink(
        x,
        adapter_a,
        expert_ids,
        rank_scratch,
        token_row_divisor=topk,
        token_lora_mapping=token_lora_mapping,
        adapter_slot=adapter_slot,
    )

    device = x.device
    _validate_projection_tensor(
        adapter_b,
        name="adapter_b",
        dtype=torch.bfloat16,
        device=device,
        ndim=3,
    )
    _validate_projection_tensor(
        route_weights,
        name="route_weights",
        dtype=torch.float32,
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
    num_experts, output_width, output_rank = (
        int(v) for v in adapter_b.shape
    )
    if num_experts != int(adapter_a.shape[0]) or output_rank != _RANK:
        raise ValueError(
            "adapter_b must have shape "
            f"({int(adapter_a.shape[0])}, output_width, {_RANK}), "
            f"got {tuple(adapter_b.shape)}"
        )
    num_routes = int(expert_ids.numel())
    if int(route_weights.numel()) != num_routes:
        raise ValueError(f"route_weights must contain {num_routes} values")
    if int(destination.shape[1]) != output_width:
        raise ValueError(
            f"destination width must be {output_width}, "
            f"got {int(destination.shape[1])}"
        )
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")

    _w4a16_static_lora_expand_token_sum_kernel[
        (int(destination.shape[0]), triton.cdiv(output_width, _EXPAND_BLOCK_N))
    ](
        rank_scratch,
        adapter_b,
        expert_ids,
        route_weights,
        destination,
        SCALE=scale,
        NUM_TOKENS=int(destination.shape[0]),
        TOPK=topk,
        OUTPUT_WIDTH=output_width,
        NUM_EXPERTS=num_experts,
        RANK=_RANK,
        BLOCK_N=_EXPAND_BLOCK_N,
        num_warps=4,
    )
    return destination


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
    token_row_divisor: int | None = None,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
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
    token_divisor = divisor if token_row_divisor is None else int(token_row_divisor)
    if token_divisor <= 0 or num_routes % token_divisor != 0:
        raise ValueError(
            "token_row_divisor must be positive and divide the route count, "
            f"got {token_divisor} for {num_routes} routes"
        )
    adapter_slot = int(adapter_slot)
    if adapter_slot < 0:
        raise ValueError("adapter_slot must be non-negative")
    if token_lora_mapping is not None:
        _validate_projection_tensor(
            token_lora_mapping,
            name="token_lora_mapping",
            dtype=(torch.int32, torch.int64),
            device=device,
            ndim=1,
        )
        required_mapping_rows = num_routes // token_divisor
        if int(token_lora_mapping.numel()) < required_mapping_rows:
            raise ValueError(
                "token_lora_mapping must contain at least "
                f"{required_mapping_rows} entries, got "
                f"{int(token_lora_mapping.numel())}"
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
        token_lora_mapping if token_lora_mapping is not None else expert_ids,
        rank_scratch,
        WIDTH=width,
        NUM_EXPERTS=num_experts,
        INPUT_ROW_DIVISOR=divisor,
        TOKEN_ROW_DIVISOR=token_divisor,
        ADAPTER_SLOT=adapter_slot,
        HAS_TOKEN_LORA_MAPPING=token_lora_mapping is not None,
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
        OUTPUT_WIDTH=output_width,
        NUM_EXPERTS=num_experts,
        HAS_ROUTE_WEIGHTS=route_weights is not None,
        RANK=_RANK,
        BLOCK_N=_EXPAND_BLOCK_N,
        num_warps=4,
    )
    return destination


def run_w4a16_static_lora_split_w13_projection(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_b: torch.Tensor,
    expert_ids: torch.Tensor,
    destination: torch.Tensor,
    rank_scratch: torch.Tensor,
    *,
    scale: float,
    input_row_divisor: int = 1,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
) -> torch.Tensor:
    """Add separate gate/up rank-4 B factors to fused ``[gate, up]`` FC1.

    vLLM deliberately owns the two B slices as independent contiguous
    allocations.  This path consumes those allocations directly, avoiding a
    per-forward concatenate and preserving CUDA-graph pointer stability.
    """

    device = x.device
    for name, tensor in (("gate_b", gate_b), ("up_b", up_b)):
        _validate_projection_tensor(
            tensor,
            name=name,
            dtype=torch.bfloat16,
            device=device,
            ndim=3,
        )
    _validate_projection_tensor(
        destination,
        name="destination",
        dtype=torch.bfloat16,
        device=device,
        ndim=2,
    )

    num_experts = int(adapter_a.shape[0])
    if gate_b.shape != up_b.shape:
        raise ValueError(
            "gate_b and up_b must have identical shapes, got "
            f"{tuple(gate_b.shape)} and {tuple(up_b.shape)}"
        )
    if int(gate_b.shape[0]) != num_experts or int(gate_b.shape[2]) != _RANK:
        raise ValueError(
            "gate_b and up_b must have shape "
            f"({num_experts}, output_width, {_RANK}), got "
            f"{tuple(gate_b.shape)}"
        )
    output_width = int(gate_b.shape[1])
    num_routes = int(expert_ids.numel())
    if tuple(destination.shape) != (num_routes, 2 * output_width):
        raise ValueError(
            "destination must have shape "
            f"{(num_routes, 2 * output_width)}, got {tuple(destination.shape)}"
        )
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")

    run_w4a16_static_lora_shrink(
        x,
        adapter_a,
        expert_ids,
        rank_scratch,
        input_row_divisor=input_row_divisor,
        token_row_divisor=input_row_divisor,
        token_lora_mapping=token_lora_mapping,
        adapter_slot=adapter_slot,
    )
    _w4a16_static_lora_expand_pair_add_kernel[
        (num_routes, triton.cdiv(output_width, _EXPAND_BLOCK_N))
    ](
        rank_scratch,
        gate_b,
        up_b,
        expert_ids,
        destination,
        SCALE=scale,
        OUTPUT_WIDTH=output_width,
        NUM_EXPERTS=num_experts,
        RANK=_RANK,
        BLOCK_N=_EXPAND_BLOCK_N,
        num_warps=4,
    )
    return destination


__all__ = [
    "prewarm_w4a16_static_lora_triton_kernels",
    "run_w4a16_static_lora_output_sum",
    "run_w4a16_static_lora_projection",
    "run_w4a16_static_lora_shrink",
    "run_w4a16_static_lora_split_w13_projection",
]
