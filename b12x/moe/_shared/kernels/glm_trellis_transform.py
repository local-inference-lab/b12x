"""Exact GLM route transforms around compact trellis projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
)


@triton.jit
def _glm_route_silu_kernel(
    gate,
    up,
    route_experts,
    gate_svh,
    up_svh,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    expert = tl.load(route_experts + route).to(tl.int64)
    gate_value = tl.load(gate + route * width + offsets, mask=mask, other=0.0)
    up_value = tl.load(up + route * width + offsets, mask=mask, other=0.0)
    gate_scale = tl.load(gate_svh + expert * width + offsets, mask=mask, other=0.0)
    up_scale = tl.load(up_svh + expert * width + offsets, mask=mask, other=0.0)
    gate_value = gate_value.to(tl.float32) * gate_scale.to(tl.float32)
    up_value = up_value.to(tl.float32) * up_scale.to(tl.float32)
    activated = gate_value * tl.sigmoid(gate_value) * up_value
    tl.store(output + route * width + offsets, activated, mask=mask)


@triton.jit
def _glm_route_scale_kernel(
    source,
    route_experts,
    scale_table,
    output,
    width: tl.constexpr,
    block_n: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    expert = tl.load(route_experts + route).to(tl.int64)
    values = tl.load(source + route * width + offsets, mask=mask, other=0.0)
    scales = tl.load(scale_table + expert * width + offsets, mask=mask, other=0.0)
    tl.store(output + route * width + offsets, values * scales, mask=mask)


@triton.jit
def _glm_topk_weighted_sum_kernel(
    routes,
    topk_weights,
    output,
    width: tl.constexpr,
    topk: tl.constexpr,
    block_n: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_n + tl.arange(0, block_n)
    mask = offsets < width
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for route_in_token in tl.static_range(topk):
        route = token * topk + route_in_token
        values = tl.load(routes + route * width + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(topk_weights + route).to(tl.float32)
        accumulator += values * weight
    tl.store(output + token * width + offsets, accumulator, mask=mask)


def run_glm_gate_up_output_transform_silu(
    gate_transformed: torch.Tensor,
    up_transformed: torch.Tensor,
    route_experts: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    gate_hadamard: torch.Tensor,
    up_hadamard: torch.Tensor,
    output: torch.Tensor,
    *,
    ones: torch.Tensor,
) -> torch.Tensor:
    """Invert gate/up H128 bases, apply expert scales, then exact SwiGLU."""

    if gate_transformed.shape != up_transformed.shape or gate_transformed.ndim != 2:
        raise ValueError("gate/up transformed outputs must be aligned rank-2 tensors")
    routes, width = (int(value) for value in gate_transformed.shape)
    device = gate_transformed.device
    for name, tensor in (
        ("gate_transformed", gate_transformed),
        ("up_transformed", up_transformed),
        ("gate_hadamard", gate_hadamard),
        ("up_hadamard", up_hadamard),
        ("output", output),
    ):
        if (
            tensor.shape != gate_transformed.shape
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {tuple(gate_transformed.shape)} "
                f"on {device}"
            )
    if (
        route_experts.shape != (routes,)
        or route_experts.dtype != torch.int32
        or route_experts.device != device
        or not route_experts.is_contiguous()
    ):
        raise TypeError("route_experts must be contiguous int32 [routes]")
    if gate_svh.shape != up_svh.shape or gate_svh.ndim != 2:
        raise ValueError("gate/up svh tables must be aligned [experts, intermediate]")
    for name, tensor in (("gate_svh", gate_svh), ("up_svh", up_svh)):
        if (
            int(tensor.shape[1]) != width
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(f"{name} must be contiguous FP16 [experts,{width}]")
    if (
        ones.shape != (width,)
        or ones.dtype != torch.float16
        or ones.device != device
        or not ones.is_contiguous()
    ):
        raise TypeError(f"ones must be contiguous FP16 [{width}] on {device}")

    _run_trellis_dense_hadamard128(
        gate_transformed,
        gate_hadamard,
        ones,
        scale_before=False,
    )
    _run_trellis_dense_hadamard128(
        up_transformed,
        up_hadamard,
        ones,
        scale_before=False,
    )
    block_n = 256
    _glm_route_silu_kernel[(routes, triton.cdiv(width, block_n))](
        gate_hadamard,
        up_hadamard,
        route_experts,
        gate_svh,
        up_svh,
        output,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    return output


def run_glm_down_input_transform(
    activation: torch.Tensor,
    route_experts: torch.Tensor,
    down_suh: torch.Tensor,
    scaled: torch.Tensor,
    rotated: torch.Tensor,
    *,
    ones: torch.Tensor,
) -> torch.Tensor:
    """Apply expert-private down ``suh`` followed by normalized H128."""

    if activation.ndim != 2:
        raise ValueError("down activation must be rank 2")
    routes, width = (int(value) for value in activation.shape)
    device = activation.device
    for name, tensor in (
        ("activation", activation),
        ("scaled", scaled),
        ("rotated", rotated),
    ):
        if (
            tensor.shape != activation.shape
            or tensor.dtype != torch.float16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                f"{name} must be contiguous FP16 {tuple(activation.shape)} on {device}"
            )
    if (
        route_experts.shape != (routes,)
        or route_experts.dtype != torch.int32
        or route_experts.device != device
        or not route_experts.is_contiguous()
    ):
        raise TypeError("route_experts must be contiguous int32 [routes]")
    if (
        down_suh.ndim != 2
        or int(down_suh.shape[1]) != width
        or down_suh.dtype != torch.float16
        or down_suh.device != device
        or not down_suh.is_contiguous()
    ):
        raise TypeError(f"down_suh must be contiguous FP16 [experts,{width}]")
    if (
        ones.shape != (width,)
        or ones.dtype != torch.float16
        or ones.device != device
        or not ones.is_contiguous()
    ):
        raise TypeError(f"ones must be contiguous FP16 [{width}] on {device}")

    block_n = 256
    _glm_route_scale_kernel[(routes, triton.cdiv(width, block_n))](
        activation,
        route_experts,
        down_suh,
        scaled,
        width=width,
        block_n=block_n,
        num_warps=4,
    )
    _run_trellis_dense_hadamard128(
        scaled,
        rotated,
        ones,
        scale_before=True,
    )
    return rotated


def run_glm_down_output_transform_sum(
    down_transformed: torch.Tensor,
    topk_weights: torch.Tensor,
    down_svh: torch.Tensor,
    down_canonical: torch.Tensor,
    output: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Invert down H128, apply shared ``svh``, and sum signed top-k routes."""

    if down_transformed.ndim != 2:
        raise ValueError("down transformed output must be rank 2")
    routes, width = (int(value) for value in down_transformed.shape)
    topk = int(topk)
    if topk <= 0 or routes % topk:
        raise ValueError("route count must be positive and divisible by topk")
    tokens = routes // topk
    device = down_transformed.device
    if (
        down_transformed.dtype != torch.float16
        or not down_transformed.is_cuda
        or not down_transformed.is_contiguous()
        or down_canonical.shape != down_transformed.shape
        or down_canonical.dtype != torch.float16
        or down_canonical.device != device
        or not down_canonical.is_contiguous()
    ):
        raise TypeError("down buffers must be aligned contiguous CUDA FP16")
    if (
        down_svh.shape != (width,)
        or down_svh.dtype != torch.float16
        or down_svh.device != device
        or not down_svh.is_contiguous()
    ):
        raise TypeError(f"down_svh must be contiguous FP16 [{width}]")
    if (
        topk_weights.shape != (tokens, topk)
        or topk_weights.dtype != torch.float32
        or topk_weights.device != device
        or not topk_weights.is_contiguous()
    ):
        raise TypeError(f"topk_weights must be contiguous FP32 [{tokens},{topk}]")
    if (
        output.shape != (tokens, width)
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.device != device
        or not output.is_contiguous()
    ):
        raise TypeError(
            f"output must be contiguous FP16/BF16 [{tokens},{width}] on {device}"
        )

    _run_trellis_dense_hadamard128(
        down_transformed,
        down_canonical,
        down_svh,
        scale_before=False,
    )
    block_n = 256
    _glm_topk_weighted_sum_kernel[(tokens, triton.cdiv(width, block_n))](
        down_canonical,
        topk_weights,
        output,
        width=width,
        topk=topk,
        block_n=block_n,
        num_warps=4,
    )
    return output


__all__ = [
    "run_glm_down_input_transform",
    "run_glm_down_output_transform_sum",
    "run_glm_gate_up_output_transform_silu",
]
