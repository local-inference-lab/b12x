"""Static routed-expert LoRA contract for the W4A16 MoE path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class W4A16StaticExpertLoRA:
    """One BF16 rank-4 adapter already sharded for tensor parallelism.

    ``w13_b`` is in logical ``[gate, up]`` order after tensor-parallel
    sharding.  vLLM stores those two output slices in independent allocations;
    that layout can be represented without a graph-time concatenate by passing
    gate rows in ``w13_b`` and up rows in ``w13_b_up``. ``w13_a`` and ``w2_b``
    are replicated across TP ranks; W13 B is sharded on its output rows and
    ``w2_a`` on its input columns.  The contract intentionally excludes expert
    parallel remapping and supports one live adapter slot.
    """

    w13_a: torch.Tensor  # [experts, rank, hidden]
    w13_b: torch.Tensor  # [experts, 2 * intermediate_tp, rank]
    w2_a: torch.Tensor  # [experts, rank, intermediate_tp]
    w2_b: torch.Tensor  # [experts, hidden, rank]
    # Optional vLLM-native split representation.  When present, ``w13_b`` is
    # the gate half [experts, intermediate_tp, rank] and this is the up half
    # with the same shape.  Both remain caller-owned contiguous tensors.
    w13_b_up: torch.Tensor | None = None
    w13_scale: float = 1.0
    w2_scale: float = 1.0
    # Optional graph-dynamic vLLM-style token map. ``-1`` selects the base
    # model and ``adapter_slot`` selects this one loaded adapter. Both int32
    # and vLLM's canonical int64 mapping are accepted. Other slots are left
    # untouched by this deliberately single-adapter contract.
    token_lora_mapping: torch.Tensor | None = None
    adapter_slot: int = 0

    @property
    def rank(self) -> int:
        if not isinstance(self.w13_a, torch.Tensor) or self.w13_a.ndim != 3:
            return 0
        return int(self.w13_a.shape[1])


def validate_w4a16_static_expert_lora(
    adapter: W4A16StaticExpertLoRA,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> None:
    """Validate the deliberately narrow static-adapter serving contract."""

    if not isinstance(adapter, W4A16StaticExpertLoRA):
        raise TypeError("adapter must be a W4A16StaticExpertLoRA")
    num_experts = int(num_experts)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    rank = adapter.rank
    if rank != 4:
        raise ValueError(f"W4A16 static expert LoRA requires rank 4, got {rank}")

    split_w13_b = adapter.w13_b_up is not None
    expected_shapes = {
        "w13_a": (num_experts, rank, hidden_size),
        "w13_b": (
            num_experts,
            intermediate_size if split_w13_b else 2 * intermediate_size,
            rank,
        ),
        "w2_a": (num_experts, rank, intermediate_size),
        "w2_b": (num_experts, hidden_size, rank),
    }
    if split_w13_b:
        expected_shapes["w13_b_up"] = (num_experts, intermediate_size, rank)
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(adapter, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    for name in ("w13_scale", "w2_scale"):
        value = float(getattr(adapter, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")

    mapping = adapter.token_lora_mapping
    if mapping is not None:
        if not isinstance(mapping, torch.Tensor):
            raise TypeError("token_lora_mapping must be a torch.Tensor")
        if mapping.ndim != 1:
            raise ValueError("token_lora_mapping must be rank 1")
        if mapping.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "token_lora_mapping must be torch.int32 or torch.int64, "
                f"got {mapping.dtype}"
            )
        if mapping.device != device:
            raise ValueError(
                f"token_lora_mapping must be on {device}, got {mapping.device}"
            )
        if not mapping.is_contiguous():
            raise ValueError("token_lora_mapping must be contiguous")
    if int(adapter.adapter_slot) < 0:
        raise ValueError("adapter_slot must be non-negative")


__all__ = [
    "W4A16StaticExpertLoRA",
    "validate_w4a16_static_expert_lora",
]
