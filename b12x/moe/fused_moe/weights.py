"""Typed checkpoint tensor bundles accepted by fused-MoE preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScaleFactors:
    """One scale boundary represented as vectors times optional gains."""

    vectors: torch.Tensor
    gains: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, torch.Tensor):
            raise TypeError("ScaleFactors.vectors must be a torch.Tensor")
        if self.gains is not None and not isinstance(self.gains, torch.Tensor):
            raise TypeError("ScaleFactors.gains must be a torch.Tensor or None")


@dataclass(frozen=True)
class TrellisWeights:
    """Layer-local views of the canonical ``b12x_trellis`` tensors.

    ``atoms`` is the rank-local ``[I_local/32, row_stride]`` uint8 payload.
    ``rate`` is a view selected from the single model-level uint8 rate tensor;
    it is never copied merely to give each layer its own rate parameter.
    """

    atoms: torch.Tensor
    rate: torch.Tensor
    input_scales: ScaleFactors
    intermediate_scales: ScaleFactors
    output_scales: ScaleFactors
    expert_transform_draws: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, torch.Tensor):
            raise TypeError("TrellisWeights.atoms must be a torch.Tensor")
        if not isinstance(self.rate, torch.Tensor):
            raise TypeError("TrellisWeights.rate must be a torch.Tensor")
        for name in (
            "input_scales",
            "intermediate_scales",
            "output_scales",
        ):
            if not isinstance(getattr(self, name), ScaleFactors):
                raise TypeError(f"TrellisWeights.{name} must be ScaleFactors")
        if self.expert_transform_draws is not None and not isinstance(
            self.expert_transform_draws, torch.Tensor
        ):
            raise TypeError(
                "TrellisWeights.expert_transform_draws must be a tensor or None"
            )


@dataclass(frozen=True)
class PackedWeights:
    """Ordinary packed MoE checkpoint tensors, without runtime policy fields."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_block_scales: torch.Tensor
    w2_block_scales: torch.Tensor
    w13_global_scales: torch.Tensor
    w2_global_scales: torch.Tensor
    input_scale: torch.Tensor
    intermediate_scale: torch.Tensor

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"PackedWeights.{name} must be a torch.Tensor")


__all__ = ["PackedWeights", "ScaleFactors", "TrellisWeights"]
