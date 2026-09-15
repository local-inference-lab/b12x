"""Typed checkpoint tensor bundles accepted by fused-MoE preparation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from b12x.moe._shared.kernels.w4a16.btx import BtxLayer
    from ._impl import B12XFP4ExpertWeights
    from .planning import WeightPlan


class WeightEncoding(str, Enum):
    """Numeric encoding of prepared weight elements."""

    FP4_E2M1 = "fp4_e2m1"
    FP6_E2M3 = "fp6_e2m3"
    TRELLIS = "trellis"


class ScaleEncoding(str, Enum):
    """Scale encoding consumed with the prepared weights."""

    E4M3_K16 = "e4m3_k16"
    E4M3_K32 = "e4m3_k32"
    E8M0_K32 = "e8m0_k32"
    E8M0_K32_E4M3_RESIDUAL = "e8m0_k32_x_e4m3_k16_residual"
    TRELLIS_SCALES = "trellis_scales"


class WeightPacking(str, Enum):
    """In-memory packing or zero-copy view exposed after preparation."""

    SOURCE_NATIVE = "source_native"
    MMA_VIEW = "mma_view"
    MMA_PACKED = "mma_packed"
    QMMA_REPACKED = "qmma_repacked"
    TRELLIS_NATIVE = "trellis_native"


@dataclass(frozen=True, kw_only=True)
class PreparedWeightFormat:
    """Numeric and physical contract produced by weight preparation."""

    weights: WeightEncoding
    scales: ScaleEncoding
    packing: WeightPacking
    available_packings: frozenset[WeightPacking]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", WeightEncoding(self.weights))
        object.__setattr__(self, "scales", ScaleEncoding(self.scales))
        object.__setattr__(self, "packing", WeightPacking(self.packing))
        available = frozenset(WeightPacking(value) for value in self.available_packings)
        if self.packing not in available:
            raise ValueError("packing must be present in available_packings")
        object.__setattr__(self, "available_packings", available)


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
    A configured group size appends a local ``I_local/group_size`` rate axis;
    the rank extent must start on a group boundary. Each rate byte stores
    independent low/high plane bit widths in its low/high nibbles. Atom rows
    concatenate expert-major gate/up/down sections, each with its low plane
    followed by its high plane. Grouped rows may end in zero padding; their
    storage and physical row stride must be aligned to 16 bytes.
    ``global_intermediate_size`` and ``intermediate_offset`` locate this rank
    on the checkpoint's intermediate axis, in channels. Nonzero coupled draws
    require that metadata so preparation slices the global sign sequence.
    """

    atoms: torch.Tensor
    rate: torch.Tensor
    input_scales: ScaleFactors
    intermediate_scales: ScaleFactors
    output_scales: ScaleFactors
    expert_transform_draws: torch.Tensor | None = None
    global_intermediate_size: int | None = None
    intermediate_offset: int = 0

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
        for name, value in (
            ("global_intermediate_size", self.global_intermediate_size),
            ("intermediate_offset", self.intermediate_offset),
        ):
            if value is None and name == "global_intermediate_size":
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"TrellisWeights.{name} must be an integer")
            minimum = 32 if name == "global_intermediate_size" else 0
            if value < minimum or value % 32:
                raise ValueError(
                    f"TrellisWeights.{name} must be a multiple of 32 at least {minimum}"
                )
        if self.intermediate_offset and self.global_intermediate_size is None:
            raise ValueError("intermediate_offset requires global_intermediate_size")


@dataclass(frozen=True)
class PackedWeights:
    """Ordinary packed MoE checkpoint tensors, without runtime policy fields.

    Activation scales are optional source metadata. A16 preparation uses unit
    scales; ModelOpt NVFP4 A4/A8 preparation requires both scale tensors.
    ``immutable_input_scales`` promises that input scale values remain unchanged
    throughout prepared bindings and graph replay; reprepare after mutation.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_block_scales: torch.Tensor
    w2_block_scales: torch.Tensor
    w13_global_scales: torch.Tensor
    w2_global_scales: torch.Tensor
    input_scale: torch.Tensor | None = None
    intermediate_scale: torch.Tensor | None = None
    immutable_input_scales: bool = False

    def __post_init__(self) -> None:
        if type(self.immutable_input_scales) is not bool:
            raise TypeError("immutable_input_scales must be boolean")
        for name in (
            "w13",
            "w2",
            "w13_block_scales",
            "w2_block_scales",
            "w13_global_scales",
            "w2_global_scales",
        ):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"PackedWeights.{name} must be a torch.Tensor")
        for name in ("input_scale", "intermediate_scale"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"PackedWeights.{name} must be a tensor or None")


@dataclass(frozen=True, kw_only=True)
class BtxWeights:
    """A whole-record BTX layer extent and its destination CUDA device."""

    layer: BtxLayer
    device: torch.device | str

    def __post_init__(self) -> None:
        from b12x.moe._shared.kernels.w4a16.btx import BtxLayer

        if not isinstance(self.layer, BtxLayer):
            raise TypeError("BTX weights require a BtxLayer extent")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("canonical BTX preparation requires a CUDA destination")
        object.__setattr__(self, "device", device)


@dataclass(frozen=True, kw_only=True)
class PreparedExperts:
    """Prepared expert tensors owned by a canonical weight plan."""

    plan: "WeightPlan"
    _impl: "B12XFP4ExpertWeights"

    def __post_init__(self) -> None:
        from ._impl import B12XFP4ExpertWeights
        from .planning import WeightPlan

        if not isinstance(self.plan, WeightPlan):
            raise TypeError("plan must be a canonical WeightPlan")
        if not isinstance(self._impl, B12XFP4ExpertWeights):
            raise TypeError("_impl must be prepared B12X expert weights")
        if self._impl.plan != self.plan._impl:
            raise ValueError("prepared experts do not match the canonical plan")

    @property
    def num_experts(self) -> int:
        return self._impl.num_experts

    @property
    def hidden_size(self) -> int:
        return self._impl.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self._impl.intermediate_size

    @property
    def device(self) -> torch.device:
        return self._impl.w1_fp4.device


__all__ = [
    "BtxWeights",
    "PackedWeights",
    "PreparedExperts",
    "PreparedWeightFormat",
    "ScaleEncoding",
    "ScaleFactors",
    "TrellisWeights",
    "WeightEncoding",
    "WeightPacking",
]
