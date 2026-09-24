"""Checkpoint-side fused-MoE weight representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from .config import TrellisConfig

if TYPE_CHECKING:
    from b12x.moe._shared.exl3_schema import Exl3Manifest


class PackedSourceFormat(str, Enum):
    """Packed checkpoint encoding, including its scale-grid contract."""

    MXFP4_E8M0_K32 = "fp4_e8m0_k32"
    MODELOPT_NVFP4 = "modelopt_nvfp4"
    COMPRESSED_TENSORS_FP4 = "compressed_tensors"
    MXFP6_E8M0_K32 = "mxfp6_e2m3"
    IQ2_XS = "iq2_xs"


class W13Layout(str, Enum):
    """Logical order of the two gated FC1 projections."""

    W13 = "w13"
    W31 = "w31"


@dataclass(frozen=True, kw_only=True)
class PackedSource:
    """Source encoding and packing of one ordinary MoE checkpoint."""

    format: PackedSourceFormat
    w13_layout: W13Layout = W13Layout.W13

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", PackedSourceFormat(self.format))
        object.__setattr__(self, "w13_layout", W13Layout(self.w13_layout))


@dataclass(frozen=True, kw_only=True)
class Exl3Source:
    """Validated EXL3 manifest describing native expert records and transforms."""

    manifest: Exl3Manifest

    def __post_init__(self) -> None:
        from b12x.moe._shared.exl3_schema import Exl3Manifest

        if not isinstance(self.manifest, Exl3Manifest):
            raise TypeError("EXL3 source requires a parsed Exl3Manifest")


WeightSource: TypeAlias = PackedSource | TrellisConfig | Exl3Source


__all__ = [
    "Exl3Source",
    "PackedSource",
    "PackedSourceFormat",
    "W13Layout",
    "WeightSource",
]
