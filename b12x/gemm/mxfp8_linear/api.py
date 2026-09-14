"""Public surface for gemm.mxfp8_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ..blockscaled._linear import (
    MXFP8LinearWeight as Weight,
)
from ..blockscaled.api import BlockscaledQuery, FixedBlockscaledQuery, mm, pack_weight, plan, query_from_call
from ._kernel import (
    is_mxfp8_linear_supported as _kernel_is_supported,
)
from . import META


def is_supported(device=None) -> bool:
    """Check the packed MXFP8 architecture and compiler dependencies."""
    kernel_supported, _ = _kernel_is_supported()
    return default_is_supported(device, archs=META.archs, requires=META.requires) and kernel_supported


__all__ = ["Weight", "BlockscaledQuery", "FixedBlockscaledQuery", "plan", "query_from_call", "mm", "pack_weight", "is_supported"]
