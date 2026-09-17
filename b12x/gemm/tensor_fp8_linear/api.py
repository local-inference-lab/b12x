"""Public surface for ``gemm.tensor_fp8_linear``."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ..blockscaled._linear import (
    TensorFP8LinearWeight as Weight,
)
from ..blockscaled.api import FixedBlockscaledQuery, mm, pack_weight, plan, plan_regimes, query_from_call
from ._kernel import (
    is_tensor_fp8_linear_supported as _kernel_is_supported,
)
from . import META


def is_supported(device=None) -> bool:
    """Return whether the SM12x tensor-FP8 linear path is available."""
    kernel_supported, _ = _kernel_is_supported()
    return default_is_supported(device, requires=META.requires) and kernel_supported


__all__ = ["Weight", "FixedBlockscaledQuery", "plan", "plan_regimes", "query_from_call", "mm", "pack_weight", "is_supported"]
