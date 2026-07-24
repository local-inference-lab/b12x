"""Public surface for gemm.bf16_gemv (docs in the op ``__init__``)."""

from __future__ import annotations

import os

import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import SMALL_M_MAX
from ._kernel import bf16_gemv_small_n  # noqa: F401  (registers the op; alias)
from ._kernel import (
    precompile_bf16_gemv_small_n as precompile,
)

# Routing thresholds for integrations (canonical home; formerly the vLLM
# plugin's constants). Unquantized bf16 linears with N <= MAX_OUT and
# K >= MIN_IN are worth routing through this small-N GEMV: catches narrow
# projections like the GDN ``in_proj_ba`` while excluding lm_head (N = vocab)
# and anything wide enough that cuBLAS tiles efficiently.
SMALL_N_GEMV_MAX_OUT = 1024
SMALL_N_GEMV_MIN_IN = 1024


def is_disabled() -> bool:
    """True when ``SPARKINFER_DISABLE_BF16_GEMV`` turns the routing off
    entirely (debug isolation switch)."""
    return os.environ.get("SPARKINFER_DISABLE_BF16_GEMV", "").lower() in (
        "1",
        "true",
        "yes",
    )


def mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for bf16 ``x (m, K)`` / ``weight (N, K)``.

    Dispatches through the opaque ``sparkinfer::bf16_gemv_small_n`` custom op
    (torch.compile- and CUDA-graph-safe); shapes the kernel does not cover
    fall back to ``F.linear`` inside the op.
    """
    return torch.ops.sparkinfer.bf16_gemv_small_n(x, weight)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, unless disabled
    via ``SPARKINFER_DISABLE_BF16_GEMV``."""
    if is_disabled():
        return False
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "mm",
    "bf16_gemv_small_n",
    "precompile",
    "is_supported",
    "is_disabled",
    "SMALL_M_MAX",
    "SMALL_N_GEMV_MAX_OUT",
    "SMALL_N_GEMV_MIN_IN",
]
