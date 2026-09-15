"""Public surface for :mod:`b12x.sequence.gdn_decode`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from b12x.preparation import Plan

from . import reference
from ._impl import (
    Binding,
    Caps,
    KdaBinding,
    bind,
    bind_kda,
    run,
    run_kda,
)
from ._preparation import plan, invocation_from_tensors
from ._tuning import GdnConfig, GdnQuery


def is_supported(device=None) -> bool:
    """True when mandatory Qwen CuTe and its Triton auxiliaries are usable."""
    return default_is_supported(device, requires=("triton",), archs=("sm103a", "sm120a", "sm121a"))


__all__ = [
    "Binding",
    "Caps",
    "GdnConfig",
    "GdnQuery",
    "KdaBinding",
    "Plan",
    "bind",
    "bind_kda",
    "is_supported",
    "plan",
    "invocation_from_tensors",
    "reference",
    "run",
    "run_kda",
]
