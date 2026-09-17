"""Public surface for :mod:`b12x.sequence.ple`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from b12x.preparation import Plan
from ._contracts import (
    LayerBinding as Binding,
    LayerCaps as Caps,
    bind_layer as bind,
    plan_layer as plan,
    run_decode,
    run_mixed,
    run_prefill,
)
from ._preparation import invocation_from_tensors
from ._tuning import PleConfig, PleQuery


def export_checkpoint(binding: Binding, *, offsets, slots) -> None:
    """Save internal prefill windows after the matching mixed PLE invocation.

    Offsets are relative to each request's query start. Negative slots and
    zero offsets disable export. Call before reusing the binding's scratch.
    Out-of-range destinations, destinations shared by enabled exports, and
    destinations overlapping live input/output state slots are skipped on
    the device, including during CUDA graph replay.
    """
    from b12x.preparation.types import require_prepared

    state = require_prepared(binding.plan, "sequence.ple")
    state.export_checkpoint(binding, offsets, slots)


def is_supported(device=None) -> bool:
    """True on supported b12x devices with Triton available."""
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PleConfig",
    "PleQuery",
    "plan",
    "invocation_from_tensors",
    "bind",
    "run_decode",
    "run_mixed",
    "run_prefill",
    "export_checkpoint",
    "is_supported",
]
