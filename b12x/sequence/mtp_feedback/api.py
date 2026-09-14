"""Public surface for :mod:`b12x.sequence.mtp_feedback`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported

from . import reference
from ._impl import Binding, Caps, Plan, bind, plan, run
from ._preparation import invocation_from_tensors
from ._tuning import MtpFeedbackConfig, MtpFeedbackQuery


def is_supported(device=None, *, contract="qwen_multistream") -> bool:
    """Check the architecture and toolchain for the selected feedback contract."""
    if contract in {"rms_concat", "rms_streams_fp8"}:
        return default_is_supported(device, archs=("sm103a", "sm120a", "sm121a"))
    if contract == "qwen_multistream":
        return default_is_supported(
            device, requires=("triton",), archs=("sm103a", "sm120a", "sm121a")
        )
    return False


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "MtpFeedbackConfig",
    "MtpFeedbackQuery",
    "plan",
    "invocation_from_tensors",
    "bind",
    "run",
    "reference",
    "is_supported",
]
