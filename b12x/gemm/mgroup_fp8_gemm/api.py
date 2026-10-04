"""Public surface for gemm.mgroup_fp8_gemm (docs in the op ``__init__``)."""
from __future__ import annotations

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from ._preparation import plan, query_from_call
from ._tuning import MGroupFP8GemmConfig, MGroupFP8GemmQuery
from . import META


def is_supported(device=None) -> bool:
    """Check SM120/SM121 capability and compiler/Triton availability.

    The supported package configuration pins CUTLASS DSL 4.7.1.

    This is an availability gate, not proof of correctness or performance on
    the device or support for every implementation/configuration. Defaults
    and configuration validation apply additional device constraints; see the
    op documentation. SM121 uses the single-body fallback and has not been
    functionally or performance-qualified for this op.
    """
    return default_is_supported(device, requires=META.requires)


def masked_mm(lhs, rhs, d, masked_m, *, plan: Plan, expected_m=None, stream=None):
    """Per-group masked FP8 GEMM: rows ``>= masked_m[g]`` of ``d[g]`` are
    contract-undefined (tiles fully past the mask are skipped, straddling
    rows compute garbage; the host never reads the device-side
    ``masked_m``). Callers must keep each count in ``[0, a.shape[1]]``;
    count values are not range-validated. A, B and D must be contiguous.
    SFA/SFB must be contiguous CUDA float32 tensors on the plan's device.
    Callers guarantee positive powers of two representable in UE8M0; there
    is no general scale-value validation. ``expected_m`` at execution does
    not replan; the query's fixed ``expected_m`` is a planning hint.
    The plan owns mutable capacity-sized scale buffers: serialize calls and
    graph replays, or use separate plans and outputs for concurrent execution."""
    state = require_prepared(plan, META.qualname)
    return state.run_masked(lhs, rhs, d, masked_m, expected_m=expected_m, stream=stream)


def contiguous_mm(lhs, rhs, d, labels, *, plan: Plan, stream=None):
    """Contiguous label-run FP8 GEMM; padding rows (label -1) come back zero.

    Labels must be -1 or in [0, num_groups), and each non-padding run must
    start at a multiple of 128. GPU validation reports violations as an
    asynchronous CUDA device error, including during graph replay.
    A, B and D must be contiguous; D has the live shape ``(m_total, n)``.
    Joint execution additionally requires 128-byte B base alignment.
    SFA/SFB must be contiguous CUDA float32 tensors on the plan's device;
    callers guarantee positive powers of two representable in UE8M0,
    without general scale-value validation. A contiguous plan owns mutable
    scale workspaces and is not reentrant: serialize calls and graph replays,
    or use separate plans and outputs for concurrent execution."""
    state = require_prepared(plan, META.qualname)
    return state.run_contiguous(lhs, rhs, d, labels, stream=stream)


__all__ = [
    "MGroupFP8GemmQuery",
    "MGroupFP8GemmConfig",
    "plan",
    "query_from_call",
    "masked_mm",
    "contiguous_mm",
    "is_supported",
]
