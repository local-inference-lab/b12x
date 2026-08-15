"""Public planned API for paged dense MLA."""

from __future__ import annotations

import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import (
    clear_dense_mla_kernel_caches,
    compile_dense_mla,
    run_dense_mla,
)
from ._scratch import Binding, Caps, Plan, Scratch
from ._scratch import (
    plan_dense_mla_scratch,
)
from .planner import Budget
from .planner import (
    infer_dense_mla_mode,
)
from ._reference import dense_mla_reference


def plan(caps: Caps) -> Plan:
    """Size fixed caller-owned storage and select shape-only kernel policy."""
    return plan_dense_mla_scratch(caps)


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors using views only; this function never allocates."""
    return plan.bind(**kwargs)


def compile(*, binding: Binding) -> None:
    """Compile every entry selected by ``binding`` without launching it."""
    compile_dense_mla(binding=binding)


def run(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dense MLA and return BF16 output plus FP32 natural-log LSE."""
    return run_dense_mla(binding=binding)


def reference(*args, **kwargs):
    """Run the FP32 paged dense-MLA oracle."""
    return dense_mla_reference(*args, **kwargs)


def infer_mode(cu_seqlens_q):
    return infer_dense_mla_mode(cu_seqlens_q)


def is_supported(device=None) -> bool:
    """True only for the fail-closed SM120/SM121 production envelope."""
    if not default_is_supported(device, requires=META.requires):
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    return tuple(torch.cuda.get_device_capability(device)) in ((12, 0), (12, 1))


def clear_caches() -> None:
    clear_dense_mla_kernel_caches()


__all__ = [
    "Binding",
    "Budget",
    "Caps",
    "Plan",
    "Scratch",
    "bind",
    "clear_caches",
    "compile",
    "infer_mode",
    "is_supported",
    "plan",
    "reference",
    "run",
]
