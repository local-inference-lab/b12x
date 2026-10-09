"""Public planned API for :mod:`b12x.attention.qsa`."""

from __future__ import annotations

from ._contract import (
    Binding,
    CacheRequirements,
    Caps,
    DraftSelectionPlan,
    DraftSelectionReuse,
    DraftSelectionState,
    LocalSelection,
    attend,
    attend_reuse,
    bind,
    cache_requirements,
    draft_selection_plan,
    invocation_from_descriptors,
    invocation_from_tensors,
    is_supported,
    plan,
    run,
    select,
    KVWriterBinding,
    bind_kv_writer,
)
from ._tuning import QsaConfig, QsaQuery
from ..paged._nvfp4_kv import NVFP4_KV_DTYPE
from b12x.preparation import Plan

__all__ = [
    "NVFP4_KV_DTYPE",
    "CacheRequirements",
    "Caps",
    "DraftSelectionPlan",
    "DraftSelectionReuse",
    "DraftSelectionState",
    "LocalSelection",
    "Plan",
    "Binding",
    "QsaConfig",
    "QsaQuery",
    "cache_requirements",
    "draft_selection_plan",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "plan",
    "bind",
    "run",
    "select",
    "attend",
    "attend_reuse",
    "is_supported",
    "KVWriterBinding",
    "bind_kv_writer",
]
