"""Prepared public surface for native paged attention."""
from __future__ import annotations

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan

from . import META
from ._forward import clear_paged_caches as clear_caches
from ._preparation import bind, invocation_from_descriptors, invocation_from_tensors, memory_requirements, plan, run
from ._scratch import (
    B12XPagedAttentionBinding as Binding,
    B12XPagedAttentionScratchCaps as Caps,
    B12XPagedDecodeGraphScratchEnvelope as DecodeGraphScratchEnvelope,
    plan_decode_graph_scratch_envelope as decode_graph_scratch_envelope,
)
from ._tuning import GqaConfig, GqaQuery
from .planner import (
    PagedDecodeGraphCapacity as DecodeGraphCapacity,
    PagedExtendGraphCapacity as ExtendGraphCapacity,
    PagedPlanBudget as Budget,
    PagedVerifyGraphCapacity as VerifyGraphCapacity,
    infer_paged_mode as infer_mode,
    plan_decode_graph_capacity as decode_graph_capacity,
    plan_extend_graph_capacity as extend_graph_capacity,
    plan_verify_graph_capacity as verify_graph_capacity,
)
from .workspace import PagedAttentionWorkspace as Workspace


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires, archs=META.archs)


__all__ = [
    "Caps", "Plan", "Binding", "Workspace", "Budget", "DecodeGraphCapacity",
    "GqaConfig", "GqaQuery", "ExtendGraphCapacity", "VerifyGraphCapacity",
    "DecodeGraphScratchEnvelope", "decode_graph_capacity", "extend_graph_capacity",
    "verify_graph_capacity", "decode_graph_scratch_envelope", "plan", "bind",
    "run", "invocation_from_descriptors", "invocation_from_tensors", "memory_requirements", "infer_mode", "is_supported", "clear_caches",
]
