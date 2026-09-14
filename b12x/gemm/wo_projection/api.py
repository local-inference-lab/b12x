"""Prepared public surface for native MXFP8 WO projection."""
from __future__ import annotations
from dataclasses import replace
from ..._lib.gating import default_is_supported
from ...preparation import Plan
from ...preparation.types import require_prepared
from .._shared.wo_mxfp8 import MXFP8Rows
from .._shared.wo_mxfp8 import WOProjectionBinding as Binding
from .._shared.wo_mxfp8 import WOProjectionInvRopeBinding as InvRopeBinding
from .._shared.wo_mxfp8 import WOProjectionMXFP8Weights as Weights
from .._shared.wo_mxfp8 import WOProjectionScratchCaps as Caps
from .._shared.wo_mxfp8 import pack_wo_projection_fp8_block_scaled_weights_mxfp8 as pack_weights
from .._shared import wo_mxfp8 as _shared
from ._preparation import plan
from ._tuning import WoProjectionConfig, WoProjectionQuery
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned WO tensors and scratch to a ready plan."""
    source = kwargs.get("source_tgd")
    state = require_prepared(plan, "gemm.wo_projection", source.device if source is not None else None)
    return replace(state.bind(**kwargs), plan=plan)


def bind_inv_rope(plan: Plan, **kwargs) -> InvRopeBinding:
    """Bind inverse-RoPE tensors and caller-owned scratch to a ready plan."""
    source = kwargs.get("o")
    state = require_prepared(plan, "gemm.wo_projection", source.device if source is not None else None)
    return replace(state.bind_inv_rope(**kwargs), plan=plan)

def run(*, binding: Binding, plan: Plan, stream=None):
    """Run a prepared WO projection; declarations and raw bindings are rejected."""
    state = require_prepared(plan, "gemm.wo_projection", binding.source_tgd.device)
    if binding.plan is not plan:
        raise ValueError("WO projection binding belongs to a different prepared plan")
    return state.run(binding, stream=stream)

def run_inv_rope(*, binding: InvRopeBinding, plan: Plan, stream=None):
    """Run a prepared inverse-RoPE WO projection."""
    state = require_prepared(plan, "gemm.wo_projection", binding.o.device)
    if binding.plan is not plan:
        raise ValueError("WO projection inverse-RoPE binding belongs to a different prepared plan")
    return state.run_inv_rope(binding, stream=stream)


def quantize_input(source_tgd, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", source_tgd.device)
    return state.quantize_a(source_tgd, out=out)


def quantize_input_inv_rope(o, positions, cos_sin_cache, *, groups, heads_per_group,
                             nope_dim=448, rope_dim=64, plan: Plan,
                             out=None):
    state = require_prepared(plan, "gemm.wo_projection", o.device)
    return state.quantize_a_inv_rope(
        o, positions, cos_sin_cache, groups=groups, heads_per_group=heads_per_group,
        nope_dim=nope_dim, rope_dim=rope_dim, out=out,
    )


def quantize_input_b(tmp_trg, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", tmp_trg.device)
    return state.quantize_b(tmp_trg, out=out)


@torch.no_grad()
def prewarm_inv_rope(
    plan: Plan,
    *,
    weights: Weights,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    positions_dtype: torch.dtype = torch.int64,
    scratch=None,
) -> None:
    """Resolve the plan's inverse-RoPE kernels before capture or resolution freeze.

    Scratch may be borrowed from a serving workspace. Temporary inputs and output
    are allocated only during this eager preparation step.
    """
    raise_if_kernel_resolution_frozen("WO inverse-RoPE prewarm")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("WO inverse-RoPE prewarm requires eager execution")
    if positions_dtype not in (torch.int32, torch.int64):
        raise ValueError("WO positions must use Int32 or Int64")
    capacity = plan.caps.max_tokens
    counts = (1,) if plan.backend == "mxfp8_tcgen05" else tuple(sorted({
        *range(1, min(capacity, 16) + 1), capacity,
    }))
    device = plan.caps.device
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        rows = max(counts)
        source = torch.ones(
            (rows, plan.caps.groups * heads_per_group, nope_dim + rope_dim),
            dtype=plan.caps.dtype, device=device,
        )
        positions = torch.zeros(rows, dtype=positions_dtype, device=device)
        output = torch.empty(rows, plan.caps.hidden, dtype=plan.caps.dtype, device=device)
        if scratch is None:
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                            for spec in plan.scratch_specs())
        for count in counts:
            binding = bind_inv_rope(
                plan, scratch=scratch, o=source[:count], positions=positions[:count],
                cos_sin_cache=cos_sin_cache, weights=weights, heads_per_group=heads_per_group,
                nope_dim=nope_dim, rope_dim=rope_dim, out=output[:count],
            )
            run_inv_rope(binding=binding)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps", "Plan", "Binding", "InvRopeBinding", "Weights", "MXFP8Rows",
    "WoProjectionConfig", "WoProjectionQuery", "plan", "bind", "bind_inv_rope",
    "run", "run_inv_rope", "pack_weights", "quantize_input",
    "quantize_input_inv_rope", "quantize_input_b", "is_supported",
]
