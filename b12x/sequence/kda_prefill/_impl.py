"""Planned contract for chunked KDA prefill: caps, plan, bind, run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.policy import PolicyContext, get_auto_policy
from .._shared.kda_math import KDA_HEAD_DIM
from .._shared.tensors import canonical_device, positive
from .._shared.delta_prefill.contract import (
    Binding as _SharedBinding, Plan as _SharedPlan, bind_tensors, materialize_plan,
)
from ._policy import CHUNK_TOKENS, KDA_PREFILL_POLICY, KdaPrefillQuery, tiles_capacity

MetadataValidation = Literal["transactional", "trusted"]


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static geometry and planned capacity of a KDA prefill plan."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    heads: int
    head_dim: int = KDA_HEAD_DIM
    model_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    qk_l2norm: bool = True
    checkpoint_export: bool = False
    null_state_index: int | None = None
    metadata_validation: MetadataValidation = "transactional"
    chunk_tokens: int = 16

    def __post_init__(self) -> None:
        device = canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"KDA prefill requires a CUDA device, got {device}")
        object.__setattr__(self, "device", device)
        for name in ("max_tokens", "max_seqs", "max_state_slots", "heads"):
            object.__setattr__(self, name, positive(name, getattr(self, name)))
        if self.max_seqs > 4096:
            raise ValueError("max_seqs must be at most 4096")
        if int(self.head_dim) != KDA_HEAD_DIM:
            raise ValueError(f"head_dim must be {KDA_HEAD_DIM}, got {self.head_dim}")
        object.__setattr__(self, "head_dim", KDA_HEAD_DIM)
        if self.model_dtype != torch.bfloat16:
            raise ValueError("model_dtype must be torch.bfloat16")
        if self.state_dtype != torch.float32:
            raise ValueError("state_dtype must be torch.float32")
        if self.chunk_tokens != CHUNK_TOKENS:
            raise ValueError(f"chunk_tokens must be {CHUNK_TOKENS}")
        if self.metadata_validation not in ("transactional", "trusted"):
            raise ValueError("metadata_validation must be 'transactional' or 'trusted'")
        object.__setattr__(self, "qk_l2norm", bool(self.qk_l2norm))
        object.__setattr__(self, "checkpoint_export", bool(self.checkpoint_export))
        if self.null_state_index is not None:
            null = int(self.null_state_index)
            if null < 0 or null >= self.max_state_slots:
                raise ValueError("null_state_index must be a valid slot index")
            object.__setattr__(self, "null_state_index", null)

    @property
    def key_heads(self) -> int:
        return self.heads

    @property
    def is_gdn(self) -> bool:
        return False

    @property
    def op_name(self) -> str:
        return "kda_prefill"

    @property
    def tiles_capacity(self) -> int:
        """Upper bound on packed chunk tiles: one partial tile per sequence."""
        return tiles_capacity(self.max_tokens, self.max_seqs)


@dataclass(frozen=True)
class Plan(_SharedPlan):
    """Fixed KDA launch policy and caller-owned workspace contract."""

    caps: Caps
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def bind(self, **kwargs) -> "Binding":
        return bind(self, **kwargs)


@dataclass(frozen=True)
class Binding(_SharedBinding):
    """Caller-owned tensors for lower-bounded KDA prefill."""

    plan: Plan


def _query(caps: Caps) -> KdaPrefillQuery:
    return KdaPrefillQuery(
        heads=caps.heads,
        head_dim=caps.head_dim,
        model_dtype=str(caps.model_dtype).removeprefix("torch."),
        state_dtype=str(caps.state_dtype).removeprefix("torch."),
        qk_l2norm=caps.qk_l2norm,
        checkpoint_export=caps.checkpoint_export,
        max_tokens=caps.max_tokens,
        max_seqs=caps.max_seqs,
    )


def _materialize_plan(caps: Caps, **kwargs) -> Plan:
    return materialize_plan(caps, plan_type=Plan, **kwargs)


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve the policy once and lay out the scratch for ``caps``."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be kda_prefill.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(KDA_PREFILL_POLICY, _query(caps))
    return _materialize_plan(
        caps,
        v_split=int(resolution.config.v_split),
        k_split=int(resolution.config.k_split),
        stages=int(resolution.config.stages),
        window_tiles=int(resolution.config.window_tiles),
        policy_resolution=resolution,
    )


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> Binding:
    """Bind live tensors to a plan without allocating or launching work.

    Live capacities come from the bound tensors: ``q.shape[0]`` tokens and
    ``cu_seqlens.numel() - 1`` sequences, each at most the planned capacity.
    """
    if not isinstance(plan, Plan):
        raise TypeError("plan must be kda_prefill.Plan")
    return bind_tensors(
        plan, binding_type=Binding, scratch=scratch, q=q, k=k, v=v,
        raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices, final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices, checkpoint_offsets=checkpoint_offsets,
        num_seqs=num_seqs, num_tokens=num_tokens, output=output,
    )


def _check_run_scalars(lower_bound: float, scale: float | None, eps: float) -> tuple[float, float, float]:
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or not -5.0 <= lower_bound_value < 0.0:
        raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound_value}")
    scale_value = KDA_HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    return lower_bound_value, scale_value, eps_value


def run(
    binding: Binding,
    *,
    lower_bound: float,
    scale: float | None = None,
    eps: float = 1e-6,
    max_live_tokens: int | None = None,
    max_live_seqs: int | None = None,
) -> torch.Tensor:
    """Run the prologue, prepare, and recurrence kernels; capture safe.

    ``max_live_tokens`` and ``max_live_seqs`` are optional host-side upper
    bounds on the device counts; they only limit how many pipeline windows are
    launched. Under transactional validation a run whose live tiles exceed the
    launched windows fails closed like any other malformed metadata; under
    trusted validation the bounds are part of the caller's contract.

    A sequence whose tiles span more than one pipeline window keeps its
    running state in its final state slot between windows, so such a
    sequence must have a non-null final slot (transactional validation flags
    a null one as an invalid slot).
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be kda_prefill.Binding")
    lower_bound_value, scale_value, eps_value = _check_run_scalars(lower_bound, scale, eps)
    windows = binding.plan.launched_windows(max_live_tokens, max_live_seqs)
    from .._shared.delta_prefill._cute_kernels import run_prefill

    run_prefill(
        binding, lower_bound=lower_bound_value, scale=scale_value, eps=eps_value, windows=windows
    )
    return binding.output


def prewarm(binding: Binding) -> None:
    """Compile every kernel specialization of ``binding`` without launching."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be kda_prefill.Binding")
    from .._shared.delta_prefill._cute_kernels import prewarm_binding

    prewarm_binding(binding)


__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "bind",
    "plan",
    "prewarm",
    "run",
]
