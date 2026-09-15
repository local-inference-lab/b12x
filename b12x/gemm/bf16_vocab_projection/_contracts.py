"""Declarative and prepared BF16 vocabulary projection boundary."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from b12x.preparation import FrozenMapping, Plan
from b12x.preparation.types import require_prepared

from ._kernel import bf16_vocab_projection  # noqa: F401
from ._tuning import Bf16VocabProjectionConfig


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    max_tokens: int
    in_features: int
    out_features: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        for name in ("max_tokens", "in_features", "out_features"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        if self.dtype != torch.bfloat16:
            raise TypeError("BF16 vocabulary projection requires torch.bfloat16")


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    source: torch.Tensor
    weight: torch.Tensor
    output: torch.Tensor | None = None


def plan(
    caps: Caps, *, invocation: FrozenMapping = FrozenMapping(),
    override: Bf16VocabProjectionConfig | None = None,
) -> Plan:
    """Declare vocabulary projection preparation without resolving a backend."""
    from ._preparation import make_plan

    return make_plan(caps, invocation=invocation, override=override)


def _bind(caps: Caps, *, plan: Plan, source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None) -> Binding:
    if source.ndim != 2 or not 0 < source.shape[0] <= caps.max_tokens:
        raise ValueError(f"source must have 1..{caps.max_tokens} rows, got {tuple(source.shape)}")
    if source.shape[1] != caps.in_features:
        raise ValueError(f"source K must be {caps.in_features}, got {source.shape[1]}")
    if tuple(weight.shape) != (caps.out_features, caps.in_features):
        raise ValueError(
            f"weight must have shape {(caps.out_features, caps.in_features)}, got {tuple(weight.shape)}"
        )
    for name, tensor in (("source", source), ("weight", weight)):
        if tensor.device != caps.device:
            raise ValueError(f"{name} must be on {caps.device}, got {tensor.device}")
        if tensor.dtype != caps.dtype:
            raise TypeError(f"{name} must have dtype {caps.dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if out is not None:
        if (out.shape != (source.shape[0], caps.out_features) or out.dtype != caps.dtype
                or out.device != caps.device or not out.is_contiguous()):
            raise ValueError("vocabulary output must match the contiguous BF16 projection")
        from ._cute import validate_output
        if not torch.compiler.is_compiling():
            validate_output(source, weight, out)
    return Binding(plan=plan, source=source, weight=weight, output=out)


def bind(plan: Plan, *, source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None) -> Binding:
    """Bind live tensors to a fully prepared vocabulary projection."""
    state = require_prepared(plan, "gemm.bf16_vocab_projection", source.device)
    return state.bind(plan=plan, source=source, weight=weight, out=out)


def run(binding: Binding) -> torch.Tensor:
    """Run only the backend selected during preparation."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be Binding")
    if binding.output is not None:
        torch.ops.b12x.bf16_vocab_projection_out(
            binding.source, binding.weight, binding.output, binding.plan.handle,
        )
        return binding.output
    return torch.ops.b12x.bf16_vocab_projection(
        binding.source, binding.weight, binding.plan.handle,
    )


__all__ = ["Binding", "Caps", "Plan", "bind", "plan", "run"]
