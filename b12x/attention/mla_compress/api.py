"""Caller-owned CSA compression, before indexer projection and main-KV RoPE.

Packed metadata is device-resident. ``live_counts`` is int32[2] containing
(token_count, request_count), and may change on graph replay. Query starts are
int32[R+1], nondecreasing, with starts[0]=0 and starts[R_live]=T_live.
``positions[r]`` is the absolute position of that request's first packed token;
positions within each request are contiguous. ``state_ids[r]`` is a stable
caller-owned request identity in [0, max_states). Nonnegative live state IDs
must be unique within one call. A negative state ID or start position masks an
entire graph-padded request; empty requests preserve their pending state.

Outputs have fixed capacity [T,512], aligned to INPUT token rows: a pair's
latent appears at its second token, including a pair completed using old state.
Non-emitting/padded rows are zero, ``emitted`` is false, and ``emitted_slots`` is
-1. Emitting rows forward caller ``destination_slots[t]`` verbatim; no cache
allocation or RoPE position policy is performed here. A -1 destination does
not suppress compression/state transitions (it only means no cache store).

For ratio 2, pending_values/gates are FP32[max_states,512]; pending_position is
int64[max_states], initialized to -1 by the caller. Only an exact predecessor
tag is read. Recycled state IDs must be reset to -1 before a new request that
does not begin at position zero. Completed groups invalidate the tag; inactive
payload bytes are unspecified. Ratio 1 needs no gates or pending buffers.

Plan compiles; bind validates and retains views/pointers without device
allocation. Warm-run before capture. Calls sharing state must be ordered on the
same CUDA stream (or externally synchronized); inputs/outputs must not alias.
No GEMM, APE, overlap, rotation, quantization, or host metadata reads occur.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.gating import default_is_supported
from b12x.policy import PolicyContext, get_auto_policy
from ._policy import MLA_COMPRESS_POLICY, MlaCompressConfig, MlaCompressQuery


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    max_tokens: int
    max_requests: int
    max_states: int
    ratio: int

    def __post_init__(self):
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("MLA compression requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        for name in ("max_tokens", "max_requests", "max_states", "ratio"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        MLA_COMPRESS_POLICY.validate_config(self.query(), MlaCompressConfig(), None)

    def query(self):
        return MlaCompressQuery(ratio=self.ratio, max_tokens=self.max_tokens,
                                max_requests=self.max_requests, max_states=self.max_states)


@dataclass(frozen=True, kw_only=True)
class Plan:
    caps: Caps
    policy_resolution: object
    _compiled: object

    def scratch_specs(self) -> tuple:
        return ()

    def bind(self, **kwargs):
        return bind(self, **kwargs)


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    values: torch.Tensor
    gates: torch.Tensor | None
    weight: torch.Tensor
    query_start_loc: torch.Tensor
    positions: torch.Tensor
    state_ids: torch.Tensor
    destination_slots: torch.Tensor
    live_counts: torch.Tensor
    pending_values: torch.Tensor | None
    pending_gates: torch.Tensor | None
    pending_position: torch.Tensor | None
    out: torch.Tensor
    emitted: torch.Tensor
    emitted_slots: torch.Tensor
    _pointers: tuple


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve immutable policy once and precompile without launching kernels."""
    from ._cute import compile_compress

    if not isinstance(caps, Caps):
        raise TypeError("caps must be Caps")
    policy = policy or get_auto_policy(caps.device)
    policy.require_device(caps.device)
    resolution = policy.resolve(MLA_COMPRESS_POLICY, caps.query())
    compiled = compile_compress(caps.ratio, caps.max_tokens, caps.max_requests,
                                caps.max_states, caps.device.index)
    return Plan(caps=caps, policy_resolution=resolution, _compiled=compiled)


def bind(plan: Plan, *, values: torch.Tensor, weight: torch.Tensor,
         query_start_loc: torch.Tensor, positions: torch.Tensor,
         state_ids: torch.Tensor, destination_slots: torch.Tensor,
         live_counts: torch.Tensor, out: torch.Tensor, emitted: torch.Tensor,
         emitted_slots: torch.Tensor, gates: torch.Tensor | None = None,
         pending_values: torch.Tensor | None = None,
         pending_gates: torch.Tensor | None = None,
         pending_position: torch.Tensor | None = None) -> Binding:
    """Validate fixed-capacity caller buffers; retain pointers without allocation."""
    from ._cute import pointers

    c = plan.caps
    tensors = dict(values=values, weight=weight, query_start_loc=query_start_loc,
                   positions=positions, state_ids=state_ids,
                   destination_slots=destination_slots, live_counts=live_counts,
                   out=out, emitted=emitted, emitted_slots=emitted_slots)
    specs = dict(values=((c.max_tokens, 512), torch.float32 if c.ratio == 2 else torch.bfloat16),
                 weight=((512,), torch.float32),
                 query_start_loc=((c.max_requests + 1,), torch.int32),
                 positions=((c.max_requests,), torch.int64),
                 state_ids=((c.max_requests,), torch.int64),
                 destination_slots=((c.max_tokens,), torch.int64),
                 live_counts=((2,), torch.int32),
                 out=((c.max_tokens, 512), torch.bfloat16),
                 emitted=((c.max_tokens,), torch.bool),
                 emitted_slots=((c.max_tokens,), torch.int64))
    if c.ratio == 2:
        tensors.update(gates=gates, pending_values=pending_values,
                       pending_gates=pending_gates, pending_position=pending_position)
        specs.update(gates=((c.max_tokens, 512), torch.float32),
                     pending_values=((c.max_states, 512), torch.float32),
                     pending_gates=((c.max_states, 512), torch.float32),
                     pending_position=((c.max_states,), torch.int64))
    elif any(t is not None for t in (gates, pending_values, pending_gates, pending_position)):
        raise ValueError("ratio 1 does not consume gates or pending state")
    for name, tensor in tensors.items():
        shape, dtype = specs[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {dtype} with shape {shape}")
        if tensor.device != c.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {c.device}")
    mutable = {"out", "emitted", "emitted_slots", "pending_values", "pending_gates", "pending_position"}
    named = tuple(tensors.items())
    for i, (left_name, left) in enumerate(named):
        for right_name, right in named[i + 1:]:
            if left_name in mutable or right_name in mutable:
                if (left.data_ptr() < right.data_ptr() + right.numel() * right.element_size()
                        and right.data_ptr() < left.data_ptr() + left.numel() * left.element_size()):
                    raise ValueError(f"{left_name} and {right_name} must not overlap")
    # Unused ratio-1 pointers retain the compiled ABI's type without new tensors.
    args = (values, gates if gates is not None else weight, weight,
            query_start_loc, positions, state_ids, destination_slots, live_counts,
            pending_values if pending_values is not None else weight,
            pending_gates if pending_gates is not None else weight,
            pending_position if pending_position is not None else positions,
            out, emitted, emitted_slots)
    return Binding(plan=plan, values=values, gates=gates, weight=weight,
                   query_start_loc=query_start_loc, positions=positions,
                   state_ids=state_ids, destination_slots=destination_slots,
                   live_counts=live_counts, pending_values=pending_values,
                   pending_gates=pending_gates, pending_position=pending_position,
                   out=out, emitted=emitted, emitted_slots=emitted_slots,
                   _pointers=pointers(args))


def run(binding: Binding) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write normalized pre-RoPE latents and emission metadata, then commit carry."""
    from ._cute import launch

    launch(binding)
    return binding.out, binding.emitted, binding.emitted_slots


def is_supported(device=None):
    return default_is_supported(device)


__all__ = ["Caps", "Plan", "Binding", "MlaCompressQuery", "MlaCompressConfig",
           "plan", "bind", "run", "is_supported"]
