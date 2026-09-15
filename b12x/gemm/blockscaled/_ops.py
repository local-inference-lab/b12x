"""Opaque prepared packed calls with exact caller-buffer mutation effects."""
from __future__ import annotations

import torch

from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x._lib.utils import cuda_stream_to_int


def _execute(source, values, scales, global_scale, activation_scale, workspace, out,
             global_scale_kind, stream, plan_handle, required_mode):
    state = require_prepared(plan_from_handle(plan_handle), "gemm.blockscaled_precision", source.device)
    resolve = getattr(state, "resolve", None)
    if resolve is not None:
        state = resolve(source)
    if required_mode != "any" and state.config.mode != required_mode:
        raise ValueError("this entry point requires an A16 prepared plan")
    if global_scale_kind != state.query.global_scale_kind:
        raise ValueError("weight scale semantics differ from prepared invocation")
    if state.query.recipe == "mxfp8" and scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e8m0fnu)
    return state.run(source, values, scales, global_scale, activation_scale=activation_scale,
                     workspace=workspace, out=out, stream=stream)


def _metadata(source, values):
    return source.new_empty((*source.shape[:-1], values.shape[0]), dtype=source.dtype)


@torch.library.custom_op("b12x::blockscaled_bf16", mutates_args=())
def _functional(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                global_scale_kind: str, stream: int | None, plan_handle: int, required_mode: str) -> torch.Tensor:
    return _execute(source, values, scales, global_scale, activation_scale, None, None,
                    global_scale_kind, stream, plan_handle, required_mode)


@_functional.register_fake
def _functional_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                     global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                     global_scale_kind: str, stream: int | None, plan_handle: int, required_mode: str) -> torch.Tensor:
    return _metadata(source, values)


@torch.library.custom_op("b12x::blockscaled_bf16_workspace", mutates_args=("workspace",))
def _functional_workspace(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                          global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                          workspace: torch.Tensor, global_scale_kind: str, stream: int | None,
                          plan_handle: int, required_mode: str) -> torch.Tensor:
    return _execute(source, values, scales, global_scale, activation_scale, workspace, None,
                    global_scale_kind, stream, plan_handle, required_mode)


@_functional_workspace.register_fake
def _functional_workspace_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                               global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                               workspace: torch.Tensor, global_scale_kind: str, stream: int | None,
                               plan_handle: int, required_mode: str) -> torch.Tensor:
    return _metadata(source, values)


@torch.library.custom_op("b12x::blockscaled_bf16_out", mutates_args=("out",))
def _out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
         global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
         out: torch.Tensor, global_scale_kind: str, stream: int | None,
         plan_handle: int, required_mode: str) -> None:
    _execute(source, values, scales, global_scale, activation_scale, None, out,
             global_scale_kind, stream, plan_handle, required_mode)


@_out.register_fake
def _out_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
              global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
              out: torch.Tensor, global_scale_kind: str, stream: int | None,
              plan_handle: int, required_mode: str) -> None:
    del source, values, scales, global_scale, activation_scale, out, global_scale_kind, stream, plan_handle


@torch.library.custom_op("b12x::blockscaled_bf16_out_workspace", mutates_args=("out", "workspace"))
def _out_workspace(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                   global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                   out: torch.Tensor, workspace: torch.Tensor, global_scale_kind: str,
                   stream: int | None, plan_handle: int, required_mode: str) -> None:
    _execute(source, values, scales, global_scale, activation_scale, workspace, out,
             global_scale_kind, stream, plan_handle, required_mode)


@_out_workspace.register_fake
def _out_workspace_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                        global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                        out: torch.Tensor, workspace: torch.Tensor, global_scale_kind: str,
                        stream: int | None, plan_handle: int, required_mode: str) -> None:
    del source, values, scales, global_scale, activation_scale, out, workspace, global_scale_kind, stream, plan_handle


def linear(source, values, scales, global_scale, *, plan: Plan,
           global_scale_kind: str, activation_global_scale=None, out=None, workspace=None, stream=None, required_mode="any"):
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    args = (source, values, scales, global_scale, activation_global_scale)
    stream_int = cuda_stream_to_int(stream)
    handle = plan.handle
    if out is None:
        if workspace is None:
            return torch.ops.b12x.blockscaled_bf16(*args, global_scale_kind, stream_int, handle, required_mode)
        return torch.ops.b12x.blockscaled_bf16_workspace(*args, workspace, global_scale_kind, stream_int, handle, required_mode)
    if workspace is None:
        torch.ops.b12x.blockscaled_bf16_out(*args, out, global_scale_kind, stream_int, handle, required_mode)
    else:
        torch.ops.b12x.blockscaled_bf16_out_workspace(*args, out, workspace, global_scale_kind, stream_int, handle, required_mode)
    return out
