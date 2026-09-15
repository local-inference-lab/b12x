"""Caller-owned scratch for tensor-scaled and compact block-scaled FP8 GEMM."""

from __future__ import annotations

import torch

from b12x._lib.utils import cuda_stream_to_int
from ._a16 import _check_tensor, _overlap, _stream_context


def prepared_layout(query, lowering, use_block):
    """Reserve padding, unit scales, and split partials for the declared capacity."""
    rows, k = query.max_rows, query.padded_in_features
    padded_bytes = rows * k if query.in_features != k else 0
    scale_start = (padded_bytes + 255) // 256 * 256
    scale_bytes = rows * (k // 128) * 4 if query.recipe == "tensor_fp8" and use_block else 0
    partial_start = (scale_start + scale_bytes + 255) // 256 * 256
    policy = lowering.policy
    partial_bytes = (policy.split_k_slices * rows * query.out_features * 4
                     if policy.split_k_slices > 1 and not policy.split_k_atomic_bf16 else 0)
    return scale_start, partial_start, partial_start + partial_bytes


def _execute(source, values, source_scale, weight_scale, weight_block_scale,
             alpha, bias, out, workspace, tensor_scaled, plan_handle, out_dtype, stream):
    from b12x.preparation import plan_from_handle, require_prepared
    state = require_prepared(plan_from_handle(plan_handle), "gemm.blockscaled.fixed", source.device)
    query = state.query
    if not query.fp8_workspace:
        raise ValueError("FP8 workspace execution requires a bounded workspace declaration")
    if tensor_scaled != (query.recipe == "tensor_fp8"):
        raise ValueError("FP8 scale recipe differs from the prepared declaration")
    if (out is not None) != (query.output_mode == "provided"):
        raise ValueError("FP8 output ownership differs from the prepared declaration")
    if (workspace is not None) != (query.workspace_form == "provided"):
        raise ValueError("FP8 workspace ownership differs from the prepared declaration")
    if source.device.type != "cuda" or source.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 GEMM requires CUDA E4M3 activation values")
    _check_tensor("source", source, source.device, torch.float8_e4m3fn)
    _check_tensor("weight", values, source.device, torch.float8_e4m3fn)
    if source.ndim != 2 or values.ndim != 2:
        raise ValueError("FP8 activation values and weights must be 2D")
    m, input_k = source.shape
    n, k = values.shape
    if min(n, input_k, k) <= 0 or input_k > k or input_k % 32 or k % 128:
        raise ValueError("FP8 GEMM requires valid K32 input and K128 weight geometry")
    if not tensor_scaled and input_k != k:
        raise ValueError("compact block-FP8 input K must equal weight K")
    if (m > query.max_rows or input_k != query.in_features or k != query.padded_in_features
            or n != query.out_features or source.device != state.device):
        raise ValueError("FP8 execution differs from its prepared capacity/geometry/device")
    if out_dtype != getattr(torch, query.output_dtype):
        raise ValueError("FP8 output dtype differs from the prepared declaration")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FP8 output must be BF16 or FP16")
    if alpha is not None:
        _check_tensor("alpha", alpha, source.device, torch.float32)
        if alpha.numel() != 1:
            raise ValueError("FP8 alpha must be scalar")
    if bias is not None:
        _check_tensor("bias", bias, source.device, out_dtype)
        if bias.shape != (n,):
            raise ValueError("FP8 bias must have shape [N]")
    borrowed = (source, values, source_scale, weight_scale, weight_block_scale, alpha, bias)
    if out is None:
        with torch.cuda.device(source.device), _stream_context(stream, source.device):
            out = torch.empty((m, n), device=source.device, dtype=out_dtype)
    _check_tensor("out", out, source.device, out_dtype)
    if out.shape != (m, n):
        raise ValueError("FP8 output must have shape [M,N]")
    if any(t is not None and _overlap(t, out) for t in borrowed):
        raise ValueError("FP8 output must not overlap inputs")
    if not m:
        return out
    scale_start, partial_start, needed = prepared_layout(query, state.dense.lowering, state.use_block)
    if workspace is None:
        workspace = state.workspace
    _check_tensor("workspace", workspace, source.device, torch.uint8)
    if workspace.numel() < needed:
        raise ValueError(f"FP8 workspace requires at least {needed} bytes")
    if any(t is not None and _overlap(t, workspace) for t in (*borrowed, out)):
        raise ValueError("FP8 workspace must not overlap inputs or output")
    if not m:
        return out
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        if input_k != k:
            padded = workspace[:m * k].view(torch.float8_e4m3fn).view(m, k)
            padded.zero_()
            padded[:, :input_k].copy_(source)
        else:
            padded = source
        block = state.use_block if tensor_scaled else True
        if tensor_scaled:
            if block:
                source_scale = workspace[scale_start:scale_start + m * (k // 128) * 4].view(torch.float32).view(m, k // 128)
                source_scale.fill_(1.0)
                weight_scale = weight_block_scale
            else:
                weight_scale = weight_scale.view(torch.float8_e8m0fnu)
                source_scale = weight_scale
        if block:
            if n % 128:
                raise ValueError("block FP8 requires N divisible by 128")
            for name, scale, shape in (
                ("activation scale", source_scale, (m, k // 128)),
                ("weight scale", weight_scale, (n // 128, k // 128)),
            ):
                _check_tensor(name, scale, source.device, torch.float32)
                if scale.shape != shape:
                    raise ValueError(f"FP8 {name} must have shape {shape}")
        partial = workspace[partial_start:needed].view(torch.float32)
        state.dense.run(
            (padded.reshape(m, k, 1), source_scale),
            (values.reshape(n, k, 1), weight_scale), out=out.view(m, n, 1),
            alpha=alpha, stream=stream, split_k_workspace=partial,
        )
        if bias is not None:
            out.add_(bias)
    return out


@torch.library.custom_op("b12x::blockscaled_fp8_workspace", mutates_args=("workspace",))
def _functional(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> torch.Tensor:
    return _execute(source, values, source_scale, weight_scale, weight_block_scale,
                    alpha, bias, None, workspace, tensor_scaled, plan_handle, out_dtype, stream)


@_functional.register_fake
def _functional_fake(source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None, weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
                     alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None, tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None):
    return source.new_empty((source.shape[0], values.shape[0]), dtype=out_dtype)


@torch.library.custom_op("b12x::blockscaled_fp8_workspace_out", mutates_args=("workspace", "out"))
def _out(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None, out: torch.Tensor,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> None:
    _execute(source, values, source_scale, weight_scale, weight_block_scale,
             alpha, bias, out, workspace, tensor_scaled, plan_handle, out_dtype, stream)


@_out.register_fake
def _out_fake(source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None, weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
              alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None, out: torch.Tensor, tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None):
    return None


@torch.library.custom_op("b12x::blockscaled_fp8_owned", mutates_args=())
def _owned(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> torch.Tensor:
    return _execute(source, values, source_scale, weight_scale, weight_block_scale,
                    alpha, bias, None, None, tensor_scaled, plan_handle, out_dtype, stream)


@_owned.register_fake
def _owned_fake(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> torch.Tensor:
    return source.new_empty((source.shape[0], values.shape[0]), dtype=out_dtype)


@torch.library.custom_op("b12x::blockscaled_fp8_owned_out", mutates_args=("out",))
def _owned_out(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, out: torch.Tensor,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> None:
    _execute(source, values, source_scale, weight_scale, weight_block_scale,
             alpha, bias, out, None, tensor_scaled, plan_handle, out_dtype, stream)


@_owned_out.register_fake
def _owned_out_fake(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, out: torch.Tensor,
    tensor_scaled: bool, plan_handle: int, out_dtype: torch.dtype, stream: int | None,
) -> None:
    return None


def linear(source, values, source_scale, weight_scale, weight_block_scale, alpha,
           *, plan, out=None, workspace=None, bias=None, tensor_scaled=False,
           out_dtype=torch.bfloat16, stream=None):
    if weight_scale.dtype == torch.float8_e8m0fnu:
        weight_scale = weight_scale.view(torch.uint8)
    args = (source, values, source_scale, weight_scale, weight_block_scale, alpha, bias, workspace)
    options = (tensor_scaled, plan.handle, out_dtype, cuda_stream_to_int(stream))
    if workspace is None:
        if out is None:
            return _owned(*args[:-1], *options)
        _owned_out(*args[:-1], out, *options)
        return out
    if out is None:
        return _functional(*args, *options)
    _out(*args, out, *options)
    return out
