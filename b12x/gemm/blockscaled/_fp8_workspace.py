"""Caller-owned scratch for tensor-scaled and compact block-scaled FP8 GEMM."""

from __future__ import annotations

import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.gating import get_compute_capability
from b12x._lib.utils import cuda_stream_to_int, get_num_sm
from ._a16 import _check_tensor, _overlap, _stream_context


def _layout(rows, n, input_k, padded_k, tensor_scaled):
    padded_bytes = rows * padded_k if input_k != padded_k else 0
    scale_start = (padded_bytes + 255) // 256 * 256
    scale_bytes = min(rows, 8) * (padded_k // 128) * 4 if tensor_scaled else 0
    partial_start = (scale_start + scale_bytes + 255) // 256 * 256
    # The dense FP8 policy splits at most four ways, only for capacities <= 8.
    partial_bytes = 4 * min(rows, 8) * n * 4
    return scale_start, partial_start, partial_start + partial_bytes


def workspace_size(weight, max_tokens: int) -> int:
    from ._linear import TensorFP8LinearWeight
    if not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    if isinstance(weight, TensorFP8LinearWeight):
        return _layout(max_tokens, weight.out_features, weight.in_features,
                       weight.padded_in_features, True)[-1]
    values, scales = weight
    if (values.ndim != 2 or values.dtype != torch.float8_e4m3fn
            or values.shape[0] <= 0 or values.shape[0] % 128
            or values.shape[1] <= 0 or values.shape[1] % 128
            or scales.dtype != torch.float32
            or scales.shape != (values.shape[0] // 128, values.shape[1] // 128)):
        raise ValueError("FP8 workspace requires compact K128 block-scaled weights")
    return _layout(max_tokens, values.shape[0], values.shape[1], values.shape[1], False)[-1]


def _execute(source, values, source_scale, weight_scale, weight_block_scale,
             alpha, bias, out, workspace, tensor_scaled, expected_m, out_dtype, stream):
    from ._linear import _use_block_fp8_recipe
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
    if expected_m is not None and expected_m <= 0:
        raise ValueError("expected_m must be positive when provided")
    if workspace is not None and expected_m is not None and m > expected_m:
        raise ValueError("FP8 workspace execution requires a covering expected_m capacity")
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
    scale_start, partial_start, needed = _layout(m, n, input_k, k, tensor_scaled)
    if workspace is None:
        # The allocator must retain scratch until work on the launch stream retires.
        with torch.cuda.device(source.device), _stream_context(stream, source.device):
            workspace = torch.empty(needed, device=source.device, dtype=torch.uint8)
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
        block = not tensor_scaled
        if tensor_scaled:
            block = get_compute_capability(source.device) != (10, 3) and _use_block_fp8_recipe(
                live_m=m, expected_m=expected_m if expected_m is not None else m,
                out_features=n, padded_in_features=k, sm_count=get_num_sm(source.device),
            )
            if block:
                source_scale = workspace[scale_start:scale_start + m * (k // 128) * 4].view(torch.float32).view(m, k // 128)
                source_scale.fill_(1.0)
                weight_scale = weight_block_scale
            else:
                # Plain FP8 consumes alpha alone; the launch ignores scale pointers.
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
        dense_gemm(
            (padded.reshape(m, k, 1), source_scale),
            (values.reshape(n, k, 1), weight_scale), out=out.view(m, n, 1),
            alpha=alpha, ab_dtype="float8_e4m3fn",
            sf_dtype="float32" if block else "float8_e8m0fnu",
            c_dtype="bfloat16" if out_dtype == torch.bfloat16 else "float16",
            sf_vec_size=128 if block else 32, block_fp8=block, plain_fp8=not block,
            expected_m=expected_m, stream=stream, _split_k_workspace=partial,
        )
        if bias is not None:
            out.add_(bias)
    return out


@torch.library.custom_op("b12x::blockscaled_fp8_workspace", mutates_args=("workspace",))
def _functional(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None,
    tensor_scaled: bool, expected_m: int | None, out_dtype: torch.dtype, stream: int | None,
) -> torch.Tensor:
    return _execute(source, values, source_scale, weight_scale, weight_block_scale,
                    alpha, bias, None, workspace, tensor_scaled, expected_m, out_dtype, stream)


@_functional.register_fake
def _functional_fake(source, values, source_scale, weight_scale, weight_block_scale,
                     alpha, bias, workspace, tensor_scaled, expected_m, out_dtype, stream):
    return source.new_empty((source.shape[0], values.shape[0]), dtype=out_dtype)


@torch.library.custom_op("b12x::blockscaled_fp8_workspace_out", mutates_args=("workspace", "out"))
def _out(
    source: torch.Tensor, values: torch.Tensor, source_scale: torch.Tensor | None,
    weight_scale: torch.Tensor, weight_block_scale: torch.Tensor | None,
    alpha: torch.Tensor | None, bias: torch.Tensor | None, workspace: torch.Tensor | None, out: torch.Tensor,
    tensor_scaled: bool, expected_m: int | None, out_dtype: torch.dtype, stream: int | None,
) -> None:
    _execute(source, values, source_scale, weight_scale, weight_block_scale,
             alpha, bias, out, workspace, tensor_scaled, expected_m, out_dtype, stream)


@_out.register_fake
def _out_fake(source, values, source_scale, weight_scale, weight_block_scale,
              alpha, bias, workspace, out, tensor_scaled, expected_m, out_dtype, stream):
    return None


def linear(source, values, source_scale, weight_scale, weight_block_scale, alpha,
           *, out=None, workspace=None, bias=None, tensor_scaled=False, expected_m=None,
           out_dtype=torch.bfloat16, stream=None):
    if weight_scale.dtype == torch.float8_e8m0fnu:
        weight_scale = weight_scale.view(torch.uint8)
    args = (source, values, source_scale, weight_scale, weight_block_scale, alpha, bias, workspace)
    options = (tensor_scaled, expected_m, out_dtype, cuda_stream_to_int(stream))
    if out is None:
        return _functional(*args, *options)
    _out(*args, out, *options)
    return out
