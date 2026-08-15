from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass

import cutlass.cute as cute
import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    pack_mxfp8_scales_for_dense_gemm,
)


@dataclass(frozen=True)
class TensorFP8LinearWeight:
    """Serialized E4M3 weight and static scale metadata for dense GEMM."""

    values: torch.Tensor
    scale_mma: torch.Tensor
    output_scale: torch.Tensor
    in_features: int
    padded_in_features: int
    out_features: int


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"tensor FP8 linear output must be bf16/fp16, got {dtype}")


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim < 2:
        raise ValueError(f"source must have at least 2 dims, got {tuple(source.shape)}")
    return source.reshape(-1, source.shape[-1]).contiguous()


def _pad_k(tensor: torch.Tensor, padded_k: int) -> torch.Tensor:
    rows, width = map(int, tensor.shape)
    if width == padded_k:
        return tensor.contiguous()
    padded = tensor.new_zeros((rows, padded_k))
    padded[:, :width] = tensor
    return padded.contiguous()


def _unit_scale_mma(rows: int, width: int, device: torch.device) -> torch.Tensor:
    scale_rows = torch.full(
        (rows, width // MXFP8_SCALE_VEC_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=rows,
        k=width,
        num_groups=1,
    )


@functools.cache
def _cached_unit_scale_mma(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    return _unit_scale_mma(rows, width, torch.device(device_type, device_index))


def _activation_scale_mma(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_scale_mma(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


def _dense_gemm_kwargs_for_n(out_features: int) -> dict[str, object]:
    if int(out_features) < 64:
        return {"mma_tiler_mn": (64, 32), "swap_ab": True}
    return {}


def is_tensor_fp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_tensor_fp8_linear_weight(
    weight: torch.Tensor,
    output_scale: torch.Tensor,
) -> TensorFP8LinearWeight:
    """Pack an E4M3 ``[N,K]`` weight with one combined dequantization scale."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("output_scale", output_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if output_scale.dtype != torch.float32 or output_scale.numel() != 1:
        raise ValueError(
            "output_scale must be one float32 value, got "
            f"dtype={output_scale.dtype}, shape={tuple(output_scale.shape)}"
        )
    if output_scale.device != weight.device:
        raise ValueError("weight and output_scale must be on the same device")
    if not bool(torch.isfinite(output_scale).all()) or bool((output_scale < 0).any()):
        raise ValueError("output_scale must be finite and non-negative")

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "tensor FP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    padded_in_features = _align_up(in_features, 128)
    values = _pad_k(weight, padded_in_features)
    scale_mma = _unit_scale_mma(
        out_features,
        padded_in_features,
        weight.device,
    )
    return TensorFP8LinearWeight(
        values=values,
        scale_mma=scale_mma,
        output_scale=output_scale.reshape(1).contiguous(),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


@torch.library.custom_op(
    "b12x::tensor_fp8_linear_fused",
    mutates_args=(),
)
def _tensor_fp8_linear_fused_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    tokens = int(source_2d.shape[0])
    source_padded = _pad_k(source_2d, int(padded_in_features))
    source_scale_mma = _activation_scale_mma(
        source_padded,
        tokens,
        int(padded_in_features),
    )
    return dense_gemm(
        (
            source_padded.reshape(tokens, padded_in_features, 1),
            source_scale_mma,
        ),
        (
            weight_values.reshape(out_features, padded_in_features, 1),
            weight_scale_mma,
        ),
        alpha=output_scale,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
        plain_fp8=True,
        **_dense_gemm_kwargs_for_n(out_features),
    )[:, :, 0]


@_tensor_fp8_linear_fused_op.register_fake
def _tensor_fp8_linear_fused_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    output_scale: torch.Tensor,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_values, weight_scale_mma, output_scale
    del padded_in_features, expected_m, stream_int
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=out_dtype,
        device=source_2d.device,
    )


def tensor_fp8_linear(
    source: torch.Tensor,
    packed_weight: TensorFP8LinearWeight,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run static per-tensor E4M3 operands through the SM12x dense GEMM."""

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, TensorFP8LinearWeight):
        raise TypeError("packed_weight must be a TensorFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if source_2d.dtype != torch.float8_e4m3fn:
        raise ValueError(f"source must be float8_e4m3fn, got {source_2d.dtype}")
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if packed_weight.values.device != source_2d.device:
        raise ValueError("source and packed weight must be on the same device")
    if expected_m is not None and int(expected_m) <= 0:
        raise ValueError("expected_m must be positive when provided")
    _output_dtype_name(out_dtype)

    out_features = int(packed_weight.out_features)
    if bias is not None:
        _check_gpu_tensor("bias", bias)
        if bias.device != source_2d.device:
            raise ValueError("bias must be on the same device as source")
        if bias.dtype != out_dtype or bias.shape != (out_features,):
            raise ValueError(
                f"bias must have shape {(out_features,)} and dtype {out_dtype}, "
                f"got shape={tuple(bias.shape)}, dtype={bias.dtype}"
            )
    if tokens == 0:
        output = torch.empty(
            (0, out_features),
            dtype=out_dtype,
            device=source_2d.device,
        )
    else:
        output = torch.ops.b12x.tensor_fp8_linear_fused(
            source_2d,
            packed_weight.values,
            packed_weight.scale_mma,
            packed_weight.output_scale,
            packed_weight.padded_in_features,
            packed_weight.out_features,
            int(expected_m) if expected_m is not None else tokens,
            out_dtype,
            cuda_stream_to_int(stream),
        )
    if bias is not None:
        output = output + bias
    return output.view(*source.shape[:-1], out_features)


def prewarm_tensor_fp8_linear(
    packed_weight: TensorFP8LinearWeight,
    token_counts: Iterable[int],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> int:
    """Compile and cache serving shapes before CUDA graph capture."""

    warmed = 0
    with torch.inference_mode():
        for tokens in sorted({int(value) for value in token_counts if int(value) > 0}):
            source = torch.zeros(
                (tokens, packed_weight.in_features),
                dtype=torch.float8_e4m3fn,
                device=packed_weight.values.device,
            )
            tensor_fp8_linear(
                source,
                packed_weight,
                out_dtype=out_dtype,
                expected_m=tokens,
                stream=stream,
            )
            warmed += 1
    return warmed


__all__ = [
    "TensorFP8LinearWeight",
    "is_tensor_fp8_linear_supported",
    "pack_tensor_fp8_linear_weight",
    "prewarm_tensor_fp8_linear",
    "tensor_fp8_linear",
]
