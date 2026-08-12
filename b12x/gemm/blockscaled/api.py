"""Public surface for gemm.blockscaled (docs in the op ``__init__``)."""

from __future__ import annotations

import torch

from ..._lib.dense_gemm import (
    dense_gemm as mm,
)
from ..._lib.dense_gemm import (
    dense_gemm_fused_quant_a as mm_fused_quant_a,
)
from ..._lib.dense_gemm import (
    dense_gemm_fused_quant_a_grouped as mm_fused_quant_a_grouped,
)
from ..._lib.gating import default_is_supported
from ..._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
from ..._lib.utils import cuda_stream_to_int
from . import META


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"block-scaled output must be bf16/fp16, got {dtype}")


@torch.library.custom_op(
    "b12x::blockscaled_mxfp4",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _blockscaled_mxfp4_op(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    m, packed_k = map(int, lhs_values.shape)
    n, rhs_packed_k = map(int, rhs_values.shape)
    if rhs_packed_k != packed_k:
        raise ValueError(
            "MXFP4 operands must have the same packed K extent, got "
            f"{packed_k} and {rhs_packed_k}"
        )
    k = packed_k * 2
    lhs_scale = as_grouped_scale_view_mx(
        lhs_scale_storage.view(torch.uint8).unsqueeze(0), m, k
    )
    rhs_scale = as_grouped_scale_view_mx(
        rhs_scale_storage.view(torch.uint8).unsqueeze(0), n, k
    )
    return mm(
        (lhs_values.reshape(m, packed_k, 1), lhs_scale),
        (rhs_values.reshape(n, packed_k, 1), rhs_scale),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=32,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_blockscaled_mxfp4_op.register_fake
def _blockscaled_mxfp4_fake(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del lhs_scale_storage, rhs_scale_storage, expected_m, stream_int
    return torch.empty(
        (lhs_values.shape[0], rhs_values.shape[0]),
        dtype=out_dtype,
        device=lhs_values.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_nvfp4",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _blockscaled_nvfp4_op(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    m, packed_k = map(int, lhs_values.shape)
    n, rhs_packed_k = map(int, rhs_values.shape)
    if rhs_packed_k != packed_k:
        raise ValueError(
            "NVFP4 operands must have the same packed K extent, got "
            f"{packed_k} and {rhs_packed_k}"
        )
    k = packed_k * 2
    lhs_scale = as_grouped_scale_view(
        lhs_scale_storage.view(torch.uint8).unsqueeze(0), m, k
    )
    rhs_scale = as_grouped_scale_view(
        rhs_scale_storage.view(torch.uint8).unsqueeze(0), n, k
    )
    return mm(
        (lhs_values.reshape(m, packed_k, 1), lhs_scale),
        (rhs_values.reshape(n, packed_k, 1), rhs_scale),
        alpha=alpha.reshape(1),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=16,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_blockscaled_nvfp4_op.register_fake
def _blockscaled_nvfp4_fake(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del lhs_scale_storage, rhs_scale_storage, alpha, expected_m, stream_int
    return torch.empty(
        (lhs_values.shape[0], rhs_values.shape[0]),
        dtype=out_dtype,
        device=lhs_values.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_block_fp8",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _blockscaled_block_fp8_op(
    lhs_values: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    m, k = map(int, lhs_values.shape)
    n, rhs_k = map(int, rhs_values.shape)
    if rhs_k != k:
        raise ValueError(
            f"block-FP8 operands must have the same K extent, got {k} and {rhs_k}"
        )
    return mm(
        (lhs_values.reshape(m, k, 1), lhs_scale),
        (rhs_values.reshape(n, k, 1), rhs_scale),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=128,
        block_fp8=True,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_blockscaled_block_fp8_op.register_fake
def _blockscaled_block_fp8_fake(
    lhs_values: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale: torch.Tensor,
    out_dtype: torch.dtype,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del lhs_scale, rhs_scale, expected_m, stream_int
    return torch.empty(
        (lhs_values.shape[0], rhs_values.shape[0]),
        dtype=out_dtype,
        device=lhs_values.device,
    )


def mm_mxfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run prequantized MXFP4 operands through an opaque B12X execution op."""
    return torch.ops.b12x.blockscaled_mxfp4(
        lhs_values,
        lhs_scale_storage,
        rhs_values,
        rhs_scale_storage,
        out_dtype,
        int(expected_m) if expected_m is not None else int(lhs_values.shape[0]),
        cuda_stream_to_int(stream),
    )


def mm_nvfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run prequantized NVFP4 operands through an opaque B12X execution op."""
    return torch.ops.b12x.blockscaled_nvfp4(
        lhs_values,
        lhs_scale_storage,
        rhs_values,
        rhs_scale_storage,
        alpha,
        out_dtype,
        int(expected_m) if expected_m is not None else int(lhs_values.shape[0]),
        cuda_stream_to_int(stream),
    )


def mm_block_fp8(
    lhs_values: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run compact 128x128 block-FP8 operands through an opaque B12X op."""
    return torch.ops.b12x.blockscaled_block_fp8(
        lhs_values,
        lhs_scale,
        rhs_values,
        rhs_scale,
        out_dtype,
        int(expected_m) if expected_m is not None else int(lhs_values.shape[0]),
        cuda_stream_to_int(stream),
    )


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "mm",
    "mm_mxfp4",
    "mm_nvfp4",
    "mm_block_fp8",
    "mm_fused_quant_a",
    "mm_fused_quant_a_grouped",
    "is_supported",
]
