from __future__ import annotations

from dataclasses import dataclass

import cutlass.cute as cute
import torch

from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.block_fp8 import (
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    empty_mxfp8_rows_for_dense_gemm,
    pack_mxfp8_scales_for_dense_gemm,
)


@dataclass(frozen=True)
class MXFP8LinearWeight:
    """ModelOpt-style MXFP8 linear weight packed for b12x dense GEMM."""

    weight: MXFP8Rows
    in_features: int
    padded_in_features: int
    out_features: int


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _c_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"b12x MXFP8 linear output dtype must be bf16/fp16, got {dtype}")


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim < 2:
        raise ValueError(f"source must have at least 2 dims, got {tuple(source.shape)}")
    return source.reshape(-1, source.shape[-1]).contiguous()


def _scale_rows_to_u8(scale_rows: torch.Tensor) -> torch.Tensor:
    if scale_rows.dtype == torch.uint8:
        return scale_rows.contiguous()
    if scale_rows.dtype == torch.float8_e8m0fnu:
        return scale_rows.view(torch.uint8).contiguous()
    raise ValueError(f"weight_scale must be uint8/e8m0, got {scale_rows.dtype}")


def _pad_weight_k(weight: torch.Tensor, padded_k: int) -> torch.Tensor:
    n, k = map(int, weight.shape)
    if k == padded_k:
        return weight.contiguous()
    padded = weight.new_zeros((n, padded_k))
    padded[:, :k] = weight
    return padded.contiguous()


def _pad_scale_rows_k(scale_rows_u8: torch.Tensor, padded_sf_k: int) -> torch.Tensor:
    n, sf_k = map(int, scale_rows_u8.shape)
    if sf_k == padded_sf_k:
        return scale_rows_u8.contiguous().view(torch.float8_e8m0fnu)
    padded = torch.full(
        (n, padded_sf_k),
        127,
        dtype=torch.uint8,
        device=scale_rows_u8.device,
    )
    padded[:, :sf_k] = scale_rows_u8
    return padded.contiguous().view(torch.float8_e8m0fnu)


def _pad_source_2d_k(source_2d: torch.Tensor, padded_k: int) -> torch.Tensor:
    tokens, k = map(int, source_2d.shape)
    if k == padded_k:
        return source_2d
    padded = source_2d.new_zeros((tokens, padded_k))
    padded[:, :k] = source_2d
    return padded.contiguous()


def is_mxfp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_mxfp8_linear_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> MXFP8LinearWeight:
    """Pack ModelOpt MXFP8 weights `[N,K]` and scales `[N,K/32]`.

    ModelOpt MXFP8 checkpoints require only `K % 32 == 0`. The underlying b12x
    dense MXFP8 GEMM consumes a 128-wide K tile, so this pads K to the next
    multiple of 128 with zero weights and UE8M0 1.0 scales.
    """

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("weight_scale", weight_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if weight_scale.ndim != 2:
        raise ValueError(
            f"weight_scale must have shape [N,K/32], got {tuple(weight_scale.shape)}"
        )

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "ModelOpt MXFP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    scale_k = in_features // MXFP8_SCALE_VEC_SIZE
    if (
        int(weight_scale.shape[0]) < out_features
        or int(weight_scale.shape[1]) < scale_k
    ):
        raise ValueError(
            "weight_scale must have at least shape "
            f"{(out_features, scale_k)}, got {tuple(weight_scale.shape)}"
        )

    padded_in_features = _align_up(in_features, 128)
    padded_scale_k = padded_in_features // MXFP8_SCALE_VEC_SIZE
    weight_values = _pad_weight_k(
        weight[:out_features, :in_features],
        padded_in_features,
    )
    scale_rows_u8 = _scale_rows_to_u8(weight_scale[:out_features, :scale_k])
    scale_rows = _pad_scale_rows_k(scale_rows_u8, padded_scale_k)
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=out_features,
        k=padded_in_features,
        num_groups=1,
    )

    return MXFP8LinearWeight(
        weight=MXFP8Rows(
            values=weight_values,
            scale_rows=scale_rows.reshape(1, out_features, padded_scale_k),
            scale_mma=scale_mma,
        ),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


@torch.library.custom_op(
    "b12x::mxfp8_linear_fused",
    mutates_args=(),
)
def _mxfp8_linear_fused_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del weight_scale_rows
    tokens = int(source_2d.shape[0])
    source_for_quant = _pad_source_2d_k(source_2d, int(padded_in_features))
    x_q = quantize_block_fp8_linear_input_mxfp8(source_for_quant)
    return dense_gemm(
        (x_q.values.reshape(tokens, padded_in_features, 1), x_q.scale_mma),
        (
            weight_values.reshape(out_features, padded_in_features, 1),
            weight_scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_c_dtype_name(source_2d.dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_mxfp8_linear_fused_op.register_fake
def _mxfp8_linear_fused_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    padded_in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> torch.Tensor:
    del stream_int
    del weight_values, weight_scale_rows, weight_scale_mma
    del in_features, padded_in_features, expected_m
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=source_2d.dtype,
        device=source_2d.device,
    )


def mxfp8_linear(
    source: torch.Tensor,
    packed_weight: MXFP8LinearWeight,
    *,
    bias: torch.Tensor | None = None,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run a ModelOpt MXFP8 linear through the native b12x dense GEMM path."""

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, MXFP8LinearWeight):
        raise TypeError("packed_weight must be an MXFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")

    out_features = int(packed_weight.out_features)
    if tokens == 0:
        output = source_2d.new_empty((0, out_features))
    else:
        output = torch.ops.b12x.mxfp8_linear_fused(
            source_2d,
            packed_weight.weight.values,
            packed_weight.weight.scale_rows,
            packed_weight.weight.scale_mma,
            packed_weight.in_features,
            packed_weight.padded_in_features,
            packed_weight.out_features,
            int(expected_m) if expected_m is not None else tokens,
            cuda_stream_to_int(stream),
        )
    if bias is not None:
        output = output + bias
    return output.view(*source.shape[:-1], out_features)


def empty_mxfp8_linear_input(
    max_tokens: int,
    in_features: int,
    *,
    device: torch.device | str,
) -> MXFP8Rows:
    """Allocate retained MXFP8 input storage for a dense linear."""

    return empty_mxfp8_rows_for_dense_gemm(
        int(max_tokens),
        int(in_features),
        num_groups=1,
        device=device,
    )


def quantize_mxfp8_linear_input_slice(
    source: torch.Tensor,
    destination: MXFP8Rows,
    *,
    destination_column: int,
) -> None:
    """Quantize one aligned feature slice into retained MXFP8 input storage.

    The temporary quantized slice is bounded by the source feature width. Its
    values and both scale layouts are copied into the corresponding 128-column
    interval of ``destination``. Standard tensor copies keep the operation
    compatible with torch functionalization while allowing the BF16 source to
    be released before the final matrix multiplication.
    """

    source_2d = _source_2d(source)
    tokens, width = map(int, source_2d.shape)
    if destination.values.ndim != 2:
        raise ValueError(
            "destination.values must have shape [capacity,K], got "
            f"{tuple(destination.values.shape)}"
        )
    capacity, destination_width = map(int, destination.values.shape)
    if capacity < tokens:
        raise ValueError(
            "destination has insufficient row capacity: "
            f"capacity={capacity}, tokens={tokens}"
        )
    destination_column = int(destination_column)
    if (
        destination_column < 0
        or destination_column % 128 != 0
        or width % 128 != 0
        or destination_column + width > destination_width
    ):
        raise ValueError(
            "source width and destination column must describe a 128-column-"
            f"aligned slice within K={destination_width}, got "
            f"offset={destination_column}, width={width}"
        )

    quantized = quantize_block_fp8_linear_input_mxfp8(source_2d)
    destination.values[:tokens, destination_column : destination_column + width].copy_(
        quantized.values
    )
    scale_column = destination_column // MXFP8_SCALE_VEC_SIZE
    scale_width = width // MXFP8_SCALE_VEC_SIZE
    destination.scale_rows[:, :tokens, scale_column : scale_column + scale_width].copy_(
        quantized.scale_rows
    )
    m_tiles = (tokens + 127) // 128
    tile_column = destination_column // 128
    tile_width = width // 128
    destination.scale_mma[
        :, :, :m_tiles, :, tile_column : tile_column + tile_width, :
    ].copy_(quantized.scale_mma)


@torch.library.custom_op(
    "b12x::mxfp8_linear_quantized",
    mutates_args=(),
)
def _mxfp8_linear_quantized_op(
    input_values: torch.Tensor,
    input_scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    tokens: int,
    in_features: int,
    out_features: int,
    expected_m: int,
    output_dtype: str,
    stream_int: int | None,
) -> torch.Tensor:
    m_tiles = (tokens + 127) // 128
    return dense_gemm(
        (
            input_values[:tokens].reshape(tokens, in_features, 1),
            input_scale_mma[:, :, :m_tiles],
        ),
        (
            weight_values.reshape(out_features, in_features, 1),
            weight_scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=output_dtype,
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
    )[:, :, 0]


@_mxfp8_linear_quantized_op.register_fake
def _mxfp8_linear_quantized_fake(
    input_values: torch.Tensor,
    input_scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    tokens: int,
    in_features: int,
    out_features: int,
    expected_m: int,
    output_dtype: str,
    stream_int: int | None,
) -> torch.Tensor:
    dtype = torch.bfloat16 if output_dtype == "bfloat16" else torch.float16
    return torch.empty(
        (tokens, out_features),
        dtype=dtype,
        device=input_values.device,
    )


@torch.library.custom_op(
    "b12x::mxfp8_linear_quantized_out",
    mutates_args=("out",),
)
def _mxfp8_linear_quantized_out_op(
    input_values: torch.Tensor,
    input_scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
    tokens: int,
    in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> None:
    """Run the retained-input GEMM directly into caller-owned storage."""
    m_tiles = (tokens + 127) // 128
    output_live = out[:tokens]
    dense_gemm(
        (
            input_values[:tokens].reshape(tokens, in_features, 1),
            input_scale_mma[:, :, :m_tiles],
        ),
        (
            weight_values.reshape(out_features, in_features, 1),
            weight_scale_mma,
        ),
        out=output_live.reshape(tokens, out_features, 1),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_c_dtype_name(out.dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
        stream=stream_int,
    )
    if bias is not None:
        output_live.add_(bias)


@_mxfp8_linear_quantized_out_op.register_fake
def _mxfp8_linear_quantized_out_fake(
    input_values: torch.Tensor,
    input_scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
    tokens: int,
    in_features: int,
    out_features: int,
    expected_m: int,
    stream_int: int | None,
) -> None:
    del (
        input_values,
        input_scale_mma,
        weight_values,
        weight_scale_mma,
        out,
        bias,
        tokens,
        in_features,
        out_features,
        expected_m,
        stream_int,
    )


def _validate_quantized_linear(
    source: MXFP8Rows,
    packed_weight: MXFP8LinearWeight,
    *,
    tokens: int | None = None,
    out: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[int, int, int, torch.dtype]:
    if not isinstance(source, MXFP8Rows):
        raise TypeError("source must be MXFP8Rows")
    if not isinstance(packed_weight, MXFP8LinearWeight):
        raise TypeError("packed_weight must be an MXFP8LinearWeight")
    if source.values.ndim != 2:
        raise ValueError(
            "source.values must have shape [capacity,K], got "
            f"{tuple(source.values.shape)}"
        )
    capacity, in_features = map(int, source.values.shape)
    live_tokens = capacity if tokens is None else int(tokens)
    if live_tokens <= 0 or live_tokens > capacity:
        raise ValueError(f"tokens must be in [1,{capacity}], got {live_tokens}")
    if in_features != int(packed_weight.padded_in_features):
        raise ValueError(
            "quantized input width does not match packed weight: "
            f"input={in_features}, weight={packed_weight.padded_in_features}"
        )
    if out is not None:
        _check_gpu_tensor("out", out)
        out_features = int(packed_weight.out_features)
        if out.ndim != 2 or out.shape[0] < live_tokens or out.shape[1] != out_features:
            raise ValueError(
                "out must have shape [capacity,N] with sufficient row capacity, "
                f"got {tuple(out.shape)}, required rows={live_tokens}, "
                f"N={out_features}"
            )
        output_dtype = out.dtype
    _c_dtype_name(output_dtype)
    out_features = int(packed_weight.out_features)
    return live_tokens, in_features, out_features, output_dtype


def mxfp8_linear_quantized(
    source: MXFP8Rows,
    packed_weight: MXFP8LinearWeight,
    *,
    tokens: int | None = None,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run a compile-safe packed linear from an already-quantized input."""

    live_tokens, in_features, out_features, output_dtype = _validate_quantized_linear(
        source,
        packed_weight,
        tokens=tokens,
        out=out,
        output_dtype=output_dtype,
    )
    result = torch.ops.b12x.mxfp8_linear_quantized(
        source.values,
        source.scale_mma,
        packed_weight.weight.values,
        packed_weight.weight.scale_mma,
        live_tokens,
        in_features,
        out_features,
        int(expected_m) if expected_m is not None else live_tokens,
        _c_dtype_name(output_dtype),
        cuda_stream_to_int(stream),
    )
    if bias is not None:
        result = result + bias
    if out is None:
        return result
    output_live = out[:live_tokens]
    output_live.copy_(result)
    return output_live


def mxfp8_linear_quantized_into(
    source: MXFP8Rows,
    packed_weight: MXFP8LinearWeight,
    *,
    out: torch.Tensor,
    tokens: int | None = None,
    bias: torch.Tensor | None = None,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run a packed linear directly into caller-owned eager output storage.

    The mutating operation avoids an intermediate dense output. Use
    :func:`mxfp8_linear_quantized` inside ``torch.compile``. The
    :func:`mxfp8_linear_quantized_into` operation is supported in eager
    execution and CUDA Graph replay, not under ``torch.compile``.
    """

    live_tokens, in_features, out_features, _ = _validate_quantized_linear(
        source,
        packed_weight,
        tokens=tokens,
        out=out,
    )
    torch.ops.b12x.mxfp8_linear_quantized_out(
        source.values,
        source.scale_mma,
        packed_weight.weight.values,
        packed_weight.weight.scale_mma,
        out,
        bias,
        live_tokens,
        in_features,
        out_features,
        int(expected_m) if expected_m is not None else live_tokens,
        cuda_stream_to_int(stream),
    )
    return out[:live_tokens]


__all__ = [
    "MXFP8LinearWeight",
    "empty_mxfp8_linear_input",
    "is_mxfp8_linear_supported",
    "mxfp8_linear",
    "mxfp8_linear_quantized",
    "mxfp8_linear_quantized_into",
    "pack_mxfp8_linear_weight",
    "quantize_mxfp8_linear_input_slice",
]
