"""Caller-owned activation packing for the serialized MXFP4 GEMM interface."""

from __future__ import annotations

import torch
import triton

from b12x._lib.utils import cuda_stream_to_int
from ._a16 import _check_tensor, _overlap, _stream_context
from . import _quantize


@torch.library.custom_op("b12x::quantize_mxfp4", mutates_args=("out_values", "out_scales"))
def _quantize_mxfp4(
    source: torch.Tensor, out_values: torch.Tensor, out_scales: torch.Tensor,
    stream: int | None,
) -> None:
    if source.device.type != "cuda" or source.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MXFP4 activation packing requires CUDA BF16/FP16 input")
    if source.ndim != 2 or source.shape[1] <= 0 or source.shape[1] % 32:
        raise ValueError("MXFP4 source must have shape [M,K] with positive K divisible by 32")
    _check_tensor("source", source, source.device)
    m, k = source.shape
    if m > 2**31 - 1:
        raise ValueError("MXFP4 row count exceeds the Int32 launch limit")
    for name, tensor in (("out_values", out_values), ("out_scales", out_scales)):
        _check_tensor(name, tensor, source.device, torch.uint8)
    if out_values.shape != (m, k // 2):
        raise ValueError("MXFP4 values must have shape [M,K/2]")
    scale_bytes = triton.cdiv(m, 128) * triton.cdiv(k // 32, 4) * 512
    if out_scales.numel() != scale_bytes:
        raise ValueError("MXFP4 scales must hold exactly the padded F8_128x4 byte extent")
    if (_overlap(source, out_values) or _overlap(source, out_scales)
            or _overlap(out_values, out_scales)):
        raise ValueError("MXFP4 input and output buffers must not overlap")
    if m == 0:
        return
    with torch.cuda.device(source.device), _stream_context(stream, source.device):
        _quantize.launch(
            _quantize._quantize,
            (source, out_values, out_scales, None, None, None, m),
            dict(INPUT_K=k, K=k, FP4=True, RECIPROCAL=False, GROUP=32, CHUNKS=16),
            (m, triton.cdiv(k // 32, 16), 1), device=source.device, num_stages=1,
        )


@_quantize_mxfp4.register_fake
def _quantize_mxfp4_fake(source: torch.Tensor, out_values: torch.Tensor, out_scales: torch.Tensor, stream: int | None):
    return None


def quantize_mxfp4(
    source: torch.Tensor, *, out_values: torch.Tensor, out_scales: torch.Tensor,
    stream: object = None,
) -> None:
    """Pack BF16/FP16 rows into caller-owned E2M1 values and F8_128x4 UE8M0 scales.

    Values have shape ``[M,K/2]``. Scale storage holds
    ``ceil(M/128) * ceil((K/32)/4) * 512`` bytes. Both buffers must be contiguous,
    aligned uint8 tensors, disjoint from each other and the contiguous source.
    All padded scale entries are overwritten. Warm one nonempty call before
    capture; compiled callables depend on dtype and K, never live M.
    """
    _quantize_mxfp4(source, out_values, out_scales, cuda_stream_to_int(stream))
