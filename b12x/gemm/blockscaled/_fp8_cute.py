"""Tensor-scaled and compact K128 FP8 GEMM with ordinary warp MMA.

The SM103 entry retains the shared TMA pipeline and arithmetic. Static weight
geometry selects a conservative tile; live rows and group strides are launch
arguments. The same entry can run on SM12x for regression qualification.
"""

import cuda.bindings.driver as cuda
import cutlass
import torch

from b12x._lib.dense_gemm import _empty_dense_gemm_output
from b12x._lib.fp8_gemm import DenseFp8Launch, compile_kernel  # noqa: F401 - compatibility exports
from b12x._lib.utils import cuda_stream_to_int
from ._sm103 import OUTPUT_TYPES, _grouped_layout, _span, pointer



def execute(lhs, rhs, out, *, ab_dtype, sf_dtype, c_dtype, sf_vec_size,
            block_fp8=False, alpha=None, stream=None):
    if ab_dtype != "float8_e4m3fn" or c_dtype not in OUTPUT_TYPES:
        raise ValueError("FP8 GEMM requires E4M3 operands and BF16/FP16/FP32 output")
    if (sf_dtype, sf_vec_size) != (("float32", 128) if block_fp8 else ("float8_e8m0fnu", 32)):
        raise ValueError("FP8 GEMM scale dtype and group size do not match the recipe")
    a, sfa = lhs
    b, sfb = rhs
    if a.device.type != "cuda" or b.device != a.device or a.ndim != 3 or b.ndim != 3:
        raise ValueError("FP8 operands must have shape [rows,K,groups] on one CUDA device")
    m, k, groups = a.shape
    n = b.shape[0]
    if min(n, k, groups) <= 0 or max(m, n, k) >= 2**31 or groups > 65535 or k % 128:
        raise ValueError("FP8 GEMM requires valid geometry and K divisible by 128")
    if b.shape[1:] != a.shape[1:] or a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError("FP8 operands must have matching K/groups and E4M3 storage")
    for tensor, rows in ((a, m), (b, n)):
        _grouped_layout(tensor, rows, k, groups)
    if groups > 1 and (a.stride(2) % 16 or b.stride(2) != n * k):
        raise ValueError("FP8 GEMM requires aligned activation groups and contiguous weight groups")
    dtype = OUTPUT_TYPES[c_dtype][0]
    if groups > 1 and n * OUTPUT_TYPES[c_dtype][1].width % 128:
        raise ValueError("unaligned FP8 output rows require one group")
    if block_fp8:
        if n % 128 or groups != 1:
            raise ValueError("compact block-FP8 requires N divisible by 128 and one group")
        for scale, shape in ((sfa, (m, k // 128)), (sfb, (n // 128, k // 128))):
            if (scale.device != a.device or scale.dtype != torch.float32
                    or tuple(scale.shape) != shape or not scale.is_contiguous()
                    or (scale.numel() and scale.data_ptr() % 16)):
                raise ValueError("compact block-FP8 scales require aligned contiguous FP32 [rows,K/128] storage")
    else:
        # Tensor scaling uses alpha exclusively. No scale allocation or transfer
        # may depend on the live row count.
        sfa = sfb = None
    if out is None:
        out = _empty_dense_gemm_output(m, n, groups, dtype=dtype, device=a.device)
    if out.device != a.device or out.dtype != dtype or tuple(out.shape) != (m, n, groups):
        raise ValueError("FP8 output must match the operand device, shape, and selected dtype")
    _grouped_layout(out, m, n, groups)
    if groups > 1 and out.stride(2) * out.element_size() % 16:
        raise ValueError("FP8 output group strides must be 16-byte aligned")
    if alpha is not None and (alpha.device != a.device or alpha.dtype != torch.float32
            or alpha.numel() != 1 or not alpha.is_contiguous() or alpha.data_ptr() % 16):
        raise ValueError("FP8 alpha must be an aligned FP32 scalar on the operand device")
    lo, hi = _span(out)
    if any(t is not None and lo < _span(t)[1] and _span(t)[0] < hi for t in (a, b, sfa, sfb, alpha)):
        raise ValueError("FP8 output must not overlap inputs")
    if m:
        _execute(a, b, sfa, sfb, out, alpha, block_fp8, c_dtype, cuda_stream_to_int(stream))
    return out


@torch.library.custom_op("b12x::fp8_warp", mutates_args=("out",))
def _execute(a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor | None,
             sfb: torch.Tensor | None, out: torch.Tensor, alpha: torch.Tensor | None,
             block_fp8: bool, c_dtype: str, stream_int: int | None) -> None:
    m, k, groups = a.shape
    with torch.cuda.device(a.device):
        props = torch.cuda.get_device_properties(a.device)
        fn = compile_kernel(b.shape[0], k, groups, c_dtype, block_fp8, alpha is None,
                            a.device.index, props.multi_processor_count, f"sm_{props.major}{props.minor}a")
        sf_type = cutlass.Float32 if block_fp8 else cutlass.Float8E8M0FNU
        types = (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, sf_type, sf_type,
                 OUTPUT_TYPES[c_dtype][1], cutlass.Float32)
        tensors = (a, b, sfa, sfb, out, alpha)
        fn(*(pointer(t, x) for t, x in zip(types, tensors, strict=True)), cutlass.Int32(m),
           cutlass.Int64(m * k if groups == 1 else a.stride(2)),
           cutlass.Int64(m * out.shape[1] if groups == 1 else out.stride(2)),
           cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream if stream_int is None else stream_int))


@_execute.register_fake
def _execute_fake(a, b, sfa, sfb, out, alpha, block_fp8, c_dtype, stream_int):
    return None
