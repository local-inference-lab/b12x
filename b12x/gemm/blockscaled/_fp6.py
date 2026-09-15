"""Native SM103 mixed FP6/FP8 GEMM with packed or byte-container operands."""


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from b12x._lib.compiler import run_compiled
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm
from ._sm103 import OUTPUT_TYPES, _grouped_layout, _scale_storage, _span, pointer


class DenseFP6Gemm(BlockscaledGemm):
    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, sfa: cute.Pointer,
                 sfb: cute.Pointer, out: cute.Pointer, alpha: cute.Pointer,
                 row_scale: cute.Pointer, rows: cutlass.Int32,
                 a_stride: cutlass.Int64, c_stride: cutlass.Int64,
                 alpha_stride: cutlass.Int64, stream: cuda.CUstream):
        self._launch(a, b, sfa, sfb, out, None, alpha, rows, a_stride,
                     c_stride, alpha_stride, stream, row_scale)


from b12x._lib.program_cache import program_cache


@program_cache
def compile_kernel(n, k, groups, a_fmt, b_fmt, a_bytes, b_bytes, c_dtype,
                   alpha_is_one, apply_row_scale, device_ordinal):
    raise_if_kernel_resolution_frozen("SM103 MXFP6 GEMM")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM103 MXFP6 GEMM must be prewarmed before graph capture")
    kernel = DenseFP6Gemm(n, k, groups, recipe="mxfp6", c_dtype=OUTPUT_TYPES[c_dtype][1],
                          a_fmt=a_fmt, b_fmt=b_fmt, a_preexpanded=a_bytes,
                          b_preexpanded=b_bytes, alpha_is_one=alpha_is_one,
                          apply_row_scale=apply_row_scale)
    types = (kernel.a_gmem_dtype, kernel.b_gmem_dtype, kernel.sf_dtype,
             kernel.sf_dtype, kernel.c_dtype, cutlass.Float32, cutlass.BFloat16)
    key = (n, k, groups, a_fmt, b_fmt, a_bytes, b_bytes, c_dtype,
           alpha_is_one, apply_row_scale, device_ordinal)
    return b12x_compile(
        kernel, *(pointer(t) for t in types), cutlass.Int32(1),
        cutlass.Int64(k), cutlass.Int64(n), cutlass.Int64(1), cuda.CUstream(0),
        compile_spec=KernelCompileSpec.from_key("gemm.blockscaled.sm103.fp6", 1, key),
        options="--gpu-arch=sm_103a",
    )


def execute(lhs, rhs, out, *, ab_dtype, sf_dtype, sf_vec_size, c_dtype,
            a_fmt=None, b_fmt=None, a_preexpanded=False, b_preexpanded=False,
            b_packed=False, alpha=None, row_scale=None, stream=None, compiled=None):
    if ab_dtype not in ("float6_e2m3fn", "float6_e3m2fn") or sf_dtype != "float8_e8m0fnu" or sf_vec_size != 32:
        raise ValueError("SM103 MXFP6 GEMM requires FP6 precision and K32 UE8M0 scales")
    if c_dtype not in OUTPUT_TYPES:
        raise ValueError("SM103 MXFP6 output must be BF16, FP16, or FP32")
    if b_packed and b_preexpanded:
        raise ValueError("b_packed and b_preexpanded are mutually exclusive")
    default = "e2m3" if ab_dtype == "float6_e2m3fn" else "e3m2"
    a_fmt, b_fmt = a_fmt or default, b_fmt or default
    if a_fmt not in ("e2m3", "e3m2", "e4m3") or b_fmt not in ("e2m3", "e3m2", "e4m3"):
        raise ValueError("SM103 MXFP6 operand formats must be e2m3, e3m2, or e4m3")
    a_bytes, b_bytes = a_preexpanded or a_fmt == "e4m3", b_preexpanded or b_fmt == "e4m3"
    if b_packed and b_fmt == "e4m3":
        raise ValueError("E4M3 weights require byte storage")
    a, sfa = lhs
    b, sfb = rhs
    if a.device.type != "cuda" or b.device != a.device or a.ndim != 3 or b.ndim != 3:
        raise ValueError("SM103 MXFP6 operands require [rows,K,groups] on one CUDA device")
    m, stored_k, groups = a.shape
    k = stored_k if a_bytes else stored_k * 4 // 3
    n = b.shape[0]
    if min(k, n, groups) <= 0 or k % 128 or k >= 2**31 or n % 8 or n > 128 * 65535 or m >= 2**31 or groups > 65535:
        raise ValueError("SM103 MXFP6 requires K divisible by 128, N by eight, and a valid grid")
    if stored_k != (k if a_bytes else k * 3 // 4) or tuple(b.shape[1:]) != (k if b_bytes else k * 3 // 4, groups):
        raise ValueError("MXFP6 operands require K byte containers or 3K/4 packed bytes")
    for tensor, fmt, byte_storage in ((a, a_fmt, a_bytes), (b, b_fmt, b_bytes)):
        if tensor.dtype not in ((torch.uint8, torch.float8_e4m3fn) if fmt == "e4m3" else (torch.uint8,)):
            raise ValueError("MXFP6 values require byte storage matching their declared format")
        _grouped_layout(tensor, tensor.shape[0], tensor.shape[1], groups)
        if not byte_storage and tensor.numel() and tensor.data_ptr() % 32:
            raise ValueError("Packed FP6 TMA operands require 32-byte alignment")
    if groups > 1 and (a.stride(2) % (16 if a_bytes else 96) or b.stride(2) != n * b.shape[1]):
        raise ValueError("MXFP6 requires aligned activation groups and contiguous weight groups")
    sfa = _scale_storage(sfa, m, k, groups, 32, a.device, torch.float8_e8m0fnu)
    sfb = _scale_storage(sfb, n, k, groups, 32, a.device, torch.float8_e8m0fnu)
    if out is None:
        from b12x._lib.dense_gemm import _empty_dense_gemm_output
        out = _empty_dense_gemm_output(m, n, groups, dtype=OUTPUT_TYPES[c_dtype][0], device=a.device)
    if out.device != a.device or out.dtype != OUTPUT_TYPES[c_dtype][0] or tuple(out.shape) != (m, n, groups):
        raise ValueError("MXFP6 output must match the operand device, shape, and selected dtype")
    _grouped_layout(out, m, n, groups)
    if alpha is not None and (alpha.device != a.device or alpha.dtype != torch.float32
            or alpha.numel() not in (1, groups) or not alpha.is_contiguous() or alpha.data_ptr() % 16):
        raise ValueError("MXFP6 alpha must be an aligned contiguous FP32 scalar or one value per group")
    if row_scale is not None and (row_scale.device != a.device or row_scale.dtype != torch.bfloat16
            or tuple(row_scale.shape) != (m,) or not row_scale.is_contiguous()
            or row_scale.numel() and row_scale.data_ptr() % 16):
        raise ValueError("MXFP6 row_scale requires aligned contiguous BF16 [M] storage")
    reads = [a, b, sfa, sfb] + [t for t in (alpha, row_scale) if t is not None]
    lo, hi = _span(out)
    if any(lo < _span(t)[1] and _span(t)[0] < hi for t in reads):
        raise ValueError("MXFP6 output must not overlap inputs")
    if m:
        if compiled is None:
            _execute(a, b, sfa, sfb, out, alpha, row_scale, a_fmt, b_fmt, a_bytes, b_bytes, c_dtype, cuda_stream_to_int(stream))
        else:
            _launch_prepared(compiled, a, b, sfa, sfb, out, alpha, row_scale, a_fmt, b_fmt, a_bytes, b_bytes, c_dtype, cuda_stream_to_int(stream))
    return out


@torch.library.custom_op("b12x::sm103_fp6", mutates_args=("out",))
def _execute(a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor, sfb: torch.Tensor,
             out: torch.Tensor, alpha: torch.Tensor | None, row_scale: torch.Tensor | None,
             a_fmt: str, b_fmt: str, a_bytes: bool, b_bytes: bool,
             c_dtype: str, stream_int: int | None) -> None:
    m, stored_k, groups = a.shape
    k = stored_k if a_bytes else stored_k * 4 // 3
    types = {"e2m3": cutlass.Float6E2M3FN, "e3m2": cutlass.Float6E3M2FN,
             "e4m3": cutlass.Float8E4M3FN}
    with torch.cuda.device(a.device):
        fn = compile_kernel(b.shape[0], k, groups, a_fmt, b_fmt, a_bytes, b_bytes,
                            c_dtype, alpha is None, row_scale is not None, a.device.index)
        _launch_prepared(fn, a, b, sfa, sfb, out, alpha, row_scale, a_fmt, b_fmt, a_bytes, b_bytes, c_dtype, stream_int)


def _launch_prepared(fn, a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor, sfb: torch.Tensor,
             out: torch.Tensor, alpha: torch.Tensor | None, row_scale: torch.Tensor | None,
             a_fmt: str, b_fmt: str, a_bytes: bool, b_bytes: bool,
             c_dtype: str, stream_int: int | None) -> None:
    m, stored_k, groups = a.shape
    k = stored_k if a_bytes else stored_k * 4 // 3
    types = {"e2m3": cutlass.Float6E2M3FN, "e3m2": cutlass.Float6E3M2FN,
             "e4m3": cutlass.Float8E4M3FN}
    with torch.cuda.device(a.device):
        dtypes = (cutlass.Uint8 if a_bytes else types[a_fmt],
                  cutlass.Uint8 if b_bytes else types[b_fmt], cutlass.Float8E8M0FNU,
                  cutlass.Float8E8M0FNU, OUTPUT_TYPES[c_dtype][1], cutlass.Float32, cutlass.BFloat16)
        # E4M3 values retain their numeric type; pre-expanded FP6 uses byte codes.
        dtypes = (types[a_fmt] if a_fmt == "e4m3" else dtypes[0],
                  types[b_fmt] if b_fmt == "e4m3" else dtypes[1], *dtypes[2:])
        a_stride = m * k if groups == 1 else a.stride(2) * (1 if a_bytes else 4) // (1 if a_bytes else 3)
        run_compiled(fn, (*(pointer(dtype, tensor) for dtype, tensor in zip(dtypes, (a, b, sfa, sfb, out, alpha, row_scale), strict=True)),
           cutlass.Int32(m), cutlass.Int64(a_stride),
           cutlass.Int64(m * out.shape[1] if groups == 1 else out.stride(2)),
           cutlass.Int64(0 if alpha is None or alpha.numel() == 1 else 1),
           cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream if stream_int is None else stream_int),))


@_execute.register_fake
def _execute_fake(a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor, sfb: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor | None, row_scale: torch.Tensor | None, a_fmt: str, b_fmt: str,
                  a_bytes: bool, b_bytes: bool, c_dtype: str, stream_int: int | None):
    return None
