"""Native unquantized projections with runtime row counts.

The original small-N GEMV's vectorized BF16 dot product is shared by every
row tile. FP32 operands use the same strided reduction without quantization.
Geometry, operand types, and bias presence specialize the kernel; live rows
and row strides do not. Both functional and caller-owned outputs use CuTe.
"""
from __future__ import annotations

from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import (
    block_reduce, get_ptr_as_int64, ld_global_v4_u32, u32_as_f32, warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 128
SMALL_M_MAX = 8  # Rows sharing a weight load, not a live-row support limit.
_DTYPES = {torch.bfloat16: BFloat16, torch.float32: Float32}
_NAMES = {torch.bfloat16: "bf16", torch.float32: "fp32"}
_KERNEL_CACHE: dict[tuple, object] = {}
_WARMED: set[tuple] = set()
_LOCK = RLock()


def _fadd(a, b):
    return a + b


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


@cute.jit
def _dot_bf16x8(source: cute.Tensor, offset: Int64, w0: Uint32,
                w1: Uint32, w2: Uint32, w3: Uint32, accumulator: Float32):
    x0, x1, x2, x3 = ld_global_v4_u32(get_ptr_as_int64(source, offset))
    result = accumulator
    for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
        result += u32_as_f32(wv << Uint32(16)) * u32_as_f32(xv << Uint32(16))
        result += u32_as_f32(wv & Uint32(0xFFFF0000)) * u32_as_f32(xv & Uint32(0xFFFF0000))
    return result


@cute.jit
def _reduce_store(value: Float32, reduction: cute.Tensor, output: cute.Tensor,
                  offset: Int64, bias: cute.Tensor, column: Int32,
                  has_bias: cutlass.Constexpr):
    total = block_reduce(warp_reduce(value, _fadd), _fadd, reduction, Float32(0.0))
    thread, _, _ = cute.arch.thread_idx()
    if Int32(thread) == Int32(0):
        if cutlass.const_expr(has_bias):
            total += Float32(bias[column])
        output[offset] = total.to(output.element_type)
    cute.arch.barrier()


class SmallNGemvKernel:
    """One CTA per output column and fixed tile of up to eight live rows."""

    def __init__(self, n: int, k: int, bf16_operands: bool, has_bias: bool):
        self.n, self.k = int(n), int(k)
        self.bf16_operands = bool(bf16_operands)
        self.has_bias = bool(has_bias)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer,
                 output: cute.Pointer, rows: Int32, x_stride: Int64,
                 weight_stride: Int64, output_stride: Int64,
                 x_column_stride: Int64, weight_column_stride: Int64,
                 vector_loads: Int32, stream: cuda.CUstream):
        self.kernel(_flat(x), _flat(weight), _flat(bias), _flat(output), rows,
                    x_stride, weight_stride, output_stride, x_column_stride,
                    weight_column_stride, vector_loads).launch(
            grid=(self.n, (rows + Int32(SMALL_M_MAX - 1)) // Int32(SMALL_M_MAX), 1),
            block=(_THREADS, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor,
               output: cute.Tensor, rows: Int32, x_stride: Int64,
               weight_stride: Int64, output_stride: Int64,
               x_column_stride: Int64, weight_column_stride: Int64,
               vector_loads: Int32):
        thread, _, _ = cute.arch.thread_idx()
        column, row_tile, _ = cute.arch.block_idx()
        tid = Int32(thread)
        first_row = Int64(row_tile) * Int64(SMALL_M_MAX)
        w_base = Int64(column) * weight_stride
        acc = cute.make_rmem_tensor((SMALL_M_MAX,), Float32)
        for r in cutlass.range_constexpr(SMALL_M_MAX):
            acc[r] = Float32(0.0)
        if cutlass.const_expr(self.bf16_operands and self.k % 8 == 0):
            if vector_loads != Int32(0):
                index = tid
                while index < Int32(self.k // 8):
                    offset = Int64(index) * Int64(8)
                    w0, w1, w2, w3 = ld_global_v4_u32(get_ptr_as_int64(weight, w_base + offset))
                    for r in cutlass.range_constexpr(SMALL_M_MAX):
                        row = first_row + Int64(r)
                        if row < Int64(rows):
                            acc[r] = _dot_bf16x8(source, row * x_stride + offset,
                                                w0, w1, w2, w3, acc[r])
                    index += Int32(_THREADS)
            else:
                self.scalar_dot(source, weight, acc, first_row, w_base, rows,
                                x_stride, x_column_stride, weight_column_stride, tid)
        else:
            self.scalar_dot(source, weight, acc, first_row, w_base, rows,
                            x_stride, x_column_stride, weight_column_stride, tid)
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(Float32, cute.make_layout((1, _THREADS // 32)), byte_alignment=16)
        for r in cutlass.range_constexpr(SMALL_M_MAX):
            row = first_row + Int64(r)
            if row < Int64(rows):
                _reduce_store(acc[r], reduction, output,
                              row * output_stride + Int64(column), bias,
                              Int32(column), self.has_bias)

    @cute.jit
    def scalar_dot(self, source: cute.Tensor, weight: cute.Tensor, acc: cute.Tensor,
                   first_row: Int64, w_base: Int64, rows: Int32, x_stride: Int64,
                   x_column_stride: Int64, weight_column_stride: Int64, tid: Int32):
        index = tid
        while index < Int32(self.k):
            value = Float32(weight[w_base + Int64(index) * weight_column_stride])
            for r in cutlass.range_constexpr(SMALL_M_MAX):
                row = first_row + Int64(r)
                if row < Int64(rows):
                    acc[r] += Float32(
                        source[row * x_stride + Int64(index) * x_column_stride]
                    ) * value
            index += Int32(_THREADS)


def _pointer(tensor: torch.Tensor):
    return make_ptr(_DTYPES[tensor.dtype], tensor.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=tensor.element_size())


def _key(x, weight, out, bias):
    device = x.device.index
    if device is None:
        device = torch.cuda.current_device()
    return (int(device), int(weight.shape[0]), int(weight.shape[1]),
            _NAMES[x.dtype], _NAMES[weight.dtype], _NAMES[out.dtype],
            None if bias is None else _NAMES[bias.dtype])


def _compile(key, x_dtype, weight_dtype, out_dtype, bias_dtype, device):
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        kernel = SmallNGemvKernel(key[1], key[2], x_dtype == weight_dtype == torch.bfloat16,
                                 bias_dtype is not None)
        raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
        types = (x_dtype, weight_dtype, x_dtype if bias_dtype is None else bias_dtype, out_dtype)
        pointers = [make_ptr(_DTYPES[dtype], 16, cute.AddressSpace.gmem,
                             assumed_align=dtype.itemsize) for dtype in types]
        with torch.cuda.device(device):
            compiled = b12x_compile(
                kernel, *pointers, Int32(1), Int64(key[2]), Int64(key[2]),
                Int64(key[1]), Int64(1), Int64(1), Int32(0), current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key("gemm.bf16_gemv_small_n", 3, key),
            )
        _KERNEL_CACHE[key] = compiled
        return compiled


def _validate(x, weight, out, bias):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("projection requires x[M,K] and weight[N,K]")
    if not x.is_cuda or weight.device != x.device or out.device != x.device:
        raise ValueError("native unquantized projection requires one CUDA device")
    if x.dtype not in _DTYPES or weight.dtype not in _DTYPES or out.dtype not in _DTYPES:
        raise TypeError("native unquantized projection supports BF16 and FP32")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0 or x.shape[0] > 2**31 - 1:
        raise ValueError("projection geometry must be positive and live rows fit int32")
    if tuple(out.shape) != (x.shape[0], weight.shape[0]):
        raise ValueError("out must have shape [M,N]")
    if out.stride(1) != 1 or out.stride(0) < out.shape[1]:
        raise ValueError("out needs unit column stride and disjoint rows")
    if any(stride < 0 for tensor in (x, weight) for stride in tensor.stride()):
        raise ValueError("projection input strides must be nonnegative")
    if bias is not None and (
        bias.device != x.device or bias.dtype not in _DTYPES
        or tuple(bias.shape) != (weight.shape[0],) or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous BF16/FP32 [N] on the input device")
    if not out.numel():
        return
    output_start = out.data_ptr()
    output_end = output_start + ((out.shape[0] - 1) * out.stride(0) + out.shape[1]) * out.element_size()
    for tensor in (x, weight) if bias is None else (x, weight, bias):
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
        start, end = tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()
        if output_start < end and start < output_end:
            raise ValueError("projection output must not overlap its inputs")


def _launch(x, weight, out, bias=None):
    _validate(x, weight, out, bias)
    if x.shape[0] == 0:
        return
    key = _key(x, weight, out, bias)
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            compiled = _KERNEL_CACHE.get(key)
            warmed = key in _WARMED
        if capturing and not warmed:
            raise RuntimeError("native unquantized projection must be warm-run before CUDA graph capture")
        if compiled is None:
            compiled = _compile(key, x.dtype, weight.dtype, out.dtype,
                                None if bias is None else bias.dtype, x.device)
        vector_loads = int(
            x.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0
            and x.stride(0) % 8 == 0 and weight.stride(0) % 8 == 0
            and x.stride(1) == 1 and weight.stride(1) == 1
        )
        run_compiled(compiled, (
            _pointer(x), _pointer(weight), _pointer(x if bias is None else bias), _pointer(out),
            int(x.shape[0]), int(x.stride(0)), int(weight.stride(0)), int(out.stride(0)),
            int(x.stride(1)), int(weight.stride(1)),
            vector_loads, current_cuda_stream(),
        ))
        if not capturing:
            with _LOCK:
                _WARMED.add(key)


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(x: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      output_dtype: torch.dtype | None = None) -> torch.Tensor:
    dtype = x.dtype if output_dtype is None else output_dtype
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=dtype, device=x.device)
    _launch(x, weight, out, bias)
    return out


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(x, weight, bias=None, output_dtype=None):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=x.dtype if output_dtype is None else output_dtype)


@torch.library.custom_op("b12x::bf16_gemv_small_n_out", mutates_args=("out",))
def bf16_gemv_small_n_out(x: torch.Tensor, weight: torch.Tensor,
                          out: torch.Tensor, bias: torch.Tensor | None = None) -> None:
    _launch(x, weight, out, bias)


@bf16_gemv_small_n_out.register_fake
def _bf16_gemv_small_n_out_fake(x, weight, out, bias=None):
    return None


def precompile_bf16_gemv_small_n(weight: torch.Tensor, log=None, *,
                                input_dtype: torch.dtype = torch.bfloat16,
                                output_dtype: torch.dtype = torch.bfloat16,
                                bias: torch.Tensor | None = None) -> None:
    """Warm one static geometry/type specialization for every live row count."""
    if weight.ndim != 2 or not weight.is_cuda or weight.dtype not in _DTYPES:
        raise ValueError("precompile requires a CUDA BF16/FP32 weight matrix")
    if input_dtype not in _DTYPES or output_dtype not in _DTYPES:
        raise TypeError("projection input/output dtype must be BF16 or FP32")
    with torch.cuda.device(weight.device):
        x = torch.zeros((1, weight.shape[1]), dtype=input_dtype, device=weight.device)
        out = torch.empty((1, weight.shape[0]), dtype=output_dtype, device=weight.device)
        key = _key(x, weight, out, bias)
        with _LOCK:
            if key in _WARMED:
                return
        _launch(x, weight, out, bias)
        torch.cuda.current_stream(weight.device).synchronize()
    if log is not None:
        log.debug("native projection warm: N=%d K=%d %s/%s -> %s",
                  weight.shape[0], weight.shape[1], input_dtype, weight.dtype, output_dtype)
