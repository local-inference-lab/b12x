"""V4.1 BF16 rotary preparation, without the V4 per-query-head RMSNorm."""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _Rotate:
    def __init__(self, heads, dim, rope_dim, inverse, ratio):
        self.heads, self.dim, self.rope_dim = heads, dim, rope_dim
        self.inverse, self.ratio = inverse, ratio

    @cute.jit
    def __call__(self, x: cute.Pointer, pos: cute.Pointer, cs: cute.Pointer,
                 out: cute.Pointer, rows: Int32, sx: Int64, sh: Int64,
                 sc: Int64, stream: cuda.CUstream):
        self.kernel(x, pos, cs, out, rows, sx, sh, sc).launch(
            grid=(rows, self.heads, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, pos: cute.Pointer, cs: cute.Pointer,
               out: cute.Pointer, rows: Int32, sx: Int64, sh: Int64, sc: Int64):
        row, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        position = Int64(pos[row])
        if cutlass.const_expr(self.ratio > 1):
            position = (position // Int64(self.ratio)) * Int64(self.ratio)
        for item in cutlass.range_constexpr((self.dim + 127) // 128):
            col = tid + item * 128
            if col < self.dim:
                offset = Int64(row) * sx + Int64(head) * sh + Int64(col)
                value = Float32(0.0)
                if position >= 0:
                    value = Float32(x[offset])
                    if col >= self.dim - self.rope_dim:
                        local = col - (self.dim - self.rope_dim)
                        partner = Float32(x[offset + Int64(1 - 2 * (local % 2))])
                        cosine = Float32(cs[position * sc + Int64(local // 2)])
                        sine = Float32(cs[position * sc + Int64(self.rope_dim // 2 + local // 2)])
                        sign = Float32(-1.0)
                        if local % 2 == 1:
                            sign = Float32(1.0)
                        if cutlass.const_expr(self.inverse):
                            sign = -sign
                        value = value * cosine + sign * partner * sine
                out[(Int64(row) * Int64(self.heads) + Int64(head)) * Int64(self.dim) + Int64(col)] = BFloat16(value)


@cache
def _compile(heads, dim, rope_dim, inverse, ratio, cs_dtype, device):
    key = (heads, dim, rope_dim, inverse, ratio, cs_dtype, device)
    entry = _Rotate(heads, dim, rope_dim, inverse, ratio)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = (BFloat16, Int64, Float32 if cs_dtype == torch.float32 else BFloat16, BFloat16)
    fake = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        raw = compile_cute(entry, *fake, Int32(1), Int64(1), Int64(1), Int64(1),
                           current_cuda_stream(), compile_spec=KernelCompileSpec.from_key(
                               "attention.compressed_sparse_mla.rotate", 1, key))
    return raw, types


def rotate(x, positions, cos_sin_cache, *, out, rope_dim=64, inverse=False, ratio=1):
    """Rotate adjacent pairs in the last rope_dim columns into caller storage.

    ``ratio`` maps a compressed emission to its group's FIRST absolute position.
    No normalization, quantization, query scaling, or padded-head synthesis occurs.
    """
    if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("rotary input/output must be BF16")
    if x.ndim not in (2, 3) or out.shape != x.shape or not out.is_contiguous():
        raise ValueError("rotary expects [T,D] or [T,H,D] and contiguous matching output")
    if positions.dtype != torch.int64 or positions.shape != (x.shape[0],):
        raise ValueError("rotary positions must be int64[T]")
    if cos_sin_cache.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("rotary table must be BF16 or FP32")
    if any(t.device != x.device for t in (positions, cos_sin_cache, out)):
        raise ValueError("rotary tensors must share a device")
    if x.data_ptr() == out.data_ptr():
        raise ValueError("rotary output must not alias its input")
    if x.shape[0] == 0:
        return out
    heads = x.shape[1] if x.ndim == 3 else 1
    raw, types = _compile(heads, x.shape[-1], rope_dim, inverse, ratio,
                          cos_sin_cache.dtype, x.device.index)
    args = tuple(make_ptr(t, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8)
                 for t, v in zip(types, (x, positions, cos_sin_cache, out), strict=True))
    with torch.cuda.device(x.device):
        run_compiled(raw, (*args, Int32(x.shape[0]), Int64(x.stride(0)),
                           Int64(x.stride(1) if x.ndim == 3 else 0),
                           Int64(cos_sin_cache.stride(0)), current_cuda_stream()))
    return out


class _Cast:
    def __init__(self, fp32):
        self.fp32 = fp32

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, size: Int64,
                 stream: cuda.CUstream):
        self.kernel(x, out, size).launch(grid=((size + Int64(255)) // Int64(256), 1, 1),
                                         block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, size: Int64):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        i = Int64(block) * Int64(256) + Int64(thread)
        if i < size:
            if cutlass.const_expr(self.fp32):
                out[i] = Float32(x[i])
            else:
                out[i] = BFloat16(x[i])


@cache
def _compile_cast(input_dtype, output_dtype, device):
    key = (input_dtype, output_dtype, device)
    entry = _Cast(output_dtype == torch.float32)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = tuple(BFloat16 if t == torch.bfloat16 else Float32
                  for t in (input_dtype, output_dtype))
    fake = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8)
                 for t in types)
    with torch.cuda.device(device):
        raw = compile_cute(entry, *fake, Int64(1), current_cuda_stream(),
                           compile_spec=KernelCompileSpec.from_key(
                               "attention.compressed_sparse_mla.cast", 1, key))
    return raw, types


def cast(x, *, out):
    """Cast contiguous BF16/FP32 storage into a caller-owned fixed buffer."""
    if x.dtype not in (torch.bfloat16, torch.float32) or out.dtype not in (
            torch.bfloat16, torch.float32):
        raise TypeError("native attention cast requires BF16/FP32 tensors")
    if (x.shape != out.shape or not x.is_contiguous() or not out.is_contiguous()
            or x.device != out.device or x.device.type != "cuda"):
        raise ValueError("native attention cast requires matching contiguous CUDA tensors")
    if x.numel() == 0:
        return out
    raw, types = _compile_cast(x.dtype, out.dtype, x.device.index)
    ptrs = tuple(make_ptr(t, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8)
                 for t, v in zip(types, (x, out), strict=True))
    with torch.cuda.device(x.device):
        run_compiled(raw, (*ptrs, Int64(x.numel()), current_cuda_stream()))
    return out
