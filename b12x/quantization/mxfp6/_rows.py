"""CuTe FP6/FP8 activation quantization with runtime row counts."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.fp6 import (
    mx_gs_numerator, quantize_block_fp6_e2m3_bytes, quantize_block_fp6_e3m2_bytes,
    quantize_block_fp6_e2m3_fast, quantize_block_fp6_e3m2_fast,
    quantize_block_fp8_e4m3_bytes,
)
from b12x._lib.intrinsics import (
    block_reduce, div_rn_f32, fabs_f32, fmax_f32, warp_reduce,
    get_ptr_as_int64, st_global_u64,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x.gemm.blockscaled._sm103 import pointer


class ScaleRows:
    def __init__(self, k, fmt, per_row):
        self.k, self.per_row = k, per_row
        self.numerator = mx_gs_numerator(fmt)

    @cute.jit
    def __call__(self, source: cute.Pointer, weight_scale: cute.Pointer,
                 scales: cute.Pointer, inverse: cute.Pointer, alpha: cute.Pointer,
                 m: cutlass.Int32, stream: cuda.CUstream):
        x = cute.make_tensor(source, cute.make_layout(cutlass.Int64(m) * self.k))
        wgs = cute.make_tensor(weight_scale, cute.make_layout(1))
        gs = cute.make_tensor(scales, cute.make_layout(m if cutlass.const_expr(self.per_row) else 1))
        inv = cute.make_tensor(inverse, cute.make_layout(m))
        a = cute.make_tensor(alpha, cute.make_layout(1))
        self.kernel(x, wgs, gs, inv, a, m).launch(
            grid=(m if cutlass.const_expr(self.per_row) else 1, 1, 1),
            block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, wgs, gs, inverse, alpha, m):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = cutlass.Int64(bid)
        start = row * self.k if cutlass.const_expr(self.per_row) else cutlass.Int64(0)
        count = self.k if cutlass.const_expr(self.per_row) else cutlass.Int64(m) * self.k
        value = cutlass.Float32(0)
        for col in cutlass.range(cutlass.Int64(tid), count, 128):
            value = fmax_f32(value, fabs_f32(cutlass.Float32(x[start + col])))
        shared = utils.SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout((1, 4)), byte_alignment=16)
        value = warp_reduce(value, fmax_f32)
        maximum = block_reduce(value, fmax_f32, shared, cutlass.Float32(0))
        if tid == 0:
            scale = div_rn_f32(cutlass.Float32(self.numerator), fmax_f32(maximum, cutlass.Float32(1e-6)))
            gs[row] = scale
            if cutlass.const_expr(self.per_row):
                inverse[row] = div_rn_f32(cutlass.Float32(1), scale).to(cutlass.BFloat16)
            if bid == 0:
                activation_scale = cutlass.Float32(1) if cutlass.const_expr(self.per_row) else scale
                alpha[0] = div_rn_f32(cutlass.Float32(1), activation_scale * wgs[0])


class QuantizeRows:
    def __init__(self, k, fmt, per_row, packed):
        self.k, self.fmt, self.per_row, self.packed = k, fmt, per_row, packed
        self.storage_k = k * 3 // 4 if packed else k

    @cute.jit
    def __call__(self, source: cute.Pointer, scales: cute.Pointer,
                 values: cute.Pointer, sf: cute.Pointer, m: cutlass.Int32,
                 grid: cutlass.Int32, stream: cuda.CUstream):
        x = cute.make_tensor(source, cute.make_layout(cutlass.Int64(m) * self.k))
        gs = cute.make_tensor(scales, cute.make_layout(m if cutlass.const_expr(self.per_row) else 1))
        q = cute.make_tensor(values, cute.make_layout(cutlass.Int64(m) * self.storage_k))
        s = cute.make_tensor(sf, cute.make_layout(cutlass.Int64(cute.ceil_div(m, 128)) * (self.k // 128) * 512))
        self.kernel(x, gs, q, s, m).launch(grid=(grid, 1, 1), block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, gs, q, sf, m):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        index = cutlass.Int64(bid) * 128 + tid
        total = cutlass.Int64(m) * (self.k // 32)
        while index < total:
            row = index // (self.k // 32)
            group = index % (self.k // 32)
            scale = gs[row] if cutlass.const_expr(self.per_row) else gs[0]
            values = cute.make_rmem_tensor(32, cutlass.Float32)
            maximum = cutlass.Float32(0)
            for i in cutlass.range_constexpr(32):
                value = cutlass.Float32(x[row * self.k + group * 32 + i])
                if cutlass.const_expr(self.per_row):
                    value = (value * scale).to(cutlass.BFloat16).to(cutlass.Float32)
                values[i] = value
                maximum = fmax_f32(maximum, fabs_f32(value))
            global_scale = cutlass.Float32(1) if cutlass.const_expr(self.per_row) else scale
            if cutlass.const_expr(self.packed):
                if cutlass.const_expr(self.fmt == "e2m3"):
                    q0, q1, q2, byte = quantize_block_fp6_e2m3_fast(values, maximum, global_scale)
                else:
                    q0, q1, q2, byte = quantize_block_fp6_e3m2_fast(values, maximum, global_scale)
                offset = row * self.storage_k + group * 24
                words = (q0, q1, q2)
            else:
                if cutlass.const_expr(self.fmt == "e2m3"):
                    q0, q1, q2, q3, byte = quantize_block_fp6_e2m3_bytes(values, maximum, global_scale)
                elif cutlass.const_expr(self.fmt == "e3m2"):
                    q0, q1, q2, q3, byte = quantize_block_fp6_e3m2_bytes(values, maximum, global_scale)
                else:
                    q0, q1, q2, q3, byte = quantize_block_fp8_e4m3_bytes(values, maximum, global_scale)
                offset = row * self.storage_k + group * 32
                words = (q0, q1, q2, q3)
            for i in cutlass.range_constexpr(len(words)):
                st_global_u64(get_ptr_as_int64(q, offset + 8*i), words[i])
            sf_offset = (row // 128) * (self.k // 128) * 512 + (group // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + group % 4
            sf[sf_offset] = byte
            index += cutlass.Int64(grid) * 128
        # TMA reads complete scale tiles, including rows beyond the live input.
        padding = cutlass.Int64(m) * (self.k // 32) + cutlass.Int64(bid) * 128 + tid
        padded_total = cutlass.Int64(cute.ceil_div(m, 128)) * 128 * (self.k // 32)
        while padding < padded_total:
            row, group = padding // (self.k // 32), padding % (self.k // 32)
            sf_offset = (row // 128) * (self.k // 128) * 512 + (group // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + group % 4
            sf[sf_offset] = cutlass.Uint8(0)
            padding += cutlass.Int64(grid) * 128


@lru_cache(maxsize=1024)
def compile_scales(k, fmt, per_row, device_ordinal, architecture):
    if fmt not in ("e2m3", "e3m2", "e4m3") or k <= 0 or k % 128 or k >= 2**31:
        raise ValueError("FP6 activation scales require a valid format and positive K divisible by 128")
    raise_if_kernel_resolution_frozen("FP6 activation scales")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP6 activation scales must be prewarmed before graph capture")
    return b12x_compile(
        ScaleRows(k, fmt, per_row), *(pointer(t) for t in (cutlass.BFloat16,
            cutlass.Float32, cutlass.Float32, cutlass.BFloat16, cutlass.Float32)),
        cutlass.Int32(1), cuda.CUstream(0),
        compile_spec=KernelCompileSpec.from_key("quantization.fp6.row_scales", 1,
            (k, fmt, per_row, device_ordinal, architecture)), options=f"--gpu-arch={architecture}")


@lru_cache(maxsize=1024)
def compile_quantizer(k, fmt, per_row, packed, device_ordinal, architecture):
    if fmt not in ("e2m3", "e3m2", "e4m3") or packed and fmt == "e4m3":
        raise ValueError("FP6 row quantization requires a valid format and storage recipe")
    if k <= 0 or k % 128 or k >= 2**31:
        raise ValueError("FP6 row quantization requires positive K divisible by 128")
    raise_if_kernel_resolution_frozen("FP6 row quantization")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP6 row quantization must be prewarmed before graph capture")
    return b12x_compile(
        QuantizeRows(k, fmt, per_row, packed), *(pointer(t) for t in (
            cutlass.BFloat16, cutlass.Float32, cutlass.Uint8, cutlass.Uint8)),
        cutlass.Int32(1), cutlass.Int32(1), cuda.CUstream(0),
        compile_spec=KernelCompileSpec.from_key("quantization.fp6.rows", 1,
            (k, fmt, per_row, packed, device_ordinal, architecture)), options=f"--gpu-arch={architecture}")
