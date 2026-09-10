"""BF16 reference rounding boundary for V4.1 index-head weights."""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _Scale:
    @cute.jit
    def __call__(self, x: cute.Pointer, y: cute.Pointer, n: Int32,
                 stream: cuda.CUstream):
        self.kernel(x, y, n).launch(grid=((n + 255) // 256, 1, 1),
                                     block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, y: cute.Pointer, n: Int32):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        i = block * 256 + thread
        if i < n:
            y[Int64(i)] = BFloat16(Float32(x[Int64(i)]) * Float32(1.0 / 64.0))


@cache
def _compile(device):
    entry = _Scale()
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=(device,))
    p = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2)
    with torch.cuda.device(device):
        return compile_cute(entry, p, p, Int32(1), current_cuda_stream(),
                            compile_spec=KernelCompileSpec.from_key(
                                "attention.compressed_sparse_mla.index_weights", 1, (device,)))


def scale_index_weights(x, *, out):
    if x.dtype != torch.bfloat16 or out.dtype != x.dtype or out.shape != x.shape:
        raise ValueError("index weights require matching BF16 tensors")
    if not x.is_contiguous() or not out.is_contiguous() or out.device != x.device:
        raise ValueError("index weights must be contiguous on one device")
    if x.numel():
        ptrs = tuple(make_ptr(BFloat16, t.data_ptr(), cute.AddressSpace.gmem,
                              assumed_align=2) for t in (x, out))
        with torch.cuda.device(x.device):
            run_compiled(_compile(x.device.index), (*ptrs, Int32(x.numel()), current_cuda_stream()))
    return out
