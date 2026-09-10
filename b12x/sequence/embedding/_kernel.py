"""CuTe row copy; model width/types are static, all row quantities are runtime."""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from ..._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from ..._lib.runtime_control import raise_if_kernel_resolution_frozen
from ..._lib.utils import current_cuda_stream, make_ptr


@dsl_user_op
def _invalid_index(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


class _Embedding:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, weight: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
                 count: cute.Pointer, capacity: Int32, table_rows: Int64,
                 row_stride: Int64, use_count: Int32, stream: cuda.CUstream):
        self.kernel(weight, ids, out, count, capacity, table_rows,
                    row_stride, use_count).launch(
            grid=(capacity, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weight: cute.Pointer, ids: cute.Pointer, out: cute.Pointer,
               count: cute.Pointer, capacity: Int32, table_rows: Int64,
               row_stride: Int64, use_count: Int32):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        live = capacity
        if use_count != 0:
            live = count[0]
        if live < 0 or live > capacity:
            _invalid_index()
        else:
            if row < live:
                index = Int64(ids[row])
                if index < 0 or index >= table_rows:
                    _invalid_index()
                else:
                    src = index * row_stride
                    dst = Int64(row) * Int64(self.width)
                    for item in cutlass.range_constexpr((self.width + 127) // 128):
                        col = tid + item * 128
                        if col < self.width:
                            out[dst + Int64(col)] = weight[src + Int64(col)]


@cache
def _compile(width, weight_dtype, id_dtype, device):
    key = (width, str(weight_dtype), str(id_dtype), device)
    entry = _Embedding(width)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    value_type = BFloat16 if weight_dtype == torch.bfloat16 else Float32
    index_type = Int32 if id_dtype == torch.int32 else Int64
    types = (value_type, index_type, value_type, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem,
                              assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int64(1),
                                Int64(width), Int32(0), current_cuda_stream(),
                                compile_spec=KernelCompileSpec.from_key(
                                    "sequence.embedding", 1, key))
    return compiled, types


def launch(weight, ids, out, num_rows):
    if ids.numel() == 0:
        return
    compiled, types = _compile(weight.shape[1], weight.dtype, ids.dtype,
                               weight.device.index)
    tensors = (weight, ids, out, ids if num_rows is None else num_rows)
    pointers = tuple(make_ptr(t, tensor.data_ptr(), cute.AddressSpace.gmem,
                              assumed_align=t.width // 8)
                     for t, tensor in zip(types, tensors, strict=True))
    with torch.cuda.device(weight.device):
        run_compiled(compiled, (*pointers, Int32(ids.numel()),
                                Int64(weight.shape[0]), Int64(weight.stride(0)),
                                Int32(num_rows is not None), current_cuda_stream()))
