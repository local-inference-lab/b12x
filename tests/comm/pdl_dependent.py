"""Dependent kernel for programmatic-dependent-launch tests.

``compile_wait_then_copy()`` returns ``copy(src, dst, use_pdl)``: a kernel
that executes ``griddepcontrol.wait`` and then copies ``src`` to ``dst`` as
16-byte packs. Launched with ``use_pdl`` behind a kernel that triggers
``griddepcontrol.launch_dependents`` early, it may start before that kernel
finishes; the wait must still make it observe the complete output.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 256


class WaitThenCopyKernel:
    @cute.jit
    def __call__(
        self,
        src_ptr: cute.Pointer,
        dst_ptr: cute.Pointer,
        packs: Int32,
        use_pdl: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = (packs + Int32(_THREADS - 1)) // Int32(_THREADS)
        self.kernel(src_ptr, dst_ptr, packs).launch(
            grid=(grid_x, 1, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
            use_pdl=use_pdl,
        )

    @cute.kernel
    def kernel(self, src_ptr: cute.Pointer, dst_ptr: cute.Pointer, packs: Int32):
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = Int32(bidx) * Int32(_THREADS) + Int32(tidx)
        if i < packs:
            src = Int64(src_ptr.toint()) + Int64(i) * Int64(16)
            dst = Int64(dst_ptr.toint()) + Int64(i) * Int64(16)
            v0, v1, v2, v3 = ld_global_v4_u32(src)
            st_global_v4_u32(dst, v0, v1, v2, v3)


def compile_wait_then_copy():
    raw = b12x_compile(
        WaitThenCopyKernel(),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        current_cuda_stream(),
    )

    def copy(src: torch.Tensor, dst: torch.Tensor, use_pdl: bool) -> None:
        nbytes = src.numel() * src.element_size()
        assert nbytes % 16 == 0 and dst.numel() * dst.element_size() == nbytes
        assert src.is_contiguous() and dst.is_contiguous()
        raw(
            make_ptr(
                cutlass.Uint32, src.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32, dst.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            nbytes // 16,
            1 if use_pdl else 0,
            current_cuda_stream(),
        )

    return copy
