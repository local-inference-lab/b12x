"""Portable execution of the exact FP6 byte-container SMEM packing helper."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm
from b12x.gemm.blockscaled._sm103 import pointer
from tests.quantization.test_fp6_workspace import pack, require_gpu


class PackingProbe(BlockscaledGemm):
    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, stream: cuda.CUstream):
        src = cute.make_tensor(source, cute.make_layout(16384))
        dst = cute.make_tensor(output, cute.make_layout(16384))
        self.probe(src, dst).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def probe(self, source, output):
        tid, _, _ = cute.arch.thread_idx()
        shared = utils.SmemAllocator().allocate_tensor(cutlass.Uint8, cute.make_layout(16384), byte_alignment=128)
        for offset in cutlass.range(tid, 16384, 128):
            shared[offset] = source[offset]
        cute.arch.barrier()
        if tid < 32:
            self._pack_fp6_smem(shared, 16384)
            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        for offset in cutlass.range(tid, 16384, 128):
            output[offset] = shared[offset]


def test_native_smem_packer_preserves_all_codes_and_padding():
    require_gpu()
    source = torch.randint(0, 256, (1024, 16), device="cuda", dtype=torch.uint8)
    output = torch.empty_like(source)
    probe = PackingProbe(128, 128, 1, recipe="mxfp6", c_dtype=cutlass.BFloat16,
                         a_fmt="e3m2", b_fmt="e2m3")
    major, minor = torch.cuda.get_device_capability()
    fn = cute.compile(probe, pointer(cutlass.Uint8), pointer(cutlass.Uint8), cuda.CUstream(0),
                      options=f"--gpu-arch=sm_{major}{minor}a")
    fn(pointer(cutlass.Uint8, source), pointer(cutlass.Uint8, output), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.testing.assert_close(output[:, :12], pack(source & 63), atol=0, rtol=0)
    torch.testing.assert_close(output[:, 12:], source[:, 12:], atol=0, rtol=0)
