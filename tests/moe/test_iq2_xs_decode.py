"""All IQ2_XS descriptors and subscales through the production CuTe intrinsic."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    packed_decode_iq2_xs_to_bfloat2x4,
    shared_ptr_to_u32,
)
from b12x._lib.quant.iq2_xs import IQ2_XS_MAGNITUDE_LUT_BYTES, iq2_xs_execution_lut
from b12x._lib.utils import current_cuda_stream
from b12x.testing.iq2_xs_reference import descriptor_vectors
from tests._reference.helpers import require_b12x


class _DecodeProbe:
    def __init__(self, shared_lut=False):
        self.shared_lut = shared_lut

    @property
    def __cache_key__(self):
        return (self.shared_lut,)

    @cute.jit
    def __call__(
        self,
        base: cutlass.Int32,
        lut: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(base, lut, out).launch(
            grid=(16384, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, base: cutlass.Int32, lut: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        table_addr = lut.iterator.toint()
        if cutlass.const_expr(self.shared_lut):
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate_array(
                cutlass.Uint32, IQ2_XS_MAGNITUDE_LUT_BYTES // 4, byte_alignment=16
            )
            table_shared = shared_ptr_to_u32(storage)
            for i in cutlass.range_constexpr(IQ2_XS_MAGNITUDE_LUT_BYTES // (256 * 16)):
                cp_async4_shared_global(
                    table_shared + (i * 256 + tid) * 16,
                    table_addr + cutlass.Int64(i * 256 + tid) * 16,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            table_addr = cutlass.Int64(table_shared)
        row = block * 256 + tid
        descriptor = cutlass.Uint32(row // 64)
        pair = (row % 4) * 2
        nibble = cutlass.Uint32((row // 4) % 16)
        other = descriptor ^ cutlass.Uint32(0xFFFF)
        q0 = descriptor | (other << 16)
        q1 = other | (descriptor << 16)
        base_pair = cutlass.Uint32(base) | (
            (cutlass.Uint32(base) ^ cutlass.Uint32(0x8000)) << 16
        )
        subscale_pair = nibble | ((15 - nibble) << 8)
        a, b, c, d = packed_decode_iq2_xs_to_bfloat2x4(
            q0, q1, base_pair, subscale_pair, table_addr, pair, shared_lut=self.shared_lut
        )
        out[row, 0], out[row, 1] = cutlass.Int32(a), cutlass.Int32(b)
        out[row, 2], out[row, 3] = cutlass.Int32(c), cutlass.Int32(d)


@pytest.mark.parametrize(
    "base_bits", [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C01, 0x7BFF, 0xFBFF]
)
@pytest.mark.parametrize("shared_lut", [False, True])
def test_all_descriptors_subscales_and_rounding(base_bits, shared_lut):
    device = require_b12x()
    table = iq2_xs_execution_lut(device, prepare=True)
    output = torch.empty((65536 * 64, 4), dtype=torch.int32, device=device)
    lut_arg, out_arg = (
        from_dlpack(table, assumed_align=16),
        from_dlpack(output, assumed_align=16),
    )
    compiled = b12x_compile(
        _DecodeProbe(shared_lut),
        cutlass.Int32(base_bits),
        lut_arg,
        out_arg,
        current_cuda_stream(),
    )
    compiled(cutlass.Int32(base_bits), lut_arg, out_arg, current_cuda_stream())
    actual = output.cpu().view(torch.bfloat16).reshape(65536, 16, 4, 8)
    vectors = descriptor_vectors().float()
    base = torch.tensor([base_bits], dtype=torch.uint16).view(torch.float16).float()[0]
    for nibble in range(16):
        s0 = (base * (nibble + 0.5)) * 0.25
        s1 = (-base * (15 - nibble + 0.5)) * 0.25
        expected = torch.stack(
            (vectors * s0, vectors.flip(0) * s0, vectors.flip(0) * s1, vectors * s1), 1
        )
        expected = (
            expected.reshape(65536, 4, 4, 2)
            .permute(0, 2, 1, 3)
            .reshape(65536, 4, 8)
            .bfloat16()
        )
        assert torch.equal(
            actual[:, nibble].view(torch.int16), expected.view(torch.int16)
        )
