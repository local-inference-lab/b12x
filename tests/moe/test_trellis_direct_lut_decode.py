"""Bit-equality of the direct state-table decode against the T12 decode.

The rate-indexed direct table precomposes the frozen XOR-Cheb rank map with
the modal T12 staircase (``direct[state] = t12[rank(state) >> 4]``), so the
two decode intrinsics must produce identical bytes for every window. The
probe drives both intrinsics on the same random ring windows and compares
the packed results.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    fp8x2_e4m3_pair_to_half2,
    fp8x2_e4m3_pair_to_bfloat2_native_sm120,
    fp8x4_e4m3_to_half2x2,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
    packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x8,
    packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x2x4,
    packed_decode_trellis_sqg_direct_lut_to_e4m3x8,
    shared_ptr_to_u32,
    st_shared_u32,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
    sqg_xor_cheb_t12_lut_cpu,
)
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x

class _DecodeProbe:
    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, t12, direct, out).launch(
            grid=(1, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx)
        t12_addr = t12.iterator.toint()
        dir_addr = direct.iterator.toint()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            wa, wb, t12_addr, self.bits, t12_in_shared=False
        )
        lo_d, hi_d = packed_decode_trellis_sqg_direct_lut_to_e4m3x8(
            wa, wb, dir_addr, self.bits, rate_indexed=True
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_direct_lut_decode_bit_equals_t12(bits: int) -> None:
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(20260812 + bits)
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    direct = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    out = torch.zeros(128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(t12, assumed_align=16),
            from_dlpack(direct, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_DecodeProbe(bits), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched


class _DirectPairsProbe:
    def __init__(self, bits, fp16):
        self.bits = bits
        self.fp16 = fp16

    @cute.jit
    def __call__(self, wins: cute.Tensor, table: cute.Tensor, out: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(wins, table, out).launch(
            grid=(32, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, wins: cute.Tensor, table: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[Uint32, 16384], 16]

        storage = allocator.allocate(Storage)
        addr = shared_ptr_to_u32(storage.words.data_ptr())
        for word in cutlass.range(tid, 16384, 256):
            st_shared_u32(addr + Int32(word) * Int32(4), Uint32(table[word]))
        cute.arch.sync_threads()
        for index in cutlass.range(cta * 256 + tid, wins.shape[0], 32 * 256):
            wa, wb = Uint32(wins[index, 0]), Uint32(wins[index, 1])
            lo, hi = packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x8(
                wa, wb, addr, self.bits
            )
            p0, p1, p2, p3 = packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x2x4(
                wa, wb, addr, self.bits
            )
            if cutlass.const_expr(self.fp16):
                h0, h1 = fp8x4_e4m3_to_half2x2(lo)
                h2, h3 = fp8x4_e4m3_to_half2x2(hi)
                f0 = fp8x2_e4m3_pair_to_half2(p0)
                f1 = fp8x2_e4m3_pair_to_half2(p1)
                f2 = fp8x2_e4m3_pair_to_half2(p2)
                f3 = fp8x2_e4m3_pair_to_half2(p3)
            else:
                h0, h1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(lo)
                h2, h3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(hi)
                f0 = fp8x2_e4m3_pair_to_bfloat2_native_sm120(p0)
                f1 = fp8x2_e4m3_pair_to_bfloat2_native_sm120(p1)
                f2 = fp8x2_e4m3_pair_to_bfloat2_native_sm120(p2)
                f3 = fp8x2_e4m3_pair_to_bfloat2_native_sm120(p3)
            out[index, 0] = Int32(lo)
            out[index, 1] = Int32(hi)
            out[index, 2] = Int32(p0)
            out[index, 3] = Int32(p1)
            out[index, 4] = Int32(p2)
            out[index, 5] = Int32(p3)
            out[index, 6] = Int32(h0)
            out[index, 7] = Int32(h1)
            out[index, 8] = Int32(h2)
            out[index, 9] = Int32(h3)
            out[index, 10] = Int32(f0)
            out[index, 11] = Int32(f1)
            out[index, 12] = Int32(f2)
            out[index, 13] = Int32(f3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("fp16", [True, False])
def test_shared_direct_pairs_preserve_every_state_and_fragment(bits, fp16):
    """Cover every 16-bit state in each window position and both fragment types."""
    require_b12x()
    states = torch.arange(1 << 16, dtype=torch.int64)
    words = torch.cat([states << (i * bits) for i in range(4)])
    wins_cpu = torch.stack((words, words.roll(123)), dim=1).to(torch.int32)
    table_cpu = sqg_xor_cheb_t12_direct_lut_cpu()[(bits - 2) << 16:(bits - 1) << 16]
    wins = wins_cpu.cuda()
    table = table_cpu.cuda().view(torch.int32)
    out = torch.full((len(words), 14), -1, dtype=torch.int32, device="cuda")
    args = (from_dlpack(wins, assumed_align=16), from_dlpack(table, assumed_align=16),
            from_dlpack(out, assumed_align=16), current_cuda_stream())
    compiled = b12x_compile(_DirectPairsProbe(bits, fp16), *args)
    compiled(*args)
    result = out.cpu().to(torch.int64) & 0xFFFFFFFF
    windows = wins_cpu.to(torch.int64) & 0xFFFFFFFF
    expected_bytes = torch.stack([
        table_cpu[((windows[:, 1 if i < 4 else 0] >> ((3 - (i & 3)) * bits))
                   & 0xFFFF)].to(torch.int64)
        for i in range(8)
    ], dim=1)
    expected_pairs = expected_bytes[:, ::2] | (expected_bytes[:, 1::2] << 8)
    torch.testing.assert_close(result[:, 2:6], expected_pairs, rtol=0, atol=0)
    torch.testing.assert_close(result[:, 0], expected_pairs[:, 0] | (expected_pairs[:, 1] << 16))
    torch.testing.assert_close(result[:, 1], expected_pairs[:, 2] | (expected_pairs[:, 3] << 16))
    torch.testing.assert_close(result[:, 6:10], result[:, 10:14], rtol=0, atol=0)
