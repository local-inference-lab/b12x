"""Native mixed-cache staging must preserve the FP4/FP8 dequantization contract."""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.utils import SmemAllocator
import pytest
import torch

from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention._shared.mla.decode_math import _nvfp4_pair_bfloat2
from tests._reference.helpers import require_b12x


class StagedMixedPair:
    """Exercise source tags and byte offsets through the production helper."""

    @cute.jit
    def __call__(self, q: cute.Pointer, sf: cute.Pointer,
                 tag: cute.Pointer, out: cute.Pointer, stream):
        self.kernel(q, sf, tag, out).launch(
            grid=(2048, 1, 1), block=(32, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, q: cute.Pointer, sf: cute.Pointer,
               tag: cute.Pointer, out: cute.Pointer):
        lane = Int32(cute.arch.thread_idx()[0])
        i = Int64(cute.arch.block_idx()[0]) * Int64(32) + lane
        staging = SmemAllocator().allocate_tensor(
            Uint32, cute.make_layout(32 * 136), 16,
        )
        staging[lane * 136] = q[i]
        staging[lane * 136 + 64] = sf[i]
        staging[lane * 136 + 128] = sf[i]
        staging[lane * 136 + 132] = tag[i]
        cute.arch.sync_threads()
        out[i] = _nvfp4_pair_bfloat2(
            shared_ptr_to_u32(staging.iterator), lane, Int32(0), Float32(17.0),
            kv_smem_stride=544,
        )


@pytest.mark.parametrize("source", ["indexed", "swa", "invalid"])
def test_staged_pairs_exhaustive_and_graph_mutation(source):
    """All scale bytes, including UE8M0 zero/subnormals/NaN, match FP64."""
    require_b12x()
    codes = torch.arange(256, device="cuda", dtype=torch.int64).repeat(256)
    scales = torch.arange(256, device="cuda", dtype=torch.int64).repeat_interleave(256)
    if source == "indexed":
        lut = torch.tensor(
            [0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda", dtype=torch.float64,
        )
        values = torch.stack((lut[codes & 15], lut[codes >> 4]), -1)
        factors = scales.to(torch.uint8).view(torch.float8_e4m3fn).double()
        packed = codes
        source_tag = 0
    else:
        values = torch.stack((codes, 255 - codes), -1).to(torch.uint8)
        values = values.view(torch.float8_e4m3fn).double()
        factors = torch.exp2(scales.double() - 127)
        factors[scales == 255] = float("nan")
        packed = codes | ((255 - codes) << 8)
        source_tag = 1 if source == "swa" else 2
    expected = (values * factors[:, None]).to(torch.bfloat16)
    if source == "invalid":
        expected.zero_()
    packed = packed.to(torch.uint32)
    scales = scales.to(torch.uint32)
    tags = torch.full_like(scales, source_tag)
    output = torch.empty_like(expected)
    pointers = [
        make_ptr(Uint32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for t in (packed, scales, tags, output)
    ]
    fn = compile_kernel(StagedMixedPair(), *pointers, current_cuda_stream())
    fn(*pointers, current_cuda_stream())
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*pointers, current_cuda_stream())
    # Invalid source rows must become zero even when their payload contains NaNs.
    tags.fill_(2)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
