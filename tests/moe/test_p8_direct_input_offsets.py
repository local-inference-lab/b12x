"""Live large-pool stores through the direct-input production address helper."""
import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from b12x.moe._shared.kernels.dynamic import _p8_direct_input_block_offsets
from b12x.moe._shared.trellismx.p8_native_kernel import _gptr, current_cuda_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class DirectOffsetProbe:
    @cute.jit
    def __call__(self, payload, scales, offsets, h512: Int64, stream):
        self.kernel(payload, scales, offsets, h512).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, payload, scales, offsets, h512: Int64):
        lane, _, _ = cute.arch.thread_idx()
        block, start = _p8_direct_input_block_offsets(h512, Int32(3), Int32(3))
        p = cute.make_tensor(payload, cute.make_layout((start + Int64(32),)))
        s = cute.make_tensor(scales, cute.make_layout((block + Int64(1),)))
        result = cute.make_tensor(offsets, cute.make_layout((2,)))
        if lane < 32:
            p[start + Int64(lane)] = cutlass.Uint8(lane + Int32(1))
        if lane == 0:
            s[block] = cutlass.Uint8(123)
            result[0], result[1] = block, start


@pytest.mark.parametrize("row", [2**31 // 4096 + 1, 2**31 // 128 + 1])
def test_direct_input_high_payload_scale_stores_and_graph(row):
    h512 = row * 8
    block = h512 * 16 + 15
    start = block * 32
    required = start + 32 + block + 1
    if torch.cuda.mem_get_info()[0] < required + 2 * 1024**3:
        pytest.skip(f"Live high-pool test requires {required + 2 * 1024**3} bytes")
    payload = torch.empty(start + 32, device="cuda", dtype=torch.uint8)
    scales = torch.empty(block + 1, device="cuda", dtype=torch.uint8)
    offsets = torch.empty(2, device="cuda", dtype=torch.int64)
    low_payload = torch.empty(512, device="cuda", dtype=torch.uint8)
    low_scales = torch.empty(16, device="cuda", dtype=torch.uint8)
    ptrs = (_gptr(cutlass.Uint8, payload), _gptr(cutlass.Uint8, scales),
            _gptr(cutlass.Int64, offsets))
    fn = cute.compile(DirectOffsetProbe(), *ptrs, Int64(h512), current_cuda_stream())
    fn(_gptr(cutlass.Uint8, low_payload), _gptr(cutlass.Uint8, low_scales),
       ptrs[2], 0, current_cuda_stream())
    payload[start - 32:].fill_(0xA5)
    scales[block - 1:].fill_(0xA5)
    fn(*ptrs, h512, current_cuda_stream())
    torch.cuda.synchronize()
    assert offsets.tolist() == [block, start]
    assert start > 2**31
    if row > 2**31 // 128:
        assert block > 2**31
    torch.testing.assert_close(payload[-32:], low_payload[-32:], rtol=0, atol=0)
    assert payload[-32:].tolist() == list(range(1, 33))
    assert scales[-1].item() == low_scales[-1].item() == 123
    assert torch.all(payload[start - 32:start] == 0xA5)
    assert scales[block - 1].item() == 0xA5
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*ptrs, h512, current_cuda_stream())
    payload[-32:].zero_()
    scales[-1:].zero_()
    offsets.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(payload[-32:], low_payload[-32:], rtol=0, atol=0)
    assert scales[-1].item() == 123
    assert offsets.tolist() == [block, start]
