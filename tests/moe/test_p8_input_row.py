"""Live large-row addressing through the production coupled input helper."""
import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.trellismx.p8_native_kernel import _gptr, current_cuda_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class InputRowProbe:
    def __init__(self):
        self.owner = object.__new__(MoEDynamicKernelBackend)
        self.owner.input_warps_per_token = 2

    @cute.jit
    def __call__(self, x, packed, scales, component, row: Int32, stream):
        self.kernel(x, packed, scales, component, row).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, x, packed, scales, component, row: Int32):
        tid, _, _ = cute.arch.thread_idx()
        # Broadcast one real input row; only output row-offset arithmetic is
        # under test, avoiding a 128-GiB redundant BF16 input allocation.
        inp = cute.make_tensor(
            x, cute.make_layout((row + 1, 4096), stride=(Int64(0), Int64(1))),
        )
        payload = cute.make_tensor(
            packed, cute.make_layout(((Int64(row) + 1) * Int64(4096),)),
        )
        scale = cute.make_tensor(
            scales, cute.make_layout(((Int64(row) + 1) * Int64(128),)),
        )
        comp = cute.make_tensor(component, cute.make_layout((4096,)))
        self.owner._store_p8_full_coupled_input_row(
            inp, payload, scale, comp, row, tid // 32, tid % 32, Int32(128),
        )


@pytest.mark.parametrize("row", [2**31 // 4096 + 1, 2**31 // 128 + 1])
def test_large_payload_and_scale_row_match_rebased_helper_and_graph(row):
    required = (row + 1) * (4096 + 128)
    if torch.cuda.mem_get_info()[0] < required + 2 * 1024**3:
        pytest.skip(f"Live large-row allocation requires {required + 2 * 1024**3} bytes")
    x = torch.linspace(-0.8, 0.9, 4096, device="cuda", dtype=torch.bfloat16)
    component = torch.linspace(0.5, 1.5, 4096, device="cuda", dtype=torch.float16)
    payload = torch.empty((row + 1) * 4096, device="cuda", dtype=torch.uint8)
    scales = torch.empty((row + 1) * 128, device="cuda", dtype=torch.uint8)
    expected = torch.empty(4096, device="cuda", dtype=torch.uint8)
    expected_scales = torch.empty(128, device="cuda", dtype=torch.uint8)
    args = (_gptr(cutlass.BFloat16, x), _gptr(cutlass.Uint8, payload),
            _gptr(cutlass.Uint8, scales), _gptr(cutlass.Float16, component))
    stream = current_cuda_stream()
    fn = cute.compile(InputRowProbe(), *args, Int32(row), stream)
    fn(args[0], _gptr(cutlass.Uint8, expected),
       _gptr(cutlass.Uint8, expected_scales), args[3], 0, stream)
    payload[-4096:].fill_(0xA5)
    scales[-128:].fill_(0xA5)
    fn(*args, row, stream)
    torch.cuda.synchronize()
    assert torch.count_nonzero(expected) > 0
    torch.testing.assert_close(payload[-4096:], expected, rtol=0, atol=0)
    torch.testing.assert_close(scales[-128:], expected_scales, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args, row, current_cuda_stream())
    payload[-4096:].zero_()
    scales[-128:].zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(payload[-4096:], expected, rtol=0, atol=0)
    torch.testing.assert_close(scales[-128:], expected_scales, rtol=0, atol=0)
