"""MXFP4 packing boundaries, physical padding, and runtime-count reuse."""

import pytest
import torch

import b12x
from b12x.gemm import blockscaled
from tests._reference.helpers import require_b12x


def test_mxfp4_packing_exports_output_mutations():
    source = torch.empty(3, 160, dtype=torch.float16)
    values = torch.empty(3, 80, dtype=torch.uint8)
    scales = torch.empty(128, 8, dtype=torch.uint8)
    def run(x, q, sf):
        blockscaled.quantize_mxfp4(x, out_values=q, out_scales=sf)
        return q, sf
    graph, _ = torch._dynamo.export(run)(source, values, scales)
    assert any(node.target == torch.ops.b12x.quantize_mxfp4.default
               for node in graph.graph.nodes if node.op == "call_function")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [32, 160, 1024])
def test_mxfp4_packing_rounding_padding_and_graph(dtype, k):
    require_b12x()
    m = 129
    pattern = torch.tensor([
        0., -0., .25, -.25, .5, -.5, .75, -.75,
        1, -1, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75,
        2, -2, 2.5, -2.5, 3, -3, 3.5, -3.5,
        4, -4, 5, -5, 6, -6, 5.5, -5.5,
    ], device="cuda", dtype=dtype)
    codes = torch.tensor([
        0, 8, 0, 8, 1, 9, 2, 10, 2, 10, 2, 10, 3, 11, 4, 12,
        4, 12, 4, 12, 5, 13, 6, 14, 6, 14, 6, 14, 7, 15, 7, 15,
    ], device="cuda", dtype=torch.uint8)
    exponents = (torch.arange(m * (k // 32), device="cuda").reshape(m, -1) % 11 - 5)
    source = (pattern * torch.exp2(exponents.float())[:, :, None]).reshape(m, k).to(dtype)
    source[0, :32] = 0.
    source[0, 1:32:2] = -0.
    expected_codes = codes.expand(m, k // 32, 32).clone().reshape(m, k)
    expected_codes[0, :32] = 0
    expected_codes[0, 1:32:2] = 8
    expected_sf = (exponents + 127).to(torch.uint8)
    expected_sf[0, 0] = 0
    cols = ((k // 32 + 3) // 4) * 4
    values = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    scales = torch.empty(256 * cols, dtype=torch.uint8, device="cuda")

    def run(count):
        sf = scales[:((count + 127) // 128) * 128 * cols]
        blockscaled.quantize_mxfp4(source[:count], out_values=values[:count], out_scales=sf)
        return sf

    def check(count, sf):
        packed = expected_codes[:count, ::2] | (expected_codes[:count, 1::2] << 4)
        torch.testing.assert_close(values[:count], packed, rtol=0, atol=0)
        reference = torch.zeros_like(sf)
        row = torch.arange(count, device="cuda")[:, None]
        group = torch.arange(k // 32, device="cuda")[None, :]
        offset = ((row // 128 * (cols // 4) + group // 4) * 512
                  + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)
        reference[offset] = expected_sf[:count]
        torch.testing.assert_close(sf, reference, rtol=0, atol=0)

    run(1)
    b12x.freeze_kernel_resolution("MXFP4 packing runtime rows")
    try:
        for count in (1, 3, 7, 128, 129):
            values.fill_(255)
            scales.fill_(255)
            sf = run(count)
            check(count, sf)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sf = run(m)
        for _ in range(2):
            source.neg_()
            expected_codes.bitwise_xor_(8)
            values.fill_(255)
            scales.fill_(255)
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            check(m, sf)
    finally:
        b12x.unfreeze_kernel_resolution()


def test_mxfp4_packing_rejects_invalid_buffers_and_accepts_zero_rows():
    require_b12x()
    source = torch.empty(3, 160, device="cuda", dtype=torch.bfloat16)
    values = torch.empty(3, 80, device="cuda", dtype=torch.uint8)
    scales = torch.empty(1024, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="padded F8_128x4"):
        blockscaled.quantize_mxfp4(source, out_values=values, out_scales=scales[:512])
    with pytest.raises(ValueError, match="must not overlap"):
        blockscaled.quantize_mxfp4(
            source, out_values=source.view(torch.uint8).reshape(-1)[:240].view(3, 80),
            out_scales=scales,
        )
    with pytest.raises(ValueError, match="contiguous"):
        strided = torch.empty(3, 320, device="cuda", dtype=source.dtype)[:, ::2]
        blockscaled.quantize_mxfp4(strided, out_values=values, out_scales=scales)
    b12x.freeze_kernel_resolution("empty MXFP4 packing must not launch")
    try:
        blockscaled.quantize_mxfp4(source[:0], out_values=values[:0], out_scales=scales[:0])
    finally:
        b12x.unfreeze_kernel_resolution()
