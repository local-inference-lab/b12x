import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe._shared.kernels.materialized_nvfp4_reference import (
    unswizzle, unpack, reference, quantize_dequantize,
)
from tests.architecture.test_sm103 import make_experts


def test_prepared_scale_atom_roundtrip_with_distinct_rows_and_k_blocks():
    x = (
        torch.arange(2 * 256 * 16)
        .reshape(2, 256, 16)
        .remainder(30)
        .float()
        .to(torch.float8_e4m3fn)
    )
    assert torch.equal(unswizzle(swizzle_block_scale(x), 256, 256), x.float())


def test_fp4_codes_and_k16_scale_boundaries():
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8
    )
    actual = unpack(codes, torch.tensor([[1.0, 2.0]]))
    expected = torch.tensor(
        [[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6.0]]
    )
    torch.testing.assert_close(actual, torch.cat([expected, 2 * expected], -1))


def test_zero_expert_and_invalid_route_are_exact_zero():
    experts = make_experts()._impl
    x = torch.randn((2, 256), dtype=torch.bfloat16)
    ids = torch.tensor([[-1, 2], [0, 1]])
    actual = reference(x, experts, ids, torch.ones(2, 2))
    assert torch.equal(actual, torch.zeros_like(x))


@pytest.mark.parametrize(
    "value,global_scale,expected",
    [(0.0, 1.0, 0.0), (4096.0, 1.0, 2688.0), (-4096.0, 2.0, -2688.0),
     (6.0, 0.5, 3.0), (6.0, 2.0, 12.0), (2**-20, 1.0, 0.0)],
)
def test_quantization_scale_saturation_zero_and_underflow(value, global_scale, expected):
    actual = quantize_dequantize(torch.full((16,), value), global_scale)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.full((16,), expected), rtol=0, atol=0)


def test_quantization_fp4_ties_round_to_even():
    values = torch.tensor([6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
    x = torch.cat((values, -values))
    positive = torch.tensor([6, 0, 1, 1, 2, 2, 4, 4.0])
    expected = torch.cat((positive, -positive))
    torch.testing.assert_close(quantize_dequantize(x, 1.0), expected, rtol=0, atol=0)


def test_materialized_fc2_rounding_precedes_router_weight():
    experts = make_experts(e=1)._impl
    # Unit diagonals make both GEMMs analytically tractable: input 6 produces
    # a requantized SiLU intermediate of 36, then FC2 scales it by 1.003.
    columns = torch.arange(256)
    codes = (2 << (4 * (columns % 2))).to(torch.uint8)
    for row_offset in (0, 256):
        experts.w1_fp4[0, columns + row_offset, columns // 2] = codes
    experts.w2_fp4[0, columns, columns // 2] = codes
    experts.w2_alphas.fill_(1.003)
    x = torch.full((1, 256), 6.0, dtype=torch.bfloat16)
    actual = reference(x, experts, torch.tensor([[0]]), torch.tensor([[0.37]]))
    # BF16(36 * 1.003) = 36; BF16(36 * 0.37) = 13.3125.
    torch.testing.assert_close(actual.float(), torch.full((1, 256), 13.3125), rtol=0, atol=0)
