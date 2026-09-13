import torch

from b12x._lib.intrinsics import swizzle_block_scale
from tests._reference.sm103_moe import unswizzle, unpack, reference
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
