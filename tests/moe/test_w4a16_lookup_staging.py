"""Only SQG Trellis payloads own a lookup table at the optional LUT pointer."""

import pytest

from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel


@pytest.mark.parametrize("layout", ["packed", "modelopt", "trellis_t256"])
@pytest.mark.parametrize("enabled", [False, True])
def test_optional_lookup_staging_requires_trellis_payload(layout, enabled, monkeypatch):
    monkeypatch.setenv("B12X_SQG_XOR_CHEB_T12_SMEM", str(int(enabled)))
    kernel = W4A16FusedMoeKernel(
        size_m=128,
        hidden_size=2560,
        intermediate_size=640,
        num_experts=256,
        top_k=10,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=True,
        fc1_tile_n=128,
        fc1_tile_k=64,
        fc2_tile_n=128,
        fc2_tile_k=64,
        moe_block_size=16,
        max_m_blocks=512,
        weight_layout=layout,
        w13_layout="packed" if layout == "trellis_t256" else "w13",
        scale_format="e4m3_k32" if layout == "trellis_t256" else "e4m3_k16",
    )
    staged = enabled and layout == "trellis_t256"
    assert kernel.sqg_xor_cheb_t12_smem == staged
    assert kernel.fc1.sqg_xor_cheb_t12_smem == staged
    assert kernel.fc2.sqg_xor_cheb_t12_smem == staged
    required_words = max(kernel.fc1.shared_words, kernel.fc2.shared_words)
    assert kernel.shared_words == required_words + (1024 if staged else 0)
