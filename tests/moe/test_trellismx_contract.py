"""CPU contract tests; not a substitute for device decoder/MMA closure."""
import pytest

from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.trellismx.p8_smallm_schedule import (
    P8SmallMGeometry,
    p8_small_m_scratch_layout,
)


def make_kernel(bits, codebook="mcg", **kwargs):
    return MoEDynamicKernelBackend(
        16, (16, 128), activation="silu", quant_recipe="w4a8_trellis",
        w4a8_repacked=True, trellis_bits=bits, trellis_codebook=codebook,
        trellis_scaled=True, trellis_identity_boundary=True, **kwargs,
    )


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_mcg_rate_keeps_native_scale_contract(bits):
    kernel = make_kernel(bits)
    assert kernel.trellis_bits == bits
    assert kernel.trellis_scaled
    assert kernel.trellis_codebook == "mcg"


@pytest.mark.parametrize("bits,law", [(2, "mcg"), (5, "sqg-xor-cheb-t12"), (6, "mcg")])
def test_unsupported_rate_law_fails_before_compilation(bits, law):
    with pytest.raises(ValueError):
        make_kernel(bits, law)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_existing_sqg_rates_still_construct(bits):
    assert make_kernel(bits, "sqg-xor-cheb-t12").trellis_bits == bits


@pytest.mark.parametrize("tokens", [1, 4, 17, 128])
def test_coupled_workspace_regions_do_not_alias(tokens):
    layout = p8_small_m_scratch_layout(tokens=tokens, shared=True)
    end = 0
    for region in layout.regions:
        assert region.offset % 16 == 0
        assert region.offset >= end
        end = region.offset + region.nbytes
    assert layout.nbytes >= end


def test_fc_owners_cover_selected_experts_without_overlap():
    geometry = P8SmallMGeometry()
    assert len({geometry.fc1_owner(i) for i in range(geometry.fc1_tasks)}) == 32
    assert len({geometry.fc2_owner(i) for i in range(geometry.fc2_tasks)}) == 128
    with pytest.raises(ValueError):
        geometry.fc1_owner(geometry.fc1_tasks)


def test_environment_cannot_change_default_artifact_codebook(monkeypatch):
    monkeypatch.setenv("B12X_TRELLIS_CODEBOOK", "mcg")
    assert make_kernel(4, None).trellis_codebook == "sqg-xor-cheb-t12"


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_direct_owner_retains_padded_input_and_scale_extents(tokens):
    layout = p8_small_m_scratch_layout(tokens=tokens, shared=True, grouped=False)
    shapes = {region.name: region.shape for region in layout.regions}
    assert shapes["packed_a"] == (tokens * 8 * 16 * 4096,)
    assert shapes["scale_flat"] == ((288 + tokens * 8 + 1) * 16 * 512,)


@pytest.mark.parametrize("tokens", [1, 4, 128])
def test_grouped_owner_uses_compact_input_and_scale_planes(tokens):
    layout = p8_small_m_scratch_layout(tokens=tokens, shared=True, grouped=True)
    shapes = {region.name: region.shape for region in layout.regions}
    assert shapes["packed_a"] == (tokens * 4096,)
    assert shapes["scale_flat"] == (tokens * 128,)


def test_unsupported_tp2_rejected_before_sidecar_access():
    from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE
    with pytest.raises(ValueError, match="TP4 only"):
        P8NativeTPMoE("/does/not/exist", layer=3, tp_rank=0, world_size=2, device="cpu")


def test_noncoupled_materialized_owner_keeps_grouped_padding():
    layout = p8_small_m_scratch_layout(tokens=17, shared=True, grouped=False,
                                      tile_m=64, direct=False)
    shapes = {region.name: region.shape for region in layout.regions}
    assert shapes["packed_a"] == ((288 + 3) * 64 * 4096,)
    assert shapes["scale_flat"] == ((288 + 17 * 8 + 1) * 64 * 512,)


def test_unimplemented_grouped_m16_rejected_before_kernel_import():
    with pytest.raises(ValueError, match="P8 grouped-M16 is unsupported"):
        MoEDynamicKernelBackend(
            16, (16, 128), activation="silu", quant_recipe="w4a8_trellis",
            trellis_bits=4, trellis_codebook="mcg", trellis_scaled=True,
            w4a8_repacked=True, p8_full_coupled=True, p8_scale_sandwich=True,
            trellis_coupled=True, materialize_intermediate=True,
            direct_routing=False, share_input_across_experts=True,
            deterministic_output=True,
        )
