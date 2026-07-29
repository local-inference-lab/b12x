"""Bit-equality and regime isolation for the MX-FP6 wide-N prefill tile.

The wide-N m>16 default tile is (128,128) (``SPARKINFER_FP6_LARGE_M_TILE``
overrides for A/B). It was briefly (128,64): the sweep that chose that could
only see (128,128) starved to a single mainloop stage by a full-tile epilogue,
and once ``_choose_epilogue`` frees those bytes the wide tile wins by 17-19% on
every Behemoth shard. Tiles only change the CTA work decomposition — the
per-output-element accumulation order is identical — so outputs must be
BIT-IDENTICAL across tiles. The m<=16 decode regime takes the wave-cliff
heuristic ((16,64), or (32,128) when ceil(N/64) leaves a tiny tail wave);
`SPARKINFER_FP6_DECODE_TILE` forces a fixed tile for A/B.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@cuda_required
@pytest.mark.parametrize("m", [32, 128, 256])
def test_fp6_large_m_tile_bit_exact_vs_old_pin(m, monkeypatch):
    import sparkinfer._lib.dense_gemm as dg
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(0)
    n, k = 6144, 512
    fp6w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)
    weight = fp6w.expanded_weight()

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_TILE", (128, 128))
    y_old = fdw.dense_fp6_linear_expanded(x, weight, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_TILE", (128, 64))
    y_new = fdw.dense_fp6_linear_expanded(x, weight, *args)

    torch.testing.assert_close(y_new, y_old, rtol=0.0, atol=0.0)


def test_fp6_tile_regime_selection():
    from sparkinfer._lib.dense_gemm import _select_default_mma_tiler_mn

    common = dict(sm_count=188, is_mxfp8=False, is_mxfp6=True)
    # Decode regime: width-64 unless ceil(N/64) leaves a tiny tail wave.
    for m in (1, 2, 8, 16):
        # qkv N=7168: 112 CTAs, single wave -> (16,64).
        assert _select_default_mma_tiler_mn(m, 7168, **common) == (16, 64)
        # gate_up N=28672: 448 CTAs, healthy 72-CTA tail -> (16,64).
        assert _select_default_mma_tiler_mn(m, 28672, **common) == (16, 64)
        # o/down N=12288 was exempted to (32,128) to dodge a 4-CTA tail wave on
        # 188 SMs. Retired Jul 26 2026: that tail is an occupancy-1 artifact,
        # and _dense_gemm_target_occupancy now returns 2 for exactly these
        # shapes, so width-64 wins outright. Decode is width-64 everywhere.
        assert _select_default_mma_tiler_mn(m, 12288, **common) == (16, 64)
    # Wide-N prefill regime takes one tile for every m > 16.
    for m in (17, 32, 512, 8192):
        assert _select_default_mma_tiler_mn(m, 7168, **common) == (128, 128)
    # Narrow-N keeps the unmeasured coarse default.
    assert _select_default_mma_tiler_mn(8192, 1024, **common) == (128, 128)
    # A declared expected_m regime hint owns the decision.
    assert _select_default_mma_tiler_mn(
        1, 7168, expected_m=8192, **common
    ) == (128, 128)
    assert _select_default_mma_tiler_mn(
        8192, 7168, expected_m=8, **common
    ) == (16, 64)


@cuda_required
@pytest.mark.parametrize("m", [1, 2, 8, 16])
def test_fp6_decode_tile_bit_exact_vs_old_pin(m, monkeypatch):
    """Heuristic decode tiles are bit-identical to the old (16,128) pin.

    Runs the PACKED weight stream (the production decode path) through both
    the forced old tile and the heuristic default.
    """
    import sparkinfer._lib.dense_gemm as dg
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(1)
    n, k = 6144, 512
    fp6w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_DECODE_TILE", (16, 128))
    y_old = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_DECODE_TILE", None)
    y_new = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)

    torch.testing.assert_close(y_new, y_old, rtol=0.0, atol=0.0)


def test_choose_epilogue_only_shrinks_to_buy_a_stage(monkeypatch):
    """The full-tile epilogue must survive unless shrinking buys a stage.

    This tie-break is the whole safety argument for the policy: it is what
    leaves (128,64) prefill and (16,64) decode compiling byte-identical kernels
    while (128,128) picks up a second mainloop stage.
    """
    import sparkinfer._lib.dense_gemm as dg

    monkeypatch.setattr(dg, "_SPARKINFER_DENSE_EPI_TILE", None)
    kernel = dg.DenseGemmKernel

    # Decode/(128,64) shape of the problem: the epilogue was never binding, so
    # both candidates reach the same depth and the larger tile must win.
    seen = []

    def probe_no_gain(epi_tile, cap):
        seen.append((epi_tile, cap))
        return 3, 1

    assert kernel._choose_epilogue((128, 64), (64, 16), probe_no_gain) == (
        (128, 64),
        0,
    )
    assert seen == [((128, 64), 0), ((64, 32), 2)]

    # (128,128) shape of the problem: halving buys a stage, so take it.
    def probe_gain(epi_tile, cap):
        return (1, 1) if epi_tile == (128, 128) else (2, 2)

    assert kernel._choose_epilogue((128, 128), (64, 16), probe_gain) == (
        (64, 64),
        2,
    )

    # A shrink that buys nothing but costs nothing is still refused, because a
    # smaller epilogue means more TMA stores for the same shared memory.
    def probe_equal_deeper(epi_tile, cap):
        return 2, 2

    assert kernel._choose_epilogue((128, 128), (64, 16), probe_equal_deeper) == (
        (128, 128),
        0,
    )


def test_choose_epilogue_refuses_sub_atom_tiles():
    """A sub-atom epilogue must never be selected, however much it would buy.

    MmaMPerEpiM is epi_m // mma_tile_m, so an epilogue shorter than one atom
    floors the accumulator copy loop to zero trips and the TMA stores
    uninitialized shared memory. There is no exception and no compile error -
    it is silently NaN - so the guard is a hard precondition, not a heuristic.
    Decode's (16,64) on a 16x16 atom is the live case.
    """
    import sparkinfer._lib.dense_gemm as dg

    def probe_huge_gain(epi_tile, cap):
        return (1, 1) if cap == 0 else (99, 2)

    for tile in ((16, 64), (16, 128), (16, 32)):
        assert dg.DenseGemmKernel._choose_epilogue(
            tile, (16, 16), probe_huge_gain
        ) == (tile, 0)


def test_choose_epilogue_refuses_the_m1_register_store_path():
    """A non-smem-staged epilogue must keep the full tile.

    use_m1_non_tma_c swaps the TMA store for a direct store out of registers,
    so sC is not the staging buffer the policy assumes. Sizing a sub-tiled
    multi-stage epilogue around it writes garbage rows without erroring. Live
    case: the narrow-N (n <= 1536) coarse (128,128) tile serving m=1, where the
    atom guard alone would happily allow (64,64).
    """
    import sparkinfer._lib.dense_gemm as dg

    def probe_gain(epi_tile, cap):
        return (1, 1) if cap == 0 else (2, 2)

    assert dg.DenseGemmKernel._choose_epilogue(
        (128, 128), (64, 16), probe_gain, stages_through_smem=False
    ) == ((128, 128), 0)
    # ... and the same tile does shrink when the epilogue really is staged.
    assert dg.DenseGemmKernel._choose_epilogue(
        (128, 128), (64, 16), probe_gain, stages_through_smem=True
    ) == ((64, 64), 2)


def test_choose_epilogue_defers_to_env_override(monkeypatch):
    import sparkinfer._lib.dense_gemm as dg

    monkeypatch.setattr(dg, "_SPARKINFER_DENSE_EPI_TILE", (64, 64))

    def probe_gain(epi_tile, cap):  # pragma: no cover - must not be consulted
        raise AssertionError("policy ran despite an explicit override")

    assert dg.DenseGemmKernel._choose_epilogue(
        (128, 128), (64, 16), probe_gain
    ) == ((64, 64), 0)
    # The override is an upper bound per mode, so a tile smaller than the
    # request clamps instead of raising - a serving process compiles the decode
    # tile in the same interpreter as the prefill tile it was aimed at.
    assert dg.DenseGemmKernel._choose_epilogue(
        (16, 64), (16, 16), probe_gain
    ) == ((16, 64), 0)


def test_parse_tile_env_guard(monkeypatch):
    from sparkinfer._lib.dense_gemm import _parse_tile_env

    monkeypatch.setenv("SPARKINFER_FP6_LARGE_M_TILE", "64x128")
    assert _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64)) == (64, 128)
    monkeypatch.setenv("SPARKINFER_FP6_LARGE_M_TILE", "bogus")
    with pytest.raises(ValueError, match="128x64"):
        _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64))
    monkeypatch.delenv("SPARKINFER_FP6_LARGE_M_TILE")
    assert _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64)) == (128, 64)
