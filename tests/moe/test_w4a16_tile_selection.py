from b12x.moe._shared.kernels.w4a16.kernel import _candidate_tile_fits


def _fits(*, tile_k: int, tile_n: int, cta_threads: int) -> bool:
    return _candidate_tile_fits(
        problem_n=4096,
        problem_k=512,
        cta_m_blocks=1,
        tile_n=tile_n,
        tile_k=tile_k,
        cta_threads=cta_threads,
        max_shared_mem=1 << 30,
        scale_format="e8m0_k32",
        weight_layout="modelopt",
        weight_bits=4,
    )


def test_wave_balanced_fc2_tile_is_valid_as_an_explicit_pin() -> None:
    assert _fits(tile_k=32, tile_n=512, cta_threads=256)


def test_other_sub64_k_tiles_remain_unsupported() -> None:
    assert not _fits(tile_k=32, tile_n=256, cta_threads=128)
    assert not _fits(tile_k=16, tile_n=512, cta_threads=128)
