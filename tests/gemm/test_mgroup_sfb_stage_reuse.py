import pytest


def make_body(**overrides):
    from b12x._lib.dense_gemm import DenseGemmKernel
    options = dict(sf_vec_size=32, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
                   tile_k=128, mgroup_labels=True)
    options.update(overrides)
    return DenseGemmKernel(**options)


def test_sfb_stage_reuse_default_and_separate_generic_flag():
    current = make_body()
    candidate = make_body(mgroup_sfb_stage_reuse=True)
    assert not current.mgroup_sfb_stage_reuse
    assert candidate.mgroup_sfb_stage_reuse
    assert not candidate.sfb_k_reuse
    assert not candidate.direct_sfb_representative
    assert candidate.tile_shape_mnk == current.tile_shape_mnk == (128, 128, 128)
    assert candidate.epi_tile == current.epi_tile == (128, 128)
    with pytest.raises(ValueError):
        make_body(sfb_k_reuse=True)


@pytest.mark.parametrize('overrides', [
    dict(mgroup_labels=False),
    dict(mgroup_labels=False, mgroup_masked=True),
    dict(mgroup_joint=True, mgroup_capacity_tiles=128),
    dict(mgroup_compact_sfb=True),
    dict(mgroup_compact_masked=True),
    dict(mgroup_sfa_prefetch=True),
    dict(mgroup_role_local_scheduler=True),
    dict(mma_tiler_mn=(64, 128)),
    dict(mma_tiler_mn=(128, 64)),
    dict(tile_k=64),
    dict(sf_vec_size=16),
    dict(sfb_k_reuse=True),
    dict(swap_ab=True),
    dict(load_path='cpasync'),
])
def test_sfb_stage_reuse_scope(overrides):
    with pytest.raises(ValueError):
        make_body(mgroup_sfb_stage_reuse=True, **overrides)
