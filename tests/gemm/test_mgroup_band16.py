import pytest


def body(**overrides):
    from b12x._lib.dense_gemm import DenseGemmKernel
    kwargs = dict(sf_vec_size=32, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), tile_k=128, mgroup_labels=True)
    kwargs.update(overrides)
    return DenseGemmKernel(**kwargs)


def test_band16_default_and_scope():
    assert not body().mgroup_band16
    candidate = body(mgroup_band16=True)
    assert candidate.mgroup_band16
    assert not candidate.mgroup_joint and not candidate.mgroup_sfb_stage_reuse
    assert candidate.tile_shape_mnk == (128, 128, 128)
    assert candidate.epi_tile == (128, 128)


@pytest.mark.parametrize('override', [
    dict(mgroup_labels=False), dict(mgroup_masked=True), dict(mgroup_joint=True),
    dict(mgroup_compact_sfb=True), dict(mgroup_compact_masked=True),
    dict(mgroup_sfa_prefetch=True), dict(mgroup_role_local_scheduler=True),
    dict(mgroup_sfb_stage_reuse=True), dict(mma_tiler_mn=(64, 128)),
    dict(tile_k=64), dict(sf_vec_size=16), dict(sfb_k_reuse=True),
    dict(direct_one_m_tile_scheduler=True), dict(single_work_tile_per_cta=True),
])
def test_band16_rejects_other_paths(override):
    with pytest.raises(ValueError):
        body(mgroup_band16=True, **override)
