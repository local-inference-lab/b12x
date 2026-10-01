import pytest


def body(**overrides):
    from b12x._lib.dense_gemm import DenseGemmKernel
    kwargs = dict(sf_vec_size=32, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), tile_k=128, mgroup_labels=True)
    kwargs.update(overrides)
    return DenseGemmKernel(**kwargs)


def test_deferred_wait_default():
    assert not body().mgroup_deferred_wait
    candidate = body(mgroup_deferred_wait=True)
    assert candidate.mgroup_deferred_wait
    assert not candidate.mgroup_band16 and not candidate.mgroup_sfb_stage_reuse
    assert candidate.epi_tile == (128, 128)


@pytest.mark.parametrize('override', [
    dict(mgroup_labels=False), dict(mgroup_masked=True), dict(mgroup_joint=True),
    dict(mgroup_compact_sfb=True), dict(mgroup_compact_masked=True),
    dict(mgroup_sfa_prefetch=True), dict(mgroup_role_local_scheduler=True),
    dict(mgroup_sfb_stage_reuse=True), dict(mgroup_band16=True),
    dict(mma_tiler_mn=(64, 128)), dict(tile_k=64), dict(sf_vec_size=16),
    dict(sfb_k_reuse=True), dict(b_packed=True), dict(block_fp8=True), dict(weight_only='mxfp8'),
])
def test_deferred_wait_scope(override):
    with pytest.raises(ValueError):
        body(mgroup_deferred_wait=True, **override)
