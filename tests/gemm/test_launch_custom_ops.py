from __future__ import annotations

import pytest

from b12x.norm.mhc._policy import MhcConfig


def test_mhc_projection_cache_key_uses_planned_geometry_not_live_rows() -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    config = MhcConfig(
        backend="tf32_tma",
        decode_partials_schedule="default",
        projection_tile_m=64,
        projection_tile_n=24,
        projection_tile_k=64,
        projection_num_stages=2,
        projection_num_m_warps=4,
        projection_num_n_warps=1,
        projection_k_splits=8,
    )
    geometry = (
        4_096,
        64,
        config.projection_tile_m,
        config.projection_tile_n,
        config.projection_tile_k,
        config.projection_num_stages,
        config.projection_num_m_warps,
        config.projection_num_n_warps,
        config.projection_k_splits,
    )

    assert residual_kernels._prefill_tf32_project_kernel(
        *geometry
    ) is residual_kernels._prefill_tf32_project_kernel(*geometry)
    assert {
        residual_kernels.mhc_prefill_tf32_project_splits(
            tokens=tokens,
            hidden_size=4_096,
            config=config,
        )
        for tokens in (2_304, 2_511, 3_071)
    } == {8}


def test_mhc_decode_split_n_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", "4")
    monkeypatch.setenv("B12X_MHC_DECODE_TILE_N", "3")
    assert residual_kernels._selected_post_pre_decode_split_n(
        num_tokens=16,
        hidden_size=4096,
        compute_capability=(12, 1),
    ) == (4, 3)


def test_mhc_sm121_decode_split_n_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    monkeypatch.delenv("B12X_MHC_DECODE_TILE_N", raising=False)
    select = residual_kernels._selected_post_pre_decode_split_n

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == (0, 0)
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == (4, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == (8, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == (0, 0)
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == (0, 0)


def test_mhc_decode_finalize_threads_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_FINALIZE_THREADS", "128")
    assert (
        residual_kernels._selected_mhc_decode_finalize_threads(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
        )
        == 128
    )


def test_mhc_sm121_decode_finalize_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_FINALIZE_THREADS", raising=False)
    select = residual_kernels._selected_mhc_decode_finalize_threads

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 0
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 512
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 128
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 0
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 0


def test_mhc_sm121_decode_partial_group_policy(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=2, hidden_size=4096, compute_capability=(12, 1)) == 4
    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 9
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 25
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 25


def test_mhc_decode_partial_group_policy_preserves_sm120(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 4
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 4


@pytest.mark.parametrize(
    ("tokens", "partials_per_cta"),
    (
        (1, 1),
        (2, 2),
        (4, 3),
        (8, 5),
        (16, 7),
        (24, 13),
        (95, 13),
        (96, 4),
        (128, 4),
    ),
)
def test_mhc_profiled_decode_partial_group_schedule(
    monkeypatch,
    tokens: int,
    partials_per_cta: int,
) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)

    assert residual_kernels._selected_post_pre_partials_per_cta(
        num_tokens=tokens,
        hidden_size=4096,
        compute_capability=(12, 0),
        schedule="hidden4096_m128_v1",
    ) == partials_per_cta


def test_mhc_decode_partial_group_environment_override(monkeypatch) -> None:
    import b12x.norm.mhc._kernels as residual_kernels

    monkeypatch.setenv("B12X_MHC_PARTIALS_PER_CTA", "7")
    assert (
        residual_kernels._selected_post_pre_partials_per_cta(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
            schedule="hidden4096_m128_v1",
        )
        == 7
    )
