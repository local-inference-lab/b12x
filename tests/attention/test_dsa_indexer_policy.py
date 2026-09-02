"""Host-side contract of the raced DSA indexer policy and its plan plumbing."""

from __future__ import annotations

import pytest

from b12x.attention.dsa_indexer._policy import (
    DSA_INDEXER_POLICY,
    INDEXER_FUSED_CTAS_WAVES,
    INDEXER_SUPERTILE_ALIGN,
    DsaIndexerConfig,
    DsaIndexerQuery,
)
from b12x.attention.dsa_indexer.scratch import (
    _PAGED_INDEX_TILE_BLOCK_K,
    B12XIndexerScratchCaps,
    _apply_indexer_policy,
)
from b12x.policy import DeviceIdentity, FrozenMapping

GB10 = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="NVIDIA GB10",
)


def _query(**fields: object) -> DsaIndexerQuery:
    base: dict[str, object] = {
        "source_layout": "paged",
        "mode": "decode",
        "dtype": "bfloat16",
        "kv_dtype": "uint8",
        "num_q_heads": 32,
        "num_idx_heads": 1,
        "max_q_rows": 4,
        "max_k_rows": 0,
        "max_page_table_width": 512,
        "top_k": 512,
        "page_size": 64,
        "score_mode": "dsa",
        "shared_page_table": False,
    }
    base.update(fields)
    return DsaIndexerQuery(**base)


def test_all_auto_config_is_the_heuristic_and_round_trips() -> None:
    config = DSA_INDEXER_POLICY.heuristic(_query(), GB10)
    assert config == DsaIndexerConfig()
    assert DsaIndexerConfig.from_profile(FrozenMapping(config.to_dict())) == config
    assert set(config.to_dict()) == DSA_INDEXER_POLICY.config_fields
    assert INDEXER_SUPERTILE_ALIGN == _PAGED_INDEX_TILE_BLOCK_K


@pytest.mark.parametrize(
    ("config", "query", "match"),
    [
        (DsaIndexerConfig(route="fused"), _query(), "unsupported DSA indexer route"),
        (
            DsaIndexerConfig(route="paged_fused"),
            _query(mode="prefill", shared_page_table=True),
            "decode-only",
        ),
        (
            DsaIndexerConfig(route="paged_tiled"),
            _query(source_layout="contiguous", max_k_rows=8192),
            "paged source layout",
        ),
        (DsaIndexerConfig(supertile_k=1000), _query(), "multiple of"),
        (
            DsaIndexerConfig(fused_ctas_per_group=INDEXER_FUSED_CTAS_WAVES * 48 + 1),
            _query(max_q_rows=1),
            "oversubscribes",
        ),
        (DsaIndexerConfig(fused_merge_threshold=-2), _query(), ">= -1"),
        (DsaIndexerConfig(two_level_fold="maybe"), _query(), "two_level_fold"),
    ],
)
def test_validate_rejects_configs_the_runtime_cannot_launch(
    config, query, match
) -> None:
    with pytest.raises(ValueError, match=match):
        DSA_INDEXER_POLICY.validate_config(query, config, GB10)


def test_validate_accepts_a_full_double_wave_at_the_budget() -> None:
    DSA_INDEXER_POLICY.validate_config(
        _query(max_q_rows=4),
        DsaIndexerConfig(
            route="paged_fused",
            fused_ctas_per_group=INDEXER_FUSED_CTAS_WAVES * 48 // 4,
            fused_merge_threshold=0,
        ),
        GB10,
    )


def _caps(**fields: object) -> B12XIndexerScratchCaps:
    base: dict[str, object] = {
        "device": "cpu",
        "source_layout": "paged",
        "num_q_heads": 32,
        "max_q_rows": 4,
        "max_page_table_width": 512,
        "topk": 512,
    }
    base.update(fields)
    return B12XIndexerScratchCaps(**base)


def test_profile_config_fills_only_the_auto_caps(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    monkeypatch.delenv("B12X_FUSED_INDEXER", raising=False)
    profile = DsaIndexerConfig(
        route="paged_fused",
        supertile_k=8192,
        fused_ctas_per_group=12,
        fused_merge_threshold=0,
        two_level_fold="off",
    )
    assert _apply_indexer_policy(_caps(), profile) == profile
    explicit = _apply_indexer_policy(
        _caps(
            route="paged_tiled",
            supertile_k=16384,
            fused_ctas_per_group=3,
            fused_merge_threshold=7,
            two_level_fold="on",
        ),
        profile,
    )
    assert explicit == DsaIndexerConfig(
        route="paged_tiled",
        supertile_k=16384,
        fused_ctas_per_group=3,
        fused_merge_threshold=7,
        two_level_fold="on",
    )


def test_debug_environment_still_overrides_the_profile(monkeypatch) -> None:
    profile = DsaIndexerConfig(route="paged_fused", supertile_k=8192)
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "4096")
    monkeypatch.setenv("B12X_FUSED_INDEXER", "0")
    resolved = _apply_indexer_policy(_caps(), profile)
    # The env supertile is read by the runtime resolver, so the plan leaves
    # the caps value untouched; the disabled fused route falls back to auto.
    assert resolved.supertile_k == 0
    assert resolved.route == "auto"
