"""Experimental mapped launches preserve arithmetic and prepared ownership."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.moe.hybrid_prefill_operator import weight_schedule
from b12x.moe.fused_moe._cache_tuning import ExpertCacheConfig, ExpertCacheQuery, TUNING
from b12x.moe.fused_moe._preparation import _W4A16PrimaryLaunches


def query():
    return ExpertCacheQuery(experts=256, resident=80, hidden=2048, intermediate=512,
                            max_tokens=256, top_k=10, w13_layout="w31",
                            checkpoint_fingerprint="a" * 64, profile_hash="b" * 64)


def test_variant_requires_explicit_selection():
    q = query()
    assert TUNING.default_config(q, None) == ExpertCacheConfig()
    TUNING.validate_config(q, ExpertCacheConfig(cold_prefill="two_cta"), None)
    assert TUNING.knobs[-1].values == ("fused",)
    with pytest.raises(ValueError, match="variant"):
        TUNING.validate_config(q, ExpertCacheConfig(cold_prefill="unknown"), None)
    with pytest.raises(ValueError, match="16 tokens"):
        TUNING.validate_config(replace(q, max_tokens=4), ExpertCacheConfig(cold_prefill="two_cta"), None)


def test_live_rows_reuse_launch_and_decode_keeps_baseline():
    launches = _W4A16PrimaryLaunches(tokens=256, route_mode="packed",
        packed=SimpleNamespace(schedule_whole_tiles=True), packed_mapped=object(),
        direct=None, direct_mapped=None, topk_sum=object(), mapped_topk_sum=object(),
        route_pack=object(), cold_prefill=object())
    for rows in (1, 4, 16, 63, 64, 128, 256):
        for mapped in (False, True):
            got = launches.select(tokens=rows, route_ids_dtype=torch.int32,
                                  has_route_map=mapped, activation_amax=None)[0]
            expected = (launches.cold_prefill if rows >= 16 else launches.packed_mapped) if mapped else launches.packed
            assert got is expected
    with pytest.raises(ValueError, match="maxima"):
        launches.select(tokens=64, route_ids_dtype=torch.int32, has_route_map=True, activation_amax=object())


def test_traffic_counts_route_blocks_not_token_count():
    result = weight_schedule([0] * 9 + [1], hidden=128, intermediate=128,
                             block_rows=8, fc1_tile=(64, 128), fc2_tile=(64, 128))
    assert result["active_experts"] == 2
    assert [x["blocks"] for x in result["experts"]] == [2, 1]
    assert result["fc1"]["tile_requests"] == 12
    assert result["fc1"]["repeated_tile_requests"] == 4
    assert result["fc1"]["unique_weight_scale_bytes"] == 2 * 256 * 128 * 9 // 16
    assert result["fc1"]["scheduled_weight_scale_bytes"] == 3 * 256 * 128 * 9 // 16
    assert result["fc2"]["scheduled_weight_scale_bytes"] * 2 == result["fc1"]["scheduled_weight_scale_bytes"]


def test_narrow_scale_tiles_repeat_sectors_without_repeating_logical_bytes():
    common = dict(hidden=2048, intermediate=512, block_rows=8)
    wide = weight_schedule([0] * 9, **common, fc1_tile=(128, 128), fc2_tile=(128, 128))
    narrow = weight_schedule([0] * 9, **common, fc1_tile=(128, 64), fc2_tile=(128, 64))
    for phase in ("fc1", "fc2"):
        assert wide[phase]["scheduled_weight_scale_bytes"] == narrow[phase]["scheduled_weight_scale_bytes"]
        scales = 2 * wide[phase]["scale_bytes_per_expert"]
        field = "inferred_scale_sector_bytes_without_cross_tile_reuse"
        assert wide[phase][field] == scales
        assert narrow[phase][field] == 2 * scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
@pytest.mark.parametrize("variant", ["two_cta", "two_cta_pipeline3"])
@pytest.mark.parametrize("geometry", [(128, 128), (2048, 512)])
def test_two_cta_graph_promotion_and_live_routes(tmp_path, variant, geometry):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    from tests.moe.test_prepared_expert_cache import _graph_parity, source
    _graph_parity(tmp_path, torch.int32, source(*geometry, 64), 64, 4, 32,
                  ExpertCacheConfig(cold_prefill=variant))
