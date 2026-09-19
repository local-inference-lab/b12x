"""Host checks for reproducible routing and honest observed cold fractions."""

import pytest

from benchmarks.moe.sm120_residency_spectrum import policy_routes, routes


@pytest.mark.parametrize(
    "live,topk,cold",
    [(1, 10, 0), (1, 10, 1), (4, 6, 7), (128, 10, 20), (128, 10, 1280)],
)
def test_route_fixture_has_exact_cold_count_and_unique_topk(live, topk, cold):
    ids = routes(512, 256, live, topk, cold)
    assert ids.shape == (live, topk)
    assert int((ids >= 256).sum()) == cold
    assert all(len(set(row)) == topk for row in ids.tolist())
    assert int(ids.min()) >= 0 and int(ids.max()) < 512
    assert ids.equal(routes(512, 256, live, topk, cold))


def test_policy_patterns_include_reuse_shift_and_wasted_promotions():
    def fixture(pattern, epoch):
        return policy_routes(512, 256, 8, 10, pattern, epoch, 8)

    assert (fixture("steady_hot", 0) < 256).all()
    assert fixture("steady_cold", 0).equal(fixture("steady_cold", 7))
    assert (fixture("phase_shift", 3) < 256).all()
    assert (fixture("phase_shift", 4) >= 256).all()
    assert set(fixture("rotating_cold", 0).flatten().tolist()).isdisjoint(
        fixture("rotating_cold", 1).flatten().tolist()
    )


def test_route_fixture_rejects_impossible_distinct_topk():
    with pytest.raises(ValueError, match="at least top-k"):
        routes(16, 8, 1, 10, 1)
