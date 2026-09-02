from __future__ import annotations

from b12x.policy.generation.crash_guard import (
    clear_inflight,
    load_crashed,
    mark_inflight,
    promote_inflight,
)


def test_leftover_inflight_marker_becomes_a_skipped_candidate(tmp_path) -> None:
    """A run that died mid-candidate must not race that candidate again."""
    assert promote_inflight(tmp_path) is None
    mark_inflight(tmp_path, "case-a", "cand-1")
    clear_inflight(tmp_path)
    assert promote_inflight(tmp_path) is None
    assert load_crashed(tmp_path) == set()

    mark_inflight(tmp_path, "case-a", "cand-2")
    assert promote_inflight(tmp_path) == ("case-a", "cand-2")
    assert promote_inflight(tmp_path) is None
    mark_inflight(tmp_path, "case-b", "cand-3")
    assert promote_inflight(tmp_path) == ("case-b", "cand-3")
    assert load_crashed(tmp_path) == {("case-a", "cand-2"), ("case-b", "cand-3")}
