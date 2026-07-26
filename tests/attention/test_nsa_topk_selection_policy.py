from __future__ import annotations

import pytest

from sparkinfer.attention.nsa_indexer.tiled_topk import (
    _resolve_selection_policy,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "exact"),
        ("", "exact"),
        ("exact", "exact"),
        (" EXACT ", "exact"),
        ("bounded_compat", "bounded_compat"),
        (" BOUNDED_COMPAT ", "bounded_compat"),
    ],
)
def test_resolve_selection_policy(raw: str | None, expected: str) -> None:
    assert _resolve_selection_policy(raw) == expected


def test_resolve_selection_policy_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match="SPARKINFER_NSA_TOPK_SELECTION_POLICY must be one of",
    ):
        _resolve_selection_policy("legacy")
