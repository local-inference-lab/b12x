"""Phase timing never attributes mixed execution or sums TP rank durations."""

from copy import deepcopy

import pytest

from benchmarks.moe.summarize_hybrid_prefill import summarize


def fixture():
    rows = [
        dict(
            requests=["cache-0"],
            scheduled=[64],
            computed=[n],
            prompt_lengths=[128],
            phase="prefill",
            prompt_tokens=64,
            prompt_chunks=[64],
            model_device_ms=2,
            model_start_ms=t,
            model_end_ms=t + 2,
        )
        for n, t in ((0, 0), (64, 5))
    ]
    rank = dict(iterations=rows, requests={"cache-0": dict(complete=True)})
    return [
        dict(
            kind="configuration",
            arguments=dict(concurrency=1, tp_size=2, admission="together"),
        ),
        dict(kind="request", index=0, ttft_ns=12_000_000),
        dict(kind="phase_timing", result=[rank, deepcopy(rank)]),
        dict(kind="complete"),
    ]


def test_rank_time_and_delivery_are_distinct():
    result = summarize(fixture())["groups"][0]
    assert result["pure_model_ms_per_rank"] == [4, 4]
    assert result["pure_model_tokens_per_s"] == 32000
    assert result["prompt_span_ms_per_rank"] == [7, 7]
    assert result["ttft_ms"] == [12]


def test_mixed_iteration_is_not_apportioned():
    records = fixture()
    for rank in records[2]["result"]:
        rank["iterations"][1]["phase"] = "mixed"
    result = summarize(records)["groups"][0]
    assert result["pure_prefill_tokens"] == 64
    assert result["pure_model_ms_per_rank"] == [2, 2]
    assert result["mixed_device_ms_per_rank"] == [2, 2]
    assert result["prompt_span_tokens_per_s"] is None


def test_tp_metadata_disagreement_rejected():
    records = fixture()
    records[2]["result"][1]["iterations"][0]["scheduled"] = [32]
    with pytest.raises(ValueError, match="metadata differs"):
        summarize(records)
