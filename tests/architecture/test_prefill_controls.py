"""Chunking controls preserve the default and reject unplanned capacity."""

import sys

import pytest


@pytest.mark.parametrize("chunk", [0, 64, -1, 257])
def test_explicit_prefill_chunk_within_prepared_capacity(tmp_path, monkeypatch, chunk):
    from benchmarks.moe import expert_cache_serving as harness

    observed = []

    async def run(args):
        observed.append(args.prefill_chunk_tokens)

    monkeypatch.setattr(harness, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serving",
            "--model",
            "checkpoint",
            "--mode",
            "static",
            "--profile",
            "profile.json",
            "--prompts",
            "prompts.jsonl",
            "--output",
            str(tmp_path / "unstarted.jsonl"),
            "--capacity",
            "256",
            "--prefill-chunk-tokens",
            str(chunk),
        ],
    )
    if 0 <= chunk <= 256:
        harness.main()
        assert observed == [chunk]
    else:
        with pytest.raises(SystemExit) as error:
            harness.main()
        assert error.value.code == 2
        assert not observed
