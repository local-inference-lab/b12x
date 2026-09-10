"""Contract checks for the published cold-prefill evidence clients."""

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def client(name, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "docs/evidence/nvfp4-immutable-input"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module(name)


def test_cache_observation_is_required(monkeypatch):
    module = client("measure_token_id_prefill", monkeypatch)
    with pytest.raises(RuntimeError, match="cache-hit evidence"):
        module.check_uncached({}, {}, {})
    assert module.check_uncached({}, {}, {
        "prompt_tokens_details": {"cached_tokens": 0}}) == 0
    metric = "vllm:prefix_cache_hits_total"
    assert module.check_uncached({metric: 13}, {metric: 13}, {}) == 0
    with pytest.raises(RuntimeError, match="cached tokens"):
        module.check_uncached({metric: 13}, {metric: 14}, {})


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_prefill_rejects_invalid_duration(duration, monkeypatch, tmp_path):
    module = client("measure_exact_cold_prefill", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["measure", "--base-url", "http://unused",
        "--tokens", "32768", "--duration", duration, "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 2


def test_slow_warmup_receipt_cannot_elide_all_measured_samples(monkeypatch, tmp_path):
    module = client("measure_exact_cold_prefill", monkeypatch)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["measure", "--base-url", "http://unused",
        "--tokens", "32768", "--duration", "30", "--output", str(output)])
    ticks = iter([0.0, 10.0, 10.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: next(ticks), time_ns=lambda: 123))
    counts = iter([0, 32768, 32768, 65536])
    monkeypatch.setattr(module, "metrics", lambda url: {
        "local_compute": next(counts), "local_cache_hit": 0, "external_kv_transfer": 0})

    class Transport:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return self

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield "data: " + json.dumps({"usage": {
                "prompt_tokens": 32768, "completion_tokens": 1},
                "choices": [{"text": "token"}]})
            yield "data: [DONE]"

    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: Transport())
    module.main()
    record = json.loads(output.read_text())
    assert record["status"] == "qualified" and len(record["samples"]) == 1
    assert record["warmup"]["ttft_seconds"] == 10.0
    assert record["samples"][0]["ttft_seconds"] == 1.0
