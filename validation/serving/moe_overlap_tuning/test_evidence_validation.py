"""Reject invalid observations before they become benchmark summaries.

These CPU-only cases corrupt exported data or construct a preparation trace.
They validate evidence parsing, not serving performance or kernel correctness.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("startup_costs", ROOT / "extract_startup_costs.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("field", ["tok_per_sec", "aggregate_tps", "server_steps_per_s",
                                   "server_spec_accept_length"])
def test_audit_rejects_nonfinite_measurement(tmp_path, field, value):
    results = json.loads((ROOT / "results.json").read_text())
    arm = results["arms"]["baseline"]
    cell = arm["prefill"][0] if field == "tok_per_sec" else arm["decode"][0]["cells"][0]
    cell[field] = value
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    completed = subprocess.run([sys.executable, str(ROOT / "audit_evidence.py"), str(path)],
                               text=True, capture_output=True)
    assert completed.returncode != 0
    assert "ValueError" in completed.stderr


def trace(tmp_path, *, failed=False, measurement=0.5):
    events = [
        {"event": "begin", "rank": 0},
        {"event": "request_begin", "request": "moe", "component": "moe.decode",
         "query": {"num_tokens": 4}},
        {"event": "batch_end", "request": "moe", "seconds": {"autotuning": measurement}},
        {"event": "complete", "rank": 0, "failed": failed, "elapsed_s": 1,
         "seconds": {"autotuning": measurement}},
    ]
    path = tmp_path / "preparation.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    return path


@pytest.mark.parametrize("failed", [True, None])
def test_startup_cost_requires_successful_phase(tmp_path, failed):
    with pytest.raises(ValueError, match="successful"):
        startup.summarize(trace(tmp_path, failed=failed))


def test_successful_phase_preserves_measured_cost(tmp_path):
    result = startup.summarize(trace(tmp_path))
    assert result["phases"][0]["measurement_s"] == 0.5
    assert result["phases"][0]["small_moe_measurement_s"] == 0.5
    assert result["phases"][0]["failed"] is False


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_startup_cost_rejects_nonfinite_counter(tmp_path, value):
    with pytest.raises(ValueError):
        startup.summarize(trace(tmp_path, measurement=value))
