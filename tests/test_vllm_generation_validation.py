"""Generation receipts reject changed checkpoints, missing requests and tokens."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.validate_vllm_generation import compare_runs, install_replay_counter


def test_replay_inspection_preserves_results_and_counts_only_successes():
    output = object()

    def replay(desc):
        if desc == "failed":
            raise RuntimeError("replay failed")
        return output

    manager = SimpleNamespace(run_fullgraph=replay)
    install_replay_counter(manager)
    assert manager.run_fullgraph("batch_4") is output
    install_replay_counter(manager)
    assert manager.run_fullgraph("batch_4") is output
    with pytest.raises(RuntimeError, match="replay failed"):
        manager.run_fullgraph("failed")
    assert dict(manager._generation_validation_replays) == {"batch_4": 2}


@pytest.fixture
def receipt():
    return {
        "status": "passed",
        "target_checkpoint": {
            "path": "/models/target",
            "files": {"weights.safetensors": {"sha256": "a", "bytes": 8}},
        },
        "prompts": ["What is 17 plus 25?"],
        "sampling": {"temperature": 0, "max_tokens": 8},
        "counts": [1],
        "chat_template_kwargs": {"enable_thinking": False},
        "runs": [
            [{"prompt_token_ids": [1, 2], "token_ids": [42], "finish_reason": "stop"}]
        ],
    }


def test_receipts_can_compare_identical_checkpoints_at_different_paths(receipt):
    candidate = deepcopy(receipt)
    candidate["target_checkpoint"]["path"] = "/another/host/target"
    assert compare_runs(receipt, candidate) == []


def test_receipt_rejects_changed_weight_bytes(receipt):
    candidate = deepcopy(receipt)
    candidate["target_checkpoint"]["files"]["weights.safetensors"]["sha256"] = "b"
    with pytest.raises(ValueError, match="target_checkpoint"):
        compare_runs(receipt, candidate)


@pytest.mark.parametrize("field", ["prompt_token_ids", "token_ids", "finish_reason"])
def test_receipt_reports_exact_token_or_termination_mismatch(receipt, field):
    candidate = deepcopy(receipt)
    candidate["runs"][0][0][field] = "length" if field == "finish_reason" else [99]
    assert compare_runs(receipt, candidate) == [
        {
            "run": 0,
            "request": 0,
            "field": field,
            "expected": receipt["runs"][0][0][field],
            "actual": candidate["runs"][0][0][field],
        }
    ]


@pytest.mark.parametrize("missing_run", [False, True])
def test_receipt_rejects_incomplete_generation(receipt, missing_run):
    candidate = deepcopy(receipt)
    if missing_run:
        candidate["runs"].clear()
    else:
        candidate["runs"][0].clear()
    with pytest.raises(ValueError):
        compare_runs(receipt, candidate)


def test_receipt_rejects_failed_reference(receipt):
    candidate = deepcopy(receipt)
    receipt["status"] = "failed"
    with pytest.raises(ValueError, match="did not pass"):
        compare_runs(receipt, candidate)
