"""Sanitizer completion cannot be inferred from a quiet or partial log."""

from scripts.qualify_hybrid_sanitizer import classify, diagnostic_inventory
import json
import pytest


def test_requires_completion_from_every_rank_and_zero_summaries():
    assert (
        classify(0, False, {0: [0], 1: [0]}, {0, 1}, 2, False) == "whole_program_pass"
    )
    assert classify(0, False, {0: [0], 1: [0]}, {0}, 2, False) == "incomplete"
    assert classify(0, False, {}, {0, 1}, 2, False) == "incomplete"
    assert classify(0, False, {0: [0, 0]}, {0, 1}, 2, False) == "incomplete"
    assert classify(99, False, {0: [120], 1: [120]}, {0, 1}, 2, False) == "failed"
    assert classify(124, True, {0: [0]}, {0}, 2, False) == "timeout"


def test_filter_never_qualifies_whole_program():
    assert classify(0, False, {0: [0], 1: [0]}, {0, 1}, 2, True) == "component_pass"


def test_completed_sanitizer_error_flushes_peers_but_failed_application_retires(tmp_path):
    from scripts.qualify_hybrid_sanitizer import retired_incomplete_rank

    (tmp_path / "rank-0-exit.json").write_text(json.dumps({"returncode": 99}))
    assert retired_incomplete_rank(tmp_path, 3) == dict(rank=0, returncode=99)
    (tmp_path / "rank-0-complete.json").write_text("{}")
    assert retired_incomplete_rank(tmp_path, 3) is None
    (tmp_path / "rank-2-exit.json").write_text(json.dumps({"returncode": 1}))
    assert retired_incomplete_rank(tmp_path, 3) == dict(rank=2, returncode=1)


def test_checkpoint_diagnostics_do_not_change_kernel_compile_identity(monkeypatch):
    from b12x._lib.compiler import _compile_environment_key

    try:
        monkeypatch.setenv("CHECKPOINT_TEST_PROGRESS", "/first/receipt")
        monkeypatch.setenv("CHECKPOINT_TEST_ORACLE_DEVICE", "cuda")
        _compile_environment_key.cache_clear()
        before = _compile_environment_key()
        monkeypatch.setenv("CHECKPOINT_TEST_PROGRESS", "/second/receipt")
        monkeypatch.setenv("CHECKPOINT_TEST_ORACLE_DEVICE", "cpu")
        _compile_environment_key.cache_clear()
        assert _compile_environment_key() == before
    finally:
        _compile_environment_key.cache_clear()


def test_probe_attribution_requires_stack_and_complete_accounting():
    log = """========= COMPUTE-SANITIZER
========= Program hit cudaErrorNoKernelImageForDevice (error 209) due to "no kernel image is available for execution on the device" on CUDA API call to cudaFuncGetAttributes.
=========     Host Frame: ncclInitKernelsForDevice(int, int, unsigned long*) in enqueue.cc:87 in libnccl.so.2.31.2
=========
========= ERROR SUMMARY: 1 error
"""
    assert diagnostic_inventory(log)["initialization_probe_only"]
    assert not diagnostic_inventory(
        log.replace("ncclInitKernelsForDevice", "application")
    )["initialization_probe_only"]
    assert not diagnostic_inventory(log.replace("SUMMARY: 1", "SUMMARY: 2"))[
        "initialization_probe_only"
    ]
    assert not diagnostic_inventory(
        log.replace(
            "========= ERROR SUMMARY",
            "========= Invalid __global__ read of size 4 bytes\n=========\n========= ERROR SUMMARY",
        )
    )["initialization_probe_only"]
    assert classify(99, False, {0: [1]}, {0}, 1, False) == "failed"


@pytest.mark.parametrize(
    "stage,counts", [("cuda", [1]), ("tp-layer", [0]), ("tp-layer", [513])]
)
def test_placement_scope_rejects_invalid_or_unrelated_controls(tmp_path, stage, counts):
    from scripts.qualify_hybrid_sanitizer import main

    output = tmp_path / "unstarted"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--stage",
                stage,
                "--output",
                str(output),
                "--sanitizer",
                "/unavailable",
                "--resident-counts",
                *map(str, counts),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()
def test_distributed_layers_require_fresh_process_groups():
    import pytest

    from scripts.qualify_hybrid_sanitizer import validate_layer_scope

    for layer in (0, 24, 47):
        validate_layer_scope("tp-layer", [layer])
    validate_layer_scope("layer", [0, 24, 47])
    for layers in ([], [0, 24, 47]):
        with pytest.raises(ValueError, match="one layer per fresh distributed job"):
            validate_layer_scope("tp-layer", layers)
