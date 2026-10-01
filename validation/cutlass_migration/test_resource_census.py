"""Offline integrity checks for cross-compiler artifact comparisons."""

import copy

import pytest

from validation.cutlass_migration.resource_census import (
    _embedded_cuda_elf,
    comparison_key,
)


def test_comparison_normalizes_only_package_runtime_and_disabled_pyir():
    baseline = {
        "compile_options": ["opt-level=3"],
        "compile_environment": [["CUTE_DSL_LIBS", "/venv/nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"]],
        "compile_kwargs": {"__dsl_compile_options_key": ["opt-level=3"]},
        "compile_kwargs_hash": "baseline-hash",
        "compile_spec": {"capacity": 128},
    }
    candidate = copy.deepcopy(baseline)
    candidate["compile_environment"] = []
    candidate["compile_options"].append("enable-pyir=false")
    candidate["compile_kwargs"]["__dsl_compile_options_key"].append("enable-pyir=false")
    candidate["compile_kwargs_hash"] = "candidate-hash"
    snapshot = copy.deepcopy(candidate)
    assert comparison_key(candidate) == comparison_key(baseline)
    assert candidate == snapshot

    candidate["compile_options"][-1] = "enable-pyir=true"
    assert comparison_key(candidate) != comparison_key(baseline)
    candidate = copy.deepcopy(snapshot)
    candidate["compile_spec"]["capacity"] = 256
    assert comparison_key(candidate) != comparison_key(baseline)
    candidate = copy.deepcopy(snapshot)
    candidate["compile_environment"] = [["CUTE_DSL_LIBS", "/custom/libcute_dsl_runtime.so"]]
    assert comparison_key(candidate) != comparison_key(baseline)


@pytest.mark.parametrize("payload", [b"", b"\x7fELF\x02\x01\x01\x41", b"\x7fELF\x02\x01\x01\x41" * 2])
def test_cuda_elf_extraction_rejects_missing_truncated_or_ambiguous_objects(payload):
    with pytest.raises(ValueError):
        _embedded_cuda_elf(payload)
