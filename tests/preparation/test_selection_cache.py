"""Tuning decision versions and source-based compiled artifacts are independent."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import importlib
import json

import pytest
import torch

from b12x._lib import compiler
from b12x._lib.compile_plan import ProgramKey, record_program
from b12x._lib.compile_pool import CompilationPlan, CompileJob
from b12x.preparation import DetectedDevice, PreparationSession, PreparedCall
from b12x.preparation._cache import SelectionCache, cache_identity
from .test_defaults import contract
from .test_session import _deterministic_timer, declaration


@pytest.fixture
def cache_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_args: nullcontext())
    monkeypatch.setattr(compiler, "_device_uuid_key", lambda ordinal: ("device_uuid", f"gpu-{ordinal}"))
    monkeypatch.delenv("B12X_TUNING_CACHE_VERSION", raising=False)


def _save_choice(cache):
    cache.save(
        "shape", assignment={"width": 2}, config={"width": 2},
        coverage={"cartesian_count": 3, "legal_count": 3, "effective_count": 3, "measured_count": 3},
        programs=(ProgramKey("cute", "selected-source-key"),),
    )


def test_manual_version_selects_a_distinct_decision_cache_without_deleting_choices(cache_device, tmp_path, monkeypatch):
    original = SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0))
    _save_choice(original)
    assert original.identity["tuning_cache_version"] == 1
    assert SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0)).get("shape") is not None
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
    retune = SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0))
    assert retune.path != original.path
    assert retune.get("shape") is None
    assert original.path.is_file()
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "1")
    assert SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0)).get("shape") is not None


@pytest.mark.parametrize("version", ["0", "-1", "", "1.5", "invalid"])
def test_invalid_tuning_version_fails_before_cache_access(cache_device, monkeypatch, version):
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", version)
    with pytest.raises(ValueError, match="B12X_TUNING_CACHE_VERSION must be a positive integer"):
        cache_identity({}, 0)


def test_decisions_remain_device_and_model_specific(cache_device):
    assert cache_identity({"model": "a"}, 0) != cache_identity({"model": "b"}, 0)
    assert cache_identity({"model": "a"}, 0) != cache_identity({"model": "a"}, 1)


def test_source_and_toolchain_changes_rekey_kernels_without_rekeying_decisions(cache_device, tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(compiler, "_b12x_package_fingerprint", compiler._compute_b12x_package_fingerprint)
    monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("compiler-a",))
    compile_context = lambda: compiler._static_compile_cache_context.__wrapped__(object())
    compiler._compile_environment_key.cache_clear()
    try:
        decision = cache_identity({}, 0)
        kernel = compile_context()
        source.write_text("VALUE = 2\n")
        changed_source = compile_context()
        assert changed_source != kernel
        assert cache_identity({}, 0) == decision
        monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("compiler-b",))
        changed_toolchain = compile_context()
        assert changed_toolchain != changed_source
        assert cache_identity({}, 0) == decision
        monkeypatch.setenv("NVCC_APPEND_FLAGS", "-lineinfo")
        compiler._compile_environment_key.cache_clear()
        assert compile_context() != changed_toolchain
        assert cache_identity({}, 0) == decision
    finally:
        compiler._compile_environment_key.cache_clear()


def test_manual_tuning_version_does_not_change_kernel_compilation_context(cache_device, monkeypatch):
    monkeypatch.setattr(compiler, "_b12x_package_fingerprint", lambda: "same-kernel-source")
    monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("same-compiler",))
    compiler._compile_environment_key.cache_clear()
    try:
        before = compiler._static_compile_cache_context.__wrapped__(object())
        decision = cache_identity({}, 0)
        monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
        compiler._compile_environment_key.cache_clear()
        assert compiler._static_compile_cache_context.__wrapped__(object()) == before
        assert cache_identity({}, 0) != decision
    finally:
        compiler._compile_environment_key.cache_clear()


def test_matching_malformed_decision_still_fails_closed(cache_device, tmp_path):
    identity = cache_identity({}, 0)
    cache = SelectionCache(tmp_path, identity)
    _save_choice(cache)
    payload = json.loads(cache.path.read_text())
    payload["records"]["shape"]["coverage"]["measured_count"] = 1
    cache.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only completed exhaustive"):
        SelectionCache(tmp_path, identity)


def test_cached_choice_rebuilds_changed_program_without_racing(cache_device, tmp_path, monkeypatch):
    session_module = importlib.import_module("b12x.preparation.session")
    _deterministic_timer(monkeypatch)
    available, built = set(), []
    source_version = "a"

    def describe(job):
        return CompilationPlan(job, (ProgramKey("cute", f"{source_version}-{job.args[0]}"),))

    def build(plans):
        for plan in plans:
            for program in plan.programs:
                built.append(program.key)
                available.add(program)

    monkeypatch.setattr(session_module, "describe_compilation", describe)
    monkeypatch.setattr(session_module, "compiled_program_available", lambda program: program in available)
    monkeypatch.setattr(session_module, "compile_in_process", build)

    def prepare():
        def materialize(selection, device):
            program = ProgramKey("cute", f"{source_version}-{selection.config.width}")
            return SimpleNamespace(value=selection.config.width * 3, program=program)

        def factory(state):
            def run():
                assert state.program in available
                record_program(state.program)
                return state.value
            return PreparedCall(run=run, produce=lambda: None)

        plan = replace(
            declaration(tuning=contract(default=2)),
            _compile_jobs=lambda config, device: (CompileJob.create("test.compiler:compile", config.width),),
            _materialize=materialize,
        )
        with PreparationSession(device=DetectedDevice(None, None), compile_workers=0) as session:
            session._cache = SelectionCache(tmp_path, cache_identity({}, 0))
            result = session.prepare((plan.request(name="query", prepare_call=factory, benchmark_call=factory),))
            return result.selections["query"].source, result.benchmarked_candidates

    assert prepare() == ("tuned", 3)
    assert set(built) == {"a-1", "a-2", "a-4"}
    source_version = "b"
    built.clear()
    assert prepare() == ("cached", 0)
    assert built == ["b-2"]
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
    built.clear()
    assert prepare() == ("tuned", 3)
    assert set(built) == {"b-1", "b-4"}
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "3")
    built.clear()
    assert prepare() == ("tuned", 3)
    assert built == []
