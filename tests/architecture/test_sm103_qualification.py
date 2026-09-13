"""Offline checks for the explicit SM103 execution and evidence boundary."""

import hashlib
import json

import pytest
import torch

from scripts import qualify_sm103 as qualification


def test_preparation_never_creates_a_cuda_context(tmp_path, monkeypatch):
    def refuse():
        pytest.fail("preparing qualification must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    output = tmp_path / "prepared"
    assert qualification.main(["--output-dir", str(output)]) == 0
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "prepared"
    assert receipt["runtime_qualified"] is False
    assert receipt["components"] == list(qualification.SUITES)
    assert all(row["status"] == "prepared" for row in receipt["results"])


def test_execution_requires_an_explicit_physical_gpu(tmp_path):
    with pytest.raises(SystemExit) as error:
        qualification.main(["--execute", "--output-dir", str(tmp_path / "run")])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_non_sm103_execution_is_recorded_as_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="physical SM103"):
        qualification.main(
            ["--execute", "--device-uuid", "GPU-synthetic", "--output-dir", str(output)]
        )
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["runtime_qualified"] is False


def test_compile_artifact_integrity_rejects_tampering(tmp_path):
    directory = tmp_path / "kernel"
    directory.mkdir()
    payload = b"compiled"
    files = {}
    for name in ("kernel.ptx", "kernel.cubin"):
        (directory / name).write_bytes(payload)
        files[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {"callables": 1, "artifacts": [{"name": "kernel", "files": files}]}
    qualification.verify_compile_artifacts(tmp_path / "manifest.json", manifest)
    (directory / "kernel.cubin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="identity mismatch"):
        qualification.verify_compile_artifacts(tmp_path / "manifest.json", manifest)


def test_skipped_tests_remain_visible_in_qualification_counts(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>'
    )
    assert qualification.junit_counts(path) == dict(
        tests=3, failures=0, errors=0, skipped=1
    )
