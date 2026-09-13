"""Offline checks for the explicit SM103 execution and evidence boundary."""

import hashlib
import json
import shutil
import subprocess
import sys

import pytest
import torch

from scripts import qualify_sm103 as qualification
from scripts._sm103_source import package_source_sha256, source_identity


def test_source_archive_prepares_outside_its_directory(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "b12x").mkdir()
    (source / "b12x" / "__init__.py").write_text("# archive package\n")
    for name in ("qualify_sm103.py", "_sm103_source.py"):
        shutil.copyfile(qualification.ROOT / "scripts" / name, source / "scripts" / name)
    output = tmp_path / "prepared"
    subprocess.run(
        [sys.executable, str(source / "scripts/qualify_sm103.py"),
         "--output-dir", str(output)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "prepared"
    assert receipt["source_revision"] is None
    assert receipt["git_status"] is None
    assert receipt["source_sha256"] == package_source_sha256(source)
    assert receipt["worktree"] == str(source)


def test_exported_revision_does_not_inherit_enclosing_checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=Archive Test",
         "-c", "user.email=archive@example.invalid", "commit", "-q", "--allow-empty",
         "-m", "Create enclosing checkout"], check=True,
    )
    source = tmp_path / "export"
    source.mkdir()
    identity = source_identity(source)
    assert identity["source_revision"] is None
    assert identity["git_status"] is None
    archival = source / ".git_archival.txt"
    archival.write_text("$Format:%H$\n")
    assert source_identity(source)["source_revision"] is None
    archival.write_text("a" * 40 + "\n")
    identity = source_identity(source)
    assert identity["source_revision"] == "a" * 40
    assert identity["source_kind"] == "archive"
    assert identity["git_status"] is None


def test_checkout_identity_uses_explicit_source_root(tmp_path, monkeypatch):
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=qualification.ROOT, text=True
    ).strip()
    monkeypatch.chdir(tmp_path)
    identity = source_identity(qualification.ROOT)
    assert identity["source_revision"] == expected
    assert identity["source_kind"] == "git"
    assert isinstance(identity["git_status"], list)


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
