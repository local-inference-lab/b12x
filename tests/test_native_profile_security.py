"""Filesystem security tests for native profile capture helpers (issue #175).

Verifies that default capture directories are private and unpredictable,
that explicit output directories are validated against pre-existing entries
and unsafe parents, that all sensitive file writes use exclusive, no-follow,
mode-0600 semantics relative to a pinned directory fd, and that post-creation
path swaps cannot redirect writes.  No live inference server is required;
HTTP operations are stubbed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
VLLM_SCRIPT = SCRIPTS_DIR / "capture_vllm_native_profile.py"
SGLANG_SCRIPT = SCRIPTS_DIR / "capture_sglang_native_profile.py"
SHELL_SCRIPT = SCRIPTS_DIR / "capture_sglang_native_profile.sh"


def _load_script(path: Path):
    """Load a standalone script as an importable module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vllm():
    return _load_script(VLLM_SCRIPT)


@pytest.fixture
def sglang():
    return _load_script(SGLANG_SCRIPT)


@pytest.fixture(params=["vllm", "sglang"])
def helper(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def umask_000(tmp_path):
    """Run under permissive umask 000 to verify modes are explicit, not umask-dependent.

    Depends on tmp_path so that directory creation happens before umask is set.
    """
    old = os.umask(0)
    yield
    os.umask(old)


@pytest.fixture
def umask_0277(tmp_path):
    """Run under restrictive umask 0277 to verify fchmod overrides umask.

    Depends on tmp_path so that directory creation happens before umask is set.
    """
    old = os.umask(0o277)
    yield
    os.umask(old)


# ---------------------------------------------------------------------------
# Unit: _secure_write_text / _secure_open (basename + dir_fd interface)
# ---------------------------------------------------------------------------


class TestSecureWriteText:
    def test_creates_file_with_mode_0600(self, helper, tmp_path, umask_000):
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            helper._secure_write_text("secret.json", fd, '{"prompt": "SECRET"}\n')
            assert (path / "secret.json").read_text() == '{"prompt": "SECRET"}\n'
            assert stat.S_IMODE((path / "secret.json").stat().st_mode) == 0o600
        finally:
            os.close(fd)

    def test_creates_file_with_mode_0600_restrictive_umask(self, helper, tmp_path, umask_0277):
        """Even under restrictive umask 0277, fchmod forces exact 0600."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            helper._secure_write_text("secret.json", fd, '{"prompt": "SECRET"}\n')
            assert (path / "secret.json").read_text() == '{"prompt": "SECRET"}\n'
            assert stat.S_IMODE((path / "secret.json").stat().st_mode) == 0o600
        finally:
            os.close(fd)

    def test_rejects_preexisting_regular_file(self, helper, tmp_path):
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            planted = path / "planted.json"
            planted.write_text("attacker data")
            with pytest.raises(RuntimeError, match="refusing to overwrite"):
                helper._secure_write_text("planted.json", fd, "victim data")
            assert planted.read_text() == "attacker data"
        finally:
            os.close(fd)

    def test_rejects_symlink_to_file(self, helper, tmp_path):
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            target = tmp_path / "target.txt"
            target.write_text("original secret")
            link = path / "request.json"
            link.symlink_to(target)
            with pytest.raises(RuntimeError, match="refusing to overwrite"):
                helper._secure_write_text("request.json", fd, "victim data")
            assert target.read_text() == "original secret"
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Unit: _prepare_output_dir
# ---------------------------------------------------------------------------


class TestPrepareOutputDir:
    def test_default_dir_is_unpredictable(self, helper, tmp_path, monkeypatch):
        """Two default calls must produce distinct, non-timestamp-predictable directories."""
        monkeypatch.setattr(helper.tempfile, "tempdir", str(tmp_path))
        d1, fd1 = helper._prepare_output_dir(None, "test-prefix-")
        d2, fd2 = helper._prepare_output_dir(None, "test-prefix-")
        try:
            assert d1 != d2
            assert d1.name.startswith("test-prefix-")
            assert d2.name.startswith("test-prefix-")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            assert d1.name != f"test-prefix-{timestamp}"
            assert d2.name != f"test-prefix-{timestamp}"
        finally:
            os.close(fd1)
            os.close(fd2)

    def test_default_dir_mode_0700(self, helper, tmp_path, monkeypatch, umask_000):
        monkeypatch.setattr(helper.tempfile, "tempdir", str(tmp_path))
        d, fd = helper._prepare_output_dir(None, "test-prefix-")
        try:
            assert stat.S_IMODE(d.stat().st_mode) == 0o700
        finally:
            os.close(fd)

    def test_default_dir_mode_0700_restrictive_umask(self, helper, tmp_path, monkeypatch, umask_0277):
        """Even under restrictive umask 0277, fchmod forces exact 0700."""
        monkeypatch.setattr(helper.tempfile, "tempdir", str(tmp_path))
        d, fd = helper._prepare_output_dir(None, "test-prefix-")
        try:
            assert stat.S_IMODE(d.stat().st_mode) == 0o700
        finally:
            os.close(fd)

    def test_explicit_dir_creates_mode_0700(self, helper, tmp_path, umask_000):
        new_dir = tmp_path / "new_output"
        result, fd = helper._prepare_output_dir(str(new_dir), "test-")
        try:
            assert result == new_dir
            assert stat.S_IMODE(result.stat().st_mode) == 0o700
        finally:
            os.close(fd)

    def test_explicit_dir_creates_mode_0700_restrictive_umask(self, helper, tmp_path, umask_0277):
        """Even under restrictive umask 0277, fchmod forces exact 0700."""
        new_dir = tmp_path / "new_output"
        result, fd = helper._prepare_output_dir(str(new_dir), "test-")
        try:
            assert result == new_dir
            assert stat.S_IMODE(result.stat().st_mode) == 0o700
        finally:
            os.close(fd)

    def test_explicit_dir_rejects_existing_directory(self, helper, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        with pytest.raises(RuntimeError, match="refusing to use existing"):
            helper._prepare_output_dir(str(existing), "test-")

    def test_explicit_dir_rejects_existing_file(self, helper, tmp_path):
        existing = tmp_path / "existing.txt"
        existing.write_text("data")
        with pytest.raises(RuntimeError, match="refusing to use existing"):
            helper._prepare_output_dir(str(existing), "test-")

    def test_explicit_dir_rejects_symlink(self, helper, tmp_path):
        target = tmp_path / "target_dir"
        target.mkdir()
        link = tmp_path / "link_dir"
        link.symlink_to(target)
        with pytest.raises(RuntimeError, match="refusing to use existing"):
            helper._prepare_output_dir(str(link), "test-")

    def test_explicit_dir_rejects_missing_parent(self, helper, tmp_path):
        """Parent directory must already exist; no implicit creation."""
        leaf = tmp_path / "nonexistent_parent" / "output"
        with pytest.raises(RuntimeError, match="parent directory does not exist"):
            helper._prepare_output_dir(str(leaf), "test-")

    def test_explicit_dir_rejects_parent_symlink(self, helper, tmp_path):
        """A parent that is a symlink must be rejected (O_NOFOLLOW)."""
        real_parent = tmp_path / "real_parent"
        real_parent.mkdir()
        link_parent = tmp_path / "link_parent"
        link_parent.symlink_to(real_parent)
        leaf = link_parent / "output"
        with pytest.raises(RuntimeError, match="refusing to follow symlink"):
            helper._prepare_output_dir(str(leaf), "test-")

    def test_explicit_dir_rejects_group_writable_parent(self, helper, tmp_path):
        """A group-writable parent is unsafe and must be rejected."""
        parent = tmp_path / "parent"
        parent.mkdir()
        os.chmod(parent, 0o770)
        leaf = parent / "output"
        try:
            with pytest.raises(RuntimeError, match="group or world writable"):
                helper._prepare_output_dir(str(leaf), "test-")
        finally:
            os.chmod(parent, 0o755)

    def test_explicit_dir_rejects_world_writable_parent(self, helper, tmp_path):
        """A world-writable parent is unsafe and must be rejected."""
        parent = tmp_path / "parent"
        parent.mkdir()
        os.chmod(parent, 0o777)
        leaf = parent / "output"
        try:
            with pytest.raises(RuntimeError, match="group or world writable"):
                helper._prepare_output_dir(str(leaf), "test-")
        finally:
            os.chmod(parent, 0o755)


# ---------------------------------------------------------------------------
# Post-create path-swap: writes must stay on pinned inode
# ---------------------------------------------------------------------------


class TestPathSwap:
    def test_post_create_path_swap_writes_stay_on_pinned_inode(self, helper, tmp_path, umask_000):
        """After directory creation, swapping the path with a symlink must not redirect writes.

        Proves fd identity: the pinned fd's inode matches the original
        directory's inode, not the attacker's replacement.
        """
        out_dir = tmp_path / "real_output"
        path, fd = helper._prepare_output_dir(str(out_dir), "test-")
        try:
            original_inode = os.fstat(fd).st_ino

            # Attacker swaps: rename real dir away, create symlink to attacker dir.
            attacker_dir = tmp_path / "attacker"
            attacker_dir.mkdir()
            saved_dir = tmp_path / "saved_output"
            os.rename(str(path), str(saved_dir))
            os.symlink(str(attacker_dir), str(path))

            # Pinned fd still refers to the original inode.
            assert os.fstat(fd).st_ino == original_inode

            # Write via pinned fd — must land in original inode, not symlink target.
            helper._secure_write_text("secret.json", fd, "SECRET_175")

            # File must NOT appear in attacker directory.
            assert not (attacker_dir / "secret.json").exists()

            # File MUST appear in the saved (original) directory.
            assert (saved_dir / "secret.json").exists()

            # File MUST be readable via pinned fd (explicit O_RDONLY | O_NOFOLLOW).
            read_fd = os.open("secret.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                content = os.read(read_fd, 1024).decode("utf-8")
            finally:
                os.close(read_fd)
            assert content == "SECRET_175"
        finally:
            os.close(fd)

    def test_intermediate_symlink_retarget_after_create_writes_stay_real(self, helper, tmp_path, umask_000):
        """An attacker retargeting an intermediate symlink after creation must not redirect writes.

        Setup:
          real/sub/  — genuine parent chain (mode 0700)
          link → real — attacker-controllable symlink in a trusted parent
          Output is created via link/sub/out (pinned fd targets real/sub/out).
          Attacker then repoints link → attacker (which also has sub/).

        Writes via the pinned fd must land under real/, not under attacker/.
        """
        real_parent = tmp_path / "real"
        real_sub = real_parent / "sub"
        real_sub.mkdir(parents=True, mode=0o700)

        # Symlink link → real in the trusted tmp_path parent.
        link = tmp_path / "link"
        link.symlink_to(str(real_parent))

        # Create output dir via the symlinked path: link/sub/out.
        out_arg = link / "sub" / "out"
        path, fd = helper._prepare_output_dir(str(out_arg), "test-")
        try:
            original_inode = os.fstat(fd).st_ino

            # Attacker prepares attacker/sub/ and retargets link → attacker.
            attacker_parent = tmp_path / "attacker"
            attacker_sub = attacker_parent / "sub"
            attacker_sub.mkdir(parents=True)
            link.unlink()
            link.symlink_to(str(attacker_parent))

            # Pinned fd still refers to the original inode (real/sub/out).
            assert os.fstat(fd).st_ino == original_inode

            # Write via pinned fd — must land under real/, not attacker/.
            helper._secure_write_text("secret.json", fd, "SECRET_175")

            # Artifact must exist only under the real path.
            assert (real_sub / "out" / "secret.json").exists()
            assert (real_sub / "out" / "secret.json").read_text() == "SECRET_175"

            # Artifact must NOT exist under the attacker's retargeted path.
            assert not (attacker_sub / "out" / "secret.json").exists()
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# StreamRequest: symlink rejection + positive mode-0600 tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal bounded-connection response returning configured stream lines."""

    def __init__(self, lines=None):
        self._lines = lines or []
        self.status = 200
        self.length = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def readline(self, _size=-1):
        if self._lines:
            return self._lines.pop(0)
        return b""

    def close(self):
        pass


class _FakeSocket:
    def settimeout(self, _timeout):
        pass

    def shutdown(self, _how):
        pass

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, outcome):
        self.sock = _FakeSocket()
        self._outcome = outcome

    def request(self, *_args, **_kwargs):
        pass

    def getresponse(self):
        if callable(self._outcome):
            return self._outcome()
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    def close(self):
        pass


def _stub_stream_connection(helper, monkeypatch, outcome):
    monkeypatch.setattr(
        helper,
        "_create_connection",
        lambda *_args, **_kwargs: _FakeConnection(outcome),
    )


class TestStreamRequestSymlinkRejection:
    def test_stream_log_rejects_symlink(self, helper, tmp_path, monkeypatch):
        """A symlinked stream.log must not be followed."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            target = tmp_path / "stream_target.txt"
            target.write_text("original stream data")
            log_link = path / "stream.log"
            log_link.symlink_to(target)

            _stub_stream_connection(helper, monkeypatch, _FakeResponse())

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=1,
            )
            stream.start()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            assert target.read_text() == "original stream data"
            assert stream.error is not None
            assert "refusing to overwrite" in stream.error
        finally:
            os.close(fd)

    def test_error_log_rejects_symlink(self, helper, tmp_path, monkeypatch):
        """A symlinked error log must not be followed; failure is caught and reported."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            target = tmp_path / "error_target.txt"
            target.write_text("original error data")
            error_link = path / "request.error.txt"
            error_link.symlink_to(target)

            _stub_stream_connection(
                helper, monkeypatch, ConnectionError("simulated failure")
            )

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=1,
            )
            stream.start()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            assert target.read_text() == "original error data"
            assert stream.error is not None
            assert "ConnectionError" in stream.error
            assert "failed to write error log" in stream.error
        finally:
            os.close(fd)


class TestStreamRequestPositive:
    def test_stream_log_created_mode_0600(self, helper, tmp_path, monkeypatch):
        """A successful stream writes stream.log with mode 0600."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            lines = [b"data: {\"token\": \"hello\"}\n", b""]
            _stub_stream_connection(helper, monkeypatch, _FakeResponse(lines))

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=1,
            )
            stream.start()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            log = path / "stream.log"
            assert log.exists()
            assert stat.S_IMODE(log.stat().st_mode) == 0o600
            assert "hello" in log.read_text()
            assert stream.error is None
        finally:
            os.close(fd)

    def test_error_log_created_mode_0600(self, helper, tmp_path, monkeypatch):
        """A failed stream writes error log with mode 0600."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            _stub_stream_connection(
                helper, monkeypatch, ConnectionError("simulated failure")
            )

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=1,
            )
            stream.start()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            err = path / "request.error.txt"
            assert err.exists()
            assert stat.S_IMODE(err.stat().st_mode) == 0o600
            assert "ConnectionError" in err.read_text()
            assert stream.error is not None
            assert "ConnectionError" in stream.error
        finally:
            os.close(fd)


class TestStreamRequestFdLifecycle:
    """Verify StreamRequest owns an independent dup'd dir_fd that survives
    caller fd closure, so log/error writes never see EBADF or a reused fd."""

    def test_log_lands_after_caller_closes_fd(self, helper, tmp_path, monkeypatch):
        """Caller closes its dir_fd before the stream finishes; log still lands in pinned dir."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            # Use a response that yields one line then blocks.
            lines = [b"data: {\"token\": \"hello\"}\n"]
            release = threading.Event()

            class _DelayedResponse:
                status = 200
                length = 0

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def readline(self, _size=-1):
                    if lines:
                        return lines.pop(0)
                    release.wait(timeout=5.0)
                    return b""

                def close(self):
                    pass

            _stub_stream_connection(helper, monkeypatch, _DelayedResponse())

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=5,
            )
            stream.start()

            # Wait for first token (first line consumed), then close caller fd
            # while the stream thread is still blocked in readline.
            assert stream.wait_for_first_token(5.0)

            # Close the caller's fd — stream must have its own dup'd fd.
            os.close(fd)
            fd = -1  # Mark as closed so finally doesn't double-close.

            # Release the stream so it can finish writing.
            release.set()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            # Log must have landed in the pinned directory despite caller fd close.
            log = path / "stream.log"
            assert log.exists()
            assert stat.S_IMODE(log.stat().st_mode) == 0o600
            assert "hello" in log.read_text()
            assert stream.error is None
        finally:
            if fd != -1:
                os.close(fd)

    def test_error_lands_after_caller_closes_fd(self, helper, tmp_path, monkeypatch):
        """Caller closes its dir_fd before the error write; error log still lands."""
        path, fd = helper._prepare_output_dir(str(tmp_path / "out"), "test-")
        try:
            release = threading.Event()

            def delayed_error():
                release.wait(timeout=5.0)
                raise ConnectionError("simulated failure after delay")

            _stub_stream_connection(helper, monkeypatch, delayed_error)

            stream = helper.StreamRequest(
                url="http://fake",
                headers={"Content-Type": "application/json"},
                payload={"model": "test", "prompt": "test"},
                log_basename="stream.log",
                error_basename="request.error.txt",
                dir_fd=fd,
                timeout_s=5,
            )
            stream.start()

            # Close the caller's fd while the stream connection is blocked.
            os.close(fd)
            fd = -1

            # Release so the connection raises and the error handler writes.
            release.set()
            stream.finished_event.wait(timeout=5.0)
            stream.stop()

            # Error log must have landed despite caller fd close.
            err = path / "request.error.txt"
            assert err.exists()
            assert stat.S_IMODE(err.stat().st_mode) == 0o600
            assert "ConnectionError" in err.read_text()
            assert stream.error is not None
            assert "ConnectionError" in stream.error
        finally:
            if fd != -1:
                os.close(fd)


# ---------------------------------------------------------------------------
# vLLM main() integration
# ---------------------------------------------------------------------------


class _FakeStream:
    """Stub StreamRequest that simulates immediate first token and completion."""

    def __init__(self, **kwargs):
        self.first_token_event = threading.Event()
        self.finished_event = threading.Event()
        self.error = None

    def start(self):
        self.first_token_event.set()
        self.finished_event.set()

    def stop(self):
        pass

    def wait_for_first_token(self, timeout_s):
        return True


def _stub_http(module, monkeypatch):
    monkeypatch.setattr(module, "resolve_model", lambda *a, **k: "test-model")
    monkeypatch.setattr(module, "http_json", lambda *a, **k: (200, "{}"))
    monkeypatch.setattr(module, "StreamRequest", _FakeStream)


class TestVllmMainFlow:
    def test_default_capture_files_are_0600(self, vllm, monkeypatch, tmp_path, umask_000):
        _stub_http(vllm, monkeypatch)
        monkeypatch.setattr(vllm.tempfile, "tempdir", str(tmp_path))

        monkeypatch.setattr(sys, "argv", [
            "capture_vllm_native_profile.py",
            "--base-url", "http://127.0.0.1:8000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--capture-seconds", "0.01",
        ])
        vllm.main()

        dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
        for f in out_dir.iterdir():
            assert stat.S_IMODE(f.stat().st_mode) == 0o600
        assert "SECRET_175" in (out_dir / "request.json").read_text()

    def test_precreated_timestamp_candidate_untouched(self, vllm, monkeypatch, tmp_path):
        """An attacker-precreated predictable directory must not be used."""
        _stub_http(vllm, monkeypatch)
        monkeypatch.setattr(vllm.tempfile, "tempdir", str(tmp_path))

        old_pattern = tmp_path / f"vllm-native-decode-{time.strftime('%Y%m%d-%H%M%S')}"
        old_pattern.mkdir(parents=True, exist_ok=True)
        planted_file = old_pattern / "request.json"
        planted_file.write_text("ATTACKER_DATA")

        monkeypatch.setattr(sys, "argv", [
            "capture_vllm_native_profile.py",
            "--base-url", "http://127.0.0.1:8000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--capture-seconds", "0.01",
        ])
        vllm.main()

        assert planted_file.read_text() == "ATTACKER_DATA"
        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d != old_pattern]
        assert len(dirs) == 1

    def test_explicit_safe_dir_works(self, vllm, monkeypatch, tmp_path, umask_000):
        _stub_http(vllm, monkeypatch)
        out_dir = tmp_path / "capture"

        monkeypatch.setattr(sys, "argv", [
            "capture_vllm_native_profile.py",
            "--base-url", "http://127.0.0.1:8000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--capture-seconds", "0.01",
            "--out-dir", str(out_dir),
        ])
        vllm.main()

        assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
        for f in out_dir.iterdir():
            assert stat.S_IMODE(f.stat().st_mode) == 0o600

    def test_explicit_existing_dir_rejected_before_network(self, vllm, monkeypatch, tmp_path):
        """Rejecting an existing directory must happen before any HTTP call."""
        _stub_http(vllm, monkeypatch)
        existing = tmp_path / "existing"
        existing.mkdir()
        planted = existing / "request.json"
        planted.write_text("ATTACKER")

        monkeypatch.setattr(sys, "argv", [
            "capture_vllm_native_profile.py",
            "--base-url", "http://127.0.0.1:8000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--capture-seconds", "0.01",
            "--out-dir", str(existing),
        ])
        with pytest.raises(RuntimeError, match="refusing to use existing"):
            vllm.main()
        assert planted.read_text() == "ATTACKER"


# ---------------------------------------------------------------------------
# SGLang main() integration
# ---------------------------------------------------------------------------


class TestSglangMainFlow:
    def test_default_capture_files_are_0600(self, sglang, monkeypatch, tmp_path, umask_000):
        _stub_http(sglang, monkeypatch)
        monkeypatch.setattr(sglang.tempfile, "tempdir", str(tmp_path))

        monkeypatch.setattr(sys, "argv", [
            "capture_sglang_native_profile.py",
            "--base-url", "http://127.0.0.1:30000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
        ])
        sglang.main()

        dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
        for f in out_dir.iterdir():
            assert stat.S_IMODE(f.stat().st_mode) == 0o600
        assert "SECRET_175" in (out_dir / "request.json").read_text()

    def test_default_server_output_dir_is_omitted(self, sglang, monkeypatch, tmp_path):
        """The client never derives a server pathname from its pinned local directory."""
        _stub_http(sglang, monkeypatch)
        monkeypatch.setattr(sglang.tempfile, "tempdir", str(tmp_path))

        profile_payloads = []

        def capturing_http_json(
            url, *, headers, payload=None, timeout_s=3600, **_bounds
        ):
            if url.endswith("/start_profile"):
                profile_payloads.append(payload)
            return (200, "{}")

        monkeypatch.setattr(sglang, "http_json", capturing_http_json)
        monkeypatch.setattr(sys, "argv", [
            "capture_sglang_native_profile.py",
            "--base-url", "http://127.0.0.1:30000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
        ])
        sglang.main()

        assert len(profile_payloads) == 1
        assert "output_dir" not in profile_payloads[0]

    def test_retargeted_local_ancestor_is_not_sent_to_server(
        self, sglang, monkeypatch, tmp_path
    ):
        """A server cannot re-resolve the client output path after an ancestor swap."""
        _stub_http(sglang, monkeypatch)
        chain = tmp_path / "chain"
        parent = chain / "owned"
        parent.mkdir(parents=True)
        out_dir = parent / "capture"
        original_chain = tmp_path / "original-chain"
        attacker_trace = chain / "owned" / "capture" / "server.trace"

        profile_payloads = []

        def retargeting_http_json(
            url, *, headers, payload=None, timeout_s=3600, **_bounds
        ):
            if url.endswith("/start_profile"):
                os.rename(chain, original_chain)
                (chain / "owned" / "capture").mkdir(parents=True)
                profile_payloads.append(payload)
                if "output_dir" in payload:
                    Path(payload["output_dir"], "server.trace").write_text("trace")
            return (200, "{}")

        monkeypatch.setattr(sglang, "http_json", retargeting_http_json)
        monkeypatch.setattr(sys, "argv", [
            "capture_sglang_native_profile.py",
            "--base-url", "http://127.0.0.1:30000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--out-dir", str(out_dir),
        ])
        sglang.main()

        assert len(profile_payloads) == 1
        assert "output_dir" not in profile_payloads[0]
        assert not attacker_trace.exists()
        assert (original_chain / "owned" / "capture" / "manifest.json").is_file()

    def test_explicit_server_output_dir_preserved(self, sglang, monkeypatch, tmp_path):
        """An explicit --server-output-dir is forwarded as-is."""
        _stub_http(sglang, monkeypatch)
        out_dir = tmp_path / "capture"
        server_dir = "/remote/server/path"

        captured_payloads = []

        def capturing_http_json(
            url, *, headers, payload=None, timeout_s=3600, **_bounds
        ):
            if payload is not None and "output_dir" in payload:
                captured_payloads.append(payload)
            return (200, "{}")

        monkeypatch.setattr(sglang, "http_json", capturing_http_json)

        monkeypatch.setattr(sys, "argv", [
            "capture_sglang_native_profile.py",
            "--base-url", "http://127.0.0.1:30000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
            "--out-dir", str(out_dir),
            "--server-output-dir", server_dir,
        ])
        sglang.main()

        assert captured_payloads[0]["output_dir"] == server_dir
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["server_output_dir"] == server_dir

    def test_precreated_timestamp_candidate_untouched(self, sglang, monkeypatch, tmp_path):
        _stub_http(sglang, monkeypatch)
        monkeypatch.setattr(sglang.tempfile, "tempdir", str(tmp_path))

        old_pattern = tmp_path / f"sglang-native-decode-{time.strftime('%Y%m%d-%H%M%S')}"
        old_pattern.mkdir(parents=True, exist_ok=True)
        planted_file = old_pattern / "request.json"
        planted_file.write_text("ATTACKER_DATA")

        monkeypatch.setattr(sys, "argv", [
            "capture_sglang_native_profile.py",
            "--base-url", "http://127.0.0.1:30000",
            "--model", "test-model",
            "--prompt", "SECRET_175",
        ])
        sglang.main()

        assert planted_file.read_text() == "ATTACKER_DATA"
        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d != old_pattern]
        assert len(dirs) == 1


# ---------------------------------------------------------------------------
# Shell helper behavioral tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_curl_dir(tmp_path):
    """Create a fake curl (Python shim) that records the response-file mode
    via os.stat and returns HTTP 200.  Portable — no BSD stat needed."""
    d = tmp_path / "fake_bin"
    d.mkdir()
    mode_log = tmp_path / "mode_log.txt"
    curl_script = d / "curl"
    curl_script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, stat\n"
        "if '--version' in sys.argv:\n"
        "    print('curl 8.4.0 test-shim')\n"
        "    raise SystemExit(0)\n"
        "output_file = None\n"
        "args = sys.argv[1:]\n"
        'i = 0\n'
        "while i < len(args):\n"
        '    if args[i] == "-o" and i + 1 < len(args):\n'
        "        output_file = args[i + 1]\n"
        "    i += 1\n"
        "if output_file:\n"
        "    st = os.stat(output_file)\n"
        f'    with open({str(mode_log)!r}, "a") as log:\n'
        '        log.write(f"{oct(stat.S_IMODE(st.st_mode))[2:]}\\n")\n'
        "    with open(output_file, \'w\') as f:\n"
        "        f.write(\'{\\\"status\\\":\\\"ok\\\"}\')\n"
        "print(\'200\')\n"
    )
    curl_script.chmod(0o755)
    return d


class TestShellHelperBehavioral:
    @pytest.mark.parametrize("action", ["start", "stop"])
    def test_response_file_mode_600_and_no_residue(
        self, action, tmp_path, fake_curl_dir
    ):
        """The shell helper must create the response file with mode 600 and leave no residue."""
        env = dict(os.environ, TMPDIR=str(tmp_path))
        env["PATH"] = f"{fake_curl_dir}:{env.get('PATH', '')}"

        result = subprocess.run(
            [
                str(SHELL_SCRIPT),
                action,
                "--base-url", "http://127.0.0.1:0",
                "--mode", "decode",
            ],
            capture_output=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0
        assert b'{"status":"ok"}' in result.stdout

        # Verify response file was mode 600.
        mode_log = tmp_path / "mode_log.txt"
        assert mode_log.exists()
        modes = mode_log.read_text().strip().split("\n")
        assert all(m.strip() == "600" for m in modes)
        assert len(modes) >= 1

        # Verify no residue in TMPDIR.
        residue = list(tmp_path.glob("sglang_prof_resp.*"))
        assert residue == []
