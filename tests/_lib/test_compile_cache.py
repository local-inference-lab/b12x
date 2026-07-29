"""Compile-cache rooting: the highest-risk mechanical edit of the restructure.

Wrong fingerprint root ⇒ silent stale disk-cache hits (running old kernels),
so these assertions are load-bearing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

compiler = importlib.import_module("sparkinfer._lib.compiler")


def test_package_root_is_the_sparkinfer_package():
    root = compiler._PACKAGE_ROOT
    assert root.name == "sparkinfer", root
    assert (root / "_lib" / "compiler.py").is_file()


def test_fingerprint_tracks_source_edits(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("x = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)

    before = compiler._compute_sparkinfer_package_fingerprint()

    (pycache / "kernel.cpython-312.pyc").write_bytes(b"ignored")
    assert compiler._compute_sparkinfer_package_fingerprint() == before, (
        "__pycache__ must not affect the fingerprint"
    )

    (tmp_path / "kernel.py").write_text("x = 2\n")
    after = compiler._compute_sparkinfer_package_fingerprint()
    assert after != before, "editing any source must change the fingerprint"


def test_cache_dir_resolution_order(monkeypatch):
    for name in ("SPARKINFER_COMPILE_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    assert compiler._cute_compile_cache_dir() == (
        Path.home() / ".cache" / "sparkinfer" / "compile"
    )

    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert compiler._cute_compile_cache_dir() == Path("/xdg/sparkinfer/compile")

    monkeypatch.setenv("SPARKINFER_COMPILE_CACHE_DIR", "/explicit")
    assert compiler._cute_compile_cache_dir() == Path("/explicit")


def test_disk_cache_key_includes_device_arch(monkeypatch):
    compile_callable = object()
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable, _ordinal=None: (
            "package",
            "toolchain",
            ("arch", 12, 0, "gpu"),
            (),
            (),
        ),
    )

    payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_cache_key_includes_device_arch,
        (),
        {},
    )

    assert payload[0] == "sparkinfer_cute_compile_cache_v3"
    assert payload[4] == ("arch", 12, 0, "gpu")


def test_device_arch_key_retries_after_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert compiler._device_arch_key() == ("arch", "unavailable")
    # An unavailable probe must not be memoized, or an early call would pin a
    # bogus identity for the rest of the process.
    assert compiler._DEVICE_ARCH_KEYS == {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "SM121")
    assert compiler._device_arch_key() == ("arch", 12, 1, "SM121")
    assert compiler._DEVICE_ARCH_KEYS == {0: ("arch", 12, 1, "SM121")}


def test_device_arch_key_is_memoized_per_ordinal(monkeypatch):
    """A second GPU must not inherit the first GPU's architecture identity.

    torch resolves get_device_capability()/get_device_name() against the current
    device when no argument is given, so a process-wide cache would freeze the
    first device probed. Mixed rigs make this observable: two boards can share a
    compute capability but report different device names.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_ARCH_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    boards = {0: (12, 0, "MAX-Q"), 1: (12, 0, "FULL")}
    current = {"index": 0}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: boards[current["index"] if device is None else device][:2],
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda device=None: boards[current["index"] if device is None else device][2],
    )

    assert compiler._device_arch_key() == ("arch", 12, 0, "MAX-Q")
    # Selecting a different board must re-probe rather than return the memoized
    # identity of board 0.
    current["index"] = 1
    assert compiler._device_arch_key() == ("arch", 12, 0, "FULL")
    # Explicit ordinals resolve independently of whichever device is current.
    assert compiler._device_arch_key(0) == ("arch", 12, 0, "MAX-Q")
    assert compiler._DEVICE_ARCH_KEYS == {
        0: ("arch", 12, 0, "MAX-Q"),
        1: ("arch", 12, 0, "FULL"),
    }
