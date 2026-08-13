"""Security contract tests for auxiliary-file handling in FP6 export.

Covers issue #172: the exporters must never follow attacker-authored symlinks
or copy special file types when copying auxiliary files (tokenizer.json,
generation_config.json, etc.) into the output bundle. Every eligible
auxiliary symlink — absolute, relative, broken, same-repository HF cache
blob link — is rejected with :class:`UnsafeAuxiliaryFileError`; only ordinary
regular files are copied and real subdirectories are silently skipped.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch

from b12x.quantization.mxfp6.fp6_safetensors_export import (
    UnsafeAuxiliaryFileError,
    _copy_aux_files,
    export_dense_model_to_fp6_safetensors,
)


# ---------------------------------------------------------------------------
# Minimal checkpoint helpers (same shape as test_model_fp6_converter).
# ---------------------------------------------------------------------------

def _write_ckpt(path: pathlib.Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))


def _fake_dense(path: pathlib.Path, *, layers: int, k: int, n: int) -> None:
    t: dict[str, torch.Tensor] = {}
    for i in range(layers):
        t[f"model.language_model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{i}.mlp.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16)
    _write_ckpt(path, t)


SENTINEL = b"EXTERNAL_SECRET_SENTINEL_BYTES_42"


# ---------------------------------------------------------------------------
# Capability check.
# ---------------------------------------------------------------------------

def test_capabilities_available() -> None:
    """The safe-copy path requires O_NOFOLLOW, O_DIRECTORY, O_NONBLOCK,
    fd-relative os.open (os.supports_dir_fd), and fd-based os.listdir
    (os.supports_fd).  If any are missing the helper must fail closed.
    """
    for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"):
        assert hasattr(os, flag), f"missing required OS flag: {flag}"
    assert os.open in os.supports_dir_fd, "os.open must support dir_fd"
    assert os.listdir in os.supports_fd, "os.listdir must support fd"


def test_fails_closed_if_dir_fd_unsupported(tmp_path: pathlib.Path, monkeypatch) -> None:
    """If os.open does not support dir_fd, _copy_aux_files fails closed."""
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(UnsafeAuxiliaryFileError, match="dir_fd"):
        _copy_aux_files(src, dst)


def test_fails_closed_if_listdir_fd_unsupported(tmp_path: pathlib.Path, monkeypatch) -> None:
    """If os.listdir does not support fd, _copy_aux_files fails closed."""
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    monkeypatch.setattr(os, "supports_fd", set())
    with pytest.raises(UnsafeAuxiliaryFileError, match="listdir"):
        _copy_aux_files(src, dst)


# ---------------------------------------------------------------------------
# _copy_aux_files — unit-level policy tests.
# ---------------------------------------------------------------------------

def test_regular_aux_files_copied_byte_for_byte(tmp_path: pathlib.Path) -> None:
    """Ordinary regular auxiliary files are preserved exactly."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_bytes(b'{"hello": "world"}')
    (src / "tokenizer_config.json").write_bytes(b'{"name": "cfg"}')
    (src / "generation_config.json").write_bytes(b'{"temp": 0}')
    (src / "chat_template.jinja").write_bytes(b"{{ messages }}")
    # Files that must be skipped (exporter generates its own).
    (src / "config.json").write_bytes(b'{"should": "skip"}')
    (src / "model.safetensors").write_bytes(b"\x00" * 16)
    (src / "model.safetensors.index.json").write_bytes(b'{"skip": true}')

    dst = tmp_path / "dst"
    dst.mkdir()
    _copy_aux_files(src, dst)

    assert (dst / "tokenizer.json").read_bytes() == b'{"hello": "world"}'
    assert (dst / "tokenizer_config.json").read_bytes() == b'{"name": "cfg"}'
    assert (dst / "generation_config.json").read_bytes() == b'{"temp": 0}'
    assert (dst / "chat_template.jinja").read_bytes() == b"{{ messages }}"

    # Skipped files must not appear in the output.
    assert not (dst / "config.json").exists()
    assert not (dst / "model.safetensors").exists()
    assert not (dst / "model.safetensors.index.json").exists()


def test_real_subdirectories_silently_skipped(tmp_path: pathlib.Path) -> None:
    """Real subdirectories (not symlinks) are skipped, preserving old behavior."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_bytes(b"ok")
    # A real subdirectory — must be silently skipped, not an error.
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_bytes(b"ref: refs/heads/main")
    (src / ".cache").mkdir()
    (src / ".cache" / "data").write_bytes(b"cached")

    dst = tmp_path / "dst"
    dst.mkdir()
    _copy_aux_files(src, dst)

    assert (dst / "tokenizer.json").read_bytes() == b"ok"
    assert not (dst / ".git").exists()
    assert not (dst / ".cache").exists()


def test_absolute_symlink_rejected(tmp_path: pathlib.Path) -> None:
    """An absolute symlink to an external file is refused."""
    src = tmp_path / "src"
    src.mkdir()
    sentinel = tmp_path / "secret.txt"
    sentinel.write_bytes(SENTINEL)
    (src / "tokenizer.json").symlink_to(sentinel)

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(src, dst)

    # Sentinel bytes must not enter the output.
    assert not (dst / "tokenizer.json").exists()


def test_relative_symlink_outside_source_rejected(tmp_path: pathlib.Path) -> None:
    """A relative symlink escaping the source directory is refused."""
    src = tmp_path / "src"
    src.mkdir()
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    (secret_dir / "secret.txt").write_bytes(SENTINEL)
    (src / "tokenizer.json").symlink_to(pathlib.Path("..") / "secrets" / "secret.txt")

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(src, dst)
    assert not (dst / "tokenizer.json").exists()


def test_relative_symlink_inside_source_rejected(tmp_path: pathlib.Path) -> None:
    """A relative symlink whose target is inside the source dir is still refused.

    The policy rejects *all* symlinks regardless of where the target lives.
    A same-repository link is not trusted merely because its target is nearby.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "real_tokenizer.json").write_bytes(SENTINEL)
    (src / "tokenizer.json").symlink_to("real_tokenizer.json")

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(src, dst)
    assert not (dst / "tokenizer.json").exists()


def test_broken_symlink_rejected(tmp_path: pathlib.Path) -> None:
    """A dangling symlink (target does not exist) is rejected as a symlink.

    ``O_NOFOLLOW`` rejects the final component because it *is* a symlink,
    regardless of whether the target exists (ELOOP on Linux/macOS).  The
    broken link is refused, not silently skipped or followed.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").symlink_to(tmp_path / "nonexistent")

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(src, dst)
    assert not (dst / "tokenizer.json").exists()


def test_vanished_file_propagates_file_not_found(tmp_path: pathlib.Path) -> None:
    """A regular file deleted between listdir and open propagates ENOENT.

    ENOENT on a *non-symlink* (the entry vanished) is a changing-source
    condition, not a symlink safety violation.  The original ``OSError`` is
    propagated as-is rather than masked as ``UnsafeAuxiliaryFileError``.
    """
    import b12x.quantization.mxfp6.fp6_safetensors_export as mod

    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_bytes(b"ok")

    real_open = os.open
    deleted: list[bool] = []

    def deleting_open(path, flags, *args, **kwargs):
        if isinstance(path, str) and path == "tokenizer.json" and not deleted:
            deleted.append(True)
            os.unlink(src / "tokenizer.json")
        return real_open(path, flags, *args, **kwargs)

    dst = tmp_path / "dst"
    dst.mkdir()
    orig = mod._candidate_open
    mod._candidate_open = deleting_open
    try:
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            _copy_aux_files(src, dst)
    finally:
        mod._candidate_open = orig
    assert deleted, "monkeypatched open was never called for tokenizer.json"
    assert not (dst / "tokenizer.json").exists()


def test_hf_cache_snapshot_symlink_rejected(tmp_path: pathlib.Path) -> None:
    """Standard Hugging Face cache layout (snapshot -> blob) is rejected.

    HF hub cache stores files as ``snapshots/<rev>/<name>`` symlinking to
    ``../../blobs/<hash>``.  These are client-generated links to content-
    addressed blobs outside the snapshot directory.  The chosen explicit
    policy is to reject them and require materialization as real files.
    """
    cache_root = tmp_path / "hub" / "models--org--model"
    blobs = cache_root / "blobs"
    snapshots = cache_root / "snapshots" / "abc123def"
    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)

    blob = blobs / "deadbeef"
    blob.write_bytes(SENTINEL)
    (snapshots / "tokenizer.json").symlink_to(
        pathlib.Path("..") / ".." / "blobs" / "deadbeef"
    )

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(snapshots, dst)
    assert not (dst / "tokenizer.json").exists()


def test_symlink_object_not_copied(tmp_path: pathlib.Path) -> None:
    """The symlink itself must not be copied as a link into the output."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_bytes(b"real")
    (src / "tokenizer.json").symlink_to("real.txt")

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError):
        _copy_aux_files(src, dst)

    # No file and no symlink should appear at the destination.
    assert not (dst / "tokenizer.json").exists()
    assert not (dst / "tokenizer.json").is_symlink()


def test_symlink_to_directory_rejected(tmp_path: pathlib.Path) -> None:
    """A symlink to a directory is refused (only *real* directories are skipped)."""
    src = tmp_path / "src"
    src.mkdir()
    target_dir = tmp_path / "subdir"
    target_dir.mkdir()
    (src / "tokenizer.json").symlink_to(target_dir)

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        _copy_aux_files(src, dst)


def test_fifo_rejected_with_accurate_error(tmp_path: pathlib.Path) -> None:
    """A FIFO (named pipe) is rejected with an accurately named error.

    O_NONBLOCK prevents the open from blocking; the post-open fstat then
    identifies it as a FIFO and refuses to copy.  This test patches
    ``_candidate_open`` with a wrapper that asserts ``O_NONBLOCK`` is present
    in the flags before delegating to the real ``os.open`` — otherwise a
    regression that drops ``O_NONBLOCK`` would hang pytest on the FIFO open.
    """
    import b12x.quantization.mxfp6.fp6_safetensors_export as mod

    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_bytes(b"real")
    fifo_path = src / "pipe.json"
    os.mkfifo(fifo_path)

    real_open = os.open

    def nonblock_asserting_open(path, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK, (
            "candidate open must include O_NONBLOCK to avoid blocking on FIFO"
        )
        return real_open(path, flags, *args, **kwargs)

    dst = tmp_path / "dst"
    dst.mkdir()
    orig = mod._candidate_open
    mod._candidate_open = nonblock_asserting_open
    try:
        with pytest.raises(UnsafeAuxiliaryFileError, match="FIFO"):
            _copy_aux_files(src, dst)
    finally:
        mod._candidate_open = orig


def test_source_root_symlink_rejected(tmp_path: pathlib.Path) -> None:
    """A symlinked source directory root is rejected by O_NOFOLLOW on the root."""
    real_dir = tmp_path / "real_model"
    real_dir.mkdir()
    (real_dir / "tokenizer.json").write_bytes(b"benign")
    link = tmp_path / "linked_model"
    link.symlink_to(real_dir)

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises((UnsafeAuxiliaryFileError, OSError), match="linked_model"):
        _copy_aux_files(link, dst)
    assert not (dst / "tokenizer.json").exists()


def test_mix_regular_and_symlink_aborts_on_symlink(tmp_path: pathlib.Path) -> None:
    """If any auxiliary entry is a symlink, the copy aborts with an error."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer_config.json").write_bytes(b"ok")
    sentinel = tmp_path / "secret.txt"
    sentinel.write_bytes(SENTINEL)
    (src / "tokenizer.json").symlink_to(sentinel)

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(UnsafeAuxiliaryFileError):
        _copy_aux_files(src, dst)

    # The symlink target must never appear in output.
    if (dst / "tokenizer.json").exists():
        assert dst.joinpath("tokenizer.json").read_bytes() != SENTINEL


# ---------------------------------------------------------------------------
# Race resistance: monkeypatched open that swaps regular file to symlink.
# ---------------------------------------------------------------------------

def test_swap_race_regular_to_symlink_rejected(tmp_path: pathlib.Path) -> None:
    """A regular file swapped to a symlink *during* the open is rejected.

    The implementation opens each candidate with ``O_NOFOLLOW`` relative to the
    source directory descriptor — there is no separate ``lstat`` before the
    open, so there is no TOCTOU window.  This test monkeypatches the
    module-local ``_candidate_open`` seam to swap the target file to an
    external symlink immediately before the real ``os.open`` runs, proving
    the O_NOFOLLOW open rejects it atomically.  The capability check and
    source-root open continue to use the real ``os.open`` undisturbed.
    """
    import b12x.quantization.mxfp6.fp6_safetensors_export as mod

    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_bytes(b"benign")
    sentinel = tmp_path / "secret.txt"
    sentinel.write_bytes(SENTINEL)

    real_open = os.open
    swapped: list[bool] = []

    def swapping_open(path, flags, *args, **kwargs):
        # Assert the production code passes the expected safety flags and
        # a source-directory descriptor — not a bare CWD-relative open.
        assert kwargs.get("dir_fd") is not None, (
            "candidate open must use dir_fd (source-directory descriptor)"
        )
        assert flags & os.O_NOFOLLOW, "candidate open must include O_NOFOLLOW"
        # Swap the regular file to a symlink right before the real open.
        if isinstance(path, str) and path == "tokenizer.json" and not swapped:
            swapped.append(True)
            os.unlink(src / "tokenizer.json")
            (src / "tokenizer.json").symlink_to(sentinel)
        return real_open(path, flags, *args, **kwargs)

    dst = tmp_path / "dst"
    dst.mkdir()
    orig = mod._candidate_open
    mod._candidate_open = swapping_open
    try:
        with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
            _copy_aux_files(src, dst)
    finally:
        mod._candidate_open = orig

    assert swapped, "monkeypatched open was never called for tokenizer.json"
    assert not (dst / "tokenizer.json").exists()
    # Sentinel must never appear in output.
    if (dst / "tokenizer.json").exists():
        assert (dst / "tokenizer.json").read_bytes() != SENTINEL


# ---------------------------------------------------------------------------
# End-to-end: the public dense exporter enforces the same policy.
# ---------------------------------------------------------------------------

def test_dense_export_rejects_symlinked_aux(tmp_path: pathlib.Path) -> None:
    """export_dense_model_to_fp6_safetensors refuses a symlinked auxiliary file."""
    src = tmp_path / "d"
    _fake_dense(src, layers=1, k=64, n=64)
    sentinel = tmp_path / "secret.txt"
    sentinel.write_bytes(SENTINEL)
    (src / "tokenizer.json").symlink_to(sentinel)

    out = tmp_path / "outd"
    with pytest.raises(UnsafeAuxiliaryFileError, match="tokenizer.json"):
        export_dense_model_to_fp6_safetensors(
            src, out, device="cpu", use_gpu=False, verbose=False
        )

    # Sentinel bytes must not be anywhere in the output directory.
    if out.exists():
        for f in out.rglob("*"):
            if f.is_file():
                assert SENTINEL not in f.read_bytes(), (
                    f"sentinel bytes leaked into {f}"
                )


def test_dense_export_regular_aux_preserved(tmp_path: pathlib.Path) -> None:
    """A regular (non-symlink) auxiliary file is copied through the exporter."""
    src = tmp_path / "d"
    _fake_dense(src, layers=1, k=64, n=64)
    (src / "tokenizer.json").write_bytes(b'{"model": "test"}')
    (src / "generation_config.json").write_bytes(b'{"temp": 1}')

    out = tmp_path / "outd"
    export_dense_model_to_fp6_safetensors(
        src, out, device="cpu", use_gpu=False, verbose=False
    )

    assert (out / "tokenizer.json").read_bytes() == b'{"model": "test"}'
    assert (out / "generation_config.json").read_bytes() == b'{"temp": 1}'
    # The exporter writes its own config.json — the source one must not clobber it.
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "W6A6"
