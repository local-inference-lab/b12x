"""Emit a full HF safetensors checkpoint quantized to b12x MX-FP6 (W6A6).

Unlike the per-layer ``.pt`` converters in
:mod:`b12x.quantization.mxfp6.model_fp6` (useful for kernel validation),
this writes a *complete, loadable* model directory: sharded
``model-*.safetensors`` + ``model.safetensors.index.json`` + a ``config.json``
patched with a ModelOpt-mirror ``quantization_config``, plus all the original
auxiliary files (tokenizer, generation config, ...).

Quantized linears follow the :mod:`b12x.quantization.mxfp6.fp6_checkpoint`
schema (``.weight`` / ``.weight_scale`` / ``.weight_scale_2`` / ``.input_scale``).
Everything else (norms, embeddings, router gates, shared experts, attention for
MoE, SSM/linear-attention projections, lm_head, MTP heads) is copied through in
its original dtype and recorded in ``quantization_config.exclude_modules``.

* MoE: routed experts are written as **per-expert** keys
  (``...experts.{e}.gate_proj.weight`` + scales), mirroring ModelOpt NVFP4 and the
  b12x FP4 loader, regardless of whether the source stored them packed.
* Dense: MLP (+ optional attention) linears are quantized in place.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import torch

from .fp6_checkpoint import (
    QUANT_ALGO,
    build_quantization_config,
    dequantize_linear_from_fp6,
    quantize_linear_to_fp6,
)
from .model_fp6 import (
    SafetensorsModel,
    _split_gate_up,
    discover_moe_experts,
)

_TILE = 128
_GATE_PROJ = "gate_proj"
_UP_PROJ = "up_proj"
_DOWN_PROJ = "down_proj"

_LAYER_IDX_RE = re.compile(r"\.layers\.\d+\.")
_EXPERT_IDX_RE = re.compile(r"\.experts\.\d+\.")

# Dense quantization follows the LLM-Compressor "golden rule": quantize every 2-D
# Linear weight EXCEPT this ignore list (mirrors targets="Linear", ignore=[...]).
# ``lm_head`` is always skipped; ``embed``/norms/conv/rotary are not Linear matmuls;
# ``linear_attn`` / ``visual`` are skipped by default (opt back in via flags).
_DENSE_IGNORE_ALWAYS = (
    r"(?:^|\.)lm_head\.",
    r"(?:^|\.)(?:embed_tokens|embeddings?|word_embeddings|wte)\.",
    # Positional / patch embedding *tables* are 2-D but are NOT Linear matmuls
    # (vLLM builds them as plain Parameters / nn.Embedding, with no input_scale),
    # so they must stay un-quantized even when --include-vision strips ``.visual.``.
    r"(?:^|\.)pos_embed",
    r"(?:^|\.)position_embeddings?\.",
    r"(?:^|\.)patch_embed\.",
    r"(?:^|\.)mtp(?:\.|_)",
    r"\.mlp\.gate\.weight$",            # MoE router gate (NOT mlp.gate_proj)
    r"\.shared_expert_gate\.",
    r"\.(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|norm|layernorm|ln_?\w*)\.",
    r"\.rotary",
    r"\.conv\d?d?\.",
)
_RE_LINEAR_ATTN = r"\.linear_attn\."
_RE_VISION = r"\.visual\."
_RE_ATTENTION = r"\.self_attn\."


def _dense_ignore_re(
    *,
    include_attention: bool,
    include_linear_attn: bool,
    include_vision: bool,
    extra_ignores: tuple[str, ...] = (),
) -> re.Pattern:
    """Build the dense ignore regex for the requested scope (golden-rule mirror)."""
    parts = list(_DENSE_IGNORE_ALWAYS) + list(extra_ignores)
    if not include_linear_attn:
        parts.append(_RE_LINEAR_ATTN)
    if not include_vision:
        parts.append(_RE_VISION)
    if not include_attention:
        parts.append(_RE_ATTENTION)
    return re.compile("(" + "|".join(parts) + ")", re.IGNORECASE)


def _module_pattern(key: str) -> str:
    """Collapse a tensor key to a layer/expert-agnostic module glob pattern."""
    k = key
    for suffix in (".weight", ".bias", ".weight_scale", ".weight_scale_2", ".input_scale"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
            break
    k = _LAYER_IDX_RE.sub(".layers.*.", k)
    k = _EXPERT_IDX_RE.sub(".experts.*.", k)
    return k


@dataclass
class ExportReport:
    arch: str
    out_dir: str
    layers: list[int] = field(default_factory=list)
    quantized_tensors: int = 0
    copied_tensors: int = 0
    shards: int = 0
    total_bytes: int = 0
    skipped: list = field(default_factory=list)
    error_groups: dict = field(default_factory=dict)


def _error_group(name: str) -> str:
    """Collapse a module name to a sensitivity group for error aggregation."""
    proj = name.rsplit(".", 1)[-1]
    if _EXPERT_IDX_RE.search(name + "."):
        return f"experts.{proj}"
    for tag in ("shared_expert", "self_attn", "linear_attn", "visual", "mlp"):
        if f".{tag}." in f".{name}.":
            return f"{tag}.{proj}"
    return proj


class _ErrorStats:
    """Per-group quantization error accumulator (relative Frobenius RMSE).

    Group error is ``sqrt(sum ||W - W_hat||^2 / sum ||W||^2)`` over the group's
    tensors (energy-weighted, so large tensors dominate the way they dominate
    the forward pass), plus the single worst tensor by its own relative error.
    """

    def __init__(self) -> None:
        self._acc: dict[str, dict] = {}

    def add(self, name: str, w: torch.Tensor, w_hat: torch.Tensor) -> None:
        err_sq = (w_hat - w.float()).pow(2).sum().item()
        ref_sq = w.float().pow(2).sum().item()
        rel = (err_sq / ref_sq) ** 0.5 if ref_sq > 0 else 0.0
        g = self._acc.setdefault(
            _error_group(name),
            {"tensors": 0, "err_sq": 0.0, "ref_sq": 0.0, "worst": 0.0, "worst_key": ""},
        )
        g["tensors"] += 1
        g["err_sq"] += err_sq
        g["ref_sq"] += ref_sq
        if rel > g["worst"]:
            g["worst"] = rel
            g["worst_key"] = name

    def summary(self) -> dict[str, dict]:
        out = {}
        for name, g in self._acc.items():
            rel = (g["err_sq"] / g["ref_sq"]) ** 0.5 if g["ref_sq"] > 0 else 0.0
            out[name] = {
                "tensors": g["tensors"],
                "rel_rmse": rel,
                "worst": g["worst"],
                "worst_key": g["worst_key"],
            }
        return out

    def print_table(self) -> None:
        rows = sorted(
            self.summary().items(), key=lambda kv: kv[1]["rel_rmse"], reverse=True
        )
        print(
            f"[fp6-error] {'group':<28}{'tensors':>8}  "
            f"{'rel-RMSE':>9}  {'worst':>9}  worst tensor"
        )
        for name, s in rows:
            print(
                f"[fp6-error] {name:<28}{s['tensors']:>8}  "
                f"{s['rel_rmse']:>9.5f}  {s['worst']:>9.5f}  {s['worst_key']}"
            )

# ---------------------------------------------------------------------------
# Secure fd-anchored export helpers (issue #173)
#
# Every public FP6 export entrypoint builds the complete checkpoint in a
# private, mode-0700 sibling *staging* directory and publishes it with a
# single atomic directory rename.  All file operations are anchored to
# held directory descriptors (``parent_fd``, ``staging_fd``) so that an
# attacker cannot redirect writes by renaming or symlinking path components
# after creation.
#
# Security properties:
#   * Every ancestor of the output path is walked with openat(O_NOFOLLOW |
#     O_DIRECTORY), rejecting symlinked components.
#   * The staging directory is created exclusively via os.mkdir(dir_fd=) and
#     its inode is pinned; all cleanup verifies the inode before acting.
#   * safetensors.save() serializes to bytes in memory; the bytes are written
#     to a fresh fd opened with O_CREAT|O_EXCL relative to staging_fd, then
#     renamed to the final name — never unlinked and re-opened.
#   * The final destination leaf must not exist (checked via stat with
#     dir_fd=parent_fd); publication is a single os.replace directory rename.
#   * On failure, only the owned staging directory (identified by inode) is
#     cleaned up — never a path-resolved replacement.
# ---------------------------------------------------------------------------
import contextlib
import ctypes
import ctypes.util
import stat
import sys


# Platform-specific atomic no-replace directory rename.
def _init_rename_no_replace():
    libc_path = ctypes.util.find_library("c")
    if libc_path is None:
        return None
    libc = ctypes.CDLL(libc_path, use_errno=True)
    if sys.platform == "darwin":
        try:
            fn = libc.renameatx_np
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        except AttributeError:
            return None
        def _mac_rename(from_fd, from_name, to_fd, to_name):
            result = fn(from_fd, from_name.encode(), to_fd, to_name.encode(), 0x0004)
            if result != 0:
                err = ctypes.get_errno()
                if err == 17:
                    raise FileExistsError(f"output destination already exists: {to_name}")
                raise OSError(err, os.strerror(err))
        return _mac_rename
    if sys.platform.startswith("linux"):
        try:
            fn = libc.renameat2
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        except AttributeError:
            return None
        def _linux_rename(from_fd, from_name, to_fd, to_name):
            result = fn(from_fd, from_name.encode(), to_fd, to_name.encode(), 0x1)
            if result != 0:
                err = ctypes.get_errno()
                if err == 17:
                    raise FileExistsError(f"output destination already exists: {to_name}")
                raise OSError(err, os.strerror(err))
        return _linux_rename
    return None


_RENAME_NO_REPLACE = _init_rename_no_replace()


def _resolve_canonical(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.realpath(str(path), strict=False))


def _resolve_absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(str(path)))


def _openat_nofollow_dir(fd: int, name: str) -> int:
    return os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY, dir_fd=fd)


def _inode(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino)


def _is_safe_parent(fd: int) -> bool:
    """True if the directory is safe against attacker entry substitution.

    Safe if owned by current user and not group/world-writable, OR if
    sticky bit set AND owned by root or current user (so the directory
    owner cannot rename other users' entries).
    """
    st = os.fstat(fd)
    mode = st.st_mode
    uid = os.getuid()
    if mode & stat.S_ISVTX:
        # Sticky bit: only entry owner can rename/unlink. But the directory
        # owner can still rename entries they own. Require trusted owner.
        return st.st_uid in (0, uid)
    return bool(st.st_uid == uid and not (mode & (stat.S_IWGRP | stat.S_IWOTH)))


def _fsync_fd(fd: int) -> None:
    """fsync a file descriptor, ignoring errors on unsupported platforms."""
    with contextlib.suppress(OSError):
        os.fsync(fd)


def _fsync_dir(fd: int) -> None:
    """fsync a directory fd for durability of its entries."""
    _fsync_fd(fd)


@contextlib.contextmanager
def _fd_cmgr(fd: int):
    try:
        yield fd
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _walk_and_pin_source(src_abs: pathlib.Path) -> tuple[int, set[tuple[int, int]]]:
    """Walk source path with openat, returning (src_dir_fd, src_inodes)."""
    comps = [p for p in src_abs.parts[1:] if p]
    if not comps:
        raise ValueError("source path has no components")
    root_fd = os.open("/", os.O_RDONLY | os.O_NOFOLLOW)
    fd = root_fd
    try:
        for comp in comps:
            next_fd = _openat_nofollow_dir(fd, comp)
            if fd != root_fd:
                os.close(fd)
            fd = next_fd
        result_fd = fd
        fd = None  # transfer ownership
        return result_fd, {_inode(result_fd)}
    finally:
        if fd is not None and fd != root_fd:
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def _walk_to_parent(
    out_abs: pathlib.Path,
    src_inodes: set[tuple[int, int]],
) -> tuple[int, str, list[str]]:
    """Walk ancestors of *out_abs*, creating missing dirs.

    Returns (parent_fd, leaf_name, created_ancestors) where created_ancestors
    tracks dirs we created so they can be cleaned up on failure.
    """
    comps = [p for p in out_abs.parts[1:] if p]
    if not comps:
        raise ValueError("output path has no leaf component")
    leaf = comps[-1]
    root_fd = os.open("/", os.O_RDONLY | os.O_NOFOLLOW)
    fd = root_fd
    created: list[str] = []
    try:
        for comp in comps[:-1]:
            if _inode(fd) in src_inodes:
                raise ValueError(
                    f"output path overlaps source directory at component '{comp}'"
                )
            try:
                next_fd = _openat_nofollow_dir(fd, comp)
            except FileNotFoundError:
                os.mkdir(comp, dir_fd=fd)
                created.append(comp)
                next_fd = _openat_nofollow_dir(fd, comp)
            if fd != root_fd:
                os.close(fd)
            fd = next_fd
        if _inode(fd) in src_inodes:
            raise ValueError("output parent overlaps source directory")
        result_fd = fd
        fd = None  # transfer ownership
        return result_fd, leaf, created
    finally:
        if fd is not None and fd != root_fd:
            os.close(fd)
        os.close(root_fd)


def _staging_name(leaf: str) -> str:
    import random
    import string
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f".b12x-ws-{leaf}-{suffix}"


def _write_all(fd: int, data: bytes) -> None:
    mv = memoryview(data)
    total = 0
    while total < len(mv):
        written = os.write(fd, mv[total:])
        if written <= 0:
            raise OSError("short write returned non-positive count")
        total += written


_DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024


class _SecureExportContext:
    """fd-anchored workspace for atomic checkpoint publication.

    Creates a private mode-0700 workspace directory in the output parent.
    The checkpoint payload is built as a child of the workspace.  Publication
    renames the payload child from the (inaccessible) workspace fd to the
    final leaf name in the parent — the attacker cannot swap the payload
    because it lives inside the mode-0700 workspace.
    """

    def __init__(self, src_path: pathlib.Path, out_dir: pathlib.Path, *, src_dir_fd: int | None = None):
        self.src_abs = _resolve_canonical(pathlib.Path(src_path))
        self.out_abs = _resolve_absolute(pathlib.Path(out_dir))
        self.parent_fd: int | None = None
        self.workspace_fd: int | None = None
        self._payload_fd: int | None = None
        self.workspace_name: str | None = None
        self.payload_name: str | None = None
        self.leaf_name: str | None = None
        self.workspace_identity: tuple[int, int] | None = None
        self._payload_identity: tuple[int, int] | None = None
        self.src_inodes: set[tuple[int, int]] = set()
        self._published = False
        self._src_dir_fd: int | None = src_dir_fd
        self._owns_src_fd = src_dir_fd is None  # we close it only if we opened it
        self._exit_stack: contextlib.ExitStack | None = None
        self._created_ancestors: list[str] = []

    @property
    def staging_fd(self) -> int | None:
        """The payload directory fd (for ShardWriter compatibility)."""
        return self._payload_fd

    def __enter__(self) -> "_SecureExportContext":
        self._exit_stack = contextlib.ExitStack()
        stack = self._exit_stack
        try:
            # Pin source if not already provided externally.
            if self._src_dir_fd is not None:
                self.src_inodes = {_inode(self._src_dir_fd)}
            else:
                self._src_dir_fd, self.src_inodes = _walk_and_pin_source(self.src_abs)
                self._owns_src_fd = True
            if self._owns_src_fd:
                stack.callback(os.close, self._src_dir_fd)

            # Walk to output parent, creating missing ancestors.
            self.parent_fd, self.leaf_name, self._created_ancestors = _walk_to_parent(
                self.out_abs, self.src_inodes
            )
            stack.callback(os.close, self.parent_fd)

            # Validate parent safety.
            if not _is_safe_parent(self.parent_fd):
                raise PermissionError(
                    f"output parent directory is not safe: it must be owned by "
                    f"the current user with no group/world write, or have the "
                    f"sticky bit set with a trusted owner; refusing to export "
                    f"to {self.out_abs}"
                )

            # Check leaf existence and overlap.
            try:
                leaf_stat = os.lstat(self.leaf_name, dir_fd=self.parent_fd)
                if (leaf_stat.st_dev, leaf_stat.st_ino) in self.src_inodes:
                    raise ValueError(
                        f"output {self.out_abs} overlaps source model directory"
                    )
                raise FileExistsError(
                    f"output destination {self.out_abs} already exists; "
                    f"refusing to overwrite"
                )
            except FileNotFoundError:
                pass

            # Create private mode-0700 workspace sibling.
            self.workspace_name = _staging_name(self.leaf_name)
            os.mkdir(self.workspace_name, mode=0o700, dir_fd=self.parent_fd)
            try:
                self.workspace_fd = _openat_nofollow_dir(
                    self.parent_fd, self.workspace_name
                )
            except BaseException:
                with contextlib.suppress(OSError):
                    os.rmdir(self.workspace_name, dir_fd=self.parent_fd)
                raise
            stack.callback(os.close, self.workspace_fd)
            self.workspace_identity = _inode(self.workspace_fd)

            # Create payload (the actual checkpoint dir) inside workspace.
            self.payload_name = "payload"
            os.mkdir(self.payload_name, mode=0o700, dir_fd=self.workspace_fd)
            self._payload_fd = _openat_nofollow_dir(
                self.workspace_fd, self.payload_name
            )
            self._payload_identity = _inode(self._payload_fd)
            stack.callback(os.close, self._payload_fd)
            return self
        except BaseException:
            self._cleanup_on_error()
            raise

    def _cleanup_on_error(self) -> None:
        """Clean up on __enter__ failure."""
        # Remove payload if created (inside workspace, fd-anchored).
        if self._payload_fd is not None and self.payload_name is not None:
            with contextlib.suppress(OSError):
                if _inode(self._payload_fd) == getattr(self, "_payload_identity", None):
                    for entry in os.listdir(self._payload_fd):
                        with contextlib.suppress(OSError):
                            os.unlink(entry, dir_fd=self._payload_fd)
                    with contextlib.suppress(OSError):
                        os.rmdir(self.payload_name, dir_fd=self.workspace_fd)
        # Remove workspace if created.
        if self.workspace_fd is not None and self.workspace_name is not None:
            with contextlib.suppress(OSError):
                if _inode(self.workspace_fd) == self.workspace_identity:
                    for entry in os.listdir(self.workspace_fd):
                        if entry != self.payload_name:
                            with contextlib.suppress(OSError):
                                os.unlink(entry, dir_fd=self.workspace_fd)
                    with contextlib.suppress(OSError):
                        os.rmdir(self.workspace_name, dir_fd=self.parent_fd)
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
        self.parent_fd = None
        self.workspace_fd = None
        self._payload_fd = None
        self._src_dir_fd = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        original_exc = exc_val
        try:
            if exc_type is None and not self._published:
                try:
                    self._publish()
                except BaseException:
                    self._cleanup()
                    raise
            elif not self._published:
                self._cleanup()
        except BaseException:
            if original_exc is not None:
                pass
            elif exc_type is None:
                raise
        finally:
            if self._exit_stack is not None:
                self._exit_stack.close()
                self._exit_stack = None
            self.parent_fd = None
            self.workspace_fd = None
            self._payload_fd = None
            self._src_dir_fd = None
        return None

    def _publish(self) -> None:
        """Atomically rename payload child to the final leaf name."""
        if _RENAME_NO_REPLACE is None:
            raise RuntimeError(
                "platform does not support atomic no-replace rename; "
                "cannot safely publish checkpoint"
            )
        # Verify workspace identity.
        if _inode(self.workspace_fd) != self.workspace_identity:
            raise RuntimeError("workspace directory identity changed; aborting")
        # fsync payload dir and workspace dir before rename.
        _fsync_dir(self._payload_fd)
        _fsync_dir(self.workspace_fd)
        # Atomic no-replace rename: payload_name (in workspace) -> leaf_name (in parent).
        _RENAME_NO_REPLACE(
            self.workspace_fd, self.payload_name,
            self.parent_fd, self.leaf_name,
        )
        # fsync parent dir to commit the rename.
        _fsync_dir(self.parent_fd)
        self._published = True
        # Remove now-empty workspace.
        with contextlib.suppress(OSError):
            os.rmdir(self.workspace_name, dir_fd=self.parent_fd)

    def _cleanup(self) -> None:
        """Remove payload and workspace, verifying identity."""
        # Remove payload contents (fd-anchored).
        if self._payload_fd is not None and self.payload_name is not None:
            try:
                if _inode(self._payload_fd) == getattr(self, "_payload_identity", _inode(self._payload_fd)):
                    for entry in os.listdir(self._payload_fd):
                        with contextlib.suppress(OSError):
                            os.unlink(entry, dir_fd=self._payload_fd)
                    with contextlib.suppress(OSError):
                        os.rmdir(self.payload_name, dir_fd=self.workspace_fd)
            except OSError:
                pass
        # Remove workspace.
        if self.workspace_fd is not None and self.workspace_name is not None:
            try:
                if _inode(self.workspace_fd) == self.workspace_identity:
                    for entry in os.listdir(self.workspace_fd):
                        with contextlib.suppress(OSError):
                            os.unlink(entry, dir_fd=self.workspace_fd)
                    with contextlib.suppress(OSError):
                        os.rmdir(self.workspace_name, dir_fd=self.parent_fd)
            except OSError:
                pass

    # -- fd-anchored write helpers ------------------------------------------

    def write_bytes(self, filename: str, data: bytes) -> int:
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o644, dir_fd=self._payload_fd)
        try:
            _write_all(fd, data)
            _fsync_fd(fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(filename, dir_fd=self._payload_fd)
            raise
        return fd

    def write_text(self, filename: str, text: str) -> None:
        fd = self.write_bytes(filename, text.encode("utf-8"))
        os.close(fd)

    def save_safetensors(self, save_fn, tensors: dict, filename: str) -> None:
        data = save_fn(tensors, metadata={"format": "pt"})
        fd = self.write_bytes(filename, data)
        os.close(fd)

    def rename_in_staging(self, old_name: str, final_name: str) -> None:
        os.replace(old_name, final_name, src_dir_fd=self._payload_fd, dst_dir_fd=self._payload_fd)

    def copy_aux_files(self, src_dir_fd: int) -> None:
        """Copy aux files from source dirfd into payload, rejecting symlinks/FIFOs."""
        if src_dir_fd is None:
            raise OSError("source directory fd is not available")
        skip_names = {"model.safetensors.index.json", "config.json"}
        for entry in os.listdir(src_dir_fd):
            if entry in skip_names or entry.endswith(".safetensors") or entry.endswith(".safetensors.index.json"):
                continue
            # Open with O_NOFOLLOW | O_NONBLOCK (rejects symlinks, prevents FIFO block).
            try:
                src_entry_fd = os.open(
                    entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_dir_fd
                )
            except OSError:
                continue  # symlink or special file — skip
            try:
                st = os.fstat(src_entry_fd)
                if (st.st_mode & 0o170000) != 0o100000:
                    continue  # not a regular file
                # Read all content from the source fd.
                data = b""
                while True:
                    chunk = os.read(src_entry_fd, 65536)
                    if not chunk:
                        break
                    data += chunk
                # Write to payload via fd.
                dst_fd = os.open(entry, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o644, dir_fd=self._payload_fd)
                try:
                    _write_all(dst_fd, data)
                    os.fchmod(dst_fd, st.st_mode)
                    os.utime(dst_fd, (st.st_atime, st.st_mtime))
                    _fsync_fd(dst_fd)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.close(dst_fd)
                    with contextlib.suppress(OSError):
                        os.unlink(entry, dir_fd=self._payload_fd)
                    raise
                os.close(dst_fd)
            finally:
                os.close(src_entry_fd)

    def fsync_staging(self) -> None:
        """fsync the payload directory after all files are written."""
        if self._payload_fd is not None:
            _fsync_dir(self._payload_fd)

    @property
    def out_path(self) -> pathlib.Path:
        return self.out_abs
    @property
    def staging_name(self) -> str | None:
        """Workspace name for test compatibility."""
        return self.workspace_name

class _ShardWriter:
    """Stream tensors to size-capped safetensors shards, then finalize the index."""

    def __init__(self, ctx: _SecureExportContext, *, max_shard_bytes: int):
        from safetensors.torch import save as _save_bytes

        self._save = _save_bytes  # returns bytes, not writes to file
        self._ctx = ctx
        self.max_shard_bytes = max_shard_bytes
        self._buf: dict[str, torch.Tensor] = {}
        self._buf_bytes = 0
        self._shard_keys: list[list[str]] = []  # provisional shard idx -> keys
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        # Check size BEFORE detach/cpu transfer to avoid GPU OOM.
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.max_shard_bytes:
            raise ValueError(
                f"tensor {key} is {nbytes} bytes which exceeds the shard cap "
                f"of {self.max_shard_bytes} bytes; reduce max_shard_bytes or "
                f"split the model"
            )
        t = tensor.detach().cpu().contiguous()
        if self._buf_bytes and self._buf_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buf[key] = t
        self._buf_bytes += nbytes
        self.total_bytes += nbytes

    def add_many(self, tensors: dict[str, torch.Tensor]) -> None:
        for key, tensor in tensors.items():
            self.add(key, tensor)

    def _flush(self) -> None:
        if not self._buf:
            return
        idx = len(self._shard_keys)
        name = f"model-{idx:05d}.safetensors"
        self._ctx.save_safetensors(self._save, self._buf, name)
        self._shard_keys.append(list(self._buf.keys()))
        self._buf = {}
        self._buf_bytes = 0

    def finalize(self) -> int:
        """Flush, rename shards to HF ``-of-N`` form, write the index. Returns shard count."""
        self._flush()
        n = len(self._shard_keys)
        for idx, keys in enumerate(self._shard_keys):
            old_name = f"model-{idx:05d}.safetensors"
            final = (
                "model.safetensors"
                if n == 1
                else f"model-{idx + 1:05d}-of-{n:05d}.safetensors"
            )
            self._ctx.rename_in_staging(old_name, final)
            for key in keys:
                self.weight_map[key] = final
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": self.weight_map,
        }
        self._ctx.write_text(
            "model.safetensors.index.json",
            json.dumps(index, indent=2),
        )
        self._ctx.fsync_staging()
        return n


def _write_config(
    src_config: dict, ctx: _SecureExportContext, quant_config: dict
) -> None:
    cfg = dict(src_config)
    cfg["quantization_config"] = quant_config
    ctx.write_text("config.json", json.dumps(cfg, indent=2))




def _emit_quantized_linear(
    writer: _ShardWriter,
    report: ExportReport,
    name: str,
    weight_bf16: torch.Tensor,
    *,
    source_format: str,
    use_gpu: bool,
    block_scale_rule: str = "mse",
    error_stats: Optional[_ErrorStats] = None,
) -> bool:
    """Quantize one ``(out, in)`` weight and stream its four FP6 keys.

    Returns ``True`` if the weight was quantized, ``False`` if it was copied
    through as BF16 (non-2-D, or a dimension the quantizer can't handle). The
    caller must record copied weights in ``exclude_modules``.
    """
    # Safety net for opt-in extra scope: anything that isn't a 2-D matmul
    # (conv1d, embeddings, ...) is copied through untouched.
    if weight_bf16.ndim != 2:
        report.skipped.append(
            {"key": name + ".weight", "reason": f"ndim {weight_bf16.ndim} not 2-D"}
        )
        writer.add(name + ".weight", weight_bf16)
        report.copied_tensors += 1
        return False
    out_f, in_f = weight_bf16.shape
    # MSE joint encode is torch-only (no TMA tile constraint). Ceil+GPU needs
    # out/in multiples of 128; ceil torch / mse only need in % 32 and in % 4.
    needs_tma_tile = use_gpu and block_scale_rule == "ceil"
    quantizable = (
        (out_f % _TILE == 0 and in_f % _TILE == 0)
        if needs_tma_tile
        else (in_f % 32 == 0 and in_f % 4 == 0)
    )
    if not quantizable:
        report.skipped.append(
            {"key": name + ".weight", "reason": f"shape {(out_f, in_f)} unquantizable"}
        )
        writer.add(name + ".weight", weight_bf16)
        report.copied_tensors += 1
        return False
    qt = quantize_linear_to_fp6(
        weight_bf16,
        source_format=source_format,
        use_gpu=use_gpu,
        block_scale_rule=block_scale_rule,
    )
    if error_stats is not None:
        dev = weight_bf16.device
        w_hat = dequantize_linear_from_fp6(
            qt.weight.to(dev),
            qt.weight_scale.to(dev),
            fmt=qt.fmt,
            weight_scale_2=qt.weight_scale_2,
        )
        error_stats.add(name, weight_bf16, w_hat)
        del w_hat
    writer.add_many(qt.to_state_dict(name))
    report.quantized_tensors += 1
    return True


def export_moe_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    activation: str = "silu",
    gate_up_order: str = "gate_up",
    include_attention: bool = True,
    include_linear_attn: bool = False,
    include_vision: bool = False,
    skip_experts: bool = False,
    skip_shared_expert: bool = False,
    report_error: bool = False,
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    use_gpu: bool = True,
    block_scale_rule: str = "mse",
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
    dry_run: bool = False,
    verbose: bool = True,
) -> ExportReport:
    """Quantize routed experts to per-expert FP6 keys + the non-expert Linears.

    Routed experts are always quantized. In addition, the same golden-rule walk
    used by the dense exporter is applied to every *non-expert* 2-D Linear weight
    (attention, shared experts, ...): quantize unless it matches the ignore set.
    ``include_linear_attn`` / ``include_vision`` opt the linear-attention / vision
    projections into FP6 too (off by default). Leaving the large ``linear_attn``
    block in BF16 keeps it at 16-bit -- heavier than the FP8 build's 8-bit -- which
    is why an experts-only FP6 MoE does not shrink below FP8; enable the flag to
    cross under it. Quality trade-off: validate downstream.

    ``skip_experts`` / ``skip_shared_expert`` are sensitivity-sweep knobs: they
    copy that group through in its original dtype (and record it in
    ``exclude_modules`` so serving BF16-falls-back) so a KLD run isolates the
    contribution of the remaining quantized groups. Not for production exports.
    ``report_error`` dequantizes every emitted tensor and prints a per-group
    relative-RMSE table (also returned on ``report.error_groups``).
    """
    # Pin source directory by inode before any reads.
    src_abs = _resolve_canonical(pathlib.Path(model_path))
    src_dir_fd, _src_inodes = _walk_and_pin_source(src_abs)
    try:
        model = SafetensorsModel(model_path, src_dir_fd=src_dir_fd)
    except BaseException:
        os.close(src_dir_fd)
        raise
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise ValueError(f"no MoE experts discovered under {model_path}")
    is_gated = activation == "silu"

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    report = ExportReport(arch="moe", out_dir=str(out_dir), layers=list(layers))
    if verbose:
        print(
            f"[moe-export] experts={scheme.num_experts} layers={len(layers)} "
            f"packed={scheme.packed} fused={scheme.is_fused} "
            f"gate_up_order={gate_up_order} src={source_format}"
        )

    # Source keys the per-expert FP6 emission replaces (dropped from the copy
    # pass). With skip_experts the source expert tensors are copied through
    # unmodified instead (the copy pass records their exclude patterns).
    drop_keys: set[str] = set()
    for layer in layers if not skip_experts else []:
        if scheme.packed:
            drop_keys.add(scheme.packed_key(layer, scheme.gate_name))
            if not scheme.is_fused:
                drop_keys.add(scheme.packed_key(layer, scheme.up_name))
            drop_keys.add(scheme.packed_key(layer, scheme.down_name))
        else:
            for e in range(scheme.num_experts):
                drop_keys.add(scheme.expert_key(layer, e, scheme.gate_name))
                if not scheme.is_fused:
                    drop_keys.add(scheme.expert_key(layer, e, scheme.up_name))
                drop_keys.add(scheme.expert_key(layer, e, scheme.down_name))

    # Golden-rule selection for the NON-expert weights (attention, shared experts,
    # optionally linear-attn / vision). Routed experts are handled by the dedicated
    # per-expert pass below, so they are excluded here.
    ignore_re = _dense_ignore_re(
        include_attention=include_attention,
        include_linear_attn=include_linear_attn,
        include_vision=include_vision,
        extra_ignores=(r"\.shared_expert\.",) if skip_shared_expert else (),
    )
    quant_keys: set[str] = set()
    for key in model.keys():
        if key in drop_keys or _EXPERT_IDX_RE.search(key):
            continue
        if not key.endswith(".weight") or ignore_re.search(key):
            continue
        shape = tuple(model.shape_of(key))
        if len(shape) != 2:
            continue
        out_f, in_f = shape
        needs_tma_tile = use_gpu and block_scale_rule == "ceil"
        ok = (
            (out_f % _TILE == 0 and in_f % _TILE == 0)
            if needs_tma_tile
            else (in_f % 32 == 0 and in_f % 4 == 0)
        )
        if ok:
            quant_keys.add(key)

    expert_quant = (
        0 if skip_experts
        else len(layers) * scheme.num_experts * (3 if is_gated else 2)
    )
    if verbose:
        print(
            f"[moe-export] non-expert quantizable Linear weights={len(quant_keys)} "
            f"(attention={include_attention} linear_attn={include_linear_attn} "
            f"vision={include_vision} skip_experts={skip_experts} "
            f"skip_shared_expert={skip_shared_expert} "
            f"block_scale_rule={block_scale_rule})"
        )

    if dry_run:
        report.quantized_tensors = expert_quant + len(quant_keys)
        os.close(src_dir_fd)
        return report

    with _SecureExportContext(
        pathlib.Path(model_path), pathlib.Path(out_dir), src_dir_fd=src_dir_fd
    ) as ctx:
        writer = _ShardWriter(ctx, max_shard_bytes=max_shard_bytes)
        exclude_patterns: set[str] = set()
        error_stats = _ErrorStats() if report_error else None

        # 1) Walk non-expert tensors: quantize golden-rule Linears, copy the rest BF16.
        for key in model.keys():
            if key in drop_keys:
                continue
            if key in quant_keys:
                name = key[: -len(".weight")]
                w = model.get_tensor(key).to(device, torch.bfloat16)
                quantized = _emit_quantized_linear(
                    writer, report, name, w, source_format=source_format,
                    use_gpu=use_gpu, block_scale_rule=block_scale_rule,
                    error_stats=error_stats,
                )
                if not quantized:
                    exclude_patterns.add(_module_pattern(key))
                del w
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                writer.add(key, model.get_tensor(key))
                report.copied_tensors += 1
                exclude_patterns.add(_module_pattern(key))

        # 2) Emit per-expert FP6 keys for the quantized layers.
        for layer in layers if not skip_experts else []:
            gate_e, up_e, down_e = _layer_expert_matrices(
                model, scheme, layer, device, is_gated, gate_up_order
            )
            prefix = scheme.prefix_template.format(L=layer)
            for e in range(scheme.num_experts):
                _emit_quantized_linear(
                    writer, report, f"{prefix}.experts.{e}.{_DOWN_PROJ}",
                    down_e[e], source_format=source_format, use_gpu=use_gpu,
                    block_scale_rule=block_scale_rule, error_stats=error_stats,
                )
                _emit_quantized_linear(
                    writer, report, f"{prefix}.experts.{e}.{_GATE_PROJ}",
                    gate_e[e], source_format=source_format, use_gpu=use_gpu,
                    block_scale_rule=block_scale_rule, error_stats=error_stats,
                )
                if is_gated:
                    _emit_quantized_linear(
                        writer, report, f"{prefix}.experts.{e}.{_UP_PROJ}",
                        up_e[e], source_format=source_format, use_gpu=use_gpu,
                        block_scale_rule=block_scale_rule, error_stats=error_stats,
                    )
            if verbose:
                print(f"  layer {layer}: {scheme.num_experts} experts -> FP6")
            del gate_e, up_e, down_e
            if device == "cuda":
                torch.cuda.empty_cache()

        report.shards = writer.finalize()
        report.total_bytes = writer.total_bytes
        quant_config = build_quantization_config(
            source_format=source_format,
            exclude_modules=sorted(exclude_patterns),
            block_scale_rule=block_scale_rule,
        )
        _write_config(model.config, ctx, quant_config)
        ctx.copy_aux_files(ctx._src_dir_fd)
        out = ctx.out_path

    if error_stats is not None:
        report.error_groups = error_stats.summary()
        if verbose:
            error_stats.print_table()
    if verbose:
        print(
            f"[moe-export] done: quantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def _layer_expert_matrices(
    model: SafetensorsModel,
    scheme,
    layer: int,
    device: str,
    is_gated: bool,
    gate_up_order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-expert ``(gate (E,N,K), up (E,N,K), down (E,K,N))`` bf16 matrices.

    ``gate``/``up`` are in ModelOpt linear orientation (out=N, in=K); ``down`` is
    (out=K, in=N). For non-gated activations ``up`` is returned but unused.
    """
    if scheme.packed:
        down = model.get_tensor(scheme.packed_key(layer, scheme.down_name)).to(
            device, torch.bfloat16
        )  # (E, K, N)
        if scheme.is_fused:
            gate_up = model.get_tensor(scheme.packed_key(layer, scheme.gate_name)).to(
                device, torch.bfloat16
            )  # (E, 2N, K)
            gate, up = _split_gate_up(gate_up, dim=1, order=gate_up_order)
        else:
            gate = model.get_tensor(scheme.packed_key(layer, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            up = model.get_tensor(scheme.packed_key(layer, scheme.up_name)).to(
                device, torch.bfloat16
            )
        return gate.contiguous(), up.contiguous(), down.contiguous()

    gate_list, up_list, down_list = [], [], []
    for e in range(scheme.num_experts):
        down_list.append(
            model.get_tensor(scheme.expert_key(layer, e, scheme.down_name)).to(
                device, torch.bfloat16
            )
        )
        if scheme.is_fused:
            gate_up = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            g, u = _split_gate_up(gate_up, dim=0, order=gate_up_order)
        else:
            g = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            u = model.get_tensor(scheme.expert_key(layer, e, scheme.up_name)).to(
                device, torch.bfloat16
            )
        gate_list.append(g)
        up_list.append(u)
    gate = torch.stack(gate_list, dim=0)
    up = torch.stack(up_list, dim=0) if is_gated else gate
    down = torch.stack(down_list, dim=0)
    return gate, up, down


def export_dense_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    include_attention: bool = True,
    include_linear_attn: bool = False,
    include_vision: bool = False,
    report_error: bool = False,
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    use_gpu: bool = True,
    block_scale_rule: str = "mse",
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
    dry_run: bool = False,
    verbose: bool = True,
) -> ExportReport:
    """Quantize dense MLP (+ optional attention) linears; copy everything else through.

    ``include_linear_attn`` / ``include_vision`` opt the 2-D matmul projections of
    linear-attention layers / the vision tower into FP6 too (off by default). These
    are normally the dominant BF16 residue in multimodal-hybrid checkpoints, so
    enabling them is what brings the FP6 size below the FP8 equivalent. They are a
    quality trade-off (FP6 is lower precision than the FP8 those layers usually
    ship in) — validate downstream before relying on them.
    """
    # Pin source directory by inode before any reads.
    src_abs = _resolve_canonical(pathlib.Path(model_path))
    src_dir_fd, _src_inodes = _walk_and_pin_source(src_abs)
    try:
        model = SafetensorsModel(model_path, src_dir_fd=src_dir_fd)
    except BaseException:
        os.close(src_dir_fd)
        raise

    # Golden-rule selection (mirrors LLM-Compressor targets="Linear", ignore=[...]):
    # quantize every 2-D ``.weight`` whose dims the quantizer can handle and that is
    # not in the ignore set. No reliance on hardcoded module names, so it adapts to
    # any architecture ("insert any model").
    ignore_re = _dense_ignore_re(
        include_attention=include_attention,
        include_linear_attn=include_linear_attn,
        include_vision=include_vision,
    )
    layer_idx_re = re.compile(r"\.layers\.(\d+)\.")

    quant_keys: set[str] = set()
    for key in model.keys():
        if not key.endswith(".weight") or ignore_re.search(key):
            continue
        shape = tuple(model.shape_of(key))
        if len(shape) != 2:
            continue
        out_f, in_f = shape
        needs_tma_tile = use_gpu and block_scale_rule == "ceil"
        ok = (
            (out_f % _TILE == 0 and in_f % _TILE == 0)
            if needs_tma_tile
            else (in_f % 32 == 0 and in_f % 4 == 0)
        )
        if not ok:
            continue
        if limit_layers is not None:
            lm = layer_idx_re.search(key)
            if lm and int(lm.group(1)) >= limit_layers:
                continue
        quant_keys.add(key)

    layers = sorted(
        {int(m.group(1)) for k in quant_keys if (m := layer_idx_re.search(k))}
    )
    report = ExportReport(arch="dense", out_dir=str(out_dir), layers=list(layers))

    if verbose:
        print(
            f"[dense-export] quantizable Linear weights={len(quant_keys)} "
            f"layers={len(layers)} (attention={include_attention} "
            f"linear_attn={include_linear_attn} vision={include_vision}) "
            f"src={source_format} block_scale_rule={block_scale_rule}"
        )

    if dry_run:
        report.quantized_tensors = len(quant_keys)
        return report

    with _SecureExportContext(pathlib.Path(model_path), pathlib.Path(out_dir)) as ctx:
        writer = _ShardWriter(ctx, max_shard_bytes=max_shard_bytes)
        exclude_patterns: set[str] = set()
        error_stats = _ErrorStats() if report_error else None

        for key in model.keys():
            if key in quant_keys:
                name = key[: -len(".weight")]
                w = model.get_tensor(key).to(device, torch.bfloat16)
                quantized = _emit_quantized_linear(
                    writer, report, name, w, source_format=source_format,
                    use_gpu=use_gpu, block_scale_rule=block_scale_rule,
                    error_stats=error_stats,
                )
                if not quantized:
                    # Copied through as BF16 -> must be excluded from the quant config.
                    exclude_patterns.add(_module_pattern(key))
                del w
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                writer.add(key, model.get_tensor(key))
                report.copied_tensors += 1
                exclude_patterns.add(_module_pattern(key))

        report.shards = writer.finalize()
        report.total_bytes = writer.total_bytes
        quant_config = build_quantization_config(
            source_format=source_format,
            exclude_modules=sorted(exclude_patterns),
            block_scale_rule=block_scale_rule,
        )
        _write_config(model.config, ctx, quant_config)
        ctx.copy_aux_files(ctx._src_dir_fd)
        out = ctx.out_path

    if error_stats is not None:
        report.error_groups = error_stats.summary()
        if verbose:
            error_stats.print_table()
    if verbose:
        print(
            f"[dense-export] done: quantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def dequantize_fp6_checkpoint_to_bf16(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    device: str = "cuda",
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
    verbose: bool = True,
) -> ExportReport:
    """Decode an FP6 checkpoint back to a plain BF16 HF checkpoint.

    Inverse of the FP6 exporters: every quantized linear (the four-key
    ``.weight``/``.weight_scale``/``.weight_scale_2``/``.input_scale`` set) is
    dequantized to a single BF16 ``.weight``; everything else is copied through.
    The output runs on stock vLLM with no b12x involvement, so a KLD
    against the original BF16 model isolates *weight* quantization error from
    the runtime W6A6 *activation* quantization error.
    """
    model = SafetensorsModel(model_path)
    qcfg = model.config.get("quantization_config") or {}
    if qcfg.get("quant_algo") != QUANT_ALGO:
        raise ValueError(
            f"{model_path} is not a b12x FP6 checkpoint "
            f"(quant_algo={qcfg.get('quant_algo')!r})"
        )
    fmt = qcfg.get("weight_format", "e2m3")

    quantized = {
        k[: -len(".weight_scale")]
        for k in model.keys()
        if k.endswith(".weight_scale") and not k.endswith(".weight_scale_2")
    }
    report = ExportReport(arch="dequant-bf16", out_dir=str(out_dir))
    if verbose:
        print(
            f"[fp6-dequant] quantized linears={len(quantized)} fmt={fmt} "
            f"-> BF16 {out_dir}"
        )

    with _SecureExportContext(pathlib.Path(model_path), pathlib.Path(out_dir)) as ctx:
        writer = _ShardWriter(ctx, max_shard_bytes=max_shard_bytes)
        drop_suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")

        for key in model.keys():
            base = key
            for suffix in drop_suffixes:
                if key.endswith(suffix):
                    base = key[: -len(suffix)]
                    break
            if base != key and base in quantized:
                continue  # consumed by the dequantized .weight emission
            if key.endswith(".weight") and key[: -len(".weight")] in quantized:
                name = key[: -len(".weight")]
                packed = model.get_tensor(key).to(device)
                scale = model.get_tensor(name + ".weight_scale").to(device)
                ws2_key = name + ".weight_scale_2"
                ws2 = model.get_tensor(ws2_key).to(device) if model.has(ws2_key) else None
                w = dequantize_linear_from_fp6(
                    packed, scale, fmt=fmt, weight_scale_2=ws2
                )
                writer.add(key, w.to(torch.bfloat16))
                report.quantized_tensors += 1
                del packed, scale, w
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                writer.add(key, model.get_tensor(key))
                report.copied_tensors += 1

        report.shards = writer.finalize()
        report.total_bytes = writer.total_bytes
        cfg = dict(model.config)
        cfg.pop("quantization_config", None)
        ctx.write_text("config.json", json.dumps(cfg, indent=2))
        ctx.copy_aux_files(ctx._src_dir_fd)
        out = ctx.out_path

    if verbose:
        print(
            f"[fp6-dequant] done: dequantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def export_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    arch: str = "auto",
    **kwargs,
) -> ExportReport:
    """Dispatch to the MoE or dense exporter (``arch`` ``"auto"``/``"moe"``/``"dense"``)."""
    if arch == "auto":
        model = SafetensorsModel(model_path)
        arch = "moe" if discover_moe_experts(model) is not None else "dense"
    if arch == "moe":
        return export_moe_model_to_fp6_safetensors(model_path, out_dir, **kwargs)
    if arch == "dense":
        return export_dense_model_to_fp6_safetensors(model_path, out_dir, **kwargs)
    raise ValueError(f"unknown arch {arch!r} (expected auto/moe/dense)")
