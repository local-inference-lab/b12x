"""JIT binding for the B12X-owned K6/MCG small-M CUDA kernel."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import secrets
import stat
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

import torch


_SOURCE_DIR = Path(__file__).resolve().parent / "csrc"
_SOURCE = _SOURCE_DIR / "trellis_k6_small.cu"
_VENDORED_FILES = tuple(sorted((_SOURCE_DIR / "vendor").rglob("*.[ch]*")))


_GLM_K6_DECODE_SMS = {
    # Q/indexer projection on the target stream.
    (2048, 4096): 128,
    # TP4 shared-expert FC1/FC2 run beside the target stream. These are the
    # rank-local dimensions after column/row parallel slicing, not the full
    # 4096/2048-wide shared MLP dimensions. The budgets match the E2E-optimal
    # ExLlama autotuner result; using all 188 SMs serializes the graph branches.
    (6144, 1024): 64,
    (512, 6144): 96,
}


def _default_num_sms(size_k: int, size_n: int, available_sms: int) -> int:
    """Select the measured GLM K6 decode overlap budget when applicable."""
    target = _GLM_K6_DECODE_SMS.get((size_k, size_n))
    return available_sms if target is None else min(available_sms, target)


@lru_cache(maxsize=None)
def _available_sms(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _extension_name() -> str:
    digest = hashlib.sha256()
    for path in (_SOURCE, *_VENDORED_FILES):
        if path.is_file():
            digest.update(path.relative_to(_SOURCE_DIR).as_posix().encode())
            digest.update(path.read_bytes())
    return f"b12x_trellis_k6_{digest.hexdigest()[:12]}"


def _trellis_build_dir() -> str:
    """Return the exact normalized absolute path of a secure B12x-owned dir.

    Never inherits an unchecked ``TORCH_EXTENSIONS_DIR``.  The default
    lives under the same per-user cache root as the compile cache.
    """
    env_dir = os.environ.get("B12X_TRELLIS_BUILD_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        cache_root = os.environ.get("B12X_COMPILE_CACHE_DIR")
        if cache_root:
            path = Path(cache_root).parent / "trellis_build"
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            if xdg:
                path = Path(xdg) / "b12x" / "trellis_build"
            else:
                path = Path.home() / ".cache" / "b12x" / "trellis_build"
    return _validate_secure_build_dir(path)


def _ancestor_is_root_sticky(st: os.stat_result) -> bool:
    """Return whether a shared ancestor has root-owned sticky isolation."""
    return st.st_uid == 0 and bool(st.st_mode & stat.S_ISVTX)


def _validate_secure_build_dir(path: Path) -> str:
    """Walk ancestors from ``/`` with ``O_DIRECTORY|O_NOFOLLOW`` and fstat.

    Ancestors must be euid/root-owned and not group/world-writable.
    The final directory must be euid-owned with no group/other bits
    (mode ``0700``).  Missing components are created with ``0700``.

    Returns the exact normalized absolute path string.
    """
    abs_path = Path(os.path.abspath(str(path)))
    parts = [p for p in abs_path.parts if p not in ("", "/")]
    if not parts:
        raise RuntimeError("trellis build dir must not be /")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i, part in enumerate(parts):
            is_final = i == len(parts) - 1
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
            except FileNotFoundError:
                created = False
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(fd)
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
            os.close(fd)
            fd = next_fd
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode):
                raise RuntimeError(
                    f"trellis build path component {part!r} is not a directory"
                )
            if is_final:
                if st.st_uid != os.geteuid():
                    raise RuntimeError(
                        f"trellis build dir {abs_path} owned by uid {st.st_uid}, "
                        f"not euid {os.geteuid()}"
                    )
                if st.st_mode & 0o077:
                    raise RuntimeError(
                        f"trellis build dir {abs_path} has group/other bits; "
                        f"refusing non-private build dir"
                    )
            else:
                if st.st_uid != os.geteuid() and st.st_uid != 0:
                    raise RuntimeError(
                        f"trellis build ancestor {part!r} owned by uid {st.st_uid}"
                    )
                root_sticky = _ancestor_is_root_sticky(st)
                if st.st_mode & stat.S_IWOTH and not root_sticky:
                    raise RuntimeError(
                        f"trellis build ancestor {part!r} is world-writable"
                    )
                if st.st_mode & stat.S_IWGRP and not root_sticky:
                    raise RuntimeError(
                        f"trellis build ancestor {part!r} is group-writable"
                    )
    finally:
        with suppress(OSError):
            os.close(fd)
    return str(abs_path)

def _open_and_validate_so(dir_fd: int, so_name: str) -> int:
    """Open a .so file relative to *dir_fd* with O_NOFOLLOW|O_NONBLOCK,
    fstat-validate as euid-owned regular nlink==1, and return the fd.

    Does NOT check mode here — the linker may emit 0755; we fchmod
    to 0600 before any import.
    """
    fd = os.open(
        so_name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        dir_fd=dir_fd,
    )
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(
                f"trellis .so {so_name} is not a regular file"
            )
        if st.st_uid != os.geteuid():
            raise RuntimeError(
                f"trellis .so {so_name} owned by uid {st.st_uid}"
            )
        if st.st_nlink != 1:
            raise RuntimeError(
                f"trellis .so {so_name} has {st.st_nlink} hard links"
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _find_and_validate_so(dir_fd: int, ext_name: str) -> tuple[int, str]:
    """Find exactly one produced .so in *dir_fd*, validate, return (fd, name).

    Accepts ``<ext_name>.so`` or ``<ext_name>_v<N>.so`` (versioned).
    Fails closed if no or multiple .so files found.
    """
    matches = [
        e for e in os.listdir(dir_fd)
        if e.startswith(ext_name) and e.endswith(".so")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 .so for {ext_name}, found {matches}"
        )


    so_name = matches[0]
    fd = _open_and_validate_so(dir_fd, so_name)
    return fd, so_name


def _load_python_extension_from_fd(name: str, fd: int):
    """Execute a CPython extension from the already validated open inode."""
    inode_path = f"/dev/fd/{fd}"
    loader = importlib.machinery.ExtensionFileLoader(name, inode_path)
    spec = importlib.util.spec_from_file_location(name, inode_path, loader=loader)
    if spec is None:
        raise ImportError(f"cannot create extension spec for {name}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _extension():
    """Build and load the Trellis K6 extension with full integrity checks.

    Trust contract: verifies integrity (ownership, permissions, link
    count, no-follow) but not provenance.  A same-euid writer can
    produce a self-consistent malicious .so; defend with a trusted
    producer/signature or process isolation.
    """
    from torch.utils.cpp_extension import (
        _write_ninja_file_and_build_library,
        FileBaton,
    )
    from torch.utils.cpp_extension import _is_cuda_file

    build_base = _trellis_build_dir()
    generation = f".gen_{secrets.token_hex(8)}"
    build_directory = _validate_secure_build_dir(
        Path(os.path.join(build_base, generation))
    )
    ext_name = _extension_name()
    verbose = os.environ.get("B12X_JIT_VERBOSE", "0") == "1"
    cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    cflags = ["-O3"]
    # Build-only: call _write_ninja_file_and_build_library directly
    # to avoid _jit_compile's _import_module_from_library which executes
    # the .so before we can validate it.
    with_cuda = any(_is_cuda_file(s) for s in [str(_SOURCE)])
    baton = FileBaton(os.path.join(build_directory, "lock"))
    if baton.try_acquire():
        try:
            old_umask = os.umask(0o077)
            try:
                _write_ninja_file_and_build_library(
                    name=ext_name,
                    sources=[str(_SOURCE)],
                    extra_cflags=cflags,
                    extra_cuda_cflags=cuda_cflags,
                    extra_sycl_cflags=[],
                    extra_ldflags=[],
                    extra_include_paths=[str(_SOURCE_DIR)],
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    with_sycl=False,
                )
            finally:
                os.umask(old_umask)
        finally:
            baton.release()
    else:
        baton.wait()

    # Retain the generation dirfd and open the produced .so
    # descriptor-relative with no-follow/nlink checks.
    gen_fd = _validate_secure_build_dir_fd(Path(build_directory))
    try:
        so_fd, so_name = _find_and_validate_so(gen_fd, ext_name)
        try:
            # Normalize mode to 0600 before any import.
            os.fchmod(so_fd, 0o600)
            # Verify mode after fchmod.
            st = os.fstat(so_fd)
            if st.st_mode & 0o077:
                raise RuntimeError(
                    f"trellis .so {so_name} mode not 0600 after fchmod"
                )
            # Execute PyInit_<ext_name> through CPython's extension loader
            # while the validated inode fd remains open. This source exports
            # PYBIND11_MODULE, not TORCH_LIBRARY operators.
            module = _load_python_extension_from_fd(ext_name, so_fd)
        finally:
            os.close(so_fd)
    finally:
        os.close(gen_fd)
    return module


def _validate_secure_build_dir_fd(path: Path) -> int:
    """Like _validate_secure_build_dir but returns the retained fd."""
    abs_path = Path(os.path.abspath(str(path)))
    parts = [p for p in abs_path.parts if p not in ("", "/")]
    if not parts:
        raise RuntimeError("trellis build dir must not be /")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i, part in enumerate(parts):
            is_final = i == len(parts) - 1
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
            except FileNotFoundError:
                created = False
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(fd)
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
            os.close(fd)
            fd = next_fd
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode):
                raise RuntimeError(
                    f"trellis build component {part!r} not a directory"
                )
            if is_final:
                if st.st_uid != os.geteuid():
                    raise RuntimeError(
                        f"trellis build dir owned by uid {st.st_uid}"
                    )
                if st.st_mode & 0o077:
                    raise RuntimeError(
                        "trellis build dir has group/other bits"
                    )
            else:
                if st.st_uid != os.geteuid() and st.st_uid != 0:
                    raise RuntimeError(
                        f"trellis ancestor {part!r} owned by uid {st.st_uid}"
                    )
                root_sticky = _ancestor_is_root_sticky(st)
                if st.st_mode & stat.S_IWOTH and not root_sticky:
                    raise RuntimeError(
                        f"trellis ancestor {part!r} is world-writable"
                    )
                if st.st_mode & stat.S_IWGRP and not root_sticky:
                    raise RuntimeError(
                        f"trellis ancestor {part!r} is group-writable"
                    )
        return fd
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        raise

def run_k6_mcg(
    x: torch.Tensor,
    trellis: torch.Tensor,
    output: torch.Tensor,
    suh: torch.Tensor,
    rotated_input: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    *,
    num_sms: int = 0,
) -> None:
    """Launch the capture-safe K6/MCG kernel on Torch's current stream."""
    capability = torch.cuda.get_device_capability(x.device)
    if capability != (12, 0):
        raise NotImplementedError(
            "Trellis K6 small-M kernel is built for sm_120 only; "
            f"device reports sm_{capability[0]}{capability[1]}"
        )
    if num_sms <= 0:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        num_sms = _default_num_sms(
            int(x.shape[1]),
            int(output.shape[1]),
            _available_sms(int(device_index)),
        )
    _extension().launch_k6_mcg(
        x,
        trellis,
        output,
        suh,
        rotated_input,
        svh,
        locks,
        int(num_sms),
    )


__all__ = ["run_k6_mcg"]
