#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def _open_dir_fd(path, dir_fd: Optional[int] = None) -> int:
    """Open a directory file descriptor with O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC.

    O_NOFOLLOW rejects a *final-component* symlink (raises ELOOP).
    Intermediate path components are not directly controlled here, but
    their risk is bounded by the caller: the parent is opened and
    verified (owned by euid, not group/world-writable) before the leaf
    is created or pinned relative to it.
    """
    flags = os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise RuntimeError(f"refusing to follow symlink: {path}") from exc
        raise


def _verify_parent_fd(fd: int, parent) -> None:
    """Verify a pinned parent directory fd is owned by euid and not group/world-writable."""
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"parent is not a directory: {parent}")
    if st.st_uid != os.geteuid():
        raise RuntimeError(
            f"parent not owned by current user: {parent} "
            f"(use a home or private parent directory, or omit --out-dir)"
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(
            f"parent is group or world writable: {parent} "
            f"(use a home or private parent directory, or omit --out-dir)"
        )


def _verify_leaf_fd(fd: int, path) -> None:
    """Verify a pinned leaf directory fd is a directory, owned by euid, mode 0700."""
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"output is not a directory: {path}")
    if st.st_uid != os.geteuid():
        raise RuntimeError(f"output not owned by current user: {path}")
    if stat.S_IMODE(st.st_mode) != 0o700:
        raise RuntimeError(f"output directory mode is {oct(stat.S_IMODE(st.st_mode))}, expected 0700: {path}")


def _prepare_output_dir(out_arg: Optional[str], prefix: str) -> tuple[Path, int]:
    """Return (path, pinned_dir_fd) for a private, mode-0700 output directory.

    When *out_arg* is ``None`` an unpredictable directory is created via
    :func:`tempfile.mkdtemp`, then pinned and verified.

    When *out_arg* is provided the parent directory must already exist, be
    owned by the current user, and not be group/world-writable.  The leaf
    must not already exist (no directory, file, or symlink).  It is created
    with mode 0700 via the pinned parent fd, then pinned and verified.
    """
    if out_arg is None:
        path = Path(tempfile.mkdtemp(prefix=prefix))
        fd = _open_dir_fd(path)
    else:
        path = Path(out_arg).absolute()
        # Leaf must not exist or be a symlink.
        if path.is_symlink() or path.exists():
            raise RuntimeError(f"refusing to use existing path as output directory: {path}")
        parent = path.parent
        if not parent.is_dir():
            raise RuntimeError(f"parent directory does not exist: {parent}")
        parent_fd = _open_dir_fd(parent)
        try:
            _verify_parent_fd(parent_fd, parent)
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
            # Pin the newly created leaf via the parent fd before closing it,
            # so the fd refers to the exact inode we just created, not a
            # path that could be swapped before reopening.
            fd = _open_dir_fd(path.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)

    # Enforce mode 0700 and verify the pinned leaf.
    try:
        os.fchmod(fd, 0o700)
        _verify_leaf_fd(fd, path)
    except Exception:
        os.close(fd)
        raise
    return path, fd


def _secure_open(basename: str, dir_fd: int):
    """Open a new file for writing relative to *dir_fd* with exclusive, no-follow, mode-0600 semantics.

    The file must not already exist; symlinks are never followed.  The raw
    fd is fchmod'd to exactly 0600 (undoing any umask) and then wrapped via
    :func:`os.fdopen`.  The raw fd is closed if fdopen fails.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(basename, flags, 0o600, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ELOOP):
            raise RuntimeError(
                f"refusing to overwrite existing file or follow symlink: {basename}"
            ) from exc
        raise
    try:
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def _secure_write_text(basename: str, dir_fd: int, content: str) -> None:
    """Write *content* to a brand-new file relative to *dir_fd* with exclusive, no-follow, mode-0600 semantics."""
    with _secure_open(basename, dir_fd) as f:
        f.write(content)


class StreamRequest:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        log_basename: str,
        error_basename: str,
        dir_fd: int,
        timeout_s: int = 3600,
    ) -> None:
        self.url = url
        self.headers = headers
        self.payload = payload
        self.log_basename = log_basename
        self.error_basename = error_basename
        self._caller_fd = dir_fd
        self._owned_fd: Optional[int] = None
        self.timeout_s = timeout_s
        self.first_token_event = threading.Event()
        self.finished_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._response = None
        self.error: Optional[str] = None

    def start(self) -> None:
        # Dup the caller's dir_fd so the stream owns an independent directory
        # capability that survives even if main() closes its fd while the
        # daemon thread is still running.
        self._owned_fd = os.dup(self._caller_fd)
        os.set_inheritable(self._owned_fd, False)
        try:
            self._thread.start()
        except Exception:
            os.close(self._owned_fd)
            self._owned_fd = None
            raise

    def stop(self) -> None:
        self._stop_event.set()
        if self._response is not None:
            with contextlib.suppress(Exception):
                self._response.close()
        self._thread.join(timeout=5)

    def wait_for_first_token(self, timeout_s: float) -> bool:
        return self.first_token_event.wait(timeout_s)

    def _run(self) -> None:
        data = json.dumps(self.payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self._response = response
                with _secure_open(self.log_basename, self._owned_fd) as log_file:
                    while not self._stop_event.is_set():
                        line = response.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="replace")
                        log_file.write(text)
                        log_file.flush()
                        if text.startswith("data: ") and text.strip() != "data: [DONE]":
                            self.first_token_event.set()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                _secure_write_text(self.error_basename, self._owned_fd, self.error + "\n")
            except Exception as write_exc:
                self.error += f" (additionally failed to write error log: {write_exc})"
        finally:
            os.close(self._owned_fd)
            self.finished_event.set()


def http_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: Optional[dict[str, Any]] = None,
    timeout_s: int = 3600,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def resolve_model(base_url: str, headers: dict[str, str]) -> str:
    status, body = http_json(f"{base_url}/v1/models", headers=headers, payload=None)
    if status != 200:
        raise RuntimeError(f"/v1/models failed with status {status}: {body.strip()}")
    payload = json.loads(body)
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no models")
    return models[0]["id"]


def build_request(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }


def build_profile_request(
    *,
    mode: str,
    server_output_dir: Optional[str],
    num_steps: int,
    include_cpu: bool,
) -> dict[str, Any]:
    activities = ["GPU"]
    if include_cpu:
        activities = ["CPU", "GPU"]
    request = {
        "num_steps": num_steps,
        "activities": activities,
        "profile_by_stage": True,
        "profile_prefix": mode,
        "profile_stages": ["decode"] if mode == "decode" else ["prefill"],
    }
    if server_output_dir is not None:
        request["output_dir"] = server_output_dir
    return request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach to an already-running SGLang server, start one streaming "
            "request, wait for the first token, then trigger SGLang's native profiler."
        )
    )
    parser.add_argument("--base-url", required=True, help="Server base URL, for example http://127.0.0.1:30000")
    parser.add_argument("--mode", choices=("decode", "mtp"), default="decode")
    parser.add_argument("--model", default=None, help="Model id. Auto-detected from /v1/models when omitted.")
    parser.add_argument("--prompt", default=None, help="Inline prompt. Ignored when --prompt-file is set.")
    parser.add_argument("--prompt-file", default=None, help="Read the prompt from a file.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Local output directory for logs and metadata. "
            "The parent directory must already exist, be owned by you, "
            "and not be group/world-writable.  The directory itself is "
            "created fresh with mode 0700; an existing path is rejected."
        ),
    )
    parser.add_argument(
        "--server-output-dir",
        default=None,
        help=(
            "Explicit server-visible path for native traces. When omitted, the "
            "server uses its configured output directory. This path is resolved "
            "by the already-running server and is a separate trusted server-side "
            "contract; the client cannot pin or secure it."
        ),
    )
    parser.add_argument("--api-key", default=None, help="Optional bearer token. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--include-cpu", action="store_true", help="Profile CPU + GPU instead of GPU only.")
    parser.add_argument("--first-token-timeout", type=float, default=60.0)
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return (
        "Repeat the exact word token separated by spaces. "
        "Do not stop early. Keep emitting tokens until the token limit is reached."
    )


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    out_dir, dir_fd = _prepare_output_dir(args.out_dir, f"sglang-native-{args.mode}-")
    server_output_dir = args.server_output_dir
    try:
        prompt = load_prompt(args)
        model = args.model or resolve_model(base_url, headers)
        request_body = build_request(model, prompt, args.max_tokens)
        profile_body = build_profile_request(
            mode=args.mode,
            server_output_dir=server_output_dir,
            num_steps=args.num_steps,
            include_cpu=args.include_cpu,
        )

        _secure_write_text("request.json", dir_fd, json.dumps(request_body, indent=2) + "\n")
        _secure_write_text("start_profile.json", dir_fd, json.dumps(profile_body, indent=2) + "\n")

        print(f"output dir: {out_dir}")
        print(f"model: {model}")
        print(f"mode: {args.mode}")
        print("starting streaming request...")

        stream = StreamRequest(
            url=f"{base_url}/v1/completions",
            headers=headers,
            payload=request_body,
            log_basename="stream.log",
            error_basename="request.error.txt",
            dir_fd=dir_fd,
        )
        stream.start()
        started_at = time.time()

        if not stream.wait_for_first_token(args.first_token_timeout):
            stream.stop()
            detail = ""
            if stream.error:
                detail = f" request error: {stream.error}"
            raise RuntimeError(f"timed out waiting for the first streamed token.{detail}")

        first_token_at = time.time()
        print("first streamed token observed; starting native profile...")

        status, body = http_json(
            f"{base_url}/start_profile",
            headers=headers,
            payload=profile_body,
            timeout_s=3600,
        )
        _secure_write_text("profile.response", dir_fd, body)
        if status != 200:
            stream.stop()
            raise RuntimeError(f"/start_profile failed with status {status}: {body.strip()}")

        finished_at = time.time()
        stream.stop()

        manifest = {
            "base_url": base_url,
            "mode": args.mode,
            "model": model,
            "out_dir": str(out_dir),
            "server_output_dir": server_output_dir,
            "started_at_epoch_s": started_at,
            "first_token_at_epoch_s": first_token_at,
            "profile_finished_at_epoch_s": finished_at,
            "request_body": request_body,
            "profile_body": profile_body,
            "notes": [
                "The background request was started before profiling and profiling began after the first streamed token.",
                "For speculative requests, native SGLang profiling buckets TARGET_VERIFY and DRAFT_EXTEND under the prefill-family stage.",
                "On older SGLang profiler paths, profile_stages is ignored and profiling still buckets into prefill-family vs decode.",
            ],
        }
        _secure_write_text("manifest.json", dir_fd, json.dumps(manifest, indent=2) + "\n")

        print("native profiling completed.")
        print(f"stream log: {out_dir / 'stream.log'}")
        print(f"manifest: {out_dir / 'manifest.json'}")
    finally:
        os.close(dir_fd)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
