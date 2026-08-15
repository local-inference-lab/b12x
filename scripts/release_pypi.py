#!/usr/bin/env python3
"""release_pypi.py — credential-owning PyPI release launcher.

This module is the security boundary for issue #170.  It is invoked by
scripts/release_pypi.sh (a thin bash wrapper).  A shell body cannot
safely bootstrap credential handling because Bash startup hooks,
imported functions, and spoofed SHELLOPTS execute before the script
body.  This Python launcher owns all credential handling:

  * Parses .env safely in Python (no shell execution possible).
  * Scrubs credentials from the environment before any subprocess.
  * Runs venv/pip/build/twine-check with a credential-free environment.
  * Runs twine upload with an allowlisted environment containing only
    the credential and a fixed PyPI endpoint.
  * Handles signals by forwarding to the upload child and reaping.
  * Removes pre-existing dist/ content before build.

Byte-level .env grammar: BOM is stripped if present; NUL/control bytes
in credential lines cause abort; malformed credential-looking lines
abort generically with no fallback to inherited or prior values.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT_DIR / ".venv-release"
VENV_PYTHON = str(VENV_DIR / "bin" / "python")
DIST_DIR = ROOT_DIR / "dist"

# Allowlisted environment keys for the upload child.  Only these are
# passed to twine upload — nothing else from the inherited environment.
_UPLOAD_ENV_KEYS = frozenset({
    "LANG",
    "LC_ALL",
})

# Credential keys we recognize in .env.
_CRED_KEYS = frozenset({"TWINE_USERNAME", "TWINE_PASSWORD"})

# BOM marker (UTF-8).
_BOM = b"\xef\xbb\xbf"
_PYPI_UPLOAD_URL = "https://upload.pypi.org/legacy/"


class EnvError(Exception):
    """Raised for malformed .env content."""


# ---------------------------------------------------------------------------
# .env parser — byte-level, non-evaluating, fail-closed for credentials.
# ---------------------------------------------------------------------------
def _parse_env_value(raw: str) -> str:
    """Parse a .env value without evaluating it.

    Grammar:
      unquoted         value                         # inline comment
      single-quoted    'value with # literal'        # comment
      double-quoted    "value with # literal"        # comment
      empty            (empty)                       # or # comment

    Returns the parsed value.  Raises EnvError on malformed input.
    """
    i = 0
    length = len(raw)

    # Skip leading whitespace.
    while i < length and raw[i] in " \t":
        i += 1

    # Empty value or comment-only.
    if i >= length or raw[i] == "#":
        return ""

    ch = raw[i]

    if ch == "'":
        # Single-quoted: literal until closing quote.
        i += 1
        start = i
        while i < length and raw[i] != "'":
            i += 1
        if i >= length:
            raise EnvError("unterminated single quote")
        result = raw[start:i]
        i += 1
        # After closing quote: only whitespace and optional comment.
        while i < length:
            c = raw[i]
            if c in " \t":
                i += 1
            elif c == "#":
                break
            else:
                raise EnvError("trailing content after single quote")
        return result

    if ch == '"':
        # Double-quoted: literal (no expansion) until closing quote.
        i += 1
        start = i
        while i < length and raw[i] != '"':
            i += 1
        if i >= length:
            raise EnvError("unterminated double quote")
        result = raw[start:i]
        i += 1
        while i < length:
            c = raw[i]
            if c in " \t":
                i += 1
            elif c == "#":
                break
            else:
                raise EnvError("trailing content after double quote")
        return result

    # Unquoted: read until whitespace, then only comment allowed.
    # A quote character in an unquoted value is malformed.
    start = i
    while i < length:
        c = raw[i]
        if c in " \t":
            break
        if c in "'\"":
            raise EnvError("embedded quote in unquoted value")
        i += 1
    result = raw[start:i]
    while i < length:
        c = raw[i]
        if c in " \t":
            i += 1
        elif c == "#":
            break
        else:
            raise EnvError("trailing content after unquoted value")
    return result


def _is_credential_key(key: str) -> bool:
    """Check if a key is exactly TWINE_USERNAME or TWINE_PASSWORD."""
    return key in _CRED_KEYS


def _looks_like_credential_line(line: str) -> bool:
    """Detect a credential-looking line even if malformed.

    After optional 'export ' prefix, the line starts with exactly
    TWINE_USERNAME or TWINE_PASSWORD followed by any non-identifier
    character (or end of line).  This catches +=, [0], #typo, missing =,
    whitespace before =, etc.  But NOT TWINE_PASSWORD_BACKUP.
    """
    stripped = line
    # Strip optional 'export ' prefix.
    if stripped.startswith("export ") or stripped.startswith("export\t"):
        stripped = stripped[6:].lstrip(" \t")
    for key in _CRED_KEYS:
        if stripped.startswith(key):
            rest = stripped[len(key):]
            # Next char must be non-identifier (not alnum or _) or EOL.
            if not rest or rest[0] not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789_"
            ):
                return True
    return False


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse .env file byte-level.  Returns dict of credential values.

    For credential keys: malformed lines abort (EnvError).
    For unknown keys: silently ignored.
    """
    data = path.read_bytes()

    # Strip exactly one UTF-8 BOM if present at the start.
    if data.startswith(_BOM):
        data = data[len(_BOM):]

    # Decode as UTF-8; reject invalid bytes.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise EnvError("invalid UTF-8 in .env") from None

    # Reject NUL bytes.
    if "\x00" in text:
        raise EnvError("NUL byte in .env")

    credentials: dict[str, str] = {}

    for line in text.splitlines():
        # Strip CR (CRLF compatibility).
        if line.endswith("\r"):
            line = line[:-1]
        # Trim leading/trailing whitespace.
        line = line.strip()
        # Skip blank and comment lines.
        if not line or line.startswith("#"):
            continue

        # Reject control bytes in ANY line (before credential detection).
        if any(ord(c) < 0x20 and c not in "\t" for c in line):
            raise EnvError("control byte in .env line")

        # Check for credential-looking lines BEFORE generic skipping.
        if _looks_like_credential_line(line):
            # Must contain '='.
            if "=" not in line:
                raise EnvError("malformed credential line")
            key, _, raw_val = line.partition("=")
            # Strip optional 'export ' prefix from key.
            if key.startswith("export ") or key.startswith("export\t"):
                key = key[6:].lstrip(" \t")
            # Key must be exactly TWINE_USERNAME or TWINE_PASSWORD.
            if key not in _CRED_KEYS:
                raise EnvError("malformed credential key")
            # Verify the line is exactly KEY=... (not +=, [0]=, etc.).
            stripped = line
            if stripped.startswith("export ") or stripped.startswith("export\t"):
                stripped = stripped[6:].lstrip(" \t")
            if not stripped.startswith(f"{key}="):
                raise EnvError("malformed credential operator")
            # Parse value.
            try:
                value = _parse_env_value(raw_val)
            except EnvError:
                raise EnvError("malformed credential value") from None
            credentials[key] = value
            continue

        # Non-credential lines: only parse if they look like KEY=VALUE.
        if "=" not in line:
            continue
        key, _, _ = line.partition("=")
        if key.startswith("export ") or key.startswith("export\t"):
            key = key[6:].lstrip(" \t")
        # Unknown keys are silently ignored.

    return credentials


# ---------------------------------------------------------------------------
# Environment management
# ---------------------------------------------------------------------------
def _scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Remove all credential and injection vectors from env."""
    scrubbed = dict(env)
    # Remove credentials.
    for k in list(scrubbed):
        if k.startswith("TWINE_") or k.startswith("RELEASE_TWINE_"):
            del scrubbed[k]
    # Remove internal scratch names.
    for k in ("_env_line", "_env_key", "_env_raw", "_env_val",
              "_env_stripped", "_release_upload_pid", "_release_upload_pgid"):
        scrubbed.pop(k, None)
    # Remove interpreter, shell, package-manager, and dynamic-loader injection
    # vectors. Prefix matching covers platform/version-specific loader knobs.
    injection_prefixes = (
        "BASH_FUNC_",
        "DYLD_",
        "LD_",
        "PYTHON",
        "PIP_",
        "TWINE_",
    )
    for k in list(scrubbed):
        if k.startswith(injection_prefixes):
            del scrubbed[k]
    for k in (
        "BASH_ENV",
        "ENV",
        "PS4",
        "BASH_XTRACEFD",
        "SHELLOPTS",
        "BASHOPTS",
    ):
        scrubbed.pop(k, None)
    return scrubbed


def _build_upload_env(
    twine_username: str, twine_password: str,
    base_env: dict[str, str],
) -> dict[str, str]:
    """Build an allowlisted environment for the upload child."""
    upload_env: dict[str, str] = {}
    for key in _UPLOAD_ENV_KEYS:
        if key in base_env:
            upload_env[key] = base_env[key]
    upload_env["TWINE_USERNAME"] = twine_username
    upload_env["TWINE_PASSWORD"] = twine_password
    return upload_env


# ---------------------------------------------------------------------------
# Signal handling for upload
# ---------------------------------------------------------------------------
_upload_proc: subprocess.Popen | None = None


def _forward_signal(signum: int, _frame: object) -> None:
    """Forward a signal to the entire upload process group and reap it."""
    global _upload_proc
    try:
        if _upload_proc is not None and _upload_proc.poll() is None:
            try:
                os.killpg(_upload_proc.pid, signum)
                _upload_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(_upload_proc.pid, signal.SIGKILL)
                _upload_proc.wait(timeout=1)
    finally:
        os._exit(128 + signum)


def _install_signal_handlers() -> None:
    """Install signal handlers that forward to the upload child."""
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _forward_signal)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    os.chdir(ROOT_DIR)

    # -------------------------------------------------------------------
    # Phase 1: Resolve credentials.
    # -------------------------------------------------------------------
    twine_username = os.environ.get("TWINE_USERNAME", "")
    twine_password = os.environ.get("TWINE_PASSWORD", "")

    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        try:
            env_creds = parse_dotenv(env_path)
        except EnvError as e:
            print(f"Malformed .env: {e}", file=sys.stderr)
            return 1
        # .env overrides environment.
        if "TWINE_USERNAME" in env_creds:
            twine_username = env_creds["TWINE_USERNAME"]
        if "TWINE_PASSWORD" in env_creds:
            twine_password = env_creds["TWINE_PASSWORD"]

    if not twine_username:
        twine_username = "__token__"
    if not twine_password:
        print("TWINE_PASSWORD must be set in .env or the environment",
              file=sys.stderr)
        return 1

    # -------------------------------------------------------------------
    # Phase 2: Build a credential-free environment for pre-upload steps.
    # -------------------------------------------------------------------
    clean_env = _scrub_env(dict(os.environ))

    # -------------------------------------------------------------------
    # Phase 3: Remove pre-existing dist/.
    # -------------------------------------------------------------------
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    # -------------------------------------------------------------------
    # Phase 4: Create venv if needed (credential-free).
    # -------------------------------------------------------------------
    if not (VENV_DIR / "bin" / "python").exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            env=clean_env, check=True,
        )

    # -------------------------------------------------------------------
    # Phase 5: Install build tools and build (credential-free).
    # -------------------------------------------------------------------
    subprocess.run(
        [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"],
        env=clean_env, check=True,
    )
    subprocess.run(
        [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "build", "twine"],
        env=clean_env, check=True,
    )
    subprocess.run(
        [VENV_PYTHON, "-m", "build"],
        cwd=str(ROOT_DIR), env=clean_env, check=True,
    )
    dist_files = sorted(DIST_DIR.glob("*"))
    subprocess.run(
        [VENV_PYTHON, "-m", "twine", "check", *map(str, dist_files)],
        cwd=str(ROOT_DIR), env=clean_env, check=True,
    )

    # -------------------------------------------------------------------
    # Phase 6: Upload with allowlisted environment.
    # -------------------------------------------------------------------
    upload_env = _build_upload_env(twine_username, twine_password, clean_env)

    _install_signal_handlers()

    global _upload_proc
    _upload_proc = subprocess.Popen(
        [
            VENV_PYTHON,
            "-m",
            "twine",
            "upload",
            "--non-interactive",
            "--repository-url",
            _PYPI_UPLOAD_URL,
            *map(str, dist_files),
        ],
        cwd=str(ROOT_DIR),
        env=upload_env,
        start_new_session=True,
    )
    exit_code = _upload_proc.wait()
    _upload_proc = None

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
