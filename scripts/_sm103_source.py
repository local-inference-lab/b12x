"""Record checkout or archive provenance without requiring Git metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


def package_source_sha256(root: Path) -> str:
    sources = sorted((root / "b12x").rglob("*.py"))
    if not sources:
        raise ValueError(f"no b12x package sources found in {root}")
    return hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()


def source_identity(root: Path) -> dict[str, object]:
    """Identify this source root; never inherit an enclosing checkout's commit.

    Archive revisions describe the exported base. Actual package hashes remain
    authoritative for the files being compiled or tested, including edits made
    after extraction. An archive has no Git working-tree status.
    """
    root = root.resolve()

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    identity: dict[str, object] = {
        "source_revision": None,
        "source_kind": "directory",
        "git_status": None,
        "worktree": str(root),
        "source_identity_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() == root:
            identity.update(
                source_revision=git("rev-parse", "HEAD"),
                source_kind="git",
                git_status=git("status", "--porcelain").splitlines(),
            )
            return identity
    except (OSError, subprocess.CalledProcessError):
        pass
    archival = root / ".git_archival.txt"
    if archival.is_file():
        revision = archival.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            identity.update(source_revision=revision, source_kind="archive")
    return identity
