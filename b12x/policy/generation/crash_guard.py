"""Survive candidate kernels that poison the CUDA context.

A sticky CUDA error (illegal address, launch failure) ends the generating
process. Before each candidate launch the session records it as in flight;
when the command is rerun on the same work directory, an in-flight marker left
behind is promoted to the crashed list and that candidate is skipped with an
explicit error instead of being raced again.
"""

from __future__ import annotations

import json
from pathlib import Path

_INFLIGHT = "inflight-candidate.json"
_CRASHED = "crashed-candidates.json"

CRASHED_ERROR = "crashed the CUDA context in an earlier run"


def load_crashed(work_dir: Path) -> set[tuple[str, str]]:
    path = Path(work_dir) / _CRASHED
    if not path.exists():
        return set()
    entries = json.loads(path.read_text())
    return {(str(item["case_id"]), str(item["candidate_id"])) for item in entries}


def mark_inflight(work_dir: Path, case_id: str, candidate_id: str) -> None:
    path = Path(work_dir) / _INFLIGHT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case_id": case_id, "candidate_id": candidate_id}))


def clear_inflight(work_dir: Path) -> None:
    path = Path(work_dir) / _INFLIGHT
    if path.exists():
        path.unlink()


def promote_inflight(work_dir: Path) -> tuple[str, str] | None:
    """Move a leftover in-flight marker onto the crashed list; return it."""
    path = Path(work_dir) / _INFLIGHT
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    crashed_path = Path(work_dir) / _CRASHED
    entries = json.loads(crashed_path.read_text()) if crashed_path.exists() else []
    entries.append({"case_id": entry["case_id"], "candidate_id": entry["candidate_id"]})
    crashed_path.write_text(json.dumps(entries, indent=1))
    path.unlink()
    return (str(entry["case_id"]), str(entry["candidate_id"]))


__all__ = [
    "CRASHED_ERROR",
    "clear_inflight",
    "load_crashed",
    "mark_inflight",
    "promote_inflight",
]
