"""Survive candidate kernels that poison the CUDA context.

A sticky CUDA error (illegal address, launch failure) ends the generating
process. Before each candidate launch the session records it as in flight;
when the command is rerun on the same work directory, an in-flight marker left
behind is promoted to the crashed list and that candidate is skipped with an
explicit error instead of being raced again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_INFLIGHT_PREFIX = "inflight-candidate"
_CRASHED = "crashed-candidates.json"

CRASHED_ERROR = "crashed the CUDA context in an earlier run"


def load_crashed(work_dir: Path) -> set[tuple[str, str]]:
    path = Path(work_dir) / _CRASHED
    if not path.exists():
        return set()
    entries = json.loads(path.read_text())
    return {(str(item["case_id"]), str(item["candidate_id"])) for item in entries}


def _inflight_path(work_dir: Path, worker: int | None = None) -> Path:
    # One marker per measuring process: parallel workers share the work dir.
    pid = os.getpid() if worker is None else int(worker)
    return Path(work_dir) / f"{_INFLIGHT_PREFIX}-{pid}.json"


def mark_inflight(
    work_dir: Path, case_id: str, candidate_id: str, *, worker: int | None = None
) -> None:
    path = _inflight_path(work_dir, worker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case_id": case_id, "candidate_id": candidate_id}))


def clear_inflight(work_dir: Path, *, worker: int | None = None) -> None:
    path = _inflight_path(work_dir, worker)
    if path.exists():
        path.unlink()


def promote_inflight(
    work_dir: Path, *, worker: int | None = None
) -> tuple[str, str] | None:
    """Move leftover in-flight markers onto the crashed list.

    Promotes every marker left behind by a dead run, or only ``worker``'s own
    marker when a live worker blames itself. Returns the last promoted
    (case_id, candidate_id), or ``None`` when nothing was in flight.
    """
    promoted: tuple[str, str] | None = None
    if worker is None:
        paths = sorted(Path(work_dir).glob(f"{_INFLIGHT_PREFIX}*.json"))
    else:
        paths = [_inflight_path(work_dir, worker)]
    for path in paths:
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        crashed_path = Path(work_dir) / _CRASHED
        entries = json.loads(crashed_path.read_text()) if crashed_path.exists() else []
        entries.append(
            {"case_id": entry["case_id"], "candidate_id": entry["candidate_id"]}
        )
        crashed_path.write_text(json.dumps(entries, indent=1))
        path.unlink()
        promoted = (str(entry["case_id"]), str(entry["candidate_id"]))
    return promoted


__all__ = [
    "CRASHED_ERROR",
    "clear_inflight",
    "load_crashed",
    "mark_inflight",
    "promote_inflight",
]
