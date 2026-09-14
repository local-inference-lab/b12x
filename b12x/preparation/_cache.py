"""Versioned tuning decisions, independent of executable artifact caches."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .types import FrozenMapping


def _json(value):
    if isinstance(value, FrozenMapping):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def cache_identity(namespace: Mapping[str, object], device_ordinal: int):
    import torch
    from b12x._lib.compiler import _device_uuid_key

    raw_version = os.environ.get("B12X_TUNING_CACHE_VERSION", "1")
    try:
        version = int(raw_version)
    except ValueError:
        raise ValueError("B12X_TUNING_CACHE_VERSION must be a positive integer") from None
    if version <= 0:
        raise ValueError("B12X_TUNING_CACHE_VERSION must be a positive integer")
    with torch.cuda.device(device_ordinal):
        uuid = _device_uuid_key(device_ordinal)
        if uuid is None:
            raise RuntimeError("preparation requires a resolved physical CUDA device")
        return {
            "schema_version": 5, "tuning_cache_version": version,
            "namespace": dict(namespace), "device": uuid,
            "visible_ordinal": device_ordinal,
        }


class SelectionCache:
    """Locked reload/merge/atomic publication prevents concurrent lost updates."""

    def __init__(self, root: str | Path, identity: Mapping[str, object]):
        self.identity = json.loads(_json(identity))
        if self.identity.get("schema_version") != 5:
            raise ValueError("preparation selection cache requires schema 5")
        version = self.identity.get("tuning_cache_version")
        if type(version) is not int or version <= 0:
            raise ValueError("selection cache requires a positive tuning_cache_version")
        self.path = Path(root) / f"{digest(self.identity)}.json"
        self.records = self._read()

    def _validate(self, records):
        if not isinstance(records, dict):
            raise ValueError("selection records must be a mapping")
        for key, record in records.items():
            if not isinstance(key, str) or not isinstance(record, dict):
                raise ValueError("malformed selection record")
            if set(record) != {"assignment", "config", "coverage", "programs"}:
                raise ValueError("selection record fields differ from schema")
            FrozenMapping(record["assignment"])
            FrozenMapping(record["config"])
            coverage = record["coverage"]
            fields = {"cartesian_count", "legal_count", "effective_count", "measured_count"}
            if not isinstance(coverage, dict) or set(coverage) != fields:
                raise ValueError("selection coverage fields differ from schema")
            if any(type(value) is not int or value < 0 for value in coverage.values()):
                raise ValueError("selection counts must be nonnegative integers")
            if not (
                coverage["cartesian_count"] >= coverage["legal_count"]
                >= coverage["effective_count"] == coverage["measured_count"] > 1
            ):
                raise ValueError("only completed exhaustive multi-candidate races are cacheable")
            programs = record["programs"]
            if not isinstance(programs, list):
                raise ValueError("selection program dependencies must be a list")
            for program in programs:
                if (not isinstance(program, list) or len(program) != 2
                        or program[0] not in ("cute", "triton")
                        or not isinstance(program[1], str) or not program[1]):
                    raise ValueError("invalid selected program identity")
        return records

    def _read(self):
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text())
        if not isinstance(payload, dict) or set(payload) != {"identity", "records"}:
            raise ValueError("invalid preparation selection cache")
        if payload["identity"] != self.identity:
            raise ValueError("selection identity differs from its cache key")
        return self._validate(payload["records"])

    def get(self, key):
        record = self.records.get(key)
        return None if record is None else FrozenMapping(record)

    def save(self, key, *, assignment, config, coverage, programs):
        update = {
            key: {
                "assignment": FrozenMapping(assignment).to_dict(),
                "config": FrozenMapping(config).to_dict(),
                "coverage": dict(coverage),
                "programs": [list(item) for item in sorted({
                    (program.dialect, program.key) for program in programs
                })],
            }
        }
        self._validate(update)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self._read()
            records.update(update)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.path.parent, suffix=".tmp", delete=False,
                ) as stream:
                    temporary = stream.name
                    stream.write(_json({"identity": self.identity, "records": records}))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            self.records = records
