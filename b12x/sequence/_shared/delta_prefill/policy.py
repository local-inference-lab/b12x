"""Launch configuration shared by the delta-rule prefill policies."""
from __future__ import annotations

from dataclasses import dataclass
from b12x.policy.types import FrozenMapping
from .workspace import BACKEND

@dataclass(frozen=True)
class PrefillConfig:
    """Backend selection plus the recurrence kernel's launch geometry.

    ``v_split`` is the number of value rows one recurrence CTA owns (smaller
    splits launch more CTAs per sequence and head at the cost of re-reading
    the prepared tiles from L2); ``k_split`` is how many warps share each
    sixteen-row group by splitting its key columns (more warps shorten the
    per-tile tensor-core chain at the cost of shared-memory reductions);
    ``stages`` is the tile prefetch depth; ``window_tiles`` is the number of
    consecutive banded tile positions one pipeline window covers. The prepare
    kernel of a window runs concurrently with the recurrence of that window
    and the next window's prepare, and two windows of prepared records form
    the workspace ring, so the window size bounds the ring's footprint.
    """

    backend: str = BACKEND
    v_split: int = 64
    k_split: int = 1
    stages: int = 3
    window_tiles: int = 64

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "PrefillConfig":
        keys = set(payload.keys())
        if "backend" not in keys or not keys <= {
            "backend", "v_split", "k_split", "stages", "window_tiles"
        }:
            raise ValueError(
                "Delta-rule prefill profiles require backend and accept only v_split, "
                "k_split, stages, and window_tiles"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        values = {}
        for name, default in (("v_split", 64), ("k_split", 1), ("stages", 3), ("window_tiles", 64)):
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            values[name] = int(value)
        return cls(backend=backend, **values)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "v_split": int(self.v_split),
            "k_split": int(self.k_split),
            "stages": int(self.stages),
            "window_tiles": int(self.window_tiles),
        }
