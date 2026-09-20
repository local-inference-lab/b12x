"""Quiescent batch fills with canonical victim recovery."""

from threading import Lock
from uuid import uuid4
from time import perf_counter_ns

import torch

from b12x.moe.residency import (
    ResidencySlotSnapshot,
    ResidencyUpdateError,
    updated_slot_map,
)


class CanonicalSlotUpdates:
    """Commit disjoint fills together; restore all victims on partial failure.

    The serving engine must stop submissions and drain every reader before
    calling apply. A failed restore poisons this preparation and requires lane
    reload. Canonical backing is immutable and contains every logical expert.
    """

    def __init__(
        self,
        *,
        resident,
        canonical,
        mapping,
        expert_map,
        before,
        after,
        transfer,
        max_pairs,
    ):
        self.resident, self.canonical, self.mapping = resident, canonical, mapping
        self.before, self.after, self.transfer = before, after, transfer
        self.max_pairs = max_pairs
        self._map = tuple(tuple(row) for row in expert_map)
        self._identity, self._generation, self._healthy = uuid4().hex, 0, True
        self._lock = Lock()
        updated_slot_map(self._snapshot(), (), backing_mode="canonical")
        self._resident = tuple(
            {n: v[r] for n, v in resident.items()}
            for r in range(sum(t == 0 for t, _ in self._map))
        )
        self._canonical = tuple(
            {n: v[e] for n, v in canonical.items()} for e in range(len(self._map))
        )
        if set(resident) != set(canonical) or mapping.shape != (len(self._map), 2):
            raise ValueError("canonical fill payload or map geometry differs")
        for name, value in canonical.items():
            if (
                value.device.type != "cpu"
                or value.dtype != torch.uint8
                or not value.is_contiguous()
                or value.shape[0] != len(self._map)
                or resident[name].shape != (len(self._resident), *value.shape[1:])
            ):
                raise ValueError(
                    "canonical fills require complete contiguous CPU byte backing"
                )

    def _snapshot(self):
        return ResidencySlotSnapshot(
            preparation_id=self._identity,
            generation=self._generation,
            expert_map=self._map,
            healthy=self._healthy,
        )

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def require_healthy(self):
        if not self._healthy:
            raise RuntimeError("expert cache is unavailable; reload the lane")

    def apply(self, pairs, *, expected, quiescent=False):
        if quiescent is not True:
            raise ValueError("canonical fill requires paused graph producers")
        if self.transfer.capturing():
            raise RuntimeError("canonical fill cannot execute during graph capture")
        pairs = tuple(pairs)
        with self._lock:
            self.require_healthy()
            if expected != self._snapshot():
                raise ValueError("stale canonical cache preparation or generation")
            if not 0 < len(pairs) <= self.max_pairs:
                raise ValueError("fill batch exceeds prepared pair capacity")
            next_map = updated_slot_map(expected, pairs, backing_mode="canonical")
            overwritten = []
            started = mark = perf_counter_ns()
            self.last_timings_ns = {}

            def stamp(name):
                nonlocal mark
                now = perf_counter_ns()
                self.last_timings_ns[name] = now - mark
                mark = now

            try:
                self.transfer.synchronize()
                stamp("initial_drain")
                self.transfer.copy(self.before, self.mapping)
                stamp("map_d2h_enqueue")
                self.transfer.synchronize()
                stamp("map_d2h_completion")
                if tuple(map(tuple, self.before.tolist())) != self._map:
                    self._healthy = False
                    raise RuntimeError(
                        "device map differs from authoritative generation"
                    )
                self.after.copy_(torch.tensor(next_map, dtype=torch.int32))
                stamp("map_validation_and_encoding")
                for candidate, victim in pairs:
                    slot = self._map[victim][1]
                    overwritten.append((slot, victim))
                    for name, value in self._canonical[candidate].items():
                        self.transfer.copy(self._resident[slot][name], value)
                stamp("payload_enqueue")
                self.transfer.synchronize()
                stamp("payload_completion")
                self.transfer.copy(self.mapping, self.after)
                stamp("map_publication_enqueue")
                self.transfer.synchronize()
                stamp("map_publication_completion")
                self._map = next_map
                self._generation += 1
            except BaseException as error:
                try:
                    self.transfer.synchronize()
                    for slot, victim in overwritten:
                        for name, value in self._canonical[victim].items():
                            self.transfer.copy(self._resident[slot][name], value)
                    if overwritten:
                        self.transfer.synchronize()
                        self.transfer.copy(self.mapping, self.before)
                        self.transfer.synchronize()
                except BaseException as recovery:
                    self._healthy = False
                    raise ResidencyUpdateError(
                        "canonical fill recovery failed; reload every rank",
                        resumable=False,
                    ) from recovery
                raise ResidencyUpdateError(
                    "canonical fill failed; victims restored"
                    if self._healthy
                    else "canonical map diverged; reload every rank",
                    resumable=self._healthy,
                ) from error
            self.last_timings_ns["total"] = perf_counter_ns() - started
            return self._snapshot()
