"""Research-only quiescent fills with immutable canonical backing.

Every logical expert retains its original row in the backing tier. A successful
fill changes only one resident row and the fixed-address map. Failure after an
overwrite restores the victim from canonical backing before restoring the map.
No engine integration, concurrent replacement or production cache API is implied.
"""

from threading import Lock
from uuid import uuid4

import torch

from b12x.moe.residency import ResidencySlotSnapshot, ResidencyUpdateError


class CanonicalFills:
    """Single-pair experiment; owners must keep all backing bytes immutable.

    Backing row E always contains canonical expert E, including hot experts.
    CPU views alias the exact prepared backing bytes, verified before capture.
    Separate pageable source views must match those bytes at construction and
    remain immutable. Staging is one reusable, preallocated expert payload.
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
        source=None,
        staging=None,
    ):
        self.resident, self.canonical, self.mapping = resident, canonical, mapping
        self.before, self.after, self.transfer = before, after, transfer
        self.source = canonical if source is None else source
        self.staging = staging
        self._map = tuple(tuple(row) for row in expert_map)
        self._identity, self._generation, self._healthy = uuid4().hex, 0, True
        self._lock = Lock()
        if mapping.dtype != torch.int32 or mapping.shape != (len(self._map), 2):
            raise ValueError("fill mapping must be E x 2 int32")
        hot = sorted(row for tier, row in self._map if tier == 0)
        if hot != list(range(len(hot))) or any(
            t not in (0, 1) or (t == 1 and r != e) for e, (t, r) in enumerate(self._map)
        ):
            raise ValueError(
                "canonical backing row must equal expert ID; resident rows must be dense"
            )
        for fields in (
            resident,
            self.source,
            staging if staging is not None else canonical,
        ):
            if set(fields) != set(canonical):
                raise ValueError("canonical payload field sets differ")
        for name, value in canonical.items():
            if (
                value.device.type != "cpu"
                or value.dtype != torch.uint8
                or not value.is_contiguous()
                or value.shape[0] != len(self._map)
            ):
                raise ValueError(
                    "canonical backing requires contiguous CPU byte views for every expert"
                )
            for fields, rows in ((resident, len(hot)), (self.source, len(self._map))):
                v = fields[name]
                if (
                    v.shape != (rows, *value.shape[1:])
                    or v.dtype != value.dtype
                    or not v.is_contiguous()
                ):
                    raise ValueError("canonical and resident payload geometry differs")
            if not torch.equal(value, self.source[name]):
                raise ValueError("fill source differs from verified canonical bytes")
            if staging is not None and (
                staging[name].shape != value.shape[1:]
                or staging[name].dtype != value.dtype
            ):
                raise ValueError("staging row differs from payload")
        # Retained views avoid reconfiguration of payload descriptors at exchange time.
        self._resident_rows = [
            {n: f[r] for n, f in resident.items()} for r in range(len(hot))
        ]
        self._canonical_rows = [
            {n: f[e] for n, f in canonical.items()} for e in range(len(self._map))
        ]
        self._source_rows = [
            {n: f[e] for n, f in self.source.items()} for e in range(len(self._map))
        ]

    def snapshot(self):
        with self._lock:
            return ResidencySlotSnapshot(
                preparation_id=self._identity,
                generation=self._generation,
                expert_map=self._map,
                healthy=self._healthy,
            )

    def promote(self, candidate, victim, *, expected, quiescent=False):
        if quiescent is not True:
            raise ValueError("fill requires paused graph producers (quiescent=True)")
        if self.transfer.capturing():
            raise RuntimeError("fill cannot execute during graph capture")
        if (
            any(
                type(e) is not int or not 0 <= e < len(self._map)
                for e in (candidate, victim)
            )
            or candidate == victim
        ):
            raise ValueError("fill requires distinct valid canonical expert IDs")
        with self._lock:
            if not self._healthy:
                raise RuntimeError("fill state is unavailable; reload the lane")
            actual = ResidencySlotSnapshot(
                preparation_id=self._identity,
                generation=self._generation,
                expert_map=self._map,
                healthy=True,
            )
            if expected != actual:
                raise ValueError("stale canonical fill preparation or generation")
            if self._map[candidate] != (1, candidate) or self._map[victim][0] != 0:
                raise ValueError(
                    "fill requires a backing candidate and resident victim"
                )
            slot = self._map[victim][1]
            modified = False
            try:
                self.transfer.synchronize()
                self.transfer.copy(self.before, self.mapping)
                self.transfer.synchronize()
                if tuple(map(tuple, self.before.tolist())) != self._map:
                    self._healthy = False
                    raise RuntimeError(
                        "device map differs from authoritative generation"
                    )
                next_map = list(self._map)
                next_map[candidate] = (0, slot)
                next_map[victim] = (1, victim)
                self.after.copy_(self.before)
                self.after[candidate, 0], self.after[candidate, 1] = next_map[candidate]
                self.after[victim, 0], self.after[victim, 1] = next_map[victim]
                fields = self._source_rows[candidate]
                if self.staging is not None:
                    for name, value in fields.items():
                        self.transfer.copy(self.staging[name], value)
                    self.transfer.synchronize()
                    fields = self.staging
                modified = (
                    True  # A submitted copy may fail after modifying its destination.
                )
                for name, value in fields.items():
                    self.transfer.copy(self._resident_rows[slot][name], value)
                self.transfer.synchronize()
                self.transfer.copy(self.mapping, self.after)
                self.transfer.synchronize()
                self._map = tuple(next_map)
                self._generation += 1
            except BaseException as error:
                try:
                    self.transfer.synchronize()
                    if modified:
                        for name, value in self._canonical_rows[victim].items():
                            self.transfer.copy(self._resident_rows[slot][name], value)
                        self.transfer.synchronize()
                        self.transfer.copy(self.mapping, self.before)
                        self.transfer.synchronize()
                except BaseException as recovery:
                    self._healthy = False
                    raise ResidencyUpdateError(
                        "canonical victim recovery failed; reload the lane",
                        resumable=False,
                    ) from recovery
                raise ResidencyUpdateError(
                    "canonical fill failed; "
                    + (
                        "victim and map restored"
                        if self._healthy
                        else "map untrusted; reload required"
                    ),
                    resumable=self._healthy,
                ) from error
            return ResidencySlotSnapshot(
                preparation_id=self._identity,
                generation=self._generation,
                expert_map=self._map,
                healthy=True,
            )
