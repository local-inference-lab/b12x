"""Quiescent, journaled exchanges of fixed physical expert rows.

This control-plane mechanism requires the engine to stop every producer using
these slabs, including raw CUDA graph replay. It is not a concurrent cache.
"""
from ..residency.contracts import ResidencySlotSnapshot, ResidencyUpdateError
import math
from threading import Lock
from uuid import uuid4

import torch

from ._residency_storage import align, tier_layout, update_host_bytes


class _CudaTransfer:
    def __init__(self, device):
        self.device = device

    def capturing(self):
        with torch.cuda.device(self.device):
            return torch.cuda.is_current_stream_capturing()

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def copy(self, destination, source):
        from cuda.bindings import runtime as cuda
        with torch.cuda.device(self.device):
            # Default infers mapped-host vs device pointers. No GPU kernel or
            # temporary device tensor is resolved by control-plane copies.
            error, = cuda.cudaMemcpyAsync(destination.data_ptr(), source.data_ptr(),
                source.numel()*source.element_size(), cuda.cudaMemcpyKind.cudaMemcpyDefault,
                torch.cuda.current_stream(self.device).cuda_stream)
            if error != cuda.cudaError_t.cudaSuccess:
                raise RuntimeError(f"expert slot copy failed: {error}")


class _SlotUpdates:
    def __init__(self, *, tiers, mapping, expert_map, journal, before, after, transfer, owner=None):
        self.tiers, self.mapping, self.journal = tiers, mapping, journal
        self.before, self.after, self.transfer, self.owner = before, after, transfer, owner
        self._map = tuple(tuple(row) for row in expert_map)
        self._identity = uuid4().hex
        self._generation = 0
        self._healthy = True
        self._busy = False
        self._lock = Lock()
        self.max_pairs = next(iter(journal.values())).shape[0] // 2
        # Every payload field participates; adding a field without its rollback
        # region must fail preparation rather than permit partial exchanges.
        for tier in tiers:
            if tier is None:
                continue
            if set(tier.fields) != set(journal):
                raise ValueError("slot payload and rollback field sets differ")
            for name, value in tier.fields.items():
                if value.dtype != journal[name].dtype or value.shape[1:] != journal[name].shape[1:]:
                    raise ValueError("slot payload and rollback geometry differ")

    def require_healthy(self):
        if not self._healthy or self._busy:
            raise RuntimeError("residency slots are unavailable; keep all graph producers paused")

    def snapshot(self):
        with self._lock:
            return ResidencySlotSnapshot(preparation_id=self._identity, generation=self._generation, expert_map=self._map,
                                         healthy=self._healthy)

    def exchange(self, pairs, *, expected, quiescent=False):
        if quiescent is not True:
            raise ValueError("expert exchange requires paused graph producers (quiescent=True)")
        if self.transfer.capturing():
            raise RuntimeError("expert exchange cannot run during CUDA graph capture")
        if not isinstance(expected, ResidencySlotSnapshot):
            raise TypeError("exchange requires an expected ResidencySlotSnapshot")
        pairs = tuple(tuple(pair) for pair in pairs)
        if not pairs or len(pairs) > self.max_pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("exchange requires one or more pairs within prepared capacity")
        ids = tuple(e for pair in pairs for e in pair)
        if any(type(e) is not int or not 0 <= e < len(self._map) for e in ids) or len(set(ids)) != len(ids):
            raise ValueError("exchange requires distinct valid canonical expert IDs")
        with self._lock:
            self.require_healthy()
            if (expected.preparation_id != self._identity or expected.generation != self._generation
                    or expected.expert_map != self._map or not expected.healthy):
                raise ValueError("stale expert residency preparation or generation")
            self._busy = True
            modified = False
            journal_ready = False
            try:
                # Stop submissions first, then drain every stream on this device.
                self.transfer.synchronize()
                self.transfer.copy(self.before, self.mapping)
                self.transfer.synchronize()
                if tuple(tuple(row) for row in self.before.tolist()) != self._map:
                    self._healthy = False
                    raise RuntimeError("device residency map differs from its authoritative generation")
                self.after.copy_(self.before)
                locations = [self._map[e] for e in ids]
                for n, (tier, row) in enumerate(locations):
                    for name, field in self.tiers[tier].fields.items():
                        self.transfer.copy(self.journal[name][n], field[row])
                self.transfer.synchronize()
                journal_ready = True
                next_map = list(self._map)
                for left, right in pairs:
                    next_map[left], next_map[right] = next_map[right], next_map[left]
                    for expert in (left, right):
                        self.after[expert, 0], self.after[expert, 1] = next_map[expert]
                # Mark before the first write: a copy may enqueue successfully
                # and then report failure. Its destination still needs rollback.
                modified = True
                for n, (tier, row) in enumerate(locations):
                    for name, field in self.tiers[tier].fields.items():
                        self.transfer.copy(field[row], self.journal[name][n ^ 1])
                self.transfer.synchronize()
                self.transfer.copy(self.mapping, self.after)
                self.transfer.synchronize()
                self._map = tuple(next_map)
                self._generation += 1
            except BaseException as error:
                try:
                    # Drain even a failed staging attempt before its journal can
                    # be reused; a CUDA failure here makes resumption unsafe.
                    self.transfer.synchronize()
                    if modified and journal_ready:
                        for n, (tier, row) in enumerate(locations):
                            for name, field in self.tiers[tier].fields.items():
                                self.transfer.copy(field[row], self.journal[name][n])
                        self.transfer.synchronize()
                        self.transfer.copy(self.mapping, self.before)
                        self.transfer.synchronize()
                    self._map = expected.expert_map
                    self._generation = expected.generation
                except BaseException as rollback_error:
                    self._healthy = False
                    raise ResidencyUpdateError("expert exchange rollback failed; stop the lane and reload",
                        resumable=False) from rollback_error
                raise ResidencyUpdateError("expert exchange failed; " +
                    ("original placement restored" if self._healthy else "slot state is untrusted; reload required"),
                    resumable=self._healthy) from error
            finally:
                self._busy = False
            return ResidencySlotSnapshot(preparation_id=self._identity, generation=self._generation, expert_map=self._map, healthy=True)


def materialize_updates(query, tiers, mapping, placement, device):
    if not query.max_swap_pairs:
        return None
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    owner = MappedHostAllocation((update_host_bytes(query),), torch.uint8, device)
    try:
        fields, size = tier_layout(2*query.max_swap_pairs, query.hidden, query.intermediate)
        journal = {name: owner.host_view[offset:offset+math.prod(shape)].view(shape)
                   for name, offset, shape in fields}
        map_bytes = query.experts*8
        before = owner.host_view[size:size+map_bytes].view(torch.int32).view(query.experts, 2)
        size += align(map_bytes)
        after = owner.host_view[size:size+map_bytes].view(torch.int32).view(query.experts, 2)
        return _SlotUpdates(tiers=tiers, mapping=mapping, expert_map=placement.expert_map,
            journal=journal, before=before, after=after, transfer=_CudaTransfer(device), owner=owner)
    except BaseException:
        owner.close()
        raise


def _updates(plan):
    from b12x.preparation import require_prepared
    state = require_prepared(plan, "moe.expert_residency")
    if state.updates is None:
        raise ValueError("expert exchanges were not declared before preparation")
    return state.updates


def residency_slot_snapshot(plan):
    """Read host generation metadata; no synchronization or device transfer."""
    return _updates(plan).snapshot()


def exchange_expert_slots(plan, pairs, *, expected, quiescent=False):
    """Exchange disjoint canonical expert pairs at an engine-owned pause.

    A cross-tier pair promotes one expert and evicts the other. Same-tier pairs
    permute rows. Tier capacities, tensor descriptors and captured addresses
    stay fixed. The caller must not resume raw graphs after a nonresumable error.
    """
    return _updates(plan).exchange(pairs, expected=expected, quiescent=quiescent)
