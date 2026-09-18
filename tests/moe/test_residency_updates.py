"""Fault-injected host transactions and explicit preparation admission."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._residency_updates import _SlotUpdates, ResidencyUpdateError
from b12x.moe.fused_moe._residency_storage import accounting, tier_layout
from tests.moe.test_expert_residency import declaration


class HostTransfer:
    def __init__(self, fail=0, persistent=False):
        self.calls = 0
        self.fail = fail
        self.persistent = persistent
        self.capture = False

    def capturing(self): return self.capture
    def synchronize(self): pass

    def copy(self, destination, source):
        self.calls += 1
        destination.copy_(source)
        if self.calls == self.fail or (self.persistent and self.calls >= self.fail):
            raise OSError("injected failure after a destination write")


def fixture(*, transfer=None):
    tiers = tuple(SimpleNamespace(fields={name: torch.arange(2*size, dtype=torch.uint8).reshape(2, size)+tier*40+i*10
        for i, (name, size) in enumerate((("w13", 8), ("w2", 4), ("s13", 2), ("s2", 1)))}) for tier in (0, 1))
    mapping = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.int32)
    journal = {name: torch.empty(4, value.shape[1], dtype=value.dtype) for name, value in tiers[0].fields.items()}
    updates = _SlotUpdates(tiers=tiers, mapping=mapping, expert_map=mapping.tolist(), journal=journal,
        before=torch.empty_like(mapping), after=torch.empty_like(mapping), transfer=transfer or HostTransfer())
    return updates


def payloads(updates):
    return {e: {name: value[row].clone() for name, value in updates.tiers[tier].fields.items()}
            for e, (tier, row) in enumerate(updates.snapshot().expert_map)}


def assert_payloads(updates, original):
    for e, fields in payloads(updates).items():
        for name, value in fields.items():
            torch.testing.assert_close(value, original[e][name], rtol=0, atol=0)


def test_batch_exchange_preserves_identity_and_same_tier_permutations():
    u = fixture()
    original, start = payloads(u), u.snapshot()
    for pairs in (((0, 1), (2, 3)), ((0, 2),), ((0, 1), (2, 3))):
        previous = u.snapshot()
        result = u.exchange(pairs, expected=previous, quiescent=True)
        assert result.generation == previous.generation+1
        assert tuple(map(tuple, u.mapping.tolist())) == result.expert_map
        assert_payloads(u, original)
    with pytest.raises(ValueError, match="stale"):
        u.exchange(((0, 1),), expected=start, quiescent=True)
    with pytest.raises(ValueError, match="stale"):
        u.exchange(((0, 1),), expected=fixture().snapshot(), quiescent=True)


@pytest.mark.parametrize("fail", [1, 2, 8, 17, 18, 25, 33, 34])
def test_failure_at_staging_payload_or_publication_restores_entire_batch(fail):
    u = fixture(transfer=HostTransfer(fail))
    original, snapshot = payloads(u), u.snapshot()
    with pytest.raises(ResidencyUpdateError) as caught:
        u.exchange(((0, 1), (2, 3)), expected=snapshot, quiescent=True)
    assert caught.value.resumable
    assert u.snapshot() == snapshot and tuple(map(tuple, u.mapping.tolist())) == snapshot.expert_map
    assert_payloads(u, original)
    u.exchange(((0, 1),), expected=snapshot, quiescent=True)
    assert_payloads(u, original)


def test_failed_rollback_and_external_map_mutation_poison_state():
    for u in (fixture(transfer=HostTransfer(18, persistent=True)), fixture()):
        snapshot = u.snapshot()
        if not u.transfer.persistent:
            u.mapping[0, 1] = 999
        with pytest.raises(ResidencyUpdateError) as caught:
            u.exchange(((0, 1), (2, 3)), expected=snapshot, quiescent=True)
        assert not caught.value.resumable and not u.snapshot().healthy
        with pytest.raises(RuntimeError, match="unavailable"):
            u.exchange(((0, 1),), expected=snapshot, quiescent=True)


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5])
def test_completion_failure_restores_the_previous_generation(boundary):
    class CompletionFailure(HostTransfer):
        completions = 0

        def synchronize(self):
            self.completions += 1
            if self.completions == boundary:
                raise OSError("injected completion failure")

    u = fixture(transfer=CompletionFailure())
    original, snapshot = payloads(u), u.snapshot()
    with pytest.raises(ResidencyUpdateError) as caught:
        u.exchange(((0, 1), (2, 3)), expected=snapshot, quiescent=True)
    assert caught.value.resumable and u.snapshot() == snapshot
    assert tuple(map(tuple, u.mapping.tolist())) == snapshot.expert_map
    assert_payloads(u, original)


@pytest.mark.parametrize("pairs", [(), ((0, 0),), ((0, 1), (1, 2)), ((-1, 1),),
    ((0, 2**40),), ((True, 1),), ((0, 1, 2),), ((0, 1), (2, 3), (0, 3))])
def test_invalid_transactions_write_nothing(pairs):
    u = fixture()
    with pytest.raises(ValueError): u.exchange(pairs, expected=u.snapshot(), quiescent=True)
    assert u.transfer.calls == 0


def test_pause_and_capture_guards():
    u = fixture()
    with pytest.raises(ValueError, match="paused"):
        u.exchange(((0, 1),), expected=u.snapshot())
    u.transfer.capture = True
    with pytest.raises(RuntimeError, match="capture"):
        u.exchange(((0, 1),), expected=u.snapshot(), quiescent=True)
    assert u.transfer.calls == 0


def test_updates_are_opt_in_pure_and_fully_budgeted(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("declaration initialized CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    static, _ = declaration()
    assert accounting(static.query).update_host_bytes == 0
    plan, _ = declaration(updates=moe.ResidencyUpdateCapacity(max_pairs=2))
    memory = accounting(plan.query)
    assert memory.update_host_bytes == tier_layout(4, 256, 256)[1]+512
    assert memory.hbm_total_bytes == accounting(static.query).hbm_total_bytes
    assert plan.contract.query_schema_version == 2 and plan.prepared is None
    budget = moe.ExpertMemoryBudget(hbm_bytes=memory.hbm_total_bytes, grace_bytes=memory.grace_total_bytes)
    budget.admit(memory)
    with pytest.raises(ValueError, match="Grace"):
        replace(budget, grace_bytes=budget.grace_bytes-1).admit(memory)
    with pytest.raises(ValueError, match="capacity"):
        declaration(updates=moe.ResidencyUpdateCapacity(max_pairs=3))
    with pytest.raises((ValueError, RuntimeError)):
        moe.residency_slot_snapshot(plan)


def test_payload_field_drift_cannot_omit_rollback_metadata():
    u = fixture()
    u.tiers[0].fields["bias"] = torch.zeros((2, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="field sets"):
        _SlotUpdates(tiers=u.tiers, mapping=u.mapping, expert_map=u.snapshot().expert_map,
            journal=u.journal, before=u.before, after=u.after, transfer=HostTransfer())
