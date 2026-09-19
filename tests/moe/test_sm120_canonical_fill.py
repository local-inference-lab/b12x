"""Canonical victim recovery and native same-graph fill qualification."""

import pytest
import torch

from b12x.moe.residency import ResidencyUpdateError
from benchmarks.moe.sm120_canonical_fill import CanonicalFills
from tests.moe.test_residency_updates import HostTransfer


def fixture(transfer=None, staged=False):
    canonical = {
        name: torch.arange(4 * n, dtype=torch.uint8).view(4, n)
        for name, n in [("weights", 16), ("scales", 4)]
    }
    resident = {name: value[:2].clone() for name, value in canonical.items()}
    mapping = torch.tensor([[0, 0], [0, 1], [1, 2], [1, 3]], dtype=torch.int32)
    return CanonicalFills(
        resident=resident,
        canonical=canonical,
        mapping=mapping,
        expert_map=mapping.tolist(),
        before=torch.empty_like(mapping),
        after=torch.empty_like(mapping),
        transfer=transfer or HostTransfer(),
        staging={name: torch.empty_like(value[0]) for name, value in canonical.items()}
        if staged
        else None,
    )


def check(u):
    assert tuple(map(tuple, u.mapping.tolist())) == u.snapshot().expert_map
    for e, (tier, row) in enumerate(u.snapshot().expert_map):
        for name, value in u.canonical.items():
            actual = u.resident[name][row] if tier == 0 else value[row]
            torch.testing.assert_close(actual, value[e], atol=0, rtol=0)


@pytest.mark.parametrize("staged", [False, True])
def test_repeated_promotion_eviction_and_stale_generation(staged):
    u = fixture(staged=staged)
    ptrs = tuple(
        x.data_ptr() for x in (*u.resident.values(), *u.canonical.values(), u.mapping)
    )
    original = u.snapshot()
    for candidate, victim in [(2, 0), (3, 1), (0, 2), (1, 3), (2, 0)]:
        start = u.snapshot()
        assert (
            u.promote(candidate, victim, expected=start, quiescent=True).generation
            == start.generation + 1
        )
        check(u)
    assert ptrs == tuple(
        x.data_ptr() for x in (*u.resident.values(), *u.canonical.values(), u.mapping)
    )
    with pytest.raises(ValueError, match="stale"):
        u.promote(0, 2, expected=original, quiescent=True)
    with pytest.raises(ValueError, match="paused"):
        u.promote(0, 2, expected=u.snapshot())


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("failure", [1, 2, 3, 4])
def test_partial_write_and_publication_restore_victim(failure, staged):
    u = fixture(HostTransfer(fail=failure), staged=staged)
    start = u.snapshot()
    with pytest.raises(ResidencyUpdateError) as caught:
        u.promote(2, 0, expected=start, quiescent=True)
    assert caught.value.resumable and start == u.snapshot()
    check(u)
    u.promote(2, 0, expected=start, quiescent=True)
    check(u)


def test_recovery_failure_and_foreign_map_poison():
    for u in (fixture(HostTransfer(fail=2, persistent=True)), fixture()):
        start = u.snapshot()
        if not u.transfer.persistent:
            u.mapping[0, 1] = 99
        with pytest.raises(ResidencyUpdateError) as caught:
            u.promote(2, 0, expected=start, quiescent=True)
        assert not caught.value.resumable and not u.snapshot().healthy
        with pytest.raises(RuntimeError, match="unavailable"):
            u.promote(2, 0, expected=start, quiescent=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_native_fill_reuses_graph_and_recovers(tmp_path, monkeypatch, dtype):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    from benchmarks.moe.sm120_residency_poc import Experiment, load_layer
    from benchmarks.moe.sm120_residency_spectrum import Graphs
    from tests.moe.test_sm120_residency_poc import checkpoint_fields, write_layer

    write_layer(tmp_path, checkpoint_fields())
    source, _ = load_layer(tmp_path, "layer.experts", 4)
    e = Experiment(
        source,
        hot=2,
        capacity=4,
        topk=4,
        id_dtype=dtype,
        canonical_backing=True,
        backing_write_combined=False,
        journal_write_combined=False,
        cache_dir=tmp_path / "cache",
    )
    graphs = []
    try:
        graphs = [Graphs(e, n) for n in (1, 4, 2)]
        pointers = e.pointers()
        for candidate, victim in [(2, 0), (3, 1), (0, 2), (1, 3)]:
            before = e.updates.snapshot()
            e.updates.promote(candidate, victim, expected=before, quiescent=True)
            for g in graphs:
                g.inputs(torch.arange(g.live * 4).reshape(g.live, 4) % 4)
                e.a.mul_(0.9375)
                g.validate(allocator=True)
            assert pointers == e.pointers()
        original = e.updates.snapshot()
        copy = e.updates.transfer.copy
        # Include a failure after map publication, not just before payload copy.
        for fail in (2, 4, 8):
            calls = 0

            def failing(dst, src):
                nonlocal calls
                calls += 1
                copy(dst, src)
                if calls == fail:
                    raise OSError("injected submitted-write failure")

            monkeypatch.setattr(e.updates.transfer, "copy", failing)
            with pytest.raises(ResidencyUpdateError) as error:
                e.updates.promote(2, 0, expected=original, quiescent=True)
            assert error.value.resumable and e.updates.snapshot() == original
            graphs[0].validate(allocator=True)
        monkeypatch.setattr(e.updates.transfer, "copy", copy)
    finally:
        for g in graphs:
            g.close()
        e.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
def test_native_fill_sanitizer(tmp_path):
    """Small native graph case for bounded memcheck/synccheck runs."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe as moe
    from benchmarks.moe.sm120_residency_poc import Experiment, load_layer
    from tests.moe.test_sm120_residency_poc import checkpoint_fields, write_layer

    write_layer(tmp_path, checkpoint_fields())
    source, _ = load_layer(tmp_path, "layer.experts", 4)
    e = Experiment(
        source,
        hot=2,
        capacity=1,
        topk=4,
        canonical_backing=True,
        backing_write_combined=False,
        journal_write_combined=False,
        cache_dir=tmp_path / "cache",
    )
    try:
        graph, control = e.capture(1)
        pointers = e.pointers()
        for candidate, victim in [(2, 0), (0, 2)]:
            e.ids.copy_(
                torch.tensor([[candidate, candidate, -1, 2**40]], device="cuda")
            )
            before = torch.cuda.memory_stats()
            with kernel_resolution_guard("canonical native sanitizer replay"):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed"):
                assert before[key] == after[key]
            reference = moe.run(binding=control)
            torch.testing.assert_close(e.output, reference, atol=0, rtol=0)
            assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
            e.updates.promote(
                candidate, victim, expected=e.updates.snapshot(), quiescent=True
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(e.output, reference, atol=0, rtol=0)
            assert pointers == e.pointers()
    finally:
        e.close()


@pytest.mark.parametrize("failure", [5, 6])
def test_staged_copy_and_publication_failure_restore(failure):
    u = fixture(HostTransfer(fail=failure), staged=True)
    start = u.snapshot()
    with pytest.raises(ResidencyUpdateError) as caught:
        u.promote(2, 0, expected=start, quiescent=True)
    assert caught.value.resumable and u.snapshot() == start
    check(u)


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5])
def test_staged_completion_failure_restores_map(boundary):
    class FailingCompletion(HostTransfer):
        completions = 0

        def synchronize(self):
            self.completions += 1
            if self.completions == boundary:
                raise OSError("completion failed")

    u = fixture(FailingCompletion(), staged=True)
    original = u.snapshot()
    with pytest.raises(ResidencyUpdateError) as caught:
        u.promote(2, 0, expected=original, quiescent=True)
    assert caught.value.resumable and original == u.snapshot()
    check(u)


def test_canonical_fill_rejects_foreign_bytes():
    u = fixture()
    source = {n: v.clone() for n, v in u.canonical.items()}
    source["scales"][0, 0] ^= 1
    with pytest.raises(ValueError, match="verified canonical"):
        CanonicalFills(
            resident=u.resident,
            canonical=u.canonical,
            source=source,
            mapping=u.mapping,
            expert_map=u.snapshot().expert_map,
            before=u.before,
            after=u.after,
            transfer=u.transfer,
        )
