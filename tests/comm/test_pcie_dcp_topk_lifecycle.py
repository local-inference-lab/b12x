"""Focused lifecycle/close tests for the DCP top-k IPC channel.

Issue #181: collective teardown ordering, error propagation, lifecycle state,
retry from FAILED, launch/replay/capture exclusion, destructor quarantine.
"""

from __future__ import annotations

import gc
import threading

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _ChannelLifecycle,
)
from b12x.comm.pcie.pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    _OwnedSharedBuffer,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeIPC:
    def __init__(
        self, *, close_fail: int = 0, free_fail: int = 0
    ) -> None:
        self.events: list[tuple[str, int]] = []
        self._cf = close_fail
        self._ff = free_fail

    @property
    def closed(self):
        return [p for op, p in self.events if op == "close"]

    @property
    def freed(self):
        return [p for op, p in self.events if op == "free"]

    def cudaIpcCloseMemHandle(self, ptr):
        self.events.append(("close", ptr))
        if ptr == self._cf:
            raise RuntimeError(f"injected unmap failure for {ptr}")

    def cudaFree(self, ptr):
        self.events.append(("free", ptr))
        if ptr == self._ff:
            raise RuntimeError(f"injected free failure for {ptr}")


def _buf(local=1000, remote=0):
    return _OwnedSharedBuffer(
        local_ptr=local,
        peer_ptrs=(local, remote) if remote else (local,),
        remote_ptrs=(remote,) if remote else (),
    )


def _ch(ipc, owned, *, group=None, ws=2, rank=0, ticket=False):
    c = PCIeDCPTopKOwnerExchange(
        rank=rank, world_size=ws, device="cpu",
        signal_ptrs=tuple(10 + i * 10 for i in range(ws)),
        staging0_ptrs=tuple(30 + i * 10 for i in range(ws)),
        staging1_ptrs=tuple(62 + i * 10 for i in range(ws)),
        max_rows=8, topk=4, ipc=ipc, owned_buffers=owned,
        exchange_group=group, stream_affine=False,
    )
    if ticket:
        c._has_collective_ipc_ticket = True
    return c


class _CP:
    """Fake 2-rank control plane with per-round shared barriers.

    When any rank raises during a barrier or exchange, all barriers are
    aborted so the peer thread also receives a BrokenBarrierError instead
    of hanging indefinitely.
    """

    def __init__(self, ws=2):
        self.ws = ws
        self._bi = 0
        self._ei = 0
        self._lk = threading.Lock()
        self._b: list[threading.Barrier] = []
        self._eb: list[threading.Barrier] = []
        self._ed: list[list[object]] = []
        self._local = threading.local()

    def _get_barrier(self):
        i = getattr(self._local, "barrier_index", 0)
        self._local.barrier_index = i + 1
        with self._lk:
            while i >= len(self._b):
                self._b.append(threading.Barrier(self.ws))
            return self._b[i]

    def dist_barrier(self, *, group=None):
        b = self._get_barrier()
        try:
            b.wait(timeout=10)
        except threading.BrokenBarrierError:
            raise RuntimeError(
                "collective barrier broken by peer failure"
            ) from None

    def gather(self, obj, group=None):
        i = getattr(self._local, "exchange_index", 0)
        self._local.exchange_index = i + 1
        with self._lk:
            while i >= len(self._eb):
                self._eb.append(threading.Barrier(self.ws))
                self._ed.append([None] * self.ws)
            b, d = self._eb[i], self._ed[i]
        try:
            s = b.wait(timeout=10)
            d[s] = obj
            b.wait(timeout=10)
            b.wait(timeout=10)
        except threading.BrokenBarrierError:
            raise RuntimeError(
                "collective exchange broken by peer failure"
            ) from None
        return list(d)

    def abort(self):
        """Abort all barriers so peer threads don't hang."""
        for b in self._b:
            b.abort()
        for b in self._eb:
            b.abort()

    def reset(self):
        with self._lk:
            self._local = threading.local()
            self._b.clear()
            self._eb.clear()
            self._ed.clear()


def _patch(monkeypatch, cp):
    monkeypatch.setattr("b12x.comm.pcie.pcie_dcp_topk.dist.barrier", cp.dist_barrier)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", cp.gather
    )


def _rank(monkeypatch, cp, rank, *, cf=0, ff=0):
    ipc = _FakeIPC(close_fail=cf, free_fail=ff)
    local = 1000 + rank * 1000
    remote = 1000 + (1 - rank) * 1000
    c = _ch(ipc, [_buf(local, remote)], group=object(), rank=rank, ticket=True)
    return c, ipc


# ---------------------------------------------------------------------------
# Single-rank tests (world_size=1 fake control plane, group=object())
# ---------------------------------------------------------------------------


def test_unmap_before_free(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    c.close()
    assert sorted(ipc.closed) == [2000]
    assert sorted(ipc.freed) == [1000]
    ci = [i for i, (op, _) in enumerate(ipc.events) if op == "close"]
    fi = [i for i, (op, _) in enumerate(ipc.events) if op == "free"]
    assert max(ci) < min(fi)
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED


def test_unmap_failure_blocks_free(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(close_fail=2000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC import close"):
        c.close()
    assert ipc.freed == []
    assert c._lifecycle_state == _ChannelLifecycle.FAILED
    assert [b.local_ptr for b in c._owned_buffers] == [1000]


def test_free_failure_retains(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(free_fail=1000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC export free"):
        c.close()
    assert 1000 in ipc.freed
    assert 1000 in [b.local_ptr for b in c._owned_buffers]
    assert c._lifecycle_state == _ChannelLifecycle.FAILED


def test_no_ipc_runtime_blocks(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    c = _ch(None, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="CUDA runtime is unavailable"):
        c.close()
    assert c._lifecycle_state == _ChannelLifecycle.FAILED


def test_local_only_no_group_can_close():
    """A channel with only local exports (no remote imports) closes fine."""
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    c.close()
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED
    assert ipc.freed == [1000]


def test_remote_imports_require_group_at_construction():
    ipc = _FakeIPC()
    with pytest.raises(ValueError, match="exchange_group is required"):
        _ch(ipc, [_buf(1000, 2000)], group=None, ticket=True)
    assert ipc.events == []


# ---------------------------------------------------------------------------
# Double / concurrent close
# ---------------------------------------------------------------------------


def test_double_close_idempotent():
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    c.close()
    first = list(ipc.freed)
    c.close()
    assert ipc.freed == first
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED


def test_close_after_failure_re_raises(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(close_fail=2000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC import close"):
        c.close()
    with pytest.raises(RuntimeError, match="IPC import close"):
        c.close()


def test_closed_rejects_stage():
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.stage_candidates(
            torch.zeros(8, 4, dtype=torch.int32),
            torch.zeros(8, 4, dtype=torch.float32),
        )


def test_failed_rejects_stage(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(close_fail=2000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC import close"):
        c.close()
    with pytest.raises(RuntimeError, match="failed"):
        c.stage_candidates(
            torch.zeros(8, 4, dtype=torch.int32),
            torch.zeros(8, 4, dtype=torch.float32),
        )


def test_concurrent_close_single_gen():
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    res: list[Exception | None] = []

    def do():
        try:
            c.close()
            res.append(None)
        except Exception as e:
            res.append(e)

    t1 = threading.Thread(target=do)
    t2 = threading.Thread(target=do)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED
    assert all(r is None for r in res)
    assert len(ipc.freed) == 1


def test_concurrent_close_failure_propagates(monkeypatch):
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(free_fail=1000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    errs: list[Exception | None] = []

    def do():
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            errs.append(e)

    t1 = threading.Thread(target=do)
    t2 = threading.Thread(target=do)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert c._lifecycle_state == _ChannelLifecycle.FAILED
    assert all(e is not None for e in errs)


# ---------------------------------------------------------------------------
# Retry from FAILED
# ---------------------------------------------------------------------------


def test_retry_from_failed(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, ipc0 = _rank(monkeypatch, cp, 0, ff=1000)
    c1, ipc1 = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def do(c):
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            cp.abort()
            errs.append(e)

    t0 = threading.Thread(target=do, args=(c0,))
    t1 = threading.Thread(target=do, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is not None for e in errs)
    assert c0._lifecycle_state == _ChannelLifecycle.FAILED

    cp.reset()
    ipc0._ff = 0
    rerrs: list[Exception | None] = []

    def do_retry(c):
        try:
            c.retry_close()
            rerrs.append(None)
        except Exception as e:
            cp.abort()
            rerrs.append(e)

    t0 = threading.Thread(target=do_retry, args=(c0,))
    t1 = threading.Thread(target=do_retry, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is None for e in rerrs), f"retry: {rerrs}"
    assert c0._lifecycle_state == _ChannelLifecycle.CLOSED
    assert c1._lifecycle_state == _ChannelLifecycle.CLOSED


# ---------------------------------------------------------------------------
# Launch vs close
# ---------------------------------------------------------------------------


class _Blocking(PCIeDCPTopKOwnerExchange):
    def __init__(self, **kw):
        super().__init__(
            rank=0, world_size=2, device="cpu",
            signal_ptrs=(10, 20), staging0_ptrs=(30, 40),
            staging1_ptrs=(62, 72), max_rows=8, topk=4, **kw,
        )
        sh = (self.max_owner_rows, self.world_size * self.topk)
        self._candidate_views = tuple(
            (torch.empty(sh, dtype=torch.int32),
             torch.empty(sh, dtype=torch.float32))
            for _ in range(2)
        )
        self._ls = threading.Event()
        self._lr = threading.Event()

    def _launch_stage(self, *a, **kw):
        self._ls.set()
        self._lr.wait(timeout=5)


def test_close_waits_for_launch():
    c = _Blocking(stream_affine=False)
    done = threading.Event()

    def launch():
        c.stage_candidates(
            torch.zeros(8, 4, dtype=torch.int32),
            torch.zeros(8, 4, dtype=torch.float32),
            threads=128, block_limit=1,
        )
        done.set()

    def close():
        c._ls.wait(timeout=5)
        c._lr.set()
        c.close()

    tl = threading.Thread(target=launch)
    tc = threading.Thread(target=close)
    tl.start()
    c._ls.wait(timeout=5)
    tc.start()
    tl.join(timeout=10)
    tc.join(timeout=10)
    assert done.is_set()
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED


def test_closing_rejects_launch():
    c = _Blocking(stream_affine=False)
    with c._lifecycle_lock:
        c._lifecycle_state = _ChannelLifecycle.CLOSING
        c._close_generation += 1
    with pytest.raises(RuntimeError, match="closing"):
        c.stage_candidates(
            torch.zeros(8, 4, dtype=torch.int32),
            torch.zeros(8, 4, dtype=torch.float32),
            threads=128, block_limit=1,
        )


# ---------------------------------------------------------------------------
# Replay vs close
# ---------------------------------------------------------------------------


class _FakeGraph:
    def __init__(self):
        self.n = 0

    def replay(self):
        self.n += 1


def test_unregistered_graph_replay_is_rejected():
    c = _Blocking(stream_affine=False)
    with pytest.raises(RuntimeError, match="was not registered"):
        c.replay(_FakeGraph())


def test_registered_graph_replays():
    c = _Blocking(stream_affine=False)
    graph = _FakeGraph()
    c.register_graph(graph)
    c.replay(graph)
    assert graph.n == 1


def test_replay_rejected_after_close():
    c = _Blocking(stream_affine=False)
    c._has_collective_ipc_ticket = True
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.replay(_FakeGraph())


def test_close_waits_for_replay():
    c = _Blocking(stream_affine=False)
    c._has_collective_ipc_ticket = True
    rs = threading.Event()
    rr = threading.Event()
    cd = threading.Event()

    def replay():
        with c._replay_gate():
            rs.set()
            rr.wait(timeout=5)

    def close():
        rs.wait(timeout=5)
        rr.set()
        c.close()
        cd.set()

    tr = threading.Thread(target=replay)
    tc = threading.Thread(target=close)
    tr.start()
    rs.wait(timeout=5)
    tc.start()
    tr.join(timeout=10)
    tc.join(timeout=10)
    assert cd.is_set()
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED


# ---------------------------------------------------------------------------
# Destructor quarantine
# ---------------------------------------------------------------------------


def test_destructor_quarantines_incomplete():
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    cid = id(c)
    del c
    gc.collect()
    r = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(cid, None)
    assert r is not None
    r._coordinated_close_complete = True
    del r


def test_destructor_no_quarantine_after_close():
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    c.close()
    cid = id(c)
    del c
    gc.collect()
    assert cid not in _ABANDONED_PCIE_RUNTIME_QUARANTINE


def test_destructor_quarantines_failed(monkeypatch):
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(close_fail=2000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC import close"):
        c.close()
    cid = id(c)
    del c
    gc.collect()
    r = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(cid, None)
    assert r is not None
    r._coordinated_close_complete = True
    del r


def test_destructor_quarantines_locally_freed_failed(monkeypatch):
    _ABANDONED_PCIE_RUNTIME_QUARANTINE.clear()
    cp = _CP(ws=1)
    _patch(monkeypatch, cp)
    ipc = _FakeIPC(free_fail=1000)
    c = _ch(ipc, [_buf(1000, 2000)], group=object(), ticket=True)
    with pytest.raises(RuntimeError, match="IPC export free"):
        c.close()
    cid = id(c)
    del c
    gc.collect()
    r = _ABANDONED_PCIE_RUNTIME_QUARANTINE.pop(cid, None)
    assert r is not None
    r._coordinated_close_complete = True
    del r


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager():
    ipc = _FakeIPC()
    c = _ch(ipc, [_buf(1000)], group=None, ticket=True)
    with c as ctx:
        assert ctx is c
        assert c._lifecycle_state == _ChannelLifecycle.OPEN
    assert c._lifecycle_state == _ChannelLifecycle.CLOSED


# ---------------------------------------------------------------------------
# 2-rank tests
# ---------------------------------------------------------------------------


def test_two_rank_unmap_before_free(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, ipc0 = _rank(monkeypatch, cp, 0)
    c1, ipc1 = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def do(c):
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            errs.append(e)

    t0 = threading.Thread(target=do, args=(c0,))
    t1 = threading.Thread(target=do, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is None for e in errs), f"errors: {errs}"
    for ipc in (ipc0, ipc1):
        ci = [i for i, (op, _) in enumerate(ipc.events) if op == "close"]
        fi = [i for i, (op, _) in enumerate(ipc.events) if op == "free"]
        if ci and fi:
            assert max(ci) < min(fi)
    assert c0._lifecycle_state == _ChannelLifecycle.CLOSED
    assert c1._lifecycle_state == _ChannelLifecycle.CLOSED


def test_two_rank_asymmetric_unmap(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, ipc0 = _rank(monkeypatch, cp, 0, cf=2000)
    c1, ipc1 = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def do(c):
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            cp.abort()
            errs.append(e)

    t0 = threading.Thread(target=do, args=(c0,))
    t1 = threading.Thread(target=do, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is not None for e in errs), f"expected errors: {errs}"
    assert ipc0.freed == []
    assert ipc1.freed == []
    assert c0._lifecycle_state == _ChannelLifecycle.FAILED
    assert c1._lifecycle_state == _ChannelLifecycle.FAILED



def test_two_rank_asymmetric_free(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, ipc0 = _rank(monkeypatch, cp, 0, ff=1000)
    c1, ipc1 = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def do(c):
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            errs.append(e)
            cp.abort()

    t0 = threading.Thread(target=do, args=(c0,))
    t1 = threading.Thread(target=do, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is not None for e in errs)
    assert 1000 in ipc0.freed
    assert 1000 in [b.local_ptr for b in c0._owned_buffers]
    assert c0._lifecycle_state == _ChannelLifecycle.FAILED
    assert c1._lifecycle_state == _ChannelLifecycle.FAILED


def test_two_rank_concurrent_close(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, _ = _rank(monkeypatch, cp, 0)
    c1, _ = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def close_twice():
        def once():
            try:
                c0.close()
                errs.append(None)
            except Exception as e:
                errs.append(e)

        ta = threading.Thread(target=once)
        tb = threading.Thread(target=once)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)

    c1_errs: list[Exception] = []

    def close1():
        try:
            c1.close()
        except Exception as exc:
            c1_errs.append(exc)

    t0 = threading.Thread(target=close_twice)
    t1 = threading.Thread(target=close1)
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is None for e in errs), f"c0 errors: {errs}"
    assert not c1_errs, f"c1 errors: {c1_errs}"
    assert c0._lifecycle_state == _ChannelLifecycle.CLOSED
    assert c1._lifecycle_state == _ChannelLifecycle.CLOSED


def test_two_rank_retry_after_failure(monkeypatch):
    cp = _CP(ws=2)
    _patch(monkeypatch, cp)
    c0, ipc0 = _rank(monkeypatch, cp, 0, ff=1000)
    c1, ipc1 = _rank(monkeypatch, cp, 1)
    errs: list[Exception | None] = []

    def do(c):
        try:
            c.close()
            errs.append(None)
        except Exception as e:
            cp.abort()
            errs.append(e)

    t0 = threading.Thread(target=do, args=(c0,))
    t1 = threading.Thread(target=do, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is not None for e in errs)
    assert c0._lifecycle_state == _ChannelLifecycle.FAILED

    cp.reset()
    ipc0._ff = 0
    rerrs: list[Exception | None] = []

    def do_retry(c):
        try:
            c.retry_close()
            rerrs.append(None)
        except Exception as e:
            rerrs.append(e)

    t0 = threading.Thread(target=do_retry, args=(c0,))
    t1 = threading.Thread(target=do_retry, args=(c1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert all(e is None for e in rerrs), f"retry: {rerrs}"
    assert c0._lifecycle_state == _ChannelLifecycle.CLOSED
    assert c1._lifecycle_state == _ChannelLifecycle.CLOSED
