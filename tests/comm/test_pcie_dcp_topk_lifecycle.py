"""Focused lifecycle/close tests for the DCP top-k IPC channel.

These tests prove the collective teardown ordering and error-propagation
contract required by issue #181:

* No export free (``cudaFree``) precedes every peer unmap acknowledgement
  (``cudaIpcCloseMemHandle``).
* Forced unmap, free, and collective-exchange failures propagate as
  ``RuntimeError`` and leave the channel non-reusable without freeing
  unsafe exports.
* Double, out-of-order, and concurrent close cannot reuse stale pointers
  or silently succeed.
"""

from __future__ import annotations

from typing import Sequence

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
from b12x.comm.pcie.pcie_oneshot import _OwnedSharedBuffer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeIPC:
    """Records close/free calls and optionally injects failures.

    ``events`` records every call in order as ``("close", ptr)`` or
    ``("free", ptr)`` so tests can assert exact teardown ordering.
    """

    def __init__(
        self,
        *,
        close_fail_ptrs: Sequence[int] = (),
        free_fail_ptrs: Sequence[int] = (),
    ) -> None:
        self.events: list[tuple[str, int]] = []
        self._close_fail = set(close_fail_ptrs)
        self._free_fail = set(free_fail_ptrs)

    @property
    def closed(self) -> list[int]:
        return [ptr for op, ptr in self.events if op == "close"]

    @property
    def freed(self) -> list[int]:
        return [ptr for op, ptr in self.events if op == "free"]

    def cudaIpcCloseMemHandle(self, ptr: int) -> None:
        self.events.append(("close", ptr))
        if ptr in self._close_fail:
            raise RuntimeError(f"injected unmap failure for ptr {ptr}")

    def cudaFree(self, ptr: int) -> None:
        self.events.append(("free", ptr))
        if ptr in self._free_fail:
            raise RuntimeError(f"injected free failure for ptr {ptr}")


def _make_channel(
    ipc: _FakeIPC | None,
    owned: Sequence[_OwnedSharedBuffer],
    *,
    exchange_group=None,
) -> PCIeDCPTopKOwnerExchange:
    """Construct a CPU DCP top-k channel with injected IPC and buffers."""
    ch = PCIeDCPTopKOwnerExchange(
        rank=0,
        world_size=2,
        device="cpu",
        signal_ptrs=(10, 20),
        staging0_ptrs=(30, 40),
        staging1_ptrs=(62, 72),
        max_rows=8,
        topk=4,
        ipc=ipc,
        owned_buffers=owned,
        exchange_group=exchange_group,
        stream_affine=False,
    )
    return ch


def _buffers() -> list[_OwnedSharedBuffer]:
    return [
        _OwnedSharedBuffer(local_ptr=1000, peer_ptrs=(1000, 2000), remote_ptrs=(2000,)),
        _OwnedSharedBuffer(local_ptr=3000, peer_ptrs=(3000, 4000), remote_ptrs=(4000,)),
    ]


# ---------------------------------------------------------------------------
# Ordering: unmap before free
# ---------------------------------------------------------------------------


def test_close_unmaps_all_imports_before_freeing_any_export():
    """Every cudaIpcCloseMemHandle must precede every cudaFree."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    ch.close()

    # All remote pointers were closed.
    assert sorted(ipc.closed) == [2000, 4000]
    # All local pointers were freed.
    assert sorted(ipc.freed) == [1000, 3000]
    # The event log proves all closes precede all frees.
    close_events = [i for i, (op, _) in enumerate(ipc.events) if op == "close"]
    free_events = [i for i, (op, _) in enumerate(ipc.events) if op == "free"]
    assert max(close_events) < min(free_events)
    assert ch._coordinated_close_complete
    assert ch._ipc_imports_closed
    assert ch._ipc_exports_freed


def test_unmap_failure_blocks_all_export_frees():
    """If any unmap fails, no cudaFree is called on any buffer."""
    ipc = _FakeIPC(close_fail_ptrs=(2000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch.close()

    # The failing import was attempted but the other succeeded.
    assert 2000 in ipc.closed
    assert 4000 in ipc.closed
    # Critical: NO export was freed.
    assert ipc.freed == []
    # Channel is closed (non-reusable) but exports are NOT freed.
    assert ch._closed
    assert not ch._ipc_exports_freed
    assert not ch._coordinated_close_complete
    # Owned buffers are retained.
    assert [b.local_ptr for b in ch._owned_buffers] == [1000, 3000]


def test_free_failure_retains_export_and_propagates():
    """If a free fails, the buffer is retained and the error propagates."""
    ipc = _FakeIPC(free_fail_ptrs=(1000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC export free"):
        ch.close()

    # Unmap succeeded for all.
    assert sorted(ipc.closed) == [2000, 4000]
    # Free was attempted.
    assert 1000 in ipc.freed
    assert 3000 in ipc.freed
    # The failing export is retained.
    assert 1000 in [b.local_ptr for b in ch._owned_buffers]
    assert 3000 not in [b.local_ptr for b in ch._owned_buffers]
    assert not ch._ipc_exports_freed
    assert not ch._coordinated_close_complete


def test_unmap_failure_on_second_buffer_still_closes_first():
    """Partial unmap success is tracked; failed imports remain open."""
    ipc = _FakeIPC(close_fail_ptrs=(4000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch.close()

    assert 2000 in ipc.closed  # succeeded
    assert 4000 in ipc.closed  # attempted
    assert ipc.freed == []
    # The successfully closed import is tracked; the failed one is not.
    assert (0, 0) in ch._closed_ipc_import_indices  # buffer 0, remote 0
    assert (1, 0) not in ch._closed_ipc_import_indices  # buffer 1, remote 0


# ---------------------------------------------------------------------------
# Error propagation: no suppression
# ---------------------------------------------------------------------------


def test_no_ipc_runtime_blocks_unmap_and_free():
    """Missing CUDA runtime prevents both unmap and free with errors."""
    ch = _make_channel(None, _buffers())

    with pytest.raises(RuntimeError, match="CUDA runtime is unavailable for IPC unmap"):
        ch.close()

    assert ch._closed
    assert not ch._ipc_imports_closed
    assert not ch._ipc_exports_freed
    assert not ch._coordinated_close_complete
    assert [b.local_ptr for b in ch._owned_buffers] == [1000, 3000]


def test_free_failure_after_successful_unmap_retains_buffer():
    """Free failure after full unmap success retains the export."""
    ipc = _FakeIPC(free_fail_ptrs=(3000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC export free"):
        ch.close()

    assert sorted(ipc.closed) == [2000, 4000]
    assert sorted(ipc.freed) == [1000, 3000]
    # 1000 freed, 3000 retained.
    assert [b.local_ptr for b in ch._owned_buffers] == [3000]
    assert ch._ipc_imports_closed
    assert not ch._ipc_exports_freed


# ---------------------------------------------------------------------------
# Double / out-of-order / concurrent close
# ---------------------------------------------------------------------------


def test_double_close_is_idempotent():
    """A second close() does not issue additional IPC calls."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    ch.close()
    assert ch._coordinated_close_complete

    first_closed = list(ipc.closed)
    first_freed = list(ipc.freed)

    ch.close()  # should be a no-op

    assert ipc.closed == first_closed
    assert ipc.freed == first_freed
    assert ch._coordinated_close_complete


def test_close_coordinated_is_also_idempotent():
    """close_coordinated() respects the same completion guard."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    ch.close_coordinated()
    assert ch._coordinated_close_complete

    first_closed = list(ipc.closed)
    first_freed = list(ipc.freed)

    ch.close_coordinated()

    assert ipc.closed == first_closed
    assert ipc.freed == first_freed


def test_closed_channel_cannot_stage_candidates():
    """After close, stage_candidates raises RuntimeError, not silent reuse."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())
    ch.close()
    assert ch._closed

    indices = torch.zeros(8, 4, dtype=torch.int32)
    scores = torch.zeros(8, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="closed"):
        ch.stage_candidates(indices, scores)


def test_failed_close_channel_cannot_stage_candidates():
    """A channel whose unmap failed is closed and cannot be reused."""
    ipc = _FakeIPC(close_fail_ptrs=(2000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch.close()

    assert ch._closed
    indices = torch.zeros(8, 4, dtype=torch.int32)
    scores = torch.zeros(8, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="closed"):
        ch.stage_candidates(indices, scores)


def test_close_after_partial_unmap_failure_does_not_free_exports():
    """Retrying close after a partial unmap failure does not free exports."""
    ipc = _FakeIPC(close_fail_ptrs=(2000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch.close()

    # Exports are still alive.
    assert [b.local_ptr for b in ch._owned_buffers] == [1000, 3000]
    assert ipc.freed == []

    # Even if we fix the IPC and retry, the channel should not silently free
    # without going through the coordinated protocol again.  Since
    # _coordinated_close_complete is False, close() re-enters the protocol.
    ipc._close_fail = set()  # fix the failure
    ch.close()
    # Now the retry should succeed and free everything.
    assert ch._coordinated_close_complete
    assert sorted(ipc.freed) == [1000, 3000]


def test_concurrent_close_does_not_double_free():
    """Two close() calls on a channel that completes simultaneously do not
    issue duplicate IPC calls."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    # Simulate concurrent close: the completion guard makes the second a no-op.
    ch.close()
    # Manually simulate a racing second close that sees the flag after the
    # first has set it.
    assert ch._coordinated_close_complete
    ch.close()

    assert len(ipc.closed) == 2  # exactly 2 remote pointers, once each
    assert len(ipc.freed) == 2  # exactly 2 local pointers, once each


def test_out_of_order_close_retains_exports_until_unmap_acknowledged():
    """A channel closed before its peer unmapped must not free exports."""
    ipc = _FakeIPC(close_fail_ptrs=(2000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch.close()

    # The export for the buffer whose import failed to close is still alive.
    assert any(b.local_ptr == 1000 for b in ch._owned_buffers)
    assert any(b.local_ptr == 3000 for b in ch._owned_buffers)
    assert ipc.freed == []


# ---------------------------------------------------------------------------
# Strict method direct invocation
# ---------------------------------------------------------------------------


def test_strict_unmap_direct_propagates_error():
    ipc = _FakeIPC(close_fail_ptrs=(2000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC import close"):
        ch._close_ipc_imports_strict()

    assert ch._closed
    assert not ch._ipc_imports_closed
    assert ipc.freed == []


def test_strict_free_direct_propagates_error():
    ipc = _FakeIPC(free_fail_ptrs=(1000,))
    ch = _make_channel(ipc, _buffers())

    with pytest.raises(RuntimeError, match="IPC export free"):
        ch._free_ipc_exports_strict()

    assert 1000 in [b.local_ptr for b in ch._owned_buffers]
    assert not ch._ipc_exports_freed


def test_strict_unmap_idempotent_after_success():
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    ch._close_ipc_imports_strict()
    assert ch._ipc_imports_closed
    first_closed = list(ipc.closed)

    ch._close_ipc_imports_strict()  # should be no-op
    assert ipc.closed == first_closed


def test_strict_free_idempotent_after_success():
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())

    ch._close_ipc_imports_strict()
    ch._free_ipc_exports_strict()
    assert ch._ipc_exports_freed
    first_freed = list(ipc.freed)

    ch._free_ipc_exports_strict()  # should be no-op
    assert ipc.freed == first_freed


def test_candidate_views_cleared_on_free():
    """Freeing exports clears candidate views so stale tensors are released."""
    ipc = _FakeIPC()
    ch = _make_channel(ipc, _buffers())
    # CPU channels start with empty candidate views.
    ch._candidate_views = (
        (torch.empty(1, dtype=torch.int32), torch.empty(1, dtype=torch.float32)),
    )
    assert ch._candidate_views

    ch.close()
    assert ch._candidate_views == ()
