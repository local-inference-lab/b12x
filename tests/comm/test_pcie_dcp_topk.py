from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _SIGNAL_BYTES,
    _candidate_staging_layout,
    owner_stage_reference,
)
from b12x.comm.pcie._dcp_cute_common import signal_bytes


class _FakeOwner(PCIeDCPTopKOwnerExchange):
    def __init__(self) -> None:
        super().__init__(
            rank=1,
            world_size=2,
            device="cpu",
            signal_ptrs=(10, 20),
            staging0_ptrs=(30, 40),
            staging1_ptrs=(62, 72),
            max_rows=8,
            topk=4,
        )
        shape = (self.max_owner_rows, self.world_size * self.topk)
        self._candidate_views = tuple(
            (
                torch.empty(shape, dtype=torch.int32),
                torch.empty(shape, dtype=torch.float32),
            )
            for _ in range(2)
        )
        self.stage_calls: list[tuple] = []

    def _launch_stage(
        self,
        local_indices,
        local_scores,
        *,
        slot,
        rows,
        threads,
        blocks,
        wait_for_prior_consumer,
    ):
        owner_rows = rows // self.world_size
        row_slice = slice(self.rank * owner_rows, (self.rank + 1) * owner_rows)
        views = self._candidate_views[slot]
        views[0][:owner_rows].copy_(
            local_indices[row_slice].repeat(1, self.world_size)
        )
        views[1][:owner_rows].copy_(
            local_scores[row_slice].repeat(1, self.world_size)
        )
        self.stage_calls.append((slot, threads, blocks, wait_for_prior_consumer))


def _make_owner() -> PCIeDCPTopKOwnerExchange:
    return _FakeOwner()


def test_candidate_staging_layout_is_aligned():
    owner = _candidate_staging_layout(
        signal_bytes=513,
        max_rows=8,
        topk=4,
        world_size=2,
    )
    assert owner.staging0_offset == 768
    assert owner.plane_bytes == 128
    assert owner.slot_bytes == 256
    assert owner.staging1_offset == 1024
    assert owner.slab_bytes == 1280


def test_topk_signal_layout_matches_the_shared_barrier_abi():
    assert signal_bytes(128) == _SIGNAL_BYTES


def test_owner_stage_reference_preserves_rank_major_order_and_score_bits():
    world_size, rows, topk = 4, 8, 4
    indices = torch.empty(world_size, rows, topk, dtype=torch.int32)
    score_bits = torch.empty(world_size, rows, topk, dtype=torch.int32)
    for rank in range(world_size):
        for row in range(rows):
            indices[rank, row].fill_(1000 * rank + 10 * row)
            score_bits[rank, row].fill_(0x3F000000 + rank * 16 + row)
    scores = score_bits.view(torch.float32)

    owner_indices, owner_scores = owner_stage_reference(indices, scores, 2)

    assert owner_indices.shape == (2, world_size * topk)
    assert owner_scores.shape == owner_indices.shape
    for owner_row, global_row in enumerate((4, 5)):
        for rank in range(world_size):
            rank_slice = slice(rank * topk, (rank + 1) * topk)
            assert torch.equal(
                owner_indices[owner_row, rank_slice], indices[rank, global_row]
            )
            assert torch.equal(
                owner_scores[owner_row, rank_slice].view(torch.int32),
                score_bits[rank, global_row],
            )


def test_owner_dispatches_and_disposes():
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10

    candidate_indices, candidate_scores = owner.stage_candidates(
        indices,
        scores,
        threads=128,
        block_limit=32,
    )
    assert torch.equal(candidate_indices, indices[4:].repeat(1, 2))
    assert torch.equal(candidate_scores, scores[4:].repeat(1, 2))
    assert owner.stage_calls == [(0, 128, 1, False)]

    owner.close()
    assert owner._closed


@pytest.mark.parametrize("eager_calls, expected_slot", [(0, 0), (1, 1), (2, 0)])
def test_graph_capture_pins_the_next_staging_slot_and_enables_prior_wait(
    monkeypatch,
    eager_calls,
    expected_slot,
):
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    capture_state = False
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_topk._is_current_stream_capturing",
        lambda _device: capture_state,
    )
    monkeypatch.setattr(owner, "prepare_graph", lambda **kwargs: None)
    monkeypatch.setattr(
        "b12x.comm.pcie._dcp_topk_cute.is_topk_stage_prepared",
        lambda *args: True,
    )

    for _ in range(eager_calls):
        owner.stage_candidates(indices, scores)
    capture_state = True
    with owner.capture():
        first_indices, first_scores = owner.stage_candidates(indices, scores)
    capture_state = False
    second_indices, second_scores = owner.stage_candidates(indices + 100, scores + 1)
    capture_state = True
    with owner.capture():
        third_indices, third_scores = owner.stage_candidates(indices + 200, scores + 2)

    assert owner._graph_slot == expected_slot
    assert first_indices.data_ptr() == second_indices.data_ptr()
    assert second_indices.data_ptr() == third_indices.data_ptr()
    assert first_scores.data_ptr() == second_scores.data_ptr()
    assert second_scores.data_ptr() == third_scores.data_ptr()
    eager_stages = [
        (slot % 2, 512, 1, False) for slot in range(eager_calls)
    ]
    assert owner.stage_calls == eager_stages + [
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
    ]
    assert torch.equal(third_indices, (indices + 200)[4:].repeat(1, 2))


def test_owner_rejects_invalid_contracts():
    owner = _make_owner()
    indices = torch.zeros(8, 4, dtype=torch.int32)
    scores = torch.zeros(8, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices, scores[:4])
    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices.flatten(), scores.flatten())
    with pytest.raises(ValueError, match="local_indices must be int32"):
        owner.stage_candidates(indices.float(), scores)
    with pytest.raises(ValueError, match="local_scores must be float32"):
        owner.stage_candidates(indices, scores.bfloat16())
    with pytest.raises(ValueError, match="divisible"):
        owner.stage_candidates(indices[:7], scores[:7])
    with pytest.raises(ValueError, match="threads"):
        owner.stage_candidates(indices, scores, threads=31)
    with pytest.raises(ValueError, match="block_limit"):
        owner.stage_candidates(indices, scores, block_limit=129)


def test_configuration_rejects_invalid_capacity_and_topk():
    with pytest.raises(ValueError, match="multiple of 4"):
        _candidate_staging_layout(
            signal_bytes=256,
            max_rows=8,
            topk=6,
            world_size=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        PCIeDCPTopKOwnerExchange(
            rank=0,
            world_size=4,
            device="cpu",
            signal_ptrs=(1, 2, 3, 4),
            staging0_ptrs=(5, 6, 7, 8),
            staging1_ptrs=(9, 10, 11, 12),
            max_rows=6,
            topk=4,
        )


# ---------------------------------------------------------------------------
# Issue #178: collective DCP Top-K contract agreement before IPC allocation.
# ---------------------------------------------------------------------------
#
# The tests below verify that every rank exchanges and compares a versioned
# fixed-schema contract *before* any cudaIpcOpenMemHandle occurs.  A
# single-field mismatch in capacity, offset, wire mode/codec, topology, or
# schedule must fail every rank coherently.  An exact match must proceed to
# allocation.

from b12x.comm.pcie import pcie_dcp_topk as _topk_mod
from b12x.comm.pcie import pcie_oneshot as _oneshot_mod


def _base_contract() -> tuple:
    """Return the canonical contract for world=2, max_rows=8, topk=4."""
    layout = _candidate_staging_layout(
        signal_bytes=_SIGNAL_BYTES,
        max_rows=8,
        topk=4,
        world_size=2,
    )
    return _topk_mod._dcp_topk_runtime_contract(
        world_size=2,
        max_rows=8,
        topk=4,
        layout=layout,
    )


def _patch_factory_gates(monkeypatch, *, peer_contract: tuple) -> None:
    """Monkeypatch collective gates so a 2-rank exchange can run on CPU.

    * ``dist.get_rank`` / ``get_world_size`` report a 2-rank group.
    * ``_run_collective_preallocation_setup`` runs ``setup()`` immediately.
    * ``_require_full_grid_residency`` is a no-op (CPU test).
    * ``_broadcast_gather_object`` returns ``[local, peer]`` so the local
      rank sees a mismatched peer contract.
    * ``_allocate_shared_buffer`` is patched to ``pytest.fail`` so the test
      proves the contract check fires *before* IPC allocation.
    """
    monkeypatch.setattr(_topk_mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(_topk_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        _topk_mod,
        "_run_collective_preallocation_setup",
        lambda **kwargs: kwargs["setup"](),
    )
    monkeypatch.setattr(
        _topk_mod,
        "_require_full_grid_residency",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        _topk_mod,
        "CudaRTLibrary",
        lambda: type("FakeIPC", (), {
            "cudaSetDevice": lambda self, device: None,
        })(),
    )
    monkeypatch.setattr(
        _oneshot_mod,
        "_broadcast_gather_object",
        lambda contract, group: [contract, peer_contract],
    )
    monkeypatch.setattr(
        _oneshot_mod.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        lambda *args, **kwargs: pytest.fail(
            "IPC allocation must not start when the contract mismatches"
        ),
    )


def _contract_field_index(field: str) -> int:
    """Return the position of a named field in the contract tuple."""
    layout = _candidate_staging_layout(
        signal_bytes=_SIGNAL_BYTES,
        max_rows=8,
        topk=4,
        world_size=2,
    )
    canonical = _topk_mod._dcp_topk_runtime_contract(
        world_size=2,
        max_rows=8,
        topk=4,
        layout=layout,
    )

    replacements: dict[str, tuple] = (
        ("capacity_max_rows", (3,)),
        ("capacity_topk", (5,)),
        ("offset_staging0", (11,)),
        ("offset_staging1", (12,)),
        ("wire_codec_index", (13,)),
        ("wire_codec_score", (14,)),
        ("topology_world", (1,)),
        ("schedule_max_blocks", (15,)),
        ("schedule_signal_bytes", (16,)),
    )
    if field not in replacements:
        raise KeyError(f"unknown field {field}")
    (idx,) = replacements[field]
    assert 0 <= idx < len(canonical), f"field {field} index out of range"
    return idx


def _mismatched_contract(field: str) -> tuple:
    """Return a contract that differs from the canonical one in exactly one field."""
    contract = list(_base_contract())
    idx = _contract_field_index(field)
    original = contract[idx]
    if isinstance(original, int):
        contract[idx] = original + 1
    elif isinstance(original, str):
        contract[idx] = original + "_mismatched"
    elif isinstance(original, tuple):
        contract[idx] = original + (999,)
    else:
        contract[idx] = "__MISMATCHED__"
    return tuple(contract)


@pytest.mark.parametrize(
    "field, label",
    [
        ("capacity_max_rows", "capacity"),
        ("capacity_topk", "capacity"),
        ("offset_staging0", "offset"),
        ("offset_staging1", "offset"),
        ("wire_codec_index", "wire mode/codec"),
        ("wire_codec_score", "wire mode/codec"),
        ("topology_world", "topology"),
        ("schedule_max_blocks", "schedule"),
        ("schedule_signal_bytes", "schedule"),
    ],
)
def test_contract_single_field_mismatch_fails_before_ipc_allocation(
    monkeypatch, field, label,
):
    """A single-field mismatch in {label} must fail every rank before
    cudaIpcOpenMemHandle."""
    _patch_factory_gates(monkeypatch, peer_contract=_mismatched_contract(field))
    with pytest.raises(RuntimeError, match="contract differs across ranks"):
        PCIeDCPTopKOwnerExchange.from_exchange_group(
            exchange_group=object(),
            device=torch.device("cuda:0"),
            max_rows=8,
            topk=4,
        )


def test_contract_exact_match_proceeds_to_allocation(monkeypatch):
    """When every field matches, the factory must proceed past the contract
    check to IPC allocation."""
    shared = _oneshot_mod._OwnedSharedBuffer(
        local_ptr=1000,
        peer_ptrs=(1000, 2000),
        remote_ptrs=(2000,),
    )

    class FakeIPC:
        def cudaSetDevice(self, device):
            pass
        def cudaFree(self, ptr):
            pass
        def cudaIpcCloseMemHandle(self, ptr):
            pass

    monkeypatch.setattr(_topk_mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(_topk_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        _topk_mod,
        "_run_collective_preallocation_setup",
        lambda **kwargs: kwargs["setup"](),
    )
    monkeypatch.setattr(
        _topk_mod,
        "_require_full_grid_residency",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        _oneshot_mod,
        "_broadcast_gather_object",
        lambda contract, group: [contract, contract],
    )
    monkeypatch.setattr(
        _oneshot_mod.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        lambda *args, **kwargs: shared,
    )
    monkeypatch.setattr(_topk_mod, "CudaRTLibrary", lambda: FakeIPC())
    monkeypatch.setattr(
        _topk_mod,
        "_tensor_from_cuda_pointer",
        lambda ptr, shape, **kwargs: torch.empty(shape, dtype=kwargs.get("dtype", torch.int32)),
    )

    owner = PCIeDCPTopKOwnerExchange.from_exchange_group(
        exchange_group=object(),
        device=torch.device("cuda:0"),
        max_rows=8,
        topk=4,
    )
    assert owner.world_size == 2
    assert owner.max_rows == 8
    assert owner.topk == 4
    owner._coordinated_close_complete = True
