from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _SIGNAL_BYTES,
    _candidate_staging_layout,
    _dcp_topk_runtime_contract,
    _dcp_topk_topology_record,
    _verify_dcp_topk_topology,
    owner_stage_reference,
)
from b12x.comm.pcie._dcp_cute_common import signal_bytes

import b12x.comm.pcie.pcie_dcp_topk as _topk_mod


class _FakeOwner(PCIeDCPTopKOwnerExchange):
    def __init__(self):
        super().__init__(
            rank=1, world_size=2, device="cpu",
            signal_ptrs=(10, 20), staging0_ptrs=(30, 40), staging1_ptrs=(62, 72),
            max_rows=8, topk=4,
        )
        shape = (self.max_owner_rows, self.world_size * self.topk)
        self._candidate_views = tuple(
            (torch.empty(shape, dtype=torch.int32), torch.empty(shape, dtype=torch.float32))
            for _ in range(2)
        )
        self.stage_calls = []

    def _launch_stage(self, local_indices, local_scores, *, slot, rows, threads, blocks, wait_for_prior_consumer):
        owner_rows = rows // self.world_size
        row_slice = slice(self.rank * owner_rows, (self.rank + 1) * owner_rows)
        views = self._candidate_views[slot]
        views[0][:owner_rows].copy_(local_indices[row_slice].repeat(1, self.world_size))
        views[1][:owner_rows].copy_(local_scores[row_slice].repeat(1, self.world_size))
        self.stage_calls.append((slot, threads, blocks, wait_for_prior_consumer))


def _make_owner():
    return _FakeOwner()


def test_candidate_staging_layout_is_aligned():
    owner = _candidate_staging_layout(signal_bytes=513, max_rows=8, topk=4, world_size=2)
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
            assert torch.equal(owner_indices[owner_row, rank_slice], indices[rank, global_row])
            assert torch.equal(owner_scores[owner_row, rank_slice].view(torch.int32), score_bits[rank, global_row])


def test_owner_dispatches_and_disposes():
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    candidate_indices, candidate_scores = owner.stage_candidates(indices, scores)
    assert torch.equal(candidate_indices, indices[4:].repeat(1, 2))
    assert torch.equal(candidate_scores, scores[4:].repeat(1, 2))
    assert owner.stage_calls == [(0, 512, 1, False)]
    owner.close()
    assert owner._closed


@pytest.mark.parametrize("eager_calls, expected_slot", [(0, 0), (1, 1), (2, 0)])
def test_graph_capture_pins_the_next_staging_slot_and_enables_prior_wait(
    monkeypatch, eager_calls, expected_slot,
):
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    capture_state = False
    monkeypatch.setattr("b12x.comm.pcie.pcie_dcp_topk._is_current_stream_capturing", lambda _device: capture_state)
    monkeypatch.setattr(owner, "prepare_graph", lambda: None)
    monkeypatch.setattr("b12x.comm.pcie._dcp_topk_cute.is_topk_stage_prepared", lambda *args: True)
    for _ in range(eager_calls):
        owner.stage_candidates(indices, scores)
    capture_state = True
    with owner.capture(rows=8):
        first_indices, first_scores = owner.stage_candidates(indices, scores)
    capture_state = False
    second_indices, second_scores = owner.stage_candidates(indices + 100, scores + 1)
    capture_state = True
    with owner.capture(rows=8):
        third_indices, third_scores = owner.stage_candidates(indices + 200, scores + 2)
    assert owner._graph_slot == expected_slot
    assert first_indices.data_ptr() == second_indices.data_ptr()
    assert second_indices.data_ptr() == third_indices.data_ptr()
    eager_stages = [(slot % 2, 512, 1, False) for slot in range(eager_calls)]
    assert owner.stage_calls == eager_stages + [
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
    ]
    assert torch.equal(third_indices, (indices + 200)[4:].repeat(1, 2))


def test_owner_rejects_invalid_inputs():
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


def test_configuration_rejects_invalid_capacity_and_topk():
    with pytest.raises(ValueError, match="multiple of 4"):
        _candidate_staging_layout(signal_bytes=256, max_rows=8, topk=6, world_size=2)
    with pytest.raises(ValueError, match="divisible"):
        PCIeDCPTopKOwnerExchange(
            rank=0, world_size=4, device="cpu",
            signal_ptrs=(1, 2, 3, 4), staging0_ptrs=(5, 6, 7, 8),
            staging1_ptrs=(9, 10, 11, 12), max_rows=6, topk=4,
        )


def test_constructor_rejects_invalid_launch_protocol():
    with pytest.raises(ValueError, match="threads"):
        PCIeDCPTopKOwnerExchange(
            rank=0, world_size=2, device="cpu",
            signal_ptrs=(1, 2), staging0_ptrs=(3, 4), staging1_ptrs=(5, 6),
            max_rows=8, topk=4, threads=31,
        )
    with pytest.raises(ValueError, match="block_limit"):
        PCIeDCPTopKOwnerExchange(
            rank=0, world_size=2, device="cpu",
            signal_ptrs=(1, 2), staging0_ptrs=(3, 4), staging1_ptrs=(5, 6),
            max_rows=8, topk=4, block_limit=129,
        )


def test_cuda_direct_constructor_requires_exchange_group():
    with pytest.raises(ValueError, match="exchange_group is required"):
        PCIeDCPTopKOwnerExchange(
            rank=0, world_size=2, device="cuda:0",
            signal_ptrs=(1, 2), staging0_ptrs=(3, 4), staging1_ptrs=(5, 6),
            max_rows=8, topk=4,
        )


def test_cuda_direct_constructor_with_group_requires_factory():
    with pytest.raises(ValueError, match="from_exchange_group"):
        PCIeDCPTopKOwnerExchange(
            rank=0, world_size=2, device="cuda:0",
            signal_ptrs=(1, 2), staging0_ptrs=(3, 4), staging1_ptrs=(5, 6),
            max_rows=8, topk=4, exchange_group=object(),
        )


# ---------------------------------------------------------------------------
# Issue #178: collective DCP Top-K contract agreement before IPC allocation.
# ---------------------------------------------------------------------------


def _base_layout():
    return _candidate_staging_layout(signal_bytes=_SIGNAL_BYTES, max_rows=8, topk=4, world_size=2)


def _base_contract():
    layout = _base_layout()
    return _dcp_topk_runtime_contract(
        world_size=2, max_rows=8, topk=4, layout=layout, threads=512, block_limit=128,
    )


_CONTRACT_FIELDS = {
    "abi_version": 1,
    "topology_world": 2,
    "capacity_max_rows": 4,
    "capacity_topk": 6,
    "shape_candidate": 8,
    "stride_candidate": 9,
    "offset_staging0": 15,
    "offset_staging1": 16,
    "wire_codec_index": 17,
    "wire_codec_score": 18,
    "signal_max_blocks": 19,
    "signal_bytes": 20,
    "schedule_threads": 23,
    "schedule_block_limit": 24,
    "stream_affine": 25,
}


def _mismatched_contract(field):
    contract = list(_base_contract())
    idx = _CONTRACT_FIELDS[field]
    original = contract[idx]
    if isinstance(original, bool):
        contract[idx] = not original
    elif isinstance(original, int):
        contract[idx] = original + 1
    elif isinstance(original, str):
        contract[idx] = original + "_mismatched"
    elif isinstance(original, tuple):
        contract[idx] = tuple(v + 1 for v in original)
    else:
        contract[idx] = "__MISMATCHED__"
    return tuple(contract)


@pytest.mark.parametrize("field, label", [
    ("capacity_max_rows", "capacity"),
    ("capacity_topk", "capacity"),
    ("offset_staging0", "offset"),
    ("offset_staging1", "offset"),
    ("wire_codec_index", "wire mode/codec"),
    ("wire_codec_score", "wire mode/codec"),
    ("topology_world", "topology"),
    ("signal_max_blocks", "signal ABI"),
    ("signal_bytes", "signal ABI"),
    ("schedule_threads", "schedule"),
    ("schedule_block_limit", "schedule"),
    ("stream_affine", "stream affinity"),
    ("abi_version", "ABI version"),
    ("shape_candidate", "shape/stride"),
    ("stride_candidate", "shape/stride"),
])
def test_contract_single_field_mismatch_detection(field, label):
    canonical = _base_contract()
    mismatched = _mismatched_contract(field)
    assert canonical != mismatched
    diffs = [i for i, (a, b) in enumerate(zip(canonical, mismatched, strict=True)) if a != b]
    assert len(diffs) == 1
    assert diffs[0] == _CONTRACT_FIELDS[field]


def test_contract_exact_match_is_identical():
    assert _base_contract() == _base_contract()


def test_contract_includes_all_required_fields():
    contract = _base_contract()
    assert len(contract) == 27
    assert isinstance(contract[0], int)  # version
    assert isinstance(contract[1], int)  # ABI version
    assert isinstance(contract[2], int)  # world_size
    assert isinstance(contract[3], tuple)  # supported world sizes
    assert all(isinstance(contract[i], int) for i in range(4, 8))  # capacity
    assert isinstance(contract[8], tuple)  # shape
    assert isinstance(contract[9], tuple)  # stride
    assert isinstance(contract[10], int)  # score plane offset
    assert all(isinstance(contract[i], int) for i in range(11, 14))  # alloc sizes
    assert all(isinstance(contract[i], int) for i in range(14, 17))  # offsets
    assert isinstance(contract[17], str)  # wire codec
    assert isinstance(contract[18], str)  # wire codec
    assert all(isinstance(contract[i], int) for i in range(19, 23))  # signal ABI
    assert isinstance(contract[23], int)  # threads
    assert isinstance(contract[24], int)  # block_limit
    assert isinstance(contract[25], bool)  # stream_affine (always True)
    assert isinstance(contract[26], int)  # IPC alignment


def test_topology_record_structure():
    record = _dcp_topk_topology_record(rank=0, device=torch.device("cpu"))
    assert len(record) == 3
    assert record[0] == 0
    assert isinstance(record[1], str)
    assert isinstance(record[2], str)


def test_verify_topology_rejects_duplicate_gpu(monkeypatch):
    record_a = (0, "host", "GPU-uuid-0")
    record_b = (1, "host", "GPU-uuid-0")
    monkeypatch.setattr(_topk_mod, "_broadcast_gather_object", lambda obj, group: [record_a, record_b])
    with pytest.raises(RuntimeError, match="duplicate GPU UUID"):
        _verify_dcp_topk_topology(exchange_group=object(), topology=record_a)


def test_verify_topology_rejects_multi_host(monkeypatch):
    record_a = (0, "host-a", "GPU-uuid-0")
    record_b = (1, "host-b", "GPU-uuid-1")
    monkeypatch.setattr(_topk_mod, "_broadcast_gather_object", lambda obj, group: [record_a, record_b])
    with pytest.raises(RuntimeError, match="multiple hosts"):
        _verify_dcp_topk_topology(exchange_group=object(), topology=record_a)


def test_verify_topology_rejects_empty_uuid(monkeypatch):
    record_a = (0, "host", "")
    record_b = (1, "host", "GPU-uuid-1")
    monkeypatch.setattr(_topk_mod, "_broadcast_gather_object", lambda obj, group: [record_a, record_b])
    with pytest.raises(RuntimeError, match="UUID"):
        _verify_dcp_topk_topology(exchange_group=object(), topology=record_a)


def test_verify_topology_rejects_noncontiguous_ranks(monkeypatch):
    record_a = (0, "host", "GPU-uuid-0")
    record_b = (2, "host", "GPU-uuid-1")
    monkeypatch.setattr(_topk_mod, "_broadcast_gather_object", lambda obj, group: [record_a, record_b])
    with pytest.raises(RuntimeError, match="contiguous"):
        _verify_dcp_topk_topology(exchange_group=object(), topology=record_a)


def test_verify_topology_accepts_unique(monkeypatch):
    record_a = (0, "host", "GPU-uuid-0")
    record_b = (1, "host", "GPU-uuid-1")
    monkeypatch.setattr(_topk_mod, "_broadcast_gather_object", lambda obj, group: [record_a, record_b])
    _verify_dcp_topk_topology(exchange_group=object(), topology=record_a)


def test_require_collective_contract_rejects_mismatch(monkeypatch):
    """Exercise the actual _require_collective_contract production gate."""
    canonical = _base_contract()
    mismatched = _mismatched_contract("capacity_max_rows")
    monkeypatch.setattr(
        _topk_mod,
        "_broadcast_gather_object",
        lambda contract, group: [contract, mismatched],
    )
    with pytest.raises(RuntimeError, match="contract differs across ranks"):
        _topk_mod._require_collective_contract(
            owner="test",
            exchange_group=object(),
            contract=canonical,
        )


def test_require_collective_contract_accepts_match(monkeypatch):
    """Exact match must pass the production gate."""
    canonical = _base_contract()
    monkeypatch.setattr(
        _topk_mod,
        "_broadcast_gather_object",
        lambda contract, group: [contract, contract],
    )
    _topk_mod._require_collective_contract(
        owner="test",
        exchange_group=object(),
        contract=canonical,
    )
