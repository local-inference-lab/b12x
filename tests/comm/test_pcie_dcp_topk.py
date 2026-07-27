from __future__ import annotations

import pytest
import torch

from sparkinfer.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _candidate_staging_layout,
    owner_stage_reference,
)


class _FakeExt:
    def __init__(self) -> None:
        self.stage_calls: list[tuple] = []
        self.dispose_owner_calls: list[int] = []

    def init_dcp_topk_owner_exchange(self, *args):
        return 1234

    def stage_owner_candidates(
        self, pointer, local_indices, local_scores, threads, block_limit
    ):
        world_size = 2
        rank = 1
        owner_rows = local_indices.shape[0] // world_size
        row_slice = slice(rank * owner_rows, (rank + 1) * owner_rows)
        indices = local_indices[row_slice].repeat(1, world_size)
        scores = local_scores[row_slice].repeat(1, world_size)
        self.stage_calls.append((pointer, threads, block_limit))
        return indices, scores

    def dispose_owner_exchange(self, pointer):
        self.dispose_owner_calls.append(pointer)


def _make_owner(ext: _FakeExt | None = None) -> PCIeDCPTopKOwnerExchange:
    return PCIeDCPTopKOwnerExchange(
        rank=1,
        world_size=2,
        device="cpu",
        signal_ptrs=(10, 20),
        staging0_ptrs=(30, 40),
        staging1_ptrs=(50, 60),
        max_rows=8,
        topk=4,
        ext_module=ext or _FakeExt(),
    )


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
    ext = _FakeExt()
    owner = _make_owner(ext)
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
    assert ext.stage_calls == [(1234, 128, 32)]

    owner.close()
    assert ext.dispose_owner_calls == [1234]


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
            ext_module=_FakeExt(),
        )
