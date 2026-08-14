"""Gated two-rank NCCL/CUDA test for DCP Top-K collective contract validation.

Run with:

    B12X_PCIE_TEST_DCP_TOPK_CONTRACT=1 CUDA_VISIBLE_DEVICES=0,1 \
    python -m pytest tests/comm/test_pcie_dcp_topk_contract_gpu.py -xvs
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
from b12x.comm.pcie import pcie_oneshot as _oneshot_mod


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _expect_collective_rejection(monkeypatch_target, operation) -> None:
    """Assert every rank rejects before IPC open, then rendezvous."""
    # Track whether IPC allocation was attempted.
    ipc_opened = [False]
    original_alloc = _oneshot_mod.PCIeOneshotAllReduce._allocate_shared_buffer

    def tracking_alloc(*args, **kwargs):
        ipc_opened[0] = True
        return original_alloc(*args, **kwargs)

    _oneshot_mod.PCIeOneshotAllReduce._allocate_shared_buffer = tracking_alloc
    try:
        with pytest.raises(RuntimeError):
            operation()
    finally:
        _oneshot_mod.PCIeOneshotAllReduce._allocate_shared_buffer = original_alloc
    assert not ipc_opened[0], "IPC allocation must not start when the contract mismatches"
    dist.barrier()


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    import b12x.comm.pcie.pcie_dcp_topk as _topk_mod
    group = dist.group.WORLD
    try:
        _expect_collective_rejection(
            None,
            lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                exchange_group=group, device=device,
                max_rows=8 if rank == 0 else 16, topk=4,
            ),
        )

        # Divergent topk.
        _expect_collective_rejection(
            None,
            lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                exchange_group=group, device=device,
                max_rows=8, topk=4 if rank == 0 else 8,
            ),
        )

        # Divergent threads.
        _expect_collective_rejection(
            None,
            lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                exchange_group=group, device=device,
                max_rows=8, topk=4,
                threads=512 if rank == 0 else 256,
            ),
        )

        # Divergent block_limit.
        _expect_collective_rejection(
            None,
            lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                exchange_group=group, device=device,
                max_rows=8, topk=4,
                block_limit=128 if rank == 0 else 64,
            ),
        )

        # Asymmetric argument parse failure: rank 1 passes object() for
        # max_rows whose int() raises. Coordinated error envelope must
        # reject every rank before IPC allocation.
        _expect_collective_rejection(
            None,
            lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                exchange_group=group, device=device,
                max_rows=8 if rank == 0 else object(), topk=4,
            ),
        )

        original_contract = _topk_mod._dcp_topk_runtime_contract
        # Asymmetric contract field mutation: rank 1's contract builder
        # returns a contract with a mutated staging offset. Must fail every
        # rank before IPC allocation via _require_collective_contract.
        def mutating_contract(*args, **kwargs):
            contract = original_contract(*args, **kwargs)
            if rank == 1:
                contract_list = list(contract)
                contract_list[15] = contract_list[15] + 256  # staging0_offset
                return tuple(contract_list)
            return contract

        _topk_mod._dcp_topk_runtime_contract = mutating_contract
        try:
            _expect_collective_rejection(
                None,
                lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                    exchange_group=group, device=device, max_rows=8, topk=4,
                ),
            )
        finally:
            _topk_mod._dcp_topk_runtime_contract = original_contract

        # Asymmetric topology: rank 1 returns a duplicate UUID.
        original_topology = _topk_mod._dcp_topk_topology_record

        def dup_uuid_topology(*, rank, device):
            record = original_topology(rank=rank, device=device)
            return (record[0], record[1], "DUPLICATE-UUID")

        _topk_mod._dcp_topk_topology_record = dup_uuid_topology
        try:
            _expect_collective_rejection(
                None,
                lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                    exchange_group=group, device=device, max_rows=8, topk=4,
                ),
            )
        finally:
            _topk_mod._dcp_topk_topology_record = original_topology

        # Asymmetric contract-builder failure: rank 1's contract builder
        # raises. Must fail every rank before IPC allocation.

        original_contract = _topk_mod._dcp_topk_runtime_contract

        def failing_contract(*args, **kwargs):
            if rank == 1:
                raise RuntimeError("injected contract builder failure")
            return original_contract(*args, **kwargs)

        _topk_mod._dcp_topk_runtime_contract = failing_contract
        try:
            _expect_collective_rejection(
                None,
                lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                    exchange_group=group, device=device, max_rows=8, topk=4,
                ),
            )
        finally:
            _topk_mod._dcp_topk_runtime_contract = original_contract

        # Injected prepare_topk_stage (compiler) failure on rank 1.
        import b12x.comm.pcie._dcp_topk_cute as _cute_mod
        original_prepare = _cute_mod.prepare_topk_stage

        def failing_prepare(*args, **kwargs):
            if rank == 1:
                raise RuntimeError("injected compiler failure")
            original_prepare(*args, **kwargs)

        _cute_mod.prepare_topk_stage = failing_prepare
        try:
            _expect_collective_rejection(
                None,
                lambda: PCIeDCPTopKOwnerExchange.from_exchange_group(
                    exchange_group=group, device=device, max_rows=8, topk=4,
                ),
            )
        finally:
            _cute_mod.prepare_topk_stage = original_prepare

        # Direct CUDA constructor without exchange_group rejected.
        with pytest.raises(ValueError, match="exchange_group is required"):
            PCIeDCPTopKOwnerExchange(
                rank=rank, world_size=world_size, device=device,
                signal_ptrs=tuple(0 for _ in range(world_size)),
                staging0_ptrs=tuple(0 for _ in range(world_size)),
                staging1_ptrs=tuple(0 for _ in range(world_size)),
                max_rows=8, topk=4,
            )
        dist.barrier()

        # Direct CUDA constructor WITH exchange_group but no factory token.
        with pytest.raises(ValueError, match="from_exchange_group"):
            PCIeDCPTopKOwnerExchange(
                rank=rank, world_size=world_size, device=device,
                signal_ptrs=tuple(0 for _ in range(world_size)),
                staging0_ptrs=tuple(0 for _ in range(world_size)),
                staging1_ptrs=tuple(0 for _ in range(world_size)),
                max_rows=8, topk=4, exchange_group=group,
            )
        dist.barrier()

        # Exact match: construction proceeds.

        # Post-allocation constructor failure: inject _tensor_from_cuda_pointer
        # failure on rank 1. Every rank must receive setup failure, rendezvous,
        # then successfully construct a fresh owner to prove the collective
        # remains usable and ownership was not leaked/double-freed.
        original_tensor_fn = _topk_mod._tensor_from_cuda_pointer

        class _FailTensor:
            pass

        def failing_tensor_fn(pointer, shape, *, dtype, device):
            if rank == 1:
                raise RuntimeError("injected tensor construction failure")
            return original_tensor_fn(pointer, shape, dtype=dtype, device=device)

        _topk_mod._tensor_from_cuda_pointer = failing_tensor_fn
        try:
            with pytest.raises(RuntimeError):
                PCIeDCPTopKOwnerExchange.from_exchange_group(
                    exchange_group=group, device=device, max_rows=8, topk=4,
                )
            dist.barrier()
        finally:
            _topk_mod._tensor_from_cuda_pointer = original_tensor_fn

        # Fresh construction after rollback must succeed.
        fresh_owner = PCIeDCPTopKOwnerExchange.from_exchange_group(
            exchange_group=group, device=device, max_rows=8, topk=4,
        )
        assert fresh_owner.world_size == world_size
        fresh_owner.close_coordinated()
        dist.barrier()
        owner = PCIeDCPTopKOwnerExchange.from_exchange_group(
            exchange_group=group, device=device, max_rows=8, topk=4,
        )
        assert owner.world_size == world_size

        # Eager stage with matching rows.
        indices = torch.arange(8 * 4, dtype=torch.int32, device=device).reshape(8, 4)
        scores = torch.arange(8 * 4, dtype=torch.float32, device=device).reshape(8, 4)
        candidate_indices, candidate_scores = owner.stage_candidates(indices, scores)
        assert candidate_indices.shape == (4, world_size * 4)
        dist.barrier()

        # Eager stage divergence: rank 0 rows=8, rank 1 rows=4.
        wrong_indices = torch.arange(4 * 4, dtype=torch.int32, device=device).reshape(4, 4)
        wrong_scores = torch.arange(4 * 4, dtype=torch.float32, device=device).reshape(4, 4)
        with pytest.raises(RuntimeError, match="pre-launch verdict differs"):
            owner.stage_candidates(
                indices if rank == 0 else wrong_indices,
                scores if rank == 0 else wrong_scores,
            )
        dist.barrier()

        # Asymmetric invalid capture rows: rank 0 passes 8, rank 1 passes 7
        # (not divisible by world_size). The pre-capture collective must
        # reject every rank coherently.
        with pytest.raises(RuntimeError), owner.capture(rows=8 if rank == 0 else 7):
            pass
        dist.barrier()

        # Capture-body failure: rank 1 passes wrong-row tensor inside
        # capture. The post-capture exchange must reject every rank.
        capture_owner = PCIeDCPTopKOwnerExchange.from_exchange_group(
            exchange_group=group, device=device, max_rows=8, topk=4,
        )
        graph2 = torch.cuda.CUDAGraph(keep_graph=True)
        dist.barrier()
        body_error = None
        try:
            with capture_owner.capture(rows=8), torch.cuda.graph(graph2):
                # Rank 1 passes 4-row tensor but capture agreed on 8.
                stage_indices = indices if rank == 0 else wrong_indices
                stage_scores = scores if rank == 0 else wrong_scores
                capture_owner.stage_candidates(stage_indices, stage_scores)
        except Exception as exc:
            body_error = exc
        assert body_error is not None
        dist.barrier()
        capture_owner.close_coordinated()

        # Graph capture with matching rows.
        graph_owner = PCIeDCPTopKOwnerExchange.from_exchange_group(
            exchange_group=group, device=device, max_rows=8, topk=4,
        )
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        dist.barrier()
        with graph_owner.capture(rows=8), torch.cuda.graph(graph):
            graph_indices, graph_scores = graph_owner.stage_candidates(indices, scores)
        assert graph.raw_cuda_graph() is not None
        graph_owner.close_coordinated()
        owner.close_coordinated()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_dcp_topk_contract_divergence_rejects_all_ranks_before_ipc() -> None:
    if os.getenv("B12X_PCIE_TEST_DCP_TOPK_CONTRACT") != "1":
        pytest.skip("set B12X_PCIE_TEST_DCP_TOPK_CONTRACT=1 to run this test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip("two CUDA devices are required")
    mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
