"""Correctness for the lossless BF16 PCIe two-shot all-reduce.

Run with torchrun on 2, 4 or 8 GPUs:

    NCCL_ALGO=Ring python -m torch.distributed.run --nproc-per-node=4 \
        tests/comm/test_pcie_twoshot_bf16.py

Every all-reduce is checked against the exact fp32 sum of the gathered
inputs: the kernel accumulates in fp32 in a fixed rank order and rounds once,
so each output element must lie within one bf16 rounding of the exact sum,
and repeated calls (eager and graph replay) must be bitwise identical.
"""

from __future__ import annotations

from contextlib import nullcontext
import os

import pytest
import torch
import torch.distributed as dist

from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout

ROW_ELEMS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_ROW_ELEMS", "4096"))
MAX_ROWS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_MAX_ROWS", "512"))
ROWS = (8, 16, 32, 64, 96, 128, 192, 256)


def test_layout_covers_supported_ranks_and_scales_with_capacity() -> None:
    layouts = {}
    for world_size in (2, 4, 8):
        base = _make_layout(64, ROW_ELEMS, world_size)
        assert base.pack_stride > 0 and base.slot_bytes > 0
        assert base.reduced_offset > 0 and base.slab_bytes >= 2 * base.slot_bytes
        taller = _make_layout(128, ROW_ELEMS, world_size)
        assert taller.slab_bytes > base.slab_bytes
        wider = _make_layout(64, 2 * ROW_ELEMS, world_size)
        assert wider.slab_bytes > base.slab_bytes
        layouts[world_size] = base
    assert layouts[2].pack_stride > layouts[4].pack_stride
    assert layouts[4].pack_stride > layouts[8].pack_stride
    assert layouts[2].slab_bytes > layouts[4].slab_bytes
    assert layouts[4].slab_bytes > layouts[8].slab_bytes


def test_all_reduce_only_capture_prepares_only_pull_launchers(monkeypatch) -> None:
    pull_launcher_requests: list[tuple[object, ...]] = []
    collective_contracts: list[tuple[object, ...]] = []

    def unexpected_generic_launcher(*_args):
        raise AssertionError("all_reduce must not use the generic two-shot launcher")

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot_bf16._is_current_stream_capturing",
        lambda _device: False,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot_bf16.get_twoshot_bf16_launcher",
        unexpected_generic_launcher,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot_bf16.get_twoshot_bf16_allreduce_launcher",
        lambda *args: pull_launcher_requests.append(args),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot_bf16._require_collective_contract",
        lambda **kwargs: collective_contracts.append(kwargs["contract"]),
    )

    pool = object.__new__(PCIeTwoShotBF16)
    pool.rank = 1
    pool.world_size = 4
    pool.device = torch.device("cuda", 0)
    pool.exchange_group = None
    pool.row_elems = 16
    pool._slot = 1
    pool._device_slot_selection = False
    pool._device_slot_bias = 0
    pool._capture_context_depth = 0
    pool._closed = False

    with pool.capture(operations=("all_reduce",), threads=256):
        assert pool._capture_context_depth == 1

    assert pool._capture_context_depth == 0
    assert pull_launcher_requests == [
        (4, 1, True, 0, 256, 16, 0),
        (4, 1, True, 1, 256, 16, 0),
    ]
    assert collective_contracts == [(("all_reduce",), 256, False, 1)]


def _payload(seed: int, rows: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, ROW_ELEMS, generator=gen, dtype=torch.float32) * 3.0
    return x.to(device=device, dtype=torch.bfloat16)


def _exact_sum(x: torch.Tensor, world: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    return sum(g.float() for g in gathered)


def _assert_one_rounding(out: torch.Tensor, exact: torch.Tensor) -> None:
    err = (out.float() - exact).abs()
    bound = exact.abs() * 2.0**-8 + 1e-5
    worst = (err - bound).max().item()
    assert bool((err <= bound).all()), (
        f"output exceeds one bf16 rounding by {worst:.3e}"
    )


def _check_all_reduce(
    pool: PCIeTwoShotBF16, rank: int, world: int, rows: int, step: int
) -> None:
    x = _payload(1000 * step + 7 * rows + rank, rows, pool.device)
    assert pool.accepts(x)
    exact = _exact_sum(x, world)
    out = pool.all_reduce(x)
    torch.cuda.synchronize()
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    _assert_one_rounding(out, exact)
    again = pool.all_reduce(x)
    torch.cuda.synchronize()
    assert torch.equal(out, again), "two-shot all-reduce must be deterministic"
    # The bf16 NCCL ring rounds after every hop; the two-shot rounds once.
    ref = x.clone()
    dist.all_reduce(ref)
    assert (out.float() - exact).abs().max() <= (ref.float() - exact).abs().max() + 1e-5


def _check_graph_capture(pool: PCIeTwoShotBF16, rank: int, world: int) -> None:
    rows = min(64, MAX_ROWS)
    rows -= rows % world
    assert rows > 0
    static = _payload(4242 + rank, rows, pool.device)
    exact = _exact_sum(static, world)
    eager = pool.all_reduce(static)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with pool.capture(operations=("all_reduce",)):
        with torch.cuda.stream(stream):
            for _ in range(3):
                pool.all_reduce(static)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = pool.all_reduce(static)
    torch.cuda.synchronize()
    dist.barrier()
    captured_address = captured.data_ptr()
    allocator_reports_request_count = torch.cuda.get_allocator_backend() == "native"
    for _ in range(3):
        allocated_before_replay = torch.cuda.memory_allocated(pool.device)
        allocation_count_before_replay = (
            int(torch.cuda.memory_stats(pool.device)["allocation.all.allocated"])
            if allocator_reports_request_count
            else None
        )
        graph.replay()
        torch.cuda.synchronize()
        if allocation_count_before_replay is not None:
            assert (
                int(torch.cuda.memory_stats(pool.device)["allocation.all.allocated"])
                == allocation_count_before_replay
            )
        assert torch.cuda.memory_allocated(pool.device) == allocated_before_replay
        dist.barrier()
        assert captured.data_ptr() == captured_address
        assert torch.equal(captured, eager), "graph replay must match the eager result"
    _assert_one_rounding(captured, exact)


def _check_rejects_divergent_graph_slot_bias(pool: PCIeTwoShotBF16, rank: int) -> None:
    if rank == 0:
        pool._slot += 1
    try:
        with (
            pytest.raises(
                RuntimeError,
                match="graph slot selection contract differs across ranks",
            ),
            pool.capture(),
        ):
            pass
    finally:
        if rank == 0:
            pool._slot -= 1
    dist.barrier()


def _check_reduce_scatter_all_gather(
    pool: PCIeTwoShotBF16, rank: int, world: int
) -> None:
    rows = MAX_ROWS
    assert rows > 0 and rows % world == 0
    local_rows = rows // world
    payload = _payload(5300 + rank, rows, pool.device)
    exact = _exact_sum(payload, world)

    shard = torch.empty(
        local_rows,
        ROW_ELEMS,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    returned_shard = pool.reduce_scatter(payload, out=shard)
    torch.cuda.synchronize()
    assert returned_shard is shard
    expected_shard = exact[rank * local_rows : (rank + 1) * local_rows]
    _assert_one_rounding(shard, expected_shard)

    gathered = torch.empty_like(payload)
    returned_gather = pool.all_gather(shard, out=gathered)
    torch.cuda.synchronize()
    assert returned_gather is gathered
    reference_shards = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(reference_shards, shard)
    assert torch.equal(gathered, torch.cat(reference_shards, dim=0))


def _check_rejects_unsupported(pool: PCIeTwoShotBF16, world: int) -> None:
    device = pool.device
    assert not pool.accepts(
        torch.empty(MAX_ROWS + world, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    )
    assert not pool.accepts(
        torch.empty(world, ROW_ELEMS, dtype=torch.float16, device=device)
    )
    assert not pool.accepts(
        torch.empty(world, ROW_ELEMS + 8, dtype=torch.bfloat16, device=device)
    )
    if world > 1:
        assert not pool.accepts(
            torch.empty(world + 1, ROW_ELEMS, dtype=torch.bfloat16, device=device)
        )
    storage = torch.empty(
        world * ROW_ELEMS + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    misaligned = storage[1:].view(world, ROW_ELEMS)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0
    assert not pool.accepts(misaligned)


def _check_rejects_invalid_outputs(pool: PCIeTwoShotBF16, world: int) -> None:
    x = _payload(6100 + pool.rank, world, pool.device)
    wrong_shape = torch.empty(
        x.numel(),
        dtype=torch.bfloat16,
        device=pool.device,
    )
    with pytest.raises(ValueError, match="output shape"):
        pool.all_reduce(x, out=wrong_shape)

    storage = torch.empty(
        x.numel() + 1,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    misaligned = storage[1:].view_as(x)
    with pytest.raises(ValueError, match="16-byte aligned"):
        pool.all_reduce(x, out=misaligned)

    local_rows = x.shape[0] // world
    with pytest.raises(TypeError, match="output must be bfloat16"):
        pool.reduce_scatter(
            x,
            out=torch.empty(
                local_rows,
                ROW_ELEMS,
                dtype=torch.float16,
                device=pool.device,
            ),
        )

    shard = _payload(6200 + pool.rank, local_rows, pool.device)
    with pytest.raises(ValueError, match="output shape"):
        pool.all_gather(shard, out=x[:local_rows])

    overlap_storage = torch.empty(
        x.numel() + 8,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    overlap_input = overlap_storage[: x.numel()].view_as(x)
    overlap_output = overlap_storage[8:].view_as(x)
    with pytest.raises(ValueError, match="output must not overlap input"):
        pool.all_reduce(overlap_input, out=overlap_output)

    with pytest.raises(ValueError, match="output must not overlap payload"):
        pool.reduce_scatter(x, out=x[:local_rows])

    gather_storage = torch.empty_like(x)
    gather_payload = gather_storage[:local_rows]
    with pytest.raises(ValueError, match="output must not overlap payload"):
        pool.all_gather(gather_payload, out=gather_storage)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # The error comparison below is specifically against NCCL's BF16 ring.
    os.environ["NCCL_ALGO"] = "Ring"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    pool = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=MAX_ROWS,
        row_elems=ROW_ELEMS,
    )
    pool.prepare_graph()
    _check_rejects_unsupported(pool, world)
    _check_rejects_invalid_outputs(pool, world)
    _check_reduce_scatter_all_gather(pool, rank, world)
    executed_rows = []
    for step in range(3):  # exercises the double-buffered staging slots
        for rows in ROWS:
            if rows % world == 0 and rows <= MAX_ROWS:
                _check_all_reduce(pool, rank, world, rows, step)
                if rows not in executed_rows:
                    executed_rows.append(rows)
    _check_all_reduce(pool, rank, world, MAX_ROWS, step=3)
    if MAX_ROWS not in executed_rows:
        executed_rows.append(MAX_ROWS)
    _check_rejects_divergent_graph_slot_bias(pool, rank)
    _check_graph_capture(pool, rank, world)
    dist.barrier()
    if rank == 0:
        print(
            "pcie_twoshot_bf16 correctness OK "
            f"({world} ranks, all_reduce_rows={tuple(executed_rows)}, "
            f"workspace_max_rows={MAX_ROWS})"
        )
    pool.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
