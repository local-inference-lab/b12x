"""Correctness for the lossless BF16 PCIe two-shot all-reduce.

Run with torchrun on 2, 4 or 8 GPUs:

    python -m torch.distributed.run --nproc-per-node=4 \
        tests/comm/test_pcie_twoshot_bf16.py

Every all-reduce is checked against the exact fp32 sum of the gathered
inputs: the kernel accumulates in fp32 in a fixed rank order and rounds once,
so each output element must lie within one bf16 rounding of the exact sum,
and repeated calls (eager and graph replay) must be bitwise identical.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16, _make_layout

ROW_ELEMS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_ROW_ELEMS", "4096"))
MAX_ROWS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_MAX_ROWS", "512"))
ROWS = (8, 16, 32, 64, 96, 128, 192, 256)


def test_layout_scales_with_rows_and_ranks() -> None:
    base = _make_layout(64, ROW_ELEMS, 4)
    assert base.pack_stride > 0 and base.slot_bytes > 0
    assert base.reduced_offset >= 0 and base.slab_bytes >= base.slot_bytes
    taller = _make_layout(128, ROW_ELEMS, 4)
    assert taller.slab_bytes > base.slab_bytes
    wider = _make_layout(64, 2 * ROW_ELEMS, 4)
    assert wider.slab_bytes > base.slab_bytes


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
    rows = 64
    static = _payload(4242 + rank, rows, pool.device)
    exact = _exact_sum(static, world)
    eager = pool.all_reduce(static)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with pool.capture():
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
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.equal(captured, eager), "graph replay must match the eager result"
    _assert_one_rounding(captured, exact)


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


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
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
    for step in range(3):  # exercises the double-buffered staging slots
        for rows in ROWS:
            if rows % world == 0 and rows <= MAX_ROWS:
                _check_all_reduce(pool, rank, world, rows, step)
    _check_graph_capture(pool, rank, world)
    dist.barrier()
    if rank == 0:
        print(f"pcie_twoshot_bf16 correctness OK ({world} ranks, rows {ROWS})")
    pool.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
