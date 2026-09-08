"""Programmatic dependent launches behind the PCIe collectives on nine GPUs.

The collective kernels trigger ``griddepcontrol.launch_dependents`` as their
first statement. A dependent launched with the programmatic-stream-
serialization attribute may therefore start while the collective still waits
on its peers; its ``griddepcontrol.wait`` must make it observe the complete
output. Every all-reduce below runs once alone and once with the dependent
copy kernel behind it (with and without the attribute), eagerly and inside a
CUDA graph, and both the collective's output and the dependent's copy are
compared bitwise with the run alone. Set ``B12X_RUN_PCIE_TP9_TEST=1`` on nine
idle GPUs; ``B12X_PCIE_PDL_ITERATIONS`` (default 40) sets the repeat count.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, port: int) -> None:
    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16
    from tests.comm.pdl_dependent import compile_wait_then_copy

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=9,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    group = dist.group.WORLD
    iterations = int(os.getenv("B12X_PCIE_PDL_ITERATIONS", "40"))
    copy = compile_wait_then_copy()

    def record(stage: str) -> None:
        torch.cuda.synchronize(device)
        dist.barrier()
        if rank == 0:
            print(json.dumps({"stage": stage, "world_size": 9}), flush=True)

    def pattern(rows: int, width: int, scale: int) -> torch.Tensor:
        # Position-dependent integers keep every partial sum exact in bf16.
        return (
            (torch.arange(rows * width, device=device).remainder(5) + 1) * scale
        ).to(torch.bfloat16).view(rows, width)

    def scale_of(iteration: int) -> int:
        # Rank 8 varies its input so a stale replay is visible.
        return rank + 1 + (iteration if rank == 8 else 0)

    def total_of(iteration: int) -> int:
        return 45 + iteration

    for mode in ("pull", "push"):
        twoshot = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group,
            device=device,
            max_rows=49149,
            row_elems=8,
        )
        twoshot.all_reduce_mode = mode
        for rows, width in ((4, 7168), (4, 3584), (16, 7168)):
            inp = pattern(rows, width, scale_of(0))
            out = torch.empty_like(inp)
            dependent = torch.empty_like(inp)
            assert twoshot.accepts(inp)
            for use_pdl in (True, False):
                for iteration in range(iterations):
                    inp.copy_(pattern(rows, width, scale_of(iteration)))
                    expected = pattern(rows, width, total_of(iteration))
                    dependent.zero_()
                    twoshot.all_reduce(inp, out=out)
                    copy(out, dependent, use_pdl)
                    torch.cuda.synchronize(device)
                    assert torch.equal(out, expected), (mode, rows, width, use_pdl)
                    assert torch.equal(dependent, expected), (
                        mode,
                        rows,
                        width,
                        use_pdl,
                        "dependent read an incomplete two-shot output",
                    )
            for use_pdl in (True, False):
                graph = torch.cuda.CUDAGraph()
                with twoshot.capture(), torch.cuda.graph(graph):
                    twoshot.all_reduce(inp, out=out)
                    copy(out, dependent, use_pdl)
                for iteration in range(3):
                    inp.copy_(pattern(rows, width, scale_of(iteration)))
                    dependent.zero_()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    expected = pattern(rows, width, total_of(iteration))
                    assert torch.equal(out, expected), (mode, rows, width, use_pdl)
                    assert torch.equal(dependent, expected), (
                        mode,
                        rows,
                        width,
                        use_pdl,
                        "graph dependent read an incomplete two-shot output",
                    )
                del graph
        torch.cuda.synchronize(device)
        del twoshot
        record(f"twoshot_{mode}_pdl_dependent_passed")

    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=group,
        device=device,
        max_input_bytes=1 << 20,
        max_concurrent_channels=3,
    )
    pool.prepare_channels(("eager", "graph-attr", "graph-noattr"))
    inp = torch.full((4, 7168), rank + 1, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    dependent = torch.empty_like(inp)
    for use_pdl in (True, False):
        for iteration in range(iterations):
            inp.fill_(scale_of(iteration))
            dependent.zero_()
            pool.all_reduce(inp, out=out, channel_id="eager")
            copy(out, dependent, use_pdl)
            torch.cuda.synchronize(device)
            expected = torch.full_like(inp, total_of(iteration))
            assert torch.equal(out, expected), ("oneshot", use_pdl)
            assert torch.equal(dependent, expected), (
                "oneshot",
                use_pdl,
                "dependent read an incomplete one-shot output",
            )
    for use_pdl, channel in ((True, "graph-attr"), (False, "graph-noattr")):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            pool.all_reduce(inp, out=out, channel_id=channel)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with (
            pool.capture(stream, channel_id=channel),
            torch.cuda.graph(graph, stream=stream),
        ):
            pool.all_reduce(inp, out=out, channel_id=channel)
            copy(out, dependent, use_pdl)
        for iteration in range(3):
            inp.fill_(scale_of(iteration))
            dependent.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            expected = torch.full_like(inp, total_of(iteration))
            assert torch.equal(out, expected), ("oneshot graph", use_pdl)
            assert torch.equal(dependent, expected), (
                "oneshot graph",
                use_pdl,
                "graph dependent read an incomplete one-shot output",
            )
        del graph
    pool.close()
    record("oneshot_pdl_dependent_passed")
    dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1",
    reason="requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1",
)
def test_tp9_pdl_dependents_observe_complete_collectives() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 9:
        pytest.skip("nine CUDA devices are required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=9, join=True)
