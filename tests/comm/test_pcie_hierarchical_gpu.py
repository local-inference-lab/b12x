from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_hierarchical import (
    OpCatalog,
    OpCatalogEntry,
    PCIeHierarchicalAllReduce,
    _pick_blocks,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_HIERARCHICAL_TEST") != "1",
    reason=(
        "set B12X_RUN_PCIE_HIERARCHICAL_TEST=1 to run TP12 hierarchical GPU tests"
    ),
)

WORLD_SIZE = 12


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _expected(elements: int, world_size: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(elements, device=device, dtype=torch.float32) / 1024.0
    values = [
        (base + float(source_rank)).to(torch.bfloat16).float()
        for source_rank in range(world_size)
    ]
    return torch.stack(values).sum(dim=0).to(torch.bfloat16)


def _worker(rank: int, world_size: int, port: int, mode: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ["B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER"] = str(
        int(mode == "double")
    )
    os.environ["B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION"] = str(
        int(mode == "deferred")
    )
    os.environ["B12X_PCIE_HIERARCHICAL_BF16X2"] = "1"
    os.environ["B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS"] = "7168"
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    # Build a catalog with all element counts used in the test.
    catalog_entries = tuple(
        OpCatalogEntry(
            op_id=f"op_{elements}",
            order=order,
            elements=elements,
            blocks=_pick_blocks(elements),
        )
        for order, elements in enumerate(
            sorted(set([3583, 3584, 7168, 7169]))
        )
    )
    runtime = PCIeHierarchicalAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_elements=7169,
        catalog=OpCatalog(entries=catalog_entries),
    )
    try:
        # Odd/even BF16x2 tails and the scalar path above the vector threshold.
        for elements in (3583, 7168, 7169, 3583):
            base = torch.arange(
                elements, device=device, dtype=torch.float32
            ) / 1024.0
            inp = (base + float(rank)).to(torch.bfloat16)
            out = torch.empty_like(inp)
            output_address = out.data_ptr()
            actual = runtime.all_reduce(inp, out=out)
            torch.cuda.synchronize(device)
            assert actual is out
            assert out.data_ptr() == output_address
            assert bool(torch.isfinite(out).all())
            assert bool(torch.count_nonzero(out))
            torch.testing.assert_close(
                out,
                _expected(elements, world_size, device),
                rtol=0,
                atol=0,
            )

        graph_input = torch.full(
            (3584,), float(rank) + 0.25, device=device, dtype=torch.bfloat16
        )
        graph_output = torch.empty_like(graph_input)
        graph_input_address = graph_input.data_ptr()
        graph_output_address = graph_output.data_ptr()
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            runtime.all_reduce(graph_input, out=graph_output)
        for step in range(2):
            graph_input.fill_(float(rank) + 0.25 + float(step))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            expected_value = sum(
                float(source_rank) + 0.25 + float(step)
                for source_rank in range(world_size)
            )
            expected = torch.full_like(graph_output, expected_value)
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            assert graph_input.data_ptr() == graph_input_address
            assert graph_output.data_ptr() == graph_output_address
    finally:
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ("baseline", "double", "deferred"))
def test_pcie_hierarchical_eager_and_graph_correctness(mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(
            f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _free_port(), mode),
        nprocs=WORLD_SIZE,
        join=True,
    )
