from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_ONESHOT_OPPOSITE_ORDER") != "1",
    reason=(
        "set SPARKINFER_RUN_PCIE_ONESHOT_OPPOSITE_ORDER=1 to run the "
        "four-rank opposite-order PCIe oneshot GPU test"
    ),
)

WORLD_SIZE = 4
EAGER_ITERS = int(os.getenv("SPARKINFER_PCIE_OPPOSITE_EAGER_ITERS", "64"))
GRAPH_REPLAYS = int(os.getenv("SPARKINFER_PCIE_OPPOSITE_GRAPH_REPLAYS", "64"))
TEST_TIMEOUT_SECONDS = float(
    os.getenv("SPARKINFER_PCIE_OPPOSITE_TIMEOUT_SECONDS", "180")
)
NUMEL = int(os.getenv("SPARKINFER_PCIE_OPPOSITE_NUMEL", "32768"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _expected(base: float, world_size: int, device: torch.device) -> torch.Tensor:
    rank_sum = world_size * (world_size - 1) // 2
    return torch.tensor(
        world_size * base + rank_sum,
        dtype=torch.float32,
        device=device,
    )


def _assert_reduced(
    out: torch.Tensor,
    base: float,
    world_size: int,
    device: torch.device,
) -> None:
    torch.testing.assert_close(
        out,
        _expected(base, world_size, device).expand_as(out),
        rtol=0,
        atol=0,
    )


def _rank_order(rank: int) -> tuple[str, str]:
    # Every rank executes the same ordered sequence within each channel. Only
    # the cross-channel enqueue order differs, which is valid because each
    # stream owns a distinct signal/staging pair.
    return ("a", "b") if rank % 2 == 0 else ("b", "a")


def _run_eager_opposite_order(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    # Channel construction exchanges IPC handles. All ranks construct A then B
    # so channel identities pair correctly before their GPU enqueue orders
    # intentionally diverge.
    channel_a = pool.for_stream(stream_a)
    dist.barrier()
    channel_b = pool.for_stream(stream_b)
    dist.barrier()
    assert channel_a is not channel_b
    assert channel_a._signal_ptrs != channel_b._signal_ptrs

    inputs = {
        "a": torch.empty(NUMEL, device=device, dtype=torch.float32),
        "b": torch.empty(NUMEL, device=device, dtype=torch.float32),
    }
    outputs = {name: torch.empty_like(inp) for name, inp in inputs.items()}
    streams = {"a": stream_a, "b": stream_b}
    channels = {"a": channel_a, "b": channel_b}

    for iteration in range(EAGER_ITERS):
        bases = {
            "a": float(4 * (iteration % 32)),
            "b": float(1024 + 8 * (iteration % 32)),
        }
        for name in _rank_order(rank):
            with torch.cuda.stream(streams[name]):
                inputs[name].fill_(bases[name] + rank)
                channels[name].all_reduce(inputs[name], out=outputs[name])

        # Both valid channel collectives are now enqueued on every rank. A
        # device-wide wait makes a deadlock visible to the parent timeout.
        torch.cuda.synchronize(device)
        for name in ("a", "b"):
            _assert_reduced(outputs[name], bases[name], world_size, device)


def _capture_channel(
    pool: PCIeOneshotAllReducePool,
    stream: torch.cuda.Stream,
    inp: torch.Tensor,
    out: torch.Tensor,
) -> tuple[object, torch.cuda.CUDAGraph]:
    graph = torch.cuda.CUDAGraph()
    with pool.capture(stream) as channel, torch.cuda.graph(graph, stream=stream):
        channel.all_reduce(inp, out=out)
    return channel, graph


def _run_graph_opposite_order(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    inputs = {
        "a": torch.empty(NUMEL, device=device, dtype=torch.float32),
        "b": torch.empty(NUMEL, device=device, dtype=torch.float32),
    }
    outputs = {name: torch.empty_like(inp) for name, inp in inputs.items()}

    # Capture channel creation also exchanges IPC handles, so all ranks capture
    # A then B. Replays may subsequently be enqueued in opposite channel order.
    channel_a, graph_a = _capture_channel(
        pool,
        stream_a,
        inputs["a"],
        outputs["a"],
    )
    stream_a.synchronize()
    dist.barrier()
    channel_b, graph_b = _capture_channel(
        pool,
        stream_b,
        inputs["b"],
        outputs["b"],
    )
    stream_b.synchronize()
    dist.barrier()
    assert channel_a is not channel_b
    assert channel_a._signal_ptrs != channel_b._signal_ptrs

    streams = {"a": stream_a, "b": stream_b}
    graphs = {"a": graph_a, "b": graph_b}
    for iteration in range(GRAPH_REPLAYS):
        bases = {
            "a": float(256 + 4 * (iteration % 32)),
            "b": float(2048 + 8 * (iteration % 32)),
        }
        for name in _rank_order(rank):
            with torch.cuda.stream(streams[name]):
                inputs[name].fill_(bases[name] + rank)
                graphs[name].replay()

        torch.cuda.synchronize(device)
        for name in ("a", "b"):
            _assert_reduced(outputs[name], bases[name], world_size, device)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=TEST_TIMEOUT_SECONDS),
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=NUMEL * torch.float32.itemsize,
        max_size=NUMEL * torch.float32.itemsize,
    )
    try:
        _run_eager_opposite_order(pool, device, rank, world_size)
        dist.barrier()
        _run_graph_opposite_order(pool, device, rank, world_size)
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        pool.close()
        dist.destroy_process_group()


def _spawn_with_timeout(port: int) -> None:
    context = mp.spawn(
        _worker,
        args=(WORLD_SIZE, port),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if context.join(timeout=min(1.0, remaining), grace_period=5):
            return

    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5)
    for process in context.processes:
        if process.is_alive():
            process.kill()
        process.join()
    pytest.fail(
        "four-rank opposite-order eager/graph PCIe oneshot test exceeded "
        f"{TEST_TIMEOUT_SECONDS:.0f}s; probable collective deadlock"
    )


def test_pcie_oneshot_four_rank_opposite_channel_order() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(
            f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}"
        )
    _spawn_with_timeout(_free_port())
