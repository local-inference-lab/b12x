from __future__ import annotations

import ctypes
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_ONESHOT_TORTURE") != "1",
    reason="set SPARKINFER_RUN_PCIE_ONESHOT_TORTURE=1 to run PCIe oneshot CUDA torture tests",
)

TORTURE_EAGER_ITERS = int(
    os.getenv("SPARKINFER_PCIE_ONESHOT_TORTURE_EAGER_ITERS", "256")
)
TORTURE_GRAPH_REPLAYS = int(
    os.getenv("SPARKINFER_PCIE_ONESHOT_TORTURE_GRAPH_REPLAYS", "256")
)
TORTURE_MULTISTREAM_ITERS = int(
    os.getenv("SPARKINFER_PCIE_ONESHOT_TORTURE_MULTISTREAM_ITERS", "256")
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_constant(tensor: torch.Tensor, value: float) -> None:
    expected = torch.full_like(tensor, value)
    torch.testing.assert_close(tensor, expected, rtol=1e-2, atol=1e-2)


def _rank_sum(world_size: int) -> int:
    return world_size * (world_size - 1) // 2


def _local_eager_words(
    channel, stream: torch.cuda.Stream, *, offset: int = 0
) -> tuple[int, int]:
    assert channel._ipc is not None
    assert channel._eager_ptrs is not None
    words = (ctypes.c_uint64(), ctypes.c_uint64())
    for word, slot_ptrs in zip(words, channel._eager_ptrs, strict=True):
        channel._ipc.cudaMemcpyAsync(
            ctypes.addressof(word),
            slot_ptrs[channel.rank] + offset,
            ctypes.sizeof(word),
            int(stream.cuda_stream),
        )
    stream.synchronize()
    return words[0].value, words[1].value


def _run_eager(
    pool: PCIeOneshotAllReducePool, device: torch.device, rank: int, world_size: int
) -> None:
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    numels = (8, 256, 4096, 32768)
    rank_sum = _rank_sum(world_size)

    for dtype in dtypes:
        for numel in numels:
            inp = torch.empty(numel, device=device, dtype=dtype)
            out = torch.empty_like(inp)
            for iteration in range(TORTURE_EAGER_ITERS):
                base = float((iteration % 64) * 3)
                inp.fill_(base + rank)
                pool.all_reduce(inp, out=out, channel_id="eager:default")
                torch.cuda.synchronize(device)
                _assert_constant(out, world_size * base + rank_sum)


def _run_graph_scratch_reuse(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream = torch.cuda.Stream(device=device)
    rank_sum = _rank_sum(world_size)
    layers = 17
    numel = 4096
    dtype = torch.bfloat16
    sources = [torch.empty(numel, device=device, dtype=dtype) for _ in range(layers)]
    scratch = torch.empty(numel, device=device, dtype=dtype)
    outs = [torch.empty(numel, device=device, dtype=dtype) for _ in range(layers)]

    def fill_sources(iteration: int) -> None:
        for layer, source in enumerate(sources):
            source.fill_(float((iteration % 32) * 5 + layer) + rank)

    fill_sources(0)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with (
        pool.capture(stream, channel_id="graph:torture") as channel,
        torch.cuda.graph(graph, stream=stream),
    ):
        for layer in range(layers):
            scratch.copy_(sources[layer])
            channel.all_reduce(scratch, out=outs[layer])
    stream.synchronize()

    for iteration in range(TORTURE_GRAPH_REPLAYS):
        with torch.cuda.stream(stream):
            fill_sources(iteration)
            graph.replay()
        stream.synchronize()
        for layer, out in enumerate(outs):
            base = float((iteration % 32) * 5 + layer)
            _assert_constant(out, world_size * base + rank_sum)

    # A graph with an odd number of collectives must swap which slot receives
    # the final layer on every replay. Both slots are overwritten by this
    # 17-layer graph, so merely counting changed slots cannot identify the
    # selected parity: the final two layer markers must exchange positions.
    snapshots = [_local_eager_words(channel, stream)]
    expected_words = [
        tuple(int(source.view(torch.uint64)[0].item()) for source in sources[-2:])
    ]
    for iteration in (101, 102):
        with torch.cuda.stream(stream):
            fill_sources(iteration)
            graph.replay()
        stream.synchronize()
        snapshots.append(_local_eager_words(channel, stream))
        expected_words.append(
            tuple(int(source.view(torch.uint64)[0].item()) for source in sources[-2:])
        )
        for layer, out in enumerate(outs):
            base = float((iteration % 32) * 5 + layer)
            _assert_constant(out, world_size * base + rank_sum)

    final_slots = []
    for snapshot, expected in zip(snapshots, expected_words, strict=True):
        assert set(snapshot) == set(expected), (
            f"staging slots {snapshot} do not contain the final layer markers "
            f"{expected}"
        )
        final_slots.append(snapshot.index(expected[-1]))
    assert all(
        final_slots[index] != final_slots[index + 1]
        for index in range(len(final_slots) - 1)
    ), f"odd-collective graph did not alternate final staging slots: {final_slots}"


def _run_multistream(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    pool.for_stream(stream_a, channel_id="eager:a")
    pool.for_stream(stream_b, channel_id="eager:b")
    rank_sum = _rank_sum(world_size)

    inp_a = torch.empty(2048, device=device, dtype=torch.float16)
    out_a = torch.empty_like(inp_a)
    inp_b = torch.empty(2048, device=device, dtype=torch.bfloat16)
    out_b = torch.empty_like(inp_b)

    for iteration in range(TORTURE_MULTISTREAM_ITERS):
        base_a = float(iteration % 64)
        base_b = float(100 + (iteration % 64) * 2)
        with torch.cuda.stream(stream_a):
            inp_a.fill_(base_a + rank)
            pool.all_reduce(inp_a, out=out_a, channel_id="eager:a")
        with torch.cuda.stream(stream_b):
            inp_b.fill_(base_b + rank)
            pool.all_reduce(inp_b, out=out_b, channel_id="eager:b")
        stream_a.synchronize()
        stream_b.synchronize()
        _assert_constant(out_a, world_size * base_a + rank_sum)
        _assert_constant(out_b, world_size * base_b + rank_sum)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=1 << 20,
        max_concurrent_channels=2,
    )
    try:
        pool.prepare_channels(("eager:default", "eager:a", "eager:b", "graph:torture"))
        _run_eager(pool, device, rank, world_size)
        dist.barrier()
        _run_graph_scratch_reuse(pool, device, rank, world_size)
        dist.barrier()
        _run_multistream(pool, device, rank, world_size)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_oneshot_eager_graph_and_multistream_torture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    available = torch.cuda.device_count()
    requested = int(os.getenv("SPARKINFER_PCIE_ONESHOT_TORTURE_WORLD_SIZE", "2"))
    if requested not in (2, 4, 6, 8, 10):
        pytest.skip("PCIe oneshot only supports world sizes 2, 4, 6, 8, and 10")
    if available < requested:
        pytest.skip(f"need {requested} CUDA devices, found {available}")
    mp.spawn(_worker, args=(requested, _free_port()), nprocs=requested, join=True)
