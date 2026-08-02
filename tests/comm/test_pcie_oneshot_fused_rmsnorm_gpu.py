from __future__ import annotations

import ctypes
import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import runtime as cudart

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_ONESHOT_RMS_TEST") != "1",
    reason=(
        "set SPARKINFER_RUN_PCIE_ONESHOT_RMS_TEST=1 to run PCIe oneshot fused "
        "RMSNorm GPU tests"
    ),
)
TEST_TIMEOUT_SECONDS = float(
    os.getenv("SPARKINFER_PCIE_ONESHOT_RMS_TIMEOUT_SECONDS", "300")
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = inp.clone()
    dist.all_reduce(reduced)
    residual_out = reduced + residual
    variance = residual_out.float().square().mean(dim=-1, keepdim=True)
    out = residual_out.float() * torch.rsqrt(variance + epsilon)
    out = (out * weight.float()).to(inp.dtype)
    return out, residual_out


def _make_inputs(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(rows * hidden_size, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden_size)
    inp = (torch.sin(values * 0.013 + rank * 0.17 + iteration * 0.11) * 0.25).to(dtype)
    residual = (torch.cos(values * 0.007 + rank * 0.03 + iteration * 0.19) * 0.5).to(
        dtype
    )
    weight = torch.linspace(0.5, 1.5, hidden_size, device=device, dtype=dtype)
    return inp, residual, weight


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


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


def _dim3_tuple(value) -> tuple[int, int, int]:
    return int(value.x), int(value.y), int(value.z)


def _cuda_graph_kernel_chain(
    graph: torch.cuda.CUDAGraph,
) -> tuple[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    tuple[tuple[int, int, int], tuple[int, int, int]],
]:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    assert result == cudart.cudaError_t.cudaSuccess
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    assert result == cudart.cudaError_t.cudaSuccess
    assert returned_nodes == num_nodes
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_nodes = []
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        assert result == cudart.cudaError_t.cudaSuccess
        if node_type == kernel_type:
            kernel_nodes.append(node)
    assert len(kernel_nodes) == 2, (
        "staged fused capture must contain one slot-control node followed by "
        f"one worker node, found {len(kernel_nodes)} kernel nodes"
    )

    edge_query = cudart.cudaGraphGetEdges(graph_handle)
    assert edge_query[0] == cudart.cudaError_t.cudaSuccess
    num_edges = int(edge_query[-1])
    edges = cudart.cudaGraphGetEdges(graph_handle, num_edges)
    result, from_nodes, to_nodes, returned_edges = (
        edges[0],
        edges[1],
        edges[2],
        edges[-1],
    )
    assert result == cudart.cudaError_t.cudaSuccess
    assert returned_edges == num_edges
    kernel_edges = []
    for source, destination in zip(
        from_nodes[:num_edges],
        to_nodes[:num_edges],
        strict=True,
    ):
        result, source_type = cudart.cudaGraphNodeGetType(source)
        assert result == cudart.cudaError_t.cudaSuccess
        result, destination_type = cudart.cudaGraphNodeGetType(destination)
        assert result == cudart.cudaError_t.cudaSuccess
        if source_type == kernel_type and destination_type == kernel_type:
            kernel_edges.append((source, destination))
    assert len(kernel_edges) == 1, (
        "staged fused capture must directly order its control node before its "
        f"worker node, found {len(kernel_edges)} kernel-to-kernel edges"
    )

    def geometry(node) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        result, params = cudart.cudaGraphKernelNodeGetParams(node)
        if result != cudart.cudaError_t.cudaSuccess and hasattr(
            cudart, "cudaGraphNodeGetParams"
        ):
            fallback_result, node_params = cudart.cudaGraphNodeGetParams(node)
            assert fallback_result == cudart.cudaError_t.cudaSuccess
            result, params = fallback_result, node_params.kernel
        assert result == cudart.cudaError_t.cudaSuccess
        return _dim3_tuple(params.gridDim), _dim3_tuple(params.blockDim)

    control, worker = map(geometry, kernel_edges[0])
    assert control == ((1, 1, 1), (1, 1, 1))
    assert worker[0][0] > 1
    assert worker[1][0] >= 64
    return control, worker


def _run_eager(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    epsilon = 1e-6
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for rows, hidden_size in (
            (1, 6144),
            (1, 8192),
            (2, 6144),
            (3, 6144),
            (4, 6144),
            (5, 6144),
            (6, 6144),
            (8, 6144),
            (3, 128),
        ):
            if rows * hidden_size * dtype.itemsize > 128 * 1024:
                continue
            inp, residual, weight = _make_inputs(
                rows,
                hidden_size,
                dtype,
                device,
                rank,
            )
            expected_out, expected_residual = _reference(
                inp,
                residual,
                weight,
                epsilon,
            )
            torch.cuda.synchronize(device)
            out, residual_out = pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                channel_id="eager:fused-rmsnorm",
            )
            torch.cuda.synchronize(device)
            _assert_close(out, expected_out, dtype)
            _assert_close(residual_out, expected_residual, dtype)


def _run_graph(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    rows = 4
    hidden_size = 6144
    epsilon = 1e-6
    dtype = torch.bfloat16
    inp, residual, weight = _make_inputs(
        rows,
        hidden_size,
        dtype,
        device,
        rank,
    )
    out = torch.empty_like(inp)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with (
        pool.capture(channel_id="graph:fused-rmsnorm") as channel,
        torch.cuda.graph(graph),
    ):
        pool.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=out,
            residual_out=residual,
            channel_id="graph:fused-rmsnorm",
        )
    # Staged capture is now control + worker, in that order. The registered
    # path still launches only its worker, but public graph capture deliberately
    # requires stable eager staging buffers and therefore exercises this
    # two-node contract.
    _cuda_graph_kernel_chain(graph)
    stream = torch.cuda.current_stream(device)
    offset = 0
    if os.getenv("SPARKINFER_PCIE_ONESHOT_PUSH", "0") not in ("", "0"):
        remote_source = (rank + 1) % dist.get_world_size()
        offset = remote_source * inp.numel() * inp.element_size()
    snapshots = [_local_eager_words(channel, stream, offset=offset)]

    for iteration in range(3):
        next_inp, next_residual, _ = _make_inputs(
            rows,
            hidden_size,
            dtype,
            device,
            rank,
            iteration=iteration + 1,
        )
        inp.copy_(next_inp)
        residual.copy_(next_residual)
        expected_out, expected_residual = _reference(
            inp,
            residual,
            weight,
            epsilon,
        )
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_close(out, expected_out, dtype)
        _assert_close(residual, expected_residual, dtype)
        snapshots.append(_local_eager_words(channel, stream, offset=offset))

    changed_slots = [
        {
            slot
            for slot, (before, after) in enumerate(
                zip(snapshots[index], snapshots[index + 1], strict=True)
            )
            if before != after
        }
        for index in range(3)
    ]
    assert all(len(changed) == 1 for changed in changed_slots)
    assert all(
        changed_slots[index] != changed_slots[index + 1]
        for index in range(len(changed_slots) - 1)
    )


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
        max_input_bytes=128 * 1024,
        max_size=128 * 1024,
        max_concurrent_channels=2,
    )
    try:
        pool.prepare_channels(("eager:fused-rmsnorm", "graph:fused-rmsnorm"))
        pool.for_stream(channel_id="eager:fused-rmsnorm")
        _run_eager(pool, device, rank)
        dist.barrier()
        _run_graph(pool, device, rank)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_oneshot_fused_add_rms_norm_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("SPARKINFER_PCIE_ONESHOT_RMS_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 6, 8, 10):
        pytest.skip("PCIe oneshot only supports world sizes 2, 4, 6, 8, and 10")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    context = mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
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
        "PCIe oneshot fused RMSNorm test exceeded "
        f"{TEST_TIMEOUT_SECONDS:.0f}s; probable collective deadlock"
    )
