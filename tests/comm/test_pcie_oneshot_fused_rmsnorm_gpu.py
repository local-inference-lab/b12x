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

from b12x.comm.pcie.pcie_oneshot import (
    _PACKED_GATHER_MIN_ROWS,
    _SCATTER_GATHER_MIN_ROWS,
    _SCATTER_GATHER_SLOT_MULTIPLIER,
    PCIeOneshotAllReducePool,
    _CuTeOneshotBackend,
    _eager_payload_shards,
)
from tests.comm.pdl_dependent import compile_wait_then_copy


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_RMS_TEST") != "1",
    reason=(
        "set B12X_RUN_PCIE_ONESHOT_RMS_TEST=1 to run PCIe oneshot fused "
        "RMSNorm GPU tests"
    ),
)
TEST_TIMEOUT_SECONDS = float(os.getenv("B12X_PCIE_ONESHOT_RMS_TIMEOUT_SECONDS", "300"))


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


def _make_split_view_inputs(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inp, contiguous_residual, weight = _make_inputs(
        rows,
        hidden_size,
        dtype,
        device,
        rank,
        iteration,
    )
    combined = torch.full(
        (rows, hidden_size * 2),
        -7.0,
        dtype=dtype,
        device=device,
    )
    padding, residual = combined.split(hidden_size, dim=-1)
    residual.copy_(contiguous_residual)
    return inp, residual, weight, padding


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
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
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
    assert len(kernel_nodes) == 1, (
        "CuTe staged fused capture must fold device slot selection into its "
        f"worker, found {len(kernel_nodes)} kernel nodes"
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

    worker = geometry(kernel_nodes[0])
    assert worker[0][0] > 1
    assert worker[1][0] >= 64
    return worker


def _run_eager(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    epsilon = 1e-6
    cases = [
        (dtype, rows, hidden_size)
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
        for rows, hidden_size in (
            (1, 6144),
            (1, 8192),
            (2, 4096),
            (4, 4096),
            (8, 4096),
            (2, 6144),
            (3, 6144),
            (4, 6144),
            (5, 6144),
            (6, 6144),
            (8, 6144),
            (3, 128),
        )
        if rows * hidden_size * dtype.itemsize <= 128 * 1024
    ]
    if os.getenv("B12X_PCIE_ONESHOT_RMS_TOPOLOGY_ONLY", "0") not in ("", "0"):
        hidden_size = 4096 if dist.get_world_size() == 2 else 6144
        cases = [(torch.bfloat16, rows, hidden_size) for rows in (1, 2, 4, 8)]

    for dtype, rows, hidden_size in cases:
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
        wrong_device = (rank + 1) % dist.get_world_size()
        exercise_device_guard = (
            dtype == torch.float16 and rows == 1 and hidden_size == 6144
        )
        if exercise_device_guard:
            torch.cuda.set_device(wrong_device)
        out, residual_out = pool.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            channel_id="eager:fused-rmsnorm",
        )
        if exercise_device_guard:
            assert torch.cuda.current_device() == wrong_device
            torch.cuda.set_device(rank)
        torch.cuda.synchronize(device)
        _assert_close(out, expected_out, dtype)
        _assert_close(residual_out, expected_residual, dtype)


def _run_graph(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    rows = int(os.getenv("B12X_PCIE_ONESHOT_RMS_GRAPH_ROWS", "4"))
    hidden_size = int(os.getenv("B12X_PCIE_ONESHOT_RMS_GRAPH_HIDDEN", "6144"))
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
    # CuTe folds device-side slot selection into the worker, so staged capture
    # preserves the one-kernel production topology.
    _cuda_graph_kernel_chain(graph)
    stream = torch.cuda.current_stream(device)
    world_size = dist.get_world_size()
    topology_remote = (
        (
            world_size == 2
            and os.getenv("B12X_PCIE_TP2_REMOTE_PUSH", "0") not in ("", "0")
        )
        or (
            world_size == 4
            and os.getenv("B12X_PCIE_TP4_REMOTE_PUSH", "0") not in ("", "0")
        )
        or (
            world_size == 8
            and os.getenv("B12X_PCIE_TP8_OWNER_REDUCE", "1") not in ("", "0")
        )
    )
    offset = 0
    if os.getenv("B12X_PCIE_ONESHOT_PUSH", "0") not in ("", "0") or topology_remote:
        if world_size == 8 and topology_remote:
            # TP8 owner mode publishes owner data into each rank's staging
            # slot, so probe the island owner rather than a neighboring rank.
            remote_source = 0 if rank < 4 else 4
        else:
            remote_source = (rank + 1) % world_size
        offset = remote_source * channel.eager_buffer_bytes
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


def _run_tp8_graph_mode_transition(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    """Replay C1 fallback followed by C4 owner reduction on one channel."""

    if dist.get_world_size() != 8 or os.getenv("B12X_PCIE_TP8_OWNER_REDUCE", "1") in (
        "",
        "0",
    ):
        return

    hidden_size = 6144
    epsilon = 1e-6
    dtype = torch.bfloat16
    stream = torch.cuda.Stream(device=device)
    channel_id = "graph:fused-transition"
    channel = pool.for_stream(stream, channel_id=channel_id)

    calls = []
    modes = []
    state = channel._ext._state(channel._ptr)
    for rows, iteration in ((1, 41), (4, 43)):
        inp, residual, weight = _make_inputs(
            rows,
            hidden_size,
            dtype,
            device,
            rank,
            iteration=iteration,
        )
        out = torch.empty_like(inp)
        calls.append((inp, residual, weight, out))
        modes.append(_CuTeOneshotBackend._fused_launch_plan(state, inp).variant.mode)
        with torch.cuda.stream(stream):
            channel.prepare_graph_fused_add_rms_norm(inp)
    assert modes == ["stage_pull", "stage_tp8_owner"]

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with (
        pool.capture(stream=stream, channel_id=channel_id),
        torch.cuda.graph(graph, stream=stream),
    ):
        for inp, residual, weight, out in calls:
            pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                out=out,
                residual_out=residual,
                stream=stream,
                channel_id=channel_id,
            )

    for replay in range(3):
        expected = []
        for rows, (inp, residual, weight, _out) in zip((1, 4), calls, strict=True):
            next_inp, next_residual, _ = _make_inputs(
                rows,
                hidden_size,
                dtype,
                device,
                rank,
                iteration=47 + replay,
            )
            inp.copy_(next_inp)
            residual.copy_(next_residual)
            expected.append(_reference(inp, residual, weight, epsilon))

        graph.replay()
        stream.synchronize()
        for (_inp, residual, _weight, out), (
            expected_out,
            expected_residual,
        ) in zip(calls, expected, strict=True):
            _assert_close(out, expected_out, dtype)
            _assert_close(residual, expected_residual, dtype)


def _run_tp8_split_view_graph(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    if dist.get_world_size() != 8:
        return

    hidden_size = 6144
    epsilon = 1e-6
    dtype = torch.bfloat16
    stream = torch.cuda.Stream(device=device)
    channel_id = "graph:split-residual"
    channel = pool.for_stream(stream, channel_id=channel_id)
    calls = []
    for rows in (4, 8, 16):
        inp, residual, weight, padding = _make_split_view_inputs(
            rows,
            hidden_size,
            dtype,
            device,
            rank,
        )
        assert residual.stride() == (hidden_size * 2, 1)
        out = torch.empty_like(inp)
        calls.append((inp, residual, weight, padding, out))
        with torch.cuda.stream(stream):
            channel.prepare_graph_fused_add_rms_norm(inp)

    graph = torch.cuda.CUDAGraph()
    with (
        pool.capture(stream=stream, channel_id=channel_id),
        torch.cuda.graph(graph, stream=stream),
    ):
        for inp, residual, weight, _padding, out in calls:
            pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                out=out,
                residual_out=residual,
                stream=stream,
                channel_id=channel_id,
            )

    for replay in range(3):
        expected = []
        for rows, (inp, residual, weight, padding, _out) in zip(
            (4, 8, 16), calls, strict=True
        ):
            next_inp, next_residual, _, _ = _make_split_view_inputs(
                rows,
                hidden_size,
                dtype,
                device,
                rank,
                iteration=101 + replay,
            )
            inp.copy_(next_inp)
            residual.copy_(next_residual)
            padding.fill_(-7.0)
            expected.append(_reference(inp, residual, weight, epsilon))

        allocated_bytes = torch.cuda.memory_allocated(device)
        graph.replay()
        stream.synchronize()
        assert torch.cuda.memory_allocated(device) == allocated_bytes
        for (_inp, residual, _weight, padding, out), (
            expected_out,
            expected_residual,
        ) in zip(calls, expected, strict=True):
            _assert_close(out, expected_out, dtype)
            _assert_close(residual, expected_residual, dtype)
            assert torch.all(padding == -7.0)


def _run_pdl_dependent(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    """Both one-shot kernels execute ``griddepcontrol.launch_dependents`` as
    their first statement. A dependent kernel that executes
    ``griddepcontrol.wait`` before reading the allreduce output must observe
    the complete output whether or not its launch carries the
    programmatic-stream-serialization attribute, and the allreduce result
    itself must not change. Per iteration on fresh inputs, eagerly and under
    CUDA-graph replay, for the plain and the fused kernel: the one-shot output
    is compared bitwise with the same one-shot run without a dependent (the
    kernels sum in a fixed order) and within tolerance with a torch reference,
    and the dependent's copy is compared bitwise with the one-shot output."""
    iterations = int(os.getenv("B12X_PCIE_ONESHOT_PDL_ITERATIONS", "40"))
    rows, hidden_size, dtype, epsilon = 4, 6144, torch.bfloat16, 1e-6
    copy = compile_wait_then_copy()
    inp, residual, weight = _make_inputs(rows, hidden_size, dtype, device, rank)
    out = torch.empty_like(inp)
    residual_out = torch.empty_like(inp)
    plain_out = torch.empty_like(inp)
    dependent_out = torch.empty_like(inp)

    def plain(channel_id="eager:fused-rmsnorm", stream=None):
        pool.all_reduce(inp, out=plain_out, stream=stream, channel_id=channel_id)

    def fused(channel_id="eager:fused-rmsnorm", stream=None):
        pool.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=out,
            residual_out=residual_out,
            stream=stream,
            channel_id=channel_id,
        )

    def expected(iteration, name, allreduce, produced):
        """Torch references for fresh inputs, then the one-shot's own output
        for the same inputs without a dependent behind it."""
        next_inp, next_residual, _ = _make_inputs(
            rows, hidden_size, dtype, device, rank, iteration=iteration
        )
        inp.copy_(next_inp)
        residual.copy_(next_residual)
        reduced = inp.clone()
        dist.all_reduce(reduced)
        fused_out, _ = _reference(inp, residual, weight, epsilon)
        torch.cuda.synchronize(device)
        allreduce()
        torch.cuda.synchronize(device)
        alone = produced.clone()
        # Restore the inputs so the measured run sees the same values.
        inp.copy_(next_inp)
        residual.copy_(next_residual)
        torch.cuda.synchronize(device)
        return (reduced if name == "plain" else fused_out), alone

    chains = (
        ("plain", plain, plain_out),
        ("fused", fused, out),
    )
    for name, allreduce, produced in chains:
        for use_pdl in (True, False):
            # Eager: allreduce, then the dependent on the same stream.
            for iteration in range(iterations):
                reference, alone = expected(iteration + 1, name, allreduce, produced)
                dependent_out.zero_()
                allreduce()
                copy(produced, dependent_out, use_pdl)
                torch.cuda.synchronize(device)
                assert torch.equal(produced, alone), (
                    name,
                    use_pdl,
                    iteration,
                    "allreduce output changed",
                )
                _assert_close(produced, reference, dtype)
                assert torch.equal(dependent_out, produced), (
                    name,
                    use_pdl,
                    iteration,
                    "dependent read an incomplete output",
                )
            # Graph: the chain captured once, replayed on fresh inputs.
            # Each independently replayable graph needs its own channel id,
            # bound to its own stream.
            graph_channel = f"graph:pdl-{name}-{'attr' if use_pdl else 'noattr'}"
            stream = torch.cuda.Stream(device)
            channel = pool.for_stream(stream, channel_id=graph_channel)
            with torch.cuda.stream(stream):
                if name == "plain":
                    channel.prepare_graph_all_reduce(inp)
                else:
                    channel.prepare_graph_fused_add_rms_norm(inp)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with (
                pool.capture(stream=stream, channel_id=graph_channel),
                torch.cuda.graph(graph, stream=stream),
            ):
                allreduce(graph_channel, stream)
                copy(produced, dependent_out, use_pdl)
            for iteration in range(iterations):
                reference, alone = expected(1000 + iteration, name, allreduce, produced)
                dependent_out.zero_()
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.equal(produced, alone), (
                    name,
                    use_pdl,
                    iteration,
                    "graph allreduce output changed",
                )
                _assert_close(produced, reference, dtype)
                assert torch.equal(dependent_out, produced), (
                    name,
                    use_pdl,
                    iteration,
                    "graph dependent read an incomplete output",
                )
            del graph
            torch.cuda.synchronize(device)
            dist.barrier()


_GRAPH_SCATTER_GATHER_ROWS = (4, 8, 16)


def _graph_scatter_gather_channel(rows: int) -> str:
    return f"graph:scatter-gather:{rows}"


def _run_scatter_gather_modes(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    """Execute both scatter-gather kernel modes eagerly and under graph
    replay. The pool must hold scatter-gather eager storage (at world size
    8 this needs ``B12X_PCIE_TP8_OWNER_REDUCE=0``), so that rows from
    ``_SCATTER_GATHER_MIN_ROWS`` below ``_PACKED_GATHER_MIN_ROWS`` plan
    ``stage_scatter_gather`` (fp32 gather region) and rows from
    ``_PACKED_GATHER_MIN_ROWS`` plan ``stage_scatter_gather_packed``."""

    world_size = dist.get_world_size()
    hidden_size = 6144
    epsilon = 1e-6
    assert _eager_payload_shards(world_size) == _SCATTER_GATHER_SLOT_MULTIPLIER

    def expected_mode(rows: int) -> str:
        if rows >= _PACKED_GATHER_MIN_ROWS:
            return "stage_scatter_gather_packed"
        if rows >= _SCATTER_GATHER_MIN_ROWS:
            return "stage_scatter_gather"
        return "stage_pull"

    eager_channel = pool.for_stream(channel_id="eager:scatter-gather")
    eager_state = eager_channel._ext._state(eager_channel._ptr)
    assert eager_state.scatter_gather_storage
    modes_seen = set()
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (2, 3, 4, 5, 6, 8, 16):
            inp, residual, weight = _make_inputs(
                rows, hidden_size, dtype, device, rank, iteration=200 + rows
            )
            mode = _CuTeOneshotBackend._fused_launch_plan(eager_state, inp).variant.mode
            assert mode == expected_mode(rows), (rows, mode)
            modes_seen.add(mode)
            expected_out, expected_residual = _reference(inp, residual, weight, epsilon)
            out, residual_out = pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                channel_id="eager:scatter-gather",
            )
            torch.cuda.synchronize(device)
            _assert_close(out, expected_out, dtype)
            _assert_close(residual_out, expected_residual, dtype)
    assert modes_seen == {
        "stage_pull",
        "stage_scatter_gather",
        "stage_scatter_gather_packed",
    }

    dtype = torch.bfloat16
    for rows in _GRAPH_SCATTER_GATHER_ROWS:
        # Each independently replayable graph needs its own logical channel,
        # bound to its own stream.
        channel_id = _graph_scatter_gather_channel(rows)
        stream = torch.cuda.Stream(device=device)
        channel = pool.for_stream(stream, channel_id=channel_id)
        state = channel._ext._state(channel._ptr)
        assert state.scatter_gather_storage
        inp, residual, weight = _make_inputs(
            rows, hidden_size, dtype, device, rank, iteration=300 + rows
        )
        assert _CuTeOneshotBackend._fused_launch_plan(
            state, inp
        ).variant.mode == expected_mode(rows)
        out = torch.empty_like(inp)
        with torch.cuda.stream(stream):
            channel.prepare_graph_fused_add_rms_norm(inp)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            pool.capture(stream=stream, channel_id=channel_id),
            torch.cuda.graph(graph, stream=stream),
        ):
            pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                out=out,
                residual_out=residual,
                stream=stream,
                channel_id=channel_id,
            )
        # One kernel per collective: both scatter-gather rounds and their
        # barriers run inside the fused worker.
        _cuda_graph_kernel_chain(graph)
        for replay in range(3):
            next_inp, next_residual, _ = _make_inputs(
                rows, hidden_size, dtype, device, rank, iteration=310 + replay
            )
            inp.copy_(next_inp)
            residual.copy_(next_residual)
            expected_out, expected_residual = _reference(inp, residual, weight, epsilon)
            allocated_bytes = torch.cuda.memory_allocated(device)
            graph.replay()
            stream.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated_bytes
            _assert_close(out, expected_out, dtype)
            _assert_close(residual, expected_residual, dtype)
        dist.barrier()


def _scatter_gather_worker(rank: int, world_size: int, port: int) -> None:
    # The transport policy is read from the environment when the channels'
    # eager storage is installed; owner reduction off makes world size 8 use
    # scatter-gather storage like world sizes 2 and 4.
    os.environ["B12X_PCIE_TP8_OWNER_REDUCE"] = "0"
    os.environ.pop("B12X_PCIE_SCATTER_GATHER", None)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=TEST_TIMEOUT_SECONDS),
    )
    graph_channels = tuple(
        _graph_scatter_gather_channel(rows) for rows in _GRAPH_SCATTER_GATHER_ROWS
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=192 * 1024,
        max_size=192 * 1024,
        max_concurrent_channels=1 + len(graph_channels),
    )
    try:
        pool.prepare_channels(("eager:scatter-gather", *graph_channels))
        _run_scatter_gather_modes(pool, device, rank)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


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
        max_input_bytes=192 * 1024,
        max_size=192 * 1024,
        max_concurrent_channels=4,
    )
    try:
        pool.prepare_channels(
            (
                "eager:fused-rmsnorm",
                "graph:fused-rmsnorm",
                "graph:fused-transition",
                "graph:split-residual",
                "graph:pdl-plain-attr",
                "graph:pdl-plain-noattr",
                "graph:pdl-fused-attr",
                "graph:pdl-fused-noattr",
            )
        )
        pool.for_stream(channel_id="eager:fused-rmsnorm")
        _run_eager(pool, device, rank)
        dist.barrier()
        _run_graph(pool, device, rank)
        dist.barrier()
        _run_tp8_graph_mode_transition(pool, device, rank)
        dist.barrier()
        _run_tp8_split_view_graph(pool, device, rank)
        dist.barrier()
        _run_pdl_dependent(pool, device, rank)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def _spawn_and_join(worker, world_size: int) -> None:
    context = mp.spawn(
        worker,
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


def _world_size_or_skip(supported: tuple[int, ...]) -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_ONESHOT_RMS_WORLD_SIZE", "2"))
    if world_size not in supported:
        pytest.skip(f"this test supports world sizes {supported}")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    return world_size


def test_pcie_oneshot_fused_add_rms_norm_eager_and_graph() -> None:
    _spawn_and_join(_worker, _world_size_or_skip((2, 4, 6, 8, 10)))


def test_pcie_oneshot_fused_add_rms_norm_scatter_gather_modes() -> None:
    """Both scatter-gather kernel modes run on the GPUs with the pool
    configured for scatter-gather storage (``B12X_PCIE_TP8_OWNER_REDUCE=0``
    inside the workers), at world size 2, 4 or 8."""
    _spawn_and_join(_scatter_gather_worker, _world_size_or_skip((2, 4, 8)))
