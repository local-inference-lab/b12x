"""Correctness tests for the RoCE one-shot all-reduce.

Run with torchrun on two or more nodes that share a RoCE fabric, for example
from every node of a DGX Spark cluster::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        -m pytest -x tests/comm/test_roce_oneshot_gpu.py

Without a torchrun environment the tests skip.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires a torchrun launch with WORLD_SIZE >= 2",
)


@pytest.fixture(scope="module")
def runtime():
    from b12x.comm import roce

    if not roce.is_supported():
        pytest.skip("RoCE all-reduce needs an integrated GPU with an active RDMA device")
    if not dist.is_initialized():
        # A short timeout turns a rank that failed early into an error on every
        # rank instead of a silent hang in the next collective.
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rt = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD, device=device, max_size=1 << 20
    )
    rt.prepare((torch.bfloat16, torch.float32, torch.float16))
    yield rt
    rt.close()
    dist.barrier()


def _tolerance(dtype: torch.dtype, world: int) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5 * world
    if dtype == torch.bfloat16:
        return 1e-2, 2e-2 * world
    return 2e-3, 4e-3 * world


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("numel_bytes", [16, 4096, 48 * 1024, 256 * 1024, 1 << 20])
def test_matches_nccl_and_is_rank_identical(runtime, dtype, numel_bytes):
    world = dist.get_world_size()
    rank = dist.get_rank()
    numel = numel_bytes // torch.tensor([], dtype=dtype).element_size()
    for trial in range(4):
        torch.manual_seed(1000 * trial + rank)
        inp = torch.randn(numel, dtype=dtype, device=runtime.device)
        expected = inp.clone()
        dist.all_reduce(expected)
        out = runtime.all_reduce(inp)
        torch.cuda.synchronize()
        rtol, atol = _tolerance(dtype, world)
        if not torch.allclose(out, expected, rtol=rtol, atol=atol):
            print(
                f"[rank {rank} trial {trial} {dtype} {numel_bytes}B] MISMATCH stats={runtime.stats()}\n"
                f"  inp={inp[:8].tolist()}\n  out={out[:8].tolist()}\n  exp={expected[:8].tolist()}",
                flush=True,
            )
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        gathered = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(gathered, out)
        for peer_out in gathered:
            assert torch.equal(peer_out, out), "ranks must produce bit-identical output"


def test_rejects_ineligible_inputs(runtime):
    huge = torch.zeros((runtime.max_size // 2) + 8, dtype=torch.bfloat16, device=runtime.device)
    assert not runtime.should_allreduce(huge)
    odd = torch.zeros(3, dtype=torch.bfloat16, device=runtime.device)
    assert not runtime.should_allreduce(odd)
    strided = torch.zeros(64, 8, dtype=torch.bfloat16, device=runtime.device)[:, ::2]
    assert not runtime.should_allreduce(strided)
    ok = torch.zeros(4096, dtype=torch.bfloat16, device=runtime.device)
    assert runtime.should_allreduce(ok)


def test_cuda_graph_replay(runtime):
    world = dist.get_world_size()
    rank = dist.get_rank()
    static_in = torch.zeros(24 * 1024, dtype=torch.bfloat16, device=runtime.device)
    static_out = torch.empty_like(static_in)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            runtime.all_reduce(static_in, out=static_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        for _ in range(3):
            runtime.all_reduce(static_in, out=static_out)
            static_in.copy_(static_out)
    torch.cuda.synchronize()
    dist.barrier()
    for replay in range(5):
        torch.manual_seed(7 * replay + rank)
        seed = torch.randn_like(static_in) * 0.01
        static_in.copy_(seed)
        expected = seed.clone()
        for _ in range(3):
            dist.all_reduce(expected)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_out, expected, rtol=2e-2, atol=2e-2 * world**3)
