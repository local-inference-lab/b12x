"""Kimi-K3 M<=8 single-projection gather A/B: B12X mb-8 pool vs NCCL.

The production M8 DSpark verify performs one small TP16 all-gather per
attention layer (KDA f_a shards and the MLA merged projection). The B12X
single-projection fast path historically covered only M=1; this harness
measures the same widths at batch 1 and 8 through one shared max_batch_size=8
IPC pool signature, byte-exact against NCCL, both CUDA-graph captured.
"""

from __future__ import annotations

import argparse
import os
import socket

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_PROTO", "LL,LL128,Simple")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool

from benchmark_kimi_k3_projection_gather import (
    _assert_exact,
    _capture,
    _capture_with_pools,
    _measure,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# Production M8 verify widths (bytes moved per rank row): the KDA f_a shard,
# the padded MLA merged qkv_a projection, and the routed-down latent shard.
SHAPES = (
    ("kda_f_a_bf16", 8, torch.bfloat16),
    ("qkv_a_bf16_raw_bytes_padded", 272, torch.float8_e4m3fn),
    ("routed_down_bf16", 224, torch.bfloat16),
)


def _input(
    *,
    rank: int,
    batch: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        raw = (
            torch.arange(batch * width, dtype=torch.int64, device=device)
            + 17 * rank
        ).to(torch.uint8)
        return raw.view(torch.float8_e4m3fn).reshape(batch, 1, width)
    base = torch.arange(batch * width, dtype=dtype, device=device)
    return (base * 0.001 + rank + 0.25).reshape(batch, 1, width)


def _benchmark_shape_batched(
    *,
    name: str,
    batch: int,
    width: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[float, float]:
    local_input = _input(
        rank=rank, batch=batch, width=width, dtype=dtype, device=device
    )
    custom_output = torch.empty(
        (batch, world_size, width), dtype=dtype, device=device
    )
    nccl_output = torch.empty(
        (world_size * batch, 1, width), dtype=dtype, device=device
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=8,
        total_heads=world_size,
        head_dim=width,
        query_head_dim=width,
    )
    channel_id = f"k3-m8-gather:{name}:b{batch}"
    pool.prepare_channels((channel_id,))
    try:

        def custom_fn() -> None:
            pool.all_gather_heads(
                local_input,
                custom_output,
                threads=512,
                block_limit=8,
                channel_id=channel_id,
            )

        def nccl_fn() -> None:
            dist.all_gather_into_tensor(
                nccl_output, local_input, group=dist.group.WORLD
            )

        custom_fn()
        nccl_fn()
        torch.cuda.synchronize(device)
        expected = (
            nccl_output.view(world_size, batch, width).permute(1, 0, 2).contiguous()
        )
        _assert_exact(custom_output.contiguous(), expected)

        custom_graph = _capture_with_pools(custom_fn, ((pool, channel_id),))
        nccl_graph = _capture(nccl_fn)
        custom_output.zero_()
        nccl_output.zero_()
        custom_graph.replay()
        nccl_graph.replay()
        torch.cuda.synchronize(device)
        expected = (
            nccl_output.view(world_size, batch, width).permute(1, 0, 2).contiguous()
        )
        _assert_exact(custom_output.contiguous(), expected)

        custom_us = _measure(
            custom_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        nccl_us = _measure(
            nccl_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        return custom_us, nccl_us
    finally:
        pool.close()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    iterations: int,
    samples: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    bootstrap = torch.zeros(8192, dtype=torch.bfloat16, device=device)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize(device)
    try:
        if rank == 0:
            print(
                "name,batch,world_size,local_bytes,custom_us,nccl_us,speedup",
                flush=True,
            )
        for batch in (1, 8):
            for name, width, dtype in SHAPES:
                custom_us, nccl_us = _benchmark_shape_batched(
                    name=name,
                    batch=batch,
                    width=width,
                    dtype=dtype,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    samples=samples,
                )
                if rank == 0:
                    local_bytes = batch * width * dtype.itemsize
                    print(
                        f"{name},{batch},{world_size},{local_bytes},"
                        f"{custom_us:.3f},{nccl_us:.3f},"
                        f"{nccl_us / custom_us:.3f}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"needs {args.world_size} CUDA devices, "
            f"found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.warmup,
            args.iterations,
            args.samples,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
