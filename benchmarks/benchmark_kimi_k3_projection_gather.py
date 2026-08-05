from __future__ import annotations

import argparse
import socket
import statistics
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capture(
    fn: Callable[[], None],
    *,
    pool: PCIeDCPA2APool | None = None,
    channel_id: str | None = None,
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if pool is None:
        with torch.cuda.graph(graph):
            fn()
    else:
        assert channel_id is not None
        with pool.capture(channel_id=channel_id), torch.cuda.graph(graph):
            fn()
    return graph


def _capture_with_pools(
    fn: Callable[[], None],
    bindings: tuple[tuple[PCIeDCPA2APool, str], ...],
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ExitStack() as stack:
        for pool, channel_id in bindings:
            stack.enter_context(pool.capture(channel_id=channel_id))
        with torch.cuda.graph(graph):
            fn()
    return graph


def _measure(
    graph: torch.cuda.CUDAGraph,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> float:
    timings: list[float] = []
    for _ in range(samples):
        dist.barrier()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize(device)
        elapsed_us = (time.perf_counter() - started) * 1e6 / iterations
        maximum = torch.tensor(elapsed_us, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        timings.append(float(maximum.item()))
    return statistics.median(timings)


def _input(
    *,
    rank: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        # The router transports FP32 logits byte-for-byte through the FP8 copy
        # dispatch. Exercise every byte position without numerically converting
        # the payload.
        raw = (torch.arange(width, dtype=torch.int64, device=device) + 17 * rank).to(
            torch.uint8
        )
        return raw.view(torch.float8_e4m3fn).reshape(1, 1, width)
    return torch.full(
        (1, 1, width),
        rank + 0.25,
        dtype=dtype,
        device=device,
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float8_e4m3fn:
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise AssertionError("raw-byte FP8 gather differs from NCCL")
        return
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _benchmark_shape(
    *,
    name: str,
    width: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[float, float]:
    local_input = _input(rank=rank, width=width, dtype=dtype, device=device)
    custom_output = torch.empty((1, world_size, width), dtype=dtype, device=device)
    nccl_output = torch.empty((world_size, 1, width), dtype=dtype, device=device)
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=1,
        total_heads=world_size,
        head_dim=width,
        query_head_dim=width,
    )
    channel_id = f"kimi-k3-projection:{name}"
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
        _assert_exact(custom_output.flatten(), nccl_output.flatten())

        custom_graph = _capture(custom_fn, pool=pool, channel_id=channel_id)
        nccl_graph = _capture(nccl_fn)
        custom_output.zero_()
        custom_graph.replay()
        torch.cuda.synchronize(device)
        _assert_exact(custom_output.flatten(), nccl_output.flatten())

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


def _benchmark_projection_pair(
    *,
    batch: int,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[float, float]:
    down_local = torch.full(
        (batch, 224), rank + 0.25, dtype=torch.bfloat16, device=device
    )
    router_local = torch.arange(batch * 56, dtype=torch.float32, device=device).view(
        batch, 56
    )
    router_local.add_(rank * 1000)
    router_raw = router_local.view(torch.float8_e4m3fn).view(batch, 1, 224)

    down_output = torch.empty(
        (batch, world_size, 224), dtype=torch.bfloat16, device=device
    )
    router_output = torch.empty(
        (batch, world_size, 224), dtype=torch.float8_e4m3fn, device=device
    )
    paired_down = torch.empty(
        (batch, world_size * 224), dtype=torch.bfloat16, device=device
    )
    paired_router = torch.empty(
        (batch, world_size * 56), dtype=torch.float32, device=device
    )

    down_pool: PCIeDCPA2APool | None = None
    router_pool: PCIeDCPA2APool | None = None
    pair_pool: PCIeDCPA2APool | None = None
    try:
        down_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=batch,
            total_heads=world_size,
            head_dim=224,
            query_head_dim=224,
        )
        router_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=batch,
            total_heads=world_size,
            head_dim=224,
            query_head_dim=224,
        )
        pair_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=batch,
            total_heads=world_size,
            head_dim=672,
            query_head_dim=672,
        )
        down_channel = f"k3-projection-pair:b{batch}:down"
        router_channel = f"k3-projection-pair:b{batch}:router"
        pair_channel = f"k3-projection-pair:b{batch}:paired"
        down_pool.prepare_channels((down_channel,))
        router_pool.prepare_channels((router_channel,))
        pair_pool.prepare_channels((pair_channel,))
    except Exception:
        for pool in (pair_pool, router_pool, down_pool):
            if pool is not None:
                with suppress(Exception):
                    pool.close()
        raise
    try:

        def separate_fn() -> None:
            down_pool.all_gather_heads(
                down_local.view(batch, 1, 224),
                down_output,
                threads=512,
                block_limit=8,
                channel_id=down_channel,
            )
            router_pool.all_gather_heads(
                router_raw,
                router_output,
                threads=512,
                block_limit=8,
                channel_id=router_channel,
            )

        def paired_fn() -> None:
            pair_pool.all_gather_pair(
                down_local,
                router_local,
                paired_down,
                paired_router,
                threads=512,
                channel_id=pair_channel,
            )

        separate_fn()
        paired_fn()
        torch.cuda.synchronize(device)
        _assert_exact(paired_down.flatten(), down_output.flatten())
        if not torch.equal(
            paired_router.view(torch.uint8),
            router_output.flatten(1).view(torch.float32).view(torch.uint8),
        ):
            raise AssertionError("paired FP32 router gather differs byte-for-byte")

        separate_graph = _capture_with_pools(
            separate_fn,
            ((down_pool, down_channel), (router_pool, router_channel)),
        )
        paired_graph = _capture_with_pools(
            paired_fn,
            ((pair_pool, pair_channel),),
        )
        paired_down.zero_()
        paired_router.zero_()
        paired_graph.replay()
        torch.cuda.synchronize(device)
        _assert_exact(paired_down.flatten(), down_output.flatten())
        if not torch.equal(
            paired_router.view(torch.uint8),
            router_output.flatten(1).view(torch.float32).view(torch.uint8),
        ):
            raise AssertionError("captured paired router gather differs")

        separate_us = _measure(
            separate_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        paired_us = _measure(
            paired_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        return separate_us, paired_us
    finally:
        for pool in (pair_pool, router_pool, down_pool):
            if pool is not None:
                with suppress(Exception):
                    pool.close()


def _benchmark_qkv_padded_projection(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[float, float]:
    local_width = 132
    padded_width = 136
    local_input = torch.full(
        (1, local_width), rank + 0.25, dtype=torch.bfloat16, device=device
    )
    custom_output = torch.empty(
        (1, world_size, padded_width), dtype=torch.bfloat16, device=device
    )
    nccl_output = torch.empty(
        (world_size, 1, local_width), dtype=torch.bfloat16, device=device
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=1,
        total_heads=world_size,
        head_dim=padded_width,
        query_head_dim=padded_width,
    )
    channel_id = "k3-qkv-a:padded"
    pool.prepare_channels((channel_id,))
    try:
        captured_actual: torch.Tensor | None = None

        def custom_fn() -> torch.Tensor:
            nonlocal captured_actual
            padded = torch.nn.functional.pad(
                local_input, (0, padded_width - local_width)
            ).view(1, 1, padded_width)
            gathered = pool.all_gather_heads(
                padded,
                custom_output,
                threads=512,
                block_limit=8,
                channel_id=channel_id,
            )
            captured_actual = (
                gathered.narrow(-1, 0, local_width).contiguous().flatten(1)
            )
            return captured_actual

        def nccl_fn() -> None:
            dist.all_gather_into_tensor(
                nccl_output, local_input, group=dist.group.WORLD
            )

        actual = custom_fn()
        nccl_fn()
        torch.cuda.synchronize(device)
        _assert_exact(actual.flatten(), nccl_output.flatten())

        custom_graph = _capture_with_pools(
            custom_fn,
            ((pool, channel_id),),
        )
        nccl_graph = _capture(nccl_fn)
        custom_output.zero_()
        nccl_output.zero_()
        custom_graph.replay()
        nccl_graph.replay()
        torch.cuda.synchronize(device)
        assert captured_actual is not None
        _assert_exact(captured_actual.flatten(), nccl_output.flatten())
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


def _benchmark_f_a_projection(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[float, float, int, float]:
    hidden_size = 7168
    local_width = 8
    hidden = torch.linspace(
        -0.5,
        0.5,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    ).view(1, hidden_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + rank)
    local_weight = torch.randn(
        (local_width, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.01)
    full_weight = torch.empty(
        (world_size * local_width, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    dist.all_gather_into_tensor(full_weight, local_weight)

    local_output = torch.empty((1, local_width), dtype=torch.bfloat16, device=device)
    sharded_output = torch.empty(
        (1, world_size, local_width), dtype=torch.bfloat16, device=device
    )
    replicated_output = torch.empty(
        (1, world_size * local_width), dtype=torch.bfloat16, device=device
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=1,
        total_heads=world_size,
        head_dim=local_width,
        query_head_dim=local_width,
    )
    channel_id = "k3-f-a-projection:sharded"
    pool.prepare_channels((channel_id,))
    try:

        def sharded_fn() -> None:
            torch.mm(hidden, local_weight.T, out=local_output)
            pool.all_gather_heads(
                local_output.view(1, 1, local_width),
                sharded_output,
                threads=512,
                block_limit=8,
                channel_id=channel_id,
            )

        def replicated_fn() -> None:
            torch.mm(hidden, full_weight.T, out=replicated_output)

        sharded_fn()
        replicated_fn()
        torch.cuda.synchronize(device)
        sharded_flat = sharded_output.flatten(1)
        different = int((sharded_flat != replicated_output).sum().item())
        max_abs = float(
            (sharded_flat.float() - replicated_output.float()).abs().max().item()
        )

        sharded_graph = _capture_with_pools(
            sharded_fn,
            ((pool, channel_id),),
        )
        replicated_graph = _capture(replicated_fn)
        sharded_output.zero_()
        replicated_output.zero_()
        sharded_graph.replay()
        replicated_graph.replay()
        torch.cuda.synchronize(device)
        sharded_flat = sharded_output.flatten(1)
        different = int((sharded_flat != replicated_output).sum().item())
        max_abs = float(
            (sharded_flat.float() - replicated_output.float()).abs().max().item()
        )
        sharded_us = _measure(
            sharded_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        replicated_us = _measure(
            replicated_graph,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        return sharded_us, replicated_us, different, max_abs
    finally:
        pool.close()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    iterations: int,
    samples: int,
    pair_only: bool,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    shapes = (
        ("kda_f_a_bf16", 8, torch.bfloat16),
        # 132 BF16 qkv_a values occupy 264 bytes; one 8-byte zero tail makes
        # the raw copy payload meet the kernel's 16-byte transaction contract.
        ("qkv_a_bf16_raw_bytes_padded", 272, torch.float8_e4m3fn),
        ("routed_down_bf16", 224, torch.bfloat16),
        # 56 FP32 router values occupy 224 raw bytes.
        ("router_fp32_raw_bytes", 224, torch.float8_e4m3fn),
        # One packed gather can carry routed-down and router payloads together.
        ("routed_down_plus_router_raw_bytes", 672, torch.float8_e4m3fn),
    )
    try:
        if pair_only:
            if rank == 0:
                print(
                    "name,world_size,local_bytes,paired_us,separate_us,speedup",
                    flush=True,
                )
            for batch in (1, 8):
                separate_us, paired_us = _benchmark_projection_pair(
                    batch=batch,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    samples=samples,
                )
                if rank == 0:
                    print(
                        f"router_down_pair_b{batch},"
                        f"{world_size},{batch * 672},{paired_us:.3f},"
                        f"{separate_us:.3f},{separate_us / paired_us:.3f}",
                        flush=True,
                    )
            return
        if rank == 0:
            print(
                "name,world_size,local_bytes,custom_us,nccl_us,speedup",
                flush=True,
            )
        for name, width, dtype in shapes:
            custom_us, nccl_us = _benchmark_shape(
                name=name,
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
                local_bytes = width * dtype.itemsize
                print(
                    f"{name},{world_size},{local_bytes},{custom_us:.3f},"
                    f"{nccl_us:.3f},{nccl_us / custom_us:.3f}",
                    flush=True,
                )
        for batch in (1, 8):
            if rank == 0 and batch == 1:
                print("# section=projection_pair", flush=True)
                print(
                    "name,world_size,local_bytes,paired_us,separate_us,speedup",
                    flush=True,
                )
            separate_us, paired_us = _benchmark_projection_pair(
                batch=batch,
                rank=rank,
                world_size=world_size,
                device=device,
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            )
            if rank == 0:
                print(
                    f"router_down_pair_b{batch},"
                    f"{world_size},{batch * 672},{paired_us:.3f},"
                    f"{separate_us:.3f},{separate_us / paired_us:.3f}",
                    flush=True,
                )
        qkv_custom_us, qkv_nccl_us = _benchmark_qkv_padded_projection(
            rank=rank,
            world_size=world_size,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        if rank == 0:
            print("# section=qkv_padded_projection", flush=True)
            print(
                "name,world_size,local_bytes,custom_us,nccl_us,speedup",
                flush=True,
            )
            print(
                "qkv_a_padded_end_to_end,"
                f"{world_size},264,{qkv_custom_us:.3f},{qkv_nccl_us:.3f},"
                f"{qkv_nccl_us / qkv_custom_us:.3f}",
                flush=True,
            )
        sharded_us, replicated_us, different, max_abs = _benchmark_f_a_projection(
            rank=rank,
            world_size=world_size,
            device=device,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
        )
        if rank == 0:
            print("# section=f_a_projection", flush=True)
            print(
                "name,world_size,replicated_us,sharded_us,"
                "sharded_over_replicated,different_elements,max_abs",
                flush=True,
            )
            print(
                "f_a_projection,"
                f"{world_size},{replicated_us:.3f},{sharded_us:.3f},"
                f"{sharded_us / replicated_us:.3f},{different},{max_abs:.8g}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--pair-only", action="store_true")
    args = parser.parse_args()
    if args.world_size not in (2, 4, 8, 16):
        raise SystemExit("world size must be one of 2, 4, 8, or 16")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.warmup,
            args.iterations,
            args.samples,
            args.pair_only,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
