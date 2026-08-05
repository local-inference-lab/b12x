"""Exact TP16 Kimi-K3 hierarchical all-reduce sequence benchmark.

This model-free harness captures the alternating K3 decode shapes in one
CUDA graph, validates the custom grouped BF16 reduction bit for bit, replays
it under stress, and reports rank-max latency.  Run each mode as a separate
invocation so each process owns only one IPC runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from statistics import median

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "LL")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "8")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.cpp_extension import load

from b12x.comm.pcie import AllReduce
from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from b12x.comm.pcie.pcie_oneshot import _broadcast_gather_object


DEFAULT_SHAPES = (7168, 3584, 7168)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _lane_peers(rank: int, world_size: int) -> tuple[int, ...]:
    """Three local peers plus the same lane in every other island."""

    if world_size not in (12, 16) or not 0 <= rank < world_size:
        raise ValueError(f"invalid lane topology rank={rank}, TP={world_size}")
    island = rank // 4
    local_rank = rank % 4
    local = set(range(island * 4, island * 4 + 4))
    cross_island = set(range(local_rank, world_size, 4))
    return tuple(sorted((local | cross_island) - {rank}))


@lru_cache(maxsize=1)
def _load_lane_extension():
    source = Path(__file__).with_name("kimi_k3_lane_allreduce_poc.cu")
    return load(
        name="b12x_kimi_k3_lane_allreduce_poc_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=os.getenv("B12X_K3_LANE_POC_VERBOSE_BUILD", "0") == "1",
    )


class LaneDistributedAllReducePOC:
    """Benchmark-only IPC owner for the lane-distributed CUDA POC."""

    algorithm = "lane-poc"

    def __init__(
        self,
        *,
        exchange_group,
        device: torch.device,
        max_elements: int,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = device
        self.max_elements = int(max_elements)
        self._ext = _load_lane_extension()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(device.index or 0)
        self._runtime = 0
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False

        slab_bytes = int(self._ext.slab_bytes(self.max_elements))
        peer_ptrs = [0] * self.world_size
        try:
            self._local_ptr = self._ipc.cudaMalloc(slab_bytes)
            self._ipc.cudaMemset(self._local_ptr, 0, slab_bytes)
            local_handle = self._ipc.cudaIpcGetMemHandleBytes(self._local_ptr)
            handles = _broadcast_gather_object(local_handle, exchange_group)
            peer_ptrs[self.rank] = self._local_ptr
            for peer in _lane_peers(self.rank, self.world_size):
                ptr = self._ipc.cudaIpcOpenMemHandleBytes(handles[peer])
                peer_ptrs[peer] = ptr
                self._remote_ptrs.append(ptr)
            self._runtime = int(
                self._ext.init_runtime(
                    peer_ptrs,
                    self.rank,
                    self.max_elements,
                )
            )
        except Exception:
            for ptr in self._remote_ptrs:
                with suppress(Exception):
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            self._remote_ptrs.clear()
            if self._local_ptr:
                with suppress(Exception):
                    self._ipc.cudaFree(self._local_ptr)
                self._local_ptr = 0
            raise

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self._closed
            or inp.device != self.device
            or inp.dtype != torch.bfloat16
            or not inp.is_contiguous()
            or not 0 < inp.numel() <= self.max_elements
        ):
            raise ValueError("input does not satisfy lane POC requirements")
        if out is None:
            out = torch.empty_like(inp)
        if (
            out.device != inp.device
            or out.dtype != inp.dtype
            or out.shape != inp.shape
            or not out.is_contiguous()
        ):
            raise ValueError("lane POC output must match its input")
        blocks = 16 if inp.numel() <= 4096 else 32
        self._ext.all_reduce_bf16(self._runtime, inp, out, blocks)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        if self._runtime:
            self._ext.destroy_runtime(self._runtime)
            self._runtime = 0
        for ptr in self._remote_ptrs:
            self._ipc.cudaIpcCloseMemHandle(ptr)
        self._remote_ptrs.clear()
        dist.barrier(group=self.group)
        if self._local_ptr:
            self._ipc.cudaFree(self._local_ptr)
            self._local_ptr = 0
        dist.barrier(group=self.group)


def _grouped_reference(
    inp: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    island_partials: list[torch.Tensor] = []
    for island_base in range(0, world_size, 4):
        partial = torch.zeros_like(inp, dtype=torch.float32)
        for peer in range(island_base, island_base + 4):
            partial.add_(gathered[peer].float())
        island_partials.append(partial)
    result = torch.zeros_like(inp, dtype=torch.float32)
    for partial in island_partials:
        result.add_(partial)
    return result.to(torch.bfloat16)


def _assert_outputs(
    outputs: list[torch.Tensor],
    references: list[torch.Tensor],
    inputs: list[torch.Tensor],
    inputs_before: list[torch.Tensor],
    device: torch.device,
) -> None:
    local_ok = torch.ones((), dtype=torch.int32, device=device)
    local_max_delta = torch.zeros((), dtype=torch.float32, device=device)
    local_rank_delta = torch.zeros((), dtype=torch.float32, device=device)
    for actual, expected, inp, before in zip(
        outputs,
        references,
        inputs,
        inputs_before,
        strict=True,
    ):
        if not torch.equal(actual, expected) or not torch.equal(inp, before):
            local_ok.zero_()
        local_max_delta = torch.maximum(
            local_max_delta,
            (actual.float() - expected.float()).abs().max(),
        )
        rank0 = actual.clone()
        dist.broadcast(rank0, src=0)
        local_rank_delta = torch.maximum(
            local_rank_delta,
            (actual.float() - rank0.float()).abs().max(),
        )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    dist.all_reduce(local_max_delta, op=dist.ReduceOp.MAX)
    dist.all_reduce(local_rank_delta, op=dist.ReduceOp.MAX)
    if int(local_ok.item()) != 1:
        raise AssertionError(
            "sequence changed an input or differs from the exact grouped "
            f"reference (max delta {local_max_delta.item()})"
        )
    if float(local_rank_delta.item()) != 0.0:
        raise AssertionError(
            "sequence output differs across ranks "
            f"(max delta {local_rank_delta.item()})"
        )


def _rank_max_timings(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)

    result: list[float] = []
    rank_value = torch.empty((), dtype=torch.float64, device=device)
    for _ in range(samples):
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        rank_value.fill_(start.elapsed_time(end) * 1e3 / iters)
        dist.all_reduce(rank_value, op=dist.ReduceOp.MAX)
        result.append(float(rank_value.item()))
    return result


def _worker(
    rank: int,
    world_size: int,
    port: int,
    mode: str,
    layers: int,
    gap_cycles: int,
    stress_replays: int,
    warmup: int,
    iters: int,
    samples: int,
    with_dcp_pool: bool,
    shapes: tuple[int, ...],
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    # Match production: let NCCL establish its mappings before the bounded
    # custom IPC topology is opened.
    bootstrap = torch.zeros(max(shapes), dtype=torch.bfloat16, device=device)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize(device)

    dcp_pool = None
    runtime = None
    try:
        if with_dcp_pool:
            # Reproduce the serving process's co-resident TP16 DCP IPC slab so
            # the POC cannot win by assuming an unrealistically empty peer map.
            dcp_pool = PCIeDCPA2APool.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_batch_size=1,
                total_heads=96,
                head_dim=512,
                query_head_dim=576,
            )
        if mode == "lane":
            runtime = LaneDistributedAllReducePOC(
                exchange_group=dist.group.WORLD,
                device=device,
                max_elements=max(shapes),
            )
            actual_mode = "lane"
        else:
            runtime = AllReduce.from_exchange_group(
                exchange_group=dist.group.WORLD,
                device=device,
                max_size=max(shapes) * torch.bfloat16.itemsize,
            )
            if runtime.algorithm != "hierarchical":
                raise AssertionError(f"unexpected algorithm {runtime.algorithm}")
            actual_mode = (
                "double"
                if runtime._runtime.double_buffered
                else ("deferred" if runtime._runtime.deferred_consumption else "legacy")
            )
        if actual_mode != mode:
            raise AssertionError(f"requested {mode}, constructed {actual_mode}")

        inputs: list[torch.Tensor] = []
        references: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for call, elements in enumerate(shapes):
            generator = torch.Generator(device=device).manual_seed(
                0x4B33 + rank * 17 + call
            )
            inp = torch.randn(
                elements,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            inputs.append(inp)
            references.append(_grouped_reference(inp, world_size))
            outputs.append(torch.empty_like(inp))
        inputs_before = [inp.clone() for inp in inputs]

        # Exercise the exact mixed-grid sequence eagerly before capture.
        for _ in range(layers):
            for inp, out in zip(inputs, outputs, strict=True):
                runtime.all_reduce(inp, out=out)
                if gap_cycles:
                    torch.cuda._sleep(gap_cycles)
        torch.cuda.synchronize(device)
        _assert_outputs(outputs, references, inputs, inputs_before, device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(layers):
                for inp, out in zip(inputs, outputs, strict=True):
                    runtime.all_reduce(inp, out=out)
                    if gap_cycles:
                        torch.cuda._sleep(gap_cycles)

        for _ in range(stress_replays):
            graph.replay()
        torch.cuda.synchronize(device)
        _assert_outputs(outputs, references, inputs, inputs_before, device)

        timings = _rank_max_timings(
            graph,
            device,
            warmup=warmup,
            iters=iters,
            samples=samples,
        )
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_outputs(outputs, references, inputs, inputs_before, device)

        peer_count = torch.tensor(
            runtime.mapped_peer_count,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(peer_count, op=dist.ReduceOp.MAX)
        if rank == 0:
            calls = layers * len(shapes)
            med = float(median(timings))
            if mode == "lane":
                # The benchmark-only lane POC has fixed launch geometry in
                # kimi_k3_lane_allreduce_poc.cu and does not implement BF16x2.
                wait_nanosleep_cycles = 64
                configured_scalar_threads = 256
                effective_threads = [256 for _ in shapes]
                vectorized_bf16x2 = False
                vectorized_bf16x2_max_elements = None
            else:
                wait_nanosleep_cycles = int(
                    os.getenv(
                        "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES",
                        "24",
                    )
                )
                vectorized_bf16x2 = (
                    os.getenv(
                        "B12X_PCIE_HIERARCHICAL_BF16X2",
                        "1",
                    )
                    == "1"
                )
                vectorized_bf16x2_max_elements = int(
                    os.getenv(
                        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
                        "7168",
                    )
                )
                configured_scalar_threads = int(
                    os.getenv("B12X_PCIE_HIERARCHICAL_THREADS", "224")
                )
                effective_threads = [
                    (
                        112
                        if vectorized_bf16x2
                        and elements <= vectorized_bf16x2_max_elements
                        else configured_scalar_threads
                    )
                    for elements in shapes
                ]
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "world_size": world_size,
                        "shapes": shapes,
                        "layers_per_graph": layers,
                        "collectives_per_graph": calls,
                        "gap_cycles_after_each_collective": gap_cycles,
                        "stress_replays": stress_replays,
                        "samples_graph_us_rank_max": timings,
                        "median_graph_us_rank_max": med,
                        "median_us_per_layer": med / layers,
                        "median_us_per_collective": med / calls,
                        "max_mapped_peers": int(peer_count.item()),
                        "dcp_pool_co_resident": with_dcp_pool,
                        "bit_exact_grouped_reference": True,
                        "bit_exact_across_ranks": True,
                        "input_immutable": True,
                        "wait_nanosleep_cycles": wait_nanosleep_cycles,
                        "configured_scalar_threads": configured_scalar_threads,
                        "effective_threads_per_block_by_shape": effective_threads,
                        "vectorized_bf16x2": vectorized_bf16x2,
                        "vectorized_bf16x2_max_elements": (
                            vectorized_bf16x2_max_elements
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dcp_pool is not None:
            dcp_pool.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(12, 16), default=16)
    parser.add_argument(
        "--mode",
        choices=("legacy", "double", "deferred", "lane"),
        required=True,
    )
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gap-cycles", type=int, default=100_000)
    parser.add_argument("--stress-replays", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--with-dcp-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep a production-shaped TP16 DCP IPC pool co-resident",
    )
    parser.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=DEFAULT_SHAPES,
        help="ordered BF16 element counts captured in each layer",
    )
    args = parser.parse_args()
    if (
        min(
            args.layers,
            args.stress_replays,
            args.warmup,
            args.iters,
            args.samples,
        )
        < 0
        or args.layers == 0
        or args.iters == 0
        or args.samples == 0
    ):
        raise ValueError("layers/iters/samples must be positive; counts nonnegative")
    if args.gap_cycles < 0:
        raise ValueError("gap cycles must be nonnegative")
    if not args.shapes or min(args.shapes) <= 0:
        raise ValueError("shapes must contain positive element counts")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"TP{args.world_size} benchmark needs {args.world_size} CUDA "
            f"devices, found {torch.cuda.device_count()}"
        )

    os.environ["B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER"] = (
        "1" if args.mode == "double" else "0"
    )
    os.environ["B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION"] = (
        "1" if args.mode == "deferred" else "0"
    )
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            args.mode,
            args.layers,
            args.gap_cycles,
            args.stress_replays,
            args.warmup,
            args.iters,
            args.samples,
            args.with_dcp_pool,
            tuple(args.shapes),
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
