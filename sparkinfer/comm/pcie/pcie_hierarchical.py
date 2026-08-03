"""Bounded-degree hierarchical TP12/TP16 all-reduce runtime.

This runtime is specialized for single-node topologies with three or four
contiguous four-GPU PCIe islands. Unlike the ordinary oneshot collective, it
does not map every rank into every CUDA context: non-leaders map one peer and
island leaders map five peers at TP12 or six at TP16. The collective is
CUDA-graph capturable and stages arbitrary BF16 inputs into fixed IPC storage
before reducing them.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import _broadcast_gather_object


ISLAND_SIZE = 4
SUPPORTED_WORLD_SIZES = (12, 16)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_hierarchical.cu")
    verbose = os.getenv("SPARKINFER_PCIE_HIERARCHICAL_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sparkinfer_pcie_hierarchical_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _selected_peers(rank: int, world_size: int) -> tuple[int, ...]:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"hierarchical all-reduce supports {SUPPORTED_WORLD_SIZES}, "
            f"got TP{world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island = rank // ISLAND_SIZE
    local_rank = rank % ISLAND_SIZE
    leader = island * ISLAND_SIZE
    if local_rank != 0:
        return (leader,)
    local_peers = tuple(range(leader, leader + ISLAND_SIZE))
    peer_leaders = tuple(range(0, world_size, ISLAND_SIZE))
    return tuple(sorted(set(local_peers + peer_leaders) - {rank}))


def _pick_blocks(elements: int) -> int:
    """Select the measured K3 decode launch geometry."""

    if elements <= 0:
        raise ValueError("elements must be positive")
    return 16 if elements <= 4096 else 32


class PCIeHierarchicalAllReduce:
    """Single-channel BF16 TP12/TP16 all-reduce with bounded peer degree."""

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_elements: int,
        blocks: Optional[int] = None,
        ext_module=None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        )
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "hierarchical all-reduce requires "
                f"{SUPPORTED_WORLD_SIZES}, got TP{self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("hierarchical all-reduce requires a CUDA device")
        if blocks is not None and blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if max_elements <= 0:
            raise ValueError("max_elements must be positive")

        self.max_elements = int(max_elements)
        self.blocks = None if blocks is None else int(blocks)
        self._ext = ext_module or _load_extension()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
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
            for peer in _selected_peers(self.rank, self.world_size):
                remote_ptr = self._ipc.cudaIpcOpenMemHandleBytes(handles[peer])
                peer_ptrs[peer] = remote_ptr
                self._remote_ptrs.append(remote_ptr)
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

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return (
            not self._closed
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and 0 < inp.numel() <= self.max_elements
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        blocks: Optional[int] = None,
        stream: object = None,
    ) -> torch.Tensor:
        del stream
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy hierarchical all-reduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device})"
            )
        if blocks is not None:
            selected_blocks = int(blocks)
        elif self.blocks is not None:
            selected_blocks = self.blocks
        else:
            selected_blocks = _pick_blocks(inp.numel())
        if selected_blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if out is None:
            out = torch.empty_like(inp)
        if (
            out.device != inp.device
            or out.dtype != inp.dtype
            or out.shape != inp.shape
            or not out.is_contiguous()
        ):
            raise ValueError(
                "output must match input shape/dtype/device and be contiguous"
            )
        self._ext.all_reduce_bf16(
            self._runtime,
            inp,
            out,
            selected_blocks,
        )
        return out

    def for_stream(self, stream: object = None) -> "PCIeHierarchicalAllReduce":
        """Compatibility with the vLLM PCIe runtime interface.

        Synchronization generations live in device memory, so captured graphs
        do not require host-side channel patching.  The runtime remains a
        single ordered channel; callers must not overlap collective streams.
        """

        del stream
        return self

    @contextmanager
    def capture(self, stream: object = None):
        del stream
        yield self

    def register_graph_buffers(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
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

    def __enter__(self) -> "PCIeHierarchicalAllReduce":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            with suppress(Exception):
                self.close()
