"""Pull-based island reduce-scatter TP16 all-reduce runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._island_rs_cute import (
    HEADER_BYTES,
    MAX_BLOCKS,
    get_island_rs_launcher,
    island_rs_peers,
)
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import _broadcast_gather_object, _normalize_device


SUPPORTED_WORLD_SIZES = (16,)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)
_ISLAND_SIZE = 4
_ALIGNMENT = 256


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _threads_from_env() -> int:
    # 512 is the measured TP16 optimum: each phase moves only about 86KB, so
    # the kernel needs enough concurrent reads to cover the PCIe round trip.
    # At [8, 7168] the same kernel costs 49.8 us with 128 threads and 25.9 with
    # 512; 1024 gives nothing back.
    raw = os.getenv("B12X_PCIE_ISLAND_RS_THREADS", "512")
    threads = int(raw)
    if not 32 <= threads <= 1024 or threads % 32:
        raise ValueError(
            "B12X_PCIE_ISLAND_RS_THREADS must be a multiple of 32 in [32, 1024]"
        )
    return threads


def _wait_nanosleep_cycles_from_env() -> int:
    cycles = int(os.getenv("B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES", "24"))
    if not 0 <= cycles <= 1024:
        raise ValueError(
            "B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES must be in [0, 1024]"
        )
    return cycles


def _pick_blocks(elements: int) -> int:
    """Measured TP16 launch geometry; 32 blocks regress, 8 under-fill."""

    if elements <= 4096:
        return 8
    return 16


# Below this the leader-gather hierarchy is still ahead (18.16 vs 20.51 us at
# one row of 7168), because it has a shorter critical path for the ranks that
# are not the leader. Above it the leader's link is the bottleneck and the
# equal-quarter split wins: 22.44 vs 30.36 us at four rows, 26.03 vs 46.79 at
# eight. Expressed in elements so it does not depend on the model's hidden size.
CROSSOVER_ELEMENTS = 14_336


class PCIeIslandRSAllReduce:
    """Equal-quarter BF16 TP16 all-reduce with no leader hotspot.

    Every rank owns one quarter of the vector and reaches six peers: the three
    other lanes of its four-GPU island, and the same lane in the three other
    islands. Peer degree is bounded at six, and no rank carries the whole
    vector on behalf of the others.
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_elements: int,
        blocks: Optional[int] = None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"island reduce-scatter requires TP16, got TP{self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("island reduce-scatter requires a CUDA device")
        if blocks is not None and blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if max_elements <= 0 or max_elements % 2:
            raise ValueError("max_elements must be a positive even count")

        self.max_elements = int(max_elements)
        self.blocks = None if blocks is None else int(blocks)
        self.threads = _threads_from_env()
        self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()

        max_pairs = self.max_elements // 2
        # Quarter stride in bf16x2 words, shared by the scatter and exchange
        # regions so both index by (source, word) with a compile-time stride.
        self.quarter_capacity = _align_up(
            (max_pairs + _ISLAND_SIZE - 1) // _ISLAND_SIZE, 8
        )
        quarter_bytes = self.quarter_capacity * 4
        self.stage_offset = _align_up(HEADER_BYTES)
        self.part_offset = _align_up(self.stage_offset + self.max_elements * 2)
        self.final_offset = _align_up(self.part_offset + quarter_bytes)
        slab_bytes = _align_up(self.final_offset + self.max_elements * 2)

        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._slab_ptrs: tuple[int, ...] = ()
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False
        self._launcher = None

        peer_ptrs = [0] * self.world_size
        try:
            self._local_ptr = self._ipc.cudaMalloc(slab_bytes)
            self._ipc.cudaMemset(self._local_ptr, 0, slab_bytes)
            local_handle = self._ipc.cudaIpcGetMemHandleBytes(self._local_ptr)
            handles = _broadcast_gather_object(local_handle, exchange_group)
            peer_ptrs[self.rank] = self._local_ptr
            for peer in island_rs_peers(self.rank, self.world_size):
                remote_ptr = self._ipc.cudaIpcOpenMemHandleBytes(handles[peer])
                peer_ptrs[peer] = remote_ptr
                self._remote_ptrs.append(remote_ptr)
            self._slab_ptrs = tuple(peer_ptrs)
            with torch.cuda.device(self.device):
                self._launcher = get_island_rs_launcher(
                    self.world_size,
                    self.rank,
                    self.device.index or 0,
                    threads=self.threads,
                    wait_nanosleep_cycles=self.wait_nanosleep_cycles,
                )
        except Exception:
            self.close()
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
            and inp.numel() % 2 == 0
            and inp.data_ptr() % 4 == 0
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        blocks: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy island reduce-scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            out = torch.empty_like(inp)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or out.shape != inp.shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError("output must match input and be 4-byte aligned")
        if blocks is not None:
            selected = int(blocks)
        elif self.blocks is not None:
            selected = self.blocks
        else:
            selected = _pick_blocks(inp.numel())
        if selected not in SUPPORTED_BLOCKS or selected > MAX_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        with torch.cuda.device(self.device):
            self._launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self.stage_offset,
                self.part_offset,
                self.final_offset,
                self.quarter_capacity,
                inp.numel(),
                selected,
            )
        return out

    def for_stream(self, stream: object = None) -> "PCIeIslandRSAllReduce":
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
            with suppress(Exception):
                torch.cuda.synchronize(self.device)
        for ptr in self._remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        self._remote_ptrs.clear()
        if self._local_ptr:
            with suppress(Exception):
                self._ipc.cudaFree(self._local_ptr)
            self._local_ptr = 0


__all__ = ["PCIeIslandRSAllReduce", "SUPPORTED_WORLD_SIZES"]
