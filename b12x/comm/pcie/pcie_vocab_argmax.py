"""Bounded-degree TP16 vocabulary argmax runtime."""

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


WORLD_SIZE = 16
ISLAND_SIZE = 4
MAX_BATCH_SIZE = 8


def _exchange_ipc_handles(
    local_handle: object,
    group: ProcessGroup,
) -> list[object]:
    """Exchange CUDA IPC handles over NCCL or a metadata-only Gloo group."""

    backend = str(dist.get_backend(group=group)).lower()
    if "nccl" in backend:
        return _broadcast_gather_object(local_handle, group)
    if "gloo" not in backend:
        raise ValueError(
            "vocabulary argmax IPC exchange requires an NCCL or Gloo group, "
            f"got {backend}"
        )

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_objects: list[list[object | None]] = [[None] for _ in range(world_size)]
    all_objects[rank][0] = local_handle
    for index, src_rank in enumerate(dist.get_process_group_ranks(group)):
        dist.broadcast_object_list(all_objects[index], src=src_rank, group=group)
    return [entry[0] for entry in all_objects]


def _wait_nanosleep_cycles_from_env() -> int:
    raw = os.getenv("B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES", "24")
    try:
        cycles = int(raw)
    except ValueError as exc:
        raise ValueError(
            "B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES must be an integer"
        ) from exc
    if not 0 <= cycles <= 1024:
        raise ValueError(
            "B12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES must be in [0, 1024]"
        )
    return cycles


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_vocab_argmax.cu")
    wait_cycles = _wait_nanosleep_cycles_from_env()
    return load(
        name=f"b12x_pcie_vocab_argmax_ext_ns{wait_cycles}",
        sources=[str(source)],
        extra_cuda_cflags=[
            "-O3",
            f"-DB12X_PCIE_VOCAB_ARGMAX_NANOSLEEP_CYCLES={wait_cycles}",
        ],
        extra_ldflags=["-lcuda"],
        verbose=(os.getenv("B12X_PCIE_VOCAB_ARGMAX_VERBOSE_BUILD", "0") == "1"),
    )


def _selected_peers(rank: int, world_size: int = WORLD_SIZE) -> tuple[int, ...]:
    """Return three island peers plus the same lane in other islands."""

    if world_size != WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError(
            f"vocabulary argmax requires TP{WORLD_SIZE}, got rank={rank}, "
            f"TP={world_size}"
        )
    island = rank // ISLAND_SIZE
    lane = rank % ISLAND_SIZE
    local = set(range(island * ISLAND_SIZE, (island + 1) * ISLAND_SIZE))
    cross_island = set(range(lane, world_size, ISLAND_SIZE))
    return tuple(sorted((local | cross_island) - {rank}))


class PCIeVocabParallelArgmax:
    """Fuse BF16 add and exact global greedy sampling across TP16 shards."""

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
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
        if self.world_size != WORLD_SIZE:
            raise ValueError(
                f"vocabulary argmax requires TP{WORLD_SIZE}, got TP{self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("vocabulary argmax requires a CUDA device")
        if local_vocab_size <= 0 or local_vocab_size * self.world_size >= 1 << 31:
            raise ValueError("global vocabulary must fit a positive int32 index")
        if not 0 < max_batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be in [1, {MAX_BATCH_SIZE}]")

        self.local_vocab_size = int(local_vocab_size)
        self.max_batch_size = int(max_batch_size)
        self._ext = ext_module or _load_extension()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._runtime = 0
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False

        slab_bytes = int(self._ext.slab_bytes())
        peer_ptrs = [0] * self.world_size
        try:
            self._local_ptr = self._ipc.cudaMalloc(slab_bytes)
            self._ipc.cudaMemset(self._local_ptr, 0, slab_bytes)
            local_handle = self._ipc.cudaIpcGetMemHandleBytes(self._local_ptr)
            handles = _exchange_ipc_handles(local_handle, exchange_group)
            peer_ptrs[self.rank] = self._local_ptr
            for peer in _selected_peers(self.rank, self.world_size):
                remote_ptr = self._ipc.cudaIpcOpenMemHandleBytes(handles[peer])
                peer_ptrs[peer] = remote_ptr
                self._remote_ptrs.append(remote_ptr)
            self._runtime = int(
                self._ext.init_runtime(
                    peer_ptrs,
                    self.rank,
                    self.local_vocab_size,
                    self.max_batch_size,
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

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
        ext_module=None,
    ) -> "PCIeVocabParallelArgmax":
        return cls(
            exchange_group=exchange_group,
            device=device,
            local_vocab_size=local_vocab_size,
            max_batch_size=max_batch_size,
            ext_module=ext_module,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        local_vocab_size: int,
        max_batch_size: int = MAX_BATCH_SIZE,
        ext_module=None,
    ) -> "PCIeVocabParallelArgmax":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            local_vocab_size=local_vocab_size,
            max_batch_size=max_batch_size,
            ext_module=ext_module,
        )

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    def fused_add_argmax(
        self,
        base: torch.Tensor,
        bias: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return exact global argmax of the BF16-rounded local sum."""

        if self._closed:
            raise RuntimeError("vocabulary argmax runtime is closed")
        if base.device != self.device or bias.device != self.device:
            raise ValueError("inputs must be on the runtime device")
        if base.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
            raise ValueError("inputs must be BF16")
        if base.ndim != 2 or base.shape != bias.shape:
            raise ValueError("inputs must have matching [batch, local_vocab] shapes")
        batch, local_vocab = (int(value) for value in base.shape)
        if not 0 < batch <= self.max_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds capacity {self.max_batch_size}"
            )
        if local_vocab != self.local_vocab_size:
            raise ValueError(
                f"local vocabulary must be {self.local_vocab_size}, got {local_vocab}"
            )
        if base.stride(1) != 1 or bias.stride(1) != 1:
            raise ValueError("input last dimensions must be contiguous")
        if base.stride(0) <= 0 or bias.stride(0) <= 0:
            raise ValueError("input row strides must be positive")
        if out is None:
            out = torch.empty(batch, dtype=torch.int64, device=self.device)
        if (
            out.device != self.device
            or out.dtype != torch.int64
            or out.shape != (batch,)
            or not out.is_contiguous()
        ):
            raise ValueError("output must be contiguous int64 [batch] on the device")
        self._ext.fused_add_argmax(self._runtime, base, bias, out)
        return out

    def for_stream(self, stream: object = None) -> "PCIeVocabParallelArgmax":
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

    def __enter__(self) -> "PCIeVocabParallelArgmax":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            with suppress(Exception):
                self.close()


__all__ = ["PCIeVocabParallelArgmax"]
