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
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from ._hierarchical_cute import get_hierarchical_launcher
from .pcie_oneshot import _broadcast_gather_object, _normalize_device


ISLAND_SIZE = 4
SUPPORTED_WORLD_SIZES = (12, 16)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)
_ALIGNMENT = 256
_HEADER_BYTES = 69_888


def _wait_nanosleep_cycles_from_env() -> int:
    raw = os.getenv("B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES", "24")
    try:
        cycles = int(raw)
    except ValueError as exc:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES must be an integer"
        ) from exc
    if not 0 <= cycles <= 1024:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES must be in [0, 1024]"
        )
    return cycles


def _threads_from_env() -> int:
    raw = os.getenv("B12X_PCIE_HIERARCHICAL_THREADS", "224")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_THREADS must be an integer"
        ) from exc
    if not 32 <= threads <= 1024 or threads % 32 != 0:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_THREADS must be a multiple of 32 "
            "in [32, 1024]"
        )
    return threads


def _vectorized_bf16x2_from_env() -> bool:
    raw = os.getenv("B12X_PCIE_HIERARCHICAL_BF16X2", "1")
    if raw not in ("0", "1"):
        raise ValueError("B12X_PCIE_HIERARCHICAL_BF16X2 must be 0 or 1")
    return raw == "1"


def _vectorized_bf16x2_max_elements_from_env() -> int:
    raw = os.getenv(
        "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS",
        "7168",
    )
    try:
        max_elements = int(raw)
    except ValueError as exc:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS must be an integer"
        ) from exc
    if not 0 <= max_elements <= 1 << 30:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS must be in [0, 2**30]"
        )
    return max_elements


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class _SlabLayout:
    stage: tuple[int, int]
    partial: tuple[int, int]
    final: tuple[int, int]
    bytes: int


def _make_layout(max_elements: int) -> _SlabLayout:
    """Mirror the native header/stage/partial/final layout exactly."""

    if max_elements <= 0:
        raise ValueError("max_elements must be positive")
    stages: list[int] = []
    partials: list[int] = []
    finals: list[int] = []
    offset = _align_up(_HEADER_BYTES)
    for _ in range(2):
        stages.append(offset)
        partials.append(_align_up(stages[-1] + int(max_elements) * 2))
        finals.append(_align_up(partials[-1] + int(max_elements) * 4))
        offset = _align_up(finals[-1] + int(max_elements) * 2)
    return _SlabLayout(
        stage=(stages[0], stages[1]),
        partial=(partials[0], partials[1]),
        final=(finals[0], finals[1]),
        bytes=offset,
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


def _buffer_modes_from_env() -> tuple[bool, bool]:
    """Return mutually exclusive experimental synchronization modes."""

    double_buffered = (
        os.getenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "0") == "1"
    )
    deferred_consumption = (
        os.getenv(
            "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
            "0",
        )
        == "1"
    )
    if double_buffered and deferred_consumption:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER and "
            "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION are "
            "mutually exclusive"
        )
    return double_buffered, deferred_consumption


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
        del ext_module  # native-extension injection is obsolete
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
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
        (
            self.double_buffered,
            self.deferred_consumption,
        ) = _buffer_modes_from_env()
        self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()
        self.threads = _threads_from_env()
        self.vectorized_bf16x2 = _vectorized_bf16x2_from_env()
        self.vectorized_bf16x2_max_elements = (
            _vectorized_bf16x2_max_elements_from_env()
        )
        self._layout = _make_layout(self.max_elements)
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._slab_ptrs: tuple[int, ...] = ()
        self._launchers: dict[bool, object] = {}
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False

        slab_bytes = self._layout.bytes
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
            # Keep the fixed slab addresses in the captured kernel node's
            # parameter bank.  The CuTe rank specialization references only
            # this rank's mapped peers, so unmapped entries remain zero and
            # are dead at compile time.
            self._slab_ptrs = tuple(peer_ptrs)
            # Resolve and load the only reachable specialization before the
            # channel is exposed.  A first call made under CUDA graph capture
            # must never compile or load a module.
            with torch.cuda.device(self.device):
                for vectorized in ({False, True} if self.vectorized_bf16x2 else {False}):
                    self._launchers[vectorized] = get_hierarchical_launcher(
                        self.world_size,
                        self.rank,
                        self.device.index or 0,
                        threads=(112 if vectorized else self.threads),
                        wait_nanosleep_cycles=self.wait_nanosleep_cycles,
                        double_buffered=self.double_buffered,
                        deferred_consumption=self.deferred_consumption,
                        vectorized_bf16x2=vectorized,
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
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del stream, channel_id
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
        if self.double_buffered and selected_blocks != _pick_blocks(inp.numel()):
            raise ValueError(
                "double-buffered hierarchical all-reduce requires the "
                "automatic K3 launch geometry"
            )
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
        assert len(self._slab_ptrs) == self.world_size
        vectorized = (
            self.vectorized_bf16x2
            and inp.numel() <= self.vectorized_bf16x2_max_elements
            and inp.data_ptr() % 4 == 0
            and out.data_ptr() % 4 == 0
        )
        launcher = self._launchers[vectorized]
        with torch.cuda.device(self.device):
            launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self._layout.stage[0],
                self._layout.partial[0],
                self._layout.final[0],
                self._layout.stage[1],
                self._layout.partial[1],
                self._layout.final[1],
                inp.numel(),
                selected_blocks,
            )
        return out

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Accept semantic owner names without allocating additional channels.

        The hierarchical runtime has one ordered channel. Owner names provide
        API compatibility for callers that serialize TP12/TP16 collectives;
        they do not permit overlapping collective streams.
        """

        del channel_ids

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ) -> "PCIeHierarchicalAllReduce":
        """Compatibility with the vLLM PCIe runtime interface.

        Synchronization generations live in device memory, so captured graphs
        do not require host-side channel patching.  The runtime remains a
        single ordered channel; callers must not overlap collective streams.
        """

        del stream, channel_id
        return self

    @contextmanager
    def capture(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        del stream, channel_id
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
        self._slab_ptrs = ()
        self._launchers.clear()
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
        # Never enter distributed barriers from asymmetric interpreter
        # teardown. Explicit/context-manager close owns coordinated release.
        return None
