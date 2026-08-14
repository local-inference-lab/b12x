"""Exact owner-sharded PCIe transport for DCP sparse top-k."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from threading import Condition, RLock
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from ._dcp_cute_common import signal_bytes
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    _align_up,
    _current_stream_key,
    _device_guard,
    _exchange_close_failures,
    _finish_collective_runtime_setup,
    _format_close_error,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _raise_coordinated_close_error,
    _raise_local_cleanup_errors,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_close_phase,
    _run_collective_preallocation_setup,
)


SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8)
_MAX_BLOCKS = 128
_SIGNAL_BYTES = signal_bytes(_MAX_BLOCKS)

_DCP_TOPK_CONTRACT_VERSION = 1


def _validate_launch_config(*, threads: int, block_limit: int, world_size: int) -> None:
    if threads < world_size or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a multiple of 32 in [32, 512]")
    if block_limit <= 0 or block_limit > 128:
        raise ValueError("block_limit must be in [1, 128]")


@dataclass(frozen=True)
class _DoubleBufferLayout:
    signal_bytes: int
    staging0_offset: int
    staging1_offset: int
    slot_bytes: int
    slab_bytes: int
    plane_bytes: int = 0


def _candidate_staging_layout(
    *,
    signal_bytes: int,
    max_rows: int,
    topk: int,
    world_size: int,
) -> _DoubleBufferLayout:
    if signal_bytes <= 0:
        raise ValueError("signal_bytes must be positive")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"unsupported world size {world_size}")
    if max_rows <= 0 or max_rows % world_size != 0:
        raise ValueError("max_rows must be positive and divisible by world_size")
    if topk <= 0 or topk % 4 != 0:
        raise ValueError("topk must be a positive multiple of 4")

    max_owner_rows = max_rows // world_size
    plane_bytes = max_owner_rows * world_size * topk * 4
    slot_bytes = _align_up(plane_bytes * 2, IPC_SLAB_ALIGNMENT)
    staging0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
    staging1_offset = staging0_offset + slot_bytes
    return _DoubleBufferLayout(
        signal_bytes=signal_bytes,
        staging0_offset=staging0_offset,
        staging1_offset=staging1_offset,
        slot_bytes=slot_bytes,
        slab_bytes=staging1_offset + slot_bytes,
        plane_bytes=plane_bytes,
    )


_DCP_TOPK_CONTRACT_FIELDS = (
    "version",
    "topology_world",
    "topology_supported_world_sizes",
    "capacity_max_rows",
    "capacity_max_owner_rows",
    "capacity_topk",
    "capacity_candidate_plane_elems",
    "allocation_slab_bytes",
    "allocation_slot_bytes",
    "allocation_plane_bytes",
    "offset_signal_bytes",
    "offset_staging0",
    "offset_staging1",
    "wire_codec_index",
    "wire_codec_score",
    "schedule_max_blocks",
    "schedule_signal_bytes",
)


def _dcp_topk_runtime_contract(
    *,
    world_size: int,
    max_rows: int,
    topk: int,
    layout: _DoubleBufferLayout,
) -> tuple:
    """Build the versioned fixed-schema contract every rank must agree on.

    The tuple covers every allocation size, offset, capacity, dtype/wire
    codec, topology constraint, and schedule/launch knob that a peer needs
    to interpret remote memory correctly.  All ranks must exchange and
    compare this contract before any IPC allocation or kernel launch so a
    divergent or compromised rank cannot turn peer DMA into out-of-bounds
    access.
    """
    max_owner_rows = int(max_rows) // int(world_size)
    candidate_plane_elems = max_owner_rows * int(world_size) * int(topk)
    return (
        _DCP_TOPK_CONTRACT_VERSION,
        # topology
        int(world_size),
        SUPPORTED_WORLD_SIZES,
        # capacity / row + expert knobs
        int(max_rows),
        max_owner_rows,
        int(topk),
        candidate_plane_elems,
        # allocation sizes
        int(layout.slab_bytes),
        int(layout.slot_bytes),
        int(layout.plane_bytes),
        # offsets
        int(layout.signal_bytes),
        int(layout.staging0_offset),
        int(layout.staging1_offset),
        # wire codec
        "int32_indices",
        "float32_scores",
        # schedule / launch knobs
        _MAX_BLOCKS,
        int(_SIGNAL_BYTES),
    )


def _tensor_from_cuda_pointer(
    pointer: int,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a non-owning tensor view over channel-owned CUDA storage."""
    numel = 1
    for extent in shape:
        numel *= int(extent)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    storage = torch._C._construct_storage_from_data_pointer(
        int(pointer),
        device,
        int(nbytes),
    )
    stride: list[int] = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= int(extent)
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        {
            "dtype": dtype,
            "size": tuple(int(value) for value in shape),
            "stride": tuple(reversed(stride)),
            "storage_offset": 0,
        },
        storage,
    )


def owner_stage_reference(
    rank_indices: torch.Tensor,
    rank_scores: torch.Tensor,
    owner_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one DCP owner's exact row-major candidate planes."""
    if rank_indices.shape != rank_scores.shape or rank_indices.ndim != 3:
        raise ValueError(
            "rank candidates must have matching [world, rows, topk] shapes"
        )
    world_size, rows, _ = rank_indices.shape
    if rows % world_size != 0:
        raise ValueError("rows must be divisible by world size")
    if not 0 <= owner_rank < world_size:
        raise ValueError("invalid owner rank")
    owner_rows = rows // world_size
    row_slice = slice(owner_rank * owner_rows, (owner_rank + 1) * owner_rows)
    return (
        rank_indices[:, row_slice].transpose(0, 1).flatten(1).contiguous(),
        rank_scores[:, row_slice].transpose(0, 1).flatten(1).contiguous(),
    )


class _ChannelLifecycle(Enum):
    """Lifecycle state for the IPC channel.

    OPEN — channel accepts launches/captures and is not closing.
    CLOSING — one thread owns the collective teardown; launches are
    rejected and concurrent close callers wait for the result.
    CLOSED — teardown completed successfully; exports are freed.
    FAILED — teardown failed; imports/exports are retained.  The channel
    is permanently non-launchable but ``retry_close()`` may re-enter the
    collective teardown from this state.
    """

    OPEN = 0
    CLOSING = 1
    CLOSED = 2
    FAILED = 3


class _IPCChannel:
    def _init_channel(
        self,
        *,
        device: torch.device | int | str,
        exchange_group: Optional[ProcessGroup],
        ipc: Optional[CudaRTLibrary],
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]],
        stream_affine: bool,
    ) -> None:
        owned_buffers_list = list(owned_buffers or ())
        if exchange_group is None and any(
            shared.remote_ptrs for shared in owned_buffers_list
        ):
            raise ValueError(
                "exchange_group is required when the channel owns remote "
                "CUDA IPC imports"
            )
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self._ipc = ipc
        self._owned_buffers = owned_buffers_list
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        self._lifecycle_lock = RLock()
        self._lifecycle_cond = Condition(self._lifecycle_lock)
        self._lifecycle_state = _ChannelLifecycle.OPEN
        self._close_generation = 0
        self._close_error: Optional[BaseException] = None
        self._launch_depth = 0
        self._capture_context_depth = 0
        self._replay_depth = 0
        # Graphs registered by this channel for gated replay.
        self._registered_graphs: set[int] = set()
        # Collective ticket: set on global init success, never before.
        self._has_collective_ipc_ticket = False

    # ------------------------------------------------------------------
    # Launch gate — held from the closed-check through kernel enqueue so
    # close cannot transition to CLOSING while a launch is in flight.
    # ------------------------------------------------------------------

    @contextmanager
    def _launch_gate(self):
        with self._lifecycle_lock:
            if self._lifecycle_state != _ChannelLifecycle.OPEN:
                raise RuntimeError(
                    f"{type(self).__name__} is "
                    f"{self._lifecycle_state.name.lower()}; cannot stage candidates"
                )
            self._launch_depth += 1
        try:
            yield
        finally:
            with self._lifecycle_lock:
                self._launch_depth -= 1
                if self._launch_depth == 0:
                    self._lifecycle_cond.notify_all()

    # ------------------------------------------------------------------
    # Replay gate — held while a captured graph replays so close waits.
    # ------------------------------------------------------------------

    @contextmanager
    def _replay_gate(self):
        with self._lifecycle_lock:
            if self._lifecycle_state != _ChannelLifecycle.OPEN:
                raise RuntimeError(
                    f"{type(self).__name__} is "
                    f"{self._lifecycle_state.name.lower()}; cannot replay"
                )
            self._replay_depth += 1
        try:
            yield
        finally:
            with self._lifecycle_lock:
                self._replay_depth -= 1
                if self._replay_depth == 0:
                    self._lifecycle_cond.notify_all()

    # ------------------------------------------------------------------
    # Close transition — atomically stop launches/replays/captures, wait
    # for in-flight work, and let one thread enter the collective.
    # ------------------------------------------------------------------

    def _begin_close(self, *, allow_retry: bool = False) -> bool:
        """Transition to CLOSING and return True if this caller owns the
        collective teardown.

        From OPEN: this caller owns the collective.
        From FAILED with allow_retry=True: this caller owns a retry.
        From CLOSING: wait for the active generation, observe its result.
        From CLOSED: return False (idempotent).
        From FAILED without allow_retry: re-raise the stored error.
        """
        with self._lifecycle_lock:
            if self._lifecycle_state == _ChannelLifecycle.CLOSED:
                return False
            if self._lifecycle_state == _ChannelLifecycle.FAILED:
                if not allow_retry:
                    assert self._close_error is not None
                    raise self._close_error
                # Retry: fall through to claim the collective.
            elif self._lifecycle_state == _ChannelLifecycle.CLOSING:
                generation = self._close_generation
                while (
                    self._lifecycle_state == _ChannelLifecycle.CLOSING
                    and self._close_generation == generation
                ):
                    self._lifecycle_cond.wait()
                if self._lifecycle_state == _ChannelLifecycle.FAILED:
                    assert self._close_error is not None
                    raise self._close_error
                return False
            # Claim the collective.
            self._lifecycle_state = _ChannelLifecycle.CLOSING
            self._close_generation += 1
            self._close_error = None
            # Wait for in-flight launches, replays, and capture contexts.
            while (
                self._launch_depth > 0
                or self._replay_depth > 0
                or self._capture_context_depth > 0
            ):
                self._lifecycle_cond.wait()
            return True

    def _finish_close(self, *, error: Optional[BaseException] = None) -> None:
        """Complete the CLOSING→CLOSED/FAILED transition."""
        with self._lifecycle_lock:
            if error is not None:
                self._lifecycle_state = _ChannelLifecycle.FAILED
                self._close_error = error
            else:
                self._lifecycle_state = _ChannelLifecycle.CLOSED
                self._coordinated_close_complete = True
            self._lifecycle_cond.notify_all()

    def _bind_stream(self) -> None:
        if not self._stream_affine or self.device.type != "cuda":
            return
        if _is_current_stream_capturing(self.device):
            return
        stream_key = _current_stream_key(self.device)
        if stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
        elif self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                f"{type(self).__name__} is stream-affine; create one instance "
                "per CUDA stream"
            )

    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        if self._ipc is None:
            return not any(shared.remote_ptrs for shared in self._owned_buffers)
        return all(
            (buffer_index, remote_index) in closed
            for buffer_index, shared in enumerate(self._owned_buffers)
            for remote_index, _ in enumerate(shared.remote_ptrs)
        )

    def _quiesce_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _has_ipc_exports(self) -> bool:
        return bool(getattr(self, "_owned_buffers", ()))

    def _has_remote_imports(self) -> bool:
        return any(
            shared.remote_ptrs for shared in getattr(self, "_owned_buffers", ())
        )

    def _close_ipc_imports(self) -> None:
        """Legacy best-effort close retained for non-strict callers."""
        if self._ipc_imports_closed:
            return
        self._closed = True
        if self._ipc is not None:
            for shared in self._owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        self._ipc.cudaIpcCloseMemHandle(ptr)
        self._ipc_imports_closed = True

    def _close_ipc_imports_strict(self) -> None:
        """Close every IPC import without suppressing errors."""
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        closed = self._closed_import_indices()
        if self._ipc is not None:
            for buffer_index, shared in enumerate(self._owned_buffers):
                for remote_index, ptr in enumerate(shared.remote_ptrs):
                    key = (buffer_index, remote_index)
                    if key in closed:
                        continue
                    try:
                        self._ipc.cudaIpcCloseMemHandle(ptr)
                    except Exception as exc:
                        failures.append((f"CUDA IPC import {ptr}", exc))
                    else:
                        closed.add(key)
        elif any(shared.remote_ptrs for shared in self._owned_buffers):
            failures.append(
                (
                    "CUDA IPC imports",
                    RuntimeError("CUDA runtime is unavailable for IPC unmap"),
                )
            )

        if not failures and self._all_python_ipc_imports_closed(closed):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors("PCIe DCP top-k", "IPC import close", failures)

    def _free_ipc_exports(self) -> None:
        """Legacy best-effort free retained for non-strict callers."""
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports()
        self._candidate_views = ()
        if self._ipc is not None:
            for shared in self._owned_buffers:
                with suppress(Exception):
                    self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()
        self._ipc_exports_freed = True

    def _free_ipc_exports_strict(self) -> None:
        """Free exported IPC storage only after all imports are closed."""
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining: list[_OwnedSharedBuffer] = []
        self._candidate_views = ()
        if self._ipc is not None:
            for shared in self._owned_buffers:
                try:
                    self._ipc.cudaFree(shared.local_ptr)
                except Exception as exc:
                    remaining.append(shared)
                    failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        elif self._owned_buffers:
            remaining = list(self._owned_buffers)
            failures.append(
                (
                    "CUDA IPC exports",
                    RuntimeError("CUDA runtime is unavailable for export free"),
                )
            )
        self._owned_buffers = remaining
        if not remaining:
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors("PCIe DCP top-k", "IPC export free", failures)

    def _fail_closed_multi_rank_without_group(self) -> None:
        """Reject freeing multi-rank IPC exports without a teardown group.

        Only applies when the channel has remote imports (peer pointers to
        another rank's export).  A channel with only local exports and no
        remote imports can close without a group.
        """
        world_size = getattr(self, "world_size", 1)
        if (
            world_size >= 2
            and self._has_remote_imports()
            and self.exchange_group is None
        ):
            self._closed = True
            raise RuntimeError(
                f"{type(self).__name__} has remote IPC imports for "
                f"world_size={world_size} but no exchange group; cannot "
                "collectively acknowledge peer unmap before freeing exports"
            )

    def _coordinated_close(self, *, allow_retry: bool = False) -> None:
        """Collective teardown with explicit lifecycle state."""
        if not self._begin_close(allow_retry=allow_retry):
            return
        error: Optional[BaseException] = None
        try:
            self._fail_closed_multi_rank_without_group()

            quiesce_error: BaseException | None = None
            try:
                self._quiesce_device()
            except Exception as exc:
                quiesce_error = exc

            if self.exchange_group is not None:
                quiesce_status = _exchange_close_failures(
                    (str(quiesce_error),) if quiesce_error else (),
                    exchange_group=self.exchange_group,
                    phase="DCP top-k quiescence",
                )
                if any(quiesce_status):
                    _raise_coordinated_close_error(
                        "DCP top-k quiescence",
                        quiesce_status,
                        local_errors=(
                            [(0, quiesce_error)] if quiesce_error else []
                        ),
                        exports_retained=True,
                    )
                dist.barrier(group=self.exchange_group)
            elif quiesce_error is not None:
                raise quiesce_error

            unmap_errors = _run_close_phase(
                (self,),
                strict_method="_close_ipc_imports_strict",
                legacy_method="_close_ipc_imports",
            )
            unmap_status = _exchange_close_failures(
                tuple(
                    _format_close_error(index, error)
                    for index, error in unmap_errors
                ),
                exchange_group=self.exchange_group,
                phase="IPC unmap",
            )
            if any(unmap_status):
                _raise_coordinated_close_error(
                    "IPC unmap",
                    unmap_status,
                    local_errors=unmap_errors,
                    exports_retained=True,
                )

            if self.exchange_group is not None:
                dist.barrier(group=self.exchange_group)

            free_errors = _run_close_phase(
                (self,),
                strict_method="_free_ipc_exports_strict",
                legacy_method="_free_ipc_exports",
            )
            free_status = _exchange_close_failures(
                tuple(
                    _format_close_error(index, error)
                    for index, error in free_errors
                ),
                exchange_group=self.exchange_group,
                phase="IPC export free",
            )
            if any(free_status):
                _raise_coordinated_close_error(
                    "IPC export free",
                    free_status,
                    local_errors=free_errors,
                    exports_retained=False,
                )

            if self.exchange_group is not None:
                dist.barrier(group=self.exchange_group)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish_close(error=error)

    def close(self) -> None:
        self._coordinated_close()

    def close_coordinated(self) -> None:
        """Collectively close peer mappings before freeing exported storage."""
        self._coordinated_close()

    def retry_close(self) -> None:
        """Re-enter the collective teardown from FAILED state.

        Launches remain permanently disabled.  Every rank must call this
        method simultaneously to retry the quiesce/unmap/free phases.
        """
        self._coordinated_close(allow_retry=True)

    def __enter__(self) -> "_IPCChannel":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        # GC cannot synchronize or enter a distributed barrier.  Retain
        # any channel whose collective close did not complete so stale
        # pointers survive until process teardown.
        if getattr(self, "_coordinated_close_complete", False):
            return
        if self._has_ipc_exports() or getattr(
            self, "_has_collective_ipc_ticket", False
        ):
            _quarantine[id(self)] = self


class PCIeDCPTopKOwnerExchange(_IPCChannel):
    """Write exact DCP candidates directly into each row owner's IPC slab.

    CUDA graphs captured from one instance share a device epoch and must be
    replayed serially. Use a separate instance for independently replayable
    graphs. Returned candidate views alias channel-owned staging and their
    consumers must be enqueued on the capture stream before the next replay.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        staging0_ptrs: Sequence[int],
        staging1_ptrs: Sequence[int],
        max_rows: int,
        topk: int,
        exchange_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        ext_module=None,
        stream_affine: bool = True,
    ) -> None:
        # Compatibility-only: native extension injection was removed.
        del ext_module
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if max_rows <= 0 or max_rows % world_size != 0:
            raise ValueError("max_rows must be positive and divisible by world_size")
        if topk <= 0 or topk % 4 != 0:
            raise ValueError("topk must be a positive multiple of 4")
        if not (
            len(signal_ptrs) == len(staging0_ptrs) == len(staging1_ptrs) == world_size
        ):
            raise ValueError("signal and staging pointers must match world size")

        self._init_channel(
            device=device,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            stream_affine=stream_affine,
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_rows = int(max_rows)
        self.max_owner_rows = self.max_rows // self.world_size
        self.topk = int(topk)
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._staging_ptrs = (
            tuple(int(ptr) for ptr in staging0_ptrs),
            tuple(int(ptr) for ptr in staging1_ptrs),
        )
        self._candidate_plane_elems = (
            self.max_owner_rows * self.world_size * self.topk
        )
        self._next_slot = 0
        self._graph_slot: Optional[int] = None

        candidate_shape = (self.max_owner_rows, self.world_size * self.topk)
        if self.device.type == "cuda":
            self._candidate_views = tuple(
                (
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank],
                        candidate_shape,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank] + self._candidate_plane_elems * 4,
                        candidate_shape,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
                for slot_ptrs in self._staging_ptrs
            )
        else:
            # CPU construction remains useful for validation-only unit tests;
            # no launch is possible until tests install synthetic views.
            self._candidate_views = ()

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        topk: int,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPTopKOwnerExchange":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            if device_obj.type != "cuda":
                raise ValueError("DCP top-k owner exchange requires a CUDA device")
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            if max_rows <= 0 or max_rows % world_size != 0:
                raise ValueError(
                    "max_rows must be positive and divisible by world_size"
                )
            if topk <= 0 or topk % 4 != 0:
                raise ValueError("topk must be a positive multiple of 4")
            return device_obj

        device_obj = _run_collective_preallocation_setup(
            owner="PCIe DCP top-k argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe DCP top-k",
            required_sms=_MAX_BLOCKS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(device_obj.index or 0)
            prepared_layout = _candidate_staging_layout(
                signal_bytes=_SIGNAL_BYTES,
                max_rows=max_rows,
                topk=topk,
                world_size=world_size,
            )
            return prepared_ipc, prepared_layout

        ipc, layout = _run_collective_preallocation_setup(
            owner="PCIe DCP top-k",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe DCP top-k channel layout",
            exchange_group=exchange_group,
            contract=_dcp_topk_runtime_contract(
                world_size=world_size,
                max_rows=max_rows,
                topk=topk,
                layout=layout,
            ),
        )
        slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            layout.slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        runtime: Optional[PCIeDCPTopKOwnerExchange] = None
        init_error: BaseException | None = None
        try:
            # Construct WITHOUT slab ownership so a partial __init__ failure
            # does not create a quarantined object with live exports.  The
            # slab is attached only after the collective init verdict.
            runtime = cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=slab.peer_ptrs,
                staging0_ptrs=tuple(
                    ptr + layout.staging0_offset for ptr in slab.peer_ptrs
                ),
                staging1_ptrs=tuple(
                    ptr + layout.staging1_offset for ptr in slab.peer_ptrs
                ),
                max_rows=max_rows,
                topk=topk,
                exchange_group=exchange_group,
                ipc=ipc,
                owned_buffers=None,
                ext_module=ext_module,
                stream_affine=stream_affine,
            )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            # On rollback, clear any partial ownership the constructor may
            # have set (e.g. candidate views) so __del__ does not quarantine
            # a duplicate.
            if runtime is not None:
                runtime._owned_buffers.clear()
                runtime._candidate_views = ()
                runtime._has_collective_ipc_ticket = False

        _finish_collective_runtime_setup(
            owner="PCIe DCP top-k",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=slab,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )
        assert runtime is not None
        # Atomically attach the slab and set the collective ticket only on
        # global init success.
        runtime._owned_buffers = [slab]
        runtime._has_collective_ipc_ticket = True
        return runtime

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        topk: int,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeDCPTopKOwnerExchange":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            max_rows=max_rows,
            topk=topk,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    def stage_candidates(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        threads: int = 512,
        block_limit: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage exact candidates and return channel-owned owner views.

        Consumers must be enqueued on this channel's stream before another
        stage call or graph replay; the returned tensors are aliasing views,
        not snapshots. Capture must run inside :meth:`capture`. On first CUDA
        graph capture the channel permanently pins one staging slot. Every
        later launch begins with the native peer barrier, so no rank can
        overwrite that slot until every rank's prior same-stream consumer has
        retired. The barrier stays inside the one transport kernel and
        therefore adds no CUDA graph node.
        """
        with self._launch_gate(), _device_guard(self.device):
            return self._stage_candidates_on_device(
                local_indices,
                local_scores,
                threads=threads,
                block_limit=block_limit,
            )

    def prepare_graph(self, *, threads: int = 512) -> None:
        """Compile the exact graph launcher before capture begins."""
        with self._launch_gate():
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "prepare_graph() must be called before CUDA graph capture"
                )
            _validate_launch_config(
                threads=int(threads),
                block_limit=1,
                world_size=self.world_size,
            )
            with _device_guard(self.device):
                self._bind_stream()
                from ._dcp_topk_cute import prepare_topk_stage

                prepare_topk_stage(
                    self.world_size,
                    self.rank,
                    self.topk,
                    int(threads),
                )

    @contextmanager
    def capture(self, *, threads: int = 512):
        """Own one serialized graph capture without adding graph nodes.

        Enter this context before ``torch.cuda.graph`` and call
        :meth:`register_graph` after capture. Graphs captured from this
        instance share one epoch and must never replay concurrently through
        any path other than :meth:`replay`. Close cannot run while a capture
        context is active.
        """
        # Atomically reserve the capture token under the lifecycle lock.
        # This prevents close from transitioning to CLOSING between the
        # OPEN check and the depth increment.
        with self._lifecycle_lock:
            if self._lifecycle_state != _ChannelLifecycle.OPEN:
                raise RuntimeError(
                    f"{type(self).__name__} is "
                    f"{self._lifecycle_state.name.lower()}; cannot capture"
                )
            if self._capture_context_depth:
                raise RuntimeError(
                    "overlapping DCP top-k capture contexts are not allowed"
                )
            self._capture_context_depth = 1
        try:
            self.prepare_graph(threads=threads)
        except Exception:
            with self._lifecycle_lock:
                self._capture_context_depth = 0
                self._lifecycle_cond.notify_all()
            raise
        try:
            yield self
        finally:
            with self._lifecycle_lock:
                self._capture_context_depth = 0
                self._lifecycle_cond.notify_all()

    def register_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        """Register a captured CUDA graph for gated replay.

        The channel tracks registered graphs so close can wait for active
        replays and prevent use-after-free of captured staging pointers.
        """
        with self._lifecycle_lock:
            if self._lifecycle_state != _ChannelLifecycle.OPEN:
                raise RuntimeError(
                    f"{type(self).__name__} is "
                    f"{self._lifecycle_state.name.lower()}; cannot register graph"
                )
            self._registered_graphs.add(id(graph))

    def replay(self, graph: torch.cuda.CUDAGraph) -> None:
        """Replay a registered CUDA graph under the lifecycle gate.

        Close cannot quiesce or free while a replay is in flight.
        """
        with self._replay_gate():
            if id(graph) not in self._registered_graphs:
                raise RuntimeError(
                    "graph was not registered with this channel; call "
                    "register_graph() after capture"
                )
            graph.replay()

    def _stage_candidates_on_device(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        threads: int,
        block_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_indices.device != self.device or local_scores.device != self.device:
            raise ValueError("inputs must be on the runtime device")
        if local_indices.dtype != torch.int32:
            raise ValueError("local_indices must be int32")
        if local_scores.dtype != torch.float32:
            raise ValueError("local_scores must be float32")
        if local_indices.shape != local_scores.shape or local_indices.ndim != 2:
            raise ValueError("inputs must have matching [rows, topk] shapes")
        rows, topk = (int(value) for value in local_indices.shape)
        if rows <= 0 or rows > self.max_rows or rows % self.world_size != 0:
            raise ValueError(
                f"rows must be in (0, {self.max_rows}] and divisible by world_size"
            )
        if topk != self.topk:
            raise ValueError(f"topk {topk} does not match configured topk {self.topk}")
        if not local_indices.is_contiguous() or not local_scores.is_contiguous():
            raise ValueError("inputs must be contiguous")
        _validate_launch_config(
            threads=int(threads),
            block_limit=int(block_limit),
            world_size=self.world_size,
        )
        self._bind_stream()
        owner_packs = (rows // self.world_size) * (self.topk // 4)
        blocks = max(1, min(int(block_limit), (owner_packs + threads - 1) // threads))
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            if self._capture_context_depth <= 0:
                raise RuntimeError(
                    "DCP top-k CUDA graph capture requires an active "
                    "owner.capture() context"
                )
            from ._dcp_topk_cute import is_topk_stage_prepared

            if not is_topk_stage_prepared(
                self.world_size,
                self.rank,
                self.topk,
                int(threads),
            ):
                raise RuntimeError(
                    "cold DCP top-k CUDA graph capture is not allowed; "
                    "enter owner.capture() before torch.cuda.graph()"
                )
        if capturing and self._graph_slot is None:
            self._graph_slot = self._next_slot
        if self._graph_slot is not None:
            slot = self._graph_slot
            wait_for_prior_consumer = True
        else:
            slot = self._next_slot
            self._next_slot ^= 1
            wait_for_prior_consumer = False
        self._launch_stage(
            local_indices,
            local_scores,
            slot=slot,
            rows=rows,
            threads=int(threads),
            blocks=blocks,
            wait_for_prior_consumer=wait_for_prior_consumer,
        )
        candidate_views = self._candidate_views[slot] if self._candidate_views else ()
        if not candidate_views:
            raise RuntimeError("DCP top-k candidate views require a CUDA device")
        owner_rows = rows // self.world_size
        candidate_indices = candidate_views[0][:owner_rows]
        candidate_scores = candidate_views[1][:owner_rows]
        expected = (rows // self.world_size, self.world_size * self.topk)
        _validate_candidate_view(
            candidate_indices,
            expected=expected,
            dtype=torch.int32,
            device=self.device,
            label="index",
        )
        _validate_candidate_view(
            candidate_scores,
            expected=expected,
            dtype=torch.float32,
            device=self.device,
            label="score",
        )
        return candidate_indices, candidate_scores

    def _launch_stage(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
        *,
        slot: int,
        rows: int,
        threads: int,
        blocks: int,
        wait_for_prior_consumer: bool,
    ) -> None:
        from ._dcp_topk_cute import stage_owner_candidates

        with torch.cuda.device(self.device):
            stage_owner_candidates(
                world_size=self.world_size,
                rank=self.rank,
                topk=self.topk,
                threads=threads,
                local_indices_ptr=local_indices.data_ptr(),
                local_scores_ptr=local_scores.data_ptr(),
                candidate_ptrs=self._staging_ptrs[slot],
                signal_ptrs=self._signal_ptrs,
                rows=rows,
                candidate_plane_elems=self._candidate_plane_elems,
                blocks=blocks,
                wait_for_prior_consumer=wait_for_prior_consumer,
            )


def _validate_candidate_view(
    tensor: torch.Tensor,
    *,
    expected: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> None:
    if (
        tuple(tensor.shape) != expected
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(f"channel returned an invalid candidate {label} view")


def _release_failed_allocations(
    owned: Sequence[_OwnedSharedBuffer],
    ipc: CudaRTLibrary,
) -> None:
    for shared in owned:
        for ptr in shared.remote_ptrs:
            with suppress(Exception):
                ipc.cudaIpcCloseMemHandle(ptr)
        with suppress(Exception):
            ipc.cudaFree(shared.local_ptr)
