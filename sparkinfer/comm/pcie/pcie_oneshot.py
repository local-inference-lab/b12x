"""PCIe oneshot allreduce runtime with optional crossover autotuning."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary


logger = logging.getLogger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_MAX_SIZE = 8 * 1024 * 1024
DEFAULT_RANK_DATA_BYTES = 8 * 1024 * 1024
AUTOTUNE_CEILING = 1 * 1024 * 1024
AUTOTUNE_FINE_STEP = 8 * 1024
IPC_SLAB_ALIGNMENT = 256
ONESHOT_REQUIRED_SMS = 36


def parse_pcie_oneshot_max_size(value: str | int | None) -> Optional[int]:
    """Parse a byte-size string, or return ``None`` for ``auto``."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if value.lower() == "auto":
        return None
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _current_stream_key(
    device: torch.device | int | str, stream: object = None
) -> Optional[int]:
    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        return None
    if stream is None:
        stream = torch.cuda.current_stream(device_obj)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(stream)


def _push_mode_enabled() -> bool:
    """Match the extension's SPARKINFER_PCIE_ONESHOT_PUSH transport toggle.

    Push transport writes each rank's input into a per-source shard of every
    peer's eager slot, so each slot must hold world_size * max_size bytes.
    """

    return os.getenv("SPARKINFER_PCIE_ONESHOT_PUSH", "0") not in ("", "0")


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    if inp.is_contiguous():
        return True
    storage = inp.untyped_storage()
    return (
        storage.nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _resolve_exchange_group(
    exchange_group: Optional[ProcessGroup],
    process_group: Optional[ProcessGroup],
) -> Optional[ProcessGroup]:
    if (
        exchange_group is not None
        and process_group is not None
        and exchange_group is not process_group
    ):
        raise ValueError("pass only one of exchange_group or process_group")
    return exchange_group if exchange_group is not None else process_group


def _group_ranks(group: ProcessGroup) -> list[int]:
    world_size = dist.get_world_size(group=group)
    if hasattr(dist, "get_process_group_ranks"):
        # Process-group rank is an index into the order returned here.  Sorting
        # silently changes that mapping for non-monotonic groups and makes the
        # handle in slot N belong to a different peer than group rank N.
        ranks = list(dist.get_process_group_ranks(group=group))
        if len(ranks) != world_size:
            raise RuntimeError("process-group rank list does not match world size")
        return ranks
    if hasattr(dist, "get_global_rank"):
        return [
            dist.get_global_rank(group, group_rank) for group_rank in range(world_size)
        ]
    return list(range(world_size))


def _object_broadcast_device(group: ProcessGroup) -> torch.device | str:
    try:
        try:
            backend = dist.get_backend(group=group)
        except TypeError:
            backend = dist.get_backend(group)
    except Exception as exc:
        raise RuntimeError(
            "PCIe oneshot IPC exchange requires a CUDA/NCCL process group"
        ) from exc
    backend_name = str(backend).lower()
    if "nccl" not in backend_name:
        raise RuntimeError(
            f"PCIe oneshot IPC exchange requires an NCCL process group, got {backend}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PCIe oneshot IPC exchange requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _broadcast_gather_object(local_object: object, group: ProcessGroup) -> list[object]:
    # `broadcast_object_list` is more robust than `all_gather_object` for
    # the host-side exchange that happens around IPC setup and graph capture.
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_objects: list[list[object | None]] = [[None] for _ in range(world_size)]
    all_objects[rank][0] = local_object
    device = _object_broadcast_device(group)
    for index, src_rank in enumerate(_group_ranks(group)):
        dist.broadcast_object_list(
            all_objects[index], src=src_rank, group=group, device=device
        )
    return [entry[0] for entry in all_objects]


@dataclass(frozen=True)
class _OwnedSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


def _format_setup_error(error: BaseException | None) -> tuple[str, ...]:
    if error is None:
        return ()
    return (f"{type(error).__name__}: {error}",)


def _exchange_setup_failures(
    local_error: BaseException | None,
    *,
    exchange_group: ProcessGroup,
    phase: str,
) -> tuple[tuple[str, ...], ...]:
    """Collect a setup verdict without allowing one rank to return early."""

    try:
        gathered = _broadcast_gather_object(
            _format_setup_error(local_error), exchange_group
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to exchange peer {phase} status; CUDA IPC exports were retained"
        ) from exc

    statuses: list[tuple[str, ...]] = []
    for group_rank, status in enumerate(gathered):
        if not isinstance(status, (tuple, list)):
            raise RuntimeError(
                f"invalid peer {phase} status from group rank {group_rank}; "
                "CUDA IPC exports were retained"
            )
        statuses.append(tuple(str(item) for item in status))
    return tuple(statuses)


def _require_full_grid_residency(
    *,
    owner: str,
    required_sms: int,
    device: torch.device,
    exchange_group: Optional[ProcessGroup],
) -> None:
    """Reject devices where a peer-waiting worker grid may not all reside.

    The PCIe worker kernels use ``__launch_bounds__(512, 1)`` and zero dynamic
    shared memory, so every visible SM can host at least one worker CTA.  A
    device with at least the extension's maximum block count can therefore
    make the complete grid resident regardless of block scheduling order.  We
    intentionally reject smaller/MIG-like slices instead of assuming CUDA
    schedules matching block indices in the same order on every rank.
    """

    local_error: BaseException | None = None
    visible_sms = 0
    try:
        visible_sms = int(
            torch.cuda.get_device_properties(device).multi_processor_count
        )
        test_visible_sms = int(
            os.getenv("SPARKINFER_PCIE_TEST_VISIBLE_SM_COUNT", "0") or "0"
        )
        if test_visible_sms > 0:
            visible_sms = min(visible_sms, test_visible_sms)
        if visible_sms < required_sms:
            raise RuntimeError(
                f"{owner} requires at least {required_sms} visible SMs so every "
                "peer-waiting CTA can be resident; this device/MIG slice exposes "
                f"{visible_sms}"
            )
    except Exception as exc:
        local_error = exc

    if exchange_group is None:
        if local_error is not None:
            raise local_error
        return

    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase=f"{owner} resident-grid capability",
    )
    if any(statuses):
        raise RuntimeError(
            _setup_failure_message(
                owner,
                "resident-grid capability",
                statuses,
                exports_retained=False,
            )
        ) from local_error


def _setup_failure_message(
    owner: str,
    phase: str,
    statuses: Sequence[Sequence[str]],
    *,
    exports_retained: bool,
) -> str:
    details = [
        f"group rank {group_rank}: " + " | ".join(failures)
        for group_rank, failures in enumerate(statuses)
        if failures
    ]
    retention = "; CUDA IPC exports were retained" if exports_retained else ""
    return f"{owner} {phase} failed{retention}: " + "; ".join(details)


def _abort_collective_ipc_setup(
    *,
    owner: str,
    setup_phase: str,
    setup_statuses: Sequence[Sequence[str]],
    exchange_group: ProcessGroup,
    ipc: CudaRTLibrary,
    local_ptr: int | None,
    remote_ptrs: Sequence[int],
    local_error: BaseException | None,
    local_cleanup: Optional[Callable[[], None]] = None,
) -> None:
    """Undo a failed collective IPC setup without freeing a mapped export.

    Every rank calls this function.  Each successfully opened import is closed
    exactly once.  Export ownership is released only after every group rank
    reports that all of its imports were unmapped; otherwise the raw CUDA
    allocation is deliberately retained until context teardown.
    """

    unmap_failures: list[str] = []
    if local_cleanup is not None:
        try:
            local_cleanup()
        except Exception as exc:
            unmap_failures.append(f"native runtime: {type(exc).__name__}: {exc}")
    for ptr in remote_ptrs:
        try:
            ipc.cudaIpcCloseMemHandle(ptr)
        except Exception as exc:
            unmap_failures.append(f"CUDA IPC import {ptr}: {type(exc).__name__}: {exc}")

    try:
        unmap_statuses = _exchange_setup_failures(
            RuntimeError(" | ".join(unmap_failures)) if unmap_failures else None,
            exchange_group=exchange_group,
            phase=f"{setup_phase} rollback unmap",
        )
    except Exception as exchange_error:
        error = RuntimeError(
            _setup_failure_message(
                owner,
                setup_phase,
                setup_statuses,
                exports_retained=True,
            )
        )
        raise error from (local_error or exchange_error)

    if any(unmap_statuses):
        error = RuntimeError(
            _setup_failure_message(
                owner,
                f"{setup_phase} rollback unmap",
                unmap_statuses,
                exports_retained=True,
            )
        )
        raise error from local_error

    free_error: BaseException | None = None
    if local_ptr is not None:
        try:
            ipc.cudaFree(local_ptr)
        except Exception as exc:
            free_error = exc
    free_statuses = _exchange_setup_failures(
        free_error,
        exchange_group=exchange_group,
        phase=f"{setup_phase} rollback export free",
    )
    if any(free_statuses):
        error = RuntimeError(
            _setup_failure_message(
                owner,
                f"{setup_phase} rollback export free",
                free_statuses,
                exports_retained=False,
            )
        )
        raise error from (local_error or free_error)

    error = RuntimeError(
        _setup_failure_message(
            owner,
            setup_phase,
            setup_statuses,
            exports_retained=False,
        )
    )
    raise error from local_error


def _finish_collective_runtime_setup(
    *,
    owner: str,
    exchange_group: ProcessGroup,
    ipc: CudaRTLibrary,
    shared: _OwnedSharedBuffer,
    local_error: BaseException | None,
    local_cleanup: Optional[Callable[[], None]] = None,
) -> None:
    """Require every rank to construct its native runtime before returning."""

    statuses = _exchange_setup_failures(
        local_error,
        exchange_group=exchange_group,
        phase=f"{owner} native initialization",
    )
    if any(statuses):
        _abort_collective_ipc_setup(
            owner=owner,
            setup_phase="native initialization",
            setup_statuses=statuses,
            exchange_group=exchange_group,
            ipc=ipc,
            local_ptr=shared.local_ptr,
            remote_ptrs=shared.remote_ptrs,
            local_error=local_error,
            local_cleanup=local_cleanup,
        )


def _coordinated_close_channels(
    channels: Sequence[object],
    *,
    exchange_group: Optional[ProcessGroup],
    device: torch.device,
) -> None:
    """Strictly release exports only after every peer reports successful unmap."""
    unique_channels = tuple(dict.fromkeys(channels))
    if not unique_channels:
        return

    if exchange_group is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier(group=exchange_group)

    # Older DCP channels still expose the legacy best-effort protocol. Keep
    # their existing ordering until they receive their own lifecycle migration;
    # oneshot and twoshot opt into the strict peer-status protocol below.
    uses_strict_protocol = any(
        hasattr(channel, "_close_ipc_imports_strict") for channel in unique_channels
    )
    if not uses_strict_protocol:
        for channel in unique_channels:
            channel._close_ipc_imports()
        if exchange_group is not None:
            dist.barrier(group=exchange_group)
        for channel in unique_channels:
            channel._free_ipc_exports()
        if exchange_group is not None:
            dist.barrier(group=exchange_group)
        return

    unmap_errors = _run_close_phase(
        unique_channels,
        strict_method="_close_ipc_imports_strict",
        legacy_method="_close_ipc_imports",
    )
    unmap_status = _exchange_close_failures(
        tuple(_format_close_error(index, error) for index, error in unmap_errors),
        exchange_group=exchange_group,
        phase="IPC unmap",
    )
    if any(unmap_status):
        _raise_coordinated_close_error(
            "IPC unmap",
            unmap_status,
            local_errors=unmap_errors,
            exports_retained=True,
        )

    if exchange_group is not None:
        dist.barrier(group=exchange_group)

    free_errors = _run_close_phase(
        unique_channels,
        strict_method="_free_ipc_exports_strict",
        legacy_method="_free_ipc_exports",
    )
    free_status = _exchange_close_failures(
        tuple(_format_close_error(index, error) for index, error in free_errors),
        exchange_group=exchange_group,
        phase="IPC export free",
    )
    if any(free_status):
        _raise_coordinated_close_error(
            "IPC export free",
            free_status,
            local_errors=free_errors,
            exports_retained=False,
        )

    if exchange_group is not None:
        dist.barrier(group=exchange_group)
    for channel in unique_channels:
        if hasattr(channel, "_close_ipc_imports_strict"):
            channel._coordinated_close_complete = True


def _run_close_phase(
    channels: Sequence[object],
    *,
    strict_method: str,
    legacy_method: str,
) -> list[tuple[int, Exception]]:
    errors: list[tuple[int, Exception]] = []
    for index, channel in enumerate(channels):
        method = getattr(channel, strict_method, None)
        if method is None:
            method = getattr(channel, legacy_method)
        try:
            method()
        except Exception as exc:
            errors.append((index, exc))
    return errors


def _format_close_error(index: int, error: Exception) -> str:
    return f"channel {index}: {type(error).__name__}: {error}"


def _exchange_close_failures(
    local_failures: tuple[str, ...],
    *,
    exchange_group: Optional[ProcessGroup],
    phase: str,
) -> tuple[tuple[str, ...], ...]:
    if exchange_group is None:
        return (local_failures,)
    try:
        gathered = _broadcast_gather_object(local_failures, exchange_group)
    except Exception as exc:
        raise RuntimeError(
            f"failed to exchange peer {phase} status; exported IPC allocations "
            "were not freed"
        ) from exc

    statuses: list[tuple[str, ...]] = []
    for rank_index, status in enumerate(gathered):
        if not isinstance(status, (tuple, list)):
            raise RuntimeError(
                f"invalid peer {phase} status from group rank {rank_index}; "
                "exported IPC allocations were not freed"
            )
        statuses.append(tuple(str(item) for item in status))
    return tuple(statuses)


def _raise_coordinated_close_error(
    phase: str,
    statuses: Sequence[Sequence[str]],
    *,
    local_errors: Sequence[tuple[int, Exception]],
    exports_retained: bool,
) -> None:
    peer_details = []
    for rank_index, failures in enumerate(statuses):
        if failures:
            peer_details.append(f"group rank {rank_index}: " + " | ".join(failures))
    retention = "; exported IPC allocations were not freed" if exports_retained else ""
    error = RuntimeError(
        f"coordinated PCIe close failed during {phase}{retention}: "
        + "; ".join(peer_details)
    )
    if local_errors:
        raise error from local_errors[0][1]
    raise error


def _raise_local_cleanup_errors(
    owner: str,
    phase: str,
    failures: Sequence[tuple[str, Exception]],
) -> None:
    details = "; ".join(
        f"{resource}: {type(error).__name__}: {error}" for resource, error in failures
    )
    error = RuntimeError(f"{owner} {phase} failed: {details}")
    raise error from failures[0][1]


@dataclass(frozen=True)
class _ChannelSharedBuffers:
    owned_buffer: _OwnedSharedBuffer
    signal_ptrs: tuple[int, ...]
    eager0_ptrs: tuple[int, ...]
    eager1_ptrs: tuple[int, ...]


@dataclass(frozen=True)
class _BenchmarkResult:
    size_bytes: int
    custom_us: float
    nccl_us: float
    winner: str


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_oneshot.cu")
    verbose = os.getenv("SPARKINFER_PCIE_ONESHOT_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sparkinfer_pcie_oneshot_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _compute_crossover_size(
    benchmark: Callable[[int], tuple[float, float]],
    *,
    ceiling_bytes: int = AUTOTUNE_CEILING,
    fine_step_bytes: int = AUTOTUNE_FINE_STEP,
) -> tuple[int, list[_BenchmarkResult]]:
    coarse_sizes = []
    current = 1024
    while current <= ceiling_bytes:
        coarse_sizes.append(current)
        current *= 2

    results: list[_BenchmarkResult] = []
    seen_sizes: set[int] = set()
    first_nccl_win: Optional[int] = None
    last_custom_win = 0

    def record(size_bytes: int) -> None:
        nonlocal first_nccl_win, last_custom_win
        custom_us, nccl_us = benchmark(size_bytes)
        winner = "custom" if custom_us < nccl_us else "NCCL"
        results.append(_BenchmarkResult(size_bytes, custom_us, nccl_us, winner))
        seen_sizes.add(size_bytes)
        if winner == "custom":
            last_custom_win = max(last_custom_win, size_bytes)
        elif first_nccl_win is None:
            first_nccl_win = size_bytes

    for size_bytes in coarse_sizes:
        record(size_bytes)

    if last_custom_win > 0 and first_nccl_win is not None:
        fine_start = last_custom_win
        fine_end = min(first_nccl_win, last_custom_win * 4)
        fine_size = fine_start + fine_step_bytes
        while fine_size < fine_end:
            aligned = (fine_size // 16) * 16
            if aligned not in seen_sizes:
                record(aligned)
            fine_size += fine_step_bytes

    results.sort(key=lambda item: item.size_bytes)
    crossover = 1024
    for result in results:
        if result.winner == "custom":
            crossover = result.size_bytes
    return crossover, results


class PCIeOneshotAllReduce:
    """Standalone unfused PCIe oneshot allreduce runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ):
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(signal_ptrs) != world_size:
            raise ValueError("signal_ptrs must match world size")
        if (eager_buffer_ptrs0 is None) != (eager_buffer_ptrs1 is None):
            raise ValueError("eager buffers must be provided as a pair")
        if eager_buffer_ptrs0 is not None and len(eager_buffer_ptrs0) != world_size:
            raise ValueError("eager_buffer_ptrs0 must match world size")
        if eager_buffer_ptrs1 is not None and len(eager_buffer_ptrs1) != world_size:
            raise ValueError("eager_buffer_ptrs1 must match world size")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.exchange_group = _resolve_exchange_group(exchange_group, process_group)
        # Compatibility alias for older callers that still refer to this as
        # `process_group`.
        self.process_group = self.exchange_group
        self.max_size = int(max_size)
        self._ipc = ipc
        if self._ipc is None and (ext_module is None or owned_buffers):
            self._ipc = CudaRTLibrary()
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._owned_buffers = list(owned_buffers or [])
        self._registered_input_ptrs: dict[int, tuple[int, ...]] = {}
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._native_ipc_import_close_failed = False
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        self._ext = ext_module or _load_extension()

        if ext_module is None and self.device.type != "cuda":
            raise ValueError("PCIe oneshot allreduce requires a CUDA device")

        self.rank_data = torch.empty(
            rank_data_bytes, dtype=torch.uint8, device=self.device
        )
        self._ptr = self._ext.init_custom_ar(
            list(self._signal_ptrs), self.rank_data, self.rank
        )

        self._eager_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
        if eager_buffer_ptrs0 is not None and eager_buffer_ptrs1 is not None:
            self._eager_ptrs = (
                tuple(int(ptr) for ptr in eager_buffer_ptrs0),
                tuple(int(ptr) for ptr in eager_buffer_ptrs1),
            )
            self._ext.register_pcie_buffers(
                self._ptr,
                list(self._eager_ptrs[0]),
                list(self._eager_ptrs[1]),
            )

    @classmethod
    def from_ipc(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        device_obj = _normalize_device(device)
        resolved_group = _resolve_exchange_group(exchange_group, process_group)
        if device_obj.type == "cuda":
            _require_full_grid_residency(
                owner="PCIe oneshot",
                required_sms=ONESHOT_REQUIRED_SMS,
                device=device_obj,
                exchange_group=resolved_group,
            )
        return cls(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            signal_ptrs=signal_ptrs,
            eager_buffer_ptrs0=eager_buffer_ptrs0,
            eager_buffer_ptrs1=eager_buffer_ptrs1,
            exchange_group=exchange_group,
            process_group=process_group,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")

        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe oneshot requires a CUDA device")

        _require_full_grid_residency(
            owner="PCIe oneshot",
            required_sms=ONESHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )
        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        ext = ext_module or _load_extension()

        channel_buffers = cls._allocate_eager_channel_buffers(
            exchange_group,
            signal_bytes=ext.meta_size(),
            eager_buffer_bytes=eager_buffer_bytes,
            ipc=ipc,
        )
        owned_buffers = [channel_buffers.owned_buffer]
        runtime = object.__new__(cls)
        init_error: BaseException | None = None
        try:
            cls.__init__(
                runtime,
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=channel_buffers.signal_ptrs,
                eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
                eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
                exchange_group=exchange_group,
                ipc=ipc,
                owned_buffers=owned_buffers,
                max_size=max_size,
                rank_data_bytes=rank_data_bytes,
                ext_module=ext,
                stream_affine=stream_affine,
            )
        except Exception as exc:
            init_error = exc

        def abort_native_runtime() -> None:
            pointer = getattr(runtime, "_ptr", 0)
            if pointer:
                ext.dispose(pointer)
                runtime._ptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe oneshot",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=channel_buffers.owned_buffer,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
        )
        return runtime

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeOneshotAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            stream_affine=stream_affine,
        )

    @staticmethod
    def _allocate_shared_buffer(
        exchange_group: ProcessGroup,
        size_in_bytes: int,
        *,
        zero_fill: bool,
        ipc: CudaRTLibrary,
    ) -> _OwnedSharedBuffer:
        local_ptr: int | None = None
        local_handle: bytes | None = None
        prepare_error: BaseException | None = None
        try:
            local_ptr = ipc.cudaMalloc(size_in_bytes)
            if zero_fill:
                ipc.cudaMemset(local_ptr, 0, size_in_bytes)
            local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
        except Exception as exc:
            prepare_error = exc

        # A rank that cannot allocate or publish its export must not leave its
        # peers blocked in the handle exchange.  No imports exist yet, so
        # successful ranks can safely release their local allocation.
        prepare_statuses = _exchange_setup_failures(
            prepare_error,
            exchange_group=exchange_group,
            phase="CUDA IPC export preparation",
        )
        if any(prepare_statuses):
            free_error: BaseException | None = None
            if local_ptr is not None:
                try:
                    ipc.cudaFree(local_ptr)
                except Exception as exc:
                    free_error = exc
            free_statuses = _exchange_setup_failures(
                free_error,
                exchange_group=exchange_group,
                phase="CUDA IPC export preparation rollback",
            )
            if any(free_statuses):
                raise RuntimeError(
                    _setup_failure_message(
                        "PCIe shared buffer",
                        "export preparation rollback",
                        free_statuses,
                        exports_retained=False,
                    )
                ) from (prepare_error or free_error)
            raise RuntimeError(
                _setup_failure_message(
                    "PCIe shared buffer",
                    "export preparation",
                    prepare_statuses,
                    exports_retained=False,
                )
            ) from prepare_error

        assert local_ptr is not None
        assert local_handle is not None
        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        try:
            world_size = dist.get_world_size(group=exchange_group)
            rank = dist.get_rank(group=exchange_group)
            handles = _broadcast_gather_object(local_handle, exchange_group)
        except Exception:
            # No rank has opened an import before handle exchange completes.
            # Retain the allocation if collective progress is uncertain.
            raise

        open_error: BaseException | None = None
        for idx, handle in enumerate(handles):
            if idx == rank:
                peer_ptrs.append(local_ptr)
                continue
            try:
                remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
            except Exception as exc:
                if open_error is None:
                    open_error = RuntimeError(
                        f"failed to open CUDA IPC handle for peer group rank {idx}"
                    )
                    open_error.__cause__ = exc
                # Preserve group-rank indexing while every rank completes the
                # same open loop and reaches the collective verdict.
                peer_ptrs.append(0)
            else:
                peer_ptrs.append(remote_ptr)
                remote_ptrs.append(remote_ptr)

        if len(handles) != world_size or len(peer_ptrs) != world_size:
            open_error = open_error or RuntimeError(
                "failed to gather CUDA IPC handles for every group rank"
            )
        open_statuses = _exchange_setup_failures(
            open_error,
            exchange_group=exchange_group,
            phase="CUDA IPC import open",
        )
        if any(open_statuses):
            _abort_collective_ipc_setup(
                owner="PCIe shared buffer",
                setup_phase="CUDA IPC import open",
                setup_statuses=open_statuses,
                exchange_group=exchange_group,
                ipc=ipc,
                local_ptr=local_ptr,
                remote_ptrs=remote_ptrs,
                local_error=open_error,
            )

        return _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )

    @classmethod
    def _allocate_eager_channel_buffers(
        cls,
        exchange_group: ProcessGroup,
        *,
        signal_bytes: int,
        eager_buffer_bytes: int,
        ipc: CudaRTLibrary,
    ) -> _ChannelSharedBuffers:
        signal_bytes = int(signal_bytes)
        eager_buffer_bytes = int(eager_buffer_bytes)
        if signal_bytes <= 0:
            raise ValueError("signal_bytes must be positive")
        if eager_buffer_bytes <= 0:
            raise ValueError("eager_buffer_bytes must be positive")
        if _push_mode_enabled():
            eager_buffer_bytes *= dist.get_world_size(group=exchange_group)

        signal_offset = 0
        eager0_offset = _align_up(signal_bytes, IPC_SLAB_ALIGNMENT)
        eager1_offset = eager0_offset + _align_up(
            eager_buffer_bytes, IPC_SLAB_ALIGNMENT
        )
        slab_bytes = eager1_offset + eager_buffer_bytes
        slab = cls._allocate_shared_buffer(
            exchange_group,
            slab_bytes,
            # Clear signals before publishing the IPC handle so a peer cannot
            # post an arrival that a later local memset would erase.
            zero_fill=True,
            ipc=ipc,
        )
        return _ChannelSharedBuffers(
            owned_buffer=slab,
            signal_ptrs=tuple(ptr + signal_offset for ptr in slab.peer_ptrs),
            eager0_ptrs=tuple(ptr + eager0_offset for ptr in slab.peer_ptrs),
            eager1_ptrs=tuple(ptr + eager1_offset for ptr in slab.peer_ptrs),
        )

    @property
    def signal_ptrs(self) -> tuple[int, ...]:
        return self._signal_ptrs

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine:
            return
        if stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe oneshot allreduce channels are stream-affine; "
                "create or use a separate channel for each CUDA stream"
            )

    def _check_stream(self, stream: object = None) -> None:
        if stream is None and _is_current_stream_capturing(self.device):
            # During CUDA graph capture the current stream is a torch-owned,
            # ephemeral capture stream (true for piecewise/inductor graphs used
            # by MTP and spec-decode). Stream affinity does not apply: the
            # captured kernel replays on the caller's stream, not this one, so
            # skip the affinity guard instead of rejecting the capture stream.
            return
        self._bind_stream_key(_current_stream_key(self.device, stream))

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self.max_size:
            return False
        if inp_bytes % 16 != 0:
            return False
        return _is_weak_contiguous(inp)

    def register_buffer(self, peer_input_ptrs: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(peer_input_ptrs) != self.world_size:
            raise ValueError("peer_input_ptrs must match world size")
        ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
        local_ptr = ptrs[self.rank]
        existing = self._registered_input_ptrs.get(local_ptr)
        if existing is not None:
            if existing != ptrs:
                raise ValueError(
                    "input pointer is already registered with different peer_input_ptrs"
                )
            return
        self._ext.register_buffer(self._ptr, list(ptrs))
        self._registered_input_ptrs[local_ptr] = ptrs

    def _prepare_input(
        self,
        inp: torch.Tensor,
        peer_input_ptrs: Optional[Sequence[int]],
    ) -> None:
        local_ptr = int(inp.data_ptr())
        if peer_input_ptrs is not None:
            if len(peer_input_ptrs) != self.world_size:
                raise ValueError("peer_input_ptrs must match world size")
            ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
            if ptrs[self.rank] != local_ptr:
                raise ValueError("peer_input_ptrs[self.rank] must match inp.data_ptr()")
            self.register_buffer(ptrs)
        elif self._eager_ptrs is None and local_ptr not in self._registered_input_ptrs:
            raise ValueError(
                "peer_input_ptrs are required unless eager IPC buffers are configured "
                "or this input was already registered"
            )

    def get_graph_buffer_ipc_meta(self) -> tuple[list[int], list[int]]:
        if self._closed:
            raise RuntimeError("runtime is closed")
        handle, offsets = self._ext.get_graph_buffer_ipc_meta(self._ptr)
        return list(handle), list(offsets)

    def register_graph_buffers_from_ranks(
        self,
        handles: Sequence[Sequence[int]],
        offsets: Sequence[Sequence[int]],
    ) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(handles) != self.world_size:
            raise ValueError("handles must match world size")
        if len(offsets) != self.world_size:
            raise ValueError("offsets must match world size")
        self._ext.register_graph_buffers(
            self._ptr,
            [list(map(int, handle)) for handle in handles],
            [list(map(int, rank_offsets)) for rank_offsets in offsets],
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )

        if out is None:
            out = torch.empty_like(inp)
        if out.device != inp.device:
            raise ValueError("output tensor must be on the same device as the input")
        if out.shape != inp.shape or out.dtype != inp.dtype:
            raise ValueError("output tensor must match input shape and dtype")
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")

        self._prepare_input(inp, peer_input_ptrs)

        self._ext.all_reduce(self._ptr, inp, out, 0, 0)
        return out

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``inp``, add ``residual``, and apply RMSNorm."""

        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(
                f"input device {inp.device} does not match runtime device {self.device}"
            )
        self._check_stream()
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity "
                f"requirements (shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if inp.ndim == 0:
            raise ValueError("input must have at least one dimension")
        hidden_size = inp.shape[-1]
        if hidden_size * inp.element_size() % 16 != 0:
            raise ValueError(
                "the last input dimension must occupy a multiple of 16 bytes"
            )
        if residual.device != inp.device:
            raise ValueError("residual tensor must be on the same device as the input")
        if residual.shape != inp.shape or residual.dtype != inp.dtype:
            raise ValueError("residual tensor must match input shape and dtype")
        if not _is_weak_contiguous(residual):
            raise ValueError("residual tensor must be weak-contiguous")
        if weight.device != inp.device:
            raise ValueError("weight tensor must be on the same device as the input")
        if weight.shape != (hidden_size,) or weight.dtype != inp.dtype:
            raise ValueError(
                "weight tensor must match the input dtype and last dimension"
            )
        if not weight.is_contiguous():
            raise ValueError("weight tensor must be contiguous")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        for name, tensor in (("output", out), ("residual output", residual_out)):
            if tensor.device != inp.device:
                raise ValueError(
                    f"{name} tensor must be on the same device as the input"
                )
            if tensor.shape != inp.shape or tensor.dtype != inp.dtype:
                raise ValueError(f"{name} tensor must match input shape and dtype")
            if not _is_weak_contiguous(tensor):
                raise ValueError(f"{name} tensor must be weak-contiguous")
        if out.data_ptr() == residual_out.data_ptr():
            raise ValueError("output and residual output must not alias")

        self._prepare_input(inp, peer_input_ptrs)
        self._ext.all_reduce_fused_add_rms_norm(
            self._ptr,
            inp,
            residual,
            weight,
            out,
            residual_out,
            float(epsilon),
            0,
            0,
        )
        return out, residual_out

    @contextmanager
    def capture(self, stream: object = None):
        if self.exchange_group is None and self._eager_ptrs is None:
            raise ValueError(
                "exchange_group is required for CUDA graph capture registration"
            )
        self._check_stream(stream)
        try:
            yield
        finally:
            if self.exchange_group is not None and self._eager_ptrs is None:
                self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required to register graph buffers")
        local_meta = self.get_graph_buffer_ipc_meta()
        all_meta = _broadcast_gather_object(local_meta, self.exchange_group)
        num_buffers = [len(entry[1]) for entry in all_meta]
        if any(count != num_buffers[0] for count in num_buffers):
            raise RuntimeError(
                "graph capture registered a different number of buffers across ranks"
            )
        if num_buffers[0] == 0:
            return
        self.register_graph_buffers_from_ranks(
            [entry[0] for entry in all_meta],
            [entry[1] for entry in all_meta],
        )

    def _bench_graph_latency(
        self,
        size_bytes: int,
        nccl_group: ProcessGroup,
        stream: torch.cuda.Stream,
        warmup: int,
        iters: int,
    ) -> tuple[float, float]:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required for graph-based autotuning")
        self._check_stream(stream)

        numel = size_bytes // torch.tensor([], dtype=torch.bfloat16).element_size()
        device = self.device

        def run_custom() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
                graph_out = torch.zeros_like(graph_inp)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._ext.all_reduce(self._ptr, graph_inp, graph_out, 0, 0)
            self.register_graph_buffers()
            dist.barrier(group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        def run_nccl() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dist.all_reduce(graph_inp, group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        custom_runs = sorted(run_custom() for _ in range(3))
        nccl_runs = sorted(run_nccl() for _ in range(3))
        # Reduce timings across ranks so every rank reaches the same
        # crossover verdicts; divergent local verdicts would desynchronize
        # the sweep's collective sequence and deadlock.
        stats = torch.tensor(
            [custom_runs[1], nccl_runs[1]], dtype=torch.float64, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=nccl_group)
        return float(stats[0].item()), float(stats[1].item())

    def find_crossover_size(
        self,
        nccl_group: ProcessGroup,
        *,
        ceiling_bytes: int = AUTOTUNE_CEILING,
        fine_step_bytes: int = AUTOTUNE_FINE_STEP,
        warmup: int = 100,
        iters: int = 1000,
    ) -> int:
        if self.device.type != "cuda":
            raise ValueError("autotune requires a CUDA device")
        bench_stream = torch.cuda.Stream(device=self.device)
        crossover, results = _compute_crossover_size(
            lambda size_bytes: self._bench_graph_latency(
                size_bytes,
                nccl_group,
                bench_stream,
                warmup,
                iters,
            ),
            ceiling_bytes=ceiling_bytes,
            fine_step_bytes=fine_step_bytes,
        )
        self.max_size = crossover

        if self.rank == 0:

            def fmt_size(size_bytes: int) -> str:
                if size_bytes >= 1024 * 1024:
                    return f"{size_bytes // (1024 * 1024)}MB"
                if size_bytes >= 1024:
                    return f"{size_bytes // 1024}KB"
                return f"{size_bytes}B"

            lines = [
                f"[PCIe oneshot allreduce] Crossover benchmark ({self.world_size} GPUs, bf16):"
            ]
            for result in results:
                lines.append(
                    f"  {fmt_size(result.size_bytes):>6s}:  custom {result.custom_us:6.1f} us  "
                    f"vs  NCCL {result.nccl_us:6.1f} us  -> {result.winner} wins"
                )
            lines.append(
                f"  Setting max_size = {fmt_size(crossover)} (last size where custom AR wins)"
            )
            logger.info("\n".join(lines))
        return crossover

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

    def _close_ipc_imports_strict(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        if getattr(self, "_native_ipc_import_close_failed", False):
            failures.append(
                (
                    "native runtime",
                    RuntimeError(
                        "a native IPC import failed during best-effort teardown "
                        "and can no longer be retried"
                    ),
                )
            )
        if getattr(self, "_ptr", 0):
            try:
                self._ext.dispose(self._ptr)
            except Exception as exc:
                failures.append(("native runtime", exc))
            else:
                self._ptr = 0

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

        if (
            not failures
            and not getattr(self, "_ptr", 0)
            and self._all_python_ipc_imports_closed(closed)
        ):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors("PCIe oneshot", "IPC import close", failures)

    def _close_ipc_imports_best_effort(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        native_imports_closed = not getattr(self, "_ptr", 0)
        if getattr(self, "_ptr", 0):
            dispose = getattr(self._ext, "dispose_best_effort", None)
            if dispose is None:
                dispose = self._ext.dispose
            try:
                result = dispose(self._ptr)
            except Exception:
                pass
            else:
                self._ptr = 0
                # The native best-effort binding reports whether every graph
                # import closed. Legacy/fake dispose methods return None and
                # report failure by raising.
                native_imports_closed = result is not False
                if not native_imports_closed:
                    self._native_ipc_import_close_failed = True

        closed = self._closed_import_indices()
        if self._ipc is not None:
            for buffer_index, shared in enumerate(self._owned_buffers):
                for remote_index, ptr in enumerate(shared.remote_ptrs):
                    key = (buffer_index, remote_index)
                    if key in closed:
                        continue
                    try:
                        self._ipc.cudaIpcCloseMemHandle(ptr)
                    except Exception:
                        pass
                    else:
                        closed.add(key)

        if (
            native_imports_closed
            and not getattr(self, "_ptr", 0)
            and self._all_python_ipc_imports_closed(closed)
        ):
            self._ipc_imports_closed = True

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
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
            self._registered_input_ptrs.clear()
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors("PCIe oneshot", "IPC export free", failures)

    def close(self) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __del__(self) -> None:
        # GC cannot prove that work queued on an arbitrary CUDA stream has
        # completed, and it must not synchronize or enter a distributed
        # barrier.  Retain both mappings and exports for CUDA context teardown;
        # explicit close() is the only release path.
        return


def _is_current_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    return bool(is_capturing())


class PCIeOneshotAllReducePool:
    """Stream-affine PCIe oneshot wrapper.

    A ``PCIeOneshotAllReduce`` instance is a single ordered channel with one
    signal buffer and one double-buffered staging pair. The pool creates a
    separate channel for each CUDA stream key so multi-stream callers never
    reuse those buffers concurrently.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        ipc: Optional[CudaRTLibrary] = None,
        single_channel: bool = False,
        channel_factory: Optional[
            Callable[[Optional[int]], PCIeOneshotAllReduce]
        ] = None,
    ):
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.exchange_group = _resolve_exchange_group(exchange_group, process_group)
        self.process_group = self.exchange_group
        self.eager_buffer_bytes = int(eager_buffer_bytes)
        self.max_size = int(max_size)
        self.rank_data_bytes = int(rank_data_bytes)
        self.single_channel = bool(single_channel)
        self._channel_factory = channel_factory
        self._channels: dict[int, PCIeOneshotAllReduce] = {}
        self._all_channels: list[PCIeOneshotAllReduce] = []
        self._capture_channel_stack: list[PCIeOneshotAllReduce] = []
        self._closed = False

        self._ipc = ipc
        self._ext = ext_module
        if self._channel_factory is None:
            if self.exchange_group is None:
                raise ValueError(
                    "exchange_group is required unless channel_factory is provided"
                )
            if self.device.type != "cuda":
                raise ValueError("PCIe oneshot pool requires a CUDA device")
            _require_full_grid_residency(
                owner="PCIe oneshot",
                required_sms=ONESHOT_REQUIRED_SMS,
                device=self.device,
                exchange_group=self.exchange_group,
            )
            self._ipc = self._ipc or CudaRTLibrary()
            self._ipc.cudaSetDevice(self.device.index or 0)
            self._ext = self._ext or _load_extension()

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeOneshotAllReducePool":
        return cls(
            rank=dist.get_rank(group=exchange_group),
            world_size=dist.get_world_size(group=exchange_group),
            device=device,
            exchange_group=exchange_group,
            eager_buffer_bytes=eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeOneshotAllReducePool":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes
            if eager_buffer_bytes is None
            else eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    def _new_channel(self, stream_key: Optional[int]) -> PCIeOneshotAllReduce:
        if self._channel_factory is not None:
            channel = self._channel_factory(stream_key)
            if self.single_channel:
                channel._stream_affine = False
            channel._bind_stream_key(stream_key)
            self._all_channels.append(channel)
            return channel

        if self.exchange_group is None or self._ipc is None or self._ext is None:
            raise RuntimeError("pool is not configured to allocate channels")

        channel_buffers = PCIeOneshotAllReduce._allocate_eager_channel_buffers(
            self.exchange_group,
            signal_bytes=self._ext.meta_size(),
            eager_buffer_bytes=self.eager_buffer_bytes,
            ipc=self._ipc,
        )
        owned_buffers = [channel_buffers.owned_buffer]
        channel = object.__new__(PCIeOneshotAllReduce)
        init_error: BaseException | None = None
        try:
            PCIeOneshotAllReduce.__init__(
                channel,
                rank=self.rank,
                world_size=self.world_size,
                device=self.device,
                signal_ptrs=channel_buffers.signal_ptrs,
                eager_buffer_ptrs0=channel_buffers.eager0_ptrs,
                eager_buffer_ptrs1=channel_buffers.eager1_ptrs,
                exchange_group=self.exchange_group,
                ipc=self._ipc,
                owned_buffers=owned_buffers,
                max_size=self.max_size,
                rank_data_bytes=self.rank_data_bytes,
                ext_module=self._ext,
                stream_affine=not self.single_channel,
            )
        except Exception as exc:
            init_error = exc

        def abort_native_runtime() -> None:
            pointer = getattr(channel, "_ptr", 0)
            if pointer:
                self._ext.dispose(pointer)
                channel._ptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe oneshot pool channel",
            exchange_group=self.exchange_group,
            ipc=self._ipc,
            shared=channel_buffers.owned_buffer,
            local_error=init_error,
            local_cleanup=abort_native_runtime,
        )
        channel._bind_stream_key(stream_key)
        self._all_channels.append(channel)
        return channel

    def checkpoint_channels(
        self,
    ) -> tuple[int, dict[int, PCIeOneshotAllReduce]]:
        """Snapshot channel ownership before a throwaway graph capture."""
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot checkpoint channels during capture")
        return len(self._all_channels), dict(self._channels)

    def rollback_channels(
        self,
        checkpoint: tuple[int, dict[int, PCIeOneshotAllReduce]],
    ) -> None:
        """Close channels created after ``checkpoint`` and restore mappings.

        Callers must destroy and synchronize any graphs that reference the
        transient channels before rolling back.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._capture_channel_stack:
            raise RuntimeError("cannot roll back channels during capture")
        all_channels_len, channels = checkpoint
        if not 0 <= all_channels_len <= len(self._all_channels):
            raise ValueError("channel checkpoint does not belong to this pool")

        retained = self._all_channels[:all_channels_len]
        retained_ids = {id(channel) for channel in retained}
        transient = self._all_channels[all_channels_len:]

        channels_to_close = tuple(
            dict.fromkeys(
                channel for channel in transient if id(channel) not in retained_ids
            )
        )
        _coordinated_close_channels(
            channels_to_close,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        self._all_channels = retained
        self._channels = dict(channels)

    def for_stream(self, stream: object = None) -> PCIeOneshotAllReduce:
        if self._closed:
            raise RuntimeError("pool is closed")
        if self.single_channel:
            channel = self._channels.get(0)
            if channel is not None:
                return channel
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "PCIe oneshot pool has no channel to reuse during CUDA graph "
                    "capture; perform an eager all-reduce (or call for_stream) "
                    "before capture starts"
                )
            channel = self._new_channel(None)
            self._channels[0] = channel
            return channel

        stream_key = _current_stream_key(self.device, stream)
        channel_key = 0 if stream_key is None else int(stream_key)
        if _is_current_stream_capturing(self.device) and self._capture_channel_stack:
            # CUDA and torch/Inductor may recycle a nested capture stream key
            # between independent target and draft graph managers. The active
            # enclosing capture owns the channel even if that key still maps
            # to a channel captured by an earlier manager.
            channel = self._capture_channel_stack[-1]
            self._channels[channel_key] = channel
            return channel
        channel = self._channels.get(channel_key)
        if channel is not None:
            return channel
        if _is_current_stream_capturing(self.device):
            # Piecewise / inductor CUDA graphs (MTP, spec-decode) capture on a
            # torch-owned stream that we cannot pre-register before capture
            # starts. Reuse the channel selected by the enclosing vLLM graph
            # capture, not an arbitrary channel from another graph manager.
            # Target and draft graph managers can replay independently; if
            # their nested captures share signal/staging buffers, they race.
            if self._capture_channel_stack:
                channel = self._capture_channel_stack[-1]
                self._channels[channel_key] = channel
                return channel
            # Preserve compatibility for callers that enter CUDA capture
            # without the pool.capture() context. They must already have a
            # channel because allocating CUDA IPC storage mid-capture is
            # illegal.
            if self._channels:
                channel = next(iter(self._channels.values()))
                self._channels[channel_key] = channel
                return channel
            raise RuntimeError(
                "PCIe oneshot pool has no channel to reuse during CUDA graph "
                "capture; perform an eager all-reduce (or call for_stream) "
                "before capture starts"
            )
        channel = self._new_channel(stream_key)
        self._channels[channel_key] = channel
        return channel

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        stream: object = None,
    ) -> torch.Tensor:
        channel = self.for_stream(stream)
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return channel.all_reduce(inp, out=out, peer_input_ptrs=peer_input_ptrs)
        return channel.all_reduce(inp, out=out, peer_input_ptrs=peer_input_ptrs)

    def all_reduce_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        stream: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel = self.for_stream(stream)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return channel.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
                out=out,
                residual_out=residual_out,
                peer_input_ptrs=peer_input_ptrs,
            )

        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                return run()
        return run()

    @contextmanager
    def capture(self, stream: object = None):
        if self.single_channel:
            channel = self.for_stream(stream)
        else:
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "PCIe oneshot capture context must be entered before CUDA "
                    "graph capture starts"
                )
            stream_key = _current_stream_key(self.device, stream)
            channel_key = 0 if stream_key is None else int(stream_key)
            # A CUDA stream handle can be recycled after one graph manager
            # finishes capture. Graphs retain the channel pointers, so a new
            # manager must receive a fresh channel even when the numeric stream
            # key is identical. _all_channels keeps replaced channels alive.
            channel = self._new_channel(stream_key)
            self._channels[channel_key] = channel
        with channel.capture(stream=stream):
            self._capture_channel_stack.append(channel)
            try:
                yield channel
            finally:
                popped = self._capture_channel_stack.pop()
                if popped is not channel:
                    raise RuntimeError("PCIe oneshot capture channel stack corrupted")

    def close(self) -> None:
        if self._closed:
            return
        seen: set[int] = set()
        channels = []
        for channel in (*self._all_channels, *self._channels.values()):
            if id(channel) not in seen:
                seen.add(id(channel))
                channels.append(channel)
        _coordinated_close_channels(
            channels,
            exchange_group=self.exchange_group,
            device=self.device,
        )
        self._closed = True
        self._all_channels.clear()
        self._channels.clear()

    def __del__(self) -> None:
        # A pool may be collected while graph or stream work still references
        # its channels.  Do not unmap, synchronize, or communicate from GC.
        return


__all__ = [
    "PCIeOneshotAllReduce",
    "PCIeOneshotAllReducePool",
    "SUPPORTED_WORLD_SIZES",
    "parse_pcie_oneshot_max_size",
]
