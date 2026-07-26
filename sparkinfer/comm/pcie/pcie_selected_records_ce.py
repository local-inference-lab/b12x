"""Copy-engine PCIe exchange for destination-selected fixed-width records."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from . import pcie_dma as _pcie_dma
from ._cuda_ipc import CudaRTLibrary
from .pcie_dma import FLAG_STRIDE
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    _current_stream_key,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _align_up,
)
from .pcie_selected_records import (
    DEFAULT_BARRIER_TIMEOUT_CYCLES,
    MAX_WORLD_SIZE,
    PCIeSelectedRecordExchangeInitializationError,
    _all_ranks_succeeded,
    _allocate_shared_buffer_rank_consistent,
    _free_shared_buffer,
)


_MAX_INT64 = (1 << 63) - 1
_CONFIG_VERSION = 2
_COUNT_BYTES = 8
_POSITION_BYTES = 8
_PACKET_HEADER_BYTES = 2 * _COUNT_BYTES
_LAYER_PLANES = 3
_CONFIG_FIELDS = (
    "version",
    "world_size",
    "max_records",
    "record_bytes",
    "layered_max_records",
    "layered_record_bytes",
    "layered_slab_bytes",
    "primary_capacity",
    "overflow_capacity",
    "flags_bytes",
    "receive_primary_base",
    "staging_primary_base",
    "staging_overflow_base",
    "primary_positions_offset",
    "primary_records_offset",
    "primary_stride",
    "overflow_positions_offset",
    "overflow_records_offset",
    "overflow_stride",
    "slab_bytes",
    "flag_stride",
    "slab_alignment",
)


@dataclass(frozen=True)
class _CopyExchangeLayout:
    flags_bytes: int
    receive_primary_base: int
    staging_primary_base: int
    staging_overflow_base: int
    primary_capacity: int
    overflow_capacity: int
    primary_positions_offset: int
    primary_records_offset: int
    primary_stride: int
    overflow_positions_offset: int
    overflow_records_offset: int
    overflow_stride: int
    slab_bytes: int


def _checked_region_end(base: int, count: int, stride: int, name: str) -> int:
    if count < 0 or stride < 0 or base < 0:
        raise ValueError(f"{name} has a negative layout component")
    if count and stride > (_MAX_INT64 - base) // count:
        raise ValueError(f"{name} exceeds int64 capacity")
    return base + count * stride


def _copy_exchange_layout(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    primary_capacity: Optional[int] = None,
) -> _CopyExchangeLayout:
    world_size = int(world_size)
    max_records = int(max_records)
    record_bytes = int(record_bytes)
    if not 2 <= world_size <= MAX_WORLD_SIZE:
        raise ValueError(
            f"world_size must be in [2, {MAX_WORLD_SIZE}], got {world_size}"
        )
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_bytes <= 0:
        raise ValueError("record_bytes must be positive")
    if max_records > _MAX_INT64 // record_bytes:
        raise ValueError("max_records * record_bytes exceeds int64 capacity")

    if primary_capacity is None:
        primary_capacity = (max_records + world_size - 1) // world_size
    primary_capacity = int(primary_capacity)
    if not 1 <= primary_capacity <= max_records:
        raise ValueError("primary_capacity must be in [1, max_records]")
    overflow_capacity = max_records - primary_capacity

    flags_bytes = _align_up(
        2 * world_size * FLAG_STRIDE,
        IPC_SLAB_ALIGNMENT,
    )
    primary_positions_offset = _PACKET_HEADER_BYTES
    primary_records_offset = _align_up(
        primary_positions_offset + primary_capacity * _POSITION_BYTES,
        16,
    )
    primary_stride = _align_up(
        _checked_region_end(
            primary_records_offset,
            primary_capacity,
            record_bytes,
            "primary packet",
        ),
        IPC_SLAB_ALIGNMENT,
    )

    # Keep a valid, distinct overflow pointer even when primary_capacity covers
    # the complete pool. No overflow bytes are consumed in that configuration.
    overflow_storage_records = max(1, overflow_capacity)
    overflow_positions_offset = 0
    overflow_records_offset = _align_up(
        overflow_storage_records * _POSITION_BYTES,
        16,
    )
    overflow_stride = _align_up(
        _checked_region_end(
            overflow_records_offset,
            overflow_storage_records,
            record_bytes,
            "overflow packet",
        ),
        IPC_SLAB_ALIGNMENT,
    )

    receive_primary_base = flags_bytes
    staging_primary_base = _align_up(
        _checked_region_end(
            receive_primary_base,
            world_size,
            primary_stride,
            "primary receive area",
        ),
        IPC_SLAB_ALIGNMENT,
    )
    staging_overflow_base = _align_up(
        _checked_region_end(
            staging_primary_base,
            world_size,
            primary_stride,
            "primary staging area",
        ),
        IPC_SLAB_ALIGNMENT,
    )
    slab_bytes = _checked_region_end(
        staging_overflow_base,
        world_size,
        overflow_stride,
        "overflow staging area",
    )
    if slab_bytes > _MAX_INT64:
        raise ValueError("copy-engine selected-record slab exceeds int64 capacity")

    return _CopyExchangeLayout(
        flags_bytes=flags_bytes,
        receive_primary_base=receive_primary_base,
        staging_primary_base=staging_primary_base,
        staging_overflow_base=staging_overflow_base,
        primary_capacity=primary_capacity,
        overflow_capacity=overflow_capacity,
        primary_positions_offset=primary_positions_offset,
        primary_records_offset=primary_records_offset,
        primary_stride=primary_stride,
        overflow_positions_offset=overflow_positions_offset,
        overflow_records_offset=overflow_records_offset,
        overflow_stride=overflow_stride,
        slab_bytes=slab_bytes,
    )


def _largest_layered_capacity_for_slab(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    slab_bytes: int,
) -> tuple[int, _CopyExchangeLayout]:
    max_records = int(max_records)
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_bytes > _MAX_INT64 // _LAYER_PLANES:
        raise ValueError("layered record width exceeds int64 capacity")
    layered_record_bytes = record_bytes * _LAYER_PLANES
    low = 1
    high = max_records
    first_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=1,
        record_bytes=layered_record_bytes,
    )
    if first_layout.slab_bytes > int(slab_bytes):
        raise ValueError("base slab cannot hold one three-layer record")
    while low < high:
        candidate = (low + high + 1) // 2
        candidate_layout = _copy_exchange_layout(
            world_size=world_size,
            max_records=candidate,
            record_bytes=layered_record_bytes,
        )
        if candidate_layout.slab_bytes <= int(slab_bytes):
            low = candidate
        else:
            high = candidate - 1
    layered_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=low,
        record_bytes=layered_record_bytes,
    )
    return low, layered_layout


def _layered_layout_for_slab(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    slab_bytes: int,
    layered_max_records: Optional[int],
    fallback_layout: _CopyExchangeLayout,
) -> tuple[int, _CopyExchangeLayout]:
    if layered_max_records is None:
        try:
            return _largest_layered_capacity_for_slab(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                slab_bytes=slab_bytes,
            )
        except ValueError as exc:
            if str(exc) != "base slab cannot hold one three-layer record":
                raise
            return 0, fallback_layout

    layered_max_records = int(layered_max_records)
    if layered_max_records == 0:
        return 0, fallback_layout
    if not 1 <= layered_max_records <= int(max_records):
        raise ValueError("layered_max_records must be in [0, max_records]")
    if record_bytes > _MAX_INT64 // _LAYER_PLANES:
        raise ValueError("layered record width exceeds int64 capacity")
    layered_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=layered_max_records,
        record_bytes=record_bytes * _LAYER_PLANES,
    )
    if layered_layout.slab_bytes > int(slab_bytes):
        raise ValueError(
            "three-layer layout exceeds the existing selected-record slab "
            f"({layered_layout.slab_bytes} > {slab_bytes} bytes)"
        )
    return layered_max_records, layered_layout


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_selected_records_ce.cu")
    return load(
        name="sparkinfer_pcie_selected_records_ce_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _configuration_values(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    layout: _CopyExchangeLayout,
    layered_max_records: int,
    layered_layout: _CopyExchangeLayout,
) -> tuple[int, ...]:
    return (
        _CONFIG_VERSION,
        int(world_size),
        int(max_records),
        int(record_bytes),
        int(layered_max_records),
        int(record_bytes) * _LAYER_PLANES,
        layered_layout.slab_bytes,
        layout.primary_capacity,
        layout.overflow_capacity,
        layout.flags_bytes,
        layout.receive_primary_base,
        layout.staging_primary_base,
        layout.staging_overflow_base,
        layout.primary_positions_offset,
        layout.primary_records_offset,
        layout.primary_stride,
        layout.overflow_positions_offset,
        layout.overflow_records_offset,
        layout.overflow_stride,
        layout.slab_bytes,
        FLAG_STRIDE,
        IPC_SLAB_ALIGNMENT,
    )


def _validate_rank_configuration(
    *,
    process_group: ProcessGroup,
    device: torch.device,
    status: torch.Tensor,
    world_size: int,
    max_records: int,
    record_bytes: int,
    layout: _CopyExchangeLayout,
    layered_max_records: int,
    layered_layout: _CopyExchangeLayout,
) -> None:
    local_config: Optional[torch.Tensor] = None
    gathered_configs: Optional[torch.Tensor] = None
    local_error: Optional[Exception] = None
    try:
        local_config = torch.tensor(
            _configuration_values(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                layout=layout,
                layered_max_records=layered_max_records,
                layered_layout=layered_layout,
            ),
            dtype=torch.int64,
            device=device,
        )
        gathered_configs = torch.empty(
            world_size * len(_CONFIG_FIELDS),
            dtype=torch.int64,
            device=device,
        )
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "copy-engine selected-record configuration allocation failed on at "
            "least one rank"
        ) from local_error

    assert local_config is not None
    assert gathered_configs is not None
    try:
        dist.all_gather_into_tensor(
            gathered_configs,
            local_config,
            group=process_group,
        )
    except Exception as exc:
        raise PCIeSelectedRecordExchangeInitializationError(
            "copy-engine selected-record configuration collective failed"
        ) from exc

    rows = gathered_configs.view(world_size, len(_CONFIG_FIELDS))
    local_matches = False
    local_error = None
    try:
        local_matches = bool(torch.all(rows == local_config.unsqueeze(0)).item())
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "copy-engine selected-record configuration comparison failed on at "
            "least one rank"
        ) from local_error
    if not _all_ranks_succeeded(status, local_matches, process_group):
        try:
            rank_configs = rows.cpu().tolist()
        except Exception:
            rank_configs = "unavailable"
        raise PCIeSelectedRecordExchangeInitializationError(
            "copy-engine selected-record configuration mismatch across ranks "
            f"(fields={_CONFIG_FIELDS}, values={rank_configs})"
        )


class PCIeSelectedRecordCopyExchange:
    """Ordered pack/copy-engine/unpack exchange for selected byte records.

    ``local_indices_by_destination`` is destination-major. Its trailing
    dimensions flatten to the output record order. Each entry is either a
    non-negative index into ``records`` or negative when this source does not
    own that selected record. Across all ranks, exactly one source must own
    each destination/output position. Every rank must invoke the channel in
    the same order with the same selected-record count.

    Records are compacted per destination. A balanced primary packet is moved
    with the CUDA copy engine; owner skew beyond that packet remains in the
    source IPC slab and is read during unpack. Both exchange modes publish a
    release after unpack and defer the matching wait until the next staging
    reuse. Keeping that protocol identical makes separately captured normal and
    layered CUDA graphs safe to alternate. ``exchange_layers`` packs exactly
    three record planes into one packet. The layered packet layout overlays the
    same IPC slab with a smaller ``layered_max_records`` capacity; callers must
    use the single-layer fallback above that bound. Configurations whose slab
    cannot hold even one layered record expose a zero layered capacity while
    retaining the normal exchange.
    """

    record_ptrs_require_release = True

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        peer_slab_ptrs: Sequence[int],
        max_records: int,
        record_bytes: int,
        primary_capacity: Optional[int] = None,
        layered_max_records: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffer: Optional[_OwnedSharedBuffer] = None,
        ext_module=None,
        dma_ext_module=None,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        stream_affine: bool = True,
    ) -> None:
        layout = _copy_exchange_layout(
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            primary_capacity=primary_capacity,
        )
        layered_max_records, layered_layout = _layered_layout_for_slab(
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            slab_bytes=layout.slab_bytes,
            layered_max_records=layered_max_records,
            fallback_layout=layout,
        )
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(peer_slab_ptrs) != int(world_size):
            raise ValueError("peer_slab_ptrs must match world_size")
        if int(barrier_timeout_cycles) <= 0:
            raise ValueError("barrier_timeout_cycles must be positive")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.process_group = process_group
        self.max_records = int(max_records)
        self.record_bytes = int(record_bytes)
        self.layered_max_records = layered_max_records
        self.primary_capacity = layout.primary_capacity
        self.overflow_capacity = layout.overflow_capacity
        self.barrier_timeout_cycles = int(barrier_timeout_cycles)
        self._layout = layout
        self._layer_layout = layered_layout
        self._ipc = ipc
        self._owned_buffer = owned_buffer
        self._ext = ext_module or _load_extension()
        self._dma_ext = dma_ext_module or _pcie_dma._load_extension()
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._deferred_release_pending = False
        self._pointer_table_active = False

        slab_ptrs = tuple(int(pointer) for pointer in peer_slab_ptrs)
        self._slab_ptrs = slab_ptrs
        self._local_slab_ptr = slab_ptrs[self.rank]
        self._peer_overflow_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[source]
                    + layout.staging_overflow_base
                    + destination * layout.overflow_stride
                    for source in range(self.world_size)
                ]
                for destination in range(self.world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._local_receive_primary_ptrs = torch.tensor(
            [
                self._local_slab_ptr
                + layout.receive_primary_base
                + source * layout.primary_stride
                for source in range(self.world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._peer_staging_primary_ptrs = torch.tensor(
            [
                slab_ptrs[source]
                + layout.staging_primary_base
                + self.rank * layout.primary_stride
                for source in range(self.world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._peer_staging_overflow_ptrs = self._peer_overflow_ptrs[self.rank]
        self._peer_layer_overflow_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[source]
                    + layered_layout.staging_overflow_base
                    + destination * layered_layout.overflow_stride
                    for source in range(self.world_size)
                ]
                for destination in range(self.world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._barrier_publish_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[destination]
                    + (phase * self.world_size + self.rank) * FLAG_STRIDE
                    for destination in range(self.world_size)
                ]
                for phase in range(2)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._barrier_wait_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[self.rank]
                    + (phase * self.world_size + source) * FLAG_STRIDE
                    for source in range(self.world_size)
                ]
                for phase in range(2)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_counters = torch.zeros(
            (2, self.world_size),
            dtype=torch.int32,
            device=self.device,
        )
        self._wait_counters = torch.zeros_like(self._send_counters)

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_records: int,
        record_bytes: int,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        ext_module=None,
        dma_ext_module=None,
        stream_affine: bool = True,
        primary_capacity: Optional[int] = None,
        layered_max_records: Optional[int] = None,
    ) -> "PCIeSelectedRecordCopyExchange":
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record exchange requires a CUDA device"
            )
        device_index = (
            torch.cuda.current_device()
            if device_obj.index is None
            else int(device_obj.index)
        )
        device_obj = torch.device("cuda", device_index)

        try:
            backend = dist.get_backend(group=process_group)
        except TypeError:
            backend = dist.get_backend(process_group)
        if "nccl" not in str(backend).lower():
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record exchange requires an NCCL process "
                f"group, got {backend}"
            )

        status = torch.empty(1, dtype=torch.int32, device=device_obj)
        layout: Optional[_CopyExchangeLayout] = None
        layered_layout: Optional[_CopyExchangeLayout] = None
        resolved_layered_max_records: Optional[int] = None
        local_error: Optional[Exception] = None
        try:
            layout = _copy_exchange_layout(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                primary_capacity=primary_capacity,
            )
            resolved_layered_max_records, layered_layout = _layered_layout_for_slab(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                slab_bytes=layout.slab_bytes,
                layered_max_records=layered_max_records,
                fallback_layout=layout,
            )
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record configuration is invalid on at "
                "least one rank"
            ) from local_error
        assert layout is not None
        assert layered_layout is not None
        assert resolved_layered_max_records is not None
        _validate_rank_configuration(
            process_group=process_group,
            device=device_obj,
            status=status,
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            layout=layout,
            layered_max_records=resolved_layered_max_records,
            layered_layout=layered_layout,
        )

        ext = None
        dma_ext = None
        local_error = None
        try:
            ext = ext_module or _load_extension()
            dma_ext = dma_ext_module or _pcie_dma._load_extension()
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record CUDA extensions failed to load on "
                "at least one rank"
            ) from local_error
        assert ext is not None
        assert dma_ext is not None

        ipc: Optional[CudaRTLibrary] = None
        local_error = None
        try:
            ipc = CudaRTLibrary()
            ipc.cudaSetDevice(device_index)
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record CUDA IPC setup failed on at least one rank"
            ) from local_error
        assert ipc is not None

        shared = _allocate_shared_buffer_rank_consistent(
            process_group=process_group,
            rank=rank,
            world_size=world_size,
            device=device_obj,
            size_in_bytes=layout.slab_bytes,
            ipc=ipc,
            status=status,
        )
        exchange: Optional[PCIeSelectedRecordCopyExchange] = None
        local_error = None
        try:
            exchange = cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                peer_slab_ptrs=shared.peer_ptrs,
                max_records=max_records,
                record_bytes=record_bytes,
                primary_capacity=layout.primary_capacity,
                layered_max_records=resolved_layered_max_records,
                process_group=process_group,
                ipc=ipc,
                owned_buffer=None,
                ext_module=ext,
                dma_ext_module=dma_ext,
                barrier_timeout_cycles=barrier_timeout_cycles,
                stream_affine=stream_affine,
            )
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            if exchange is not None:
                exchange._closed = True
            _free_shared_buffer(ipc, shared)
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record runtime initialization failed on at "
                "least one rank"
            ) from local_error
        assert exchange is not None
        exchange._owned_buffer = shared
        return exchange

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine or stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe selected-record copy channels are stream-affine; create a "
                "separate channel for each CUDA stream"
            )

    def _check_stream(self) -> None:
        if self.device.type != "cuda":
            return
        stream_key = _current_stream_key(self.device)
        if (
            _is_current_stream_capturing(self.device)
            and self._stream_affine
            and self._owner_stream_key is None
        ):
            raise RuntimeError(
                "PCIe selected-record copy channels must be bound to their CUDA "
                "stream before first-use graph capture"
            )
        self._bind_stream_key(stream_key)

    def _validate(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> int:
        if self._closed:
            raise RuntimeError("PCIeSelectedRecordCopyExchange is closed")
        if records.device != self.device:
            raise ValueError("records must be on the exchange device")
        if local_indices_by_destination.device != self.device:
            raise ValueError("local indices must be on the exchange device")
        if out.device != self.device:
            raise ValueError("output must be on the exchange device")
        if records.dtype != torch.uint8 or not records.is_contiguous():
            raise ValueError("records must be contiguous uint8")
        if records.ndim < 2 or records.shape[-1] != self.record_bytes:
            raise ValueError(
                f"records must end in the configured {self.record_bytes}-byte width"
            )
        if local_indices_by_destination.dtype not in (torch.int32, torch.int64):
            raise ValueError("local indices must be int32 or int64")
        if not local_indices_by_destination.is_contiguous():
            raise ValueError("local indices must be contiguous")
        if (
            local_indices_by_destination.ndim < 2
            or local_indices_by_destination.shape[0] != self.world_size
        ):
            raise ValueError(
                "local indices must have shape [world_size, ...selected records]"
            )
        active_records = local_indices_by_destination.numel() // self.world_size
        if active_records > self.max_records:
            raise ValueError(
                f"selected record count {active_records} exceeds capacity "
                f"{self.max_records}"
            )
        if active_records > _MAX_INT64 // self.record_bytes:
            raise ValueError("active selected-record payload exceeds int64 capacity")
        if out.dtype != torch.uint8 or not out.is_contiguous():
            raise ValueError("output must be contiguous uint8")
        if out.ndim < 2 or out.shape[-1] != self.record_bytes:
            raise ValueError(
                f"output must end in the configured {self.record_bytes}-byte width"
            )
        if out.numel() != active_records * self.record_bytes:
            raise ValueError(
                "output must contain exactly one record per selected position"
            )
        return active_records

    def _active_primary_capacity(self, active_records: int) -> int:
        if active_records == 0:
            return 0
        scaled = (
            self.primary_capacity * active_records + self.max_records - 1
        ) // self.max_records
        active_primary = min(scaled, self.primary_capacity, active_records)
        active_overflow = active_records - active_primary
        if active_overflow > self.overflow_capacity:
            raise ValueError("active selected-record overflow exceeds channel capacity")
        return active_primary

    def _active_layer_primary_capacity(self, active_records: int) -> int:
        if active_records == 0:
            return 0
        layout = self._layer_layout
        scaled = (
            layout.primary_capacity * active_records + self.layered_max_records - 1
        ) // self.layered_max_records
        active_primary = min(
            scaled,
            layout.primary_capacity,
            active_records,
        )
        active_overflow = active_records - active_primary
        if active_overflow > layout.overflow_capacity:
            raise ValueError(
                "active three-layer selected-record overflow exceeds channel capacity"
            )
        return active_primary

    def _receive_primary_ptr(self, destination: int, source: int) -> int:
        return (
            self._slab_ptrs[destination]
            + self._layout.receive_primary_base
            + source * self._layout.primary_stride
        )

    def _local_staging_primary_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layout.staging_primary_base
            + destination * self._layout.primary_stride
        )

    def _local_staging_overflow_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layout.staging_overflow_base
            + destination * self._layout.overflow_stride
        )

    def _layer_receive_primary_ptr(self, destination: int, source: int) -> int:
        return (
            self._slab_ptrs[destination]
            + self._layer_layout.receive_primary_base
            + source * self._layer_layout.primary_stride
        )

    def _local_layer_staging_primary_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layer_layout.staging_primary_base
            + destination * self._layer_layout.primary_stride
        )

    def _local_layer_staging_overflow_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layer_layout.staging_overflow_base
            + destination * self._layer_layout.overflow_stride
        )

    def _barrier(self, phase: int) -> None:
        self._ext.barrier_all_peers(
            self._barrier_publish_ptrs,
            self._barrier_wait_ptrs,
            self._send_counters,
            self._wait_counters,
            int(phase),
            self.barrier_timeout_cycles,
        )

    def _publish_deferred_release(self) -> None:
        self._ext.publish_all_peers(
            self._barrier_publish_ptrs,
            self._send_counters,
            1,
        )
        self._deferred_release_pending = True

    def _wait_for_deferred_release(self) -> None:
        if self._pointer_table_active:
            raise RuntimeError(
                "selected-record pointer table is still active; enqueue its "
                "consumer and call release_record_ptrs() before channel reuse"
            )
        if not self._deferred_release_pending:
            return
        self._ext.wait_all_peers(
            self._barrier_wait_ptrs,
            self._wait_counters,
            1,
            self.barrier_timeout_cycles,
        )
        self._deferred_release_pending = False

    def _require_eager_warmup_for_capture(self) -> None:
        if (
            self.device.type == "cuda"
            and _is_current_stream_capturing(self.device)
            and not self._deferred_release_pending
        ):
            raise RuntimeError(
                "copy-engine selected-record exchange requires one eager "
                "warmup before CUDA graph capture"
            )

    def _validate_layers(
        self,
        records_by_layer: Sequence[torch.Tensor],
        local_indices_by_destination: torch.Tensor,
        outputs_by_layer: Sequence[torch.Tensor],
    ) -> int:
        if len(records_by_layer) != _LAYER_PLANES:
            raise ValueError(f"records_by_layer must contain {_LAYER_PLANES} tensors")
        if len(outputs_by_layer) != _LAYER_PLANES:
            raise ValueError(f"outputs_by_layer must contain {_LAYER_PLANES} tensors")

        active_records: Optional[int] = None
        for records, out in zip(records_by_layer, outputs_by_layer, strict=True):
            current = self._validate(records, local_indices_by_destination, out)
            if active_records is None:
                active_records = current
            elif current != active_records:
                raise ValueError("all layer outputs must have the same record count")

        assert active_records is not None
        if active_records:
            output_ranges = sorted(
                (
                    int(output.data_ptr()),
                    int(output.data_ptr()) + output.numel(),
                )
                for output in outputs_by_layer
            )
            if any(
                output_ranges[index][1] > output_ranges[index + 1][0]
                for index in range(len(output_ranges) - 1)
            ):
                raise ValueError("outputs_by_layer must use non-overlapping storage")
        if active_records > self.layered_max_records:
            raise ValueError(
                f"three-layer selected record count {active_records} exceeds "
                f"bounded capacity {self.layered_max_records}; use the "
                "single-layer exchange fallback"
            )
        record_shapes = {tuple(records.shape) for records in records_by_layer}
        if len(record_shapes) != 1:
            raise ValueError("all layer record tensors must have the same shape")
        return active_records

    def exchange(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Pack owned records, copy them to peers, and reconstruct ``out``."""
        if self._pointer_table_active:
            raise RuntimeError("cannot exchange while a record pointer table is active")
        self._check_stream()
        self._require_eager_warmup_for_capture()
        active_records = self._validate(
            records,
            local_indices_by_destination,
            out,
        )
        self._wait_for_deferred_release()
        if active_records == 0:
            self._barrier(0)
            out.zero_()
            self._publish_deferred_release()
            return out

        active_primary = self._active_primary_capacity(active_records)
        primary_copy_bytes = _align_up(
            self._layout.primary_records_offset + active_primary * self.record_bytes,
            16,
        )

        # Rotate both pack and CE issue order by source rank. At phase zero,
        # rank R targets destination R instead of every rank targeting rank 0.
        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_records(
                records,
                local_indices_by_destination[destination],
                self._local_staging_primary_ptr(destination),
                self._local_staging_overflow_ptr(destination),
                self.record_bytes,
                active_primary,
                self._layout.primary_positions_offset,
                self._layout.primary_records_offset,
                self._layout.overflow_positions_offset,
                self._layout.overflow_records_offset,
            )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._dma_ext.dma_copy(
                self._receive_primary_ptr(destination, self.rank),
                self._local_staging_primary_ptr(destination),
                primary_copy_bytes,
            )

        self._barrier(0)
        out.zero_()
        self._ext.unpack_compact_records(
            self._local_slab_ptr + self._layout.receive_primary_base,
            self._layout.primary_stride,
            self._peer_overflow_ptrs[self.rank],
            out,
            active_records,
            self.record_bytes,
            active_primary,
            self._layout.primary_positions_offset,
            self._layout.primary_records_offset,
            self._layout.overflow_positions_offset,
            self._layout.overflow_records_offset,
        )
        self._publish_deferred_release()
        return out

    def prepare_record_ptrs(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out_ptrs: torch.Tensor,
        *,
        primary_mode: str = "ce",
    ) -> torch.Tensor:
        """Build destination-order pointers without materializing output records.

        ``primary_mode='ce'`` keeps the balanced copy-engine primary packet and
        points only owner skew at peer memory. ``'peer'`` skips primary DMA and
        points at every source's IPC staging packet. Enqueue all consumers on
        this stream, then call :meth:`release_record_ptrs` before channel reuse.
        """
        if self._pointer_table_active:
            raise RuntimeError("a selected-record pointer table is already active")
        if primary_mode not in {"ce", "peer"}:
            raise ValueError("primary_mode must be 'ce' or 'peer'")
        if self._closed:
            raise RuntimeError("PCIeSelectedRecordCopyExchange is closed")
        if records.device != self.device:
            raise ValueError("records must be on the exchange device")
        if local_indices_by_destination.device != self.device:
            raise ValueError("local indices must be on the exchange device")
        if out_ptrs.device != self.device:
            raise ValueError("output pointers must be on the exchange device")
        if records.dtype != torch.uint8 or not records.is_contiguous():
            raise ValueError("records must be contiguous uint8")
        if records.ndim < 2 or records.shape[-1] != self.record_bytes:
            raise ValueError(
                f"records must end in the configured {self.record_bytes}-byte width"
            )
        if local_indices_by_destination.dtype not in (torch.int32, torch.int64):
            raise ValueError("local indices must be int32 or int64")
        if not local_indices_by_destination.is_contiguous():
            raise ValueError("local indices must be contiguous")
        if (
            local_indices_by_destination.ndim < 2
            or local_indices_by_destination.shape[0] != self.world_size
        ):
            raise ValueError(
                "local indices must have shape [world_size, ...selected records]"
            )
        active_records = local_indices_by_destination.numel() // self.world_size
        if not 0 < active_records <= self.max_records:
            raise ValueError(
                f"selected record count {active_records} is outside "
                f"[1, {self.max_records}]"
            )
        if (
            out_ptrs.dtype != torch.int64
            or not out_ptrs.is_contiguous()
            or out_ptrs.ndim != 1
            or out_ptrs.numel() != active_records
        ):
            raise ValueError(
                "out_ptrs must be a contiguous int64 vector with one entry "
                "per selected record"
            )

        self._check_stream()
        self._require_eager_warmup_for_capture()
        self._wait_for_deferred_release()
        active_primary = self._active_primary_capacity(active_records)
        primary_copy_bytes = _align_up(
            self._layout.primary_records_offset + active_primary * self.record_bytes,
            16,
        )
        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_records(
                records,
                local_indices_by_destination[destination],
                self._local_staging_primary_ptr(destination),
                self._local_staging_overflow_ptr(destination),
                self.record_bytes,
                active_primary,
                self._layout.primary_positions_offset,
                self._layout.primary_records_offset,
                self._layout.overflow_positions_offset,
                self._layout.overflow_records_offset,
            )
        if primary_mode == "ce":
            for destination_phase in range(self.world_size):
                destination = (self.rank + destination_phase) % self.world_size
                self._dma_ext.dma_copy(
                    self._receive_primary_ptr(destination, self.rank),
                    self._local_staging_primary_ptr(destination),
                    primary_copy_bytes,
                )

        self._barrier(0)
        primary_ptrs = (
            self._local_receive_primary_ptrs
            if primary_mode == "ce"
            else self._peer_staging_primary_ptrs
        )
        self._ext.build_compact_record_ptrs(
            primary_ptrs,
            self._peer_staging_overflow_ptrs,
            out_ptrs,
            self.record_bytes,
            active_primary,
            self._layout.primary_positions_offset,
            self._layout.primary_records_offset,
            self._layout.overflow_positions_offset,
            self._layout.overflow_records_offset,
        )
        self._pointer_table_active = True
        return out_ptrs

    def prepare_storage_record_ptrs(
        self,
        local_indices_by_destination: torch.Tensor,
        peer_record_base_ptrs: torch.Tensor,
        out_ptrs: torch.Tensor,
        *,
        source_record_bytes: int,
    ) -> torch.Tensor:
        """Build stable pointers into model-owned peer storage.

        This channel must be configured for eight-byte ordinal records.  Only
        ``(selected position, source-local ordinal)`` metadata crosses PCIe;
        the resulting pointers target the original model KV allocation, so the
        compact packet can be retired immediately after pointer construction.
        """
        if self.record_bytes != _POSITION_BYTES:
            raise RuntimeError(
                "storage-pointer channels require eight-byte ordinal packets"
            )
        if self._pointer_table_active:
            raise RuntimeError("a packet-backed pointer table is still active")
        if self._closed:
            raise RuntimeError("PCIeSelectedRecordCopyExchange is closed")
        if local_indices_by_destination.device != self.device:
            raise ValueError("local indices must be on the exchange device")
        if local_indices_by_destination.dtype not in (torch.int32, torch.int64):
            raise ValueError("local indices must be int32 or int64")
        if not local_indices_by_destination.is_contiguous():
            raise ValueError("local indices must be contiguous")
        if (
            local_indices_by_destination.ndim < 2
            or local_indices_by_destination.shape[0] != self.world_size
        ):
            raise ValueError(
                "local indices must have shape [world_size, ...selected records]"
            )
        active_records = local_indices_by_destination.numel() // self.world_size
        if not 0 < active_records <= self.max_records:
            raise ValueError(
                f"selected record count {active_records} is outside "
                f"[1, {self.max_records}]"
            )
        if (
            peer_record_base_ptrs.device != self.device
            or peer_record_base_ptrs.dtype != torch.int64
            or not peer_record_base_ptrs.is_contiguous()
            or peer_record_base_ptrs.ndim != 1
            or peer_record_base_ptrs.numel() != self.world_size
        ):
            raise ValueError(
                "peer_record_base_ptrs must be a contiguous int64 vector with "
                "one entry per source rank"
            )
        if (
            out_ptrs.device != self.device
            or out_ptrs.dtype != torch.int64
            or not out_ptrs.is_contiguous()
            or out_ptrs.ndim != 1
            or out_ptrs.numel() != active_records
        ):
            raise ValueError(
                "out_ptrs must be a contiguous int64 vector with one entry "
                "per selected record"
            )
        source_record_bytes = int(source_record_bytes)
        if source_record_bytes <= 0:
            raise ValueError("source_record_bytes must be positive")

        self._check_stream()
        self._require_eager_warmup_for_capture()
        self._wait_for_deferred_release()
        active_primary = self._active_primary_capacity(active_records)
        primary_copy_bytes = _align_up(
            self._layout.primary_records_offset + active_primary * self.record_bytes,
            16,
        )
        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_indices(
                local_indices_by_destination[destination],
                self._local_staging_primary_ptr(destination),
                self._local_staging_overflow_ptr(destination),
                active_primary,
                self._layout.primary_positions_offset,
                self._layout.primary_records_offset,
                self._layout.overflow_positions_offset,
                self._layout.overflow_records_offset,
            )
        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._dma_ext.dma_copy(
                self._receive_primary_ptr(destination, self.rank),
                self._local_staging_primary_ptr(destination),
                primary_copy_bytes,
            )

        self._barrier(0)
        self._ext.build_storage_record_ptrs(
            self._local_receive_primary_ptrs,
            self._peer_staging_overflow_ptrs,
            peer_record_base_ptrs,
            out_ptrs,
            source_record_bytes,
            active_primary,
            self._layout.primary_positions_offset,
            self._layout.primary_records_offset,
            self._layout.overflow_positions_offset,
            self._layout.overflow_records_offset,
        )
        # The output points into model storage rather than this packet.  Packet
        # reuse may therefore be published immediately on the same stream.
        self._publish_deferred_release()
        return out_ptrs

    def consume_record_ptrs(
        self,
        record_ptrs: torch.Tensor,
        checksums: torch.Tensor,
    ) -> torch.Tensor:
        """POC consumer that reads every byte through ``record_ptrs``."""
        if not self._pointer_table_active:
            raise RuntimeError("no selected-record pointer table is active")
        self._ext.consume_record_ptrs(record_ptrs, checksums, self.record_bytes)
        return checksums

    def consume_contiguous_records(
        self,
        records: torch.Tensor,
        checksums: torch.Tensor,
    ) -> torch.Tensor:
        """POC baseline consumer with the same work but no indirection."""
        if records.shape != (checksums.numel(), self.record_bytes):
            raise ValueError(
                f"records must have shape ({checksums.numel()}, {self.record_bytes})"
            )
        self._ext.consume_contiguous_records(records, checksums)
        return checksums

    def release_record_ptrs(self) -> None:
        """Publish packet reuse after all pointer consumers are enqueued."""
        if not self._pointer_table_active:
            raise RuntimeError("no selected-record pointer table is active")
        self._publish_deferred_release()
        self._pointer_table_active = False

    def exchange_layers(
        self,
        records_by_layer: Sequence[torch.Tensor],
        local_indices_by_destination: torch.Tensor,
        outputs_by_layer: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange three layer planes with one shared destination-position table.

        Each selected slot carries three native records in one packet. The
        arrival barrier protects packet visibility. Unpack then publishes a
        release generation without waiting; the matching wait is issued only
        before this channel reuses staging for its next exchange.
        """
        if self._pointer_table_active:
            raise RuntimeError("cannot exchange while a record pointer table is active")
        self._check_stream()
        self._require_eager_warmup_for_capture()
        active_records = self._validate_layers(
            records_by_layer,
            local_indices_by_destination,
            outputs_by_layer,
        )
        outputs = (
            outputs_by_layer[0],
            outputs_by_layer[1],
            outputs_by_layer[2],
        )
        self._wait_for_deferred_release()

        if active_records == 0:
            self._barrier(0)
            for output in outputs:
                output.zero_()
            self._publish_deferred_release()
            return outputs

        active_primary = self._active_layer_primary_capacity(active_records)
        layout = self._layer_layout
        packet_record_bytes = self.record_bytes * _LAYER_PLANES
        primary_copy_bytes = _align_up(
            layout.primary_records_offset + active_primary * packet_record_bytes,
            16,
        )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_record_layers(
                records_by_layer[0],
                records_by_layer[1],
                records_by_layer[2],
                local_indices_by_destination[destination],
                self._local_layer_staging_primary_ptr(destination),
                self._local_layer_staging_overflow_ptr(destination),
                self.record_bytes,
                active_primary,
                layout.primary_positions_offset,
                layout.primary_records_offset,
                layout.overflow_positions_offset,
                layout.overflow_records_offset,
            )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._dma_ext.dma_copy(
                self._layer_receive_primary_ptr(destination, self.rank),
                self._local_layer_staging_primary_ptr(destination),
                primary_copy_bytes,
            )

        self._barrier(0)
        for output in outputs:
            output.zero_()
        self._ext.unpack_compact_record_layers(
            self._local_slab_ptr + layout.receive_primary_base,
            layout.primary_stride,
            self._peer_layer_overflow_ptrs[self.rank],
            outputs[0],
            outputs[1],
            outputs[2],
            active_records,
            self.record_bytes,
            active_primary,
            layout.primary_positions_offset,
            layout.primary_records_offset,
            layout.overflow_positions_offset,
            layout.overflow_records_offset,
        )
        self._publish_deferred_release()
        return outputs

    def _finish_deferred_release_for_close(self) -> None:
        if self._pointer_table_active:
            self.release_record_ptrs()
        if not self._deferred_release_pending:
            return
        if self.device.type == "cuda":
            device_index = (
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
            if self._ipc is not None:
                self._ipc.cudaSetDevice(device_index)
            torch.cuda.synchronize(self.device)
        self._wait_for_deferred_release()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _release_owned_buffer(self, *, synchronize: bool) -> None:
        if self._owned_buffer is None or self._ipc is None:
            return
        if self.device.type == "cuda":
            device_index = (
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
            self._ipc.cudaSetDevice(device_index)
            if synchronize:
                with suppress(Exception):
                    torch.cuda.synchronize(self.device)
        _free_shared_buffer(self._ipc, self._owned_buffer)
        self._owned_buffer = None

    def close(self) -> None:
        if self._closed:
            return
        self._finish_deferred_release_for_close()
        self._closed = True
        self._release_owned_buffer(synchronize=True)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


@dataclass
class _TorchStorageRecordMapping:
    peer_record_base_ptrs: torch.Tensor
    storages: tuple[torch.UntypedStorage, ...]


class PCIeSelectedStoragePointerExchange:
    """Exchange compact ordinals and point directly into peer tensor storage.

    PyTorch's CUDA IPC tuple supplies the allocator base and storage offset for
    tensors living inside caching-allocator segments.  The format-aware PyTorch
    importer is retained for the mapping lifetime so both cudaMalloc and VMM
    expandable-segment allocations remain valid without copying record payloads.
    """

    record_ptrs_require_release = False

    def __init__(
        self,
        *,
        packet_exchange: PCIeSelectedRecordCopyExchange,
        process_group: ProcessGroup,
        source_record_bytes: int,
    ) -> None:
        if packet_exchange.record_bytes != _POSITION_BYTES:
            raise ValueError("storage-pointer metadata packets must be eight bytes")
        if source_record_bytes <= 0:
            raise ValueError("source_record_bytes must be positive")
        self._packet_exchange = packet_exchange
        self.process_group = process_group
        self.rank = packet_exchange.rank
        self.world_size = packet_exchange.world_size
        self.device = packet_exchange.device
        self.max_records = packet_exchange.max_records
        self.record_bytes = int(source_record_bytes)
        if packet_exchange._ipc is None:
            raise ValueError("storage-pointer exchange requires CUDA IPC")
        self._mappings: dict[tuple[int, ...], _TorchStorageRecordMapping] = {}
        self._closed = False

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_records: int,
        source_record_bytes: int,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        ext_module=None,
        dma_ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeSelectedStoragePointerExchange":
        packet_exchange = PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=process_group,
            device=device,
            max_records=max_records,
            record_bytes=_POSITION_BYTES,
            barrier_timeout_cycles=barrier_timeout_cycles,
            ext_module=ext_module,
            dma_ext_module=dma_ext_module,
            stream_affine=stream_affine,
        )
        return cls(
            packet_exchange=packet_exchange,
            process_group=process_group,
            source_record_bytes=source_record_bytes,
        )

    @staticmethod
    def _mapping_key(records: torch.Tensor) -> tuple[int, ...]:
        return (
            int(records.untyped_storage().data_ptr()),
            int(records.data_ptr()),
            int(records.storage_offset()),
            int(records.numel()),
            int(records.element_size()),
        )

    @staticmethod
    def _release_receiver_ref(shared: Sequence[Any]) -> None:
        torch.UntypedStorage._release_ipc_counter(
            shared[4],
            shared[5],
            device=shared[0],
        )

    def _map_records(self, records: torch.Tensor) -> _TorchStorageRecordMapping:
        if self._closed:
            raise RuntimeError("PCIeSelectedStoragePointerExchange is closed")
        if records.device != self.device:
            raise ValueError("records must be on the exchange device")
        if records.dtype != torch.uint8 or not records.is_contiguous():
            raise ValueError("records must be contiguous uint8")
        if records.ndim < 2 or records.shape[-1] != self.record_bytes:
            raise ValueError(
                f"records must end in the configured {self.record_bytes}-byte width"
            )
        key = self._mapping_key(records)
        cached = self._mappings.get(key)
        if cached is not None:
            return cached
        if _is_current_stream_capturing(self.device):
            raise RuntimeError(
                "model-storage CUDA IPC must be initialized by one eager "
                "selected-record call before graph capture"
            )

        storage = records.untyped_storage()
        tensor_byte_offset = int(records.storage_offset()) * int(records.element_size())
        tensor_bytes = int(records.numel()) * int(records.element_size())
        local_exports: list[tuple[tuple[Any, ...], int, int] | None] = []
        try:
            for destination in range(self.world_size):
                local_exports.append(
                    None
                    if destination == self.rank
                    else (storage._share_cuda_(), tensor_byte_offset, tensor_bytes)
                )
        except Exception:
            for exported in local_exports:
                if exported is not None:
                    with suppress(Exception):
                        self._release_receiver_ref(exported[0])
            raise

        gathered: list[list[tuple[tuple[Any, ...], int, int] | None] | None] = [
            None
        ] * self.world_size
        try:
            dist.all_gather_object(
                gathered,
                local_exports,
                group=self.process_group,
            )
        except Exception:
            for exported in local_exports:
                if exported is not None:
                    with suppress(Exception):
                        self._release_receiver_ref(exported[0])
            raise

        remote_exports: list[tuple[int, tuple[Any, ...], int]] = []
        try:
            for source, source_exports in enumerate(gathered):
                if source == self.rank:
                    continue
                if source_exports is None:
                    raise RuntimeError("missing peer CUDA IPC export list")
                exported = source_exports[self.rank]
                if exported is None or len(exported) != 3:
                    raise RuntimeError("missing receiver-specific CUDA IPC export")
                shared, source_tensor_offset, source_tensor_bytes = exported
                if len(shared) != 8:
                    raise RuntimeError("invalid receiver-specific CUDA IPC tuple")
                if (
                    source_tensor_offset < 0
                    or source_tensor_bytes < 0
                    or source_tensor_offset + source_tensor_bytes > int(shared[2])
                ):
                    raise RuntimeError(
                        "peer tensor exceeds its exported CUDA storage bounds"
                    )
                remote_exports.append((source, shared, int(source_tensor_offset)))
        except Exception:
            # all_gather_object completed, so this rank owns one receiver ref
            # from every peer even when validation fails before import.
            for source, source_exports in enumerate(gathered):
                if source == self.rank or source_exports is None:
                    continue
                exported = source_exports[self.rank]
                if exported is None or len(exported) != 3:
                    continue
                with suppress(Exception):
                    self._release_receiver_ref(exported[0])
            raise

        device_index = (
            torch.cuda.current_device()
            if self.device.index is None
            else int(self.device.index)
        )
        base_ptrs = [0] * self.world_size
        base_ptrs[self.rank] = int(records.data_ptr())
        retained_storages = [storage]
        unconsumed = {source: shared for source, shared, _ in remote_exports}
        try:
            for source, shared, source_tensor_offset in remote_exports:
                # _new_shared_cuda normally maps into shared[0] (the producer).
                # Replacing only that field asks PyTorch's allocator-aware IPC
                # importer to map the same storage in this consumer context.
                imported = torch.UntypedStorage._new_shared_cuda(
                    device_index,
                    *shared[1:],
                )
                unconsumed.pop(source)
                retained_storages.append(imported)
                base_ptrs[source] = imported.data_ptr() + source_tensor_offset
        except Exception:
            # Successful imports own and release their receiver refs through the
            # retained storage deleter.  Only exports not consumed by PyTorch's
            # importer still require an explicit decrement here.
            retained_storages.clear()
            for shared in unconsumed.values():
                with suppress(Exception):
                    self._release_receiver_ref(shared)
            raise

        mapping = _TorchStorageRecordMapping(
            peer_record_base_ptrs=torch.tensor(
                base_ptrs,
                dtype=torch.int64,
                device=self.device,
            ),
            storages=tuple(retained_storages),
        )
        self._mappings[key] = mapping
        return mapping

    def prepare_record_ptrs(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out_ptrs: torch.Tensor,
        *,
        primary_mode: str = "storage",
    ) -> torch.Tensor:
        if primary_mode != "storage":
            raise ValueError(
                "storage-pointer transport requires primary_mode='storage'"
            )
        mapping = self._map_records(records)
        return self._packet_exchange.prepare_storage_record_ptrs(
            local_indices_by_destination,
            mapping.peer_record_base_ptrs,
            out_ptrs,
            source_record_bytes=self.record_bytes,
        )

    def consume_record_ptrs(
        self,
        record_ptrs: torch.Tensor,
        checksums: torch.Tensor,
    ) -> torch.Tensor:
        self._packet_exchange._ext.consume_record_ptrs(
            record_ptrs,
            checksums,
            self.record_bytes,
        )
        return checksums

    def reset_storage_mappings(self) -> None:
        """Release mappings tied to a KV cache while keeping the packet ring."""
        if self._closed:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._mappings.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.reset_storage_mappings()
        self._packet_exchange.close()
        self._closed = True

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "PCIeSelectedRecordCopyExchange",
    "PCIeSelectedRecordExchangeInitializationError",
    "PCIeSelectedStoragePointerExchange",
]
