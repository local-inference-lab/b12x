"""Bounded-degree hierarchical TP12/TP16 all-reduce runtime.

This runtime is specialized for single-node topologies with three or four
contiguous four-GPU PCIe islands. Unlike the ordinary oneshot collective, it
does not map every rank into every CUDA context: non-leaders map one peer and
island leaders map five peers at TP12 or six at TP16. The collective is
CUDA-graph capturable and stages arbitrary BF16 inputs into fixed IPC storage
before reducing them.

Security: every field that a rank derives locally and later interprets
against remote IPC pointers is collectively agreed before allocation.  A
divergent rank fails coherently before any IPC mapping.
"""

from __future__ import annotations

import os
import socket
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from ._hierarchical_cute import (
    _COLLECTIVE_GENERATION as _NATIVE_COLLECTIVE_GENERATION,
    _FINAL_READY as _NATIVE_FINAL_READY,
    _ISLAND_SIZE as _NATIVE_ISLAND_SIZE,
    _LEADER_CONSUMED as _NATIVE_LEADER_CONSUMED,
    _LEADER_READY as _NATIVE_LEADER_READY,
    _LOCAL_ARRIVED as _NATIVE_LOCAL_ARRIVED,
    _LOCAL_CONSUMED as _NATIVE_LOCAL_CONSUMED,
    _MAX_ISLANDS as _NATIVE_MAX_ISLANDS,
    _MAX_WORLD_SIZE as _NATIVE_MAX_WORLD_SIZE,
    get_hierarchical_launcher,
)
from .pcie_oneshot import (
    _abort_collective_ipc_setup,
    _attach_retryable_setup,
    _broadcast_gather_object,
    _exchange_setup_failures,
    _normalize_device,
    _require_collective_contract,
    _require_no_retained_ipc_setup,
    _retain_failed_ipc_setup,
    _setup_failure_message,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISLAND_SIZE = 4
SUPPORTED_WORLD_SIZES = (12, 16)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)
_ALIGNMENT = 256
_HEADER_BYTES = 69_888

_HIERARCHICAL_CONTRACT_VERSION = 5
_HIERARCHICAL_ABI_VERSION = 1

# IPC handle wire size from ctypes.
from ctypes import sizeof as _ctypes_sizeof  # noqa: E402

from ._cuda_ipc import cudaIpcMemHandle_t  # noqa: E402

_IPC_HANDLE_BYTES = _ctypes_sizeof(cudaIpcMemHandle_t)

# Region dtypes/widths.
_STAGE_DTYPE = "bf16"
_STAGE_WIDTH = 2
_PARTIAL_DTYPE = "fp32"
_PARTIAL_WIDTH = 4
_FINAL_DTYPE = "bf16"
_FINAL_WIDTH = 2

_BF16X2_THREADS = 112
_PICK_BLOCKS_THRESHOLD = 4096
_NATIVE_COMPILER_KEY = "comm.pcie.hierarchical.bf16"
_NATIVE_COMPILER_VERSION = 3
_OWNER = "PCIe hierarchical all-reduce"

_STATE_OPEN = "open"
_STATE_CLOSING = "closing"
_STATE_CLOSED = "closed"

# Quarantine for abandoned runtimes (matches PCIeOneshotAllReduce pattern).
_ABANDONED_HIERARCHICAL_QUARANTINE: dict[int, object] = {}


# ---------------------------------------------------------------------------
# Environment parsing (strict validation)
# ---------------------------------------------------------------------------

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
    if elements <= 0:
        raise ValueError("elements must be positive")
    return 16 if elements <= _PICK_BLOCKS_THRESHOLD else 32


def _buffer_modes_from_env() -> tuple[bool, bool]:
    raw_db = os.getenv("B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER", "0")
    if raw_db not in ("0", "1"):
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER must be 0 or 1"
        )
    double_buffered = raw_db == "1"

    raw_dc = os.getenv(
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION",
        "0",
    )
    if raw_dc not in ("0", "1"):
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION must be 0 or 1"
        )
    deferred_consumption = raw_dc == "1"

    if double_buffered and deferred_consumption:
        raise ValueError(
            "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER and "
            "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION are "
            "mutually exclusive"
        )
    return double_buffered, deferred_consumption


# ---------------------------------------------------------------------------
# Immutable operation catalog (mandatory, slots-based, collectively agreed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OpCatalogEntry:
    """One frozen operation in the collectively agreed catalog."""

    op_id: str
    order: int
    elements: int
    blocks: int

    def __post_init__(self) -> None:
        if int(self.elements) <= 0:
            raise ValueError("elements must be positive")
        if int(self.blocks) not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if int(self.blocks) > _HEADER_BLOCK_CAPACITY():
            raise ValueError(
                f"blocks={self.blocks} exceeds header flag capacity "
                f"{_HEADER_BLOCK_CAPACITY()}"
            )

    def to_tuple(self) -> tuple:
        return (self.op_id, int(self.order), int(self.elements), int(self.blocks))


def _HEADER_BLOCK_CAPACITY() -> int:
    """Max blocks fitting in the header flag arrays."""
    return 32


@dataclass(frozen=True, slots=True)
class OpCatalog:
    """Frozen catalog of all operations agreed collectively at construction.

    Every all_reduce call must match an entry by (elements, blocks).
    The entire catalog is included in the all-rank contract tuple.
    """

    entries: tuple[OpCatalogEntry, ...]
    double_buffered: bool = False
    deferred_consumption: bool = False

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("catalog must have at least one entry")
        seen_ids: set[str] = set()
        seen_orders: set[int] = set()
        for entry in self.entries:
            if entry.op_id in seen_ids:
                raise ValueError(f"duplicate op_id {entry.op_id}")
            if entry.order in seen_orders:
                raise ValueError(f"duplicate order {entry.order}")
            seen_ids.add(entry.op_id)
            seen_orders.add(entry.order)
        if self.double_buffered and self.deferred_consumption:
            raise ValueError("double_buffered and deferred_consumption are mutually exclusive")

    def to_tuple(self) -> tuple:
        return (
            tuple(e.to_tuple() for e in self.entries),
            bool(self.double_buffered),
            bool(self.deferred_consumption),
        )

    def max_elements(self) -> int:
        return max(e.elements for e in self.entries)

    def find(self, elements: int, blocks: int) -> Optional[OpCatalogEntry]:
        for entry in self.entries:
            if entry.elements == elements and entry.blocks == blocks:
                return entry
        return None


def _make_default_catalog(max_elements: int, blocks: Optional[int] = None) -> OpCatalog:
    """Build the default single-operation catalog for backward compatibility."""
    resolved_blocks = int(blocks) if blocks is not None else _pick_blocks(max_elements)
    if resolved_blocks not in SUPPORTED_BLOCKS:
        raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
    return OpCatalog(
        entries=(OpCatalogEntry(op_id="default", order=0, elements=int(max_elements), blocks=resolved_blocks),),
    )


# ---------------------------------------------------------------------------
# ABI descriptor
# ---------------------------------------------------------------------------

def _hierarchical_abi_descriptor() -> tuple:
    return (
        _HIERARCHICAL_ABI_VERSION,
        int(_NATIVE_ISLAND_SIZE),
        int(_NATIVE_MAX_ISLANDS),
        int(_NATIVE_MAX_WORLD_SIZE),
        ISLAND_SIZE,
        _HEADER_BYTES,
        _ALIGNMENT,
        int(_NATIVE_COLLECTIVE_GENERATION),
        int(_NATIVE_LOCAL_ARRIVED),
        int(_NATIVE_LEADER_READY),
        int(_NATIVE_LEADER_CONSUMED),
        int(_NATIVE_FINAL_READY),
        int(_NATIVE_LOCAL_CONSUMED),
        128,  # flag stride
        4,    # generation stride
        _HEADER_BLOCK_CAPACITY(),
        _IPC_HANDLE_BYTES,
        _STAGE_DTYPE,
        _STAGE_WIDTH,
        _PARTIAL_DTYPE,
        _PARTIAL_WIDTH,
        _FINAL_DTYPE,
        _FINAL_WIDTH,
        _BF16X2_THREADS,
        SUPPORTED_WORLD_SIZES,
        SUPPORTED_BLOCKS,
        _PICK_BLOCKS_THRESHOLD,
        _NATIVE_COMPILER_KEY,
        _NATIVE_COMPILER_VERSION,
    )


# ---------------------------------------------------------------------------
# Physical topology manifest
# ---------------------------------------------------------------------------

def _get_gpu_identity(device: torch.device) -> tuple[str, str]:
    """Get process-independent GPU UUID and PCI address.

    Returns (uuid, pci_address) from torch.cuda.get_device_properties,
    matching the pattern used by overlap_probe.py.
    """
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, "uuid", f"fallback-{device.index}"))
    pci_address = (
        f"{getattr(props, 'pci_domain_id', 0):08x}:"
        f"{getattr(props, 'pci_bus_id', 0):02x}:"
        f"{getattr(props, 'pci_device_id', 0):02x}.0"
    )
    return uuid, pci_address


def _rank_topology_record(
    rank: int,
    world_size: int,
    device: torch.device,
    group_ranks: list[int],
) -> tuple:
    """Build a rank-indexed physical topology record.

    Includes global rank, host, GPU UUID, PCI address, device ordinal,
    island, local_rank, leader, and exact ordered peer tuple.
    """
    island = rank // ISLAND_SIZE
    local_rank = rank % ISLAND_SIZE
    leader = island * ISLAND_SIZE
    peers = _selected_peers(rank, world_size)
    device_index = int(device.index) if device.index is not None else 0
    global_rank = group_ranks[rank] if rank < len(group_ranks) else rank
    host = socket.gethostname()
    gpu_uuid, pci_address = _get_gpu_identity(device)
    return (
        int(rank),
        int(global_rank),
        host,
        gpu_uuid,
        pci_address,
        device_index,
        island,
        local_rank,
        leader,
        peers,
    )


def _validate_topology_manifest(
    gathered: list, world_size: int,
) -> None:
    """Validate the complete physical topology manifest."""
    if len(gathered) != world_size:
        raise RuntimeError(
            f"{_OWNER} topology manifest has {len(gathered)} records, "
            f"expected {world_size}"
        )
    seen_uuids: set[str] = set()
    seen_global_ranks: set[int] = set()
    seen_pci: set[str] = set()
    hosts: set[str] = set()
    for idx, record in enumerate(gathered):
        if not isinstance(record, tuple) or len(record) != 10:
            raise RuntimeError(
                f"{_OWNER} invalid topology record from rank {idx}"
            )
        (rec_rank, rec_global, rec_host, rec_uuid, rec_pci, rec_dev,
         rec_island, rec_local, rec_leader, rec_peers) = record
        if rec_rank != idx:
            raise RuntimeError(
                f"{_OWNER} topology rank mismatch at index {idx}: "
                f"rank={rec_rank}"
            )
        if rec_global in seen_global_ranks:
            raise RuntimeError(
                f"{_OWNER} duplicate global rank {rec_global} at rank {idx}"
            )
        seen_global_ranks.add(rec_global)
        hosts.add(rec_host)
        if rec_uuid in seen_uuids:
            raise RuntimeError(
                f"{_OWNER} duplicate GPU UUID {rec_uuid} at rank {idx}"
            )
        seen_uuids.add(rec_uuid)
        if rec_pci in seen_pci:
            raise RuntimeError(
                f"{_OWNER} duplicate PCI address {rec_pci} at rank {idx}"
            )
        seen_pci.add(rec_pci)
        expected_peers = _selected_peers(idx, world_size)
        if tuple(rec_peers) != expected_peers:
            raise RuntimeError(
                f"{_OWNER} topology peer mismatch at rank {idx}: "
                f"{rec_peers} != {expected_peers}"
            )
        expected_island = idx // ISLAND_SIZE
        expected_local = idx % ISLAND_SIZE
        expected_leader = expected_island * ISLAND_SIZE
        if rec_island != expected_island:
            raise RuntimeError(
                f"{_OWNER} topology island mismatch at rank {idx}"
            )
        if rec_local != expected_local:
            raise RuntimeError(
                f"{_OWNER} topology local_rank mismatch at rank {idx}"
            )
        if rec_leader != expected_leader:
            raise RuntimeError(
                f"{_OWNER} topology leader mismatch at rank {idx}"
            )
    if len(hosts) > 1:
        raise RuntimeError(
            f"{_OWNER} requires single-node topology, got hosts {hosts}"
        )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def _hierarchical_allreduce_contract(
    *,
    world_size: int,
    layout: _SlabLayout,
    catalog: OpCatalog,
    wait_nanosleep_cycles: int,
    threads: int,
    vectorized_bf16x2: bool,
    vectorized_bf16x2_max_elements: int,
) -> tuple:
    """Build the versioned fixed-schema contract tuple.

    Includes the mandatory catalog, ABI descriptor, peer graph, layout,
    and all protocol fields.  No mutable launch fields.
    """
    peer_graph = tuple(
        _selected_peers(r, world_size) for r in range(world_size)
    )
    max_elements = catalog.max_elements()
    return (
        _HIERARCHICAL_CONTRACT_VERSION,
        _hierarchical_abi_descriptor(),
        int(world_size),
        peer_graph,
        int(max_elements),
        layout.stage,
        layout.partial,
        layout.final,
        int(layout.bytes),
        catalog.to_tuple(),
        "bf16",
        vectorized_bf16x2,
        int(vectorized_bf16x2_max_elements),
        bool(catalog.double_buffered),
        bool(catalog.deferred_consumption),
        int(wait_nanosleep_cycles),
        int(threads),
    )


# ---------------------------------------------------------------------------
# Phase-tagged control-plane status exchange
# ---------------------------------------------------------------------------

_CONTROL_PROTOCOL_VERSION = 1


def _phase_status_exchange(
    *,
    exchange_group: ProcessGroup,
    phase: str,
    attempt: int,
    local_error: BaseException | None,
    exports_retained_on_failure: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """Phase-tagged status exchange with protocol version and attempt number.

    Wraps _exchange_setup_failures with a phase tag so the payload carries
    protocol version, attempt, and phase identity — preventing cross-wire
    between phases after a partial collective failure.
    """
    tagged_error: BaseException | None = local_error
    if local_error is not None:
        # The phase tag is embedded in the error text for diagnostics.
        pass
    try:
        statuses = _exchange_setup_failures(
            tagged_error,
            exchange_group=exchange_group,
            phase=f"{_OWNER} {phase} (v{_CONTROL_PROTOCOL_VERSION} attempt={attempt})",
            exports_retained_on_exchange_failure=exports_retained_on_failure,
        )
    except Exception:
        # The status exchange itself failed — this means we cannot determine
        # whether peers are in the same phase.  We must retain ownership and
        # raise with a retry ticket if any allocation has occurred.
        raise
    return statuses


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

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
        catalog: Optional[OpCatalog] = None,
    ) -> None:
        del ext_module
        self.group = exchange_group
        self._state = _STATE_OPEN
        self._close_lock = threading.Lock()
        self._close_condition = threading.Condition(self._close_lock)
        self._close_generation = 0
        self._closed_remotely = False
        self._attempt = 0

        # ---- Phase 0: Retained-setup gate ----
        # Prevent a new generation from starting before the old one is resolved.
        _require_no_retained_ipc_setup(exchange_group)

        # ---- Phase 1: Coordinated preflight ----
        # All local parsing inside the error envelope.
        local_error: BaseException | None = None
        try:
            self.rank = dist.get_rank(group=exchange_group)
            self.world_size = dist.get_world_size(group=exchange_group)
            self.device = _normalize_device(device)
            if self.world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(
                    "hierarchical all-reduce requires "
                    f"{SUPPORTED_WORLD_SIZES}, got TP{self.world_size}"
                )
            if self.device.type != "cuda":
                raise ValueError(
                    "hierarchical all-reduce requires a CUDA device"
                )
            if max_elements <= 0:
                raise ValueError("max_elements must be positive")

            # Build or validate the mandatory catalog.
            if catalog is not None:
                self._catalog = catalog
            else:
                self._catalog = _make_default_catalog(max_elements, blocks)

            # Derive capacity from catalog max, not from a mutable field.
            self.max_elements = self._catalog.max_elements()
            if max_elements < self.max_elements:
                raise ValueError(
                    f"max_elements={max_elements} is less than catalog "
                    f"max {self.max_elements}"
                )

            (
                self.double_buffered,
                _deferred_consumption,
            ) = _buffer_modes_from_env()
            # Catalog modes take precedence.
            self._catalog = OpCatalog(
                entries=self._catalog.entries,
                double_buffered=self.double_buffered,
                deferred_consumption=_deferred_consumption,
            )
            self.deferred_consumption = _deferred_consumption
            self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()
            self.threads = _threads_from_env()
            self.vectorized_bf16x2 = _vectorized_bf16x2_from_env()
            self.vectorized_bf16x2_max_elements = (
                _vectorized_bf16x2_max_elements_from_env()
            )
            self._layout = _make_layout(self.max_elements)

            assert int(_NATIVE_ISLAND_SIZE) == ISLAND_SIZE, (
                f"ISLAND_SIZE drift: Python={ISLAND_SIZE}, "
                f"native={int(_NATIVE_ISLAND_SIZE)}"
            )

            self._ipc = CudaRTLibrary()
            self._ipc.cudaSetDevice(self.device.index or 0)
        except Exception as exc:
            local_error = exc

        self._slab_ptrs: tuple[int, ...] = ()
        self._launchers: dict[bool, object] = {}
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed_imports: set[int] = set()

        preflight_statuses = _phase_status_exchange(
            exchange_group=exchange_group,
            phase="preflight",
            attempt=self._attempt,
            local_error=local_error,
            exports_retained_on_failure=False,
        )
        if any(preflight_statuses):
            raise RuntimeError(
                _setup_failure_message(
                    _OWNER,
                    "preflight",
                    preflight_statuses,
                    exports_retained=False,
                )
            ) from local_error

        # ---- Phase 2: Physical topology manifest ----
        try:
            group_ranks = self._get_group_ranks(exchange_group)
        except Exception:
            group_ranks = list(range(self.world_size))

        topology_record = _rank_topology_record(
            self.rank, self.world_size, self.device, group_ranks,
        )
        gathered_topology = _broadcast_gather_object(
            topology_record, exchange_group,
        )
        _validate_topology_manifest(gathered_topology, self.world_size)

        # ---- Phase 3: Collective contract agreement ----
        _require_collective_contract(
            owner=_OWNER,
            exchange_group=exchange_group,
            contract=_hierarchical_allreduce_contract(
                world_size=self.world_size,
                layout=self._layout,
                catalog=self._catalog,
                wait_nanosleep_cycles=self.wait_nanosleep_cycles,
                threads=self.threads,
                vectorized_bf16x2=self.vectorized_bf16x2,
                vectorized_bf16x2_max_elements=(
                    self.vectorized_bf16x2_max_elements
                ),
            ),
        )

        # ---- Phase 4: Compile before any IPC export ----
        compile_error: BaseException | None = None
        try:
            with torch.cuda.device(self.device):
                for vectorized in (
                    {False, True} if self.vectorized_bf16x2 else {False}
                ):
                    self._launchers[vectorized] = get_hierarchical_launcher(
                        self.world_size,
                        self.rank,
                        self.device.index or 0,
                        threads=(
                            _BF16X2_THREADS if vectorized else self.threads
                        ),
                        wait_nanosleep_cycles=self.wait_nanosleep_cycles,
                        double_buffered=self.double_buffered,
                        deferred_consumption=self.deferred_consumption,
                        vectorized_bf16x2=vectorized,
                    )
        except Exception as exc:
            compile_error = exc

        compile_statuses = _phase_status_exchange(
            exchange_group=exchange_group,
            phase="launcher compilation",
            attempt=self._attempt,
            local_error=compile_error,
            exports_retained_on_failure=False,
        )
        if any(compile_statuses):
            raise RuntimeError(
                _setup_failure_message(
                    _OWNER,
                    "launcher compilation",
                    compile_statuses,
                    exports_retained=False,
                )
            ) from compile_error

        # ---- Phase 5: Allocate + export ----
        slab_bytes = self._layout.bytes
        alloc_error: BaseException | None = None
        local_handle: object = None
        try:
            self._local_ptr = self._ipc.cudaMalloc(slab_bytes)
            self._ipc.cudaMemset(self._local_ptr, 0, slab_bytes)
            local_handle = self._ipc.cudaIpcGetMemHandleBytes(self._local_ptr)
        except Exception as exc:
            alloc_error = exc

        try:
            alloc_statuses = _phase_status_exchange(
                exchange_group=exchange_group,
                phase="allocation+export",
                attempt=self._attempt,
                local_error=alloc_error,
                exports_retained_on_failure=True,  # allocation may have succeeded
            )
        except Exception as exchange_error:
            # Status exchange itself failed after allocation — retain.
            self._retain_and_raise(
                exchange_group, "allocation+export status exchange",
                exchange_error, alloc_error,
            )

        if any(alloc_statuses):
            _abort_collective_ipc_setup(
                owner=_OWNER,
                setup_phase="allocation+export",
                setup_statuses=alloc_statuses,
                exchange_group=exchange_group,
                ipc=self._ipc,
                local_ptr=self._local_ptr,
                remote_ptrs=[],
                local_error=alloc_error,
            )

        # ---- Phase 6: Handle exchange + validation (single combined phase) ----
        exchange_error: BaseException | None = None
        handles: list[object] = []
        try:
            handles = _broadcast_gather_object(local_handle, exchange_group)
            if len(handles) != self.world_size:
                raise RuntimeError(
                    f"expected {self.world_size} handles, "
                    f"got {len(handles)}"
                )
            for i, handle in enumerate(handles):
                if handle is None:
                    raise RuntimeError(f"missing IPC handle from rank {i}")
                if not isinstance(handle, (bytes, bytearray)):
                    raise RuntimeError(
                        f"invalid IPC handle type from rank {i}"
                    )
                if len(handle) != _IPC_HANDLE_BYTES:
                    raise RuntimeError(
                        f"IPC handle from rank {i} is {len(handle)} bytes, "
                        f"expected {_IPC_HANDLE_BYTES}"
                    )
        except Exception as exc:
            exchange_error = exc

        try:
            exchange_statuses = _phase_status_exchange(
                exchange_group=exchange_group,
                phase="handle exchange",
                attempt=self._attempt,
                local_error=exchange_error,
                exports_retained_on_failure=True,
            )
        except Exception as exchange_exc:
            self._retain_and_raise(
                exchange_group, "handle exchange status exchange",
                exchange_exc, exchange_error,
            )

        if any(exchange_statuses):
            _abort_collective_ipc_setup(
                owner=_OWNER,
                setup_phase="handle exchange",
                setup_statuses=exchange_statuses,
                exchange_group=exchange_group,
                ipc=self._ipc,
                local_ptr=self._local_ptr,
                remote_ptrs=self._remote_ptrs,
                local_error=exchange_error,
            )

        # ---- Phase 7: Import ----
        peer_ptrs = [0] * self.world_size
        import_error: BaseException | None = None
        try:
            peer_ptrs[self.rank] = self._local_ptr
            for peer in _selected_peers(self.rank, self.world_size):
                remote_ptr = self._ipc.cudaIpcOpenMemHandleBytes(
                    handles[peer],
                )
                peer_ptrs[peer] = remote_ptr
                self._remote_ptrs.append(remote_ptr)
        except Exception as exc:
            import_error = exc

        try:
            import_statuses = _phase_status_exchange(
                exchange_group=exchange_group,
                phase="IPC import",
                attempt=self._attempt,
                local_error=import_error,
                exports_retained_on_failure=True,
            )
        except Exception as exchange_exc:
            self._retain_and_raise(
                exchange_group, "IPC import status exchange",
                exchange_exc, import_error,
            )

        if any(import_statuses):
            _abort_collective_ipc_setup(
                owner=_OWNER,
                setup_phase="IPC import",
                setup_statuses=import_statuses,
                exchange_group=exchange_group,
                ipc=self._ipc,
                local_ptr=self._local_ptr,
                remote_ptrs=self._remote_ptrs,
                local_error=import_error,
            )

        # ---- Phase 8: Readiness (status exchange IS the rendezvous) ----
        self._slab_ptrs = tuple(peer_ptrs)
        ready_error: BaseException | None = None
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                ready_error = exc

        try:
            ready_statuses = _phase_status_exchange(
                exchange_group=exchange_group,
                phase="readiness",
                attempt=self._attempt,
                local_error=ready_error,
                exports_retained_on_failure=True,
            )
        except Exception as exchange_exc:
            self._retain_and_raise(
                exchange_group, "readiness status exchange",
                exchange_exc, ready_error,
            )

        if any(ready_statuses):
            _abort_collective_ipc_setup(
                owner=_OWNER,
                setup_phase="readiness",
                setup_statuses=ready_statuses,
                exchange_group=exchange_group,
                ipc=self._ipc,
                local_ptr=self._local_ptr,
                remote_ptrs=self._remote_ptrs,
                local_error=ready_error,
            )

    def _get_group_ranks(self, group: ProcessGroup) -> list[int]:
        """Get group-rank-to-global-rank mapping."""
        if hasattr(dist, "get_process_group_ranks"):
            ranks = list(dist.get_process_group_ranks(group=group))
            if len(ranks) == self.world_size:
                return ranks
        if hasattr(dist, "get_global_rank"):
            return [
                dist.get_global_rank(group, gr)
                for gr in range(self.world_size)
            ]
        return list(range(self.world_size))

    def _retain_and_raise(
        self,
        exchange_group: ProcessGroup,
        phase: str,
        exchange_error: BaseException,
        original_error: BaseException | None,
    ) -> None:
        """Create a durable retry ticket for any status-exchange failure."""
        retained = _retain_failed_ipc_setup(
            ipc=self._ipc,
            local_ptr=self._local_ptr,
            exchange_group=exchange_group,
            owner=_OWNER,
            phase=phase,
            remote_ptrs=self._remote_ptrs,
            state="unmap",
        )
        error = RuntimeError(
            _setup_failure_message(
                _OWNER,
                phase,
                ((),) * self.world_size,
                exports_retained=True,
            )
        )
        raise _attach_retryable_setup(
            error, retained,
        ) from (original_error or exchange_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    @property
    def catalog(self) -> OpCatalog:
        return self._catalog

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return (
            self._state == _STATE_OPEN
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
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
        del blocks  # removed — schedule is frozen in the catalog

        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy hierarchical all-reduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype}, "
                f"device={inp.device})"
            )

        # Find the matching catalog entry by elements.
        # If no catalog entry matches, reject — the schedule is frozen.
        resolved_blocks = _pick_blocks(inp.numel())
        entry = self._catalog.find(inp.numel(), resolved_blocks)
        if entry is None:
            raise ValueError(
                f"input has {inp.numel()} elements with blocks="
                f"{resolved_blocks}, but no catalog entry matches; "
                f"catalog entries: {[(e.elements, e.blocks) for e in self._catalog.entries]}"
            )
        selected_blocks = entry.blocks

        if self._catalog.double_buffered and selected_blocks != _pick_blocks(inp.numel()):
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

        # Serialize state-check-plus-enqueue against close transition.
        with self._close_lock:
            if self._state != _STATE_OPEN:
                raise RuntimeError(
                    "all_reduce called after close initiated"
                )
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

    def for_stream(self, stream: object = None) -> "PCIeHierarchicalAllReduce":
        del stream
        return self
    def close(self) -> None:
        """Strictly release IPC resources with collective phase ordering.

        Fully serialized: exactly one close() caller per rank participates
        in a generation.  Concurrent callers wait on the condition for
        the result.  If a prior close failed (leaving CLOSING), a
        subsequent call retries the idempotent local progress.

        Once any unmap starts, remain CLOSING and reject every launch.
        Never restore OPEN with partially released resources.
        """

        with self._close_lock:
            if self._state == _STATE_CLOSED:
                return
            if self._state == _STATE_CLOSING:
                # A prior close attempt is in progress or failed.
                # If it's active (another thread), wait for its result.
                # If it failed (no active caller), fall through to retry.
                if self._close_active:
                    while self._state == _STATE_CLOSING and self._close_active:
                        self._close_condition.wait(timeout=30)
                    if self._state == _STATE_CLOSED:
                        return
                    # Prior close failed — fall through to retry.
            self._state = _STATE_CLOSING
            self._close_active = True
            self._close_generation += 1
            gen = self._close_generation

        try:
            self._do_coordinated_close(gen)
        finally:
            with self._close_lock:
                self._close_active = False
                self._close_condition.notify_all()

    def _do_coordinated_close(self, gen: int) -> None:
        # Phase 1: synchronize.
        sync_error: BaseException | None = None
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                sync_error = exc

        try:
            sync_statuses = _phase_status_exchange(
                exchange_group=self.group,
                phase=f"close synchronize gen={gen}",
                attempt=self._attempt,
                local_error=sync_error,
                exports_retained_on_failure=True,
            )
        except Exception:
            with self._close_lock:
                self._state = _STATE_CLOSING  # stay CLOSING, don't reopen
            raise

        if any(sync_statuses):
            with self._close_lock:
                self._state = _STATE_CLOSING  # stay CLOSING
            raise RuntimeError(
                _setup_failure_message(
                    _OWNER,
                    f"close synchronize gen={gen}",
                    sync_statuses,
                    exports_retained=True,
                )
            )

        # Phase 2: close imports.
        unmap_failures: list[str] = []
        for ptr in self._remote_ptrs:
            if ptr in self._closed_imports:
                continue
            try:
                self._ipc.cudaIpcCloseMemHandle(ptr)
                self._closed_imports.add(ptr)
            except Exception as exc:
                unmap_failures.append(
                    f"import {ptr}: {type(exc).__name__}: {exc}"
                )

        # Keep only imports that failed to close.
        self._remote_ptrs = [
            p for p in self._remote_ptrs if p not in self._closed_imports
        ]

        try:
            unmap_statuses = _phase_status_exchange(
                exchange_group=self.group,
                phase=f"close unmap gen={gen}",
                attempt=self._attempt,
                local_error=RuntimeError(" | ".join(unmap_failures))
                if unmap_failures
                else None,
                exports_retained_on_failure=True,
            )
        except Exception:
            with self._close_lock:
                self._state = _STATE_CLOSING
            raise

        if any(unmap_statuses):
            with self._close_lock:
                self._state = _STATE_CLOSING  # stay CLOSING, never reopen
            raise RuntimeError(
                _setup_failure_message(
                    _OWNER,
                    f"close unmap gen={gen}",
                    unmap_statuses,
                    exports_retained=True,
                )
            )

        self._slab_ptrs = ()
        self._launchers.clear()

        # Phase 3: free exports.
        free_error: BaseException | None = None
        live_ptr = self._local_ptr
        if self._local_ptr:
            try:
                self._ipc.cudaFree(self._local_ptr)
            except Exception as exc:
                free_error = exc
            else:
                self._local_ptr = 0
                live_ptr = 0

        retained = live_ptr != 0
        try:
            free_statuses = _phase_status_exchange(
                exchange_group=self.group,
                phase=f"close export free gen={gen}",
                attempt=self._attempt,
                local_error=free_error,
                exports_retained_on_failure=retained,
            )
        except Exception:
            with self._close_lock:
                self._state = _STATE_CLOSING
            raise

        if any(free_statuses):
            with self._close_lock:
                self._state = _STATE_CLOSING
            raise RuntimeError(
                _setup_failure_message(
                    _OWNER,
                    f"close export free gen={gen}",
                    free_statuses,
                    exports_retained=retained,
                )
            )

        # All phases succeeded.
        with self._close_lock:
            self._state = _STATE_CLOSED

    def __enter__(self) -> "PCIeHierarchicalAllReduce":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        # Quarantine an abandoned live runtime so GC cannot release
        # IPC resources from asymmetric interpreter teardown.  Only
        # quarantine if we still truly own resources — a constructor
        # that raised after _abort_collective_ipc_setup already
        # transferred ownership to a retry ticket.
        if (
            not hasattr(self, "_state")
            or self._state == _STATE_CLOSED
            or not (
                getattr(self, "_local_ptr", 0)
                or getattr(self, "_remote_ptrs", None)
            )
        ):
            return
        _ABANDONED_HIERARCHICAL_QUARANTINE[id(self)] = self
