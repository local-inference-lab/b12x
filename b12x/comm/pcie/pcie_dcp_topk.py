"""Exact owner-sharded PCIe transport for DCP sparse top-k."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from ._dcp_cute_common import (
    DCP_TOPK_ABI_VERSION,
    DCP_TOPK_MAX_BLOCKS,
    _FLAG_STRIDE,
    _MAX_RANKS,
    signal_bytes,
)
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _align_up,
    _broadcast_gather_object,
    _coordinated_close_channels,
    _current_stream_key,
    _device_guard,
    _finish_collective_runtime_setup,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_collective_preallocation_setup,
)


SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8)
_MAX_BLOCKS = DCP_TOPK_MAX_BLOCKS
_SIGNAL_BYTES = signal_bytes(_MAX_BLOCKS)
_DCP_TOPK_CONTRACT_VERSION = 1

# Internal factory token: direct CUDA construction with exchange_group is
# only permitted through _from_prepared_factory, which sets this token.
_FACTORY_TOKEN = object()


def _validate_launch_config(*, threads: int, block_limit: int, world_size: int) -> None:
    if threads < world_size or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a multiple of 32 in [32, 512]")
    if block_limit <= 0 or block_limit > _MAX_BLOCKS:
        raise ValueError(f"block_limit must be in [1, {_MAX_BLOCKS}]")


def _host_identity() -> str:
    """Return a kernel-level host identity for topology verification.

    Uses the boot ID when available (unique per host kernel) with a
    hostname fallback for platforms that lack /proc/sys/kernel/random/boot_id.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except (OSError, ValueError):
        import socket

        return socket.gethostname()


def _dcp_topk_topology_record(*, rank: int, device: torch.device) -> tuple:
    """Gather per-rank physical topology identity.

    Returns ``(group_rank, host_id, device_uuid)``.  Uses the CUDA device
    UUID and boot ID for physical identity.
    """
    host_id = _host_identity()
    device_uuid = ""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_uuid = str(getattr(props, "uuid", "")).strip()
        if not device_uuid:
            raise RuntimeError(
                "DCP top-k topology: could not determine CUDA device UUID"
            )
    return (int(rank), host_id, device_uuid)


def _verify_dcp_topk_topology(
    *, exchange_group: ProcessGroup, topology: tuple
) -> None:
    """Collectively verify topology uniqueness and IPC reachability."""
    gathered = _broadcast_gather_object(topology, exchange_group)
    if len(gathered) < 2:
        return
    host_ids = {record[1] for record in gathered}
    if len(host_ids) > 1:
        raise RuntimeError(
            "DCP top-k topology: ranks span multiple hosts; "
            "PCIe IPC requires all ranks on the same host"
        )
    device_uuids = [record[2] for record in gathered]
    if "" in device_uuids:
        raise RuntimeError(
            "DCP top-k topology: one or more ranks could not determine its "
            "CUDA device UUID; cannot verify GPU uniqueness"
        )
    if len(set(device_uuids)) != len(device_uuids):
        raise RuntimeError(
            f"DCP top-k topology: duplicate GPU UUID detected {device_uuids}"
        )
    ranks = [record[0] for record in gathered]
    if ranks != list(range(len(ranks))):
        raise RuntimeError(
            f"DCP top-k topology: group rank ordering is not contiguous: {ranks}"
        )


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


def _dcp_topk_runtime_contract(
    *,
    world_size: int,
    max_rows: int,
    topk: int,
    layout: _DoubleBufferLayout,
    threads: int,
    block_limit: int,
) -> tuple:
    """Build the versioned fixed-schema contract every rank must agree on."""
    max_owner_rows = int(max_rows) // int(world_size)
    candidate_plane_elems = max_owner_rows * int(world_size) * int(topk)
    candidate_shape = (max_owner_rows, int(world_size) * int(topk))
    candidate_stride = (int(world_size) * int(topk), 1)
    score_plane_offset = candidate_plane_elems * 4
    return (
        _DCP_TOPK_CONTRACT_VERSION,
        DCP_TOPK_ABI_VERSION,
        int(world_size),
        SUPPORTED_WORLD_SIZES,
        int(max_rows),
        max_owner_rows,
        int(topk),
        candidate_plane_elems,
        candidate_shape,
        candidate_stride,
        score_plane_offset,
        int(layout.slab_bytes),
        int(layout.slot_bytes),
        int(layout.plane_bytes),
        int(layout.signal_bytes),
        int(layout.staging0_offset),
        int(layout.staging1_offset),
        "int32_indices",
        "float32_scores",
        DCP_TOPK_MAX_BLOCKS,
        int(_SIGNAL_BYTES),
        _FLAG_STRIDE,
        _MAX_RANKS,
        int(threads),
        int(block_limit),
        True,
        IPC_SLAB_ALIGNMENT,
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
        int(pointer), device, int(nbytes),
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
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self._ipc = ipc
        self._owned_buffers = list(owned_buffers or ())
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False

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

    def _close_ipc_imports(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        if self._ipc is not None:
            for shared in self._owned_buffers:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        self._ipc.cudaIpcCloseMemHandle(ptr)
        self._ipc_imports_closed = True

    def _free_ipc_exports(self) -> None:
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

    def close(self) -> None:
        _coordinated_close_channels(
            (self,), exchange_group=self.exchange_group, device=self.device,
        )

    def close_coordinated(self) -> None:
        """Collectively close peer mappings before freeing exported storage."""
        _coordinated_close_channels(
            (self,), exchange_group=self.exchange_group, device=self.device,
        )

    def __del__(self) -> None:
        return None


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
        threads: int = 512,
        block_limit: int = 128,
        _factory_token: object = None,
    ) -> None:
        del ext_module
        device_obj = _normalize_device(device)

        # Reject public CUDA construction BEFORE touching CUDA state so
        # CPU-only CI does not initialise CUDA.
        if device_obj.type == "cuda" and exchange_group is None:
            raise ValueError(
                "exchange_group is required for a CUDA PCIe DCP top-k runtime; "
                "use from_exchange_group()"
            )
        if (
            device_obj.type == "cuda"
            and exchange_group is not None
            and _factory_token is not _FACTORY_TOKEN
        ):
            raise ValueError(
                "grouped CUDA PCIe DCP top-k construction must use "
                "from_exchange_group()"
            )

        # Only the internal prepared path may set the device.
        if device_obj.type == "cuda" and _factory_token is _FACTORY_TOKEN:
            torch.cuda.set_device(device_obj)

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
        _validate_launch_config(
            threads=int(threads), block_limit=int(block_limit), world_size=world_size,
        )

        if exchange_group is not None:
            group_rank = dist.get_rank(group=exchange_group)
            group_world_size = dist.get_world_size(group=exchange_group)
            if int(rank) != group_rank:
                raise ValueError(
                    f"supplied rank {rank} does not match process group rank "
                    f"{group_rank}"
                )
            if int(world_size) != group_world_size:
                raise ValueError(
                    f"supplied world size {world_size} does not match process "
                    f"group size {group_world_size}"
                )

        self._init_channel(
            device=device,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            stream_affine=True,
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_rows = int(max_rows)
        self.max_owner_rows = self.max_rows // self.world_size
        self.topk = int(topk)
        self._threads = int(threads)
        self._block_limit = int(block_limit)
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
        self._capture_context_depth = 0
        self._capture_rows: Optional[int] = None
        self._capture_stage_count: int = 0
        candidate_shape = (self.max_owner_rows, self.world_size * self.topk)
        if self.device.type == "cuda":
            self._candidate_views = tuple(
                (
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank], candidate_shape,
                        dtype=torch.int32, device=self.device,
                    ),
                    _tensor_from_cuda_pointer(
                        slot_ptrs[self.rank] + self._candidate_plane_elems * 4,
                        candidate_shape, dtype=torch.float32, device=self.device,
                    ),
                )
                for slot_ptrs in self._staging_ptrs
            )
        else:
            self._candidate_views = ()

    @classmethod
    def _from_prepared_factory(cls, **kwargs) -> "PCIeDCPTopKOwnerExchange":
        """Construct a runtime via the internal factory token."""
        kwargs["_factory_token"] = _FACTORY_TOKEN
        runtime = object.__new__(cls)
        runtime.__init__(**kwargs)
        return runtime

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        topk: int,
        ext_module=None,
        threads: int = 512,
        block_limit: int = 128,
    ) -> "PCIeDCPTopKOwnerExchange":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            if device_obj.type != "cuda":
                raise ValueError("DCP top-k owner exchange requires a CUDA device")
            if device_obj.type == "cuda":
                torch.cuda.set_device(device_obj)
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(f"unsupported world size {world_size}")
            n_max_rows = int(max_rows)
            n_topk = int(topk)
            n_threads = int(threads)
            n_block_limit = int(block_limit)
            if n_max_rows <= 0 or n_max_rows % world_size != 0:
                raise ValueError(
                    "max_rows must be positive and divisible by world_size"
                )
            if n_topk <= 0 or n_topk % 4 != 0:
                raise ValueError("topk must be a positive multiple of 4")
            _validate_launch_config(
                threads=n_threads, block_limit=n_block_limit, world_size=world_size,
            )
            topology = _dcp_topk_topology_record(rank=rank, device=device_obj)
            return device_obj, n_max_rows, n_topk, n_threads, n_block_limit, topology

        (
            device_obj, n_max_rows, n_topk, n_threads, n_block_limit, topology,
        ) = _run_collective_preallocation_setup(
            owner="PCIe DCP top-k argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )

        _require_full_grid_residency(
            owner="PCIe DCP top-k", required_sms=_MAX_BLOCKS,
            device=device_obj, exchange_group=exchange_group,
        )

        _verify_dcp_topk_topology(
            exchange_group=exchange_group, topology=topology,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(device_obj.index or 0)
            prepared_layout = _candidate_staging_layout(
                signal_bytes=_SIGNAL_BYTES, max_rows=n_max_rows,
                topk=n_topk, world_size=world_size,
            )
            from ._dcp_topk_cute import prepare_topk_stage
            prepare_topk_stage(world_size, rank, n_topk, n_threads)
            # Build the contract inside coordinated setup so a one-rank
            # contract-builder/schema exception is exchanged collectively
            # rather than exiting locally and stranding peers.
            prepared_contract = _dcp_topk_runtime_contract(
                world_size=world_size, max_rows=n_max_rows, topk=n_topk,
                layout=prepared_layout, threads=n_threads, block_limit=n_block_limit,
            )
            return prepared_ipc, prepared_layout, prepared_contract

        ipc, layout, contract = _run_collective_preallocation_setup(
            owner="PCIe DCP top-k", exchange_group=exchange_group, setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe DCP top-k channel layout",
            exchange_group=exchange_group,
            contract=contract,
        )
        slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group, layout.slab_bytes, zero_fill=True, ipc=ipc,
        )

        runtime: Optional[PCIeDCPTopKOwnerExchange] = None
        init_error: BaseException | None = None
        try:
            runtime = cls._from_prepared_factory(
                rank=rank, world_size=world_size, device=device_obj,
                signal_ptrs=slab.peer_ptrs,
                staging0_ptrs=tuple(
                    ptr + layout.staging0_offset for ptr in slab.peer_ptrs
                ),
                staging1_ptrs=tuple(
                    ptr + layout.staging1_offset for ptr in slab.peer_ptrs
                ),
                max_rows=n_max_rows, topk=n_topk,
                exchange_group=exchange_group, ipc=ipc,
                owned_buffers=[slab], ext_module=ext_module,
                threads=n_threads, block_limit=n_block_limit,
            )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            if runtime is not None:
                runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe DCP top-k", exchange_group=exchange_group,
            ipc=ipc, shared=slab, local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )
        assert runtime is not None
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
        threads: int = 512,
        block_limit: int = 128,
    ) -> "PCIeDCPTopKOwnerExchange":
        return cls.from_exchange_group(
            exchange_group=process_group, device=device,
            max_rows=max_rows, topk=topk, ext_module=ext_module,
            threads=threads, block_limit=block_limit,
        )

    def stage_candidates(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage exact candidates and return channel-owned owner views.

        The launch protocol (threads, block_limit) is fixed at construction
        and collectively agreed.  For eager (non-capture) calls, a collective
        pre-launch verdict exchanges (rows, blocks, slot, wait) so a
        divergent or malformed rank fails every rank before any peer kernel
        launches.  During CUDA graph capture, rows/slot/wait were
        pre-agreed in :meth:`capture` so no host collective occurs inside
        the captured path.
        """
        with _device_guard(self.device):
            return self._stage_candidates_on_device(local_indices, local_scores)

    def prepare_graph(self) -> None:
        """Compile the exact graph launcher before capture begins."""
        if self._closed:
            raise RuntimeError("PCIeDCPTopKOwnerExchange is closed")
        if _is_current_stream_capturing(self.device):
            raise RuntimeError(
                "prepare_graph() must be called before CUDA graph capture"
            )
        with _device_guard(self.device):
            self._bind_stream()
            from ._dcp_topk_cute import is_topk_stage_prepared
            if not is_topk_stage_prepared(
                self.world_size, self.rank, self.topk, int(self._threads),
            ):
                from ._dcp_topk_cute import prepare_topk_stage
                prepare_topk_stage(
                    self.world_size, self.rank, self.topk, int(self._threads),
                )

    @contextmanager
    def capture(self, *, rows: int):
        """Own one serialized graph capture without adding graph nodes.

        Before yielding, collectively agrees (rows, blocks, slot, wait) so
        the captured stage_candidates needs no host collective.  After the
        caller's ``torch.cuda.graph`` context ends (on context exit),
        exchanges capture success and stage-call count so no peer replays
        a partial graph or a graph with mismatched barrier generations.

        Every pre-capture operation is wrapped in a non-throwing envelope
        so a rank with a local failure participates in the collective
        agreement and every rank rejects coherently.

        The entire capture lifecycle is wrapped in ``_device_guard`` so
        NCCL collectives run on the owner's device, not the caller's
        current device.

        Enter this context before ``torch.cuda.graph``::

            with owner.capture(rows=128), torch.cuda.graph(graph):
                ...
        """
        with _device_guard(self.device):
            # Wrap every pre-capture operation in a non-throwing envelope.
            local_error: BaseException | None = None
            n_rows = 0
            blocks = 0
            proposed_slot = 0
            wait_for_prior_consumer = True
            try:
                if self._capture_context_depth:
                    raise RuntimeError(
                        "overlapping DCP top-k capture contexts are not allowed"
                    )
                n_rows = int(rows)
                if n_rows <= 0 or n_rows > self.max_rows or n_rows % self.world_size != 0:
                    raise ValueError(
                        f"rows must be in (0, {self.max_rows}] and divisible by world_size"
                    )
                self.prepare_graph()
                owner_packs = (n_rows // self.world_size) * (self.topk // 4)
                blocks = max(
                    1,
                    min(
                        int(self._block_limit),
                        (owner_packs + self._threads - 1) // self._threads,
                    ),
                )
                # Compute proposed slot WITHOUT mutating _graph_slot.
                proposed_slot = (
                    self._graph_slot if self._graph_slot is not None else self._next_slot
                )
            except BaseException as exc:
                local_error = exc

            # Collective pre-capture agreement: every rank exchanges
            # (has_error, rows, blocks, proposed_slot, wait).  No rank
            # commits capture state until the verdict is unanimous.
            capture_contract = (
                local_error is not None,
                n_rows,
                blocks,
                proposed_slot,
                wait_for_prior_consumer,
            )
            if self.exchange_group is not None:
                gathered = _broadcast_gather_object(
                    capture_contract, self.exchange_group
                )
                if any(peer != capture_contract for peer in gathered):
                    raise RuntimeError(
                        f"DCP top-k capture contract differs across ranks: {gathered}"
                    )
            if local_error is not None:
                raise local_error

            # Only commit capture state after unanimous agreement.
            self._graph_slot = proposed_slot
            self._capture_rows = n_rows
            self._capture_context_depth = 1
            self._capture_stage_count = 0
            capture_body_error: BaseException | None = None
            try:
                yield self
            except BaseException as exc:
                capture_body_error = exc
                raise
            finally:
                self._capture_context_depth = 0
                self._capture_rows = None
                stage_count = self._capture_stage_count
                self._capture_stage_count = 0
                if self.exchange_group is not None:
                    # Exchange (success, error_type, stage_count).
                    # All ranks must capture the same number of stage
                    # kernels (preferably exactly 1) so replay has
                    # matching barrier generations.
                    status = (
                        capture_body_error is None,
                        type(capture_body_error).__name__ if capture_body_error else "",
                        stage_count,
                    )
                    gathered = _broadcast_gather_object(
                        status, self.exchange_group
                    )
                    if any(peer[0] is not True for peer in gathered):
                        raise RuntimeError(
                            f"DCP top-k capture failed on one or more ranks: {gathered}"
                        )
                    counts = {peer[2] for peer in gathered}
                    if len(counts) > 1:
                        raise RuntimeError(
                            f"DCP top-k capture stage count differs across ranks: {gathered}"
                        )
                    if stage_count != 1:
                        raise RuntimeError(
                            f"DCP top-k capture requires exactly 1 stage call, "
                            f"got {stage_count}"
                        )

    def _eager_prelaunch_verdict(
        self,
        *,
        rows: int,
        blocks: int,
        slot: int,
        wait_for_prior_consumer: bool,
        local_error: Optional[Exception],
    ) -> None:
        """Collectively verify all ranks agree on eager launch parameters."""
        if self.exchange_group is None:
            if local_error is not None:
                raise local_error
            return
        prelaunch_contract = (
            local_error is not None,
            int(rows), int(blocks), int(slot),
            bool(wait_for_prior_consumer),
        )
        gathered = _broadcast_gather_object(
            prelaunch_contract, self.exchange_group
        )
        if any(peer != prelaunch_contract for peer in gathered):
            raise RuntimeError(
                f"DCP top-k pre-launch verdict differs across ranks: {gathered}"
            )
        if local_error is not None:
            raise local_error

    def _validate_stage_inputs(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[Optional[Exception], int]:
        """Run all per-call validation, returning (error, rows).

        Everything that can fail — including ndim access and stream binding —
        is inside the try block so a malformed rank can participate in the
        collective verdict.
        """
        rows = 0
        try:
            if self._closed:
                raise RuntimeError("PCIeDCPTopKOwnerExchange is closed")
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
                raise ValueError(
                    f"topk {topk} does not match configured topk {self.topk}"
                )
            if not local_indices.is_contiguous() or not local_scores.is_contiguous():
                raise ValueError("inputs must be contiguous")
            if int(local_indices.data_ptr()) % 16 != 0:
                raise ValueError("local_indices data pointer must be 16-byte aligned")
            if int(local_scores.data_ptr()) % 16 != 0:
                raise ValueError("local_scores data pointer must be 16-byte aligned")
            self._bind_stream()
            return None, rows
        except Exception as exc:
            return exc, rows

    def _stage_candidates_on_device(
        self,
        local_indices: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        capturing = _is_current_stream_capturing(self.device)
        if capturing:
            if self._capture_context_depth <= 0:
                raise RuntimeError(
                    "DCP top-k CUDA graph capture requires an active "
                    "owner.capture() context"
                )
            # During capture, rows/slot/wait were pre-agreed in capture().
            # Validate locally — including that tensor rows match the
            # pre-agreed capture rows — then launch with no host collective.
            local_error, tensor_rows = self._validate_stage_inputs(
                local_indices, local_scores
            )
            if local_error is not None:
                raise local_error
            rows = self._capture_rows
            assert rows is not None
            if tensor_rows != rows:
                raise ValueError(
                    f"captured stage tensor rows {tensor_rows} do not match "
                    f"pre-agreed capture rows {rows}"
                )
            slot = self._graph_slot
            assert slot is not None
            wait_for_prior_consumer = True
            owner_packs = (rows // self.world_size) * (self.topk // 4)
            blocks = max(
                1,
                min(
                    int(self._block_limit),
                    (owner_packs + self._threads - 1) // self._threads,
                ),
            )
        else:
            # Eager path: validate without raising (including ndim/stream),
            # then collective verdict.
            local_error, rows = self._validate_stage_inputs(
                local_indices, local_scores
            )
            owner_packs = (rows // self.world_size) * (self.topk // 4)
            blocks = max(
                1,
                min(
                    int(self._block_limit),
                    (owner_packs + self._threads - 1) // self._threads,
                ),
            )
            # After graph capture, all launches use the pinned graph slot
            # with wait=True to prevent overwriting before a slow consumer
            # retires.
            if self._graph_slot is not None:
                slot = self._graph_slot
                wait_for_prior_consumer = True
            else:
                slot = self._next_slot
                self._next_slot ^= 1
                wait_for_prior_consumer = False
            self._eager_prelaunch_verdict(
                rows=rows, blocks=blocks, slot=slot,
                wait_for_prior_consumer=wait_for_prior_consumer,
                local_error=local_error,
            )

        self._launch_stage(
            local_indices, local_scores, slot=slot, rows=rows,
            threads=int(self._threads), blocks=blocks,
            wait_for_prior_consumer=wait_for_prior_consumer,
        )
        if capturing:
            self._capture_stage_count += 1
        candidate_views = self._candidate_views[slot] if self._candidate_views else ()
        if not candidate_views:
            raise RuntimeError("DCP top-k candidate views require a CUDA device")
        owner_rows = rows // self.world_size
        candidate_indices = candidate_views[0][:owner_rows]
        candidate_scores = candidate_views[1][:owner_rows]
        expected = (rows // self.world_size, self.world_size * self.topk)
        _validate_candidate_view(
            candidate_indices, expected=expected,
            dtype=torch.int32, device=self.device, label="index",
        )
        _validate_candidate_view(
            candidate_scores, expected=expected,
            dtype=torch.float32, device=self.device, label="score",
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
                world_size=self.world_size, rank=self.rank, topk=self.topk,
                threads=threads,
                local_indices_ptr=local_indices.data_ptr(),
                local_scores_ptr=local_scores.data_ptr(),
                candidate_ptrs=self._staging_ptrs[slot],
                signal_ptrs=self._signal_ptrs,
                rows=rows, candidate_plane_elems=self._candidate_plane_elems,
                blocks=blocks, wait_for_prior_consumer=wait_for_prior_consumer,
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
