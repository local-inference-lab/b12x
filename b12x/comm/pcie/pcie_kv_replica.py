"""Serial, byte-exact prefill replica of owner-striped compressed KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from b12x.preparation import require_prepared

from ._cuda_ipc import CudaRTLibrary
from ._dcp_cute_common import signal_bytes
from .pcie_dcp_topk import _IPCChannel
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _align_up,
    _normalize_device,
    _require_collective_contract,
    _run_collective_preallocation_setup,
)


@dataclass(frozen=True)
class _ReplicaBinding:
    """Validated tensors and fixed layout for one paged-replica execution."""

    runtime: object
    state: object
    cache: torch.Tensor
    table: torch.Tensor
    positions: torch.Tensor
    starts: torch.Tensor
    out: torch.Tensor
    request_capacity: int
    output_pages_per_request: int


class PCIePagedKvReplica(_IPCChannel):
    """One stream-ordered channel; no independent overlapping graph replay."""

    @classmethod
    def from_process_group(
        cls, *, process_group, device, max_requests, max_tokens, stripe_alignment=128
    ):
        rank, world = dist.get_rank(process_group), dist.get_world_size(process_group)

        def validate():
            device_obj = _normalize_device(device)
            if world != 4 or device_obj.type != "cuda":
                raise ValueError("paged KV replica requires four CUDA ranks")
            if max_requests <= 0 or max_tokens <= 0 or stripe_alignment <= 0:
                raise ValueError(
                    "replica capacities and stripe alignment must be positive"
                )
            local = _align_up((max_tokens + world - 1) // world, stripe_alignment)
            offset = _align_up(signal_bytes(1), IPC_SLAB_ALIGNMENT)
            return device_obj, local, offset, offset + max_requests * local * 288

        device_obj, local, offset, slab_bytes = _run_collective_preallocation_setup(
            owner="paged KV replica",
            exchange_group=process_group,
            setup=validate,
        )
        _require_collective_contract(
            owner="paged KV replica",
            exchange_group=process_group,
            contract=(max_requests, max_tokens, stripe_alignment, local, slab_bytes),
        )
        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index)
        slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            process_group,
            slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        self = cls()
        self._init_channel(
            device=device_obj,
            exchange_group=process_group,
            ipc=ipc,
            owned_buffers=(slab,),
            # Serial startup and profiling may use different logical streams.
            # _bind_stream below preserves ordering across those handoffs.
            stream_affine=False,
        )
        self.rank, self.world_size = rank, world
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.local_capacity, self.slab_bytes = local, slab_bytes
        self.signal_ptrs = tuple(slab.peer_ptrs)
        self.staging_ptrs = tuple(ptr + offset for ptr in slab.peer_ptrs)
        self._serial_stream = None
        return self

    def _bind_stream(self):
        # Capture uses a torch-owned stream, rather than the logical stream
        # warmed by the caller. Graphs must replay serially on that logical
        # stream; independent overlapping replay remains unsupported.
        if torch.cuda.is_current_stream_capturing():
            return
        stream = torch.cuda.current_stream(self.device)
        previous = self._serial_stream
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            # Records an event at the previous stream's current tail, including
            # attention consumers enqueued after its last replica copy. This is
            # a device dependency, not a global or serving-time host synchronize.
            stream.wait_stream(previous)
        self._serial_stream = stream

    def bind(self, cache, table, positions, starts, out, *, plan):
        """Validate and retain one caller-owned tensor binding before execution."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        return self._bind_prepared(state, cache, table, positions, starts, out)

    def replicate(self, binding, *, plan, requests, max_tokens):
        """Launch a prepared binding for the live request and token counts."""
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        self._run_prepared(
            state,
            binding,
            requests=requests,
            max_tokens=max_tokens,
        )

    def _bind_prepared(self, state, cache, table, positions, starts, out):
        """Check fixed tensor ABI once and return its retained execution binding."""
        if self._closed:
            raise RuntimeError("paged KV replica channel is closed")
        page = state.query.call["page_size"]
        capacity = state.query.call["max_tokens"]
        stripe = state.query.call["stripe"]
        width = (capacity + page - 1) // page
        if (
            cache.dtype != torch.uint8
            or out.dtype != torch.uint8
            or cache.ndim not in (2, 3)
            or cache.shape[1:].numel() != page * 288
            or out.numel() != (self.max_requests * width + 1) * page * 288
        ):
            raise ValueError(
                "replica cache/output must use exact DeepSeek V4.1 indexed byte pages"
            )
        if table.ndim != 2 or table.shape[0] < 1:
            raise ValueError("replica requires a request-major page table")
        local = _align_up((capacity + self.world_size - 1) // self.world_size, stripe)
        required_columns = (local + page - 1) // page
        if table.shape[1] < required_columns:
            raise ValueError(
                "replica page table does not cover the declared local capacity"
            )
        for tensor in (cache, table, positions, starts, out):
            if tensor.device != self.device or (
                tensor is not cache and not tensor.is_contiguous()
            ):
                raise ValueError(
                    "replica tensors must be contiguous on the channel device"
                )
        if (
            cache.stride(-1) != 1
            or cache.data_ptr() % 16
            or out.data_ptr() % 16
            or cache.stride(0) < page * 288
            or cache.stride(0) % 16
            or (cache.ndim == 3 and cache.stride(1) != 288)
        ):
            raise ValueError(
                "replica requires packed records within aligned byte pages"
            )
        for tensor in (table, starts):
            if tensor.dtype != torch.int32:
                raise ValueError("replica page and request metadata must use int32")
        if (
            positions.dtype != torch.int64
            or positions.numel() < 1
            or starts.numel() < 2
        ):
            raise ValueError(
                "replica request metadata must describe at least one request"
            )
        request_capacity = min(
            self.max_requests,
            table.shape[0],
            positions.numel(),
            starts.numel() - 1,
        )
        return _ReplicaBinding(
            runtime=self,
            state=state,
            cache=cache,
            table=table,
            positions=positions,
            starts=starts,
            out=out,
            request_capacity=request_capacity,
            output_pages_per_request=width,
        )

    def _run_prepared(self, state, binding, *, requests, max_tokens):
        from ._kv_replica_cute import run_kv_replica

        if binding.runtime is not self or binding.state is not state:
            raise ValueError("replica binding belongs to a different prepared runtime")
        self._bind_stream()
        capacity = state.query.call["max_tokens"]
        if (
            not 0 < requests <= binding.request_capacity
            or not 0 < max_tokens <= capacity
        ):
            raise ValueError(
                "live replica request/token counts exceed prepared capacity"
            )
        run_kv_replica(
            state.launcher(),
            self,
            binding.cache,
            binding.table,
            binding.positions,
            binding.starts,
            binding.out,
            requests=requests,
            max_tokens=max_tokens,
            output_pages_per_request=binding.output_pages_per_request,
        )
