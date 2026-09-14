"""Serial, byte-exact prefill replica of owner-striped compressed KV."""

from __future__ import annotations

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
                raise ValueError("paged KV replica currently requires four CUDA ranks")
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
            stream_affine=True,
        )
        self.rank, self.world_size = rank, world
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.local_capacity, self.slab_bytes = local, slab_bytes
        self.signal_ptrs = tuple(slab.peer_ptrs)
        self.staging_ptrs = tuple(ptr + offset for ptr in slab.peer_ptrs)
        return self

    def replicate(
        self, cache, table, positions, starts, out, *, plan, requests, max_tokens
    ):
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        self._run_prepared(
            state,
            cache,
            table,
            positions,
            starts,
            out,
            requests=requests,
            max_tokens=max_tokens,
        )

    def _run_prepared(
        self, state, cache, table, positions, starts, out, *, requests, max_tokens
    ):
        from ._kv_replica_cute import run_kv_replica

        self._bind_stream()
        if self._closed:
            raise RuntimeError("paged KV replica channel is closed")
        page = state.query.call["page_size"]
        capacity = state.query.call["max_tokens"]
        stripe = state.query.call["stripe"]
        width = (capacity + page - 1) // page
        if not 0 < requests <= self.max_requests or not 0 < max_tokens <= capacity:
            raise ValueError(
                "live replica request/token counts exceed prepared capacity"
            )
        if (
            cache.dtype != torch.uint8
            or out.dtype != torch.uint8
            or cache.ndim not in (2, 3)
            or cache.shape[1:].numel() != page * 288
            or out.numel() != (self.max_requests * width + 1) * page * 288
        ):
            raise ValueError(
                "replica cache/output must use exact DS4.1 indexed byte pages"
            )
        if table.ndim != 2 or table.shape[0] < requests:
            raise ValueError("replica requires a request-major page table")
        local = _align_up((capacity + self.world_size - 1) // self.world_size, stripe)
        required_columns = (local + page - 1) // page
        if table.shape[1] < required_columns:
            raise ValueError(
                "replica page table does not cover the declared local capacity"
            )
        for tensor in (cache, table, positions, starts, out):
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(
                    "replica tensors must be contiguous on the channel device"
                )
        for tensor in (table, starts):
            if tensor.dtype != torch.int32:
                raise ValueError("replica page and request metadata must use int32")
        if (
            positions.dtype != torch.int64
            or positions.numel() < requests
            or starts.numel() < requests + 1
        ):
            raise ValueError(
                "replica request metadata is shorter than the live request count"
            )
        run_kv_replica(
            state.launcher(),
            self,
            cache,
            table,
            positions,
            starts,
            out,
            requests=requests,
            max_tokens=max_tokens,
            output_pages_per_request=width,
        )
