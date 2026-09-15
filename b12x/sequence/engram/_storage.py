"""Owning Engram table storage with explicit device or host placement."""

from dataclasses import dataclass

import torch

from b12x._lib.platform import probe_platform
from .._shared.disk_table import MappedHostAllocation


@dataclass
class TableStorage:
    plan: object
    memory: str
    weight: torch.Tensor
    scales: torch.Tensor
    weight_load_view: torch.Tensor
    scales_load_view: torch.Tensor
    _allocations: tuple
    closed: bool = False

    def stats(self):
        nbytes = (
            self.weight.numel() * self.weight.element_size()
            + self.scales.numel() * self.scales.element_size()
        )
        return {
            "placement": self.memory,
            "table_bytes": nbytes,
            "mapped_host_bytes": nbytes if self._allocations else 0,
            "hbm_cache_bytes": 0,
            "logical_bytes_per_lookup": 264,
            "logical_bytes_per_token": 24 * 264,
        }

    def close(self):
        """Release storage after its bindings and captured graphs are retired."""
        if not self.closed:
            for allocation in reversed(self._allocations):
                allocation.close()
            self.closed = True


def allocate_storage(plan, *, memory="device"):
    """Allocate checkpoint loading views and retain their CUDA-visible owners.

    ``grace`` requires live ATS and host-atomic capability probes. It uses
    CUDA-mapped host allocations without assuming a memory-bandwidth result.
    The returned owner must outlive every lookup binding and CUDA graph.
    """
    from .api import Plan

    if not isinstance(plan, Plan):
        raise TypeError("plan must be an Engram Plan")
    from b12x.preparation.types import require_prepared
    state = require_prepared(plan, "sequence.engram")
    if state.operation != "lookup":
        raise ValueError("table storage requires a prepared Engram lookup plan")
    if state.compact_rows:
        raise ValueError("table storage requires an uncompressed lookup declaration")
    if memory not in {"device", "mapped_host", "grace"}:
        raise ValueError("Engram memory must be device, mapped_host, or grace")
    if memory == "grace" and not probe_platform(state.caps.device).grace_coherent:
        raise NotImplementedError(
            "Grace Engram placement requires coherent SM103 CPU/GPU memory"
        )
    allocations = []
    try:

        def allocate(shape, dtype):
            if memory == "device":
                tensor = torch.empty(shape, dtype=dtype, device=state.caps.device)
                return tensor, tensor
            allocation = MappedHostAllocation(shape, dtype, state.caps.device)
            allocations.append(allocation)
            return allocation.device_view, allocation.host_view

        weight, weight_load = allocate(state.weight_shape, torch.float8_e4m3fn)
        scales, scales_load = allocate(state.scale_shape, torch.uint8)
    except Exception:
        for allocation in reversed(allocations):
            allocation.close()
        raise
    return TableStorage(
        plan, memory, weight, scales, weight_load, scales_load, tuple(allocations)
    )
