"""Persistent PLE table storage allocation."""

from __future__ import annotations

import math
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from cuda.bindings import runtime as cudart

if TYPE_CHECKING:
    from ._contracts import Plan


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host PLE storage requires "
            "torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage,
        0,
        shape,
        _contiguous_strides(shape),
    )


class _MappedHostAllocation:
    """Own one mapped page-locked allocation and its two tensor aliases."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host PLE storage requires an indexed CUDA device, got {device}"
            )
        element_size = int(torch.empty((), dtype=dtype).element_size())
        nbytes = math.prod(shape) * element_size
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )

        self.device = device
        self.nbytes = nbytes
        self._host_pointer = 0
        self._closed = False

        with torch.cuda.device(device):
            error, host_pointer = cudart.cudaHostAlloc(
                nbytes,
                cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined,
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(host_pointer)
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(host_pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(host_pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        """Release the allocation after all GPU access has completed."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        error = cudart.cudaFreeHost(self._host_pointer)[0]
        _check_cuda(error, "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if self._closed or self._host_pointer == 0 or sys.is_finalizing():
            return
        with suppress(Exception):
            self.close()


@dataclass(kw_only=True)
class TableStorage:
    """Owning persistent table tensors and their checkpoint loading views.

    Kernel-visible tensors are always CUDA tensors. For mapped-host table
    storage, a loading view is a CPU tensor over the same page-locked bytes.
    The owner must outlive every binding that references its tensors.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    weight_load_view: torch.Tensor
    weight_scale_load_view: torch.Tensor | None
    weight_scale_2_load_view: torch.Tensor | None
    mapped_host_nbytes: int
    _mapped_allocations: tuple[_MappedHostAllocation, ...]

    def close(self) -> None:
        """Synchronize and release any mapped-host allocations."""
        for allocation in reversed(self._mapped_allocations):
            allocation.close()


def _device_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def allocate_storage(plan: Plan) -> TableStorage:
    """Allocate persistent table storage according to the planned policy."""
    from ._contracts import Plan

    if not isinstance(plan, Plan):
        raise TypeError(f"plan must be Plan, got {type(plan)!r}")
    caps = plan.caps

    allocations: list[_MappedHostAllocation] = []

    def table_tensor(
        shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if caps.table_memory == "device":
            tensor = _device_tensor(shape, dtype, caps.device)
            return tensor, tensor
        allocation = _MappedHostAllocation(shape, dtype, caps.device)
        allocations.append(allocation)
        return allocation.device_view, allocation.host_view

    weight, weight_load_view = table_tensor(plan.weight_shape, plan.weight_dtype)

    weight_scale: torch.Tensor | None = None
    weight_scale_load_view: torch.Tensor | None = None
    if plan.weight_scale_shape is not None:
        assert plan.weight_scale_dtype is not None
        if caps.quant_mode == "nvfp4_group16":
            weight_scale, weight_scale_load_view = table_tensor(
                plan.weight_scale_shape, plan.weight_scale_dtype
            )
        else:
            weight_scale = _device_tensor(
                plan.weight_scale_shape, plan.weight_scale_dtype, caps.device
            )
            weight_scale_load_view = weight_scale

    weight_scale_2: torch.Tensor | None = None
    weight_scale_2_load_view: torch.Tensor | None = None
    if plan.weight_scale_2_shape is not None:
        assert plan.weight_scale_2_dtype is not None
        weight_scale_2 = _device_tensor(
            plan.weight_scale_2_shape, plan.weight_scale_2_dtype, caps.device
        )
        weight_scale_2_load_view = weight_scale_2

    return TableStorage(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        weight_load_view=weight_load_view,
        weight_scale_load_view=weight_scale_load_view,
        weight_scale_2_load_view=weight_scale_2_load_view,
        mapped_host_nbytes=sum(allocation.nbytes for allocation in allocations),
        _mapped_allocations=tuple(allocations),
    )


__all__ = ["TableStorage", "allocate_storage"]
