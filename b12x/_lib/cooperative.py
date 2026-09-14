"""Residency bounds for cooperative CuTe launches."""

from __future__ import annotations

from typing import Any


def _resident_blocks_per_sm(compiled: Any, block_threads: int) -> int:
    from cuda.bindings import driver

    from .compiler import _extract_launch_dynamic_smem_bytes

    def checked(result):
        if result[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA cooperative residency query failed: {result[0]}")
        return result[1]

    kernel_info = compiled.kernel_info
    if len(kernel_info) != 1:
        raise ValueError("Cooperative residency requires exactly one CUDA entry point")
    symbol = next(iter(kernel_info))
    executor = compiled.to(None)
    module = executor.jit_module
    if hasattr(module, "cuda_library"):
        libraries = module.cuda_library
        if len(libraries) != 1:
            raise ValueError("Cooperative residency requires one CUDA library")
        kernel = checked(
            driver.cuLibraryGetKernel(
                driver.CUlibrary(int(libraries[0])), symbol.encode()
            )
        )
    else:
        kernel = module.cuda_modules[0].kernel
    function = checked(driver.cuKernelGetFunction(kernel))

    def attribute(name):
        return int(
            checked(
                driver.cuFuncGetAttribute(
                    getattr(driver.CUfunction_attribute, name), function
                )
            )
        )

    metadata = getattr(compiled, "_b12x_launch_metadata", None)
    if metadata is None:
        metadata = _extract_launch_dynamic_smem_bytes(compiled)
        compiled._b12x_launch_metadata = metadata
    if metadata.get("status") == "exact":
        sizes = metadata["launch_dynamic_smem_bytes"].get(symbol)
        if not sizes or any(type(size) is not int or size < 0 for size in sizes):
            raise ValueError("Invalid cooperative launch shared-memory metadata")
        shared_bytes = max(sizes)
    else:
        # Older cached objects lack launcher IR. The configured SMEM maximum
        # gives a conservative bound until that object is compiled again.
        shared_bytes = attribute("CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES")
    resident = int(
        checked(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                function, block_threads, shared_bytes
            )
        )
    )
    if resident <= 0:
        raise RuntimeError("Compiled cooperative kernel cannot occupy one CUDA SM")
    return resident


def cooperative_grid_limit(compiled: Any, block_threads: int) -> int:
    """Prepare a device-specific residency bound before CUDA graph capture."""
    import torch

    device = torch.cuda.current_device()
    key = (device, block_threads)
    limits = getattr(compiled, "_b12x_cooperative_grid_limits", None)
    if limits is not None and key in limits:
        return limits[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Cooperative launch residency must be warmed before capture")
    resident = _resident_blocks_per_sm(compiled, block_threads)
    limit = resident * torch.cuda.get_device_properties(device).multi_processor_count
    if limits is None:
        limits = {}
        compiled._b12x_cooperative_grid_limits = limits
    limits[key] = limit
    return limit
