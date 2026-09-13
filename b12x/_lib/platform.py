"""Live CUDA platform probes for host-memory transports.

Compute capability alone never proves Grace coherency or RDMA registration.
Unknown or failed CUDA attribute queries are represented as false.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformCapabilities:
    compute_capability: tuple[int, int]
    integrated: bool
    host_native_atomics: bool
    pageable_memory_access: bool
    uses_host_page_tables: bool
    cpu_architecture: str

    @property
    def grace_coherent(self) -> bool:
        return (
            self.compute_capability == (10, 3)
            and self.cpu_architecture in {"aarch64", "arm64"}
            and self.host_native_atomics
            and self.pageable_memory_access
            and self.uses_host_page_tables
        )


def probe_platform(device=None) -> PlatformCapabilities:
    import platform
    import torch
    from cuda.bindings import runtime as cudart

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a platform probe")
    dev = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device or "cuda")
    )
    if dev.type != "cuda":
        raise ValueError("platform probe requires a CUDA device")
    index = torch.cuda.current_device() if dev.index is None else dev.index
    props = torch.cuda.get_device_properties(index)

    def attribute(name):
        attr = getattr(cudart.cudaDeviceAttr, name, None)
        if attr is None:
            return False
        try:
            error, value = cudart.cudaDeviceGetAttribute(attr, index)
            return error == cudart.cudaError_t.cudaSuccess and bool(value)
        except RuntimeError:
            return False

    return PlatformCapabilities(
        compute_capability=(props.major, props.minor),
        integrated=bool(getattr(props, "is_integrated", False)),
        host_native_atomics=attribute("cudaDevAttrHostNativeAtomicSupported"),
        pageable_memory_access=attribute("cudaDevAttrPageableMemoryAccess"),
        uses_host_page_tables=attribute(
            "cudaDevAttrPageableMemoryAccessUsesHostPageTables"
        ),
        cpu_architecture=platform.machine().lower(),
    )
