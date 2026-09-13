"""Platform selection for RoCEnante's mapped-memory protocol.

The Grace arm reuses the pinned-region proxy ABI and TP2 peer exchange. HBM
registration and GPUDirect ordering require a separate transport implementation.
"""

from dataclasses import dataclass

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x._lib.platform import PlatformCapabilities, probe_platform


@dataclass(frozen=True)
class TransportSelection:
    backend: str
    memory: str
    experimental: bool
    qualification: str


def select_transport(
    capabilities: PlatformCapabilities,
    *,
    world_size: int,
    requested: str = "auto",
    experimental: bool = False,
) -> TransportSelection:
    if requested not in {"auto", "spark_mapped", "grace_mapped", "hbm_gdr"}:
        raise ValueError(f"unknown RoCEnante transport {requested!r}")
    if requested == "hbm_gdr":
        raise UnsupportedArchitectureError(
            "RoCEnante HBM GPUDirect transport is not implemented; DMA-BUF registration "
            "and NIC-to-GPU visibility must be qualified before enabling it"
        )
    if capabilities.compute_capability == (12, 1) and capabilities.integrated:
        if requested in {"auto", "spark_mapped"}:
            return TransportSelection(
                "spark_mapped", "mapped_host", False, "existing Spark transport"
            )
    if capabilities.grace_coherent and requested == "grace_mapped" and experimental:
        if world_size != 2:
            raise UnsupportedArchitectureError(
                "experimental Grace transport supports TP2 only"
            )
        return TransportSelection(
            "grace_mapped", "mapped_host", True, "awaiting B300 runtime validation"
        )
    raise UnsupportedArchitectureError(
        "no qualified RoCEnante transport for this platform; Grace TP2 requires "
        "coherency probes, requested='grace_mapped', and experimental=True"
    )


__all__ = [
    "TransportSelection",
    "select_transport",
    "probe_platform",
    "PlatformCapabilities",
]
