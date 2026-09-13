"""Plan-time architecture selection for the public fused-MoE contract."""

from b12x._lib.architecture import architecture_for


def architecture_backend(device_identity):
    if device_identity is None:
        return None
    architecture = architecture_for(device_identity.compute_capability)
    if architecture is None or architecture.mma_family == "warp":
        return None
    if not architecture.implemented:
        return None
    from . import _sm103

    return _sm103
