"""CUDA architecture contracts, independent of device and compiler imports.

Resource limits describe the ISA target. Launch planning must also check the
physical device's opt-in shared-memory limit. Recognition does not qualify an
operator or imply coherent host memory, which is a platform property.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedArchitectureError(NotImplementedError):
    """The requested operator has no implementation for this architecture."""


@dataclass(frozen=True)
class Architecture:
    compute_capability: tuple[int, int]
    name: str
    compilation_target: str
    mma_family: str
    tmem_columns: int
    max_smem_per_block: int
    block_scaled_formats: frozenset[str]
    implemented: bool = True

    @property
    def has_tmem(self) -> bool:
        return self.tmem_columns > 0


_FORMATS = frozenset({"nvfp4", "mxfp4", "mxfp8"})
_ARCHITECTURES = {
    (10, 0): Architecture(
        (10, 0), "sm100", "sm_100a", "tcgen05", 512, 227 * 1024, _FORMATS, False
    ),
    (10, 3): Architecture(
        (10, 3), "sm103", "sm_103a", "tcgen05", 512, 227 * 1024, _FORMATS
    ),
    (12, 0): Architecture((12, 0), "sm120", "sm_120a", "warp", 0, 99 * 1024, _FORMATS),
    (12, 1): Architecture((12, 1), "sm121", "sm_121a", "warp", 0, 99 * 1024, _FORMATS),
}


def architecture_for(capability: tuple[int, int] | None) -> Architecture | None:
    return _ARCHITECTURES.get(capability)


def supports_architecture(
    capability: tuple[int, int] | None, archs: tuple[str, ...]
) -> bool:
    architecture = architecture_for(capability)
    return bool(
        architecture is not None
        and architecture.implemented
        and architecture.compilation_target.replace("_", "") in archs
    )


def require_kernel_architecture(
    module: str, capability: tuple[int, int] | None
) -> None:
    """Protect unplanned CuTe entry points from compiling SM12x code on SM103."""
    architecture = architecture_for(capability)
    if (
        architecture is None
        or architecture.name != "sm103"
        or not module.startswith("b12x.")
    ):
        return
    if module.startswith("b12x.moe._shared.kernels.sm103."):
        return
    if module in {"b12x.comm.roce._oneshot_cute", "b12x.comm.roce._allgather_cute"}:
        # Transport construction separately requires explicit Grace qualification.
        return
    raise UnsupportedArchitectureError(
        f"{module} has no admitted SM103 CuTe implementation; use an implemented backend"
    )


def require_component_architecture(
    component_id: str, capability: tuple[int, int]
) -> None:
    """Reject registered plans on recognized targets without an implementation.

    Synthetic unknown devices retain the policy system's heuristic contract.
    Registered metadata is authoritative for architecture coverage.
    """
    architecture = architecture_for(capability)
    if (
        architecture is None
        or not architecture.implemented
        or architecture.mma_family == "warp"
    ):
        return
    from b12x import find_op
    from b12x.policy.catalog import PLANNING_COMPONENTS

    registrations = [r for r in PLANNING_COMPONENTS if r.component_id == component_id]
    for registration in registrations:
        meta = find_op(registration.op_qualname)
        if not supports_architecture(capability, meta.archs):
            raise UnsupportedArchitectureError(
                f"b12x.{meta.qualname} has no {architecture.name} backend; "
                "select an integration fallback before allocating a plan"
            )


__all__ = [
    "Architecture",
    "UnsupportedArchitectureError",
    "architecture_for",
    "supports_architecture",
]
