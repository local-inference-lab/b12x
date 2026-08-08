"""World-size-dispatched PCIe all-reduce runtime."""

from __future__ import annotations

import os

from contextlib import ExitStack, contextmanager, suppress
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .pcie_hierarchical import (
    SUPPORTED_WORLD_SIZES as HIERARCHICAL_WORLD_SIZES,
)
from .pcie_hierarchical import (
    PCIeHierarchicalAllReduce,
)
from .pcie_island_rs import (
    CROSSOVER_ELEMENTS as ISLAND_RS_CROSSOVER_ELEMENTS,
)
from .pcie_island_rs import (
    SUPPORTED_WORLD_SIZES as ISLAND_RS_WORLD_SIZES,
)
from .pcie_island_rs import (
    PCIeIslandRSAllReduce,
)
from .pcie_oneshot import (
    DEFAULT_MAX_SIZE,
    DEFAULT_RANK_DATA_BYTES,
    SUPPORTED_WORLD_SIZES as ONESHOT_WORLD_SIZES,
    PCIeOneshotAllReducePool,
)


# The island reduce-scatter runtime keeps beating NCCL up to about this size;
# past it NCCL's flat protocol cost wins again (37.6 vs 35.3 us at twelve rows
# of 7168). Callers that gate on a byte limit should ask for this rather than
# carrying their own constant.
ISLAND_RS_MAX_BYTES = 160 * 1024


def _algorithm_override() -> str:
    """``auto`` picks per message size; the others pin one implementation."""

    choice = os.getenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "auto").strip().lower()
    if choice not in ("auto", "hierarchical", "island_rs"):
        raise ValueError(
            "B12X_PCIE_ALLREDUCE_ALGORITHM must be auto, hierarchical or "
            f"island_rs, got {choice!r}"
        )
    return choice


def recommended_max_bytes(world_size: int, *, default: int = DEFAULT_MAX_SIZE) -> int:
    """Largest all-reduce this runtime expects to win at, for this world size.

    Exists so callers do not have to hard-code a byte limit that only holds for
    the world sizes and kernels that existed when they were written.
    """

    if world_size in ISLAND_RS_WORLD_SIZES and _algorithm_override() != "hierarchical":
        return max(default, ISLAND_RS_MAX_BYTES)
    return default


MAX_DIRECT_WORLD_SIZE = 8
DIRECT_WORLD_SIZES = tuple(
    world_size
    for world_size in ONESHOT_WORLD_SIZES
    if world_size <= MAX_DIRECT_WORLD_SIZE
)
SUPPORTED_WORLD_SIZES = (*DIRECT_WORLD_SIZES, *HIERARCHICAL_WORLD_SIZES)


def _algorithm_for_world_size(world_size: int) -> str:
    if world_size in DIRECT_WORLD_SIZES:
        return "oneshot"
    if world_size in HIERARCHICAL_WORLD_SIZES:
        return "hierarchical"
    raise ValueError(
        f"unsupported PCIe all-reduce world size {world_size}; "
        f"supported world sizes are {SUPPORTED_WORLD_SIZES}"
    )


class PCIeAllReduce:
    """Select a peer-safe all-reduce implementation from the world size.

    Worlds through TP8 use the low-latency all-peer oneshot runtime. TP12 and
    TP16 use bounded-degree four-GPU islands so no CUDA context maps more than
    six peers. Other worlds fail closed instead of exceeding the CUDA peer
    connection limit.
    """

    def __init__(
        self,
        runtime: Any,
        algorithm: str,
        island_rs: Any = None,
    ) -> None:
        self._runtime = runtime
        # Optional second implementation for the same world. When present the
        # dispatcher routes by message size instead of exposing a knob.
        self._island_rs = island_rs
        self.algorithm = algorithm
        self.rank = runtime.rank
        self.world_size = runtime.world_size
        self.device = runtime.device

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeAllReduce":
        world_size = dist.get_world_size(group=exchange_group)
        algorithm = _algorithm_for_world_size(world_size)
        if algorithm == "oneshot":
            runtime = PCIeOneshotAllReducePool.from_exchange_group(
                exchange_group=exchange_group,
                device=device,
                eager_buffer_bytes=eager_buffer_bytes,
                max_size=max_size,
                rank_data_bytes=rank_data_bytes,
                ext_module=ext_module,
                single_channel=single_channel,
            )
        else:
            if max_size < torch.bfloat16.itemsize:
                raise ValueError("max_size must hold at least one BF16 element")
            runtime = PCIeHierarchicalAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=max_size // torch.bfloat16.itemsize,
                ext_module=ext_module,
            )
        island_rs = cls._maybe_island_rs(
            exchange_group=exchange_group,
            device=device,
            max_size=max_size,
        )
        return cls(runtime, algorithm, island_rs)

    @staticmethod
    def _maybe_island_rs(
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int,
    ) -> Any:
        """Attach the equal-quarter runtime where it is supported and wanted.

        Sized from :data:`ISLAND_RS_MAX_BYTES` rather than the caller's
        ``max_size`` so the band it is good at is available even when the caller
        provisioned the older, smaller limit. Returns ``None`` on any failure;
        the hierarchy alone is always a correct configuration.
        """

        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in ISLAND_RS_WORLD_SIZES:
            return None
        if _algorithm_override() == "hierarchical":
            return None
        capacity = max(int(max_size), ISLAND_RS_MAX_BYTES)
        elements = capacity // torch.bfloat16.itemsize
        try:
            return PCIeIslandRSAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=elements - (elements % 2),
            )
        except Exception:
            return None

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
    ) -> "PCIeAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=(
                max_input_bytes if eager_buffer_bytes is None else eager_buffer_bytes
            ),
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
        )

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        """Whether another runtime may safely map every rank as a peer."""

        return self.algorithm == "oneshot"

    def for_stream(self, stream: object = None):
        if self._island_rs is not None:
            self._island_rs.for_stream(stream)
        return self._runtime.for_stream(stream)

    def _use_island_rs(self, inp: torch.Tensor) -> bool:
        """Route large messages to the equal-quarter runtime.

        Small ones stay on the hierarchy, whose critical path is shorter for the
        ranks that are not the island leader; large ones would otherwise funnel
        the whole vector through that leader's PCIe link.
        """

        if self._island_rs is None:
            return False
        override = _algorithm_override()
        if override == "hierarchical":
            return False
        if override != "island_rs" and inp.numel() <= ISLAND_RS_CROSSOVER_ELEMENTS:
            return False
        return self._island_rs.should_allreduce(inp)

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._runtime.should_allreduce(inp):
            return True
        return self._island_rs is not None and self._island_rs.should_allreduce(inp)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        blocks: Optional[int] = None,
        stream: object = None,
    ) -> torch.Tensor:
        if self.algorithm == "hierarchical":
            if peer_input_ptrs is not None:
                raise ValueError(
                    "peer_input_ptrs are unavailable for hierarchical all-reduce"
                )
            if self._use_island_rs(inp):
                return self._island_rs.all_reduce(inp, out=out, blocks=blocks)
            return self._runtime.all_reduce(
                inp,
                out=out,
                blocks=blocks,
                stream=stream,
            )
        if blocks is not None:
            raise ValueError("blocks is only available for hierarchical all-reduce")
        return self._runtime.all_reduce(
            inp,
            out=out,
            peer_input_ptrs=peer_input_ptrs,
            stream=stream,
        )

    @contextmanager
    def capture(self, stream: object = None):
        with ExitStack() as stack:
            runtime = stack.enter_context(self._runtime.capture(stream=stream))
            if self._island_rs is not None:
                stack.enter_context(self._island_rs.capture(stream=stream))
            yield runtime

    def close(self) -> None:
        if self._island_rs is not None:
            with suppress(Exception):
                self._island_rs.close()
            self._island_rs = None
        self._runtime.close()

    def __getattr__(self, name: str):
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            raise AttributeError(name)
        return getattr(runtime, name)


__all__ = [
    "DIRECT_WORLD_SIZES",
    "MAX_DIRECT_WORLD_SIZE",
    "PCIeAllReduce",
    "SUPPORTED_WORLD_SIZES",
]
