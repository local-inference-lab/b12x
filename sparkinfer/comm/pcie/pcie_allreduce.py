"""World-size-dispatched PCIe all-reduce runtime."""

from __future__ import annotations

from contextlib import contextmanager
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
from .pcie_oneshot import (
    DEFAULT_MAX_SIZE,
    DEFAULT_RANK_DATA_BYTES,
    SUPPORTED_WORLD_SIZES as ONESHOT_WORLD_SIZES,
    PCIeOneshotAllReducePool,
)


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

    def __init__(self, runtime: Any, algorithm: str) -> None:
        self._runtime = runtime
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
        return cls(runtime, algorithm)

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
        return self._runtime.for_stream(stream)

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
        with self._runtime.capture(stream=stream) as runtime:
            yield runtime

    def close(self) -> None:
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
