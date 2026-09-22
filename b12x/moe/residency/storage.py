"""Host-only storage capabilities for logical expert residency.

Adapters own encoding, validation and materialization. These declarations do
not register an executable backend or confer hardware qualification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from .contracts import _integer, _text


class BackingMode(str, Enum):
    DIRECT = "direct_mapped"
    PREPARED = "prepared_canonical"
    ON_PROMOTION = "prepare_on_promotion"
    RESIDENT_ONLY = "resident_only"


@dataclass(frozen=True, kw_only=True)
class ExpertShard:
    """One rank's intermediate-axis slice; expert IDs are never sharded here."""

    experts: int
    hidden: int
    intermediate: int
    global_intermediate: int
    intermediate_start: int = 0
    rank: int = 0
    world_size: int = 1

    def __post_init__(self):
        for name in ("experts", "hidden", "intermediate", "global_intermediate", "world_size"):
            _integer(name, getattr(self, name), 1)
        _integer("rank", self.rank)
        _integer("intermediate_start", self.intermediate_start)
        if self.rank >= self.world_size or self.intermediate_start + self.intermediate > self.global_intermediate:
            raise ValueError("expert shard is outside the declared TP geometry")
        if self.world_size == 1 and (self.intermediate_start or self.intermediate != self.global_intermediate):
            raise ValueError("single-rank expert storage requires complete geometry")


@dataclass(frozen=True, kw_only=True)
class ExpertRepresentation:
    """Payload sizes include scale/metadata fields, before slab alignment.

    Shared bytes are owned once per layer, not once per logical expert. The
    adapter still supplies allocation accounting for padding and workspaces.
    """

    encoding: str
    bytes_per_expert: int
    shared_bytes: int = 0
    alignment: int = 1

    def __post_init__(self):
        _text("representation encoding", self.encoding)
        _integer("bytes_per_expert", self.bytes_per_expert, 1)
        _integer("shared_bytes", self.shared_bytes)
        _integer("alignment", self.alignment, 1)
        if self.alignment & (self.alignment - 1):
            raise ValueError("representation alignment must be a power of two")

    def payload_bytes(self, experts):
        _integer("experts", experts)
        return self.bytes_per_expert * experts + self.shared_bytes


@dataclass(frozen=True, kw_only=True)
class ExpertStorageContract:
    """Separate checkpoint bytes, executable bytes, arithmetic and local costs."""

    adapter: str
    checkpoint: str
    layer: str
    recipe: str
    shard: ExpertShard
    source: ExpertRepresentation
    backing: ExpertRepresentation | None
    resident: ExpertRepresentation
    mode: BackingMode
    direct_cold_execution: bool
    rollback: str
    source_transform: str | None = None
    promotion_transform: str | None = None

    def __post_init__(self):
        for name in ("adapter", "checkpoint", "layer", "recipe", "rollback"):
            _text(name, getattr(self, name))
        if not isinstance(self.shard, ExpertShard) or any(
            not isinstance(r, ExpertRepresentation) for r in (self.source, self.resident)
        ):
            raise TypeError("storage requires typed geometry and representations")
        object.__setattr__(self, "mode", BackingMode(self.mode))
        if type(self.direct_cold_execution) is not bool:
            raise TypeError("direct_cold_execution must be bool")
        for value in (self.source_transform, self.promotion_transform):
            if value is not None:
                _text("transform", value)
        if self.mode == BackingMode.RESIDENT_ONLY:
            if self.backing is not None or self.direct_cold_execution:
                raise ValueError("resident-only storage has no executable backing")
        elif not isinstance(self.backing, ExpertRepresentation):
            raise TypeError("host-backed storage requires a backing representation")
        if self.backing is not None and self.source != self.backing and self.source_transform is None:
            raise ValueError("different source/backing representations require a source transform")
        if (self.mode == BackingMode.RESIDENT_ONLY and self.source != self.resident
                and self.source_transform is None and self.promotion_transform is None):
            raise ValueError("different source/resident representations require a preparation transform")
        if self.mode in (BackingMode.DIRECT, BackingMode.PREPARED) and not self.direct_cold_execution:
            raise ValueError("direct/prepared canonical storage requires cold execution")
        if self.mode == BackingMode.PREPARED and self.source_transform is None:
            raise ValueError("prepared canonical storage requires a source transform")
        if self.mode == BackingMode.ON_PROMOTION and (self.direct_cold_execution or self.promotion_transform is None):
            raise ValueError("prepare-on-promotion requires a transform and no direct cold execution")
        if self.backing is not None and self.backing != self.resident and self.promotion_transform is None:
            raise ValueError("different backing/resident representations require a promotion transform")

    def require_cold_execution(self):
        if not self.direct_cold_execution:
            raise ValueError("storage adapter does not provide direct cold execution")


@runtime_checkable
class ExpertStorageSource(Protocol):
    """Immutable owners retained by preparation; row materialization is bounded.

    ``row`` returns adapter-defined canonical fields for one logical expert.
    Adapters requiring promotion transforms provide those operations through
    their prepared backend, never through the replacement policy.
    """

    @property
    def storage(self) -> ExpertStorageContract: ...

    @property
    def source_bytes(self) -> int: ...

    def validate_values(self) -> None: ...

    def row(self, expert: int) -> Mapping[str, object]: ...
