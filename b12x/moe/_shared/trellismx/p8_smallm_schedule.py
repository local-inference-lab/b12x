"""CPU-visible geometry for the developmental, exact-contract P8 M1 arm."""
from dataclasses import dataclass


@dataclass(frozen=True)
class P8SmallMGeometry:
    tokens: int = 1
    experts: int = 288
    hidden: int = 4096
    intermediate: int = 512
    topk: int = 8

    def __post_init__(self):
        if (self.tokens, self.experts, self.hidden, self.topk) != (1, 288, 4096, 8) or self.intermediate not in (512, 1024):
            raise ValueError("small-M candidate requires GLM TP4/TP2 M1 geometry")

    @property
    def physical_tiles(self):
        return self.topk

    @property
    def fc1_tasks(self):
        return self.topk * (self.intermediate // 128)

    @property
    def fc2_tasks(self):
        return self.topk * (self.hidden // 256)

    def fc1_owner(self, slot):
        if not 0 <= slot < self.fc1_tasks:
            raise ValueError("FC1 task outside capacity")
        return divmod(slot, self.intermediate // 128)

    def fc2_owner(self, slot):
        if not 0 <= slot < self.fc2_tasks:
            raise ValueError("FC2 task outside capacity")
        return divmod(slot, self.hidden // 256)


def use_small_m(enabled: bool, tokens: int) -> bool:
    return bool(enabled and tokens == 1)


@dataclass(frozen=True)
class P8ScratchRegion:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class P8ScratchLayout:
    regions: tuple[P8ScratchRegion, ...]
    nbytes: int


def p8_small_m_scratch_layout(*, intermediate: int = 512, tokens: int = 1, shared: bool = False, grouped: bool = False, tile_m: int | None = None, direct: bool | None = None) -> P8ScratchLayout:
    """Exact M1 buffer extents, each starting at a 16-byte-aligned offset.

    The route-output allocation is intentionally excluded. Padding between
    regions is also zeroed by the caller's single uint8 arena initialization.
    """
    geometry = P8SmallMGeometry(intermediate=intermediate)
    if tokens < 1 or (tokens != 1 and not shared):
        raise ValueError('multi-token layout requires shared coupled workspace')
    tile_m = (64 if grouped else 16) if tile_m is None else tile_m
    if tile_m not in (16, 64, 128):
        raise ValueError("unsupported scratch tile size")
    direct = not grouped if direct is None else direct
    physical_tiles = (geometry.experts + (tokens * geometry.topk + tile_m - 1) // tile_m
                      if not direct else tokens * geometry.topk)
    rows = physical_tiles * tile_m
    max_tasks = physical_tiles * (geometry.intermediate // 128)
    input_rows = tokens if shared and grouped else rows
    scale_elements = (tokens * (geometry.hidden // 32) if shared and grouped
                      else (geometry.experts + tokens * geometry.topk + 1) * tile_m * (geometry.hidden // 8))
    specs = [
        ("packed_a", "uint8", (input_rows * geometry.hidden,)),
        ("scale_flat", "uint8",
         (scale_elements,)),
        ("intermediate_u32", "int32",
         (rows * (geometry.intermediate + geometry.intermediate // 32) // 4,)),
    ]
    specs.extend((name, "int32", (1,)) for name in (
        "barrier_count", "barrier_epoch", "pair_head", "producers_done",
        "all_published", "task_head", "task_tail",
    ))
    specs.extend((name, "int32", (max_tasks,)) for name in (
        "task_ready", "task_expert", "task_m_tile", "task_slice_begin",
        "task_slice_count", "task_valid_rows",
    ))
    specs.extend([
        ("tile_write_count", "int32", (physical_tiles,)),
        ("row_counts", "int32", (geometry.experts,)),
        ("expert_write_rows", "int32", (geometry.experts,)),
        ("expert_tile_base", "int32", (geometry.experts + 1,)),
        ("token_map", "int32", (rows,)),
        ("token_weights", "float32", (rows,)),
        ("output", "bfloat16", (1, geometry.hidden)),
    ])
    if shared:
        specs = [spec for spec in specs if spec[0] != 'output']
    sizes = {"uint8": 1, "int32": 4, "float32": 4, "bfloat16": 2}
    regions = []
    cursor = 0
    for name, dtype, shape in specs:
        cursor = (cursor + 15) // 16 * 16
        elements = 1
        for extent in shape:
            elements *= extent
        nbytes = elements * sizes[dtype]
        regions.append(P8ScratchRegion(name, dtype, shape, cursor, nbytes))
        cursor += nbytes
    return P8ScratchLayout(tuple(regions), (cursor + 15) // 16 * 16)
