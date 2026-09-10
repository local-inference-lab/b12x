"""CPU-only layout for one-fill direct-route P8 scratch; no arithmetic changes."""
from dataclasses import replace
from math import prod
from .p8_smallm_schedule import p8_small_m_scratch_layout


def direct_scratch_layout(tokens, intermediate=512):
    if tokens not in range(1, 17):
        raise ValueError('direct scratch supports 1..16 tokens')
    base = p8_small_m_scratch_layout(intermediate=intermediate)
    scaled = {'packed_a', 'intermediate_u32', 'task_ready', 'task_expert',
              'task_m_tile', 'task_slice_begin', 'task_slice_count',
              'task_valid_rows', 'tile_write_count', 'token_map', 'token_weights'}
    sizes = {'uint8': 1, 'int32': 4, 'float32': 4, 'bfloat16': 2}
    regions, cursor = [], 0
    for region in base.regions:
        shape = region.shape
        if region.name in scaled:
            shape = (shape[0] * tokens,)
        elif region.name == 'scale_flat':
            shape = ((288 + tokens * 8 + 1) * 16 * (4096 // 8),)
        elif region.name == 'output':
            shape = (tokens, 4096)
        cursor = (cursor + 15) // 16 * 16
        size = prod(shape) * sizes[region.dtype]
        regions.append(replace(region, shape=shape, offset=cursor, nbytes=size))
        cursor += size
    return replace(base, regions=tuple(regions), nbytes=(cursor + 15) // 16 * 16)
def direct_grid_capacity(m: int, default: int = 64) -> int:
    """TP4 policy: measured direct shapes and qualified grouped capacity128."""
    if m > 16:
        return 128
    return {4: 192, 8: 128, 16: 192}.get(m, default)


def grouped_m16_scratch_layout(tokens, intermediate=512, compact_input=True, *, grouped=False):
    """Exact extents of the existing GLM TP4 grouped-M16 allocations."""
    if tokens < 1 or intermediate != 512:
        raise ValueError('grouped M16 requires positive tokens and I512')
    base = p8_small_m_scratch_layout(intermediate=intermediate)
    tiles = 288 + (tokens * 8 + 15) // 16
    rows = tiles * 16
    counts = dict(packed_a=(tokens if grouped and compact_input else rows) * 4096,
                  scale_flat=(tokens * 128 if grouped else (288 + tokens * 8 + 1) * 16 * 512),
                  intermediate_u32=rows * (intermediate + intermediate // 32) // 4,
                  tile_write_count=tiles, row_counts=288,
                  expert_write_rows=288, expert_tile_base=289,
                  token_map=rows, token_weights=rows)
    for name in ('barrier_count', 'barrier_epoch', 'pair_head', 'producers_done',
                 'all_published', 'task_head', 'task_tail'):
        counts[name] = 1
    for name in ('task_ready', 'task_expert', 'task_m_tile', 'task_slice_begin',
                 'task_slice_count', 'task_valid_rows'):
        counts[name] = tiles * (intermediate // 128)
    sizes = {'uint8': 1, 'int32': 4, 'float32': 4, 'bfloat16': 2}
    regions, cursor = [], 0
    for region in base.regions:
        shape = (tokens, 4096) if region.name == 'output' else (counts[region.name],)
        cursor = (cursor + 15) // 16 * 16
        size = prod(shape) * sizes[region.dtype]
        regions.append(replace(region, shape=shape, offset=cursor, nbytes=size))
        cursor += size
    return replace(base, regions=tuple(regions), nbytes=(cursor + 15) // 16 * 16)
