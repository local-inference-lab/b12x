"""Tile-aware direct scratch preserving one route per physical tile."""
from dataclasses import replace
from .policy_smallm_schedule import p8_small_m_scratch_layout, P8ScratchRegion

def direct_scratch_layout(tokens, intermediate=512, *, tile_m=16):
    if not 1 <= tokens <= 16 or tile_m not in (16, 32):
        raise ValueError("direct policy requires M1..16 and tile M16/M32")
    base = p8_small_m_scratch_layout(tokens=tokens, intermediate=intermediate,
                                    shared=True, grouped=False, tile_m=tile_m, direct=True)
    regions, cursor = [], 0
    for region in base.regions:
        if region.name == 'output':
            region = replace(region, shape=(tokens,4096), nbytes=tokens*4096*2)
        cursor = (cursor+15)//16*16
        regions.append(replace(region, offset=cursor))
        cursor += region.nbytes
    cursor = (cursor+15)//16*16
    regions.append(P8ScratchRegion("output", "bfloat16", (tokens,4096), cursor, tokens*4096*2))
    cursor += tokens*4096*2
    return replace(base, regions=tuple(regions), nbytes=(cursor+15)//16*16)
