"""Preparation-owned HBM and exact-size mapped Grace expert slabs."""
from dataclasses import dataclass
import math

import torch

from .residency import ExpertMemoryAccounting


def align(value):
    return (value + 255) // 256 * 256


def tier_layout(experts, hidden, intermediate):
    offset, fields = 0, []
    for name, shape in (
        ("w13", (experts, 2 * intermediate, hidden // 2)),
        ("w2", (experts, hidden, intermediate // 2)),
        ("s13", (experts, 2 * intermediate, hidden // 32)),
        ("s2", (experts, hidden, intermediate // 32)),
    ):
        fields.append((name, offset, shape))
        offset = align(offset + math.prod(shape))
    return tuple(fields), offset


def workspace_layout(query):
    r, h, i = query.max_tokens * query.max_top_k, query.hidden, query.intermediate
    offset, fields = 0, []
    for name, shape, dtype in (
        ("local_ids", (2, (r+3)//4*4), torch.int32), ("indices", (2, (r+3)//4*4), torch.int32),
        ("counts", (8,), torch.int32),
        ("q1", (r, h), torch.float8_e4m3fn), ("s1", (r, 128 * (h // 32)), torch.uint8),
        ("fc1", (r, 2 * i), torch.float32),
        ("q2", (r, i), torch.float8_e4m3fn), ("s2", (r, 128 * (i // 32)), torch.uint8),
        ("fc2", (r, h), torch.bfloat16), ("out", (query.max_tokens, h), torch.bfloat16),
    ):
        fields.append((name, offset, shape, dtype))
        offset = align(offset + math.prod(shape) * dtype.itemsize)
    return tuple(fields), offset


def accounting(query):
    hot = tier_layout(query.hot_experts, query.hidden, query.intermediate)[1]
    cold = tier_layout(query.experts - query.hot_experts, query.hidden, query.intermediate)[1]
    return ExpertMemoryAccounting(hbm_expert_bytes=hot, grace_expert_bytes=cold,
                                  scratch_bytes=workspace_layout(query)[1],
                                  route_map_bytes=align(query.experts * 2 * 4))


def validate_source(weights, query):
    from .weights import PackedWeights
    if not isinstance(weights, PackedWeights):
        raise TypeError("hierarchical packed preparation requires PackedWeights")
    e, h, i = query.experts, query.hidden, query.intermediate
    for name, shape, dtypes in (
        ("w13", (e, 2*i, h//2), (torch.uint8, torch.float4_e2m1fn_x2)),
        ("w2", (e, h, i//2), (torch.uint8, torch.float4_e2m1fn_x2)),
        ("w13_block_scales", (e, 2*i, h//32), (torch.uint8, torch.float8_e8m0fnu)),
        ("w2_block_scales", (e, h, i//32), (torch.uint8, torch.float8_e8m0fnu)),
    ):
        value = getattr(weights, name)
        if value.shape != shape or value.dtype not in dtypes or not value.is_contiguous():
            raise ValueError(f"{name} requires contiguous {shape} with dtype {dtypes}")
        if value.device.type != "cpu":
            raise ValueError("hierarchical preparation requires CPU checkpoint views to bound HBM staging")
    for name in ("w13_global_scales", "w2_global_scales", "input_scale", "intermediate_scale"):
        value = getattr(weights, name)
        if value is not None and (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() not in (1, e)):
            raise ValueError(f"{name} must be scalar or per-expert CPU FP32 metadata")


def _swizzle_scale(source):
    rows, columns = source.shape
    return source.view(torch.uint8).reshape(rows//128, 4, 32, columns//4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(rows, columns)


@dataclass(frozen=True)
class TierStorage:
    slab: torch.Tensor
    fields: dict
    owner: object | None


def materialize_tier(ids, weights, query, device, *, grace):
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    fields, nbytes = tier_layout(len(ids), query.hidden, query.intermediate)
    if not nbytes:
        return None
    owner = MappedHostAllocation((nbytes,), torch.uint8, device) if grace else None
    slab = owner.device_view if grace else torch.empty(nbytes, dtype=torch.uint8, device=device)
    destination = owner.host_view if grace else slab
    views = {}
    try:
        for name, offset, shape in fields:
            size = math.prod(shape)
            views[name] = slab[offset:offset+size].view(shape)
            target = destination[offset:offset+size].view(shape)
            source = getattr(weights, {"s13": "w13_block_scales", "s2": "w2_block_scales"}.get(name, name))
            for local, original in enumerate(ids):
                row = source[original].view(torch.uint8)
                if name in ("s13", "s2"):
                    row = _swizzle_scale(row)
                target[local].copy_(row)
        return TierStorage(slab, views, owner)
    except BaseException:
        if owner is not None:
            owner.close()
        raise
