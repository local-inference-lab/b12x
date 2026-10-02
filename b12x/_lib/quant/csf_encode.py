"""Load-time, lossless GPU packing of logical FP4 block-scale bytes.

Each row uses the lowest base covering the largest number of scale bytes.
Values outside its two-byte (MXFP4) or sixteen-byte (NVFP4) interval are
stored verbatim as exceptions. No floating-point conversion is performed.
The compact allocation requires one host synchronization; this is not a
graph-replay operation.
"""

from __future__ import annotations

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _choose_narrow_mxfp4_bases(
    S,
    Bases,
    Counts,
    rows,
    R: tl.constexpr,
    C: tl.constexpr,
    ROT: tl.constexpr,
    B: tl.constexpr,
):
    rid = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    col = tl.arange(0, B)
    row = rid // R * R + (rid % R + ROT) % R
    values = tl.load(
        S + row[:, None] * C + col[None, :],
        (rid[:, None] < rows) & (col[None, :] < C),
        0,
    ).to(tl.int32)
    # An optimal two-byte interval starts at an observed byte or one below it.
    # Comparing short rows directly avoids a 256-bin histogram per row.
    difference = values[:, :, None] - values[:, None, :]
    valid = col[None, :, None] < C
    lower = tl.sum((valid & ((difference == 0) | (difference == 1))).to(tl.int32), 1)
    upper = tl.sum((valid & ((difference == 0) | (difference == -1))).to(tl.int32), 1)
    lower = tl.where((col[None, :] < C) & (values < 255), lower, -1)
    upper = tl.where((col[None, :] < C) & (values > 0), upper, -1)
    best = tl.maximum(tl.max(lower, 1), tl.max(upper, 1))
    chosen = tl.minimum(
        tl.min(tl.where(lower == best[:, None], values, 256), 1),
        tl.min(tl.where(upper == best[:, None], values - 1, 256), 1),
    )
    tl.store(Bases + rid, chosen, rid < rows)
    tl.store(Counts + rid, C - best, rid < rows)


@tr.jit
def _pack_narrow_mxfp4_rows(
    S,
    Bases,
    Prefix,
    Fixed,
    Exceptions,
    rows,
    R: tl.constexpr,
    C: tl.constexpr,
    ROT: tl.constexpr,
    B: tl.constexpr,
):
    rid = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    expert, ordered_row = rid // R, rid % R
    row = (ordered_row + ROT) % R
    col = tl.arange(0, B)
    valid = (rid[:, None] < rows) & (col[None, :] < C)
    values = tl.load(
        S + (expert[:, None] * R + row[:, None]) * C + col[None, :], valid, 0
    ).to(tl.int32)
    base = tl.load(Bases + rid, rid < rows, 0).to(tl.int32)
    delta = values - base[:, None]
    outside = valid & ((delta < 0) | (delta >= 2))
    rank = tl.cumsum(outside.to(tl.int32), 1) - 1
    start = tl.load(Prefix + rid, rid < rows, 0)
    record = (row[:, None] * C + col[None, :]).to(tl.uint32)
    record |= values.to(tl.uint32) << 24
    tl.store(Exceptions + start[:, None] + rank, record, outside)
    slab = (expert * (R // 16) + row // 16) * 32
    tl.store(Fixed + slab + row % 16, base, rid < rows)
    packed = tl.sum(tl.where(valid & (delta == 1), 1 << col[None, :], 0), 1)
    tl.store(Fixed + slab + 16 + row % 16, packed, rid < rows)


@tr.jit
def _choose_bases(
    S,
    Bases,
    Counts,
    R: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    ROT: tl.constexpr,
    B: tl.constexpr,
):
    rid = tl.program_id(0).to(tl.int64)
    row = (rid % R + ROT) % R
    col = tl.arange(0, B)
    values = tl.load(S + (rid // R * R + row) * C + col, col < C, 0).to(tl.int32)
    hist = tl.histogram(values, 256, mask=col < C)
    ends = tl.cumsum(hist)
    base = tl.arange(0, 256)
    right = tl.gather(ends, tl.minimum(base + W - 1, 255), 0)
    left = tl.where(base > 0, tl.gather(ends, tl.maximum(base - 1, 0), 0), 0)
    covered = tl.where(base <= 256 - W, right - left, -1)
    best = tl.max(covered, 0)
    chosen = tl.min(tl.where(covered == best, base, 256), 0)
    tl.store(Bases + rid, chosen)
    tl.store(Counts + rid, C - best)


@tr.jit
def _pack_rows(
    S,
    Bases,
    Prefix,
    Fixed,
    Exceptions,
    R: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    ROT: tl.constexpr,
    B: tl.constexpr,
):
    rid = tl.program_id(0).to(tl.int64)
    expert, ordered_row = rid // R, rid % R
    row = (ordered_row + ROT) % R
    col = tl.arange(0, B)
    values = tl.load(S + (expert * R + row) * C + col, col < C, 0).to(tl.int32)
    base = tl.load(Bases + rid).to(tl.int32)
    delta = values - base
    exception = (col < C) & ((delta < 0) | (delta >= W))
    rank = tl.cumsum(exception.to(tl.int32)) - 1
    start = tl.load(Prefix + rid)
    record = (row * C + col).to(tl.uint32) | (values.to(tl.uint32) << 24)
    tl.store(Exceptions + start + rank, record, exception)
    if W == 2:
        width: tl.constexpr = tr.cdiv(C, 8)
        slab = (expert * (R // 16) + row // 16) * (16 * (1 + width))
        tl.store(Fixed + slab + row % 16, base)
        # Each lane writes one byte assembled from eight consecutive selectors.
        byte = tl.arange(0, tr.next_power_of_2(width))
        packed = tl.full((tr.next_power_of_2(width),), 0, tl.int32)
        for bit in tl.static_range(8):
            c = byte * 8 + bit
            v = tl.load(S + (expert * R + row) * C + c, c < C, 0).to(tl.int32)
            packed |= tl.where((c < C) & (v == base + 1), 1 << bit, 0)
        tl.store(Fixed + slab + 16 + row % 16 * width + byte, packed, byte < width)
    else:
        slab = (expert * (R // 128) + row // 128) * (128 * (1 + C // 2))
        tl.store(Fixed + slab + row % 32 * 4 + row % 128 // 32, base)
        lo = tl.where((delta >= 0) & (delta < W) & (col < C), delta, 0)
        hi = tl.gather(lo, tl.minimum(col + 1, B - 1), 0)
        offset = (
            128 + col // 4 * 256 + row % 32 * 8 + row % 128 // 32 * 2 + col % 4 // 2
        )
        tl.store(Fixed + slab + offset, lo | (hi << 4), (col < C) & (col % 2 == 0))


def encode_scale_bytes(
    scales: torch.Tensor, *, format: str, exception_row_rotation: int = 0
):
    """Encode logical ``[expert, row, column]`` bytes into a resident CSF batch.

    MXFP4 exception partitions can be ordered by a 64-row-aligned rotation;
    fixed streams and exception positions retain the original logical order.
    Source bytes are never modified. Dense exceptions remain exact, but may
    make the result larger than its input.
    """
    if (
        scales.dtype != torch.uint8
        or scales.ndim != 3
        or scales.device.type != "cuda"
        or not scales.is_contiguous()
    ):
        raise ValueError("CSF encoding requires contiguous CUDA uint8 [E,R,C]")
    e, r, c = scales.shape
    nv = format == "nvfp4"
    if format not in ("mxfp4", "nvfp4"):
        raise ValueError("scale format must be mxfp4 or nvfp4")
    rot = exception_row_rotation
    if (
        e <= 0
        or r <= 0
        or c <= 0
        or r * c > 1 << 24
        or r % (128 if nv else 64)
        or (c % 4 if nv else c > 255)
        or rot < 0
        or rot >= r
        or rot % 64
        or (nv and rot)
    ):
        raise ValueError("unsupported CSF scale geometry or exception rotation")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CSF encoding is a load-time operation, not graph replay")
    bases = torch.empty(e * r, dtype=torch.uint8, device=scales.device)
    counts = torch.empty(e * r, dtype=torch.int64, device=scales.device)
    args = dict(R=r, C=c, W=16 if nv else 2, ROT=rot, B=tr.next_power_of_2(c))
    if not nv and c <= 8:
        _choose_narrow_mxfp4_bases[(tr.cdiv(e * r, 32),)](
            scales,
            bases,
            counts,
            e * r,
            R=r,
            C=c,
            ROT=rot,
            B=tr.next_power_of_2(c),
        )
    else:
        _choose_bases[(e * r,)](scales, bases, counts, **args)
    prefix = torch.empty(e * r + 1, dtype=torch.int64, device=scales.device)
    prefix[0] = 0
    torch.cumsum(counts, 0, out=prefix[1:])
    exceptions = torch.empty(
        int(prefix[-1].item()), dtype=torch.uint32, device=scales.device
    )
    slab = 128 if nv else 16
    width = c // 2 if nv else tr.cdiv(c, 8)
    fixed = torch.empty(
        (e, r // slab, slab * (1 + width)), dtype=torch.uint8, device=scales.device
    )
    if not nv and c <= 8:
        _pack_narrow_mxfp4_rows[(tr.cdiv(e * r, 32),)](
            scales,
            bases,
            prefix,
            fixed,
            exceptions,
            e * r,
            R=r,
            C=c,
            ROT=rot,
            B=tr.next_power_of_2(c),
        )
    else:
        _pack_rows[(e * r,)](scales, bases, prefix, fixed, exceptions, **args)
    task_rows = 128 if nv else 64
    indices = (
        torch.arange(e, device=scales.device, dtype=torch.int64)[:, None] * r
        + torch.arange(0, r + 1, task_rows, device=scales.device)[None, :]
    )
    partitions = prefix[indices].contiguous()
    if nv:
        from .nvfp4_csf import Nvfp4CsfBatch

        result = Nvfp4CsfBatch(fixed, exceptions.view(torch.uint8), partitions, r, c)
    else:
        from .x4t_scales import X4TScaleBatch

        result = X4TScaleBatch(
            fixed, exceptions, prefix[::r].contiguous(), r, c, partitions, 64, rot
        )
    result.validate()
    return result
