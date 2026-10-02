from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._packing import _check_f32_scales


def workspace_sizes(capacity, n, k, groups, *, compact):
    if capacity < 0 or n <= 0 or k <= 0 or k % 128 or groups <= 0:
        raise ValueError("invalid grouped scale geometry")
    return ((capacity + 127) // 128 * (k // 128) * 512,
            groups * ((n + 127) // 128) * ((k + 511) // 512 if compact else k // 128) * 512)


def normalize_g1(scales, rows, columns, groups):
    if groups == 1 and tuple(scales.shape) == (1, rows, columns):
        return scales.view(rows, columns)
    return scales


@triton.jit
def _trap_if_invalid(valid):
    if tl.sum((~valid).to(tl.int32), 0) != 0:
        tl.inline_asm_elementwise("trap; mov.u32 $0, 0;", constraints="=r", args=[],
                                 dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pack_operand(src, dst, rows, COLUMNS: tl.constexpr, ATOMS: tl.constexpr,
                  ROW_ATOMS: tl.constexpr, REP: tl.constexpr, pm, pk):
    group = pm // ROW_ATOMS
    row = (pm % ROW_ATOMS).to(tl.int64) * 128 + tl.arange(0, 128)
    cols = pk * 64 + tl.arange(0, 64)
    values = tl.load(src + group.to(tl.int64) * rows.to(tl.int64) * COLUMNS
                     + row[:, None] * COLUMNS + (cols // REP)[None, :],
                     (row < rows)[:, None] & (cols < COLUMNS * REP)[None, :], other=1.)
    exponent = (values.to(tl.int32, bitcast=True) >> 23) & 255
    w = tl.reshape(exponent, (128, 16, 2, 2))
    lo, hi = tl.split(w)
    b0, b2 = tl.split(lo)
    b1, b3 = tl.split(hi)
    words = b0.to(tl.uint32) | (b1.to(tl.uint32) << 8) | (b2.to(tl.uint32) << 16) | (b3.to(tl.uint32) << 24)
    flat = tl.reshape(tl.trans(tl.reshape(words, (4, 32, 16)), (2, 1, 0)), (2048,))
    index = tl.arange(0, 2048)
    atom = pk * 16 + index // 128
    tl.store(dst + pm.to(tl.int64) * ATOMS * 128 + pk.to(tl.int64) * 2048 + index,
             flat, atom < ATOMS)


@triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["rows"])
def _mark_active_groups(labels, selector, rows, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    row = tl.arange(0, BLOCK).to(tl.int64) * 128
    label = tl.load(labels + row, row < rows, other=-1)
    tl.store(selector + 1 + group, tl.sum((label == group).to(tl.int32), 0) > 0)


@triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["rows"])
def _pack_contiguous(a, b, oa, ob, labels, selector, rows,
                     N: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr,
                     CAPACITY: tl.constexpr, COMPACT: tl.constexpr,
                     SELECTOR: tl.constexpr, CTAS: tl.constexpr, BLOCK: tl.constexpr,
                     ACTIVE_GROUPS: tl.constexpr = False):
    pid = tl.program_id(0)
    ar: tl.constexpr = tl.cdiv(CAPACITY, 128)
    ak: tl.constexpr = K // 128
    br: tl.constexpr = tl.cdiv(N, 128)
    bk: tl.constexpr = tl.cdiv(K, 512) if COMPACT else K // 128
    ap: tl.constexpr = tl.cdiv(ak, 16)
    bp: tl.constexpr = tl.cdiv(bk, 16)
    if pid < ar * ap:
        pm = pid // ap
        if pid % ap == 0:
            row = pm.to(tl.int64) * 128 + tl.arange(0, 128)
            label = tl.load(labels + row, row < rows, other=-1)
            previous = tl.load(labels + row - 1, (row > 0) & (row < rows), other=-2)
            valid = (row >= rows) | ((label >= -1) & (label < GROUPS)
                    & ((label < 0) | (label == previous) | (row % 128 == 0)))
            _trap_if_invalid(valid)
        _pack_operand(a, oa, rows, K // 32, ak, ar, 1, pm, pid % ap)
    else:
        p = pid - ar * ap
        active = True
        if ACTIVE_GROUPS:
            active = tl.load(selector + 1 + p // (br * bp)) != 0
        if active:
            _pack_operand(b, ob, tl.full((), N, tl.int32), K // 128, bk, br,
                          1 if COMPACT else 4, p // bp, p % bp)
    if SELECTOR and pid == 0:
        tile = tl.arange(0, BLOCK)
        row = tile.to(tl.int64) * 128
        group = tl.load(labels + row, row < rows, other=-1)
        prev = tl.load(labels + row - 128, (row >= 128) & (row < rows), other=-2)
        nxt = tl.load(labels + row + 128, row + 128 < rows, other=-2)
        live = (row < rows) & (group >= 0)
        isolated = tl.sum((live & (group != prev) & (group != nxt)).to(tl.int32), 0)
        count = tl.sum(live.to(tl.int32), 0)
        full = count * tl.cdiv(N, 128)
        narrow = count * tl.cdiv(N, 64)
        wf = tl.cdiv(full, CTAS)
        wn = tl.cdiv(narrow, CTAS)
        score = 3500. * (wn - wf) + 585. * (wn * (K // 128) - wf * (K // 64)) - 67. * full * (K // 64) / CTAS
        tl.store(selector, tl.where((count == 0) | (isolated < count) | (score >= 0), 0, 1))


@triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["rows"])
def _zero_flat(d, labels, rows, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = offset < rows.to(tl.int64) * N
    label = tl.load(labels + offset // N, live, other=0)
    tl.store(d + offset, 0, live & (label < 0))


def pack_contiguous(a, b, oa, ob, labels, selector, *, rows, capacity, n, k, groups,
                    compact, use_selector=False, ctas=1):
    if not 0 <= rows <= capacity or ctas < 1:
        raise ValueError("live rows exceed prepared capacity")
    _check_f32_scales("SFA", a, (rows, k // 32))
    b = normalize_g1(b, n, k // 128, groups)
    _check_f32_scales("SFB", b, (n, k // 128) if groups == 1 else (groups, n, k // 128))
    sizes = workspace_sizes(capacity, n, k, groups, compact=compact)
    for dst, size in zip((oa, ob), sizes, strict=True):
        if dst.device != a.device or dst.dtype != torch.uint8 or not dst.is_contiguous() or dst.numel() < size:
            raise ValueError("invalid prepared scale workspace")
    if labels.device != a.device or labels.dtype != torch.int32 or labels.shape != (rows,) or not labels.is_contiguous():
        raise ValueError("labels must be contiguous device int32 rows")
    if selector.device != a.device or selector.dtype != torch.int32 or selector.numel() not in (1, groups + 1):
        raise ValueError("invalid prepared selector")
    active_groups = compact and selector.numel() == groups + 1
    ar = triton.cdiv(capacity, 128)
    ak = k // 128
    br = triton.cdiv(n, 128)
    bk = triton.cdiv(k, 512) if compact else k // 128
    programs = ar * triton.cdiv(ak, 16) + groups * br * triton.cdiv(bk, 16)
    from b12x._lib.compile_plan import compile_only_launches_enabled, launch_triton

    block = triton.next_power_of_2(max(ar, 1))
    marker = None
    if active_groups:
        marker = launch_triton(_mark_active_groups, (groups,), labels, selector, rows, block, num_warps=4)
    compiled = launch_triton(_pack_contiguous, (programs,),
        a, b, oa.view(torch.int32), ob.view(torch.int32), labels,
        selector, rows, n, k, groups, capacity, compact, use_selector, ctas,
        block, ACTIVE_GROUPS=active_groups, num_warps=16)
    return (marker, compiled) if compile_only_launches_enabled() else (oa, ob)


def zero_padding(d, labels, *, capacity):
    if d.ndim != 2 or not d.is_contiguous() or labels.shape != (d.shape[0],):
        raise ValueError("cleanup requires contiguous output and matching labels")
    if d.shape[0] > capacity:
        raise ValueError("cleanup exceeds prepared capacity")
    if d.numel():
        from b12x._lib.compile_plan import compile_only_launches_enabled, launch_triton

        compiled = launch_triton(_zero_flat, (triton.cdiv(capacity * d.shape[1], 4096),),
            d, labels, d.shape[0], d.shape[1], 4096, num_warps=4)
        if compile_only_launches_enabled():
            return compiled
    return d
