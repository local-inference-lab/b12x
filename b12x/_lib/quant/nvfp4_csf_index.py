"""Build native NVFP4 exception-word indices on the source CUDA device."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _exception_masks(
    Exceptions, Bounds, Masks, R: tl.constexpr, C: tl.constexpr, B: tl.constexpr
):
    partition = tl.program_id(0).to(tl.int64)
    expert, slab = partition // (R // 128), partition % (R // 128)
    start = tl.load(Bounds + expert * (R // 128 + 1) + slab)
    end = tl.load(Bounds + expert * (R // 128 + 1) + slab + 1)
    lane = tl.arange(0, B)
    for offset in range(start, end, B):
        record = tl.load(Exceptions + offset + lane, offset + lane < end, 0)
        position = record & 0xFFFFFF
        row, column = position // C, position % C
        word = ((expert * (R // 128) + row // 128) * (C // 4) + column // 4) * 128
        word += row % 32 * 4 + row % 128 // 32
        tl.atomic_or(
            Masks + word // 32,
            (1 << (word % 32)).to(tl.uint32),
            offset + lane < end,
            sem="relaxed",
        )


@tr.jit
def _word_counts(Masks, Counts, tiles, B: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    count = tl.full((B,), 0, tl.int32)
    for group in tl.static_range(4):
        mask = tl.load(Masks + tile * 4 + group, tile < tiles, 0)
        count += tl.inline_asm_elementwise(
            "popc.b32 $0, $1;",
            constraints="=r,r",
            args=[mask],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
    tl.store(Counts + tile, count, tile < tiles)


@tr.jit
def _index_words(
    Fixed,
    Masks,
    Prefix,
    Storage,
    fixed_bytes,
    R: tl.constexpr,
    C: tl.constexpr,
    tiles,
    payload_offset,
):
    tile = tl.program_id(0).to(tl.int64)
    word = tl.arange(0, 128)
    group, bit = word // 32, word % 32
    mask = tl.load(Masks + tile * 4 + group)
    before = mask & ((1 << bit).to(tl.uint32) - 1)
    rank = tl.inline_asm_elementwise(
        "popc.b32 $0, $1;",
        constraints="=r,r",
        args=[before],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    prefix = tl.full((128,), 0, tl.int32)
    packed_prefix = tl.full((), 0, tl.uint32)
    count = tl.full((), 0, tl.int32)
    for index in tl.static_range(4):
        packed_prefix |= count.to(tl.uint32) << (8 * index)
        prefix += tl.where(
            group > index,
            tl.inline_asm_elementwise(
                "popc.b32 $0, $1;",
                constraints="=r,r",
                args=[tl.load(Masks + tile * 4 + index)],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            ),
            0,
        )
        count += tl.inline_asm_elementwise(
            "popc.b32 $0, $1;",
            constraints="=r,r",
            args=[tl.load(Masks + tile * 4 + index)],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
    start = tl.load(Prefix + tile)
    end = tl.load(Prefix + tile + 1)
    metadata = Storage + 1024 + fixed_bytes.to(tl.int64) + tile * 32
    fields = tl.arange(0, 8)
    fields_value = tl.where(
        fields == 0,
        start,
        tl.where(
            fields == 1,
            packed_prefix,
            tl.where(
                fields < 6,
                tl.load(Masks + tile * 4 + fields - 2, (fields >= 2) & (fields < 6), 0),
                tl.where(fields == 6, end, 0),
            ),
        ),
    ).to(tl.uint32)
    tl.store(metadata.to(tl.pointer_type(tl.uint32)) + fields, fields_value)
    selected = (mask & (1 << bit).to(tl.uint32)) != 0
    slab, column = tile // (C // 4), tile % (C // 4)
    fixed_offset = slab * (128 * (1 + C // 2))
    base = tl.load(Fixed + fixed_offset + word, selected, 0).to(tl.uint32)
    codes = tl.load(
        (Fixed + fixed_offset + 128).to(tl.pointer_type(tl.uint16))
        + column * 128
        + word,
        selected,
        0,
    ).to(tl.uint32)
    codes = (codes | (codes << 8)) & 0x00FF00FF
    codes = (codes | (codes << 4)) & 0x0F0F0F0F
    value = codes + base * 0x01010101
    destination = start + (prefix + rank).to(tl.int64)
    tl.store(
        (Storage + payload_offset.to(tl.int64)).to(tl.pointer_type(tl.uint32))
        + destination,
        value,
        selected,
    )


@tr.jit
def _patch_words(
    Exceptions,
    Bounds,
    Masks,
    Prefix,
    Storage,
    R: tl.constexpr,
    C: tl.constexpr,
    payload_offset,
    B: tl.constexpr,
):
    partition = tl.program_id(0).to(tl.int64)
    expert, slab = partition // (R // 128), partition % (R // 128)
    start = tl.load(Bounds + expert * (R // 128 + 1) + slab)
    end = tl.load(Bounds + expert * (R // 128 + 1) + slab + 1)
    lane = tl.arange(0, B)
    for offset in range(start, end, B):
        valid = offset + lane < end
        record = tl.load(Exceptions + offset + lane, valid, 0)
        position = record & 0xFFFFFF
        row, column = position // C, position % C
        tile = (expert * (R // 128) + row // 128) * (C // 4) + column // 4
        word = row % 32 * 4 + row % 128 // 32
        group, bit = word // 32, word % 32
        mask = tl.load(Masks + tile * 4 + group, valid, 0)
        before = mask & ((1 << bit).to(tl.uint32) - 1)
        rank = tl.inline_asm_elementwise(
            "popc.b32 $0, $1;",
            constraints="=r,r",
            args=[before],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        for index in tl.static_range(3):
            prefix_mask = tl.load(Masks + tile * 4 + index, valid & (group > index), 0)
            rank += tl.inline_asm_elementwise(
                "popc.b32 $0, $1;",
                constraints="=r,r",
                args=[prefix_mask],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
        target = tl.load(Prefix + tile, valid, 0) + rank.to(tl.int64)
        tl.store(
            Storage + payload_offset.to(tl.int64) + target * 4 + column % 4,
            record >> 24,
            valid,
        )


def build_nvfp4_index(batch):
    """Return the byte-exact prepared index without copying scale planes to CPU."""
    batch.validate()
    if batch.codec != 0 or batch.layout != 0:
        raise ValueError("Inline NVFP4 requires native-order byte-window scales")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NVFP4 index preparation must precede graph capture")
    e, r, c = batch.num_experts, batch.rows, batch.columns
    tiles = e * r * c // 512
    masks = torch.zeros(tiles * 4, device=batch.fixed.device, dtype=torch.uint32)
    exceptions = (
        batch.exceptions.view(torch.uint32)
        if batch.exceptions.numel()
        else torch.empty(0, device=batch.fixed.device, dtype=torch.uint32)
    )
    _exception_masks[(e * (r // 128),)](
        exceptions, batch.task_offsets, masks, R=r, C=c, B=256
    )
    counts = torch.empty(tiles, device=batch.fixed.device, dtype=torch.int64)
    _word_counts[(tr.cdiv(tiles, 256),)](masks, counts, tiles, B=256)
    prefix = torch.empty(tiles + 1, device=batch.fixed.device, dtype=torch.int64)
    prefix[0] = 0
    torch.cumsum(counts, 0, out=prefix[1:])
    total = int(prefix[-1].item())
    if total >= 1 << 32:
        raise ValueError("NVFP4 inline exception count exceeds uint32 indexing")
    payload_offset = 1024 + batch.fixed.numel() + tiles * 32
    storage = torch.empty(
        payload_offset + tr.cdiv(total, 4) * 16,
        device=batch.fixed.device,
        dtype=torch.uint8,
    )
    storage[:1024].zero_()
    storage[:1024].view(torch.uint32)[167] = 1
    storage[1024 : 1024 + batch.fixed.numel()].copy_(batch.fixed.view(-1))
    storage[payload_offset + total * 4 :].zero_()
    _index_words[(tiles,)](
        batch.fixed,
        masks,
        prefix,
        storage,
        fixed_bytes=batch.fixed.numel(),
        R=r,
        C=c,
        tiles=tiles,
        payload_offset=payload_offset,
        num_warps=4,
    )
    _patch_words[(e * (r // 128),)](
        exceptions,
        batch.task_offsets,
        masks,
        prefix,
        storage,
        R=r,
        C=c,
        payload_offset=payload_offset,
        B=256,
    )
    return storage
