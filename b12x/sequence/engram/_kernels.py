"""Minor GPU metadata and local FP8 gather; no full-table dequantization.

Reuse PLE packed request mapping and lag-source loading. Engram differs at
DEAD (including the current position), and EOS is an ordinary token.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..ple_hash._kernels import _request_ids_kernel, _reset_error_kernel, _source_token


@triton.jit(do_not_specialize=["prepared_tokens"])
def _compress_validate(
    ids,
    token_mask,
    token_map,
    starts,
    slots,
    history,
    num_seqs,
    num_tokens,
    compressed,
    error,
    prepared_tokens,
    T: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    CV: tl.constexpr,
):
    i = tl.program_id(0)
    ns, nt = tl.load(num_seqs), tl.load(num_tokens)
    capacity_ok = (
        (ns >= 0)
        & (ns <= S)
        & (nt >= 0)
        & (nt <= T)
        & (nt <= prepared_tokens)
        & ((ns > 0) | (nt == 0))
    )
    if i == 0:
        tl.atomic_or(error, tl.where(capacity_ok, 0, 1))
    live_seq = (i < S) & (i < ns) & capacity_ok
    begin = tl.load(starts + i, live_seq, 0)
    end = tl.load(starts + i + 1, live_seq, 0)
    slot = tl.load(slots + i, live_seq, 0)
    bad = live_seq & (
        (begin < 0)
        | (end < begin)
        | (end > nt)
        | ((i == 0) & (begin != 0))
        | ((i == ns - 1) & (end != nt))
        | (slot < 0)
        | (slot >= R)
    )
    tl.atomic_or(error, tl.where(bad, 2, 0))
    for lag in tl.static_range(3):
        h = tl.load(
            history + slot.to(tl.int64) * 3 + lag,
            live_seq & (slot >= 0) & (slot < R),
            -1,
        )
        tl.atomic_or(error, tl.where(live_seq & ((h < -1) | (h >= CV)), 4, 0))
    live = (i < T) & (i < nt) & capacity_ok
    raw = tl.load(ids + i, live, 0).to(tl.int64)
    valid = (raw >= 0) & (raw < V)
    tl.atomic_or(error, tl.where(live & ~valid, 4, 0))
    included = tl.load(token_mask + i, live, 0)
    value = tl.load(token_map + raw, live & valid & included, -1)
    tl.store(compressed + i, value, i < prepared_tokens)


@triton.jit
def _hash(
    compressed,
    starts,
    slots,
    history,
    num_tokens,
    request_ids,
    multipliers,
    primes,
    offsets,
    hashes,
    error,
    PAD: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    req = tl.load(request_ids + t)
    live = (t < tl.load(num_tokens)) & (req >= 0) & (tl.load(error) == 0)
    start = tl.load(starts + req, live, 0)
    slot = tl.load(slots + req, live, 0)
    blocked = tl.full((), False, tl.int1)
    mixed = tl.full((), 0, tl.int64)
    order = head // 8 + 2
    for lag in tl.static_range(4):
        source = _source_token(
            compressed, history, start, slot, t - start, -lag, -1, live, MAX_ORDER=4
        )
        blocked |= source == -1
        value = tl.where(blocked, PAD, source).to(tl.int64)
        multiplier = tl.load(multipliers + lag)
        mixed ^= tl.where(lag < order, value * multiplier, 0)
    prime = tl.load(primes + head)
    remainder = mixed % prime
    remainder = tl.where(remainder < 0, remainder + prime, remainder)
    result = remainder + tl.load(offsets + head)
    tl.store(hashes + t.to(tl.int64) * 24 + head, tl.where(live, result, -1))


@triton.jit(do_not_specialize=["prepared_tokens"])
def _lookup(
    weight,
    scales,
    hashes,
    num_tokens,
    out,
    prepared_tokens,
    T: tl.constexpr,
    ROWS: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    COMPACT: tl.constexpr,
    RESIDENT_SCALES: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 256)
    nt = tl.load(num_tokens)
    live = (t < nt) & (t < prepared_tokens) & (nt >= 0) & (nt <= T)
    row = tl.load(hashes + t.to(tl.int64) * 24 + head, live, -1).to(tl.int64)
    local = live & (row >= START) & (row < END) & (row < ROWS)
    if COMPACT:
        local_row = t.to(tl.int64) * 24 + head
    else:
        local_row = tl.where(local, row - tl.full((), START, tl.int64), 0)
    quant = tl.load(weight + local_row * 256 + col.to(tl.int64), local, 0.0).to(
        tl.float32
    )
    scale_row = local_row
    if RESIDENT_SCALES:
        scale_row = tl.where(local, row - tl.full((), START, tl.int64), 0)
    exponent = tl.load(
        scales + scale_row * 8 + (col // 32).to(tl.int64), local, 127
    ).to(tl.uint32)
    # E8M0 byte zero is 2^-127, not floating point zero; 255 is NaN.
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    result = (quant * scale).to(tl.bfloat16)
    tl.store(out + t.to(tl.int64) * 6144 + head * 256 + col, tl.where(local, result, 0))


@torch.library.custom_op(
    "b12x::engram_hash", mutates_args=("compressed", "request_ids", "error", "hashes")
)
def hash_op(
    ids: torch.Tensor,
    token_mask: torch.Tensor,
    token_map: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    compressed: torch.Tensor,
    request_ids: torch.Tensor,
    error: torch.Tensor,
    hashes: torch.Tensor,
    max_requests: int,
    compressed_vocab_size: int,
    pad: int,
    prepared_tokens: int = -1,
) -> None:
    t, s = ids.numel(), starts.numel() - 1
    prepared = t if prepared_tokens < 0 else prepared_tokens
    if not 0 <= prepared <= t:
        raise ValueError("prepared_tokens must be within the hash capacity")
    _reset_error_kernel[(1,)](error, num_warps=1)
    _compress_validate[(max(prepared, s),)](
        ids,
        token_mask,
        token_map,
        starts,
        slots,
        history,
        num_seqs,
        num_tokens,
        compressed,
        error,
        prepared,
        t,
        s,
        max_requests,
        token_map.numel(),
        compressed_vocab_size,
        num_warps=1,
    )
    if prepared:
        _request_ids_kernel[(prepared,)](
            starts, num_seqs, num_tokens, request_ids, error, MAX_TOKENS=t, num_warps=1
        )
        _hash[(prepared, 24)](
            compressed,
            starts,
            slots,
            history,
            num_tokens,
            request_ids,
            multipliers,
            primes,
            offsets,
            hashes,
            error,
            pad,
            num_warps=1,
        )


@hash_op.register_fake
def _hash_fake(
    ids,
    token_mask,
    token_map,
    starts,
    slots,
    history,
    num_seqs,
    num_tokens,
    multipliers,
    primes,
    offsets,
    compressed,
    request_ids,
    error,
    hashes,
    max_requests,
    compressed_vocab_size,
    pad,
    prepared_tokens=-1,
):
    return None


@torch.library.custom_op("b12x::engram_lookup", mutates_args=("out",))
def lookup_op(
    weight: torch.Tensor,
    scale_bytes: torch.Tensor,
    hashes: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    table_rows: int,
    shard_start: int,
    shard_end: int,
    compact_rows: bool = False,
    prepared_tokens: int = -1,
    clear_tail: bool = True,
    resident_scales: bool = False,
) -> None:
    capacity = hashes.shape[0] if prepared_tokens < 0 else prepared_tokens
    if not 0 <= capacity <= hashes.shape[0]:
        raise ValueError("prepared_tokens must be within the lookup capacity")
    if capacity:
        _lookup[(capacity, 24)](
            weight,
            scale_bytes,
            hashes,
            num_tokens,
            out,
            capacity,
            hashes.shape[0],
            table_rows,
            shard_start,
            shard_end,
            compact_rows,
            resident_scales,
            num_warps=4,
        )
    if clear_tail and capacity < hashes.shape[0]:
        out[capacity:].zero_()


@lookup_op.register_fake
def _lookup_fake(
    weight,
    scale_bytes,
    hashes,
    num_tokens,
    out,
    table_rows,
    shard_start,
    shard_end,
    compact_rows=False,
    prepared_tokens=-1,
    clear_tail=True,
    resident_scales=False,
):
    return None
