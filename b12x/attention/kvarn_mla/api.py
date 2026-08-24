"""KVarN packed-latent cache staging and native sparse-MLA decode (Triton).

Reads KVarN G64 tiles (2/4/5-bit packed latent, exact BF16 RoPE) directly:
``stage_*`` kernels rewrite packed or exact rows into the FP8 record format
the promoted SM120 sparse-MLA runtime already consumes, and
native_packed_k5_decode decodes straight from packed pages plus the live
exact-slot pool without a global row stage. The CuTeDSL decode grids that
back the fused path live in ``b12x.attention._shared.mla.kernel``.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..._lib.gating import default_is_supported
from . import META

_K5_GROUP = tl.constexpr(64)
_K5_LATENT_DIM = tl.constexpr(512)
_K5_ROPE_DIM = tl.constexpr(64)
_K5_BITS = 5
_K5_TILE_BYTES = 30_848
_K4_TILE_BYTES = tl.constexpr(26_752)
_K2_TILE_BYTES = tl.constexpr(18_560)
_K5_S_COL_OFFSET = 20_480
_K5_ZP_OFFSET = 21_504
_K5_S_ROW_OFFSET = 22_528
_K5_ROPE_OFFSET = 22_656
_GLM_FP8_RECORD_BYTES = 656
_GLM_FP8_SCALE_OFFSET = 512
_GLM_FP8_ROPE_OFFSET = 528
_FP8_E4M3_MAX = 448.0


def is_kvarn_mla_supported(device=None) -> bool:
    """True on SM120/SM121 with the staging/decode requirements present."""
    return default_is_supported(device, requires=META.requires)
_TL_K5_GROUP = tl.constexpr(_K5_GROUP)
_TL_K5_ROPE_DIM = tl.constexpr(_K5_ROPE_DIM)
_TL_K5_BITS = tl.constexpr(_K5_BITS)
_TL_K5_S_COL_OFFSET = tl.constexpr(_K5_S_COL_OFFSET)
_TL_K5_ZP_OFFSET = tl.constexpr(_K5_ZP_OFFSET)
_TL_K5_S_ROW_OFFSET = tl.constexpr(_K5_S_ROW_OFFSET)
_TL_K5_ROPE_OFFSET = tl.constexpr(_K5_ROPE_OFFSET)
_TL_GLM_FP8_SCALE_OFFSET = tl.constexpr(_GLM_FP8_SCALE_OFFSET)
_TL_GLM_FP8_ROPE_OFFSET = tl.constexpr(_GLM_FP8_ROPE_OFFSET)
_TL_FP8_E4M3_MAX = tl.constexpr(_FP8_E4M3_MAX)
_M4_CHUNKS_PER_SPLIT_ENV = "B12X_KVARN_MLA_M4_CHUNKS_PER_SPLIT"
_EXACT_H16_ENV = "B12X_KVARN_MLA_EXACT_H16"
_M9_HPP4_MERGE_ENV = "B12X_KVARN_MLA_M9_HPP4_MERGE"
_M5_NATIVE_ENV = "B12X_KVARN_MLA_NATIVE_M5"
_MERGE_BLOCK_D_ENV = "B12X_KVARN_MLA_MERGE_BLOCK_D"
_MERGE_BLOCK_D_ALLOWED = (64, 128, 256, 512)
_MERGE_BLOCK_D = int(os.getenv(_MERGE_BLOCK_D_ENV, "64") or "64")
if _MERGE_BLOCK_D not in _MERGE_BLOCK_D_ALLOWED:
    raise ValueError(
        f"{_MERGE_BLOCK_D_ENV} must be one of {_MERGE_BLOCK_D_ALLOWED}, "
        f"got {_MERGE_BLOCK_D}"
    )


def _merge_block_d() -> int:
    """Column tile of the split-merge kernels (64 = the incumbent retile)."""
    return _MERGE_BLOCK_D



def _native_m5_split_family_enabled() -> bool:
    raw = os.environ.get(_M5_NATIVE_ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "on", "yes"}




def _mixed_m4_chunks_per_split() -> int:
    raw = os.environ.get(_M4_CHUNKS_PER_SPLIT_ENV)
    if raw is None:
        return 3
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_M4_CHUNKS_PER_SPLIT_ENV} must be an integer in [1,4]"
        ) from exc
    if not 1 <= value <= 4:
        raise ValueError(
            f"{_M4_CHUNKS_PER_SPLIT_ENV} must be in [1,4], got {value}"
        )
    return value


def _exact_h16_enabled() -> bool:
    raw = os.environ.get(_EXACT_H16_ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "on", "yes"}




@triton.jit
def _unpack_k5(payload, value_indices, mask, bits: tl.constexpr):
    bit_positions = value_indices * bits
    byte_offsets = bit_positions // 8
    shifts = bit_positions % 8
    low = tl.load(payload + byte_offsets, mask=mask, other=0).to(tl.uint32)
    high = tl.load(payload + byte_offsets + 1, mask=mask, other=0).to(tl.uint32)
    return ((low | (high << 8)) >> shifts) & ((1 << bits) - 1)


@triton.jit
def _stage_k5_as_fp8_records_kernel(
    physical_slots_ptr,
    k5_cache_ptr,
    block_to_pool_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    output_slots_ptr,
    raw_latent_ptr,
    raw_rope_ptr,
    cache_stride_block: tl.constexpr,
    latent_pool_stride_slot: tl.constexpr,
    latent_pool_stride_token: tl.constexpr,
    rope_pool_stride_slot: tl.constexpr,
    rope_pool_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    bits: tl.constexpr,
    s_col_offset: tl.constexpr,
    zp_offset: tl.constexpr,
    s_row_offset: tl.constexpr,
    rope_offset: tl.constexpr,
    num_blocks,
    num_pool_slots,
    raw_exact: tl.constexpr,
    scatter_output: tl.constexpr,
):
    row = tl.program_id(0)
    if raw_exact:
        valid = True
    else:
        physical_slot = tl.load(physical_slots_ptr + row).to(tl.int64)
        block = physical_slot // _TL_K5_GROUP
        token = physical_slot % _TL_K5_GROUP
        valid = (physical_slot >= 0) & (block >= 0) & (block < num_blocks)
        safe_block = tl.where(valid, block, 0)
        pool_slot = tl.load(
            block_to_pool_slot_ptr + safe_block,
            mask=valid,
            other=-1,
        )
        valid &= pool_slot < num_pool_slots

    if valid:
        scale_groups = tl.arange(0, 4)
        group_dims = tl.arange(0, 128)
        dims = scale_groups[:, None] * 128 + group_dims[None, :]
        if raw_exact:
            latent = (
                tl.load(raw_latent_ptr + row.to(tl.int64) * 512 + dims)
                .to(tl.float8e4nv)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
        else:
            exact = pool_slot >= 0
            body = ~exact
            safe_pool_slot = tl.maximum(pool_slot, 0).to(tl.int64)
            record = k5_cache_ptr + block * cache_stride_block
            indices = dims * _TL_K5_GROUP + token
            codes = _unpack_k5(record, indices, body, bits).to(tl.float32)
            fp16_record = record.to(tl.pointer_type(tl.float16))
            s_col = tl.load(
                fp16_record + s_col_offset // 2 + dims,
                mask=body,
                other=0.0,
            ).to(tl.float32)
            zero = tl.load(
                fp16_record + zp_offset // 2 + dims,
                mask=body,
                other=0.0,
            ).to(tl.float32)
            s_row = tl.load(
                fp16_record + s_row_offset // 2 + token,
                mask=body,
                other=0.0,
            ).to(tl.float32)
            body_latent = (codes * s_col + zero) * s_row
            exact_latent = tl.load(
                latent_pool_ptr
                + safe_pool_slot * latent_pool_stride_slot
                + token * latent_pool_stride_token
                + dims,
                mask=exact,
                other=0.0,
            ).to(tl.float32)
            latent = (
                tl.where(exact, exact_latent, body_latent)
                .to(tl.bfloat16)
                .to(tl.float32)
            )

        amax = tl.max(tl.abs(latent), axis=1)
        scales = tl.where(amax > 0.0, amax / _TL_FP8_E4M3_MAX, 1.0)
        quantized = tl.maximum(
            tl.minimum(latent / scales[:, None], _TL_FP8_E4M3_MAX),
            -_TL_FP8_E4M3_MAX,
        ).to(tl.float8e4nv)
        output_row = (
            (tl.load(output_slots_ptr + row).to(tl.int64) if scatter_output
             else row.to(tl.int64))
        )
        output_record = output_ptr + output_row * output_stride_row
        fp8_output = output_record.to(tl.pointer_type(tl.float8e4nv))
        tl.store(fp8_output + dims, quantized)
        fp32_output = output_record.to(tl.pointer_type(tl.float32))
        tl.store(
            fp32_output + _TL_GLM_FP8_SCALE_OFFSET // 4 + scale_groups,
            scales,
        )

        rope_dims = tl.arange(0, _TL_K5_ROPE_DIM)
        if raw_exact:
            rope = tl.load(
                raw_rope_ptr + row.to(tl.int64) * _TL_K5_ROPE_DIM + rope_dims
            )
        else:
            body_rope = tl.load(
                (record + rope_offset).to(tl.pointer_type(tl.bfloat16))
                + token * _TL_K5_ROPE_DIM
                + rope_dims,
                mask=body,
                other=0.0,
            )
            exact_rope = tl.load(
                rope_pool_ptr
                + safe_pool_slot * rope_pool_stride_slot
                + token * rope_pool_stride_token
                + rope_dims,
                mask=exact,
                other=0.0,
            )
            rope = tl.where(exact, exact_rope, body_rope)
        tl.store(
            (output_record + _TL_GLM_FP8_ROPE_OFFSET).to(
                tl.pointer_type(tl.bfloat16)
            )
            + rope_dims,
            rope,
        )


_NATIVE_EXACT_PAGES_PER_REQ = tl.constexpr(16)
_NATIVE_SINK_PAGES_PER_REQ = tl.constexpr(2)
_NATIVE_TAIL_PAGES_PER_REQ = tl.constexpr(14)
_NATIVE_EXACT_PAGE_BYTES = (
    int(_K5_GROUP) * int(_K5_LATENT_DIM) + int(_K5_GROUP) * int(_K5_ROPE_DIM) * 2
)


def compact_kvarn_native_rank_nbytes(
    padded_pages: int,
    padded_exact_pages: int,
) -> int:
    if padded_pages < 0 or padded_exact_pages < 0:
        raise ValueError("native CKV page capacities must be non-negative")
    return (
        int(padded_pages) * (int(_K4_TILE_BYTES) + 4)
        + int(padded_exact_pages) * _NATIVE_EXACT_PAGE_BYTES
    )


@triton.jit
def _native_page_lookup(
    page,
    page_starts_ptr,
    page_lens_ptr,
    num_reqs,
):
    req_id = 0
    local_page = 0
    page_len = 0
    valid = False
    for req in tl.range(0, num_reqs):
        start = tl.load(page_starts_ptr + req)
        length = tl.load(page_lens_ptr + req)
        match = (page >= start) & (page < start + length)
        req_id = tl.where(match, req, req_id)
        local_page = tl.where(match, page - start, local_page)
        page_len = tl.where(match, length, page_len)
        valid |= match
    return req_id, local_page, page_len, valid


@triton.jit
def _native_exact_id(req_id, local_page, page_len, exact):
    tail_start = tl.maximum(
        page_len - _NATIVE_TAIL_PAGES_PER_REQ,
        _NATIVE_SINK_PAGES_PER_REQ,
    )
    sink = local_page < _NATIVE_SINK_PAGES_PER_REQ
    tail = local_page >= tail_start
    mapped = sink | tail
    tl.device_assert(~exact | mapped, "exact CKV page is outside sink/tail window")
    within_req = tl.where(
        sink,
        local_page,
        _NATIVE_SINK_PAGES_PER_REQ + local_page - tail_start,
    )
    tl.device_assert(
        ~exact | (within_req < _NATIVE_EXACT_PAGES_PER_REQ),
        "exact CKV page capacity exceeded",
    )
    in_window = mapped & (within_req >= 0) & (
        within_req < _NATIVE_EXACT_PAGES_PER_REQ
    )
    return in_window, req_id * _NATIVE_EXACT_PAGES_PER_REQ + within_req


@triton.jit
def _stage_compact_kvarn_native_index_kernel(
    block_table_ptr,
    page_starts_ptr,
    page_lens_ptr,
    block_to_pool_slot_ptr,
    wire_ptr,
    block_table_stride,
    padded_pages: tl.constexpr,
    padded_exact_pages: tl.constexpr,
    num_reqs,
    num_blocks,
    num_pool_slots,
):
    page = tl.program_id(0)
    req, local_page, page_len, valid = _native_page_lookup(
        page,
        page_starts_ptr,
        page_lens_ptr,
        num_reqs,
    )
    active = valid
    block = tl.load(
        block_table_ptr + req * block_table_stride + local_page,
        mask=active,
        other=-1,
    )
    valid = active & (block >= 0) & (block < num_blocks)
    tl.device_assert(~active | valid, "invalid native CKV block")
    pool_slot = tl.load(
        block_to_pool_slot_ptr + tl.maximum(block, 0),
        mask=valid,
        other=-1,
    )
    tl.device_assert(
        ~valid | (pool_slot < num_pool_slots),
        "native CKV exact pool mapping is out of range",
    )
    pool_mapped = valid & (pool_slot >= 0)
    in_window, exact_id = _native_exact_id(req, local_page, page_len, pool_mapped)
    exact = pool_mapped & in_window
    tl.device_assert(
        ~exact | (exact_id < padded_exact_pages),
        "native CKV exact wire capacity exceeded",
    )
    index_ptr = (
        wire_ptr + padded_pages * _K4_TILE_BYTES
    ).to(tl.pointer_type(tl.int32))
    tl.store(index_ptr + page, tl.where(exact, exact_id, -1))


@triton.jit
def _stage_compact_kvarn_native_packed_kernel(
    block_table_ptr,
    page_starts_ptr,
    page_lens_ptr,
    k4_cache_ptr,
    wire_ptr,
    block_table_stride,
    num_reqs,
    num_blocks,
    padded_pages: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    chunk = tl.program_id(1)
    req, local_page, _, valid = _native_page_lookup(
        page,
        page_starts_ptr,
        page_lens_ptr,
        num_reqs,
    )
    block = tl.load(
        block_table_ptr + req * block_table_stride + local_page,
        mask=valid,
        other=-1,
    )
    valid &= (block >= 0) & (block < num_blocks)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = valid & (offsets < _K4_TILE_BYTES)
    values = tl.load(
        k4_cache_ptr + tl.maximum(block, 0).to(tl.int64) * _K4_TILE_BYTES
        + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        wire_ptr + page.to(tl.int64) * _K4_TILE_BYTES + offsets, values, mask=mask
    )


@triton.jit
def _stage_compact_kvarn_native_exact_kernel(
    block_table_ptr,
    page_starts_ptr,
    page_lens_ptr,
    block_to_pool_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    wire_ptr,
    block_table_stride,
    num_reqs,
    num_blocks,
    num_pool_slots,
    padded_pages: tl.constexpr,
    padded_exact_pages: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    chunk = tl.program_id(1)
    req, local_page, page_len, valid = _native_page_lookup(
        page,
        page_starts_ptr,
        page_lens_ptr,
        num_reqs,
    )
    block = tl.load(
        block_table_ptr + req * block_table_stride + local_page,
        mask=valid,
        other=-1,
    )
    valid &= (block >= 0) & (block < num_blocks)
    pool_slot = tl.load(
        block_to_pool_slot_ptr + tl.maximum(block, 0),
        mask=valid,
        other=-1,
    )
    pool_mapped = valid & (pool_slot >= 0) & (pool_slot < num_pool_slots)
    in_window, exact_id = _native_exact_id(req, local_page, page_len, pool_mapped)
    exact = pool_mapped & in_window
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    latent_bytes = _K5_GROUP * _K5_LATENT_DIM
    rope_bytes = _K5_GROUP * _K5_ROPE_DIM * 2
    latent_mask = exact & (offsets < latent_bytes)
    rope_mask = exact & (offsets >= latent_bytes) & (
        offsets < latent_bytes + rope_bytes
    )
    latent_values = tl.load(
        latent_pool_ptr.to(tl.pointer_type(tl.uint8))
        + pool_slot.to(tl.int64) * latent_bytes
        + offsets,
        mask=latent_mask,
        other=0,
    )
    rope_offsets = offsets - latent_bytes
    rope_values = tl.load(
        rope_pool_ptr.to(tl.pointer_type(tl.uint8))
        + pool_slot * rope_bytes
        + rope_offsets,
        mask=rope_mask,
        other=0,
    )
    latent_base = padded_pages * (_K4_TILE_BYTES + 4)
    rope_base = latent_base + padded_exact_pages * latent_bytes
    tl.store(
        wire_ptr
        + latent_base
        + exact_id.to(tl.int64) * latent_bytes
        + offsets,
        latent_values,
        mask=latent_mask,
    )
    tl.store(
        wire_ptr
        + rope_base
        + exact_id.to(tl.int64) * rope_bytes
        + rope_offsets,
        rope_values,
        mask=rope_mask,
    )


@triton.jit
def _materialize_compact_kvarn_native_records_kernel(
    gathered_wire_ptr,
    rank_page_starts_ptr,
    rank_page_lens_ptr,
    rank_token_starts_ptr,
    rank_token_lens_ptr,
    output_ptr,
    num_reqs,
    page_stride_rank,
    page_stride_req,
    token_stride_rank,
    token_stride_req,
    padded_tokens: tl.constexpr,
    padded_pages: tl.constexpr,
    padded_exact_pages: tl.constexpr,
    rank_wire_bytes: tl.constexpr,
    output_stride_row: tl.constexpr,
):
    row = tl.program_id(0)
    rank = row // padded_tokens
    rank_token = row % padded_tokens
    req_id = 0
    local_token = 0
    valid = False
    for req in tl.range(0, num_reqs):
        token_offset = rank * token_stride_rank + req * token_stride_req
        start = tl.load(rank_token_starts_ptr + token_offset)
        length = tl.load(rank_token_lens_ptr + token_offset)
        match = (rank_token >= start) & (rank_token < start + length)
        req_id = tl.where(match, req, req_id)
        local_token = tl.where(match, rank_token - start, local_token)
        valid |= match
    page_offset = rank * page_stride_rank + req_id * page_stride_req
    page = (
        tl.load(rank_page_starts_ptr + page_offset)
        + local_token // _K5_GROUP
    )
    tl.device_assert(~valid | (page < padded_pages), "native CKV page overflow")
    token = local_token % _K5_GROUP
    rank_wire = gathered_wire_ptr + rank.to(tl.int64) * rank_wire_bytes
    index_ptr = (
        rank_wire + padded_pages * _K4_TILE_BYTES
    ).to(tl.pointer_type(tl.int32))
    exact_id = tl.load(index_ptr + page, mask=valid, other=-1)
    tl.device_assert(
        ~valid
        | (exact_id == -1)
        | ((exact_id >= 0) & (exact_id < padded_exact_pages)),
        "native CKV exact index is invalid",
    )
    exact = valid & (exact_id >= 0) & (exact_id < padded_exact_pages)
    body = valid & ~exact
    scale_groups = tl.arange(0, 4)
    group_dims = tl.arange(0, 128)
    dims = scale_groups[:, None] * 128 + group_dims[None, :]
    packed_record = rank_wire + page.to(tl.int64) * _K4_TILE_BYTES
    indices = dims * _K5_GROUP + token
    codes = _unpack_k5(packed_record, indices, body, 4).to(tl.float32)
    fp16_record = packed_record.to(tl.pointer_type(tl.float16))
    s_col = tl.load(
        fp16_record + 16_384 // 2 + dims,
        mask=body,
        other=0.0,
    ).to(tl.float32)
    zero = tl.load(
        fp16_record + 17_408 // 2 + dims,
        mask=body,
        other=0.0,
    ).to(tl.float32)
    s_row = tl.load(
        fp16_record + 18_432 // 2 + token,
        mask=body,
        other=0.0,
    ).to(tl.float32)
    body_latent = (codes * s_col + zero) * s_row
    latent_bytes = _K5_GROUP * _K5_LATENT_DIM
    rope_bytes = _K5_GROUP * _K5_ROPE_DIM * 2
    latent_base = padded_pages * (_K4_TILE_BYTES + 4)
    rope_base = latent_base + padded_exact_pages * latent_bytes
    exact_latent = tl.load(
        (rank_wire + latent_base).to(tl.pointer_type(tl.float8e4nv))
        + exact_id.to(tl.int64) * latent_bytes
        + token * _K5_LATENT_DIM
        + dims,
        mask=exact,
        other=0.0,
    ).to(tl.float32)
    latent = (
        tl.where(exact, exact_latent, body_latent)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    amax = tl.max(tl.abs(latent), axis=1)
    scales = tl.where(amax > 0.0, amax / _TL_FP8_E4M3_MAX, 1.0)
    quantized = tl.maximum(
        tl.minimum(latent / scales[:, None], _TL_FP8_E4M3_MAX),
        -_TL_FP8_E4M3_MAX,
    ).to(tl.float8e4nv)
    output_record = output_ptr + row.to(tl.int64) * output_stride_row
    tl.store(
        output_record.to(tl.pointer_type(tl.float8e4nv)) + dims,
        quantized,
        mask=valid,
    )
    tl.store(
        output_record.to(tl.pointer_type(tl.float32))
        + _TL_GLM_FP8_SCALE_OFFSET // 4
        + scale_groups,
        scales,
        mask=valid,
    )
    rope_dims = tl.arange(0, _K5_ROPE_DIM)
    body_rope = tl.load(
        (packed_record + 18_560).to(tl.pointer_type(tl.bfloat16))
        + token * _K5_ROPE_DIM
        + rope_dims,
        mask=body,
        other=0.0,
    )
    exact_rope = tl.load(
        (rank_wire + rope_base).to(tl.pointer_type(tl.bfloat16))
        + exact_id * (_K5_GROUP * _K5_ROPE_DIM)
        + token * _K5_ROPE_DIM
        + rope_dims,
        mask=exact,
        other=0.0,
    )
    tl.store(
        (output_record + _TL_GLM_FP8_ROPE_OFFSET).to(
            tl.pointer_type(tl.bfloat16)
        )
        + rope_dims,
        tl.where(exact, exact_rope, body_rope),
        mask=valid,
    )

@triton.jit
def _stage_bf16_sylvester_as_exact_pool_fp8_records_kernel(
    raw_latent_ptr,
    raw_rope_ptr,
    output_slots_ptr,
    output_ptr,
    output_stride_row: tl.constexpr,
):
    """Fuse the BF16 Hadacore factorization with canonical record staging."""
    row = tl.program_id(0)
    axis = tl.arange(0, 16)
    h_row = axis[:, None]
    h_col = axis[None, :]
    parity = (
        (h_row & 1) * (h_col & 1)
        + ((h_row >> 1) & 1) * ((h_col >> 1) & 1)
        + ((h_row >> 2) & 1) * ((h_col >> 2) & 1)
        + ((h_row >> 3) & 1) * ((h_col >> 3) & 1)
    ) & 1
    h16 = tl.where(parity == 0, 0.25, -0.25).to(tl.bfloat16)

    # Hadacore's 512-wide BF16 path is H16 x H16 x H2.  Each MMA
    # factor is rounded back to BF16 before the next factor.
    dims = h_row * 16 + h_col
    raw_low = tl.load(raw_latent_ptr + row * 512 + dims)
    raw_high = tl.load(raw_latent_ptr + row * 512 + 256 + dims)
    low = tl.dot(raw_low, h16).to(tl.bfloat16)
    high = tl.dot(raw_high, h16).to(tl.bfloat16)
    low = tl.dot(h16, low).to(tl.bfloat16)
    high = tl.dot(h16, high).to(tl.bfloat16)

    h2_scale = tl.full((), 0.70703125, tl.bfloat16).to(tl.float32)
    low_fp32 = low.to(tl.float32)
    high_fp32 = high.to(tl.float32)
    rotated_low = (
        low_fp32 * h2_scale + high_fp32 * h2_scale
    ).to(tl.bfloat16)
    rotated_high = (
        low_fp32 * h2_scale - high_fp32 * h2_scale
    ).to(tl.bfloat16)

    # The exact pool first rounds stored BF16 latent through E4M3, then
    # derives four independent 128-element scales from that rounded value.
    exact_low = (
        rotated_low.to(tl.float8e4nv)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    exact_high = (
        rotated_high.to(tl.float8e4nv)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    exact_low = tl.reshape(exact_low, (2, 128))
    exact_high = tl.reshape(exact_high, (2, 128))
    low_amax = tl.max(tl.abs(exact_low), axis=1)
    high_amax = tl.max(tl.abs(exact_high), axis=1)
    low_scales = tl.where(
        low_amax > 0.0, low_amax / _TL_FP8_E4M3_MAX, 1.0
    )
    high_scales = tl.where(
        high_amax > 0.0, high_amax / _TL_FP8_E4M3_MAX, 1.0
    )
    low_quantized = tl.maximum(
        tl.minimum(exact_low / low_scales[:, None], _TL_FP8_E4M3_MAX),
        -_TL_FP8_E4M3_MAX,
    ).to(tl.float8e4nv)
    high_quantized = tl.maximum(
        tl.minimum(exact_high / high_scales[:, None], _TL_FP8_E4M3_MAX),
        -_TL_FP8_E4M3_MAX,
    ).to(tl.float8e4nv)

    output_row = tl.load(output_slots_ptr + row)
    output_record = output_ptr + output_row * output_stride_row
    fp8_output = output_record.to(tl.pointer_type(tl.float8e4nv))
    half_dims = tl.arange(0, 256)
    tl.store(
        fp8_output + half_dims,
        tl.reshape(low_quantized, (256,)),
    )
    tl.store(
        fp8_output + 256 + half_dims,
        tl.reshape(high_quantized, (256,)),
    )
    scale_groups = tl.arange(0, 2)
    fp32_output = output_record.to(tl.pointer_type(tl.float32))
    tl.store(
        fp32_output + _TL_GLM_FP8_SCALE_OFFSET // 4 + scale_groups,
        low_scales,
    )
    tl.store(
        fp32_output + _TL_GLM_FP8_SCALE_OFFSET // 4 + 2 + scale_groups,
        high_scales,
    )

    rope_dims = tl.arange(0, _TL_K5_ROPE_DIM)
    rope = tl.load(raw_rope_ptr + row * _TL_K5_ROPE_DIM + rope_dims)
    tl.store(
        (output_record + _TL_GLM_FP8_ROPE_OFFSET).to(
            tl.pointer_type(tl.bfloat16)
        )
        + rope_dims,
        rope,
    )

def stage_compact_kvarn_native_history(
    block_table: torch.Tensor,
    page_starts: torch.Tensor,
    page_lens: torch.Tensor,
    k4_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    wire: torch.Tensor,
    *,
    padded_pages: int,
    padded_exact_pages: int,
) -> None:
    """Pack one rank's live K4 pages and exact overrides into native wire."""
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or not block_table.is_contiguous()
    ):
        raise ValueError("block_table must be contiguous rank-2 int32")
    for tensor, name in (
        (page_starts, "page_starts"),
        (page_lens, "page_lens"),
    ):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous flat int32")
    if page_starts.shape != page_lens.shape or page_starts.numel() == 0:
        raise ValueError("page starts/lens must have equal nonzero length")
    if (
        k4_cache.dtype != torch.uint8
        or k4_cache.ndim != 3
        or not k4_cache.is_contiguous()
        or k4_cache.stride(0) != int(_K4_TILE_BYTES)
    ):
        raise ValueError("native CKV requires contiguous K4 pages with stride 26752")
    if (
        block_to_pool_slot.dtype != torch.int32
        or block_to_pool_slot.ndim != 1
        or not block_to_pool_slot.is_contiguous()
        or block_to_pool_slot.numel() != k4_cache.shape[0]
    ):
        raise ValueError("block_to_pool_slot must map every K4 cache page")
    if (
        latent_pool.dtype != torch.float8_e4m3fn
        or latent_pool.ndim != 3
        or tuple(latent_pool.shape[1:]) != (int(_K5_GROUP), int(_K5_LATENT_DIM))
        or not latent_pool.is_contiguous()
    ):
        raise ValueError("native CKV latent pool must be contiguous E4M3 [P,64,512]")
    if (
        rope_pool.dtype != torch.bfloat16
        or rope_pool.ndim != 3
        or tuple(rope_pool.shape[1:]) != (_K5_GROUP, _K5_ROPE_DIM)
        or not rope_pool.is_contiguous()
        or rope_pool.shape[0] != latent_pool.shape[0]
    ):
        raise ValueError("native CKV RoPE pool must be contiguous BF16 [P,64,64]")
    padded_pages = int(padded_pages)
    padded_exact_pages = int(padded_exact_pages)
    expected_nbytes = compact_kvarn_native_rank_nbytes(
        padded_pages, padded_exact_pages
    )
    if (
        wire.dtype != torch.uint8
        or wire.ndim != 1
        or not wire.is_contiguous()
        or wire.numel() != expected_nbytes
    ):
        raise ValueError(
            f"native CKV wire must be flat uint8 with {expected_nbytes} bytes"
        )
    tensors = (
        block_table,
        page_starts,
        page_lens,
        k4_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        wire,
    )
    if any(tensor.device != wire.device for tensor in tensors):
        raise ValueError("all native CKV staging tensors must share one device")
    if wire.device.type != "cuda":
        raise ValueError("native CKV staging requires CUDA tensors")
    if wire.data_ptr() % 16:
        raise ValueError("native CKV wire must be 16-byte aligned")
    if padded_pages <= 0:
        raise ValueError("native CKV padded page count must be positive")
    required_exact = page_starts.numel() * int(_NATIVE_EXACT_PAGES_PER_REQ)
    if padded_exact_pages < required_exact:
        raise ValueError(
            "native CKV exact capacity must provide 16 pages per request"
        )

    num_reqs = page_starts.numel()
    _stage_compact_kvarn_native_index_kernel[(padded_pages,)](
        block_table,
        page_starts,
        page_lens,
        block_to_pool_slot,
        wire,
        block_table.stride(0),
        padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
        num_reqs=num_reqs,
        num_blocks=k4_cache.shape[0],
        num_pool_slots=latent_pool.shape[0],
        num_warps=1,
    )
    block = 256
    _stage_compact_kvarn_native_packed_kernel[
        (padded_pages, triton.cdiv(int(_K4_TILE_BYTES), block))
    ](
        block_table,
        page_starts,
        page_lens,
        k4_cache,
        wire,
        block_table.stride(0),
        num_reqs,
        k4_cache.shape[0],
        padded_pages=padded_pages,
        BLOCK=block,
        num_warps=4,
    )
    _stage_compact_kvarn_native_exact_kernel[
        (padded_pages, triton.cdiv(_NATIVE_EXACT_PAGE_BYTES, block))
    ](
        block_table,
        page_starts,
        page_lens,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        wire,
        block_table.stride(0),
        num_reqs,
        k4_cache.shape[0],
        latent_pool.shape[0],
        padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
        BLOCK=block,
        num_warps=4,
    )


def materialize_compact_kvarn_native_records(
    gathered_wire: torch.Tensor,
    rank_page_starts: torch.Tensor,
    rank_page_lens: torch.Tensor,
    rank_token_starts: torch.Tensor,
    rank_token_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    padded_tokens: int,
    padded_pages: int,
    padded_exact_pages: int,
) -> None:
    """Reproduce canonical FP8 records from the compact native CKV wire."""
    matrices = (
        (rank_page_starts, "rank_page_starts"),
        (rank_page_lens, "rank_page_lens"),
        (rank_token_starts, "rank_token_starts"),
        (rank_token_lens, "rank_token_lens"),
    )
    shape = rank_page_starts.shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("native CKV rank metadata must have shape [D,R]")
    for tensor, name in matrices:
        if tensor.dtype != torch.int32 or tensor.ndim != 2:
            raise ValueError(f"{name} must be rank-2 int32")
        if tensor.shape != shape:
            raise ValueError("native CKV rank metadata shapes must match")
    dcp_world_size, num_reqs = map(int, shape)
    padded_tokens = int(padded_tokens)
    padded_pages = int(padded_pages)
    padded_exact_pages = int(padded_exact_pages)
    rank_wire_bytes = compact_kvarn_native_rank_nbytes(
        padded_pages, padded_exact_pages
    )
    if (
        gathered_wire.dtype != torch.uint8
        or gathered_wire.ndim != 1
        or not gathered_wire.is_contiguous()
        or gathered_wire.numel() != dcp_world_size * rank_wire_bytes
    ):
        raise ValueError("gathered native CKV wire has invalid geometry")
    if (
        output.dtype != torch.uint8
        or output.ndim != 2
        or output.shape
        != (dcp_world_size * padded_tokens, _GLM_FP8_RECORD_BYTES)
        or not output.is_contiguous()
    ):
        raise ValueError("native CKV output must be canonical contiguous records")
    tensors = (
        gathered_wire,
        rank_page_starts,
        rank_page_lens,
        rank_token_starts,
        rank_token_lens,
        output,
    )
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all native CKV reader tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("native CKV reader requires CUDA tensors")
    if output.data_ptr() % 16:
        raise ValueError("native CKV output must be 16-byte aligned")
    _materialize_compact_kvarn_native_records_kernel[
        (dcp_world_size * padded_tokens,)
    ](
        gathered_wire,
        rank_page_starts,
        rank_page_lens,
        rank_token_starts,
        rank_token_lens,
        output,
        num_reqs,
        rank_page_starts.stride(0),
        rank_page_starts.stride(1),
        rank_token_starts.stride(0),
        rank_token_starts.stride(1),
        padded_tokens=padded_tokens,
        padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
        rank_wire_bytes=rank_wire_bytes,
        output_stride_row=output.stride(0),
        num_warps=4,
    )



def stage_k5_as_fp8_records(
    physical_slots: torch.Tensor,
    k5_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Stage packed KVarN MLA tiles as SparkInfer GLM FP8 cache records.

    Invalid physical slots leave the corresponding output rows unchanged.
    """
    if physical_slots.ndim != 1 or physical_slots.dtype != torch.int32:
        raise ValueError("physical_slots must be a flat int32 tensor")
    if not physical_slots.is_contiguous():
        raise ValueError("physical_slots must be contiguous")
    if k5_cache.dtype != torch.uint8 or k5_cache.ndim != 3:
        raise ValueError("k5_cache must be a rank-3 uint8 tensor")
    if not k5_cache.is_contiguous():
        raise ValueError("k5_cache must be contiguous")
    geometry = {
        18_560: (2, 8_192, 9_216, 10_240, 10_368),
        26_752: (4, 16_384, 17_408, 18_432, 18_560),
        30_848: (5, 20_480, 21_504, 22_528, 22_656),
    }.get(k5_cache.stride(0))
    if geometry is None:
        raise ValueError(
            "KVarN cache block stride must be 18560, 26752, or 30848 bytes, "
            f"got {k5_cache.stride(0)}"
        )
    if block_to_pool_slot.ndim != 1 or block_to_pool_slot.dtype != torch.int32:
        raise ValueError("block_to_pool_slot must be a flat int32 tensor")
    if not block_to_pool_slot.is_contiguous():
        raise ValueError("block_to_pool_slot must be contiguous")
    if block_to_pool_slot.numel() != k5_cache.shape[0]:
        raise ValueError("block_to_pool_slot must contain one entry per cache block")
    if latent_pool.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("latent_pool must use BF16 or float8_e4m3fn")
    if latent_pool.ndim != 3 or tuple(latent_pool.shape[1:]) != (
        _K5_GROUP,
        _K5_LATENT_DIM,
    ):
        raise ValueError("latent_pool must have shape (slots,64,512)")
    if not latent_pool.is_contiguous():
        raise ValueError("latent_pool must be contiguous")
    if rope_pool.dtype != torch.bfloat16 or rope_pool.ndim != 3:
        raise ValueError("rope_pool must be a rank-3 BF16 tensor")
    if tuple(rope_pool.shape[1:]) != (_K5_GROUP, _K5_ROPE_DIM):
        raise ValueError("rope_pool must have shape (slots,64,64)")
    if not rope_pool.is_contiguous():
        raise ValueError("rope_pool must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("latent_pool and rope_pool must have equal slot capacities")
    if output.dtype != torch.uint8 or output.ndim != 2:
        raise ValueError("output must be a rank-2 uint8 tensor")
    if output.shape != (physical_slots.numel(), _GLM_FP8_RECORD_BYTES):
        raise ValueError(
            f"output must have shape ({physical_slots.numel()},"
            f"{_GLM_FP8_RECORD_BYTES})"
        )
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    tensors = (
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
    )
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all KVarN5 staging tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("KVarN5 staging tensors must be CUDA tensors")
    aligned_tensors = (
        (physical_slots, 4, "physical_slots"),
        (k5_cache, 2, "k5_cache"),
        (block_to_pool_slot, 4, "block_to_pool_slot"),
        (latent_pool, latent_pool.element_size(), "latent_pool"),
        (rope_pool, 2, "rope_pool"),
        (output, 16, "output"),
    )
    for tensor, alignment, name in aligned_tensors:
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    if physical_slots.numel() == 0:
        return

    _stage_k5_as_fp8_records_kernel[(physical_slots.numel(),)](
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
        physical_slots,
        latent_pool,
        rope_pool,
        k5_cache.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        output.stride(0),
        bits=geometry[0],
        s_col_offset=geometry[1],
        zp_offset=geometry[2],
        s_row_offset=geometry[3],
        rope_offset=geometry[4],
        num_blocks=k5_cache.shape[0],
        num_pool_slots=latent_pool.shape[0],
        raw_exact=False,
        scatter_output=False,
        num_warps=4,
    )

def stage_bf16_as_exact_pool_fp8_records(
    latent: torch.Tensor,
    rope: torch.Tensor,
    output_slots: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Stage full raw rows with the exact-pool FP8 round and canonical record."""
    if latent.dtype != torch.bfloat16 or latent.ndim != 2:
        raise ValueError("latent must be a rank-2 BF16 tensor")
    if latent.shape[1] != _K5_LATENT_DIM or not latent.is_contiguous():
        raise ValueError("latent must be contiguous with shape (rows,512)")
    if rope.dtype != torch.bfloat16 or rope.ndim != 2:
        raise ValueError("rope must be a rank-2 BF16 tensor")
    if rope.shape != (latent.shape[0], _K5_ROPE_DIM) or not rope.is_contiguous():
        raise ValueError("rope must be contiguous with shape (rows,64)")
    if output_slots.ndim != 1 or output_slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("output_slots must be a flat int32 or int64 tensor")
    if output_slots.numel() != latent.shape[0] or not output_slots.is_contiguous():
        raise ValueError("output_slots must contain one contiguous slot per row")
    if (
        output.dtype != torch.uint8
        or output.ndim != 2
        or output.shape[1] != _GLM_FP8_RECORD_BYTES
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous uint8 records with width 656")
    tensors = (latent, rope, output_slots, output)
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all direct-current staging tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("direct-current staging tensors must be CUDA tensors")
    if output.data_ptr() % 16:
        raise ValueError("output must be 16-byte aligned")
    if latent.shape[0] == 0:
        return

    _stage_k5_as_fp8_records_kernel[(latent.shape[0],)](
        output_slots,
        output,
        output_slots,
        latent,
        rope,
        output,
        output_slots,
        latent,
        rope,
        output.stride(0),
        latent.stride(0),
        latent.stride(1),
        rope.stride(0),
        rope.stride(1),
        output.stride(0),
        bits=4,
        s_col_offset=0,
        zp_offset=0,
        s_row_offset=0,
        rope_offset=0,
        num_blocks=0,
        num_pool_slots=0,
        raw_exact=True,
        scatter_output=True,
        num_warps=4,
    )

def stage_bf16_sylvester_as_exact_pool_fp8_records(
    latent: torch.Tensor,
    rope: torch.Tensor,
    output_slots: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fuse the fixed GLM Sylvester transform with canonical FP8 staging.

    This explicit opt-in candidate implements the normalized 512-wide
    Sylvester transform used by the GLM KVarN CUDA path.  It intentionally
    accepts no matrix operand: alternate transforms are unsupported and cannot
    be mistaken for the fixed transform compiled into the kernel.
    """
    if latent.dtype != torch.bfloat16 or latent.ndim != 2:
        raise ValueError("latent must be a rank-2 BF16 tensor")
    if latent.shape[1] != _K5_LATENT_DIM or not latent.is_contiguous():
        raise ValueError("latent must be contiguous with shape (rows,512)")
    if rope.dtype != torch.bfloat16 or rope.ndim != 2:
        raise ValueError("rope must be a rank-2 BF16 tensor")
    if rope.shape != (latent.shape[0], _K5_ROPE_DIM) or not rope.is_contiguous():
        raise ValueError("rope must be contiguous with shape (rows,64)")
    if output_slots.ndim != 1 or output_slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("output_slots must be a flat int32 or int64 tensor")
    if output_slots.numel() != latent.shape[0] or not output_slots.is_contiguous():
        raise ValueError("output_slots must contain one contiguous slot per row")
    if (
        output.dtype != torch.uint8
        or output.ndim != 2
        or output.shape[1] != _GLM_FP8_RECORD_BYTES
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous uint8 records with width 656")
    tensors = (latent, rope, output_slots, output)
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all fused-current staging tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("fused-current staging tensors must be CUDA tensors")
    if output.data_ptr() % 16:
        raise ValueError("output must be 16-byte aligned")
    if latent.shape[0] == 0:
        return

    _stage_bf16_sylvester_as_exact_pool_fp8_records_kernel[
        (latent.shape[0],)
    ](
        latent,
        rope,
        output_slots,
        output,
        output.stride(0),
        num_warps=4,
    )



@triton.jit
def _native_k5_fused_merge_kernel(
    split_o, split_lse, output, output_lse,
    so_m: tl.constexpr, so_h: tl.constexpr, so_s: tl.constexpr,
    sl_m: tl.constexpr, sl_h: tl.constexpr, sl_s: tl.constexpr,
    out_m: tl.constexpr, out_h: tl.constexpr,
    ol_m: tl.constexpr, ol_h: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_SPLITS)
    lse = tl.load(
        split_lse + row * sl_m + head * sl_h + splits * sl_s,
        mask=splits < SPLITS,
        other=-float("inf"),
    )
    gmax = tl.max(lse, axis=0)
    valid = gmax != -float("inf")
    safe = tl.where(valid, gmax, 0.0)
    weights = tl.where(lse != -float("inf"), tl.exp2(lse - safe), 0.0)
    denom = tl.sum(weights, axis=0)
    norm = tl.where(denom > 0.0, weights / denom, 0.0)
    natural_lse = tl.where(
        valid, (safe + tl.log2(denom)) * 0.6931471805599453, -float("inf")
    )
    tl.store(output_lse + row * ol_m + head * ol_h, natural_lse)
    for d0 in range(0, D, BLOCK_D):
        dims = d0 + tl.arange(0, BLOCK_D)
        partial = tl.load(
            split_o + row * so_m + head * so_h
            + splits[:, None] * so_s + dims[None, :],
            mask=(lse != -float("inf"))[:, None] & (dims < D)[None, :],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(partial * norm[:, None], axis=0)
        tl.store(
            output + row * out_m + head * out_h + dims,
            merged,
            mask=dims < D,
        )


@triton.jit
def _native_k5_fused_merge_hpp4_kernel(
    split_o,
    split_lse,
    output,
    output_lse,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_s: tl.constexpr,
    sl_m: tl.constexpr,
    sl_h: tl.constexpr,
    sl_s: tl.constexpr,
    out_m: tl.constexpr,
    out_h: tl.constexpr,
    ol_m: tl.constexpr,
    ol_h: tl.constexpr,
    SPLITS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    splits = tl.arange(0, SPLITS)
    lse = tl.load(
        split_lse
        + row * sl_m
        + heads[:, None] * sl_h
        + splits[None, :] * sl_s
    )
    gmax = tl.max(lse, axis=1)
    valid = gmax != -float("inf")
    safe = tl.where(valid, gmax, 0.0)
    weights = tl.where(lse != -float("inf"), tl.exp2(lse - safe[:, None]), 0.0)
    denom = tl.sum(weights, axis=1)
    norm = tl.where(denom[:, None] > 0.0, weights / denom[:, None], 0.0)
    natural_lse = tl.where(
        valid, (safe + tl.log2(denom)) * 0.6931471805599453, -float("inf")
    )
    tl.store(output_lse + row * ol_m + heads * ol_h, natural_lse)
    for d0 in range(0, D, BLOCK_D):
        dims = d0 + tl.arange(0, BLOCK_D)
        partial = tl.load(
            split_o
            + row * so_m
            + heads[:, None, None] * so_h
            + splits[None, :, None] * so_s
            + dims[None, None, :],
            mask=(lse != -float("inf"))[:, :, None],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(partial * norm[:, :, None], axis=1)
        tl.store(
            output
            + row * out_m
            + heads[:, None] * out_h
            + dims[None, :],
            merged,
        )

def native_packed_k5_decode(
    q: torch.Tensor,
    selected_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    k5_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
    *,
    sm_scale: float,
    candidate_envelope: int,
    exact_pool_only: bool = False,
    fuse_kvarn_hadamard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native SM120 sparse MLA over K4/K5 and exact rows.

    ``exact_pool_only`` and ``fuse_kvarn_hadamard`` are Python-static launch
    modes. Fused Hadamard consumes unrotated Q-NoPE and returns unrotated output.
    Exact eager calls validate that every active selected row maps into the exact
    side pool; CUDA graph replay relies on the cache tracker's invariant.
    ``B12X_KVARN_MLA_EXACT_H16=1`` routes only exact-pool launches to
    the H16/block512 BF16-math specialization.

    """
    if q.ndim != 3:
        raise ValueError("native K5 query must be rank 3")
    rows, heads, width = map(int, q.shape)
    if not 1 <= rows <= 16 or heads != 64 or width != 576:
        raise ValueError("native K5 decode requires M in [1,16], H64, D576")
    if (
        q.dtype != torch.bfloat16
        or q.stride(2) != 1
        or q.stride(1) < width
        or q.stride(0) < heads * q.stride(1)
    ):
        raise ValueError("native K5 query must have a non-overlapping unit-inner BF16 layout")
    if selected_indices.dtype != torch.int32 or selected_indices.shape != (rows, 2048):
        raise ValueError("native K5 selected indices must be int32[M,2048]")
    if not selected_indices.is_contiguous():
        raise ValueError("native K5 selected indices must be contiguous")
    if valid_counts.dtype != torch.int32 or valid_counts.shape != (rows,):
        raise ValueError("native K5 valid counts must be int32[M]")
    if k5_cache.dtype != torch.uint8 or k5_cache.ndim != 3:
        raise ValueError("native KVarN cache must be rank-3 uint8")
    tile_bytes = int(k5_cache.shape[2])
    packed_bits_by_tile = {
        int(_K2_TILE_BYTES): 2,
        int(_K4_TILE_BYTES): 4,
        int(_K5_TILE_BYTES): 5,
    }
    packed_bits = packed_bits_by_tile.get(tile_bytes)
    if (
        k5_cache.shape[1] != 1
        or packed_bits is None
        or not k5_cache.is_contiguous()
    ):
        raise ValueError(
            "native KVarN cache must be contiguous "
            "[blocks,1,18560|26752|30848]: "
            f"shape={tuple(k5_cache.shape)} stride={tuple(k5_cache.stride())} "
            f"offset={k5_cache.storage_offset()} "
            f"contiguous={k5_cache.is_contiguous()}"
        )
    if (
        block_to_pool_slot.dtype != torch.int32
        or block_to_pool_slot.shape != (k5_cache.shape[0],)
        or not block_to_pool_slot.is_contiguous()
    ):
        raise ValueError("native K5 block map must contain one contiguous int32 per block")
    if (
        latent_pool.dtype != torch.float8_e4m3fn
        or tuple(latent_pool.shape[1:]) != (64, 512)
        or not latent_pool.is_contiguous()
    ):
        raise ValueError("native K5 exact latent pool must be contiguous E4M3[slots,64,512]")
    if (
        rope_pool.dtype != torch.bfloat16
        or tuple(rope_pool.shape[1:]) != (64, 64)
        or not rope_pool.is_contiguous()
    ):
        raise ValueError("native K5 exact RoPE pool must be contiguous BF16[slots,64,64]")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("native K5 exact pools must have equal capacities")
    if type(exact_pool_only) is not bool:
        raise TypeError("native K5 exact_pool_only must be a Python bool")
    if type(fuse_kvarn_hadamard) is not bool:
        raise TypeError("native K5 fuse_kvarn_hadamard must be a Python bool")
    if not 0 <= int(candidate_envelope) <= 2048:
        raise ValueError("native K5 candidate envelope must be in [0,2048]")
    chunks = (int(candidate_envelope) + 63) // 64
    exact_h16 = bool(exact_pool_only and _exact_h16_enabled())
    chunks_per_split = 1 if rows == 1 else (3 if rows == 4 else 4)
    if rows == 4 and (not exact_pool_only or exact_h16):
        chunks_per_split = _mixed_m4_chunks_per_split()
    elif (
        _native_m5_split_family_enabled()
        and 2 <= rows <= 7
        and (not exact_pool_only or exact_h16)
    ):
        # M5/M6/M7 share the M4 mixed split family (default-off). The native
        # grid is one CTA per row, so per-row chunk-range walk and FP32
        # accumulation order are then identical to the M=4 verify path; only
        # the runtime grid dimension changes. With the knob unset, rows 2..7
        # keep chunks_per_split=4 exactly as before (fail-closed default).
        chunks_per_split = _mixed_m4_chunks_per_split()
    num_splits = (chunks + chunks_per_split - 1) // chunks_per_split
    if num_splits > int(split_output.shape[2]) or num_splits > int(split_lse.shape[2]):
        raise ValueError("native K5 split scratch is smaller than the DCP-local plan")
    tensors = (
        selected_indices, valid_counts, k5_cache, block_to_pool_slot,
        latent_pool, rope_pool, split_output, split_lse, num_chunks_ptr,
        output, output_lse,
    )
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("native K5 tensors must share one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("native K5 decode requires SM120")
    if exact_pool_only and not torch.cuda.is_current_stream_capturing():
        prefix = selected_indices[:, : int(candidate_envelope)]
        active = (
            torch.arange(int(candidate_envelope), device=q.device)[None, :]
            < valid_counts[:, None]
        )
        max_physical = int(k5_cache.shape[0]) * 64
        if max_physical == 0 and bool(active.any().item()):
            raise ValueError("native exact-pool decode has active rows but an empty cache")
        if max_physical:
            physical_ok = (prefix >= 0) & (prefix < max_physical)
            if bool((active & ~physical_ok).any().item()):
                raise ValueError("native exact-pool decode selected an invalid physical slot")
            safe_blocks = prefix.clamp(0, max_physical - 1) // 64
            # Profiling/dummy MTP runs before ownership maps exist and presents
            # an entirely unmapped (-1) block map. The kernel masks those rows.
            # Once any exact mapping exists, retain fail-closed validation for
            # every active selected candidate so real requests cannot mix K5.
            all_unmapped = bool((block_to_pool_slot == -1).all().item())
            if not all_unmapped:
                pool_slots = block_to_pool_slot[safe_blocks]
                exact = (pool_slots >= 0) & (pool_slots < int(latent_pool.shape[0]))
                if bool((active & ~exact).any().item()):
                    raise ValueError("native exact-pool decode selected a non-exact cache row")
    if chunks == 0:
        output[:rows].zero_()
        output_lse[:rows].fill_(-float("inf"))
        return output[:rows], output_lse[:rows]

    if num_splits == 1:
        mid_out = output[:rows, :heads, :512].unsqueeze(2)
        mid_lse = output_lse[:rows, :heads].unsqueeze(2)
    else:
        mid_out = split_output[:rows, :heads, :num_splits, :512]
        mid_lse = split_lse[:rows, :heads, :num_splits]
    # Import registers the CuTe custom op; kept lazy to avoid package cycles.
    from b12x.attention._shared.mla import kernel as _native_kernel  # noqa: F401
    if exact_h16:
        torch.ops.b12x.kvarn_mla_sm120_decode_grid_exact_h16(
            q, k5_cache.view(-1), selected_indices, mid_out, mid_lse,
            valid_counts, block_to_pool_slot, latent_pool, rope_pool,
            float(sm_scale), num_splits, chunks_per_split, exact_pool_only,
            fuse_kvarn_hadamard,
        )
    else:
        torch.ops.b12x.kvarn_mla_sm120_decode_grid(
            q, k5_cache.view(-1), selected_indices, mid_out, mid_lse,
            valid_counts, block_to_pool_slot, latent_pool, rope_pool,
            float(sm_scale), num_splits, chunks_per_split, exact_pool_only,
            fuse_kvarn_hadamard, packed_bits,
        )
    if num_splits == 1:
        return output[:rows, :heads, :512], output_lse[:rows, :heads]
    use_m9_hpp4 = (
        os.getenv(_M9_HPP4_MERGE_ENV, "0") == "1"
        and packed_bits == 4
        and rows == 9
        and heads == 64
        and num_splits == 8
        and chunks_per_split == 4
    )
    if use_m9_hpp4:
        _native_k5_fused_merge_hpp4_kernel[(rows, heads // 4)](
            mid_out, mid_lse, output, output_lse,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
            output.stride(0), output.stride(1),
            output_lse.stride(0), output_lse.stride(1),
            8, 512, _merge_block_d(), 4,
            num_warps=8,
        )
    else:
        # Independent heads avoid the M1/S32 spill seen in earlier grouped
        # prototypes and remain the exact fallback for every other geometry.
        merge_warps = 8 if num_splits > 16 else 4
        block_d = _merge_block_d()
        merge_warps = max(merge_warps, min(8, block_d // 64))
        _native_k5_fused_merge_kernel[(rows, heads)](
            mid_out, mid_lse, output, output_lse,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
            output.stride(0), output.stride(1),
            output_lse.stride(0), output_lse.stride(1),
            num_splits, triton.next_power_of_2(num_splits), 512, block_d,
            num_warps=merge_warps,
        )
    return output[:rows, :heads, :512], output_lse[:rows, :heads]


__all__ = [
    "compact_kvarn_native_rank_nbytes",
    "is_kvarn_mla_supported",
    "stage_compact_kvarn_native_history",
    "materialize_compact_kvarn_native_records",
    "stage_k5_as_fp8_records",
    "stage_bf16_as_exact_pool_fp8_records",
    "stage_bf16_sylvester_as_exact_pool_fp8_records",
    "native_packed_k5_decode",
]
