"""KVarN packed-latent staging and compact native-reader contracts.

Covers the Triton surface of ``b12x.attention.kvarn_mla``:

* ``stage_k5_as_fp8_records`` rewrites packed K2/K4/K5 G64 tiles (and exact
  pool overrides) into the canonical 656-byte FP8 record format the promoted
  SM120 sparse-MLA runtime consumes;
* ``stage_compact_kvarn_native_history`` packs one rank's live pages plus
  exact overrides into the compact DCP wire, and
  ``materialize_compact_kvarn_native_records`` reproduces the canonical
  records from a gathered wire;
* the wire sizing / block-stride rejection contracts.

The references mirror the kernels' arithmetic op-for-op (fp32 low-bit unpack,
BF16 round, per-128-dim amax FP8 scales), so record bytes must match exactly.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.kvarn_mla.api import (
    _NATIVE_EXACT_PAGE_BYTES,
    compact_kvarn_native_rank_nbytes,
    materialize_compact_kvarn_native_records,
    stage_compact_kvarn_native_history,
    stage_k5_as_fp8_records,
)
from tests._reference.helpers import require_b12x

_GROUP = 64
_LATENT_DIM = 512
_ROPE_DIM = 64
_RECORD_BYTES = 656
_RECORD_SCALE_OFFSET = 512
_RECORD_ROPE_OFFSET = 528
_FP8_MAX = 448.0

_K5 = dict(bits=5, tile=30_848, codestream=20_480, s_col=20_480, zp=21_504, s_row=22_528, rope=22_656)
_K4 = dict(bits=4, tile=26_752, codestream=16_384, s_col=16_384, zp=17_408, s_row=18_432, rope=18_560)


def _pack_lowbit(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Little-endian bitpack along the flattened code axis (test-only)."""
    flat = codes.reshape(-1).to(torch.int64)
    bit_values = ((flat.unsqueeze(1) >> torch.arange(bits)) & 1).reshape(-1)
    positions = (
        torch.arange(flat.numel()).unsqueeze(1) * bits + torch.arange(bits)
    ).reshape(-1)
    out = torch.zeros((flat.numel() * bits + 7) // 8, dtype=torch.int64)
    out.scatter_add_(0, positions // 8, bit_values << (positions % 8))
    return out.to(torch.uint8)
def _unpack_lowbit(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.int64)
    lo_idx = (idx * bits) // 8
    lo = packed[lo_idx].to(torch.int64)
    hi_idx = torch.minimum(lo_idx + 1, torch.full_like(lo_idx, packed.numel() - 1))
    hi = packed[hi_idx].to(torch.int64)
    return (((lo | (hi << 8)) >> (idx * bits % 8)) & ((1 << bits) - 1)).reshape(-1)


def _make_packed_tile(
    geom: dict, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One self-consistent packed tile -> (tile bytes, latent fp32, rope bf16)."""
    bits = geom["bits"]
    codes = torch.randint(0, 1 << bits, (_LATENT_DIM, _GROUP), generator=generator)
    s_col = (0.5 + torch.rand(_LATENT_DIM, generator=generator)).to(torch.float16)
    zp = (torch.rand(_LATENT_DIM, generator=generator) - 0.5).to(torch.float16)
    s_row = (0.5 + torch.rand(_GROUP, generator=generator)).to(torch.float16)
    rope = torch.randn(_GROUP, _ROPE_DIM, generator=generator).to(torch.bfloat16)

    tile = torch.zeros(geom["tile"], dtype=torch.uint8)
    tile[: geom["codestream"]] = _pack_lowbit(codes, bits)
    fp16 = tile.view(torch.float16)
    fp16[geom["s_col"] // 2 : geom["s_col"] // 2 + _LATENT_DIM] = s_col
    fp16[geom["zp"] // 2 : geom["zp"] // 2 + _LATENT_DIM] = zp
    fp16[geom["s_row"] // 2 : geom["s_row"] // 2 + _GROUP] = s_row
    tile[geom["rope"] :].view(torch.bfloat16).copy_(rope.reshape(-1))

    # latent[token, dim] = (codes[dim, token] * s_col[dim] + zp[dim]) * s_row[token]
    latent = (
        codes.to(torch.float32).T * s_col.to(torch.float32).unsqueeze(0)
        + zp.to(torch.float32).unsqueeze(0)
    ) * s_row.to(torch.float32).unsqueeze(1)
    return tile.to(device), latent, rope


def _reference_record(latent: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Mirror the staging math: BF16 round then per-128-dim amax FP8 quant."""
    latent = latent.to(torch.bfloat16).to(torch.float32)
    grouped = latent.reshape(4, 128)
    amax = grouped.abs().amax(dim=-1)
    scales = torch.where(amax > 0.0, amax / _FP8_MAX, torch.ones_like(amax))
    codes = torch.clamp(grouped / scales.unsqueeze(-1), -_FP8_MAX, _FP8_MAX)
    row = torch.zeros(_RECORD_BYTES, dtype=torch.uint8)
    row[:_LATENT_DIM].view(torch.float8_e4m3fn).copy_(
        codes.reshape(_LATENT_DIM).to(torch.float8_e4m3fn)
    )
    row[_RECORD_SCALE_OFFSET : _RECORD_SCALE_OFFSET + 16].view(torch.float32).copy_(
        scales.reshape(-1)
    )
    row[_RECORD_ROPE_OFFSET:].view(torch.bfloat16).copy_(rope.reshape(-1))
    return row


def _reference_packed_latent(
    tile: torch.Tensor, geom: dict, token: int
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = _unpack_lowbit(tile[: geom["codestream"]], _LATENT_DIM * _GROUP, geom["bits"])
    codes = codes.reshape(_LATENT_DIM, _GROUP)
    fp16 = tile.view(torch.float16)
    s_col = fp16[geom["s_col"] // 2 : geom["s_col"] // 2 + _LATENT_DIM].to(torch.float32)
    zp = fp16[geom["zp"] // 2 : geom["zp"] // 2 + _LATENT_DIM].to(torch.float32)
    s_row = fp16[geom["s_row"] // 2 : geom["s_row"] // 2 + _GROUP].to(torch.float32)
    latent = (
        codes.to(torch.float32).T * s_col.unsqueeze(0) + zp.unsqueeze(0)
    ) * s_row.unsqueeze(1)
    rope = tile[geom["rope"] :].view(torch.bfloat16).reshape(_GROUP, _ROPE_DIM)
    return latent[token], rope[token]


def test_compact_rank_nbytes_geometry() -> None:
    # 26,752-byte K4 page + 4-byte trailing index slot; 40,960-byte exact page.
    assert _NATIVE_EXACT_PAGE_BYTES == _GROUP * _LATENT_DIM + _GROUP * _ROPE_DIM * 2
    assert compact_kvarn_native_rank_nbytes(0, 0) == 0
    assert (
        compact_kvarn_native_rank_nbytes(10, 32)
        == 10 * (26_752 + 4) + 32 * _NATIVE_EXACT_PAGE_BYTES
    )
    with pytest.raises(ValueError):
        compact_kvarn_native_rank_nbytes(-1, 4)
    with pytest.raises(ValueError):
        compact_kvarn_native_rank_nbytes(4, -1)


def test_stage_rejects_unknown_block_stride() -> None:
    device = require_b12x()
    cache = torch.zeros(2, 1, 26_752 + 64, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="block stride"):
        stage_k5_as_fp8_records(
            torch.zeros(1, dtype=torch.int32, device=device),
            cache,
            torch.full((2,), -1, dtype=torch.int32, device=device),
            torch.zeros(1, _GROUP, _LATENT_DIM, dtype=torch.bfloat16, device=device),
            torch.zeros(1, _GROUP, _ROPE_DIM, dtype=torch.bfloat16, device=device),
            torch.zeros(1, _RECORD_BYTES, dtype=torch.uint8, device=device),
        )


def _assert_record_matches(got: torch.Tensor, expect: torch.Tensor, where: str) -> None:
    """Staged latents must agree within one FP8 quantization step.

    The kernel's ``amax / 448`` fp32 division can round one ULP away from the
    torch reference; at a rounding tie that legitimately flips one E4M3 code,
    so the compared quantity is the dequantized latent (codes x scales) with a
    one-step bound. RoPE stays bit-exact.
    """
    got_codes = got[:_LATENT_DIM].view(torch.float8_e4m3fn).to(torch.float32)
    exp_codes = expect[:_LATENT_DIM].view(torch.float8_e4m3fn).to(torch.float32)
    got_scales = got[_RECORD_SCALE_OFFSET : _RECORD_SCALE_OFFSET + 16].view(
        torch.float32
    )
    exp_scales = expect[_RECORD_SCALE_OFFSET : _RECORD_SCALE_OFFSET + 16].view(
        torch.float32
    )
    assert torch.allclose(got_scales, exp_scales, rtol=2e-7, atol=0.0), (
        f"scales {where}"
    )
    groups = torch.arange(_LATENT_DIM) // 128
    got_val = got_codes * got_scales[groups]
    exp_val = exp_codes * exp_scales[groups]
    step = exp_scales[groups]  # one E4M3 step at these magnitudes
    assert torch.allclose(got_val, exp_val, rtol=0.13, atol=step.max().item() * 0.02), (
        f"latent {where}"
    )
    assert torch.equal(got[_RECORD_ROPE_OFFSET:], expect[_RECORD_ROPE_OFFSET:]), f"rope {where}"

def test_stage_k5_records_match_reference() -> None:
    device = require_b12x()
    gen = torch.Generator(device="cpu").manual_seed(7)

    num_blocks, pool_slots = 3, 2
    tiles, latents, ropes = [], [], []
    for _ in range(num_blocks):
        tile, latent, rope = _make_packed_tile(_K5, gen, device)
        tiles.append(tile)
        latents.append(latent)
        ropes.append(rope)
    k5_cache = torch.stack(tiles).reshape(num_blocks, 1, _K5["tile"]).contiguous()

    exact_latent = (
        torch.randn(pool_slots, _GROUP, _LATENT_DIM, generator=gen)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    exact_rope = (
        torch.randn(pool_slots, _GROUP, _ROPE_DIM, generator=gen).to(torch.bfloat16).to(device)
    )
    # Block 1 is exact (whole-page override through pool slot 0).
    block_to_pool_slot = torch.tensor([-1, 0, -1], dtype=torch.int32, device=device)

    physical_slots = torch.tensor(
        [5, 1 * 64 + 7, -1, num_blocks * 64, 2 * 64 + 63],
        dtype=torch.int32,
        device=device,
    )
    output = torch.full((5, _RECORD_BYTES), 0xAA, dtype=torch.uint8, device=device)

    stage_k5_as_fp8_records(
        physical_slots, k5_cache, block_to_pool_slot, exact_latent, exact_rope, output
    )
    torch.cuda.synchronize()

    expect_body0 = _reference_record(latents[0][5], ropes[0][5])
    expect_exact1 = _reference_record(
        exact_latent[0, 7].to(torch.float32).cpu(), exact_rope[0, 7].cpu()
    )
    expect_body4 = _reference_record(latents[2][63], ropes[2][63])
    _assert_record_matches(output[0].cpu(), expect_body0, "body block 0")
    _assert_record_matches(output[1].cpu(), expect_exact1, "exact block 1")
    _assert_record_matches(output[4].cpu(), expect_body4, "body block 2")
    # Invalid rows (negative slot, block past the cache) stay untouched.
    assert (output[2].cpu() == 0xAA).all()
    assert (output[3].cpu() == 0xAA).all()


def test_compact_history_materialize_round_trip() -> None:
    device = require_b12x()
    gen = torch.Generator(device="cpu").manual_seed(11)

    # Two short requests (5 and 3 pages of 64 tokens). With page_len <= 16
    # every page sits in the sink+tail window, so any page may be exact; the
    # exact id is req * 16 + local_page (sink keeps local, tail maps after it).
    num_blocks, pool_slots = 8, 3
    tiles = [_make_packed_tile(_K4, gen, device)[0] for _ in range(num_blocks)]
    k4_cache = torch.stack(tiles).reshape(num_blocks, 1, _K4["tile"]).contiguous()
    latent_pool = (
        torch.randn(pool_slots, _GROUP, _LATENT_DIM, generator=gen)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    rope_pool = (
        torch.randn(pool_slots, _GROUP, _ROPE_DIM, generator=gen).to(torch.bfloat16).to(device)
    )
    # Blocks 1, 4 (req 0) and 7 (req 1) carry exact overrides.
    block_to_pool_slot = torch.tensor([-1, 0, -1, -1, 1, -1, -1, 2], dtype=torch.int32, device=device)

    num_reqs = 2
    page_starts = torch.tensor([0, 5], dtype=torch.int32, device=device)
    page_lens = torch.tensor([5, 3], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[0, 1, 2, 3, 4, -1, -1, -1], [5, 6, 7, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )

    padded_pages = 10
    padded_exact_pages = num_reqs * 16
    wire = torch.zeros(
        compact_kvarn_native_rank_nbytes(padded_pages, padded_exact_pages),
        dtype=torch.uint8,
        device=device,
    )

    stage_compact_kvarn_native_history(
        block_table,
        page_starts,
        page_lens,
        k4_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        wire,
        padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
    )
    torch.cuda.synchronize()

    # Packed region: live pages carry their cache tile verbatim.
    for page in range(num_blocks):
        assert torch.equal(
            wire[page * _K4["tile"] : (page + 1) * _K4["tile"]],
            k4_cache[page].reshape(-1),
        )

    # Index region: exact pages resolve to req*16 + local_page.
    index = wire[
        padded_pages * _K4["tile"] : padded_pages * _K4["tile"] + 4 * padded_pages
    ].view(torch.int32).cpu()
    assert set((index >= 0).nonzero().flatten().tolist()) == {1, 4, 7}
    assert index[1].item() == 1
    assert index[4].item() == 4
    assert index[7].item() == 1 * 16 + 2

    # Exact region carries the pool bytes at their per-request slots.
    latent_base = padded_pages * (_K4["tile"] + 4)
    rope_base = latent_base + padded_exact_pages * (_GROUP * _LATENT_DIM)
    for exact_id, pool in ((1, 0), (4, 1), (18, 2)):
        assert torch.equal(
            wire[latent_base + exact_id * _GROUP * _LATENT_DIM :][
                : _GROUP * _LATENT_DIM
            ],
            latent_pool[pool].view(torch.uint8).reshape(-1),
        )
        assert torch.equal(
            wire[rope_base + exact_id * _GROUP * _ROPE_DIM * 2 :][
                : _GROUP * _ROPE_DIM * 2
            ],
            rope_pool[pool].view(torch.uint8).reshape(-1),
        )

    # Materialize a single-rank (D=1) gathered wire back into canonical records.
    padded_tokens = 576  # 512 live tokens + sentinel tail
    rank_page_starts = page_starts.reshape(1, num_reqs).contiguous()
    rank_page_lens = page_lens.reshape(1, num_reqs).contiguous()
    token_starts = torch.tensor([[0, 320]], dtype=torch.int32, device=device)
    token_lens = torch.tensor([[320, 192]], dtype=torch.int32, device=device)
    output = torch.full(
        (padded_tokens, _RECORD_BYTES), 0x5A, dtype=torch.uint8, device=device
    )
    materialize_compact_kvarn_native_records(
        wire,
        rank_page_starts,
        rank_page_lens,
        token_starts,
        token_lens,
        output,
        padded_tokens=padded_tokens,
        padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
    )
    torch.cuda.synchronize()

    wire_cpu = wire.cpu()

    def check_token(global_token: int) -> None:
        req = 0 if global_token < 320 else 1
        local = global_token - (0 if req == 0 else 320)
        page = page_starts[req].item() + local // _GROUP
        token = local % _GROUP
        exact_id = index[page].item()
        if exact_id >= 0:
            latent = (
                wire_cpu[latent_base + exact_id * _GROUP * _LATENT_DIM :]
                .view(torch.float8_e4m3fn)[token * _LATENT_DIM : (token + 1) * _LATENT_DIM]
                .to(torch.float32)
            )
            rope = (
                wire_cpu[rope_base + exact_id * _GROUP * _ROPE_DIM * 2 :]
                .view(torch.bfloat16)[token * _ROPE_DIM : (token + 1) * _ROPE_DIM]
            )
        else:
            tile = wire_cpu[page * _K4["tile"] : (page + 1) * _K4["tile"]]
            latent, rope = _reference_packed_latent(tile, _K4, token)
        expect = _reference_record(latent, rope)
        _assert_record_matches(
            output[global_token].cpu(), expect, f"token {global_token}"
        )

    # Body pages, exact pages, and both request boundaries.
    for t in (0, 63, 64, 100, 319, 320, 384, 511):
        check_token(t)
    # Padded tail rows stay untouched.
    assert (output[512:].cpu() == 0x5A).all()


def test_native_packed_decode_matches_reference() -> None:
    """Decode straight from a packed K5 tile via the CuTeDSL SM120 grid.

    This is the scale_format=KVARN_K5 path: it traces the KVarN QK/PV math
    in ``_shared/mla`` (s1/s6 KVarN helpers), so it fails at trace time if
    that dispatch surface is broken. Requires the CUTLASS DSL.
    """
    device = require_b12x()
    pytest.importorskip("cutlass")
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("native KVarN decode grid requires SM120")
    import math

    from b12x.attention.kvarn_mla import native_packed_k5_decode

    tile, latent, rope = _make_packed_tile(_K5, torch.Generator().manual_seed(5), device)
    k5_cache = tile.reshape(1, 1, _K5["tile"])
    q = torch.randn(1, 64, 576, dtype=torch.bfloat16, device=device)
    selected = torch.full((1, 2048), -1, dtype=torch.int32, device=device)
    selected[0, :64] = torch.arange(64, dtype=torch.int32, device=device)
    valid_counts = torch.tensor([64], dtype=torch.int32, device=device)
    block_to_pool_slot = torch.tensor([-1], dtype=torch.int32, device=device)
    latent_pool = torch.zeros(1, _GROUP, _LATENT_DIM, dtype=torch.float8_e4m3fn, device=device)
    rope_pool = torch.zeros(1, _GROUP, _ROPE_DIM, dtype=torch.bfloat16, device=device)
    output = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device=device)
    output_lse = torch.zeros(1, 64, dtype=torch.float32, device=device)
    split_output = torch.zeros(1, 64, 1, 512, dtype=torch.bfloat16, device=device)
    split_lse = torch.zeros(1, 64, 1, dtype=torch.float32, device=device)
    num_chunks_ptr = torch.zeros(1, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(576)

    out, lse = native_packed_k5_decode(
        q, selected, valid_counts, k5_cache, block_to_pool_slot,
        latent_pool, rope_pool, split_output, split_lse, num_chunks_ptr,
        output, output_lse, sm_scale=sm_scale, candidate_envelope=64,
    )
    torch.cuda.synchronize()

    qn = q[0, :, :512].float()
    qr = q[0, :, 512:].float()
    k_lat = latent.to(device).float()
    k_rop = rope.to(device).float()
    scores = (qn @ k_lat.T + qr @ k_rop.T) * sm_scale
    ref = torch.softmax(scores, dim=-1) @ k_lat
    ref_lse = torch.logsumexp(scores, dim=-1)
    cos = torch.nn.functional.cosine_similarity(
        out[0].float().flatten(), ref.flatten(), dim=0
    )
    assert cos.item() >= 0.9995, f"decode cosine {cos.item()}"
    assert (lse[0] - ref_lse).abs().max().item() < 0.5, (
        f"lse drift {(lse[0] - ref_lse).abs().max().item()}"
    )


def test_scaled_offsets_survive_int32_boundary() -> None:
    """Page/block/rank x byte-stride products must address past 2**31 bytes.

    Live pages and cache blocks at the tail of pool-sized arenas, per the
    64-bit addressing contract: small sequential test ids can never catch a
    32-bit offset overflow.
    """
    device = require_b12x()
    gen = torch.Generator(device="cpu").manual_seed(13)

    # --- packed K5 staging past INT32_MAX / 30848 (~69,586 blocks) ---
    tail_block = 70_000 - 1
    tile, latent, rope = _make_packed_tile(_K5, gen, device)
    arena = torch.zeros(tail_block + 1, 1, _K5["tile"], dtype=torch.uint8, device=device)
    arena[tail_block] = tile
    physical_slots = torch.tensor(
        [tail_block * 64 + 5, tail_block * 64 + 63], dtype=torch.int32, device=device
    )
    block_to_pool_slot = torch.full((tail_block + 1,), -1, dtype=torch.int32, device=device)
    output = torch.zeros(2, _RECORD_BYTES, dtype=torch.uint8, device=device)
    stage_k5_as_fp8_records(
        physical_slots,
        arena,
        block_to_pool_slot,
        torch.zeros(1, _GROUP, _LATENT_DIM, dtype=torch.bfloat16, device=device),
        torch.zeros(1, _GROUP, _ROPE_DIM, dtype=torch.bfloat16, device=device),
        output,
    )
    torch.cuda.synchronize()
    _assert_record_matches(
        output[0].cpu(), _reference_record(latent[5], rope[5]), "tail block token 5"
    )
    _assert_record_matches(
        output[1].cpu(), _reference_record(latent[63], rope[63]), "tail block token 63"
    )
    del arena

    # --- compact wire staging + materialization past INT32_MAX / 26752 pages ---
    padded_pages = 100_000  # 100,000 x 26,752 > 2**31 bytes
    num_reqs = 1
    k4_tiles = [_make_packed_tile(_K4, gen, device)[0] for _ in range(4)]
    k4_cache = torch.stack(k4_tiles).reshape(4, 1, _K4["tile"]).contiguous()
    latent_pool = (
        torch.randn(1, _GROUP, _LATENT_DIM, generator=gen).to(torch.float8_e4m3fn).to(device)
    )
    rope_pool = (
        torch.randn(1, _GROUP, _ROPE_DIM, generator=gen).to(torch.bfloat16).to(device)
    )
    b2p = torch.tensor([0, -1, -1, -1], dtype=torch.int32, device=device)
    page_starts = torch.tensor([padded_pages - 4], dtype=torch.int32, device=device)
    page_lens = torch.tensor([4], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    padded_exact_pages = num_reqs * 16
    wire = torch.zeros(
        compact_kvarn_native_rank_nbytes(padded_pages, padded_exact_pages),
        dtype=torch.uint8,
        device=device,
    )
    stage_compact_kvarn_native_history(
        block_table, page_starts, page_lens, k4_cache, b2p,
        latent_pool, rope_pool, wire,
        padded_pages=padded_pages, padded_exact_pages=padded_exact_pages,
    )
    torch.cuda.synchronize()
    for local in range(4):
        page = padded_pages - 4 + local
        assert torch.equal(
            wire[page * _K4["tile"] : (page + 1) * _K4["tile"]],
            k4_cache[local].reshape(-1),
        ), f"tail page {page} lost its tile"
    index = wire[
        padded_pages * _K4["tile"] : padded_pages * _K4["tile"] + 4 * padded_pages
    ].view(torch.int32).cpu()
    assert index[padded_pages - 4].item() == 0  # sink page 0, exact
    assert (index[padded_pages - 3 :] == -1).all()

    # --- two-rank gather: rank 1's wire base sits past 2**31 bytes ---
    rank_wire_bytes = wire.numel()
    gathered = torch.zeros(2 * rank_wire_bytes, dtype=torch.uint8, device=device)
    gathered[rank_wire_bytes:] = wire  # rank 1 carries the staged wire
    rank_page_starts = torch.tensor(
        [[0], [padded_pages - 4]], dtype=torch.int32, device=device
    )
    rank_page_lens = torch.tensor([[0], [4]], dtype=torch.int32, device=device)
    token_starts = torch.tensor([[0], [0]], dtype=torch.int32, device=device)
    token_lens = torch.tensor([[0], [256]], dtype=torch.int32, device=device)
    padded_tokens = 256
    output = torch.full(
        (2 * padded_tokens, _RECORD_BYTES), 0x77, dtype=torch.uint8, device=device
    )
    materialize_compact_kvarn_native_records(
        gathered, rank_page_starts, rank_page_lens, token_starts, token_lens,
        output, padded_tokens=padded_tokens, padded_pages=padded_pages,
        padded_exact_pages=padded_exact_pages,
    )
    torch.cuda.synchronize()

    rank1_wire = wire.cpu()
    index_rank1 = index
    for t in (0, 5, 63, 128, 255):
        page = padded_pages - 4 + t // 64
        token = t % 64
        exact_id = index_rank1[page].item()
        if exact_id >= 0:
            latent_ref = (
                rank1_wire[
                    padded_pages * (_K4["tile"] + 4) + exact_id * _GROUP * _LATENT_DIM :
                ]
                .view(torch.float8_e4m3fn)[token * _LATENT_DIM : (token + 1) * _LATENT_DIM]
                .to(torch.float32)
            )
            rope_base = padded_pages * (_K4["tile"] + 4) + padded_exact_pages * _GROUP * _LATENT_DIM
            rope_ref = (
                rank1_wire[rope_base + exact_id * _GROUP * _ROPE_DIM * 2 :]
                .view(torch.bfloat16)[token * _ROPE_DIM : (token + 1) * _ROPE_DIM]
            )
        else:
            tile_bytes = rank1_wire[page * _K4["tile"] : (page + 1) * _K4["tile"]]
            latent_ref, rope_ref = _reference_packed_latent(tile_bytes, _K4, token)
        row = padded_tokens + t  # rank 1 rows live past rank 0's rows
        _assert_record_matches(
            output[row].cpu(),
            _reference_record(latent_ref, rope_ref),
            f"rank1 tail token {t}",
        )


def test_exact_pages_outside_sink_tail_window_stay_packed() -> None:
    """Pool-marked body pages (outside sink/tail) must not write exact wire.

    With page_len=20 the mappable window is pages {0,1} (sink) and {6..19}
    (tail). Blocks 2 and 5 carry exact pool slots but sit in the body: the
    staging must keep them packed instead of computing a negative or aliased
    exact id (which ``tl.device_assert`` only catches in debug builds).
    """
    device = require_b12x()
    gen = torch.Generator(device="cpu").manual_seed(17)
    num_blocks = 20
    tiles = [_make_packed_tile(_K4, gen, device)[0] for _ in range(num_blocks)]
    k4_cache = torch.stack(tiles).reshape(num_blocks, 1, _K4["tile"]).contiguous()
    latent_pool = (
        torch.randn(2, _GROUP, _LATENT_DIM, generator=gen).to(torch.float8_e4m3fn).to(device)
    )
    rope_pool = (
        torch.randn(2, _GROUP, _ROPE_DIM, generator=gen).to(torch.bfloat16).to(device)
    )
    b2p = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
    b2p[0] = 0  # sink page 0: valid exact
    b2p[2] = 1  # body page: outside the window
    b2p[5] = 1  # body page: outside the window
    page_starts = torch.tensor([0], dtype=torch.int32, device=device)
    page_lens = torch.tensor([20], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(1, -1)
    padded_pages = 20
    padded_exact_pages = 16
    wire = torch.zeros(
        compact_kvarn_native_rank_nbytes(padded_pages, padded_exact_pages),
        dtype=torch.uint8,
        device=device,
    )
    stage_compact_kvarn_native_history(
        block_table, page_starts, page_lens, k4_cache, b2p,
        latent_pool, rope_pool, wire,
        padded_pages=padded_pages, padded_exact_pages=padded_exact_pages,
    )
    torch.cuda.synchronize()
    index = wire[
        padded_pages * _K4["tile"] : padded_pages * _K4["tile"] + 4 * padded_pages
    ].view(torch.int32).cpu()
    assert index[0].item() == 0  # sink page staged exact
    assert index[2].item() == -1  # body page stays packed
    assert index[5].item() == -1  # body page stays packed
    # Body pages 2 and 5 still carry their packed tiles verbatim.
    for page in (2, 5):
        assert torch.equal(
            wire[page * _K4["tile"] : (page + 1) * _K4["tile"]],
            k4_cache[page].reshape(-1),
        )
    # Only exact id 0 was written into the exact region.
    latent_base = padded_pages * (_K4["tile"] + 4)
    exact_region = wire[
        latent_base : latent_base + padded_exact_pages * _GROUP * _LATENT_DIM
    ]
    nonzero_pages = (
        exact_region.view(-1, _GROUP * _LATENT_DIM).any(dim=1).nonzero().flatten()
    )
    assert nonzero_pages.tolist() == [0]
