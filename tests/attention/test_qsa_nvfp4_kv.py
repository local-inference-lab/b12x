"""NVFP4 main K/V records for QSA: writer, selected attention, planned path.

The record format and writer recipe live in ``b12x.attention.paged._nvfp4_kv``.
Selected attention must read exactly the values an NVFP4 round trip produces,
so its output is compared bitwise with the BF16 path over the decoded cache.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention import qsa
from b12x.attention.paged import _nvfp4_kv as nvfp4
from b12x.attention.qsa._sparse_gqa import launch_sparse_paged_gqa
from b12x.attention.qsa.reference import sparse_paged_gqa_reference

from ..conftest import require_b12x as require_sm120
from .test_qsa_contract import (  # noqa: F401 - autouse resource fixture
    _allocate_binding,
    _caps,
    _dynamic_inputs,
    _retain_qsa_test_resources,
)

HEAD_DIM = 256
RECORD = nvfp4.record_nbytes(HEAD_DIM)
_SELECTION_WIDTH = 2051


def _record_pool(
    pages: int, page_size: int, kv_heads: int, device: torch.device, tail: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K/V record views into a pool that stores each page's K and V rows together,
    with padding after the records, as the QSA caches of an engine may."""
    raw = torch.zeros(
        (pages, 2, page_size, kv_heads * RECORD + tail),
        dtype=torch.uint8,
        device=device,
    )
    key_cache = raw[:, 0, :, : kv_heads * RECORD].unflatten(-1, (kv_heads, RECORD))
    value_cache = raw[:, 1, :, : kv_heads * RECORD].unflatten(-1, (kv_heads, RECORD))
    return raw, key_cache, value_cache


def _layered_rows(tokens: int, kv_heads: int, device: torch.device) -> torch.Tensor:
    """Rows spanning four decades of magnitude, from strided projection output."""
    generator = torch.Generator(device="cpu").manual_seed(0x51A)
    fused = torch.randn((tokens, 3 * kv_heads * HEAD_DIM + 64), generator=generator)
    fused *= torch.logspace(-3, 1, tokens)[:, None]
    return fused.to(device=device, dtype=torch.bfloat16)


def test_record_geometry_and_cache_requirements() -> None:
    assert RECORD == 148
    assert nvfp4.scale_offset(HEAD_DIM) == 128
    assert nvfp4.outer_scale_offset(HEAD_DIM) == 144
    assert nvfp4.kv_storage(qsa.NVFP4_KV_DTYPE, HEAD_DIM) == (torch.uint8, RECORD)
    requirements = qsa.cache_requirements(
        main_page_size=64, kv_heads=2, kv_dtype=qsa.NVFP4_KV_DTYPE
    )
    fp8 = qsa.cache_requirements(
        main_page_size=64, kv_heads=2, kv_dtype=torch.float8_e4m3fn
    )
    assert requirements.main_k_page_shape == (64, 2, RECORD)
    assert requirements.main_kv_page_nbytes == 2 * 64 * 2 * RECORD
    assert requirements.compressed_page_nbytes == fp8.compressed_page_nbytes
    with pytest.raises(TypeError, match="NVFP4"):
        qsa.cache_requirements(main_page_size=64, kv_dtype=torch.float16)


@pytest.mark.parametrize("shape", [(0, 1), (0, 2), (2, 0, 1), (3, 2)])
def test_reference_codecs_accept_empty_and_nested_rows(shape) -> None:
    """A cache shard can hold no pages; empty inputs must round trip."""
    rows = torch.randn(*shape, HEAD_DIM).to(torch.bfloat16)
    records = nvfp4.quantize_nvfp4_kv_torch(rows)
    decoded = nvfp4.dequantize_nvfp4_kv_torch(records, HEAD_DIM)
    assert records.shape == (*shape, RECORD)
    assert decoded.shape == rows.shape
    if rows.numel():
        error = (decoded - rows.float()).norm() / rows.float().norm()
        assert error < 0.12


@pytest.mark.parametrize("kv_heads", [1, 2])
@torch.inference_mode()
def test_writer_follows_two_level_recipe(kv_heads: int) -> None:
    device = require_sm120()
    tokens, page_size, pages = 37, 64, 3
    fused = _layered_rows(tokens, kv_heads, device)
    width = kv_heads * HEAD_DIM
    key = fused[:, :width].view(tokens, kv_heads, HEAD_DIM)
    value = fused[:, width : 2 * width].view(tokens, kv_heads, HEAD_DIM)
    key[3].zero_()
    raw, key_cache, value_cache = _record_pool(pages, page_size, kv_heads, device)
    raw.fill_(0xA5)
    slots = torch.randperm(pages * page_size, device=device)[:tokens].to(torch.int64)
    slots[5] = -1
    slots[6] = pages * page_size
    nvfp4.write_nvfp4_kv(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slots,
    )
    written = (slots >= 0) & (slots < pages * page_size)
    page, offset = slots[written] // page_size, slots[written] % page_size
    for rows, cache in ((key, key_cache), (value, value_cache)):
        records = cache[page, offset]
        expected = nvfp4.quantize_nvfp4_kv_torch(rows[written])
        # The outer scale is amax * f32(1/2688) bit-exactly.
        assert torch.equal(records[..., 144:], expected[..., 144:])
        # Group scales and payload follow the recipe; rcp.approx may move a
        # rare value across a rounding tie, never by more than one code step.
        assert (records[..., 128:144] == expected[..., 128:144]).float().mean() > 0.999
        assert (records[..., :128] == expected[..., :128]).float().mean() > 0.995
        decoded = nvfp4.dequantize_nvfp4_kv_torch(records, HEAD_DIM)
        reference = nvfp4.dequantize_nvfp4_kv_torch(expected, HEAD_DIM)
        source = rows[written].float()
        error = (decoded - source).norm(dim=-1) / source.norm(dim=-1).clamp_min(1e-30)
        reference_error = (reference - source).norm(dim=-1) / source.norm(
            dim=-1
        ).clamp_min(1e-30)
        assert error.max() < 0.12
        assert error.mean() <= reference_error.mean() * 1.01
    # An all-zero row stores a zero outer scale, zero scales and zero payload.
    zero = key_cache[slots[3] // page_size, slots[3] % page_size]
    assert torch.count_nonzero(zero) == 0
    # Skipped slots and the selector tail stay untouched.
    touched = torch.zeros(pages * page_size, dtype=torch.bool, device=device)
    touched[slots[written]] = True
    untouched = raw.view(pages, 2, page_size, -1).permute(0, 2, 1, 3)
    untouched = untouched.reshape(pages * page_size, -1)[~touched]
    assert torch.all(untouched == 0xA5)
    assert torch.all(raw[..., kv_heads * RECORD :] == 0xA5)


@torch.inference_mode()
def test_writer_replays_under_cuda_graph() -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_sm120()
    tokens, kv_heads = 8, 2
    fused = _layered_rows(tokens, kv_heads, device)
    width = kv_heads * HEAD_DIM
    key = fused[:, :width].view(tokens, kv_heads, HEAD_DIM)
    value = fused[:, width : 2 * width].view(tokens, kv_heads, HEAD_DIM)
    _, key_cache, value_cache = _record_pool(2, 64, kv_heads, device)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    kwargs = dict(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slots,
    )
    nvfp4.write_nvfp4_kv(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("NVFP4 K/V writer replay"), torch.cuda.graph(graph):
        nvfp4.write_nvfp4_kv(**kwargs)
    fused.mul_(-0.5)
    slots.add_(64)
    graph.replay()
    torch.cuda.synchronize()
    expected = nvfp4.quantize_nvfp4_kv_torch(value)
    assert torch.equal(value_cache[1, :tokens, :, 144:], expected[..., 144:])


@pytest.mark.parametrize(("kv_heads", "q_heads"), [(1, 6), (2, 12), (2, 24)])
@pytest.mark.parametrize("rows", [1, 3, 64, 65, 200])
@torch.inference_mode()
def test_selected_attention_reads_decoded_values_bitwise(
    kv_heads: int, q_heads: int, rows: int
) -> None:
    device = require_sm120()
    torch.manual_seed(20260926)
    page_size, pages = 64, 40
    tokens = pages * page_size
    _, key_cache, value_cache = _record_pool(pages, page_size, kv_heads, device)
    key = (
        torch.randn(tokens, kv_heads, HEAD_DIM, device=device)
        * torch.logspace(-2, 1, tokens, device=device)[:, None, None]
    ).to(torch.bfloat16)
    value = torch.randn(tokens, kv_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    nvfp4.write_nvfp4_kv(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=torch.arange(tokens, dtype=torch.int64, device=device),
    )
    decoded_key = nvfp4.dequantize_nvfp4_kv_torch(key_cache, HEAD_DIM).to(
        torch.bfloat16
    )
    decoded_value = nvfp4.dequantize_nvfp4_kv_torch(value_cache, HEAD_DIM).to(
        torch.bfloat16
    )
    positions = torch.randint(100, tokens, (rows,), device=device, dtype=torch.int64)
    selected = torch.full(
        (rows, _SELECTION_WIDTH), -1, dtype=torch.int32, device=device
    )
    for row in range(rows):
        count = min(_SELECTION_WIDTH, int(positions[row]) + 1)
        selected[row, :count] = torch.randperm(int(positions[row]) + 1, device=device)[
            :count
        ].to(torch.int32)
    query = torch.randn(rows, q_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    splits = 16 if rows <= 64 else 1
    table = torch.randperm(pages, device=device).to(torch.int32)[None]
    results = []
    for k_cache, v_cache in ((key_cache, value_cache), (decoded_key, decoded_value)):
        output = torch.empty(
            rows, q_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        lse = torch.empty(rows, q_heads, device=device)
        launch_sparse_paged_gqa(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=table,
            request_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            selected_positions=selected,
            query_positions=positions,
            output=output,
            output_lse=lse,
            partial_output=torch.empty(rows, splits, q_heads, HEAD_DIM, device=device),
            partial_lse=torch.empty(rows, splits, q_heads, device=device),
            softmax_scale=HEAD_DIM**-0.5,
            block_n=16,
            splits=splits,
        )
        results.append((output, lse))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_qsa_plan_writes_and_attends_nvfp4_records() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        q_heads=24,
        head_dim=256,
        index_head_dim=128,
        kv_dtype=qsa.NVFP4_KV_DTYPE,
    )
    binding = _allocate_binding(caps)
    assert binding.main_k_cache.dtype == torch.uint8
    assert tuple(binding.main_k_cache.shape[2:]) == (caps.kv_heads, RECORD)
    binding.main_block_table.copy_(
        torch.arange(caps.main_table_width, dtype=torch.int32, device=device)
        .expand(caps.max_batch, -1)
        .contiguous()
    )
    binding.compressed_block_table.copy_(
        torch.arange(caps.compressed_table_width, dtype=torch.int32, device=device)
        .expand(caps.max_batch, -1)
        .contiguous()
    )
    binding.compressed_k_cache.zero_()
    tokens = caps.num_main_cache_pages * caps.main_page_size
    rows = torch.randn(tokens, 2, caps.kv_heads, HEAD_DIM, device=device).to(
        torch.bfloat16
    )
    writer = qsa.bind_kv_writer(
        binding.plan,
        main_k_cache=binding.main_k_cache,
        main_v_cache=binding.main_v_cache,
    )
    writer.write(
        key=rows[:, 0],
        value=rows[:, 1],
        slot_mapping=torch.arange(tokens, dtype=torch.int64, device=device),
    )
    dynamic = _dynamic_inputs(binding, positions=(3, -1), request_ids=(0, -1))
    dynamic["rope_positions"][1].fill_(-1)
    binding.raw_k_ring[0, :3].copy_(
        torch.arange(3 * caps.index_head_dim, device=device)
        .reshape(3, caps.index_head_dim)
        .to(torch.bfloat16)
        / 128
    )
    binding.raw_logical_positions[0, :3].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    binding.raw_rope_positions[0, :3, 0].copy_(
        torch.arange(3, dtype=torch.int64, device=device)
    )
    main_k_before = binding.main_k_cache.clone()
    actual = qsa.run(binding, **dynamic)
    expected = sparse_paged_gqa_reference(
        dynamic["query"],
        nvfp4.dequantize_nvfp4_kv_torch(binding.main_k_cache, HEAD_DIM).to(
            torch.bfloat16
        ),
        nvfp4.dequantize_nvfp4_kv_torch(binding.main_v_cache, HEAD_DIM).to(
            torch.bfloat16
        ),
        binding.main_block_table,
        dynamic["request_ids"],
        binding.selected_positions[:2],
        dynamic["query_positions"],
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=2e-2)
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.equal(binding.main_k_cache, main_k_before)


def test_selected_attention_rejects_misaligned_nvfp4_records() -> None:
    device = require_sm120()
    raw = torch.zeros((2, 16, 1, RECORD + 1), dtype=torch.uint8, device=device)
    misaligned = raw[..., 1:]  # data pointer and strides off the four-byte grid
    rows = 1
    with pytest.raises(ValueError, match="four-byte aligned"):
        launch_sparse_paged_gqa(
            query=torch.zeros(rows, 6, HEAD_DIM, dtype=torch.bfloat16, device=device),
            key_cache=misaligned,
            value_cache=misaligned,
            block_table=torch.zeros(1, 2, dtype=torch.int32, device=device),
            request_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            selected_positions=torch.zeros(
                rows, _SELECTION_WIDTH, dtype=torch.int32, device=device
            ),
            query_positions=torch.zeros(rows, dtype=torch.int64, device=device),
            output=torch.empty(rows, 6, HEAD_DIM, dtype=torch.bfloat16, device=device),
            output_lse=None,
            partial_output=torch.empty(rows, 16, 6, HEAD_DIM, device=device),
            partial_lse=torch.empty(rows, 16, 6, device=device),
            softmax_scale=HEAD_DIM**-0.5,
            block_n=16,
            splits=16,
        )


def test_writer_binding_rejects_misaligned_nvfp4_records() -> None:
    device = require_sm120()
    caps = _caps(
        device,
        q_heads=24,
        head_dim=256,
        index_head_dim=128,
        kv_dtype=qsa.NVFP4_KV_DTYPE,
    )
    binding = _allocate_binding(caps)
    shape = tuple(binding.main_k_cache.shape)
    raw = torch.zeros((*shape[:3], RECORD + 1), dtype=torch.uint8, device=device)
    misaligned = raw[..., 1:]
    with pytest.raises(ValueError, match="four-byte aligned"):
        qsa.bind_kv_writer(
            binding.plan, main_k_cache=misaligned, main_v_cache=misaligned
        )


def test_prepared_sparse_launch_requires_planned_kv_format() -> None:
    device = require_sm120()
    cache = torch.zeros((2, 16, 1, RECORD), dtype=torch.uint8, device=device)
    rows = 1
    with pytest.raises(ValueError, match="planned kv_format"):
        launch_sparse_paged_gqa(
            query=torch.zeros(rows, 6, HEAD_DIM, dtype=torch.bfloat16, device=device),
            key_cache=cache,
            value_cache=cache,
            block_table=torch.zeros(1, 2, dtype=torch.int32, device=device),
            request_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            selected_positions=torch.zeros(
                rows, _SELECTION_WIDTH, dtype=torch.int32, device=device
            ),
            query_positions=torch.zeros(rows, dtype=torch.int64, device=device),
            output=torch.empty(rows, 6, HEAD_DIM, dtype=torch.bfloat16, device=device),
            output_lse=None,
            partial_output=torch.empty(rows, 16, 6, HEAD_DIM, device=device),
            partial_lse=torch.empty(rows, 16, 6, device=device),
            softmax_scale=HEAD_DIM**-0.5,
            block_n=16,
            splits=16,
            _prepared={},
        )


@pytest.mark.parametrize("kv_is_fp8", [False, True])
def test_paged_engine_selected_positions_keeps_its_fp8_flag(kv_is_fp8: bool) -> None:
    """The shared paged engine's entry point still takes kv_is_fp8."""
    from b12x.attention.paged.forward_extend_generic import PagedForwardKernel

    kernel = PagedForwardKernel.selected_positions(
        q_heads=24,
        kv_heads=2,
        kv_is_fp8=kv_is_fp8,
        direct_output=True,
        kv_warps=2,
        page_size=16,
        key_strides=(8192, 512, 256),
        value_strides=(8192, 512, 256),
    )
    assert kernel.kv_format == ("fp8" if kv_is_fp8 else "bf16")
    assert kernel.kv_is_fp8 is kv_is_fp8
    assert kernel.kv_is_nvfp4 is False


def test_bf16_plans_own_no_writer() -> None:
    device = require_sm120()
    binding = _allocate_binding(_caps(device, q_heads=24))
    with pytest.raises(ValueError, match="only NVFP4"):
        qsa.bind_kv_writer(
            binding.plan,
            main_k_cache=binding.main_k_cache,
            main_v_cache=binding.main_v_cache,
        )
