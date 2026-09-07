"""BF16/FP8 dual-cache prefill through the public compressed-MLA serving API."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention import compressed_sparse_mla as mla
from tests._reference.helpers import require_b12x
from tests.attention.test_attention_mla_unified_corpus import (
    _allocator_counters,
    _assert_output,
    _install_scenario,
    _make_inputs,
    _poison_inactive_topk_tails,
    _reference,
    _SM_SCALE,
)


def _relocate_pages(cache, scenarios, page_size):
    page_bytes = cache.stride(0) * cache.element_size()
    first_page = (1 << 31) // page_bytes + 1
    pool = torch.empty(
        (first_page + cache.shape[0], cache.shape[1]),
        dtype=cache.dtype,
        device=cache.device,
    )
    pool[first_page:].copy_(cache)
    # Masked RoPE gathers may read page zero before discarding the candidate.
    pool[0].copy_(cache[0])
    slot_offset = first_page * page_size
    shifted = tuple(
        torch.where(indices >= 0, indices + slot_offset, indices)
        for indices in scenarios
    )
    assert first_page * page_bytes > (1 << 31)
    assert slot_offset + cache.shape[0] * page_size < torch.iinfo(torch.int32).max
    return pool, shifted


@torch.inference_mode()
@pytest.mark.parametrize(
    "high_main,high_extra,extra_page_size,heads",
    [(True, False, 64, 16), (False, True, 2, 32), (True, True, 64, 40)],
)
@pytest.mark.parametrize("main_width", [512, 1024])
def test_dual_cache_high_pages_public_graph(
    high_main: bool, high_extra: bool, extra_page_size: int, heads: int, main_width: int,
) -> None:
    device = require_b12x()
    rows, extra_width = 2, 64
    inputs = _make_inputs(
        rows=rows, heads=heads, main_width=main_width, extra_width=extra_width,
        extra_page_size=extra_page_size, per_token=True, device=device,
    )
    _poison_inactive_topk_tails(inputs.main_index_scenarios, inputs.main_length_scenarios)
    _poison_inactive_topk_tails(inputs.extra_index_scenarios, inputs.extra_length_scenarios)
    # Relocation preserves the packed records and therefore this union oracle.
    expected = tuple(_reference(inputs, scenario) for scenario in range(2))
    if high_main:
        cache, indices = _relocate_pages(inputs.main_cache, inputs.main_index_scenarios, 64)
        inputs = replace(inputs, main_cache=cache, main_index_scenarios=indices)
    if high_extra:
        cache, indices = _relocate_pages(
            inputs.extra_cache, inputs.extra_index_scenarios, extra_page_size,
        )
        inputs = replace(inputs, extra_cache=cache, extra_index_scenarios=indices)
    assert inputs.main_cache.data_ptr() != inputs.extra_cache.data_ptr()
    plan = mla.plan(mla.Caps(
        device=device, num_q_heads=heads, max_q_rows=rows,
        max_width=main_width + extra_width, swa_width=main_width,
        indexed_width=extra_width, swa_page_size=64,
        indexed_page_size=extra_page_size, mode="extend", use_cuda_graph=True,
    ))
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = mla.bind(
        plan, scratch=scratch, q=inputs.q, swa_indices=inputs.main_indices,
        swa_lengths=inputs.main_lengths, indexed_indices=inputs.extra_indices,
        indexed_lengths=inputs.extra_lengths,
    )
    output = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)

    def run():
        return mla.run(
            binding=binding, swa_k_cache=inputs.main_cache,
            indexed_k_cache=inputs.extra_cache, sm_scale=_SM_SCALE,
            swa_page_size=64, indexed_page_size=extra_page_size,
            out=output, return_lse=True,
        )

    _install_scenario(inputs, 0)
    run()
    freeze_kernel_resolution("DSV4 dual-cache high-page prefill")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, lse = run()
        tensors = (inputs.q, inputs.main_cache, inputs.extra_cache, scratch, output, lse)
        pointers = tuple(tensor.data_ptr() for tensor in tensors)
        assert captured.data_ptr() == output.data_ptr()
        for scenario in range(2):
            _install_scenario(inputs, scenario)
            output.fill_(float("nan"))
            lse.fill_(float("nan"))
            before = _allocator_counters(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert _allocator_counters(device) == before
            assert tuple(tensor.data_ptr() for tensor in tensors) == pointers
            _assert_output(output, expected[scenario][0], label="dual-cache high pages")
            assert torch.isfinite(lse).all()
            torch.testing.assert_close(
                lse, expected[scenario][1] / math.log(2.0),
                atol=6.0e-2, rtol=2.0e-2,
            )
    finally:
        unfreeze_kernel_resolution()
