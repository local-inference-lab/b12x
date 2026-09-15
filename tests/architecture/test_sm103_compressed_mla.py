"""Host contracts for the planned DeepSeek compressed warp-MMA backend."""

from dataclasses import replace

import pytest
import torch

from b12x.attention.compressed_sparse_mla._tuning import (
    TUNING,
    SparseMlaConfig,
    SparseMlaQuery,
)
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps as Caps,
    plan_compressed_sparse_mla_scratch,
)
from b12x.preparation import DeviceIdentity

B300 = DeviceIdentity(
    vendor="nvidia",
    product_name="NVIDIA B300",
    compute_capability=(10, 3),
    sm_count=148,
)


def query(**changes):
    from tests.preparation.test_sm103_contracts import declare
    return replace(declare("compressed:deepseek_v41").query, **changes)



@pytest.mark.parametrize("rows", [1, 3, 19, 32])
@pytest.mark.parametrize("widths", [(0, 1), (1, 0), (65, 65), (128, 256)])
def test_selection_scratch_is_disjoint_at_capacity(rows, widths):
    config = SparseMlaConfig(max_chunks_per_row=4, split_chunk_size=1, single_pass=False, backend="warp")
    caps = Caps(
        device="cpu",
        num_q_heads=12,
        max_q_rows=rows,
        max_width=sum(widths),
        swa_width=widths[0],
        indexed_width=widths[1],
        max_chunks_per_row=4,
        cache_format="deepseek_v41",
    )
    plan = plan_compressed_sparse_mla_scratch(caps, execution_config=config)
    (spec,) = plan.scratch_specs()
    storage = torch.empty(spec.shape, dtype=spec.dtype)
    binding = plan.bind(
        scratch=storage,
        q=torch.empty(rows, 12, 512, dtype=torch.bfloat16),
        swa_indices=torch.empty(rows, widths[0], dtype=torch.int32),
        swa_lengths=torch.empty(rows, dtype=torch.int32),
    )
    scratch = binding.scratch
    views = (
        scratch.tmp_output,
        scratch.tmp_lse,
        scratch.final_lse,
        scratch.kv_chunk_size_ptr,
        scratch.num_chunks_ptr,
        scratch.sm_scale_tensor,
        scratch.mapped_indices,
        scratch.staged_swa_indices,
        scratch.staged_indexed_indices,
        scratch.staged_swa_lengths,
        scratch.staged_indexed_lengths,
    )
    occupied = sorted(
        (v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
        for v in views
        if v.numel()
    )
    assert all(
        left[1] <= right[0] for left, right in zip(occupied, occupied[1:], strict=False)
    )
    assert occupied[-1][1] <= storage.data_ptr() + storage.numel()
    for number, view in enumerate(views):
        view.fill_(number)
    for number, view in enumerate(views):
        assert torch.all(view == number)


@pytest.mark.parametrize("changes", [
    {"q_dtype": "float16"}, {"kv_dtype": "float8_e4m3fn"}, {"v_head_dim": 448},
    {"mode": "invalid"}, {"query_rows": 2**31}, {"num_q_heads": 65536}, {"swa_width": -1},
])
def test_warp_preparation_rejects_invalid_geometry(changes):
    with pytest.raises((ValueError, TypeError)):
        TUNING.configure(query(**changes), device=B300)



@pytest.mark.parametrize(
    "config",
    [
        SparseMlaConfig(max_chunks_per_row=4, split_chunk_size=1, single_pass=False, backend="native"),
        SparseMlaConfig(max_chunks_per_row=257, split_chunk_size=1, single_pass=False, backend="warp"),
        SparseMlaConfig(max_chunks_per_row=4, split_chunk_size=1, single_pass=False, backend="warp", v41_heads_per_block=8),
    ],
)
def test_sm103_compressed_policy_rejects_incompatible_config(config):
    with pytest.raises(ValueError):
        TUNING.validate_config(query(), config, B300)


def test_sm103_compressed_preparation_selects_implemented_splits():
    for mode in ("decode", "extend"):
        q = query(mode=mode)
        selection = TUNING.configure(q, device=B300)
        configs = [config for _, config in TUNING.iterate(selection)]
        assert configs
        for config in configs:
            assert config.backend == "warp" and config.v41_heads_per_block == 16
            TUNING.validate_config(q, config, B300)



@pytest.mark.parametrize("recipe", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("with_sink", [False, True])
def test_compressed_reference_masks_holes_without_truncating(recipe, with_sink):
    import math
    from b12x.attention._shared.mla.compressed_reference import (
        compressed_sparse_mla_reference,
        pack_compressed_sparse_mla_kv_cache_reference,
        pack_deepseek_v41_cache_reference,
    )

    values = torch.tensor([1, 9, 3], dtype=torch.bfloat16)[:, None].repeat(1, 512)
    cache = (
        pack_deepseek_v41_cache_reference(values, page_size=4, cache_kind="swa")
        if recipe == "deepseek_v41"
        else pack_compressed_sparse_mla_kv_cache_reference(
            values[:, :448].contiguous(), values[:, 448:].contiguous(), page_size=4
        )
    )
    q = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
    indices = torch.tensor([[0, -1, 2, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([4, 4], dtype=torch.int32)
    sink = torch.tensor([math.log(2)]) if with_sink else None
    result, lse = compressed_sparse_mla_reference(
        q,
        cache,
        indices,
        lengths,
        swa_page_size=4,
        cache_format=recipe,
        sm_scale=512**-0.5,
        attn_sink=sink,
        return_lse=True,
    )
    torch.testing.assert_close(
        result[0], torch.full_like(result[0], 1 if with_sink else 2), atol=0, rtol=0
    )
    assert torch.count_nonzero(result[1]) == 0
    torch.testing.assert_close(lse[0], torch.tensor([math.log(4 if with_sink else 2)]))
    if with_sink:
        torch.testing.assert_close(lse[1], sink)
    else:
        assert torch.isneginf(lse[1]).all()
