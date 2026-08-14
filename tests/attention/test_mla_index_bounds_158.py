"""Regression tests for issue #158: sparse/compressed MLA live cache index bounds.

These tests verify that the host-side fail-closed validation rejects high
positive selected indices that exceed the physical cache slot capacity, while
retaining negative sentinels (-1) as the contractual invalid marker.

The in-kernel ``slot_capacity`` bound (computed from the live cache tensor
shape) dominates during CUDA graph replay; these tests cover the host-side
guard that runs outside graph capture.
"""

from __future__ import annotations

import math

import pytest
import torch


def _get_helpers():
    """Lazy import to avoid pulling CUDA bindings on non-CUDA hosts."""
    from b12x.attention._shared.mla.api import (
        _sparse_mla_slot_capacity,
        _validate_sparse_mla_index_bounds,
    )
    return _sparse_mla_slot_capacity, _validate_sparse_mla_index_bounds


# ---------------------------------------------------------------------------
# _sparse_mla_slot_capacity unit tests (no GPU required)
# ---------------------------------------------------------------------------

def test_slot_capacity_rank3():
    """Rank-3 [pages, page_size, record] -> pages * page_size."""
    _cap = _get_helpers()[0]
    cache = torch.zeros((4, 64, 656), dtype=torch.uint8)
    assert _cap(cache, 64) == 256


def test_slot_capacity_rank2_compressed():
    """Rank-2 [pages, page_bytes] -> pages * page_size."""
    _cap = _get_helpers()[0]
    cache = torch.zeros((8, 256 * 584), dtype=torch.uint8)
    assert _cap(cache, 256) == 2048


def test_slot_capacity_single_page():
    _cap = _get_helpers()[0]
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    assert _cap(cache, 64) == 64


def test_slot_capacity_empty():
    """Zero-element cache -> capacity 0."""
    _cap = _get_helpers()[0]
    cache = torch.zeros((0, 64, 656), dtype=torch.uint8)
    assert _cap(cache, 64) == 0


# ---------------------------------------------------------------------------
# _validate_sparse_mla_index_bounds unit tests (no GPU required)
# ---------------------------------------------------------------------------

def _make_indices(rows, width, values, device="cpu"):
    idx = torch.full((rows, width), values, dtype=torch.int32, device=device)
    return idx


def test_validate_rejects_high_positive_index():
    """An index >= capacity is rejected."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, 100)
    lengths = torch.tensor([4], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_accepts_capacity_minus_one():
    """capacity-1 is the highest valid index and must pass."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, 63)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_accepts_negative_sentinel():
    """Negative sentinels (-1) are retained and must not raise."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, -1)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_accepts_mixed_valid_and_sentinel():
    """A mix of valid indices and -1 sentinels must pass."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[0, 1, -1, 63]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_accepts_negative_non_sentinel():
    """Any negative value is safe (the kernel masks idx < 0)."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[-2, -1, 0, 1]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_skipped_during_graph_capture():
    """During CUDA graph capture the host guard is skipped (kernel dominates)."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, 999_999)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=True)


def test_validate_empty_indices():
    """Empty index tensor -> no validation needed."""
    _validate = _get_helpers()[1]
    indices = torch.empty((0, 4), dtype=torch.int32)
    lengths = torch.empty((0,), dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_zero_capacity_all_sentinels():
    """Zero capacity with only -1 sentinels passes (no positive indices)."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, -1)
    lengths = torch.tensor([4], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=0, is_graph_capture=False)


def test_validate_zero_capacity_rejects_positive():
    """Zero capacity with any positive index is rejected."""
    _validate = _get_helpers()[1]
    indices = _make_indices(1, 4, 0)
    lengths = torch.tensor([4], dtype=torch.int32)
    with pytest.raises(ValueError, match="zero slot capacity"):
        _validate(indices, lengths, slot_capacity=0, is_graph_capture=False)


def test_validate_active_token_counts_masks_padding():
    """Indices past the per-token length (padding) are not checked."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[0, 1, 999, 999]], dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_active_token_counts_catches_live_oob():
    """An OOB index within the live prefix is caught even if padding is valid."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[0, 100, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_active_token_counts_zero_length():
    """A row with zero live length skips validation entirely."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[999, 999, 999, 999]], dtype=torch.int32)
    lengths = torch.tensor([0], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_big_pid_boundary():
    """A large valid index at exactly capacity-1 must pass."""
    _validate = _get_helpers()[1]
    cap = 1024 * 64
    indices = torch.tensor([[cap - 1]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=cap, is_graph_capture=False)


def test_validate_big_pid_overflow():
    """An index at exactly capacity must fail."""
    _validate = _get_helpers()[1]
    cap = 1024 * 64
    indices = torch.tensor([[cap]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        _validate(indices, lengths, slot_capacity=cap, is_graph_capture=False)


def test_validate_rejects_capacity_exact():
    """Index == capacity (not capacity-1) is out of bounds."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[64]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_capacity_mismatch_main():
    """Capacity mismatch: indices planned for a larger cache are rejected."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[0, 32, 64, 128]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        _validate(indices, lengths, slot_capacity=64, is_graph_capture=False)


def test_validate_valid_control():
    """Valid control: all indices within bounds passes silently."""
    _validate = _get_helpers()[1]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
    lengths = torch.tensor([8], dtype=torch.int32)
    _validate(indices, lengths, slot_capacity=256, is_graph_capture=False)


# ---------------------------------------------------------------------------
# GPU integration tests (require CUDA + SM120)
# ---------------------------------------------------------------------------

def _has_sm120(device: torch.device) -> bool:
    try:
        from b12x._lib.intrinsics import get_sm_version
        return get_sm_version(device) >= 120
    except Exception:
        return False


def _make_compressed_case(device, *, topk, num_pages=4, seed=42):
    """Build a compressed MLA decode case with valid indices."""
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_mla_kv_cache_reference,
    )

    page_size = 64
    n_tokens = num_pages * page_size
    heads = 8
    head_dim = 512

    gen = torch.Generator(device=device).manual_seed(seed)
    k_nope = torch.randn((n_tokens, 448), generator=gen, dtype=torch.float32, device=device).clamp(-1, 1)
    k_rope = torch.randn((n_tokens, 64), generator=gen, dtype=torch.float32, device=device).clamp(-1, 1)
    cache = pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope.to(torch.bfloat16), page_size=page_size, num_pages=num_pages
    )
    q = torch.randn((1, heads, head_dim), generator=gen, dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, n_tokens, (1, topk), generator=gen, dtype=torch.int32, device=device)
    idx[:, topk // 2:] = -1
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    return q, cache, idx, lengths, page_size, heads, head_dim


def _make_scratch(device, *, topk, heads, head_dim, page_size):
    from b12x.attention.compressed_mla._scratch import (
        B12XCompressedMLAScratchCaps,
        _compressed_mla_scratch_layout,
        _materialize_compressed_mla_scratch,
    )
    caps = B12XCompressedMLAScratchCaps(
        device=device, num_q_heads=heads, max_q_rows=1, max_width=topk,
        head_dim=head_dim, v_head_dim=head_dim,
        max_chunks_per_row=8, page_size=page_size,
    )
    layout = _compressed_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    return _materialize_compressed_mla_scratch(caps, storage, layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_mla_decode_rejects_oob_index():
    """Compressed MLA decode with an out-of-bounds positive index is rejected
    at the API boundary (fail-closed)."""
    from b12x.attention._shared.mla.compressed_api import (
        compressed_mla_decode_forward,
    )

    device = torch.device("cuda")
    if not _has_sm120(device):
        pytest.skip("SM120 sparse MLA kernel path unavailable")

    topk = 64
    q, cache, idx, lengths, page_size, heads, head_dim = _make_compressed_case(device, topk=topk)

    # Overwrite with an out-of-bounds index
    idx.fill_(int(cache.shape[0]) * page_size)

    scratch = _make_scratch(device, topk=topk, heads=heads, head_dim=head_dim, page_size=page_size)
    binding = scratch.bind(q=q, swa_indices=idx, swa_lengths=lengths)

    with pytest.raises(ValueError, match="exceeds cache slot capacity"):
        compressed_mla_decode_forward(
            binding=binding,
            swa_k_cache=cache,
            sm_scale=1.0 / math.sqrt(head_dim),
            swa_page_size=page_size,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_mla_decode_accepts_valid_indices():
    """Compressed MLA decode with valid indices within bounds succeeds."""
    from b12x.attention._shared.mla.compressed_api import (
        compressed_mla_decode_forward,
    )

    device = torch.device("cuda")
    if not _has_sm120(device):
        pytest.skip("SM120 sparse MLA kernel path unavailable")

    topk = 64
    q, cache, idx, lengths, page_size, heads, head_dim = _make_compressed_case(device, topk=topk)

    scratch = _make_scratch(device, topk=topk, heads=heads, head_dim=head_dim, page_size=page_size)
    binding = scratch.bind(q=q, swa_indices=idx, swa_lengths=lengths)

    out = compressed_mla_decode_forward(
        binding=binding,
        swa_k_cache=cache,
        sm_scale=1.0 / math.sqrt(head_dim),
        swa_page_size=page_size,
    )
    assert out.shape == (1, heads, head_dim)
