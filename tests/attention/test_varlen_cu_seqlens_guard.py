"""Regression tests for issue #155: varlen cumulative offset value validation.

Tests cover:
  - Eager rejection of negative/nonzero start, decreasing entries, wrong final
    extent, exceeded per-segment maxima, K/V packed-row mismatch.
  - Device guard neutralization on post-capture mutation (replay-safe).
  - Single-invariant NUM=3 (non-power-of-two) guard correctness.
  - Valid metadata passes both eager and device guards.
  - Failure is detected before unsafe launch/addressing (eager raises before
    the kernel is ever called).
  - Real CUDA graph capture/replay with mutation, sentinel oracle, and
    invalid->valid status reset.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from ..conftest import require_b12x as require_sm120


def _require_contiguous_backend() -> torch.device:
    device = require_sm120()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return device


def _make_varlen_inputs(
    lengths: Tuple[int, ...],
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_q = sum(lengths)
    q = torch.randn(total_q, q_heads, head_dim, dtype=dtype, device=device, generator=generator)
    k = torch.randn(total_q, kv_heads, head_dim, dtype=dtype, device=device, generator=generator)
    v = torch.randn(total_q, kv_heads, head_dim, dtype=dtype, device=device, generator=generator)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return q, k, v, cu_seqlens


# ---------------------------------------------------------------------------
# Eager value validation (CPU-safe, no GPU kernel needed)
# ---------------------------------------------------------------------------


def _eager_validate(
    q_rows: int,
    k_rows: int,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    v_rows: int | None = None,
) -> None:
    from b12x.attention._shared.contiguous.api import _validate_varlen_cu_seqlens_values

    v_shape = (v_rows, 2, 64) if v_rows is not None else None
    _validate_varlen_cu_seqlens_values(
        q_shape=(q_rows, 4, 64),
        k_shape=(k_rows, 2, 64),
        v_shape=v_shape,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
    )


def test_eager_rejects_nonzero_start_q() -> None:
    cu_q = torch.tensor([100, 102], dtype=torch.int32)
    cu_k = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="must be 0"):
        _eager_validate(102, 2, cu_q, cu_k, 10, 10)


def test_eager_rejects_nonzero_start_k() -> None:
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([5, 7], dtype=torch.int32)
    with pytest.raises(ValueError, match="must be 0"):
        _eager_validate(2, 7, cu_q, cu_k, 10, 10)


def test_eager_rejects_negative_start() -> None:
    cu_q = torch.tensor([-1, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="must be 0"):
        _eager_validate(2, 2, cu_q, cu_k, 10, 10)


def test_eager_rejects_decreasing_entries() -> None:
    cu_q = torch.tensor([0, 5, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="nondecreasing"):
        _eager_validate(3, 8, cu_q, cu_k, 10, 10)


def test_eager_rejects_wrong_final_extent_q() -> None:
    cu_q = torch.tensor([0, 5, 10], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_q\\[-1\\] must equal packed q rows 31"):
        _eager_validate(31, 10, cu_q, cu_k, 10, 10)


def test_eager_rejects_wrong_final_extent_k() -> None:
    cu_q = torch.tensor([0, 5, 10], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_k\\[-1\\] must equal packed k rows 42"):
        _eager_validate(10, 42, cu_q, cu_k, 10, 10)


def test_eager_rejects_exceeded_max_seqlen_q() -> None:
    cu_q = torch.tensor([0, 15, 20], dtype=torch.int32)
    cu_k = torch.tensor([0, 10, 20], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds max_seqlen_q=10"):
        _eager_validate(20, 20, cu_q, cu_k, 10, 10)


def test_eager_rejects_exceeded_max_seqlen_k() -> None:
    cu_q = torch.tensor([0, 10, 20], dtype=torch.int32)
    cu_k = torch.tensor([0, 15, 20], dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds max_seqlen_k=10"):
        _eager_validate(20, 20, cu_q, cu_k, 10, 10)


def test_eager_rejects_kv_row_mismatch() -> None:
    cu_q = torch.tensor([0, 5, 10], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32)
    with pytest.raises(ValueError, match="packed k rows.*must equal packed v rows"):
        _eager_validate(10, 10, cu_q, cu_k, 10, 10, v_rows=8)


def test_eager_accepts_valid_metadata() -> None:
    cu_q = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    _eager_validate(15, 15, cu_q, cu_k, 5, 5, v_rows=15)


def test_eager_accepts_single_segment() -> None:
    cu_q = torch.tensor([0, 20], dtype=torch.int32)
    cu_k = torch.tensor([0, 20], dtype=torch.int32)
    _eager_validate(20, 20, cu_q, cu_k, 25, 25, v_rows=20)


def test_eager_rejects_too_short_cu_seqlens() -> None:
    cu_q = torch.tensor([0], dtype=torch.int32)
    cu_k = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(ValueError, match="at least two entries"):
        _eager_validate(0, 0, cu_q, cu_k, 10, 10)


# ---------------------------------------------------------------------------
# Eager validation integrated into plan/bind path (GPU required)
# ---------------------------------------------------------------------------


def test_create_varlen_plan_rejects_nonzero_start() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import create_varlen_attention_plan

    q, k, v, _ = _make_varlen_inputs((5, 5), device=device)
    cu_q = torch.tensor([100, 105], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 10], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="must be 0"):
        create_varlen_attention_plan(q, k, v, cu_q, cu_k)


def test_create_varlen_plan_rejects_decreasing() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import create_varlen_attention_plan

    q, k, v, _ = _make_varlen_inputs((5, 5), device=device)
    cu_q = torch.tensor([0, 5, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="nondecreasing"):
        create_varlen_attention_plan(q, k, v, cu_q, cu_k)


def test_create_varlen_plan_rejects_wrong_final_extent() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import create_varlen_attention_plan

    q, k, v, _ = _make_varlen_inputs((5, 5), device=device)
    cu_q = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="packed q rows 10"):
        create_varlen_attention_plan(q, k, v, cu_q, cu_k)


def test_create_varlen_plan_rejects_exceeded_maxima() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import create_varlen_attention_plan

    q, k, v, _ = _make_varlen_inputs((5, 5), device=device)
    cu_q = torch.tensor([0, 15, 20], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="exceeds max_seqlen_q=5"):
        create_varlen_attention_plan(
            q, k, v, cu_q, cu_k, max_seqlen_q=5, max_seqlen_k=5
        )


# ---------------------------------------------------------------------------
# Device guard: single-invariant NUM=3 (non-power-of-two) correctness
# ---------------------------------------------------------------------------


def test_device_guard_single_invariant_num3_q_final_extent() -> None:
    """NUM=3 (BLOCK=4): mutate only Q final extent by +1 so exactly one lane
    is invalid.  The old reduction bug would accept this (sum==NUM with
    padded lanes).  The fixed reduction must reject it."""
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    # Q has wrong final extent (11 instead of 10), K is valid.
    bad_q = torch.tensor([0, 5, 11], dtype=torch.int32, device=device)
    valid_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    binding.cu_seqlens_q_guard.copy_(bad_q)
    binding.cu_seqlens_k_guard.copy_(valid_k)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=10,
        k_rows=10,
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=2,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "guard must reject single-invariant Q final extent"
    assert torch.all(binding.cu_seqlens_q_guard == 0)
    assert torch.all(binding.cu_seqlens_k_guard == 0)


# ---------------------------------------------------------------------------
# Device guard: post-capture mutation neutralization (GPU required)
# ---------------------------------------------------------------------------


def test_device_guard_neutralizes_post_capture_mutation() -> None:
    """After capture, mutate cu_seqlens to invalid values; device guard must
    neutralize the guard copies so the kernel sees safe empty segments."""
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    cu_seqlens.fill_(0)
    cu_seqlens[0] = 100
    cu_seqlens[1] = 50

    binding.cu_seqlens_q_guard.copy_(cu_seqlens)
    binding.cu_seqlens_k_guard.copy_(cu_seqlens)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=int(q.shape[0]),
        k_rows=int(k.shape[0]),
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=int(cu_seqlens.numel()) - 1,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "device guard should detect post-capture mutation"
    assert torch.all(binding.cu_seqlens_q_guard == 0)
    assert torch.all(binding.cu_seqlens_k_guard == 0)


def test_device_guard_accepts_valid_metadata() -> None:
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    binding.cu_seqlens_q_guard.copy_(cu_seqlens)
    binding.cu_seqlens_k_guard.copy_(cu_seqlens)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=int(q.shape[0]),
        k_rows=int(k.shape[0]),
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=int(cu_seqlens.numel()) - 1,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status == 0, "device guard should accept valid metadata"
    assert torch.equal(binding.cu_seqlens_q_guard, cu_seqlens)
    assert torch.equal(binding.cu_seqlens_k_guard, cu_seqlens)


def test_device_guard_detects_decreasing_entries() -> None:
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    bad = torch.tensor([0, 8, 3], dtype=torch.int32, device=device)
    binding.cu_seqlens_q_guard.copy_(bad)
    binding.cu_seqlens_k_guard.copy_(bad)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=int(q.shape[0]),
        k_rows=int(k.shape[0]),
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=int(cu_seqlens.numel()) - 1,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "device guard should detect decreasing entries"
    assert torch.all(binding.cu_seqlens_q_guard == 0)
    assert torch.all(binding.cu_seqlens_k_guard == 0)


def test_device_guard_detects_wrong_final_extent() -> None:
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    # Nondecreasing and within the per-segment maximum, but final extent 9
    # does not equal the ten packed rows.
    bad = torch.tensor([0, 5, 9], dtype=torch.int32, device=device)
    binding.cu_seqlens_q_guard.copy_(bad)
    binding.cu_seqlens_k_guard.copy_(bad)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=int(q.shape[0]),
        k_rows=int(k.shape[0]),
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=int(cu_seqlens.numel()) - 1,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "device guard should detect wrong final extent"
    assert bool((binding.cu_seqlens_q_guard == 0).all().item())


def test_device_guard_detects_exceeded_maxima() -> None:
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    bad = torch.tensor([0, 15, 20], dtype=torch.int32, device=device)
    binding.cu_seqlens_q_guard.copy_(bad)
    binding.cu_seqlens_k_guard.copy_(bad)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=20,
        k_rows=20,
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=2,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "device guard should detect exceeded maxima"


def test_device_guard_detects_nonzero_start() -> None:
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import _run_varlen_cu_seqlens_guard

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)

    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    bad = torch.tensor([100, 105, 110], dtype=torch.int32, device=device)
    binding.cu_seqlens_q_guard.copy_(bad)
    binding.cu_seqlens_k_guard.copy_(bad)
    _run_varlen_cu_seqlens_guard(
        cu_seqlens_q_guard=binding.cu_seqlens_q_guard,
        cu_seqlens_k_guard=binding.cu_seqlens_k_guard,
        metadata_status=binding.metadata_status,
        q_rows=10,
        k_rows=10,
        max_seqlen_q=int(plan.max_seqlen_q),
        max_seqlen_k=int(plan.max_seqlen_k),
        num_batch=2,
    )
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "device guard should detect nonzero start"
    assert torch.all(binding.cu_seqlens_q_guard == 0)
    assert torch.all(binding.cu_seqlens_k_guard == 0)


# ---------------------------------------------------------------------------
# Valid forward + neutral output on invalid (GPU required)
# ---------------------------------------------------------------------------


def test_valid_varlen_forward_unchanged() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import (
        b12x_varlen_attention_forward,
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    lengths = (5, 17, 9)
    q, k, v, cu_seqlens = _make_varlen_inputs(
        lengths, q_heads=4, kv_heads=2, head_dim=64, device=device, seed=37
    )
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    out, lse = b12x_varlen_attention_forward(binding=binding)
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert lse.shape == (q.shape[1], q.shape[0])
    assert not torch.isnan(out).any()
    assert not torch.isnan(lse).any()


def test_invalid_forward_produces_neutral_output() -> None:
    """When metadata is invalid, the sanitizer must zero output and set LSE
    to -inf, not leave stale/uninitialized data."""
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        b12x_varlen_attention_forward,
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
    )

    # Seed output and LSE with distinctive sentinels.
    sentinel_out = torch.full_like(binding.output, 0.123)
    sentinel_lse = torch.full_like(binding.lse, 0.456)
    binding.output.copy_(sentinel_out)
    binding.lse.copy_(sentinel_lse)

    # Mutate cu_seqlens to invalid (nonzero start).
    cu_seqlens.fill_(0)
    cu_seqlens[0] = 100

    out, lse = b12x_varlen_attention_forward(binding=binding)
    torch.cuda.synchronize()

    status = int(binding.metadata_status.item())
    assert status != 0, "guard should detect invalid metadata"
    assert torch.all(out == 0), "output must be zeroed on invalid metadata"
    assert torch.all(lse == float("-inf")), "LSE must be -inf on invalid metadata"


# ---------------------------------------------------------------------------
# Real CUDA graph capture/replay with mutation and sentinel oracle
# ---------------------------------------------------------------------------


def test_cuda_graph_replay_mutation_neutral_and_reset() -> None:
    """Capture a real CUDA graph around binding.run(), mutate live cu_seqlens
    between replays, verify neutral output on invalid and valid output on
    valid replay."""
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        b12x_varlen_attention_forward,
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )

    # Warm up on a side stream (required before graph capture).
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        b12x_varlen_attention_forward(binding=binding)
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    # Capture the graph on a non-default stream.
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            out, lse = b12x_varlen_attention_forward(binding=binding)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    # Save valid reference output.
    valid_out = out.clone()
    valid_lse = lse.clone()

    # Replay 1: mutate cu_seqlens to invalid (nonzero start).
    cu_seqlens.fill_(0)
    cu_seqlens[0] = 100
    graph.replay()
    torch.cuda.synchronize(device)

    status_invalid = int(binding.metadata_status.item())
    assert status_invalid != 0, "replay should detect invalid mutation"
    assert torch.all(out == 0), "invalid replay output must be zeroed"
    assert torch.all(lse == float("-inf")), "invalid replay LSE must be -inf"

    # Replay 2: restore valid cu_seqlens.
    cu_seqlens.copy_(torch.tensor([0, 5, 10], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize(device)

    status_valid = int(binding.metadata_status.item())
    assert status_valid == 0, "replay should accept restored valid metadata"
    # Output should be fresh valid results, not neutral.
    assert not torch.all(out == 0), "valid replay should produce real output"
    assert not torch.isnan(out).any()
    # The valid replay output should match the reference (same inputs).
    torch.testing.assert_close(out, valid_out, rtol=0, atol=0)
    # LSE should also be restored to valid values, not -inf.
    torch.testing.assert_close(lse, valid_lse, rtol=0, atol=0)



# ---------------------------------------------------------------------------
# Workspace guard buffer allocation
# ---------------------------------------------------------------------------


def test_allocate_varlen_workspace_has_guard_buffers() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import (
        allocate_varlen_attention_workspace_for_plan,
        clear_attention_caches,
        create_varlen_attention_plan,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    workspace = allocate_varlen_attention_workspace_for_plan(plan)
    assert workspace.metadata_status is not None
    assert workspace.metadata_status.shape == (1,)
    assert workspace.metadata_status.dtype == torch.int32
    assert workspace.cu_seqlens_q_guard is not None
    assert workspace.cu_seqlens_q_guard.shape == (3,)
    assert workspace.cu_seqlens_k_guard is not None
    assert workspace.cu_seqlens_k_guard.shape == (3,)


def test_scratch_plan_includes_guard_scratch() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    output_bytes = q.numel() * q.element_size()
    lse_bytes = q.shape[1] * q.shape[0] * 4
    assert spec.shape[0] > output_bytes + lse_bytes, (
        "scratch plan must include guard buffer space"
    )


# ---------------------------------------------------------------------------
# Failure before unsafe launch: eager validation raises before kernel call
# ---------------------------------------------------------------------------


def test_eager_failure_prevents_kernel_launch() -> None:
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    cu_seqlens.fill_(0)
    cu_seqlens[0] = 100

    with pytest.raises(ValueError, match="must be 0"):
        scratch_plan.bind(
            scratch=scratch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
        )


# ---------------------------------------------------------------------------
# Unmocked overlap checker (GPU required)
# ---------------------------------------------------------------------------


def test_overlap_checker_allows_disjoint_arena_views() -> None:
    """Real scratch-plan arena views (output, LSE, guards, status) are
    disjoint and must not be rejected."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous import (
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )
    from b12x.attention._shared.contiguous.api import (
        _reject_write_overlap,
    )

    clear_attention_caches()
    lengths = (5, 5)
    q, k, v, cu_seqlens = _make_varlen_inputs(lengths, device=device)
    max_seqlen = max(lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens
    )
    # Must not raise.
    _reject_write_overlap(
        [("output", binding.output), ("lse", binding.lse),
         ("metadata_status", binding.metadata_status),
         ("cu_seqlens_q_guard", binding.cu_seqlens_q_guard),
         ("cu_seqlens_k_guard", binding.cu_seqlens_k_guard)],
        [("output", binding.output), ("lse", binding.lse),
         ("metadata_status", binding.metadata_status),
         ("cu_seqlens_q_guard", binding.cu_seqlens_q_guard),
         ("cu_seqlens_k_guard", binding.cu_seqlens_k_guard),
         ("q", q), ("k", k), ("v", v),
         ("cu_seqlens_q", cu_seqlens), ("cu_seqlens_k", cu_seqlens)],
    )


def test_overlap_checker_allows_read_read_alias() -> None:
    """cu_seqlens_k aliased to cu_seqlens_q (read-only) must not be rejected."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import _reject_write_overlap

    cu = torch.zeros(3, dtype=torch.int32, device=device)
    out = torch.zeros(10, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(2, dtype=torch.float32, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    guard_q = torch.zeros(3, dtype=torch.int32, device=device)
    guard_k = torch.zeros(3, dtype=torch.int32, device=device)
    # Must not raise despite cu_seqlens_q == cu_seqlens_k (read-read alias).
    _reject_write_overlap(
        [("out", out), ("lse", lse), ("status", status),
         ("guard_q", guard_q), ("guard_k", guard_k)],
        [("out", out), ("lse", lse), ("status", status),
         ("guard_q", guard_q), ("guard_k", guard_k),
         ("cu_q", cu), ("cu_k", cu)],  # same tensor
    )


def test_overlap_checker_rejects_write_read_alias() -> None:
    """Output overlapping Q (write vs read) must be rejected."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import _reject_write_overlap

    buf = torch.zeros(100, dtype=torch.bfloat16, device=device)
    out = buf[:50]
    q = buf[30:80]  # overlaps out
    lse = torch.zeros(10, dtype=torch.float32, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    guard_q = torch.zeros(3, dtype=torch.int32, device=device)
    guard_k = torch.zeros(3, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="overlap"):
        _reject_write_overlap(
            [("out", out), ("lse", lse), ("status", status),
             ("guard_q", guard_q), ("guard_k", guard_k)],
            [("out", out), ("lse", lse), ("status", status),
             ("guard_q", guard_q), ("guard_k", guard_k), ("q", q)],
        )


def test_overlap_checker_rejects_write_write_alias() -> None:
    """Output overlapping LSE (write vs write) must be rejected."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import _reject_write_overlap

    buf = torch.zeros(100, dtype=torch.float32, device=device)
    out = buf[:50].view(torch.bfloat16)
    lse = buf[30:80]  # overlaps out
    status = torch.zeros(1, dtype=torch.int32, device=device)
    guard_q = torch.zeros(3, dtype=torch.int32, device=device)
    guard_k = torch.zeros(3, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="overlap"):
        _reject_write_overlap(
            [("out", out), ("lse", lse), ("status", status),
             ("guard_q", guard_q), ("guard_k", guard_k)],
            [("out", out), ("lse", lse), ("status", status),
             ("guard_q", guard_q), ("guard_k", guard_k)],
        )


def test_overlap_checker_rejects_gapped_view() -> None:
    """A gapped as_strided view that spans beyond its dense prefix must be
    detected by the conservative span calculation."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import _occupied_byte_range


    backing = torch.zeros(200, dtype=torch.bfloat16, device=device)
    # Create a gapped view: 10 elements with stride 4 (spans 40 elements).
    gapped = torch.as_strided(backing[:10], (10,), (4,))
    start, end = _occupied_byte_range(gapped)
    # The span must cover the gapped region, not just 10 elements.
    assert end - start > 10 * 2, "gapped view span must exceed dense prefix"


def test_overlap_checker_empty_tensor_is_legal() -> None:
    """Empty tensors must produce zero-width spans and not be falsely rejected."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import _occupied_byte_range

    empty = torch.empty(0, dtype=torch.bfloat16, device=device)
    start, end = _occupied_byte_range(empty)
    assert start == end, "empty tensor must have zero-width span"


def test_overlap_checker_empty_write_inside_nonempty_read_is_legal() -> None:
    """A zero-width range never intersects a containing nonempty range."""
    device = _require_contiguous_backend()
    from b12x.attention._shared.contiguous.api import (
        _byte_ranges_overlap,
        _reject_write_overlap,
    )

    assert not _byte_ranges_overlap(25, 25, 0, 50)
    assert not _byte_ranges_overlap(0, 50, 25, 25)

    backing = torch.zeros(100, dtype=torch.bfloat16, device=device)
    nonempty_read = backing[:50]
    empty_write = backing[25:25]
    # PyTorch represents an empty CUDA view with a null data pointer, but the
    # real checker must still accept it.
    _reject_write_overlap(
        [("empty_write", empty_write)],
        [("empty_write", empty_write), ("read", nonempty_read)],
    )

# ---------------------------------------------------------------------------
# FP16 unequal Q/K capture readiness (GPU required)
# ---------------------------------------------------------------------------


def test_fp16_unequal_qk_and_v_capture_no_jit() -> None:
    """Unequal Q/K packed extents and unequal QK/V head dimensions bind,
    then capture immediately without compiling a sanitizer specialization."""
    device = _require_contiguous_backend()
    pytest.importorskip("triton")
    from b12x.attention._shared.contiguous import (
        b12x_varlen_attention_forward,
        clear_attention_caches,
        create_varlen_attention_plan,
        plan_varlen_attention_scratch,
    )

    clear_attention_caches()
    # Unequal Q/K: Q has 12 rows, K has 8 rows.
    q_lengths = (5, 7)
    k_lengths = (3, 5)
    q, _, _, cu_q = _make_varlen_inputs(
        q_lengths, q_heads=4, kv_heads=2, head_dim=64,
        dtype=torch.float16, device=device, seed=99,
    )
    _, k, _, cu_k = _make_varlen_inputs(
        k_lengths, q_heads=4, kv_heads=2, head_dim=64,
        dtype=torch.float16, device=device, seed=100,
    )
    # Supported MHA contract: Q/K head dim 64, V/output head dim 32.
    v = torch.randn(
        (sum(k_lengths), 2, 32),
        dtype=torch.float16,
        device=device,
    ).contiguous()
    max_q = max(q_lengths)
    max_k = max(k_lengths)
    plan = create_varlen_attention_plan(
        q, k, v, cu_q, cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
    )
    scratch_plan = plan_varlen_attention_scratch(plan)
    spec = scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    # Bind prewarms guard and the exact unequal-V output sanitizer.
    binding = scratch_plan.bind(
        scratch=scratch, q=q, k=k, v=v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )

    # Capture immediately with no warm forward.
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            out, lse = b12x_varlen_attention_forward(binding=binding)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    # Replay with valid metadata - must produce real output.
    graph.replay()
    torch.cuda.synchronize(device)
    assert not torch.isnan(out).any(), "FP16 unequal Q/K replay must produce valid output"
    assert int(binding.metadata_status.item()) == 0
