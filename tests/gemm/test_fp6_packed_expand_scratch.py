"""Tests for the large-M packed-B expansion into the shared scratch buffer.

Phase B packed-B regime fix: at M > 16 the dense linear expands a 3:4-packed
weight into a shared grow-only scratch (one-pass ``ExpandPackedKernel``) and
runs the faster expanded-B GEMM; decode (M <= 16) stays on the packed stream.
The expansion bytes must be BIT-IDENTICAL to the torch reference
(``expand_mxfp6_packed_to_bytes``) and the end-to-end linear bit-identical to
the pure packed path (``SPARKINFER_PACKED_B_EXPAND_LARGE_M=0``).
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture(autouse=True)
def _isolate_expand_registries():
    """Snapshot and restore the process-wide expansion-scratch registries.

    These tests clear buffers that any warm path in the same interpreter may
    already have sized and claimed, so the clears must not outlive the module.
    """
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    saved_scratch = dict(fdw._EXPAND_SCRATCH)
    saved_assigned = dict(fdw._EXPAND_CAPTURE_ASSIGNED)
    saved_claimed = {
        key: list(value) for key, value in fdw._EXPAND_CAPTURE_CLAIMED.items()
    }
    fdw._EXPAND_SCRATCH.clear()
    fdw._EXPAND_CAPTURE_ASSIGNED.clear()
    fdw._EXPAND_CAPTURE_CLAIMED.clear()
    try:
        yield
    finally:
        for registry, snapshot in (
            (fdw._EXPAND_SCRATCH, saved_scratch),
            (fdw._EXPAND_CAPTURE_ASSIGNED, saved_assigned),
            (fdw._EXPAND_CAPTURE_CLAIMED, saved_claimed),
        ):
            registry.clear()
            registry.update(snapshot)


@cuda_required
@pytest.mark.parametrize(
    "n,k",
    [
        (7, 128),      # single unit column, odd row count
        (128, 512),
        (129, 1536),   # odd rows, wide K
        (4096, 12288), # prefill-scale shard (36 MB packed)
    ],
)
def test_expand_kernel_matches_reference(n, k):
    from sparkinfer._lib.fp6 import expand_mxfp6_packed_to_bytes
    from sparkinfer.quantization.mxfp6.fp6_expand_packed import (
        compile_fp6_expand_packed,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    packed = torch.randint(
        0, 256, (n, 3 * k // 4), dtype=torch.uint8, device=device
    )
    ref = expand_mxfp6_packed_to_bytes(packed, k)

    packed_bytes = packed.numel()
    launch = compile_fp6_expand_packed(packed_bytes)
    out = torch.empty(packed_bytes * 4 // 3, dtype=torch.uint8, device=device)
    launch(packed.reshape(-1), out)

    torch.testing.assert_close(
        out.view(n, k), ref, rtol=0.0, atol=0.0
    )


@cuda_required
def test_expand_kernel_compile_guard():
    from sparkinfer.quantization.mxfp6.fp6_expand_packed import (
        compile_fp6_expand_packed,
    )

    with pytest.raises(AssertionError, match="multiple of 48"):
        compile_fp6_expand_packed(50)


@cuda_required
@pytest.mark.parametrize("m", [17, 128, 130])
def test_large_m_scratch_expansion_matches_packed(m, monkeypatch):
    """Large-M linear over a packed weight: scratch path == pure packed path."""
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(1)
    n, k = 6144, 512
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    fp6w = fdw.quantize_dense_weight_to_fp6(w_bf16)
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)

    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", False)
    y_packed = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)
    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", True)
    y_scratch = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)
    y_pre = fdw.dense_fp6_linear_expanded(x, fp6w.expanded_weight(), *args)

    torch.testing.assert_close(y_scratch, y_packed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y_scratch, y_pre, rtol=0.0, atol=0.0)


@cuda_required
def test_scratch_reuse_and_growth(monkeypatch):
    """The scratch is reused across layers and grows monotonically.

    Two packed weights of different sizes run back-to-back at large M: the
    smaller must reuse the buffer the larger allocated (same storage), results
    stay bit-exact for both, and superseded buffers are retired (kept alive),
    never freed — a captured graph may hold raw pointers into them.
    """
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(2)
    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", True)

    k = 512
    m = 128
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)

    def _run(n):
        w = fdw.quantize_dense_weight_to_fp6(
            (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
        )
        args = (w.scale_storage, w.global_scale, w.fmt, n, k)
        y_scratch = fdw.dense_fp6_linear_expanded(x, w.packed, *args)
        y_pre = fdw.dense_fp6_linear_expanded(x, w.expanded_weight(), *args)
        torch.testing.assert_close(y_scratch, y_pre, rtol=0.0, atol=0.0)

    retired_before = len(fdw._EXPAND_SCRATCH_RETIRED)
    _run(1024)  # allocates
    assert len(fdw._EXPAND_SCRATCH) == 1
    small_buf = next(iter(fdw._EXPAND_SCRATCH.values()))
    _run(6144)  # grows: retires the small buffer
    big_buf = next(iter(fdw._EXPAND_SCRATCH.values()))
    assert big_buf.numel() >= 6144 * k
    assert len(fdw._EXPAND_SCRATCH_RETIRED) == retired_before + 1
    assert fdw._EXPAND_SCRATCH_RETIRED[-1] is small_buf
    _run(1024)  # reuses the big buffer, no growth
    assert next(iter(fdw._EXPAND_SCRATCH.values())) is big_buf


@cuda_required
def test_capture_claims_an_eager_buffer(monkeypatch):
    """Capture reuses an eager buffer instead of allocating in the graph pool.

    The scratch key carries the stream, so a capture stream can never match the
    eager entry directly; without the claim it would allocate an unplanned
    buffer into the graph's private pool on every capture.
    """
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", True)

    device = torch.device("cuda")
    eager = fdw._packed_expand_scratch(1 << 20, device)
    assert len(fdw._EXPAND_SCRATCH) == 1

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    claimed = fdw._packed_expand_scratch(1 << 19, device)

    assert claimed is eager
    assert len(fdw._EXPAND_SCRATCH) == 1


@cuda_required
def test_capture_refuses_to_allocate_unplanned_scratch(monkeypatch):
    """No eager buffer big enough: fail loudly rather than allocate in-graph."""
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", True)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="no eager buffer"):
        fdw._packed_expand_scratch(1 << 20, torch.device("cuda"))


@cuda_required
def test_small_m_stays_packed(monkeypatch):
    """Decode-regime calls (m <= 16) must not touch the expansion scratch."""
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(3)
    monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", True)

    n, k = 6144, 512
    fp6w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x = (torch.randn(1, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)
    fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)
    assert not fdw._EXPAND_SCRATCH
