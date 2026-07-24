"""Tests for the ``sparkinfer::fp6_dense_linear`` torch custom op.

Validates that the opaque op matches the eager :func:`dense_fp6_linear` path
bit-for-bit and that ``torch.compile(fullgraph=True)`` traces a model using the
op without graph breaks — the property vLLM's VLLM_COMPILE mode relies on.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _make_weight(n: int = 256, k: int = 256):
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import quantize_dense_weight_to_fp6

    torch.manual_seed(0)
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    return quantize_dense_weight_to_fp6(w_bf16)


def _op_args(w):
    return (
        w.expanded_weight(),
        w.scale_storage,
        w.global_scale,
        w.fmt,
        w.out_features,
        w.in_features,
        w.act_fmt,
    )


@cuda_required
def test_custom_op_matches_eager():
    import sparkinfer.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import dense_fp6_linear

    w = _make_weight()
    for m in (1, 3, 128):
        x = torch.randn(m, w.in_features, dtype=torch.bfloat16, device="cuda")
        ref = dense_fp6_linear(x, w)
        got = torch.ops.sparkinfer.fp6_dense_linear(x, *_op_args(w))
        assert got.shape == (m, w.out_features)
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


@cuda_required
def test_small_m_matches_padded_rows():
    """The small-M decode tile must agree with the (reference-validated) m=128 path.

    Rows are independent in the GEMM, so y(x[:m])[i] must be bit-identical to
    y(x)[i]. The activation amax is pinned to row 0 so every slice quantizes
    with the same global scale; m=1 hits the (16,128) decode tile, m=3 the MTP
    verify shape, m=5 the >4 coarse-tile path with a non-tile-multiple m.
    """
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import dense_fp6_linear

    w = _make_weight()
    x = torch.randn(128, w.in_features, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 8.0  # amax lives in row 0 -> identical a_gs for all slices
    y_full = dense_fp6_linear(x, w)
    for m in (1, 3, 5):
        y_small = dense_fp6_linear(x[:m], w)
        torch.testing.assert_close(y_small, y_full[:m], rtol=0.0, atol=0.0)


@cuda_required
def test_bytes_quantizer_matches_packed_expand():
    """``emit="bytes"`` must equal the packed quantizer's output, just unpacked.

    The bytes mode shares the entire quantization path (UE8M0 scale, cvt) with
    the packed mode and differs only in the store packing, so codes must match
    ``expand_mxfp6_packed_to_bytes(packed)`` bit-for-bit, and scales must be
    identical.
    """
    from sparkinfer._lib.fp6 import expand_mxfp6_packed_to_bytes
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6,
        _quantize_matrix_fp6_bytes,
    )

    torch.manual_seed(0)
    m, k = 128, 256
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    f8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    a_gs = (f8_max * 28.0 / x.abs().amax().to(torch.float32)).reshape(1)
    for fmt in ("e3m2", "e2m3"):
        packed, scale_packed = _quantize_matrix_fp6(x, fmt, a_gs)
        codes, scale_bytes = _quantize_matrix_fp6_bytes(x, fmt, a_gs)
        assert codes.shape == (m, k)
        expected = expand_mxfp6_packed_to_bytes(packed, k)
        torch.testing.assert_close(codes, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scale_bytes, scale_packed, rtol=0.0, atol=0.0)


@cuda_required
def test_custom_op_compile_fullgraph():
    import sparkinfer.quantization.mxfp6.fp6_dense_op  # noqa: F401

    w = _make_weight()
    args = _op_args(w)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparkinfer.fp6_dense_linear(x, *args)

    x = torch.randn(1, w.in_features, dtype=torch.bfloat16, device="cuda")
    eager = fn(x)
    # Warm the CUTE JIT outside of tracing, mirroring vLLM's warmup-then-compile
    # order; fullgraph=True asserts no graph break around the opaque op.
    compiled = torch.compile(fn, fullgraph=True)
    got = compiled(x)
    torch.testing.assert_close(got, eager, rtol=0.0, atol=0.0)


@cuda_required
def test_custom_op_fake_tensor_shape():
    import sparkinfer.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode

    w = _make_weight()
    args = _op_args(w)
    with FakeTensorMode() as mode:
        fx = mode.from_tensor(torch.empty(7, w.in_features, dtype=torch.bfloat16))
        fargs = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        out = torch.ops.sparkinfer.fp6_dense_linear(fx, *fargs)
        assert tuple(out.shape) == (7, w.out_features)
        assert out.dtype == torch.bfloat16
