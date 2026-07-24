"""Tests for the small-N bf16 GEMV (``sparkinfer::bf16_gemv_small_n``).

The decode path routes unquantized small-N bf16 linears (GDN ``in_proj_ba``)
through a one-CTA-per-column CUTE GEMV instead of cuBLAS's 16x16 WMMA pick.
Accumulation is f32 on both sides but in a different reduction order, so the
contract is "matches the f32 reference to bf16 rounding", not bitwise.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _op():
    from sparkinfer.gemm import bf16_gemv

    # The package API is lazy (install_lazy_api); touching the attribute
    # imports _kernel, which registers the torch custom op.
    bf16_gemv.bf16_gemv_small_n  # noqa: B018

    return torch.ops.sparkinfer.bf16_gemv_small_n


def _assert_matches_f32_ref(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    ref = x.float() @ w.float().t()
    assert y.dtype == torch.bfloat16
    assert y.shape == (x.shape[0], w.shape[0])
    # f32 accumulation on both sides; the only expected difference is the
    # final bf16 rounding plus reduction-order noise far below it.
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)


@cuda_required
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize(
    "n,k", [(64, 5120), (96, 5120), (128, 2048), (112, 1024), (1, 5120)]
)
def test_small_n_gemv_matches_reference(m, n, k):
    op = _op()
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f32_ref(y, x, w)


@cuda_required
def test_last_element_contributes():
    """Regression: the strided K loop must cover the entire row."""
    op = _op()
    device = torch.device("cuda")
    m, n, k = 1, 64, 5120
    x = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
    w = torch.zeros(n, k, device=device, dtype=torch.bfloat16)
    x[0, k - 1] = 3.0
    w[n - 1, k - 1] = 2.0
    y = op(x, w)
    assert y[0, n - 1].item() == pytest.approx(6.0)
    assert y[0, : n - 1].abs().max().item() == 0.0


@cuda_required
def test_large_m_falls_back_to_cublas():
    """m > SMALL_M_MAX must take the in-op F.linear fallback (prefill)."""
    from sparkinfer.gemm.bf16_gemv import SMALL_M_MAX

    op = _op()
    torch.manual_seed(1)
    device = torch.device("cuda")
    m, n, k = SMALL_M_MAX + 1, 96, 2048
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(y, ref)


@cuda_required
def test_odd_k_falls_back_to_cublas():
    op = _op()
    torch.manual_seed(2)
    device = torch.device("cuda")
    x = torch.randn(2, 100, device=device, dtype=torch.bfloat16)
    w = torch.randn(64, 100, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(y, ref)


@cuda_required
def test_noncontiguous_x():
    """A strided x view must be handled (contiguous copy inside the op)."""
    op = _op()
    torch.manual_seed(3)
    device = torch.device("cuda")
    big = torch.randn(4, 2 * 2048, device=device, dtype=torch.bfloat16)
    x = big[:, ::2]  # non-contiguous (4, 2048)
    w = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f32_ref(y, x.contiguous(), w)
