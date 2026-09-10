"""Tests for the small-N bf16 GEMV (``b12x::bf16_gemv_small_n``).

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
    from b12x.gemm import bf16_gemv

    # The package API is lazy (install_lazy_api); touching the attribute
    # imports _kernel, which registers the torch custom op.
    bf16_gemv.bf16_gemv_small_n  # noqa: B018

    return torch.ops.b12x.bf16_gemv_small_n


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
def test_noncontiguous_x():
    """The native scalar path must read a strided column view correctly."""
    op = _op()
    torch.manual_seed(3)
    device = torch.device("cuda")
    big = torch.randn(4, 2 * 2048, device=device, dtype=torch.bfloat16)
    x = big[:, ::2]  # non-contiguous (4, 2048)
    w = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f32_ref(y, x.contiguous(), w)


@cuda_required
@pytest.mark.parametrize(
    "input_dtype,weight_dtype,output_dtype",
    [(torch.bfloat16, torch.bfloat16, torch.float32),
     (torch.bfloat16, torch.float32, torch.float32),
     (torch.float32, torch.float32, torch.bfloat16)],
)
def test_unquantized_bias_and_live_rows_reuse_native_graph(
    input_dtype, weight_dtype, output_dtype,
):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.gemm import bf16_gemv

    device = torch.device("cuda")
    torch.manual_seed(41091)
    capacity, n, k = 17, 97, 131
    # Odd K and column-strided sources exercise the non-vectorized native path.
    source = torch.randn(capacity, k * 2, device=device, dtype=input_dtype)[:, ::2]
    weight = torch.randn(n, k, device=device, dtype=weight_dtype) * 0.125
    bias = torch.linspace(-0.03, 0.04, n, device=device, dtype=torch.float32)
    output = torch.empty(capacity, n + 3, device=device, dtype=output_dtype)

    def launch(rows):
        return bf16_gemv.mm(
            source[:rows], weight, bias=bias, out=output[:rows, :n]
        )

    launch(1)
    torch.cuda.synchronize()
    freeze_kernel_resolution("native unquantized projection live-row reuse")
    try:
        for rows in (0, 1, 7, capacity):
            output.fill_(123)
            actual = launch(rows)
            expected = (
                source[:rows].double() @ weight.double().T + bias.double()
            ).to(output_dtype)
            torch.testing.assert_close(
                actual, expected,
                rtol=1e-5 if output_dtype == torch.float32 else 1e-2,
                atol=2e-5 if output_dtype == torch.float32 else 1e-2,
            )
            torch.testing.assert_close(output[:, n:], torch.full_like(output[:, n:], 123))
            torch.testing.assert_close(output[rows:], torch.full_like(output[rows:], 123))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch(capacity)
        address = captured.data_ptr()
        source.mul_(0.5)
        bias.add_(0.03125)
        graph.replay()
        torch.cuda.synchronize()
        assert captured.data_ptr() == address
        expected = (source.double() @ weight.double().T + bias.double()).to(output_dtype)
        torch.testing.assert_close(
            captured, expected,
            rtol=1e-5 if output_dtype == torch.float32 else 1e-2,
            atol=2e-5 if output_dtype == torch.float32 else 1e-2,
        )
    finally:
        unfreeze_kernel_resolution()
