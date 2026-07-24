"""Tests for the small-M BF16->MX-FP6 activation quantizer.

The decode path quantizes only the real activation rows
(:mod:`sparkinfer.quantization.mxfp6.bf16_to_fp6_small_m`) instead of a full 128-row TMA
tile. Codes and swizzled SFA scale bytes for those rows must be BIT-IDENTICAL
to the validated TMA quantizer — same per-block math, only the padding work is
skipped — so the GEMM result is unchanged.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_TILE = 128


def _sf_offsets(m: int, k: int) -> torch.Tensor:
    """Flat swizzled-SFA byte offsets for rows < m (mt==0 since m <= 16)."""
    rows = torch.arange(m)
    kts = torch.arange(k // 128)
    sfbs = torch.arange(4)
    off = (
        kts[:, None, None] * 512
        + (rows[None, :, None] % 32) * 16
        + (rows[None, :, None] // 32) * 4
        + sfbs[None, None, :]
    )
    return off.flatten()


@cuda_required
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2", "e4m3"])
@pytest.mark.parametrize("m", [1, 3, 5, 16])
def test_small_m_matches_tma_quantizer(m, fmt):
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6_bytes,
        _quantize_matrix_fp6_bytes_small_m,
    )

    from sparkinfer._lib.fp6 import mx_gs_numerator

    torch.manual_seed(0)
    k = 512
    device = torch.device("cuda")

    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    # Pin the amax at the LAST element: regression for the in-kernel amax
    # scan covering the whole tensor (a partial scan would change a_gs and
    # every code below would mismatch).
    x[m - 1, k - 1] = 4.0
    amax_raw = torch.linalg.vector_norm(x, ord=float("inf")).reshape(1)
    # Fmt-aware numerator; f64 divide + f32 cast == div.rn.f32 (the host
    # recipe — torch's raw CUDA f32 division is not always correctly rounded).
    a_gs = (
        mx_gs_numerator(fmt) / amax_raw.to(torch.float32).clamp_min(1e-6).double()
    ).float()
    w_gs = torch.tensor([0.5], dtype=torch.float32, device=device)

    x_pad = torch.zeros(_TILE, k, dtype=torch.bfloat16, device=device)
    x_pad[:m].copy_(x)
    codes_ref, scale_ref = _quantize_matrix_fp6_bytes(x_pad, fmt, a_gs)
    # The small-M kernel computes the amax itself (no amax argument).
    codes_sm, scale_sm, alpha_sm = _quantize_matrix_fp6_bytes_small_m(
        x, fmt, w_gs, _TILE
    )

    assert codes_sm.shape == (_TILE, k)
    torch.testing.assert_close(codes_sm[:m], codes_ref[:m], rtol=0.0, atol=0.0)
    off = _sf_offsets(m, k).to(device)
    torch.testing.assert_close(
        scale_sm.view(-1)[off], scale_ref.view(-1)[off], rtol=0.0, atol=0.0
    )
    # In-kernel alpha must be bit-identical to the host recipe (both use
    # correctly-rounded f32 ops).
    alpha_ref = torch.reciprocal(a_gs * w_gs)
    torch.testing.assert_close(alpha_sm, alpha_ref, rtol=0.0, atol=0.0)


@cuda_required
def test_small_m_linear_end_to_end_bit_exact():
    """``dense_fp6_linear`` at m<=16 (small-M quant path) matches the m=128 rows.

    Regression for the dispatch in ``dense_fp6_linear_expanded``: the small-M
    branch must produce GEMM inputs equivalent to the padded TMA branch. Rows
    are independent, so with the amax pinned to row 0 every slice quantizes
    with the same global scale and y(x[:m])[i] must equal y(x)[i] bit-for-bit.
    """
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(0)
    w = quantize_dense_weight_to_fp6(
        torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    )
    x = torch.randn(128, w.in_features, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 8.0  # amax in row 0 -> identical a_gs for every slice
    y_full = dense_fp6_linear(x, w)  # m=128: TMA quantizer branch
    for m in (1, 3, 16):
        y_small = dense_fp6_linear(x[:m], w)  # small-M quantizer branch
        torch.testing.assert_close(y_small, y_full[:m], rtol=0.0, atol=0.0)


@cuda_required
def test_small_m_compile_guards():
    from sparkinfer.quantization.mxfp6.bf16_to_fp6_small_m import compile_bf16_to_fp6_small_m

    with pytest.raises(AssertionError, match="m<=16"):
        compile_bf16_to_fp6_small_m(17, 512)
    with pytest.raises(AssertionError, match="multiple of 128"):
        compile_bf16_to_fp6_small_m(1, 96)


@cuda_required
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2", "e4m3"])
@pytest.mark.parametrize("m", [1, 3, 5, 16])
def test_small_m_per_row_matches_host_chain(m, fmt):
    """The fused per-row kernel must reproduce the host pre-scale chain bit-for-bit.

    Host reference (the serving decode recipe before fusion): per-row bf16
    amax -> f32 clamp/div -> bf16 pre-scale -> quantize with the kernel's
    re-derived (unit) gs. The per_row=True kernel does all of it in one
    launch; codes, scale bytes, alpha AND the per-row output correction must
    match exactly.
    """
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6_bytes_small_m,
    )
    from sparkinfer._lib.fp6 import mx_gs_numerator

    torch.manual_seed(1)
    k = 512
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    x[0, 3] = 6.0  # distinct row amaxes, including one far off the others
    if m > 1:
        x[m - 1].zero_()  # degenerate all-zero row: clamp_min(1e-6) edge
    w_gs = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Host chain reference (identical to fp6_dense_weights' unfused path):
    # f64 divide + f32 cast == div.rn.f32, matching the in-kernel division.
    a_amax_pr = x.abs().amax(dim=1, keepdim=True).float()
    a_gs_pr = (mx_gs_numerator(fmt) / a_amax_pr.clamp_min_(1e-6).double()).float()
    x_pre = (x.float() * a_gs_pr).to(torch.bfloat16)
    codes_ref, scale_ref, alpha_ref = _quantize_matrix_fp6_bytes_small_m(
        x_pre, fmt, w_gs, _TILE
    )
    codes_ref = codes_ref[:m].clone()
    scale_ref = scale_ref.clone()
    alpha_ref = alpha_ref.clone()
    inv_ref = (1.0 / a_gs_pr.double()).float().to(torch.bfloat16)

    codes_pr, scale_pr, alpha_pr, inv_pr = _quantize_matrix_fp6_bytes_small_m(
        x, fmt, w_gs, _TILE, per_row=True
    )

    torch.testing.assert_close(codes_pr[:m], codes_ref, rtol=0.0, atol=0.0)
    off = _sf_offsets(m, k).to(device)
    torch.testing.assert_close(
        scale_pr.view(-1)[off], scale_ref.view(-1)[off], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(alpha_pr, alpha_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inv_pr, inv_ref, rtol=0.0, atol=0.0)


@cuda_required
@pytest.mark.parametrize("m", [1, 5, 16])
def test_small_m_per_row_linear_ab_bit_exact(m, monkeypatch):
    """dense_fp6_linear: fused in-kernel per-row path == host-chain path, bitwise."""
    from sparkinfer.quantization.mxfp6 import fp6_dense_weights as fdw

    torch.manual_seed(2)
    w = fdw.quantize_dense_weight_to_fp6(
        torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    )
    x = torch.randn(m, w.in_features, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(fdw, "_PER_ROW_IN_KERNEL", False)
    y_host = fdw.dense_fp6_linear(x, w).clone()
    monkeypatch.setattr(fdw, "_PER_ROW_IN_KERNEL", True)
    y_fused = fdw.dense_fp6_linear(x, w)
    torch.testing.assert_close(y_fused, y_host, rtol=0.0, atol=0.0)
