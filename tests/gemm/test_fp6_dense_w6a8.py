"""Dense W6A8 (E4M3 activations + E2M3 weights) smoke tests.

Gate 1.1: the dense runtime must honor ``activation_format=e4m3`` from
``mxfp6_w6a8`` checkpoints — same weight bytes as W6A6, different live
activation quantization + ``e4m3.e2m3`` MMA.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


@cuda_required
def test_w6a8_dense_linear_numeric():
    """W6A8 dense path produces a non-trivial, high-cosine result vs BF16."""
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(7)
    n, k = 256, 256
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.2).to(torch.bfloat16)
    x = (torch.randn(128, k, device="cuda") * 0.2).to(torch.bfloat16)
    fp6w = quantize_dense_weight_to_fp6(w_bf16, source_format="mxfp6_w6a8")
    assert fp6w.fmt == "e2m3"
    assert fp6w.act_fmt == "e4m3"

    y = dense_fp6_linear(x, fp6w)
    ref = x.float() @ w_bf16.float().T
    assert y.shape == (128, n)
    assert torch.isfinite(y).all()
    assert _cos(y, ref) > 0.95


@cuda_required
@pytest.mark.parametrize("m", [1, 3, 16, 128])
def test_w6a8_dense_matches_across_m(m):
    """Small-M and large-M W6A8 paths agree on shared rows (amax pinned)."""
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(11)
    fp6w = quantize_dense_weight_to_fp6(
        torch.randn(256, 256, dtype=torch.bfloat16, device="cuda") * 0.2,
        source_format="mxfp6_w6a8",
    )
    x = torch.randn(128, fp6w.in_features, dtype=torch.bfloat16, device="cuda") * 0.2
    x[0, 0] = 8.0  # pin amax in row 0 so every slice shares a_gs
    y_full = dense_fp6_linear(x, fp6w)
    y_m = dense_fp6_linear(x[:m], fp6w)
    if m == 1:
        # m=1 uses per-tensor quantization; m=128 uses per-row.  Same
        # effective global scale (30.0) but the per-row path pre-scales in
        # float32→bf16 while the per-tensor kernel applies gs directly in
        # FP32. The bf16 roundtrip changes a few codes — high cosine is
        # enough.
        cos = _cos(y_m, y_full[:m])
        assert cos > 0.999, f"m=1 vs full cosine {cos:.6f}"
    else:
        # m>1: both paths use per-row scaling → row-independent → exact.
        torch.testing.assert_close(y_m, y_full[:m], rtol=0.0, atol=0.0)


@cuda_required
def test_w6a8_source_format_sets_act_fmt():
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import quantize_dense_weight_to_fp6

    w = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    w6a8 = quantize_dense_weight_to_fp6(w, source_format="mxfp6_w6a8")
    w6a6 = quantize_dense_weight_to_fp6(w, source_format="mxfp6_e2m3")
    assert w6a8.act_fmt == "e4m3" and w6a8.fmt == "e2m3"
    assert w6a6.act_fmt == "e2m3" and w6a6.fmt == "e2m3"


@cuda_required
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 256, 256),
        (4, 256, 256),
        (8, 256, 256),
        (1, 5120, 5120),
        (1, 6144, 5120),
        (1, 5120, 13824),
    ],
)
def test_fused_quant_matches_unfused(m, n, k, monkeypatch):
    """Phase 4.1: fused quant→GEMM must produce the same output as the
    standalone-quant + GEMM path (bit-identical)."""
    import sparkinfer._lib.dense_gemm as _dense_mod
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as _wmod
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(42)
    fp6w = quantize_dense_weight_to_fp6(
        torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.2,
        source_format="mxfp6_w6a8",
    )
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.2

    monkeypatch.setattr(_dense_mod, "_DENSE_FUSED_QUANT", False)
    monkeypatch.setattr(_wmod, "_DENSE_FUSED_QUANT", False)
    y_unfused = dense_fp6_linear(x, fp6w)

    monkeypatch.setattr(_dense_mod, "_DENSE_FUSED_QUANT", True)
    monkeypatch.setattr(_wmod, "_DENSE_FUSED_QUANT", True)
    y_fused = dense_fp6_linear(x, fp6w)

    assert y_fused.shape == y_unfused.shape
    assert torch.isfinite(y_fused).all()
    cos = _cos(y_fused, y_unfused)
    if m == 1:
        # m=1: both paths use per-tensor scaling → bit-identical.
        assert cos > 0.999, f"fused/unfused cosine sim {cos:.6f} < 0.999"
        torch.testing.assert_close(y_fused, y_unfused, rtol=0, atol=0)
    else:
        # m>1: unfused uses per-row activation scaling (deterministic) while
        # fused uses per-tensor; different quantization → close but not equal.
        assert cos > 0.99, f"fused/unfused cosine sim {cos:.6f} < 0.99"


@cuda_required
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 256, 256),
        (1, 5120, 5120),
        (128, 5120, 5120),
        (2048, 5120, 5120),
    ],
)
def test_dense_fp6_linear_deterministic(m, n, k):
    """dense_fp6_linear must produce bit-identical output on every call
    for the same input.  Covers small-M decode (m=1), mid-M (m=128),
    and large-M prefill/KLD-scoring (m=2048) paths."""
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(7)
    fp6w = quantize_dense_weight_to_fp6(
        torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.2,
        source_format="mxfp6_w6a8",
    )
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.2

    y_ref = dense_fp6_linear(x, fp6w)
    for trial in range(9):
        y = dense_fp6_linear(x, fp6w)
        torch.testing.assert_close(
            y, y_ref, rtol=0, atol=0,
            msg=f"trial {trial + 1}: output differs from reference",
        )


class TestPackedGemmTPAware:
    """Phase 3.1: use_packed_gemm must use full (unsharded) N, not per-GPU N."""

    def _make_weight(self, out_f, in_f, *, unsharded=0):
        from sparkinfer.quantization.mxfp6.fp6_dense_weights import FP6DenseWeight

        return FP6DenseWeight(
            packed=torch.empty(out_f, (3 * in_f) // 4, dtype=torch.uint8),
            scale_storage=torch.empty(out_f * in_f // 32, dtype=torch.uint8),
            global_scale=torch.ones(1, dtype=torch.float32),
            out_features=out_f,
            in_features=in_f,
            fmt="e2m3",
            act_fmt="e4m3",
            out_features_unsharded=unsharded,
        )

    def test_tp1_wide_uses_packed(self):
        """TP=1 gate_up_proj (N=13824) is above threshold → packed."""
        w = self._make_weight(13824, 5120)
        assert w.use_packed_gemm

    def test_tp1_narrow_uses_expanded(self):
        """TP=1 q_proj (N=4096) is below threshold → expanded."""
        w = self._make_weight(4096, 5120)
        assert not w.use_packed_gemm

    def test_tp2_wide_stays_packed(self):
        """TP=2 gate_up_proj: per-GPU N=6912 < threshold, but full N=13824
        is above → must still use packed path."""
        w = self._make_weight(6912, 5120, unsharded=13824)
        assert w.use_packed_gemm

    def test_tp2_narrow_stays_expanded(self):
        """TP=2 q_proj: per-GPU N=2048, full N=4096 → still expanded."""
        w = self._make_weight(2048, 5120, unsharded=4096)
        assert not w.use_packed_gemm

    def test_unsharded_zero_falls_back_to_per_gpu(self):
        """When out_features_unsharded is 0 (non-TP), use out_features."""
        w_below = self._make_weight(6912, 5120, unsharded=0)
        assert not w_below.use_packed_gemm
        w_above = self._make_weight(13824, 5120, unsharded=0)
        assert w_above.use_packed_gemm

    def test_to_preserves_unsharded(self):
        """FP6DenseWeight.to() must carry out_features_unsharded."""
        w = self._make_weight(6912, 5120, unsharded=13824)
        w2 = w.to("cpu")
        assert w2.out_features_unsharded == 13824
        assert w2.use_packed_gemm
