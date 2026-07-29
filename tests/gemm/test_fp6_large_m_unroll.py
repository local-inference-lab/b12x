"""Bit-equality of the MX-FP6 large-M mainloop unroll (unroll=4 vs unroll=2).

Phase B item 3: the MXFP8 ``large_m_unroll`` tactic is now taken for MX-FP6
too (``SPARKINFER_FP6_LARGE_M_UNROLL=0`` restores the historical unroll=2
plan). The unroll is a pure codegen pragma — identical MMA order per
k-tile/k-block — so outputs must be BIT-IDENTICAL across the switch, on both
the pre-expanded and the packed-B (in-smem expand-ahead) mainloops.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@cuda_required
@pytest.mark.parametrize("weight_form", ["preexpanded", "packed"])
@pytest.mark.parametrize("m,k", [(128, 512), (256, 1536)])
def test_fp6_large_m_unroll_bit_exact(weight_form, m, k, monkeypatch):
    import sparkinfer._lib.dense_gemm as dg
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(0)
    n = 6144
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    fp6w = fdw.quantize_dense_weight_to_fp6(w_bf16)
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)

    if weight_form == "packed":
        # Keep the packed stream at large M so the unroll wraps the in-smem
        # expand-ahead pipeline (the structurally riskier codegen).
        monkeypatch.setattr(fdw, "_PACKED_B_EXPAND_LARGE_M", False)
        weight = fp6w.packed
    else:
        weight = fp6w.expanded_weight()

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_UNROLL", False)
    y_ref = fdw.dense_fp6_linear_expanded(x, weight, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_UNROLL", True)
    y_unrolled = fdw.dense_fp6_linear_expanded(x, weight, *args)

    torch.testing.assert_close(y_unrolled, y_ref, rtol=0.0, atol=0.0)
