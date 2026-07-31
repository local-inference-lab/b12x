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

    # Same guard as test_sf_copy_mode_is_bit_identical: the unroll changes
    # codegen, so if it ever falls out of the compile-cache key both arms
    # collapse onto one kernel and this compares a path against itself.
    resolved: list[object] = []
    unrolled: list[bool] = []
    _resolve = dg._get_compiled_dense_gemm_mxfp6

    def _spy(*spy_args, **spy_kwargs):
        unrolled.append(bool(spy_kwargs["policy"].large_m_unroll))
        compiled = _resolve(*spy_args, **spy_kwargs)
        resolved.append(compiled)
        return compiled

    monkeypatch.setattr(dg, "_get_compiled_dense_gemm_mxfp6", _spy)

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_UNROLL", False)
    y_ref = fdw.dense_fp6_linear_expanded(x, weight, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_UNROLL", True)
    y_unrolled = fdw.dense_fp6_linear_expanded(x, weight, *args)

    assert len(resolved) == 2, f"expected one GEMM per arm, got {len(resolved)}"
    # Separate failure mode from the cache-key check below: if the policy never
    # asked for unroll=4 on these shapes there is nothing to compare, however
    # distinct the two compiled kernels turn out to be.
    assert unrolled == [False, True], (
        f"expected policy large_m_unroll False then True, got {unrolled}; "
        f"m={m} k={k} weight_form={weight_form} did not select the unroll"
    )
    assert resolved[0] is not resolved[1], (
        "unroll=2 and unroll=4 resolved the SAME compiled kernel; the compile "
        "cache key is blind to the unroll flag, so this comparison is vacuous"
    )
    torch.testing.assert_close(y_unrolled, y_ref, rtol=0.0, atol=0.0)
