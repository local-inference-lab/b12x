"""Host admission, storage, and compile-cache contracts for native FP6."""

from inspect import signature

import pytest
import torch

from b12x._lib.architecture import require_kernel_architecture
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm.blockscaled import _fp6
from b12x.quantization.mxfp6 import _rows, allocate_fp6_linear_workspace


def test_native_fp6_admission_and_static_compile_keys():
    for module in (_fp6, _rows):
        require_kernel_architecture(module.__name__, (10, 3))
    for function in (_fp6.compile_kernel, _rows.compile_scales, _rows.compile_quantizer):
        assert not set(signature(function).parameters) & {"m", "rows", "tokens", "batch", "live_rows"}


@pytest.mark.parametrize("capacity,k,fmt,packed", [(0, 128, "e3m2", True),
    (2**31, 128, "e3m2", True), (8, 96, "e3m2", True),
    (8, 128, "e4m3", True), (8, 128, "invalid", False)])
def test_workspace_rejects_bad_geometry_without_cuda(capacity, k, fmt, packed):
    with pytest.raises(ValueError):
        allocate_fp6_linear_workspace(capacity, k, act_fmt=fmt, packed=packed)


@pytest.mark.parametrize("capture", [False, True])
def test_fp6_cache_misses_fail_before_cuda(capture, monkeypatch):
    calls = (
        (_fp6.compile_kernel, (136, 384, 2, "e3m2", "e2m3", False, False, "bfloat16", False, True, 0)),
        (_rows.compile_scales, (384, "e3m2", True, 0, "sm_103a")),
        (_rows.compile_quantizer, (384, "e3m2", True, True, 0, "sm_103a")),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    if not capture:
        freeze_kernel_resolution("FP6 cache-miss host contract")
    try:
        for fn, args in calls:
            fn.cache_clear()
            with pytest.raises(RuntimeError, match="prewarmed" if capture else "frozen"):
                fn(*args)
    finally:
        unfreeze_kernel_resolution()


def test_fp6_smem_precision_is_independent_of_global_packing():
    import cutlass
    from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm
    for fmt in ("e2m3", "e3m2"):
        for expanded in (False, True):
            kernel = BlockscaledGemm(136, 384, 2, recipe="mxfp6", c_dtype=cutlass.BFloat16,
                a_fmt="e4m3", b_fmt=fmt, b_preexpanded=expanded)
            assert kernel.a_smem_dtype.width == kernel.b_smem_dtype.width == 8
            assert kernel.b_dtype.width == 6 and kernel.pack_b_smem == expanded
            assert kernel.b_gmem_dtype.width == (8 if expanded else 6)
            assert kernel.mma_inst_shape_k == kernel.sf_vec_size == 32
