"""Host admission, geometry, and cache contracts for the FP8 warp entry."""

import cutlass
import pytest

from b12x._lib.architecture import UnsupportedArchitectureError, require_kernel_architecture
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm.blockscaled._fp8_cute import DenseFp8Launch, compile_kernel


def test_fp8_admission_does_not_enable_sm12x_blockscaled_entry():
    from b12x.gemm.tensor_fp8_linear import META
    assert "sm103a" in META.archs
    require_kernel_architecture("b12x.gemm.blockscaled._fp8_cute", (10, 3))
    with pytest.raises(UnsupportedArchitectureError):
        require_kernel_architecture("b12x._lib.dense_gemm", (10, 3))


@pytest.mark.parametrize("n,k,groups,block", [(132, 128, 2, False), (128, 96, 1, False),
    (132, 128, 1, True), (128, 128, 2, True), (0, 128, 1, False)])
def test_fp8_rejects_invalid_static_contract(n, k, groups, block):
    with pytest.raises(ValueError):
        DenseFp8Launch(n, k, groups, cutlass.BFloat16, block, False, 148)


def test_fp8_compile_miss_fails_under_frozen_resolution():
    compile_kernel.cache_clear()
    with kernel_resolution_guard("FP8 host cache miss"):
        with pytest.raises(RuntimeError, match="frozen"):
            compile_kernel(128, 128, 1, "bfloat16", False, True, 0, 148, "sm_103a")


def test_fp8_capture_requires_prewarm(monkeypatch):
    import torch
    compile_kernel.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarmed"):
        compile_kernel(128, 128, 1, "bfloat16", False, True, 0, 148, "sm_103a")
