"""Host contracts for SM103 precision dispatch, storage, and compiler admission."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.architecture import require_kernel_architecture, UnsupportedArchitectureError
from b12x.gemm.blockscaled._policy import BLOCKSCALED_POLICY, BlockscaledQuery
from b12x.policy import DeviceIdentity, PolicyContext, PolicySource


B300 = DeviceIdentity(vendor="nvidia", product_name="NVIDIA B300",
                      compute_capability=(10, 3), sm_count=148)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
def test_sm103_precision_uses_unmeasured_heuristic_without_embedded_profile(recipe):
    context = PolicyContext.for_identity(B300)
    assert context.profile_id is None
    query = BlockscaledQuery(recipe=recipe, in_features=384, out_features=136)
    resolution = context.resolve(BLOCKSCALED_POLICY, query)
    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.select(1) == (128, 64, 4)
    assert resolution.config.select(8) == (128, 64, 4)
    assert resolution.config.select(129) is None
    assert set(BLOCKSCALED_POLICY.encode_query(query)) == {"recipe", "in_features", "out_features"}
    unknown = replace(B300, compute_capability=(10, 9))
    assert not PolicyContext.for_identity(unknown).resolve(BLOCKSCALED_POLICY, query).config.a16_rows


def test_compiler_admits_only_portable_a16_and_native_dense_entry_types():
    for module in ("b12x.gemm.blockscaled._a16_cute", "b12x.gemm.blockscaled._sm103"):
        require_kernel_architecture(module, (10, 3))
    with pytest.raises(UnsupportedArchitectureError):
        require_kernel_architecture("b12x._lib.dense_gemm", (10, 3))
    from b12x.gemm.blockscaled._a16_cute import DenseA16Launch
    with pytest.raises(ValueError, match="inline"):
        DenseA16Launch(weight_only=None)


@pytest.mark.parametrize("vector", [16, 32])
@pytest.mark.parametrize("rows,k,groups", [(1, 128, 1), (129, 384, 2), (257, 1024, 3)])
def test_scale_storage_preserves_grouped_f8_128x4_layout(vector, rows, k, groups):
    from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
    from b12x.gemm.blockscaled._sm103 import _scale_storage
    physical = torch.empty(groups, ((rows + 127) // 128) * ((k // vector + 3) // 4) * 512, dtype=torch.uint8)
    view = (as_grouped_scale_view if vector == 16 else as_grouped_scale_view_mx)(physical, rows, k)
    result = _scale_storage(view, rows, k, groups, vector, physical.device, view.dtype)
    assert result.data_ptr() == physical.data_ptr() and result.is_contiguous()
    with pytest.raises(ValueError, match="shape"):
        _scale_storage(view, rows + 128, k, groups, vector, physical.device, view.dtype)


def test_native_compile_miss_fails_under_frozen_resolution():
    from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.gemm.blockscaled._sm103 import compile_kernel
    compile_kernel.cache_clear()
    freeze_kernel_resolution("host cache-miss contract")
    try:
        with pytest.raises(RuntimeError, match="frozen"):
            compile_kernel(136, 384, 1, "nvfp4", "bfloat16", 0)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("option", [dict(plain_fp8=True), dict(block_fp8=True), dict(swap_ab=True), dict(_tile_k_override=256)])
def test_unsupported_dense_modes_fail_before_compilation(monkeypatch, option):
    from b12x._lib.dense_gemm import dense_gemm
    monkeypatch.setattr("b12x._lib.gating.get_compute_capability", lambda device: (10, 3))
    tensor = SimpleNamespace(device=torch.device("cuda:0"))
    with pytest.raises(UnsupportedArchitectureError, match="override"):
        dense_gemm((tensor, tensor), (tensor, tensor), ab_dtype="float4_e2m1fn",
                   sf_dtype="float8_e4m3fn", c_dtype="bfloat16", sf_vec_size=16, **option)


def test_raw_output_rejects_interleaved_or_overlapping_groups():
    from b12x.gemm.blockscaled._sm103 import _grouped_layout
    _grouped_layout(torch.empty(2, 129, 136).permute(1, 2, 0), 129, 136, 2)
    with pytest.raises(ValueError, match="contiguous rows"):
        _grouped_layout(torch.empty(129, 136, 2), 129, 136, 2)
    with pytest.raises(ValueError, match="overlap"):
        _grouped_layout(torch.empty(129, 136, 1).expand(129, 136, 2), 129, 136, 2)


def test_sm103_timing_rejects_the_max_q_throttle_allowance():
    from benchmarks.benchmark_blockscaled_precision import _clock_checks
    fields = dict(uuid="GPU-B300", name="NVIDIA B300", pstate="P0",
                  **{"clocks.sm": "1500", "clocks.mem": "4000", "clocks_event_reasons.active": "0x0"})
    def snapshot(values):
        return dict(fields=list(values), values=list(values.values()), compute_capability=[10, 3])
    before = snapshot(fields)
    assert _clock_checks(before, before)["valid"]
    for change in ({"clocks_event_reasons.active": "0x4"}, {"pstate": "P1"},
                   {"clocks.mem": "3999"}, {"clocks.sm": "1600"}):
        assert not _clock_checks(before, snapshot(fields | change))["valid"]
