from __future__ import annotations

from dataclasses import replace

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm import tensor_fp8_linear
from b12x.gemm.tensor_fp8_linear import api as tensor_fp8_api

from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    output_scale = torch.tensor([0.0002], dtype=torch.float32, device="cuda")
    packed = tensor_fp8_linear.pack_weight(weight, output_scale)
    return source, weight, output_scale, packed


def test_mm_matches_static_tensor_fp8_reference() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260729)

    source, weight, output_scale, packed = _make_inputs(7, 128, 64)
    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()

    assert actual.shape == (7, 64)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_writes_all_rows_for_unaligned_output_stride() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814)

    source, weight, output_scale, packed = _make_inputs(3, 128, 132)
    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260730)

    source, weight, output_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.values.shape == (40, 256)
    assert torch.count_nonzero(packed.values[:, 160:]) == 0

    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_mm_uses_plain_fp8_mma_not_scale_storage() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260731)

    source, weight, output_scale, packed = _make_inputs(4, 128, 64)
    packed = replace(packed, scale_mma=torch.zeros_like(packed.scale_mma))

    actual = tensor_fp8_linear.mm(source, packed)
    expected = (source.float() @ weight.float().T) * output_scale
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-3,
    )


def test_is_supported_honors_kernel_probe(monkeypatch) -> None:
    monkeypatch.setattr(tensor_fp8_api, "default_is_supported", lambda *args, **kw: True)
    monkeypatch.setattr(
        tensor_fp8_api,
        "_kernel_is_supported",
        lambda: (False, "plain FP8 MMA unavailable"),
    )

    assert not tensor_fp8_api.is_supported()


def test_mm_default_path_captures() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260801)

    source, _, _, packed = _make_inputs(4, 128, 64)
    eager = tensor_fp8_linear.mm(source, packed).clone()
    torch.cuda.synchronize()

    tensor_fp8_linear.prewarm(packed, [4])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tensor_fp8_linear.mm(source, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
