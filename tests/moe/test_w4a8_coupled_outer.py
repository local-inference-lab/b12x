from __future__ import annotations

import pytest
import torch


def _hadamard_128(device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < 128:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h / (128.0**0.5)


def _had128(value: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    rows, width = value.shape
    return (value.float().view(rows, width // 128, 128) @ hadamard).view(
        rows, width
    )


def _h512(value: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    rows, width = value.shape
    work = value.float().view(rows, width // 512, 4, 128) @ hadamard
    h4 = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=torch.float32,
        device=value.device,
    ) * 0.5
    return torch.einsum("rbgc,gh->rbhc", work, h4).reshape(rows, width)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    rel_l2 = (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1.0e-9)
    ).item()
    assert cosine > 0.99999, cosine
    assert rel_l2 < 0.004, rel_l2


def test_trellis_midband_uses_m64() -> None:
    from b12x.moe.fused_moe._impl import _select_dynamic_tile_mn

    for m in (897, 1024, 1536, 2015):
        assert _select_dynamic_tile_mn(
            m * 16,
            384,
            "w4a8_mx",
            num_experts=896,
            activation="situ",
            trellis=True,
        ) == (64, 128)


def test_trellis_workspace_covers_smaller_m16_band() -> None:
    from b12x.moe.fused_moe._impl import _dynamic_capacity_geometry

    assert _dynamic_capacity_geometry(
        max_tokens=3080,
        num_topk=16,
        num_experts=896,
        n=384,
        quant_mode="w4a8_mx",
        activation="situ",
        deterministic_output=False,
        trellis=True,
    ) == (1791, 5373)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_routing_sanitizer_zeroes_inactive_entries() -> None:
    from b12x.moe.fused_moe._impl import sanitize_w4a8_routing

    device = torch.device("cuda")
    ids = torch.tensor([[-1, 0, 895, 896]], dtype=torch.int32, device=device)
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
    output_ids = torch.empty_like(ids)
    output_weights = torch.empty_like(weights)
    sanitize_w4a8_routing(
        ids,
        weights,
        output_ids,
        output_weights,
        num_experts=896,
    )
    assert output_ids.tolist() == [[0, 0, 895, 0]]
    assert output_weights.tolist() == [[0.0, 2.0, 3.0, 0.0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 16, 257])
def test_coupled_outer_transform_matches_torch(m: int) -> None:
    from b12x.moe.fused_moe._impl import run_w4a8_coupled_outer_transform

    device = torch.device("cuda")
    width = 3584
    torch.manual_seed(20260830 + m)
    x = torch.randn(m, width, dtype=torch.bfloat16, device=device)
    hadamard = _hadamard_128(device)

    input_scale = (torch.randn(width, device=device).sign() * 0.02).to(torch.float16)
    expected_input = _had128(
        (_h512(x.to(torch.float16), hadamard) * input_scale.float()).to(
            torch.float16
        ),
        hadamard,
    ).to(torch.bfloat16)
    actual_input = torch.empty_like(x)
    run_w4a8_coupled_outer_transform(
        x,
        actual_input,
        input_scale,
        output_transform=False,
    )
    _assert_close(actual_input, expected_input)

    output_scale = (
        torch.randint(0, 2, (width,), dtype=torch.int32, device=device)
        .mul_(2)
        .sub_(1)
        .to(torch.float16)
    )
    expected_output = _h512(
        _had128(x.to(torch.float16), hadamard) * output_scale.float(),
        hadamard,
    ).to(torch.bfloat16)
    actual_output = x.clone()
    run_w4a8_coupled_outer_transform(
        actual_output,
        actual_output,
        output_scale,
        output_transform=True,
    )
    _assert_close(actual_output, expected_output)


def test_coupled_outer_mxfp8_requires_e4m3_values() -> None:
    from b12x.moe.fused_moe._impl import (
        run_w4a8_coupled_outer_transform_mxfp8,
    )

    x = torch.empty((1, 512), dtype=torch.bfloat16)
    values = torch.empty((1, 512), dtype=torch.uint8)
    scale_rows = torch.empty((1, 16), dtype=torch.float8_e8m0fnu)
    scale = torch.empty((512,), dtype=torch.float16)

    with pytest.raises(TypeError, match="E4M3 values"):
        run_w4a8_coupled_outer_transform_mxfp8(
            x,
            values,
            scale_rows,
            scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 16, 257])
def test_coupled_outer_mxfp8_matches_two_stage_reference(m: int) -> None:
    from b12x.gemm import block_fp8_linear as bfl
    from b12x.moe.fused_moe._impl import (
        run_w4a8_coupled_outer_transform,
        run_w4a8_coupled_outer_transform_mxfp8,
    )
    from b12x.quantization import mxfp8

    device = torch.device("cuda")
    width = 3584
    torch.manual_seed(20260831 + m)
    x = torch.randn(m, width, dtype=torch.bfloat16, device=device)
    scale = (torch.randn(width, device=device).sign() * 0.02).to(torch.float16)

    transformed = torch.empty_like(x)
    run_w4a8_coupled_outer_transform(
        x,
        transformed,
        scale,
        output_transform=False,
    )
    reference = bfl.quantize_input(transformed)
    reference_scale_mma = torch.full_like(
        reference.scale_mma.view(torch.uint8), 127
    ).view(torch.float8_e8m0fnu)
    mxfp8.quantize_rows(
        transformed,
        reference.values,
        reference.scale_rows,
        reference_scale_mma,
        value_order="trellis_native_mma",
    )

    values = torch.empty(
        (m, width), dtype=torch.float8_e4m3fn, device=device
    )
    scale_rows = torch.empty(
        (m, width // 32), dtype=torch.float8_e8m0fnu, device=device
    )
    run_w4a8_coupled_outer_transform_mxfp8(
        x,
        values,
        scale_rows,
        scale,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        values.view(torch.uint8), reference.values.view(torch.uint8).reshape(m, width)
    )
    assert torch.equal(
        scale_rows.view(torch.uint8),
        reference.scale_rows.view(torch.uint8).reshape(m, width // 32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_coupled_outer_mxfp8_graph_replay_tracks_input() -> None:
    from b12x.gemm import block_fp8_linear as bfl
    from b12x.moe.fused_moe._impl import (
        run_w4a8_coupled_outer_transform,
        run_w4a8_coupled_outer_transform_mxfp8,
    )
    from b12x.quantization import mxfp8

    device = torch.device("cuda")
    m, width = 16, 3584
    torch.manual_seed(20260901)
    x = torch.randn(m, width, dtype=torch.bfloat16, device=device)
    next_x = torch.randn_like(x)
    scale = (torch.randn(width, device=device).sign() * 0.02).to(torch.float16)
    values = torch.empty(
        (m, width), dtype=torch.float8_e4m3fn, device=device
    )
    scale_rows = torch.empty(
        (m, width // 32), dtype=torch.float8_e8m0fnu, device=device
    )

    run_w4a8_coupled_outer_transform_mxfp8(x, values, scale_rows, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_w4a8_coupled_outer_transform_mxfp8(x, values, scale_rows, scale)
    x.copy_(next_x)
    graph.replay()
    torch.cuda.synchronize()

    transformed = torch.empty_like(x)
    run_w4a8_coupled_outer_transform(
        x,
        transformed,
        scale,
        output_transform=False,
    )
    reference = bfl.quantize_input(transformed)
    reference_scale_mma = torch.full_like(
        reference.scale_mma.view(torch.uint8), 127
    ).view(torch.float8_e8m0fnu)
    mxfp8.quantize_rows(
        transformed,
        reference.values,
        reference.scale_rows,
        reference_scale_mma,
        value_order="trellis_native_mma",
    )
    torch.cuda.synchronize()

    assert torch.equal(
        values.view(torch.uint8), reference.values.view(torch.uint8).reshape(m, width)
    )
    assert torch.equal(
        scale_rows.view(torch.uint8),
        reference.scale_rows.view(torch.uint8).reshape(m, width // 32),
    )
