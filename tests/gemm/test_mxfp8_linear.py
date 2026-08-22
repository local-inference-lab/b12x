"""gemm.mxfp8_linear: ModelOpt MXFP8 linear vs quantized reference, K-padding
semantics, and CUDA-graph capture of the default fused path.

Curated from b12x tests/test_gemm_mxfp8_linear.py (kept whole — it was
already tight).
"""

from __future__ import annotations

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm import mxfp8_linear
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
)

from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _quantize_modelopt_mxfp8_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    chunks = width // 32
    blocked = source.to(torch.float32).reshape(rows, chunks, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _reference_from_packed(source: torch.Tensor, packed_weight) -> torch.Tensor:
    rows, width = map(int, source.shape)
    padded_width = int(packed_weight.padded_in_features)
    if width != padded_width:
        padded = source.new_zeros((rows, padded_width))
        padded[:, :width] = source
        source = padded.contiguous()
    x_q = bfl.quantize_input(source)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values, packed_weight.weight.scale_rows
    )
    return x_deq @ w_deq.T


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = mxfp8_linear.pack_weight(weight, weight_scale)
    return source, weight_scale, packed


def test_mm_matches_quantized_reference_small_n() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260614)

    source, _, packed = _make_inputs(7, 128, 32)
    actual = mxfp8_linear.mm(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (7, 32)
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens", (2, 3, 8, 15, 16, 17, 32, 99))
def test_mm_writes_all_rows_for_unaligned_output_width(tokens: int) -> None:
    """A small-batch GEMM must store every live row when N spans multiple tiles."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814 + tokens)

    source, _, packed = _make_inputs(tokens, 7168, 132)
    actual = mxfp8_linear.mm(source, packed, expected_m=tokens)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (tokens, 132)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-2,
    )


def test_mm_unaligned_output_stride_captures_and_replays() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260822)

    source, _, packed = _make_inputs(8, 7168, 132)
    replacement = torch.randn_like(source).div_(4)
    mxfp8_linear.mm(source, packed)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_linear.mm(source, packed)
    source.copy_(replacement)
    expected = _reference_from_packed(source, packed)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=2e-2,
    )


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260615)

    source, weight_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.weight.values.shape == (40, 256)
    assert packed.weight.scale_rows.shape == (1, 40, 8)
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0, :, :5], weight_scale
    )
    assert torch.all(packed.weight.scale_rows.view(torch.uint8)[0, :, 5:] == 127)

    actual = mxfp8_linear.mm(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (3, 40)
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


def test_mm_default_fused_path_captures_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260616)

    source, _, packed = _make_inputs(1, 160, 40)

    eager = mxfp8_linear.mm(source, packed).clone()
    torch.cuda.synchronize()

    mxfp8_linear.mm(source, packed)  # warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_linear.mm(source, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_incremental_quantized_input_matches_one_shot_linear() -> None:
    """Aligned feature slices preserve one-shot MXFP8 quantization and GEMM."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260823)

    tokens = 257
    source, _, packed = _make_inputs(tokens, 256, 64)
    quantized = mxfp8_linear.empty_input(512, 256, device=source.device)
    mxfp8_linear.quantize_input_slice(
        source[:, :128].contiguous(), quantized, destination_column=0
    )
    mxfp8_linear.quantize_input_slice(
        source[:, 128:].contiguous(), quantized, destination_column=128
    )
    output_storage = torch.empty(
        (512, packed.out_features),
        device=source.device,
        dtype=torch.bfloat16,
    )
    actual = mxfp8_linear.mm_quantized(
        quantized,
        packed,
        tokens=tokens,
        out=output_storage,
        expected_m=tokens,
    ).clone()
    expected = mxfp8_linear.mm(source, packed, expected_m=tokens)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_incremental_quantized_input_replays_in_cuda_graph() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260824)

    tokens = 8
    source, _, packed = _make_inputs(tokens, 256, 64)
    replacement = torch.randn_like(source).div_(4)
    quantized = mxfp8_linear.empty_input(tokens, 256, device=source.device)
    output = torch.empty(
        (tokens, packed.out_features),
        device=source.device,
        dtype=torch.bfloat16,
    )

    def run() -> torch.Tensor:
        mxfp8_linear.quantize_input_slice(
            source[:, :128].contiguous(), quantized, destination_column=0
        )
        mxfp8_linear.quantize_input_slice(
            source[:, 128:].contiguous(), quantized, destination_column=128
        )
        return mxfp8_linear.mm_quantized(quantized, packed, out=output)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    source.copy_(replacement)
    expected = mxfp8_linear.mm(source, packed)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_incremental_quantized_input_runs_under_torch_compile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260825)

    tokens = 8
    source, _, packed = _make_inputs(tokens, 256, 64)
    quantized = mxfp8_linear.empty_input(tokens, 256, device=source.device)
    output = torch.empty(
        (tokens, packed.out_features),
        device=source.device,
        dtype=torch.bfloat16,
    )

    @torch.compile(fullgraph=True)
    def compiled(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        mxfp8_linear.quantize_input_slice(first, quantized, destination_column=0)
        mxfp8_linear.quantize_input_slice(second, quantized, destination_column=128)
        return mxfp8_linear.mm_quantized(quantized, packed, out=output)

    actual = compiled(
        source[:, :128].contiguous(),
        source[:, 128:].contiguous(),
    )
    expected = mxfp8_linear.mm(source, packed)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
