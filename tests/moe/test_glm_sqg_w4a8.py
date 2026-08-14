from __future__ import annotations

import math

import pytest
import torch

from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    empty_mxfp8_rows_for_dense_gemm,
    quantize_mxfp8_rows_torch,
)
from b12x.moe import glm_sqg_w4a8
from b12x.moe._shared.kernels.glm_trellis_w4a8 import (
    prepare_glm_route_packed_w4a8_projection,
    run_glm_route_packed_w4a8_projection,
)
from b12x.moe._shared.kernels.w4a16.kernel import pack_topk_routes_by_expert


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((12, 0), (12, 1))


def _pack_edges(edges: torch.Tensor, bits: int) -> torch.Tensor:
    tiles = edges.reshape(-1, 256).to(torch.int64) & ((1 << bits) - 1)
    symbol_shifts = torch.arange(bits - 1, -1, -1)
    word_shifts = torch.arange(15, -1, -1)
    spans = tiles.reshape(-1, 16, 16)
    bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(-1, 16, bits * 16)
    words = (bitstream.reshape(-1, 16, bits, 16) << word_shifts).sum(dim=-1)
    flat = words.reshape(-1, 16 * bits)
    packed = flat.reshape(flat.shape[0], -1, 2).flip(-1).reshape(flat.shape)
    return packed.to(torch.int16).reshape(*edges.shape[:-1], 16 * bits).contiguous()


def _tensor_core_permutation() -> torch.Tensor:
    permutation = [0] * 256
    for thread in range(32):
        rows = (
            (thread % 4) * 2,
            (thread % 4) * 2 + 1,
            (thread % 4) * 2 + 8,
            (thread % 4) * 2 + 9,
        )
        columns = (thread // 4, thread // 4 + 8)
        permutation[thread * 8 : thread * 8 + 8] = [
            rows[0] * 16 + columns[0],
            rows[1] * 16 + columns[0],
            rows[2] * 16 + columns[0],
            rows[3] * 16 + columns[0],
            rows[0] * 16 + columns[1],
            rows[1] * 16 + columns[1],
            rows[2] * 16 + columns[1],
            rows[3] * 16 + columns[1],
        ]
    return torch.tensor(permutation, dtype=torch.long)


def _decode_reference(edges: torch.Tensor, bits: int) -> torch.Tensor:
    values = edges.to(torch.int64) & ((1 << bits) - 1)
    states = torch.zeros_like(values)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * bits)
    states &= 0xFFFF
    codebook = sqg_xor_cheb_t12_direct_lut_cpu().reshape(5, 1 << 16)[bits - 2]
    decoded = (
        codebook.view(torch.float8_e4m3fn).float().index_select(0, states.flatten())
    )
    decoded = decoded.reshape_as(states)
    decoded = decoded.index_select(-1, torch.argsort(_tensor_core_permutation()))
    k_tiles, n_tiles, _ = decoded.shape
    return (
        decoded.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
        .contiguous()
    )


def test_acceptance_schedule_is_qualified_for_sm120(monkeypatch) -> None:
    monkeypatch.setenv("B12X_GLM_W4A8_ACCEPT_ARCH", "sm_120")
    monkeypatch.delenv("B12X_GLM_W4A8_KERNEL", raising=False)
    monkeypatch.delenv("B12X_GLM_W4A8_V2_BLOCKS", raising=False)
    monkeypatch.delenv("B12X_GLM_W4A8_V2_STAGES", raising=False)

    glm_sqg_w4a8.validate_glm_route_packed_w4a8_acceptance_kernel()


def test_acceptance_schedule_rejects_unqualified_override(monkeypatch) -> None:
    monkeypatch.setenv("B12X_GLM_W4A8_ACCEPT_ARCH", "120")
    monkeypatch.setenv("B12X_GLM_W4A8_V2_BLOCKS", "4")

    with pytest.raises(RuntimeError, match="qualified sm_120 schedule"):
        glm_sqg_w4a8.validate_glm_route_packed_w4a8_acceptance_kernel()


@pytest.mark.parametrize("arch", ("sm_90", "sm_103", "sm_130"))
def test_acceptance_schedule_rejects_other_architectures(monkeypatch, arch) -> None:
    monkeypatch.setenv("B12X_GLM_W4A8_ACCEPT_ARCH", arch)

    with pytest.raises(RuntimeError, match="supports SM120/SM121"):
        glm_sqg_w4a8.validate_glm_route_packed_w4a8_acceptance_kernel()


@pytest.mark.skipif(not _sm12x_available(), reason="requires SM120 or SM121")
def test_mixed_k3_k4_projection_matches_reference_and_replays_in_cuda_graph() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(0x53514734)
    size_k = 128
    size_n = 128
    topk = 2
    tokens = 4
    bits_by_expert = (3, 4)
    edges = tuple(
        torch.randint(
            0,
            1 << bits,
            (size_k // 16, size_n // 16, 256),
            dtype=torch.int16,
            generator=generator,
        )
        for bits in bits_by_expert
    )
    packed = tuple(
        _pack_edges(expert_edges, bits).to(device)
        for expert_edges, bits in zip(edges, bits_by_expert, strict=True)
    )
    prepared = prepare_glm_route_packed_w4a8_projection(
        packed,
        bits_by_expert,
        size_k=size_k,
        size_n=size_n,
    )
    topk_ids = torch.tensor(
        ((0, 1), (1, 0), (0, 1), (1, 0)),
        dtype=torch.int32,
        device=device,
    )
    packed_routes, block_experts, _ = pack_topk_routes_by_expert(
        topk_ids,
        128,
        len(bits_by_expert),
    )
    source = (torch.randn((tokens, size_k), generator=generator) * 0.03125).to(
        device=device, dtype=torch.float16
    )
    quantized = empty_mxfp8_rows_for_dense_gemm(
        tokens,
        size_k,
        device=device,
    )
    output = torch.empty((tokens * topk, size_n), dtype=torch.float16, device=device)

    def run() -> torch.Tensor:
        quantize_mxfp8_rows_cute(
            source,
            quantized.values,
            quantized.scale_rows,
            quantized.scale_mma,
            value_order="trellis_native_mma",
        )
        return run_glm_route_packed_w4a8_projection(
            quantized,
            prepared,
            packed_routes,
            block_experts,
            output,
            topk=topk,
            shared_input=True,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    canonical_quantized = quantize_mxfp8_rows_torch(source)
    source_reference = dequantize_mxfp8_rows_torch(
        canonical_quantized.values,
        canonical_quantized.scale_rows,
    )
    weights = tuple(
        _decode_reference(expert_edges, bits).to(device)
        for expert_edges, bits in zip(edges, bits_by_expert, strict=True)
    )
    expected = torch.stack(
        tuple(
            source_reference[route // topk] @ weights[int(expert)]
            for route, expert in enumerate(topk_ids.flatten().tolist())
        )
    ).to(torch.float16)
    relative_error = (actual - expected).float().norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)
