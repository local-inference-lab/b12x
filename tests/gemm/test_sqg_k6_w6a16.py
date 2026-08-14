from __future__ import annotations

import math

import pytest
import torch

from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
    sqg_xor_cheb_t12_lut_cpu,
)
from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear._small_m import (
    _configure_sqg_k6_lut,
    _extension,
)
from b12x.moe._shared.kernels.w4a16.host import (
    dense_trellis_gemm_scratch_elements_upper_bound,
    packed_gemm_scratch_elements,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
)


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


def _reconstruct_states(edges: torch.Tensor, bits: int) -> torch.Tensor:
    values = edges.to(torch.int64) & ((1 << bits) - 1)
    states = torch.zeros_like(values)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(torch.int16)


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
    states = _reconstruct_states(edges, bits)
    codebook = sqg_xor_cheb_t12_direct_lut_cpu().reshape(5, 1 << 16)[bits - 2]
    values = codebook.view(torch.float8_e4m3fn).float()
    decoded = values.index_select(0, (states.to(torch.int64) & 0xFFFF).flatten())
    decoded = decoded.reshape_as(states)
    decoded = decoded.index_select(-1, torch.argsort(_tensor_core_permutation()))
    k_tiles, n_tiles, _ = decoded.shape
    return (
        decoded.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
        .contiguous()
    )


def _identity_hadamard(
    source: torch.Tensor,
    destination: torch.Tensor,
    _left_scale,
    _right_scale,
    _scale: float,
) -> None:
    destination.copy_(source)


def _separate_hadamard(
    source: torch.Tensor,
    destination: torch.Tensor,
    left_scale,
    right_scale,
    _scale: float,
) -> None:
    _run_trellis_dense_hadamard128(
        source,
        destination,
        left_scale if left_scale is not None else right_scale,
        scale_before=left_scale is not None,
    )


def test_sqg_k6_direct_table_matches_codebook_definition() -> None:
    bits = 6
    width = 16 - bits
    history_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    codeword = torch.arange(1 << 16, dtype=torch.int64)
    history = codeword >> bits
    branch = codeword & branch_mask

    mixed = history ^ (history >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & history_mask
    syndrome = product >> (32 - bits)
    reversed_branch = torch.zeros_like(branch)
    for index in range(bits):
        reversed_branch |= ((branch >> index) & 1) << (bits - 1 - index)
    rank = ((reversed_branch ^ syndrome) << width) | phase

    expected = sqg_xor_cheb_t12_lut_cpu()[rank >> 4]
    actual = sqg_xor_cheb_t12_direct_lut_cpu().reshape(5, 1 << 16)[bits - 2]
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 GPU")
def test_sqg_k6_device_decoder_matches_every_direct_table_entry() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _configure_sqg_k6_lut(int(device_index))
    codewords = torch.arange(1 << 16, dtype=torch.int32).to(
        device=device, dtype=torch.int16
    )
    actual = _extension().decode_k6_sqg_codewords(codewords)
    labels = sqg_xor_cheb_t12_direct_lut_cpu().reshape(5, 1 << 16)[4]
    expected = labels.view(torch.float8_e4m3fn).to(device=device, dtype=torch.float16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", (1, 7, 32, 129, 8192))
def test_dense_trellis_scratch_bound_covers_every_schedule(rows: int) -> None:
    size_n = 6144
    sms = 104
    capacity = dense_trellis_gemm_scratch_elements_upper_bound(
        rows=rows,
        size_n=size_n,
        sms=sms,
    )
    for block_size in (8, 16, 32, 48, 64):
        route_slots = ((rows + block_size - 1) // block_size) * block_size
        required = packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=sms,
        )
        assert capacity >= required


@pytest.mark.skipif(not _sm12x_available(), reason="requires SM120 or SM121")
def test_sqg_k6_w6a16_matches_reference_and_replays_in_cuda_graph() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(0x53514706)
    bits = 6
    features = 128
    rows = 7
    edges = torch.randint(
        0,
        1 << bits,
        (features // 16, features // 16, 256),
        dtype=torch.int16,
        generator=generator,
    )
    packed = _pack_edges(edges, bits).to(device)
    scale = torch.ones(features, dtype=torch.float16, device=device)
    prepared = trellis_linear.prepare_weight(
        packed,
        scale,
        scale.clone(),
        codebook="sqg_xor_cheb_t12",
        params_dtype=torch.float16,
    )
    x = (torch.randn((rows, features), generator=generator) * 0.03125).to(
        device=device, dtype=torch.float16
    )
    output = torch.empty((rows, features), dtype=torch.float16, device=device)
    gemm_output = torch.empty_like(output)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty(
        trellis_linear.sqg_k6_w6a16_scratch_elements(
            rows,
            features,
            device=device,
        ),
        dtype=torch.float32,
        device=device,
    )

    actual = trellis_linear.run_sqg_k6_w6a16(
        x,
        prepared,
        output=output,
        gemm_output=gemm_output,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
        hadamard_128=_identity_hadamard,
    ).clone()
    expected_weight = _decode_reference(edges, bits).to(device)
    expected = (x.float() @ expected_weight).to(torch.float16)
    relative_error = (actual - expected).float().norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run_sqg_k6_w6a16(
            x,
            prepared,
            output=output,
            gemm_output=gemm_output,
            rotated_f16=rotated_f16,
            c_tmp=c_tmp,
            hadamard_128=_identity_hadamard,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.parametrize("rows", (1, 7, 32, 128))
@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 GPU")
def test_sqg_k6_small_m_matches_generic_pipeline_and_replays(
    rows: int,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(0x53514760 + rows)
    features = 128
    packed = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 96),
        dtype=torch.int16,
        generator=generator,
    ).to(device)
    suh = torch.randn(features, dtype=torch.float16, device=device)
    svh = torch.randn(features, dtype=torch.float16, device=device)
    prepared = trellis_linear.prepare_weight(
        packed,
        suh,
        svh,
        codebook="sqg_xor_cheb_t12",
        params_dtype=torch.float16,
    )
    x = torch.randn(
        (rows, features),
        dtype=torch.float16,
        generator=generator,
    ).to(device)
    generic_output = torch.empty_like(x)
    generic_gemm = torch.empty_like(x)
    actual_output = torch.empty_like(x)
    actual_gemm = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty(
        trellis_linear.sqg_k6_w6a16_scratch_elements(
            rows,
            features,
            device=device,
        ),
        dtype=torch.float32,
        device=device,
    )

    expected = trellis_linear.run_sqg_k6_w6a16(
        x,
        prepared,
        output=generic_output,
        gemm_output=generic_gemm,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
        hadamard_128=_separate_hadamard,
    ).clone()
    prepared.workspace.zero_()
    actual = trellis_linear.run_sqg_k6_w6a16(
        x,
        prepared,
        output=actual_output,
        gemm_output=actual_gemm,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
    ).clone()
    torch.cuda.synchronize(device)

    delta = actual.float() - expected.float()
    relative_l2 = delta.norm() / expected.float().norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    max_relative_to_range = delta.abs().max() / expected.float().abs().max().clamp_min(
        1e-12
    )
    assert float(relative_l2) < 1.5e-3
    assert float(cosine) > 0.999998
    assert float(max_relative_to_range) < 2e-3

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run_sqg_k6_w6a16(
            x,
            prepared,
            output=actual_output,
            gemm_output=actual_gemm,
            rotated_f16=rotated_f16,
            c_tmp=c_tmp,
        )
    actual_output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)
