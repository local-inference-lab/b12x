"""Deferred SM103 execution tests for native compressed Trellis projections."""

from unittest.mock import patch

import pytest
import torch

from tests._reference.trellis_decode import (
    CODEBOOK_RATES,
    codebook_tensor,
    native_weight,
)


def _require_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("physical SM103 required for tcgen05 Trellis projection")


def _reference(a, decoded, ids):
    result = torch.zeros(
        (ids.numel(), decoded.shape[1]), dtype=torch.float16, device=a.device
    )
    for row, expert in enumerate(ids.cpu().tolist()):
        if 0 <= expert < decoded.shape[0]:
            result[row] = a[row].float() @ decoded[expert].float().T
    return result


def _run_projection(codebook, bits, *, n=144, k=80, id_dtype=torch.int64):
    _require_sm103()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm

    device = torch.device("cuda", torch.cuda.current_device())
    capacity, experts = 17, 3
    generator = torch.Generator().manual_seed(72)
    cpu = torch.randint(
        -32768,
        32768,
        (experts, k // 16, n // 16, 16 * bits),
        dtype=torch.int16,
        generator=generator,
    )
    packed = cpu.to(device)
    decoded = native_weight(cpu, bits, codebook).to(device)
    lut = codebook_tensor(codebook, device)
    a = torch.randn((capacity, k + 32), device=device, dtype=torch.float16) * 0.05
    ids = (torch.arange(capacity, dtype=id_dtype, device=device) % experts).contiguous()
    ids[1], ids[3] = -1, experts
    if id_dtype == torch.int64:
        ids[5] = 2**32 + 1
    a[1, :k] = float("nan")
    output = torch.full((capacity, n + 16), 17.0, dtype=torch.float16, device=device)
    args = [
        pointer(t, v)
        for t, v in (
            (cutlass.Float16, a),
            (cutlass.Uint32, packed),
            (cutlass.Uint8, lut),
            (cutlass.Int32 if id_dtype == torch.int32 else cutlass.Int64, ids),
            (cutlass.Float16, output),
        )
    ]
    stride = (cutlass.Int64(a.stride(0)), cutlass.Int64(output.stride(0)))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(
        RoutedTrellisGemm(n, k, experts, capacity, bits=bits, codebook=codebook),
        *args,
        cutlass.Int32(1),
        *stride,
        stream,
        options="--gpu-arch=sm_103a",
    )
    expected = _reference(a[:, :k], decoded, ids)
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    with patch.object(
        cute, "compile", side_effect=AssertionError("kernel resolution is frozen")
    ):
        for live in (1, 4, 8, capacity, 3):
            output.fill_(17)
            fn(*args, cutlass.Int32(live), *stride, stream)
            torch.testing.assert_close(
                output[:live, :n], expected[:live], rtol=0.003, atol=0.01
            )
            assert torch.all(output[live:] == 17)
            assert torch.all(output[:live, n:] == 17)
        fn(*args, cutlass.Int32(capacity), *stride, stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(
                *args,
                cutlass.Int32(capacity),
                *stride,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        a[:, :k].normal_(std=0.05)
        ids.fill_(2)
        cpu.bitwise_xor_(0x214)
        packed.copy_(cpu)
        decoded = native_weight(cpu, bits, codebook).to(device)
        changed = _reference(a[:, :k], decoded, ids)
        assert not torch.equal(changed, expected)
        addresses = tuple(t.data_ptr() for t in (a, packed, lut, ids, output))
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            output.fill_(17)
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert addresses == tuple(t.data_ptr() for t in (a, packed, lut, ids, output))
        torch.testing.assert_close(output[:, :n], changed, rtol=0.003, atol=0.01)
        assert torch.all(output[:, n:] == 17)


@pytest.mark.parametrize("codebook,bits", CODEBOOK_RATES)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_trellis_projection_frozen_resolution_and_graph(codebook, bits, id_dtype):
    _run_projection(codebook, bits, id_dtype=id_dtype)


@pytest.mark.parametrize("n,k", [(2304, 5120), (5120, 2304)])
def test_v41_sized_k3_projection(n, k):
    _run_projection("mcg", 3, n=n, k=k)


def test_trellis_projection_high_weight_and_output_offsets():
    _require_sm103()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm

    if torch.cuda.mem_get_info()[0] < 19 * 1024**3:
        pytest.skip("high-offset projection requires 19 GiB of free GPU memory")
    n, k, bits = 144, 80, 3
    per_expert_words = (n // 16) * (k // 16) * 8 * bits
    experts = 2**31 // per_expert_words + 2
    device = torch.device("cuda", torch.cuda.current_device())
    cpu = torch.randint(
        -32768,
        32768,
        (1, k // 16, n // 16, 16 * bits),
        dtype=torch.int16,
        generator=torch.Generator().manual_seed(133),
    )
    packed = torch.empty(
        (experts, k // 16, n // 16, 16 * bits), device=device, dtype=torch.int16
    )
    packed[-1].copy_(cpu[0])
    decoded = native_weight(cpu, bits, "mcg").to(device)
    a = torch.randn((2, k), device=device, dtype=torch.float16) * 0.05
    ids = torch.full((2,), experts - 1, device=device, dtype=torch.int64)
    output = torch.empty((2, 2**31 + 16), device=device, dtype=torch.float16)
    lut = codebook_tensor("mcg", device)
    args = [
        pointer(t, v)
        for t, v in (
            (cutlass.Float16, a),
            (cutlass.Uint32, packed),
            (cutlass.Uint8, lut),
            (cutlass.Int64, ids),
            (cutlass.Float16, output),
        )
    ] + [
        cutlass.Int32(2),
        cutlass.Int64(k),
        cutlass.Int64(output.stride(0)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    fn = cute.compile(
        RoutedTrellisGemm(n, k, experts, 2, bits=bits, codebook="mcg"),
        *args,
        options="--gpu-arch=sm_103a",
    )
    output[:, :n].fill_(float("nan"))
    fn(*args)
    expected = a.float() @ decoded[0].float().T
    torch.testing.assert_close(output[:, :n], expected.half(), rtol=0.003, atol=0.01)
