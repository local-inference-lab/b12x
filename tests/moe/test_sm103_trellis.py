"""Exact portable t256 reconstruction; complete SM103 experts remain unsupported."""

import pytest
import torch

from tests._reference.trellis_decode import (
    CODEBOOK_RATES,
    codebook_tensor,
    native_weight,
)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_native_trellis_tile_reconstruction(bits):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis import ReconstructTrellisTiles
    from tests._reference.trellis_moe import build_trellis_weight

    device = torch.device("cuda", torch.cuda.current_device())
    packed, expected = build_trellis_weight(
        torch.Generator().manual_seed(43), 2, 32, 32, bits, device
    )
    tiles = packed.numel() // (16 * bits)
    out = torch.full((tiles, 16, 16), float("nan"), device=device, dtype=torch.bfloat16)
    lut = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    args = [
        pointer(t, x)
        for t, x in (
            (cutlass.Uint32, packed),
            (cutlass.Uint8, lut),
            (cutlass.BFloat16, out),
        )
    ]
    args += [
        cutlass.Int32(tiles),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    compiled = cute.compile(
        ReconstructTrellisTiles(bits, tiles), *args, options=f"--gpu-arch={target}"
    )
    compiled(*args)
    dense = out.reshape(2, 2, 2, 16, 16).permute(0, 2, 3, 1, 4).reshape_as(expected)
    torch.testing.assert_close(dense.float(), expected, atol=0, rtol=0)
    # The same capacity callable must honor a shorter live extent.
    out.fill_(float("nan"))
    args[-2] = cutlass.Int32(1)
    compiled(*args)
    assert torch.isfinite(out[0]).all()
    assert torch.isnan(out[1:]).all()


@pytest.mark.parametrize("codebook,bits", CODEBOOK_RATES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_codebook_reconstruction_frozen_callable_and_graph(
    codebook, bits, dtype, monkeypatch
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis import ReconstructTrellisTiles

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(813)
    cpu = torch.randint(
        -32768, 32768, (3, 5, 9, 16 * bits), dtype=torch.int16, generator=generator
    )
    cpu[0, 0, 0].fill_(0)
    cpu[0, 0, 1].fill_(-1)
    cpu[0, 0, 2, ::2] = 0x5555
    cpu[0, 0, 2, 1::2] = -21846
    expected = native_weight(cpu, bits, codebook).to(device=device, dtype=dtype)
    packed = cpu.to(device)
    lut = codebook_tensor(codebook, device)
    tiles = cpu.numel() // (16 * bits)
    out = torch.full((tiles, 16, 16), float("nan"), device=device, dtype=dtype)
    args = [
        pointer(t, x)
        for t, x in (
            (cutlass.Uint32, packed),
            (cutlass.Uint8, lut),
            (cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16, out),
        )
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    compiled = cute.compile(
        ReconstructTrellisTiles(bits, tiles, codebook=codebook),
        *args,
        cutlass.Int32(tiles),
        stream,
        options=f"--gpu-arch={target}",
    )

    def no_compile(*args, **kwargs):
        raise AssertionError("live tile count or replay triggered compilation")

    monkeypatch.setattr(cute, "compile", no_compile)
    expected_tiles = (
        expected.reshape(3, 9, 16, 5, 16).permute(0, 3, 1, 2, 4).reshape_as(out)
    )
    for live in (tiles, 1, 4, 31, tiles - 1):
        out.fill_(float("nan"))
        compiled(*args, cutlass.Int32(live), stream)
        torch.testing.assert_close(out[:live], expected_tiles[:live], atol=0, rtol=0)
        assert torch.isnan(out[live:]).all()
    compiled(*args, cutlass.Int32(tiles), stream)
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph):
        compiled(
            *args,
            cutlass.Int32(tiles),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
    cpu.bitwise_xor_(0x421)
    packed.copy_(cpu)
    mutated = native_weight(cpu, bits, codebook).to(device=device, dtype=dtype)
    mutated_tiles = (
        mutated.reshape(3, 9, 16, 5, 16).permute(0, 3, 1, 2, 4).reshape_as(out)
    )
    owners = tuple(t.data_ptr() for t in (packed, lut, out))
    replay_bytes = torch.cuda.memory_allocated()
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == replay_bytes
    assert tuple(t.data_ptr() for t in (packed, lut, out)) == owners
    assert before <= replay_bytes
    assert torch.isfinite(out).all() and torch.count_nonzero(out)
    torch.testing.assert_close(out, mutated_tiles, atol=0, rtol=0)
    assert not torch.equal(mutated_tiles, expected_tiles)
