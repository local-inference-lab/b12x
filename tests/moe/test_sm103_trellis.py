"""Portable reconstruction qualification; complete SM103 experts remain unsupported."""

import pytest
import torch


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
