"""Execute portable SM103 packing kernels on available Blackwell hardware."""

import pytest
import torch
import torch.nn.functional as F

from tests._reference.sm103_moe import unpack, unswizzle, quantize_dequantize


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("gate_first", [False, True])
def test_routed_quantizer_matches_nvfp4_reference(activation, gate_first):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("Blackwell GPU required for FP4 conversion instructions")
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.pointwise import RoutedQuantize
    from b12x._lib.architecture import architecture_for

    device = torch.device("cuda", torch.cuda.current_device())
    k, routes, top_k, e = 256, 8, 2, 4
    x = torch.randn(
        (routes if activation else routes // top_k, 2 * k if activation else k),
        device=device,
        dtype=torch.bfloat16,
    )
    ids = torch.tensor(
        [0, 1, 2, 3, -1, e, 2**32 + 1, -(2**32)], device=device, dtype=torch.int64
    )
    gs = torch.tensor([0.5, 1.0, 1.5, 2.0], device=device)
    q = torch.empty((routes, k // 2), device=device, dtype=torch.uint8)
    sf = torch.empty((routes, 128, k // 16), device=device, dtype=torch.float8_e4m3fn)
    kernel = RoutedQuantize(k, top_k, e, activation=activation, gate_first=gate_first)
    args = [
        pointer(t, v)
        for t, v in (
            (cutlass.BFloat16, x),
            (cutlass.Int64, ids),
            (cutlass.Float32, gs),
            (cutlass.Uint8, q),
            (cutlass.Float8E4M3FN, sf),
        )
    ]
    args += [
        cutlass.Int32(routes),
        cutlass.Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    compiled = cute.compile(kernel, *args, options=f"--gpu-arch={target}")
    compiled(*args)
    decoded = unpack(q, unswizzle(sf, 128, k)[:, 0])
    expected = torch.zeros_like(decoded)
    for route in range(routes):
        expert = int(ids[route])
        if not 0 <= expert < e:
            continue
        if activation:
            a, b = x[route].float().chunk(2)
            gate, up = (a, b) if gate_first else (b, a)
            inp = F.silu(gate) * up
        else:
            inp = x[route // top_k]
        expected[route] = quantize_dequantize(inp, gs[expert])
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)
