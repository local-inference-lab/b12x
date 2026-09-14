"""Execute portable SM103 packing kernels on available Blackwell hardware."""

import pytest
import torch
import torch.nn.functional as F

from b12x.moe._shared.kernels.materialized_nvfp4_reference import (
    unpack,
    unswizzle,
    quantize_dequantize,
)


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("gate_first", [False, True])
@pytest.mark.parametrize("input_kind", ["random", "saturated", "zero"])
def test_routed_quantizer_matches_nvfp4_reference(activation, gate_first, input_kind):
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
    if input_kind == "saturated":
        x.mul_(4096)
    elif input_kind == "zero":
        x.zero_()
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


def test_route_and_reduction_grids_exceed_65535_without_recompilation():
    """Large prefill uses the same callable as decode across the old grid limit."""
    from tests.conftest import require_sm103_or_sm12x

    require_sm103_or_sm12x()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.pointwise import RoutedQuantize, TopKSum

    tokens, top_k, hidden = 65537, 2, 256
    routes = tokens * top_k
    device = torch.device("cuda", torch.cuda.current_device())
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    x = torch.full((tokens, hidden), 2.0, dtype=torch.bfloat16, device=device)
    x[:, 0] = torch.arange(tokens, device=device).remainder(13).sub(6).float() / 4
    ids = torch.zeros(routes, dtype=torch.int64, device=device)
    gs = torch.ones(1, device=device)
    q = torch.empty((routes, hidden // 2), dtype=torch.uint8, device=device)
    sf = torch.empty(
        (routes, 128, hidden // 16), dtype=torch.float8_e4m3fn, device=device
    )
    quant_args = [
        pointer(dtype, value)
        for dtype, value in (
            (cutlass.BFloat16, x),
            (cutlass.Int64, ids),
            (cutlass.Float32, gs),
            (cutlass.Uint8, q),
            (cutlass.Float8E4M3FN, sf),
        )
    ] + [cutlass.Int32(2), cutlass.Int32(1), stream]
    quant = cute.compile(
        RoutedQuantize(hidden, top_k, 1), *quant_args, options=f"--gpu-arch={target}"
    )
    quant(*quant_args)
    quant_args[-3] = cutlass.Int32(routes)
    q.fill_(255)
    quant(*quant_args)
    chosen = torch.tensor([0, 65535, 65536, routes - 1], device=device)
    decoded = unpack(q[chosen], unswizzle(sf[chosen], 128, hidden)[:, 0])
    expected = quantize_dequantize(x[chosen // top_k], gs[0])
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)

    # Dyadic operands make both the fused and separate FP32 sums exact.
    projected = torch.randint(-32, 33, (routes, hidden), device=device).to(
        torch.bfloat16
    ) / 8
    weights = torch.randint(-16, 17, (tokens, top_k), device=device).float() / 16
    output = torch.full_like(x, float("nan"))
    sum_args = [
        pointer(dtype, value)
        for dtype, value in (
            (cutlass.BFloat16, projected),
            (cutlass.Float32, weights),
            (cutlass.BFloat16, output),
        )
    ] + [cutlass.Int32(1), stream]
    reduction = cute.compile(
        TopKSum(hidden, top_k), *sum_args, options=f"--gpu-arch={target}"
    )
    reduction(*sum_args)
    sum_args[-2] = cutlass.Int32(tokens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sum_args[-1] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        reduction(*sum_args)
    weights.mul_(0.5)
    output.fill_(float("nan"))
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    expected = (
        (projected.view(tokens, top_k, hidden).float() * weights[..., None])
        .sum(1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
