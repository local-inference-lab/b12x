"""Device checks for route offsets and graph-stable P8 reduction."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_live_route_beyond_int32_offset_matches_rebased_route():
    from b12x.moe._shared.trellismx.p8_native_kernel import (
        P8NativeTPMoE, _gptr, current_cuda_stream,
    )
    import cutlass

    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("Large live-offset test needs 10 GiB free")
    runtime = object.__new__(P8NativeTPMoE)
    runtime.full_coupled = True
    runtime.topk, runtime.hidden, runtime.tp_rank = 8, 4096, 0
    runtime._coupled_reducer = None
    reducer = runtime._compile_full_coupled_reducer()
    tokens = 65537
    routes = torch.zeros((tokens * 8, 4096), device="cuda")
    weights = torch.zeros(tokens * 8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(923)
    routes[-8:] = torch.randn((8, 4096), device="cuda", generator=generator)
    weights[-8:] = torch.randn(8, device="cuda", generator=generator)
    output = torch.empty((tokens, 4096), device="cuda", dtype=torch.bfloat16)
    reference = torch.empty((1, 4096), device="cuda", dtype=torch.bfloat16)

    def launch(r, w, o, m):
        reducer(_gptr(cutlass.Float32, r), _gptr(cutlass.Float32, w, 4),
                _gptr(cutlass.BFloat16, o), m, current_cuda_stream())

    launch(routes, weights, output, tokens)
    launch(routes[-8:], weights[-8:], reference, 1)
    torch.cuda.synchronize()
    assert torch.count_nonzero(reference) > 0
    torch.testing.assert_close(output[-1:], reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(routes, weights, output, tokens)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[-1:], reference, rtol=0, atol=0)
