"""Independent dense GEMM oracles and allocation-stable SM103 serving tests."""

import pytest
import torch

from b12x.gemm import blockscaled
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from tests.gemm import test_blockscaled_a16 as a16
from tests.gemm.test_blockscaled_a16 import (  # noqa: F401
    test_a16_reference,
    test_a16_frozen_callable_and_graph_replay,
    test_a16_short_scale_tile_tail,
    test_w4a16_raw_scale_identity_and_rounding,
    test_a16_rejects_tma_misalignment,
    test_a16_aot_compile_functionalizes_workspace,
    test_mxfp8_prewarm_covers_functional_and_out_under_frozen_resolution,
    test_native_weight_pairs_exhaustive,
)


@pytest.fixture(autouse=True)
def supported_a16_device(monkeypatch):
    def require():
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (12, 0), (12, 1)):
            pytest.skip("requires SM103/SM120/SM121")
    monkeypatch.setattr(a16, "require_b12x", require)


def require_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires physical SM103")


def operand(recipe, groups, rows, k):
    from b12x._lib.intrinsics import swizzle_block_scale
    vector = 16 if recipe == "nvfp4" else 32
    if recipe == "mxfp8":
        values = torch.randn(groups, rows, k, device="cuda").to(torch.float8_e4m3fn)
        decoded = values.float()
    else:
        codes = torch.randint(0, 16, (groups, rows, k), device="cuda", dtype=torch.uint8)
        values = codes[..., 0::2] | (codes[..., 1::2] << 4)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        decoded = lut[codes.long()]
    if recipe == "nvfp4":
        compact = (torch.rand(groups, rows, k // vector, device="cuda") + .125).to(torch.float8_e4m3fn)
        scales = compact.float()
    else:
        compact = torch.randint(123, 130, (groups, rows, k // vector), device="cuda", dtype=torch.uint8)
        scales = torch.exp2(compact.float() - 127)
    storage = swizzle_block_scale(compact)
    return values.permute(1, 2, 0), storage, decoded * scales.repeat_interleave(vector, -1)


def options(recipe, dtype):
    return dict(ab_dtype="float8_e4m3fn" if recipe == "mxfp8" else "float4_e2m1fn",
                sf_dtype="float8_e4m3fn" if recipe == "nvfp4" else "float8_e8m0fnu",
                sf_vec_size=16 if recipe == "nvfp4" else 32,
                c_dtype=str(dtype).removeprefix("torch."))


def check(actual, expected):
    actual = actual.float()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert float(relative) < .005 and float(cosine) > .9999
    torch.testing.assert_close(actual, expected, rtol=.008, atol=float(expected.abs().max()) * .008)


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_native_dense_counts_groups_tails_and_graphs(recipe, groups, dtype, monkeypatch):
    require_native()
    from b12x.gemm.blockscaled._sm103 import compile_kernel
    torch.manual_seed(2301)
    n, k, capacity = 136, 384, 257
    weight, weight_scales, decoded_weight = operand(recipe, groups, n, k)
    alpha = torch.linspace(.5, 1.5, groups, device="cuda")
    output = torch.empty(groups, capacity, n, device="cuda", dtype=dtype).permute(1, 2, 0)
    # Input and output group strides retain planned capacity across live counts.
    inputs = []
    for m in (1, 3, 8, 129, capacity):
        values, scales, decoded = operand(recipe, groups, m, k)
        backing = torch.empty(groups, capacity, values.shape[1], device="cuda", dtype=values.dtype)
        view = backing[:, :m].permute(1, 2, 0)
        view.copy_(values)
        inputs.append((view, scales, decoded))
    def call(item, m):
        return blockscaled.mm(item[:2], (weight, weight_scales), out=output[:m],
                              alpha=alpha, **options(recipe, dtype))
    call(inputs[0], 1)
    cache = compile_kernel.cache_info()
    freeze_kernel_resolution("SM103 dense live-count reuse")
    try:
        for item, m in zip(inputs, (1, 3, 8, 129, capacity), strict=True):
            expected = (torch.bmm(item[2], decoded_weight.transpose(1, 2)) * alpha[:, None, None]).permute(1, 2, 0)
            check(call(item, m), expected)
        assert compile_kernel.cache_info().misses == cache.misses
        item = inputs[-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(item, capacity)
        addresses = tuple(t.data_ptr() for t in (*item[:2], weight, weight_scales, alpha, output))
        for scale in (.25, 1.75):
            alpha.fill_(scale)
            output.fill_(float("nan"))
            graph.replay()
            expected = (torch.bmm(item[2], decoded_weight.transpose(1, 2)) * scale).permute(1, 2, 0)
            check(output, expected)
        assert addresses == tuple(t.data_ptr() for t in (*item[:2], weight, weight_scales, alpha, output))
        before = torch.cuda.memory_allocated()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected allocation"))
            call(item, capacity)
            graph.replay()
            torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("recipe,dtype", [
    ("nvfp4", torch.bfloat16), ("mxfp8", torch.bfloat16), ("mxfp8", torch.float16),
])
def test_packed_quantized_graph_and_independent_oracle(recipe, dtype, monkeypatch):
    a16.require_b12x()
    from benchmarks.benchmark_blockscaled_precision import _reference_weight, quantized_nvfp4_reference
    from b12x._lib.intrinsics import quant_dequant_mxfp8_torch
    weight, local, multiplier = _reference_weight(recipe, 136, 384)
    source = torch.randn(33, 384, device="cuda", dtype=dtype)
    out = torch.empty(33, 136, device="cuda", dtype=dtype)
    scratch = torch.empty(blockscaled.workspace_size(weight, 33), device="cuda", dtype=torch.uint8)
    activation_scale = torch.tensor([4.0], device="cuda")
    args = dict(activation_global_scale=activation_scale) if recipe == "nvfp4" else {}
    counts = (1, 4, 8, 17, 33)
    # SM12x retains its existing precision-regime specializations. SM103 must
    # reuse the callable warmed at the two endpoints for every interior count.
    warmup_counts = (1, 33) if torch.cuda.get_device_capability() == (10, 3) else counts
    blockscaled.prewarm(weight, warmup_counts, mode="quantized", out_dtype=dtype, workspace=scratch, **args)
    def call(m):
        return blockscaled.mm(source[:m], weight, out=out[:m], workspace=scratch, mode="quantized", **args)
    def reference(m):
        if recipe == "nvfp4":
            return quantized_nvfp4_reference(source[:m], local, multiplier, activation_scale)
        return quant_dequant_mxfp8_torch(source[:m]).float() @ local.T
    freeze_kernel_resolution("SM103 packed quantized projection")
    try:
        for m in counts:
            check(call(m), reference(m))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(33)
        for _ in range(3):
            source.normal_()
            scratch.fill_(255)
            out.fill_(float("nan"))
            graph.replay()
            check(out, reference(33))
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected allocation"))
            call(33)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("recipe", ["nvfp4", "mxfp4", "mxfp8"])
def test_native_scalar_alpha_empty_rows_and_tiny_output(recipe, monkeypatch):
    require_native()
    a, sfa, decoded_a = operand(recipe, 1, 1, 128)
    b, sfb, decoded_b = operand(recipe, 1, 8, 128)
    opts = options(recipe, torch.bfloat16)
    expected = torch.bmm(decoded_a, decoded_b.transpose(1, 2)).permute(1, 2, 0)
    out = torch.empty_like(expected, dtype=torch.bfloat16)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "ones", lambda *a, **kw: pytest.fail("implicit alpha allocation"))
        check(blockscaled.mm((a, sfa), (b, sfb), out=out, **opts), expected)
    alpha = torch.tensor([.125], device="cuda")
    check(blockscaled.mm((a, sfa), (b, sfb), alpha=alpha, **opts), expected * alpha)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    blockscaled.mm((a, sfa), (b, sfb), out=out, stream=stream, **opts)
    torch.cuda.current_stream().wait_stream(stream)
    check(out, expected)
    empty = torch.empty(0, a.shape[1], 1, device="cuda", dtype=a.dtype)
    scales = torch.empty(0, device="cuda", dtype=sfa.dtype)
    result = blockscaled.mm((empty, scales), (b, sfb), **opts)
    assert result.shape == (0, 8, 1)


def test_native_dense_row_address_past_int32_elements():
    """The final CTA row begins beyond 2^31 output elements (about 4 GiB)."""
    require_native()
    m, n, k = 4097, 524288, 128
    a = torch.full((m, k // 2, 1), 0x22, device="cuda", dtype=torch.uint8)
    b = torch.full((n, k // 2, 1), 0x22, device="cuda", dtype=torch.uint8)
    sfa = torch.full((((m + 127) // 128) * 2 * 512,), 1.0, device="cuda", dtype=torch.float8_e4m3fn)
    sfb = torch.full(((n // 128) * 2 * 512,), 1.0, device="cuda", dtype=torch.float8_e4m3fn)
    output = torch.full((m, n, 1), float("nan"), device="cuda", dtype=torch.bfloat16)
    blockscaled.mm((a, sfa), (b, sfb), out=output, **options("nvfp4", torch.bfloat16))
    assert (m - 1) * n == 2**31
    for row in (0, 127, 128, m - 2, m - 1):
        assert torch.all(output[row] == k)
