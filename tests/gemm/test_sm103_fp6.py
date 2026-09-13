"""Native FP6 GEMM oracles, live-count reuse, and graph execution on SM103."""

import pytest
import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm.blockscaled._fp6 import compile_kernel
from tests.gemm.test_sm103_blockscaled import check, require_native
from tests.quantization.test_fp6_workspace import decode, pack


def operand(fmt, packed, groups, rows, k):
    if fmt == "e4m3":
        codes = torch.randn(groups, rows, k, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        codes = torch.randint(0, 64, (groups, rows, k), device="cuda", dtype=torch.uint8)
    scales = torch.randint(122, 129, (groups, rows, k // 32), device="cuda", dtype=torch.uint8)
    reference = decode(codes, fmt) * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    values = pack(codes) if packed else codes
    return values.permute(1, 2, 0), swizzle_block_scale(scales), reference


@pytest.mark.parametrize("a_fmt,b_fmt,a_packed,b_packed", [
    ("e3m2", "e2m3", True, True), ("e2m3", "e3m2", True, True),
    ("e4m3", "e2m3", False, True), ("e3m2", "e4m3", True, False),
    ("e3m2", "e2m3", False, True), ("e2m3", "e3m2", True, False),
    ("e3m2", "e3m2", False, False),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_native_fp6_oracle_group_strides_counts_and_graph(a_fmt, b_fmt, a_packed, b_packed, dtype, monkeypatch):
    require_native()
    groups, capacity, n, k = 2, 257, 136, 384
    weight, sfb, decoded_weight = operand(b_fmt, b_packed, groups, n, k)
    alpha = torch.tensor([.25, 1.5], device="cuda")
    correction = torch.linspace(.5, 1.5, capacity, device="cuda").to(torch.bfloat16)
    out = torch.empty(groups, capacity, n, device="cuda", dtype=dtype).permute(1, 2, 0)
    items = []
    for m in (1, 3, 8, 129, capacity):
        a, sfa, decoded_a = operand(a_fmt, a_packed, groups, m, k)
        backing = torch.empty(groups, capacity, a.shape[1], device="cuda", dtype=torch.uint8)
        view = backing[:, :m].permute(1, 2, 0)
        view.copy_(a)
        items.append((view, sfa, decoded_a))
    def call(item, m):
        return dense_gemm(item[:2], (weight, sfb), out=out[:m], alpha=alpha,
            row_scale=correction[:m], ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu",
            sf_vec_size=32, c_dtype=str(dtype).removeprefix("torch."), a_fmt=a_fmt, b_fmt=b_fmt,
            a_preexpanded=not a_packed, b_preexpanded=not b_packed, b_packed=b_packed, expected_m=capacity)
    def expected(item, m):
        result = (torch.bmm(item[2], decoded_weight.transpose(1, 2)) * alpha[:, None, None]).permute(1, 2, 0)
        return (result.to(dtype).float() * correction[:m, None, None].float()).to(dtype).float()
    call(items[0], 1)
    cache = compile_kernel.cache_info()
    freeze_kernel_resolution("native FP6 live rows retain one compiled callable")
    try:
        for item, m in zip(items, (1, 3, 8, 129, capacity), strict=True):
            check(call(item, m), expected(item, m))
        assert compile_kernel.cache_info().misses == cache.misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(items[-1], capacity)
        for factor in (.5, 2.):
            alpha.fill_(factor)
            out.fill_(float("nan"))
            graph.replay()
            check(out, expected(items[-1], capacity))
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("native FP6 allocated"))
            call(items[-1], capacity)
            graph.replay()
            torch.cuda.synchronize()
        assert before["allocation.all.allocated"] == torch.cuda.memory_stats()["allocation.all.allocated"]
    finally:
        unfreeze_kernel_resolution()


def test_native_fp6_scalar_alpha_empty_rows_and_stream():
    require_native()
    a, sfa, decoded_a = operand("e3m2", True, 1, 1, 128)
    b, sfb, decoded_b = operand("e2m3", True, 1, 8, 128)
    options = dict(ab_dtype="float6_e3m2fn", a_fmt="e3m2", b_fmt="e2m3",
                   sf_dtype="float8_e8m0fnu", sf_vec_size=32, c_dtype="bfloat16")
    expected = torch.bmm(decoded_a, decoded_b.transpose(1, 2)).permute(1, 2, 0)
    check(dense_gemm((a, sfa), (b, sfb), **options), expected)
    alpha = torch.tensor([.125], device="cuda")
    out = torch.empty_like(expected, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dense_gemm((a, sfa), (b, sfb), out=out, alpha=alpha, stream=stream, **options)
    torch.cuda.current_stream().wait_stream(stream)
    check(out, expected * alpha)
    # Empty scale storage describes zero live scale tiles.
    from b12x._lib.fp6 import as_grouped_mxfp6_scale_view
    empty_sf = as_grouped_mxfp6_scale_view(torch.empty(1, 0, device="cuda", dtype=torch.uint8), 0, 128)
    assert dense_gemm((a[:0], empty_sf), (b, sfb), **options).shape == (0, 8, 1)


def test_native_fp6_output_row_past_int32_elements():
    require_native()
    m, n, k = 4097, 524288, 128
    a = pack(torch.full((m, k), 12, device="cuda", dtype=torch.uint8))[..., None]
    b = pack(torch.full((n, k), 8, device="cuda", dtype=torch.uint8))[..., None]
    sfa = torch.full((((m + 127) // 128) * 512,), 127, device="cuda", dtype=torch.uint8)
    sfb = torch.full(((n // 128) * 512,), 127, device="cuda", dtype=torch.uint8)
    out = torch.full((m, n, 1), float("nan"), device="cuda", dtype=torch.bfloat16)
    dense_gemm((a, sfa), (b, sfb), out=out, ab_dtype="float6_e3m2fn", a_fmt="e3m2", b_fmt="e2m3",
               sf_dtype="float8_e8m0fnu", sf_vec_size=32, c_dtype="bfloat16")
    assert (m - 1) * n == 2**31
    for row in (0, 127, 128, m - 2, m - 1):
        assert torch.all(out[row] == k)


def test_native_fp6_opaque_serving_op_compile_and_capture():
    require_native()
    from b12x.quantization.mxfp6 import quantize_dense_weight_to_fp6
    from b12x.quantization.mxfp6.fp6_dense_op import fp6_dense_linear
    weight = quantize_dense_weight_to_fp6(torch.randn(128, 384, device="cuda", dtype=torch.bfloat16))
    packed = weight.gemm_weight()
    def operation(x):
        return fp6_dense_linear(x, packed, weight.scale_storage, weight.global_scale,
                                weight.fmt, 128, 384, weight.act_fmt)
    source = torch.randn(8, 384, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(operation, fullgraph=True, backend="aot_eager")
    torch.testing.assert_close(compiled(source), operation(source), atol=0, rtol=0)
    freeze_kernel_resolution("FP6 opaque serving op capture is prewarmed")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = compiled(source)
        source.normal_()
        graph.replay()
        torch.testing.assert_close(result, operation(source), atol=0, rtol=0)
    finally:
        unfreeze_kernel_resolution()
