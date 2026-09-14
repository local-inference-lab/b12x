"""FP8 arithmetic, runtime count reuse, and graph replay on the shared entry."""

from dataclasses import replace

import pytest
import torch

from b12x.gemm import blockscaled, tensor_fp8_linear
from b12x.gemm.blockscaled import _fp8_cute as fp8
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from tests.gemm.test_sm103_blockscaled import check


@pytest.fixture(autouse=True)
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (12, 0), (12, 1)):
        pytest.skip("requires physical SM103/SM120/SM121")


def call(a, b, sfa, sfb, out, *, block, alpha=None, stream=None):
    options = dict(ab_dtype="float8_e4m3fn", c_dtype=str(out.dtype).removeprefix("torch."),
                   sf_dtype="float32" if block else "float8_e8m0fnu",
                   sf_vec_size=128 if block else 32, block_fp8=block, alpha=alpha, stream=stream)
    if torch.cuda.get_device_capability() == (10, 3):
        return blockscaled.mm((a, sfa), (b, sfb), out=out, plain_fp8=True, **options)
    # Execute the identical entry on its actual SM12x target for regression.
    return fp8.execute((a, sfa), (b, sfb), out, **options)


def reference(a, b, sfa, sfb, block, alpha):
    if block:
        scaled_a = a[..., 0].float() * sfa.repeat_interleave(128, dim=1)
        scaled_b = b[..., 0].float() * sfb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        result = (scaled_a @ scaled_b.T)[..., None]
    else:
        result = torch.bmm(a.permute(2, 0, 1).float(), b.permute(2, 1, 0).float()).permute(1, 2, 0)
    return result if alpha is None else result * alpha


@pytest.mark.parametrize("block,n,k,groups", [(False, 64, 128, 1), (False, 132, 256, 1),
    (False, 136, 384, 2), (False, 16386, 1024, 1), (True, 256, 384, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp8_counts_scales_groups_and_graph(block, n, k, groups, dtype, monkeypatch):
    capacity = 257
    a = torch.randn(groups, capacity, k, device="cuda").clamp(-4, 4).to(torch.float8_e4m3fn).permute(1, 2, 0)
    b = torch.randn(groups, n, k, device="cuda").clamp(-4, 4).to(torch.float8_e4m3fn).permute(1, 2, 0)
    out = torch.empty(groups, capacity, n, device="cuda", dtype=dtype).permute(1, 2, 0)
    sfa = torch.rand(capacity, k // 128, device="cuda") * .5 + .125 if block else None
    sfb = torch.rand(n // 128, k // 128, device="cuda") * 2 + .25 if block else None
    alpha = torch.tensor([.375], device="cuda")
    def run(m):
        return call(a[:m], b, sfa[:m] if block else None, sfb, out[:m], block=block, alpha=alpha)
    run(1)
    cached = fp8.compile_kernel.cache_info()
    freeze_kernel_resolution("FP8 live counts reuse immutable weight geometry")
    try:
        for m in (1, 4, 8, 17, 65, capacity):
            expected = reference(a[:m], b, sfa[:m] if block else None, sfb, block, alpha)
            check(run(m), expected)
        assert fp8.compile_kernel.cache_info().misses == cached.misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(capacity)
        tensors = [a, b, out, alpha] + ([sfa, sfb] if block else [])
        addresses = [t.data_ptr() for t in tensors]
        for _ in range(2):
            a.copy_(torch.randn_like(a, dtype=torch.float32).clamp(-4, 4).to(a.dtype))
            alpha.mul_(.5)
            if block:
                sfa.mul_(1.5)
                sfb.mul_(.75)
            expected = reference(a, b, sfa, sfb, block, alpha)
            out.fill_(float("nan"))
            graph.replay()
            check(out, expected)
        assert addresses == [t.data_ptr() for t in tensors]
        before = torch.cuda.memory_allocated()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected allocation"))
            patch.setattr(torch, "ones", lambda *a, **kw: pytest.fail("unexpected unit scale"))
            run(capacity)
            graph.replay()
            torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("block", [False, True])
def test_fp8_implicit_alpha_zero_scales_empty_and_stream(block, monkeypatch):
    a = torch.ones(4, 128, 1, device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones(128, 128, 1, device="cuda").to(a.dtype)
    out = torch.empty(4, 128, 1, device="cuda", dtype=torch.bfloat16)
    sfa = torch.ones(4, 1, device="cuda") if block else None
    sfb = torch.ones(1, 1, device="cuda") if block else None
    with monkeypatch.context() as patch:
        patch.setattr(torch, "ones", lambda *a, **kw: pytest.fail("implicit alpha allocation"))
        call(a, b, sfa, sfb, out, block=block)
    assert torch.all(out == 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    if block:
        sfb.zero_()
        stream.wait_stream(torch.cuda.current_stream())
    call(a, b, sfa, sfb, out, block=block, stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.all(out == (0 if block else 128))
    result = call(a[:0], b, sfa[:0] if block else None, sfb, out[:0], block=block)
    assert result.shape == (0, 128, 1)


@pytest.mark.parametrize("n,k", [(132, 128), (40, 160), (16386, 1024)])
def test_packed_tensor_fp8_graph_and_scale_independence(n, k):
    a = torch.randn(33, k, device="cuda").clamp(-4, 4).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").clamp(-4, 4).to(a.dtype)
    alpha = torch.tensor([.125], device="cuda")
    packed = tensor_fp8_linear.pack_weight(b, alpha)
    packed = replace(packed, scale_mma=torch.zeros_like(packed.scale_mma), block_scale=torch.zeros_like(packed.block_scale))
    native = torch.cuda.get_device_capability() == (10, 3)
    tensor_fp8_linear.prewarm(packed, (1, 33) if native else (1, 4, 8, 17, 33))
    if native:
        freeze_kernel_resolution("packed tensor-FP8 reuse")
    try:
        def run(m):
            return tensor_fp8_linear.mm(a[:m], packed, expected_m=33)
        for m in (1, 4, 8, 17, 33):
            check(run(m), (a[:m].float() @ b.float().T) * alpha)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(33)
        address = out.data_ptr()
        for _ in range(2):
            a.copy_(torch.randn_like(a, dtype=torch.float32).clamp(-4, 4).to(a.dtype))
            graph.replay()
            check(out, (a.float() @ b.float().T) * alpha)
        assert out.data_ptr() == address
    finally:
        if native:
            unfreeze_kernel_resolution()


@pytest.mark.parametrize("block", [False, True])
def test_fp8_output_rows_past_int32_elements(block):
    m, n, k = 4097, 524288, 128
    a = torch.ones(m, k, 1, device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones(n, k, 1, device="cuda").to(a.dtype)
    out = torch.full((m, n, 1), float("nan"), device="cuda", dtype=torch.bfloat16)
    sfa = torch.ones(m, 1, device="cuda") if block else None
    sfb = torch.ones(n // 128, 1, device="cuda") if block else None
    call(a, b, sfa, sfb, out, block=block)
    assert (m - 1) * n == 2**31
    for row in (0, 15, 16, m - 2, m - 1):
        assert torch.all(out[row] == k)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_serialized_compact_block_fp8_public_graph(dtype):
    a = torch.randn(17, 384, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(256, 384, device="cuda").to(a.dtype)
    sfa = torch.rand(17, 3, device="cuda") + .125
    sfb = torch.rand(2, 3, device="cuda") + .125
    def run():
        return blockscaled.mm_block_fp8(a, sfa, b, sfb, out_dtype=dtype, expected_m=17)
    check(run(), reference(a[..., None], b[..., None], sfa, sfb, True, None)[..., 0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    sfa.mul_(.5)
    sfb.mul_(1.5)
    graph.replay()
    check(out, reference(a[..., None], b[..., None], sfa, sfb, True, None)[..., 0])


@pytest.mark.parametrize("block,n,k", [(False, 132, 160), (False, 4096, 4096),
                                      (True, 256, 384), (True, 4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_caller_workspace_capacity_reuse_and_replay(block, n, k, dtype, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    a = (torch.randn(33, k, device="cuda") / 4).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda") / 4).to(a.dtype)
    alpha = torch.tensor([.125], device="cuda")
    sfa = torch.rand(33, k // 128, device="cuda") + .125 if block else None
    sfb = torch.rand(n // 128, k // 128, device="cuda") + .125 if block else None
    weight = (b, sfb) if block else tensor_fp8_linear.pack_weight(b, alpha)
    scratch = torch.empty(blockscaled.workspace_size(weight, 33), dtype=torch.uint8, device="cuda")
    output = torch.empty(33, n, device="cuda", dtype=dtype)
    caps = (1, 4, 8, 33)

    def run(m, out=None):
        capacity = next(c for c in caps if c >= m)
        if block:
            return blockscaled.mm_block_fp8(a[:m], sfa[:m], b, sfb, out_dtype=dtype,
                expected_m=capacity, out=out, workspace=scratch)
        return blockscaled.mm(a[:m], weight, out_dtype=dtype, expected_m=capacity,
                              out=out, workspace=scratch)

    if not block:
        blockscaled.prewarm(weight, caps, out_dtype=dtype, workspace=scratch)
    for m in caps:
        run(m, output[:m])
    freeze_kernel_resolution("FP8 caller-owned capacity and scratch")
    try:
        for m in (1, 3, 4, 7, 8, 9, 33):
            actual = run(m, output[:m])
            if dtype == torch.bfloat16 and torch.cuda.get_device_capability() != (10, 3):
                # The retained SM12x policy uses unordered BF16 atomic split-K sums.
                check(actual, run(m).float())
            else:
                torch.testing.assert_close(actual, run(m), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(m, output[:m])
            for _ in range(2):
                a.copy_((torch.randn_like(a, dtype=torch.float32) / 4).to(a.dtype))
                scratch.fill_(255)
                output.fill_(float("nan"))
                before = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before
                if block:
                    expected = reference(a[:m, :, None], b[:, :, None], sfa[:m], sfb, True, None)[..., 0]
                else:
                    expected = (a[:m].float() @ b.float().T) * alpha
                check(output[:m], expected)
            graph.reset()
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("block", [False, True])
def test_fp8_workspace_validation_empty_and_explicit_stream(block, monkeypatch):
    m, n, k = 3, 256, 384 if block else 160
    a = torch.ones(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones(n, k, device="cuda").to(a.dtype)
    alpha = torch.tensor([.25], device="cuda")
    sfa = torch.ones(m, k // 128, device="cuda") if block else None
    sfb = torch.ones(n // 128, k // 128, device="cuda") if block else None
    weight = (b, sfb) if block else tensor_fp8_linear.pack_weight(b, alpha)
    size = blockscaled.workspace_size(weight, m)
    scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")

    def run(source=a, output=out, workspace=scratch, hint=4, stream=None):
        if block:
            return blockscaled.mm_block_fp8(source, sfa[:source.shape[0]], b, sfb,
                out_dtype=torch.float16, out=output, workspace=workspace,
                expected_m=hint, stream=stream)
        return blockscaled.mm(source, weight, out_dtype=torch.float16, out=output,
                              workspace=workspace, expected_m=hint, stream=stream)

    with pytest.raises(ValueError, match="at least"):
        run(workspace=scratch[:-1])
    with pytest.raises(ValueError, match="covering"):
        run(hint=1)
    with pytest.raises(ValueError, match="overlap"):
        run(output=scratch[:m * n * 2].view(torch.float16).view(m, n))
    with pytest.raises(ValueError, match="contiguous"):
        run(workspace=torch.empty(size * 2, device="cuda", dtype=torch.uint8)[::2])
    with pytest.raises(ValueError, match="dtype"):
        run(workspace=scratch.float())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    run(stream=stream)
    stream.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, k if block else k * .25), rtol=0, atol=0)
    allocations = []
    empty = torch.empty
    def record_allocation(*args, **kwargs):
        allocations.append(torch.cuda.current_stream().cuda_stream)
        return empty(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", record_allocation)
        owned = run(output=None, workspace=None, stream=stream)
    assert len(allocations) >= 2 and all(s == stream.cuda_stream for s in allocations)
    # Reuse default-stream allocations while the explicit-stream call owns its scratch.
    churn = [torch.full((size,), 255, dtype=torch.uint8, device="cuda") for _ in range(4)]
    stream.synchronize()
    torch.testing.assert_close(owned, out, rtol=0, atol=0)
    del churn
    freeze_kernel_resolution("FP8 empty caller-owned execution")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("empty launch allocation"))
            assert run(source=a[:0], output=out[:0], workspace=scratch[:0]).numel() == 0
    finally:
        unfreeze_kernel_resolution()
