from __future__ import annotations

import pytest
import torch

from b12x.quantization.mxfp6 import (
    FP6DenseWeight,
    dense_fp6_linear,
    load_fp6_dense_weight,
    quantize_dense_weight_to_fp6,
    save_fp6_dense_weight,
)
from tests._reference.helpers import require_b12x

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP6 GPU tests"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


@pytest.mark.parametrize("source_format", ["mxfp6_e3m2", "mxfp6_e2m3"])
def test_dense_fp6_weight_pipeline_roundtrip_and_equivalence(
    source_format: str, tmp_path
) -> None:
    require_b12x()
    torch.manual_seed(3)
    m, n, k = 128, 256, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2

    qw = quantize_dense_weight_to_fp6(weight, source_format=source_format)
    assert qw.packed.shape == (n, 3 * k // 4)
    assert qw.out_features == n and qw.in_features == k

    y = dense_fp6_linear(x, qw)
    assert y.shape == (m, n)

    ref = x.float() @ weight.float().T
    assert _cos(y, ref) > 0.95

    # save/load round-trip: identical tensors and identical kernel output.
    path = str(tmp_path / "dense_fp6.safetensors")
    save_fp6_dense_weight(qw, path)
    loaded = load_fp6_dense_weight(path, device="cuda")
    assert isinstance(loaded, FP6DenseWeight)
    torch.testing.assert_close(loaded.packed, qw.packed, rtol=0, atol=0)
    torch.testing.assert_close(loaded.scale_storage, qw.scale_storage, rtol=0, atol=0)

    y_loaded = dense_fp6_linear(x, loaded)
    torch.testing.assert_close(y_loaded, y, rtol=0, atol=0)


def test_dense_fp6_linear_pads_non_tile_token_count() -> None:
    require_b12x()
    torch.manual_seed(5)
    m, n, k = 70, 256, 128  # m not a multiple of 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2

    qw = quantize_dense_weight_to_fp6(weight, source_format="mxfp6_e3m2")
    y = dense_fp6_linear(x, qw)
    assert y.shape == (m, n)

    ref = x.float() @ weight.float().T
    assert _cos(y, ref) > 0.95


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('per_row,epilogue', [(True, True), (False, True), (True, False)])
def test_caller_out_direct_and_correction_fallback(packed, per_row, epilogue, monkeypatch):
    require_b12x()
    from b12x.quantization.mxfp6 import fp6_dense_weights as fw
    monkeypatch.setattr(fw, '_DENSE_PER_ROW_GS', per_row)
    monkeypatch.setattr(fw, '_ROW_SCALE_EPILOGUE', epilogue)
    m, n, k = 128, 256, 512
    torch.manual_seed(704)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .2
    original_x = x.clone()
    w = quantize_dense_weight_to_fp6(torch.randn(n, k, device='cuda', dtype=x.dtype) * .2)
    weight = w.packed if packed else w.expanded_weight()
    seen = []
    original = fw.dense_gemm
    def spy(*args, **kwargs):
        import inspect
        state = inspect.currentframe().f_back.f_locals
        assert state['fresh_internal'] == per_row
        seen.append(kwargs['out'].data_ptr())
        return original(*args, **kwargs)
    monkeypatch.setattr(fw, 'dense_gemm', spy)
    def invoke(out):
        return fw.dense_fp6_linear_expanded(x, weight, w.scale_storage, w.global_scale,
                                          w.fmt, n, k, out=out, act_fmt=w.act_fmt)
    base = torch.empty(m * n + 8, device='cuda', dtype=x.dtype)
    out = base[8:].view(m, n)
    for factor in (1., 0., -.5):
        x.copy_(original_x * factor)
        expected = invoke(None)
        old_out = torch.empty_like(out)
        old_version = old_out._version
        with monkeypatch.context() as old_path:
            old_path.setattr(torch._C, '_overlaps', lambda a, b: True)
            assert invoke(old_out) is old_out
        assert seen[-1] != old_out.data_ptr() and old_out._version == old_version + 1
        torch.testing.assert_close(old_out, expected, rtol=0, atol=0)
        out.fill_(float('nan'))
        version = base._version
        assert invoke(out) is out
        assert (seen[-1] == out.data_ptr()) == (not per_row or epilogue)
        assert out._version == base._version == version + 1
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['noncontiguous', 'fp32', 'unaligned', 'broadcast',
                                  'alias_x', 'alias_weight', 'alias_scale', 'alias_global',
                                  'grad_out', 'grad_input', 'negative', 'shape_error'])
def test_caller_out_fallback_copy_semantics(kind, monkeypatch):
    require_b12x()
    from b12x.quantization.mxfp6 import fp6_dense_weights as fw
    m = n = k = 128
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .2
    w = quantize_dense_weight_to_fp6(torch.randn(n, k, device='cuda', dtype=x.dtype) * .2)
    weight, scale, gs = w.expanded_weight(), w.scale_storage, w.global_scale
    out = torch.empty(m, n, device='cuda', dtype=x.dtype)
    if kind == 'noncontiguous': out = out.t()
    if kind == 'fp32': out = out.float()
    if kind == 'unaligned': out = torch.empty(m * n + 1, device='cuda', dtype=x.dtype)[1:].view(m, n)
    if kind == 'broadcast': out = torch.empty(2, m, n, device='cuda', dtype=x.dtype)
    if kind == 'alias_x': out = x
    if kind in ('alias_weight', 'alias_scale', 'alias_global'):
        src = dict(alias_weight=weight, alias_scale=scale, alias_global=gs)[kind]
        storage = torch.empty(max(src.numel() * src.element_size(), out.numel() * 2), device='cuda', dtype=torch.uint8)
        alias = storage[:src.numel() * src.element_size()].view(src.dtype).view(src.shape)
        alias.copy_(src)
        out = storage[:m * n * 2].view(torch.bfloat16).view(m, n)
        if kind == 'alias_weight': weight = alias
        if kind == 'alias_scale': scale = alias
        if kind == 'alias_global': gs = alias
    if kind == 'grad_out': out.requires_grad_(True)
    if kind == 'grad_input': x.requires_grad_(True)
    if kind == 'negative': out = torch._neg_view(out)
    if kind == 'shape_error': out = out[:2]
    def invoke(target):
        return fw.dense_fp6_linear_expanded(x, weight, scale, gs, w.fmt, n, k,
                                          out=target, act_fmt=w.act_fmt)
    if kind == 'grad_input':
        out.fill_(13)
        version = out._version
        for target in (None, out):
            with pytest.raises(BufferError, match="Can't export tensors that require gradient"):
                invoke(target)
        with monkeypatch.context() as old_path:
            old_path.setattr(torch._C, '_overlaps', lambda a, b: True)
            with pytest.raises(BufferError, match="Can't export tensors that require gradient"):
                invoke(out)
        assert out._version == version
        torch.testing.assert_close(out, torch.full_like(out, 13), rtol=0, atol=0)
        return
    expected = invoke(None)
    seen = []
    original = fw.dense_gemm
    def spy(*args, **kwargs):
        seen.append(kwargs['out'].data_ptr())
        return original(*args, **kwargs)
    monkeypatch.setattr(fw, 'dense_gemm', spy)
    if kind in ('grad_out', 'shape_error'):
        with pytest.raises(RuntimeError): invoke(out)
    else:
        assert invoke(out) is out
        torch.testing.assert_close(out, expected.expand_as(out).to(out.dtype), rtol=0, atol=0)
    assert seen and seen[-1] != out.data_ptr()


@pytest.mark.parametrize('force_old', [False, True])
def test_caller_out_inference_and_capture_versions(force_old, monkeypatch):
    require_b12x()
    if force_old:
        monkeypatch.setattr(torch._C, '_overlaps', lambda a, b: True)
    x = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16) * .2
    original_x = x.clone()
    w = quantize_dense_weight_to_fp6(torch.randn(256, 512, device='cuda', dtype=x.dtype) * .2)
    w.expanded_weight()
    with torch.inference_mode():
        inference_out = torch.empty(128, 256, device='cuda', dtype=x.dtype)
        assert dense_fp6_linear(x, w, out=inference_out) is inference_out
        torch.testing.assert_close(inference_out, dense_fp6_linear(x, w), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='Inference'):
        dense_fp6_linear(x, w, out=inference_out)
    out = torch.empty(128, 256, device='cuda', dtype=x.dtype)
    dense_fp6_linear(x, w, out=out)
    version = out._version
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dense_fp6_linear(x, w, out=out)
    assert result is out and out._version == version + 1
    captured_version = out._version
    for factor in (2., 0., -.5):
        x.copy_(original_x * factor)
        expected = dense_fp6_linear(x, w)
        graph.replay()
        torch.cuda.synchronize()
        assert out._version == captured_version
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph.reset()
