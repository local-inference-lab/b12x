"""Independent FP6 arithmetic and fixed-capacity quantization on physical GPUs."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.quantization.mxfp6 import allocate_fp6_linear_workspace
from b12x.quantization.mxfp6 import _rows


def require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (12, 0), (12, 1)):
        pytest.skip("requires physical SM103/SM120/SM121")


def lut(fmt, device):
    mantissa, bias = (2, 3) if fmt == "e3m2" else (3, 1)
    codes = torch.arange(32, device=device)
    exponent, fraction = codes >> mantissa, (codes % (1 << mantissa)).float() / (1 << mantissa)
    return torch.where(exponent == 0, fraction * 2.0 ** (1 - bias), (1 + fraction) * torch.exp2(exponent.float() - bias))


def pack(codes):
    c = codes.long().reshape(*codes.shape[:-1], -1, 4)
    words = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    return torch.stack(tuple((words >> shift) & 255 for shift in (0, 8, 16)), -1).to(torch.uint8).flatten(-2)


def unpack(values):
    v = values.long().reshape(*values.shape[:-1], -1, 3)
    words = v[..., 0] | (v[..., 1] << 8) | (v[..., 2] << 16)
    return torch.stack(tuple((words >> shift) & 63 for shift in (0, 6, 12, 18)), -1).to(torch.uint8).flatten(-2)


def decode(codes, fmt):
    if fmt == "e4m3":
        return codes.contiguous().view(torch.float8_e4m3fn).float()
    return lut(fmt, codes.device)[(codes & 31).long()] * torch.where(codes & 32 != 0, -1., 1.)


def oracle(source, fmt, per_row, weight_scale, fixed_global_scale=None):
    """Torch arithmetic, exponent scaling, and nearest-even FP6 code selection."""
    x = source.float()
    maximum = x.abs().amax(1, keepdim=True) if per_row else x.abs().amax().reshape(1, 1)
    fmt_max = {"e3m2": 28., "e2m3": 7.5, "e4m3": 448.}[fmt]
    gs = (448 * fmt_max / maximum.clamp_min(1e-6).double()).float()
    if fixed_global_scale is not None:
        assert not per_row
        gs = fixed_global_scale.reshape(1, 1)
    inverse = (1 / gs.double()).float().to(torch.bfloat16)
    alpha = (1 / ((torch.ones_like(gs[:1]) if per_row else gs) * weight_scale).double()).float()
    x = (x * gs).to(torch.bfloat16).float() if per_row else x
    quant_gs = torch.ones_like(gs) if per_row else gs
    blocks = x.reshape(x.shape[0], -1, 32)
    ratio = ((blocks.abs().amax(-1) * quant_gs).double() / fmt_max).float()
    fraction, power = torch.frexp(ratio)
    exponent = (power - (fraction == .5).int()).clamp(-127, 127)
    sf = torch.where(ratio == 0, 0, exponent + 127).to(torch.uint8)
    multiplier = torch.where(sf == 0, 0., quant_gs * torch.exp2(127 - sf.float()))
    normalized = blocks * multiplier[..., None]
    if fmt == "e4m3":
        codes = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        positive = lut(fmt, source.device)
        distance = (normalized.abs()[..., None] - positive).abs()
        minimum = distance.amin(-1, keepdim=True)
        candidates = torch.arange(32, device=source.device)
        # A zero-distance even candidate wins an exact midpoint tie.
        priority = candidates + (candidates & 1) * 32
        selected = torch.where(distance == minimum, priority, 1000).argmin(-1)
        codes = (selected | (torch.signbit(normalized).long() << 5)).to(torch.uint8)
        codes = torch.where(multiplier[..., None] == 0, 0, codes)
    return codes.reshape_as(source), sf, gs.flatten(), inverse.flatten(), alpha.flatten()


def assert_quantized(workspace, source, weight_scale):
    codes, sf, gs, inverse, alpha = oracle(source, workspace.act_fmt, workspace.per_row, weight_scale)
    m, k = source.shape
    torch.testing.assert_close(workspace.values[:m], pack(codes) if workspace.packed else codes, rtol=0, atol=0)
    torch.testing.assert_close(workspace.global_scales[:m] if workspace.per_row else workspace.global_scales, gs, rtol=0, atol=0)
    if workspace.per_row:
        torch.testing.assert_close(workspace.inverse_scales[:m], inverse, rtol=0, atol=0)
    torch.testing.assert_close(workspace.alpha, alpha, rtol=0, atol=0)
    row = torch.arange((m + 127) // 128 * 128, device=source.device)[:, None]
    group = torch.arange(k // 32, device=source.device)[None, :]
    offsets = (row // 128) * (k // 128) * 512 + (group // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + group % 4
    actual = workspace.scale_storage[offsets]
    torch.testing.assert_close(actual[:m], sf, rtol=0, atol=0)
    assert not torch.count_nonzero(actual[m:])


@pytest.mark.parametrize("fmt,packed", [(f, p) for f in ("e2m3", "e3m2") for p in (False, True)] + [("e4m3", False)])
@pytest.mark.parametrize("per_row", [False, True])
def test_quantization_oracle_counts_graph_and_allocations(fmt, packed, per_row, monkeypatch):
    require_gpu()
    workspace = allocate_fp6_linear_workspace(129, 384, act_fmt=fmt, per_row=per_row, packed=packed)
    source = torch.randn(129, 384, device="cuda", dtype=torch.bfloat16)
    source[0].zero_()
    source[1].fill_(1e-6)
    source[2, -1] = -2.625
    source[3, :32] = torch.finfo(torch.bfloat16).max
    weight_scale = torch.tensor([.75], device="cuda")
    workspace.quantize(source[:1], weight_scale)
    cache = (_rows.compile_scales.cache_info().misses, _rows.compile_quantizer.cache_info().misses)
    with kernel_resolution_guard("FP6 rows share a capacity-independent callable"):
        for m in (1, 3, 8, 17, 129):
            workspace.scale_storage.fill_(0xA5)
            workspace.quantize(source[:m], weight_scale)
            assert_quantized(workspace, source[:m], weight_scale)
        assert cache == (_rows.compile_scales.cache_info().misses, _rows.compile_quantizer.cache_info().misses)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.quantize(source, weight_scale)
        buffers = (workspace.values, workspace.scale_storage, workspace.global_scales, workspace.inverse_scales, workspace.alpha)
        pointers = [t.data_ptr() for t in buffers]
        for factor in (.25, 1.5):
            source.normal_().mul_(factor)
            weight_scale.fill_(factor)
            for tensor in buffers:
                tensor.view(torch.uint8).fill_(0xA5)
            graph.replay()
            assert_quantized(workspace, source, weight_scale)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace quantization allocated"))
            workspace.quantize(source, weight_scale)
            graph.replay()
            torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
        assert pointers == [t.data_ptr() for t in buffers]


@pytest.mark.parametrize("fault", ["values", "scale_storage", "global_scales", "inverse_scales", "alpha", "overlap", "source_overlap", "capacity", "alignment"])
def test_workspace_rejects_invalid_storage_before_launch(fault, monkeypatch):
    require_gpu()
    workspace = allocate_fp6_linear_workspace(8, 128, packed=False)
    source = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    weight_scale = torch.ones(1, device="cuda")
    if fault in ("values", "scale_storage", "global_scales", "inverse_scales", "alpha"):
        workspace = replace(workspace, **{fault: getattr(workspace, fault).flatten()[:-1]})
    elif fault == "overlap":
        workspace = replace(workspace, alpha=workspace.global_scales[:1])
    elif fault == "source_overlap":
        workspace = replace(workspace, values=source.view(torch.uint8).flatten()[:1024].view(8, 128))
    elif fault == "capacity":
        source = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    else:
        source = torch.empty(1025, device="cuda", dtype=torch.bfloat16)[1:].view(8, 128)
    monkeypatch.setattr(_rows, "compile_scales", lambda *a: pytest.fail("invalid workspace reached compilation"))
    with pytest.raises(ValueError):
        workspace.quantize(source, weight_scale)


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2", "e4m3"])
def test_public_linear_workspace_graph(fmt, monkeypatch):
    require_gpu()
    from b12x.quantization.mxfp6 import dense_fp6_linear, quantize_dense_weight_to_fp6
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw
    monkeypatch.setattr(fdw, "_DENSE_PER_ROW_GS", True)
    weight = replace(quantize_dense_weight_to_fp6(torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)), act_fmt=fmt)
    workspace = allocate_fp6_linear_workspace(129, 384, act_fmt=fmt)
    source = torch.randn(129, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(129, 256, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(256, device="cuda")[:, None]
    blocks = torch.arange(12, device="cuda")[None, :]
    offsets = (rows // 128) * 1536 + (blocks // 4) * 512 + (rows % 32) * 16 + ((rows // 32) % 4) * 4 + blocks % 4
    weight_values = decode(unpack(weight.packed.reshape(256, -1)), weight.fmt)
    weight_values *= torch.exp2(weight.scale_storage.flatten()[offsets].float() - 127).repeat_interleave(32, -1)
    def reference(m):
        codes, sf, _, inverse, alpha = oracle(source[:m], fmt, True, weight.global_scale)
        x = decode(codes, fmt) * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)
        return ((x @ weight_values.T * alpha).to(torch.bfloat16).float() * inverse[:, None].float()).to(torch.bfloat16).float()
    def run(m):
        return dense_fp6_linear(source[:m], weight, out=out[:m], workspace=workspace, expected_m=129)
    run(1)
    with kernel_resolution_guard("FP6 public linear reuses planned workspace"):
        for m in (1, 3, 8, 17, 129):
            result = run(m)
            assert_quantized(workspace, source[:m], weight.global_scale)
            assert result.data_ptr() == out.data_ptr()
            from tests.gemm.test_sm103_blockscaled import check
            check(result, reference(m))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(129)
        source.normal_()
        graph.replay()
        captured = out.clone()
        check(out, reference(129))
        run(129)
        torch.testing.assert_close(out, captured, rtol=0, atol=0)


def test_quantizer_offsets_past_int32_elements_and_bytes():
    require_gpu()
    # Both source element offsets and packed output byte offsets cross 2^31.
    m, k = 65537, 65536
    if torch.cuda.mem_get_info()[0] < 14 * 1024**3:
        pytest.skip("requires 14 GiB free for the Int64 address boundary")
    workspace = allocate_fp6_linear_workspace(m, k, act_fmt="e3m2", packed=True)
    source = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    source[-1].normal_()
    weight_scale = torch.ones(1, device="cuda")
    workspace.quantize(source, weight_scale)
    codes, sf, gs, inverse, _ = oracle(source[-1:], "e3m2", True, weight_scale)
    torch.testing.assert_close(workspace.values[-1:], pack(codes), atol=0, rtol=0)
    torch.testing.assert_close(workspace.global_scales[-1:], gs, atol=0, rtol=0)
    torch.testing.assert_close(workspace.inverse_scales[-1:], inverse, atol=0, rtol=0)
    group = torch.arange(k // 32, device="cuda")
    offsets = ((m - 1) // 128) * (k // 128) * 512 + (group // 4) * 512 + group % 4
    torch.testing.assert_close(workspace.scale_storage[offsets], sf[0], atol=0, rtol=0)


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_weight_preparation_and_subnormal_scale_boundaries(fmt):
    require_gpu()
    from b12x.quantization.mxfp6._linear_workspace import quantize_fp6_weight
    from b12x.quantization.mxfp6.fp6_dense_weights import _quantize_matrix_fp6
    source = torch.randn(128, 384, device="cuda", dtype=torch.bfloat16)
    fmt_max = {"e2m3": 7.5, "e3m2": 28.}[fmt]
    for index, factor in enumerate((0., .5, 1., 1.0078125, 1.5, 2., 2.015625)):
        source[index].fill_(fmt_max * 2.**-127 * factor)
    global_scale = torch.ones(1, device="cuda")
    codes, sf, *_ = oracle(source, fmt, False, global_scale, fixed_global_scale=global_scale)
    values, scales = quantize_fp6_weight(source, fmt, global_scale)
    torch.testing.assert_close(values, pack(codes), atol=0, rtol=0)
    from b12x._lib.intrinsics import swizzle_block_scale
    torch.testing.assert_close(scales, swizzle_block_scale(sf[None]).flatten(), atol=0, rtol=0)
    legacy_values, legacy_scales = _quantize_matrix_fp6(source, fmt, global_scale)
    torch.testing.assert_close(values, legacy_values.reshape_as(values), atol=0, rtol=0)
    torch.testing.assert_close(scales, legacy_scales.flatten(), atol=0, rtol=0)
