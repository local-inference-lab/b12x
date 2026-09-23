"""WO quantizer oracles and deferred native SM103 projection qualification."""

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.preparation import FrozenMapping
from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import (
    empty_dense_gemm_mnl_view, empty_mxfp8_rows_for_dense_gemm,
    quantize_wo_projection_weights_mxfp8_torch,
)
from b12x.gemm.wo_projection import _quant_cute as quant
from tests.gemm.test_sm103_blockscaled import check, require_native


@pytest.fixture(autouse=True)
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (12, 0), (12, 1)):
        pytest.skip("requires physical SM103/SM120/SM121")


def reference_quant(grouped):
    """Independent FP32-to-E4M3 rounding with per-32 ceil-power-of-two scales."""
    groups, rows, width = grouped.shape
    chunks = grouped.float().reshape(groups, rows, width // 32, 32)
    amax = chunks.abs().amax(-1)
    exponent = torch.where(amax == 0, 0, torch.ceil(torch.log2(amax / 448))).clamp(-127, 127)
    scales = torch.exp2(exponent)
    values = (chunks / scales[..., None]).to(torch.float8_e4m3fn).reshape(groups, rows, width)
    return values, (exponent + 127).to(torch.uint8)


def rotate(source, positions, cache, nope, rope):
    values = source.float().clone()
    pairs = values[..., nope:].reshape(*source.shape[:-1], rope // 2, 2)
    cs = cache[positions.long()].float().unsqueeze(1)
    cos, sin = cs[..., :rope // 2], cs[..., rope // 2:]
    first = pairs[..., 0].clone()
    second = pairs[..., 1].clone()
    pairs[..., 0] = first * cos + second * sin
    pairs[..., 1] = second * cos - first * sin
    return values


def assert_rows(storage, grouped):
    expected_values, expected_rows = reference_quant(grouped)
    g, m, k = grouped.shape
    actual_values = storage.values.permute(2, 0, 1) if storage.values.ndim == 3 else storage.values.unsqueeze(0)
    torch.testing.assert_close(actual_values.view(torch.uint8), expected_values.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(storage.scale_rows.view(torch.uint8), expected_rows, atol=0, rtol=0)
    r = torch.arange((m + 127) // 128 * 128, device=grouped.device)[:, None]
    c = torch.arange(k // 32, device=grouped.device)[None, :]
    for group in range(g):
        mma_rows = storage.scale_mma.view(torch.uint8)[r % 32, r // 32 % 4, r // 128, c % 4, c // 4, group]
        torch.testing.assert_close(mma_rows[:m], expected_rows[group], atol=0, rtol=0)
        assert torch.all(mma_rows[m:] == 127)


def decode(storage):
    values = storage.values.permute(2, 0, 1) if storage.values.ndim == 3 else storage.values.unsqueeze(0)
    return values.float() * torch.exp2(storage.scale_rows.view(torch.uint8).float() - 127).repeat_interleave(32, -1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode,positions_dtype,cache_dtype", [
    ("grouped", None, None), ("group_major", None, None),
    ("rope", torch.int32, torch.bfloat16), ("rope", torch.int64, torch.bfloat16),
    ("rope", torch.int32, torch.float32), ("rope", torch.int64, torch.float32),
])
def test_quantization_bytes_padding_frozen_counts_and_graph(dtype, mode, positions_dtype, cache_dtype, monkeypatch):
    torch.manual_seed(6137)
    groups, span = 3, 128
    items = []
    for m in (1, 3, 8, 9, 16, 127, 128, 129):
        if mode == "group_major":
            source = empty_dense_gemm_mnl_view(m, span, groups, device="cuda", dtype=dtype)
            source.copy_(torch.randn_like(source) * .25)
            storage = empty_mxfp8_rows_for_dense_gemm(m, span * groups, device="cuda")
        else:
            padded = torch.full((m, groups + m % 2, span), float("nan"), device="cuda", dtype=dtype)
            source = padded[:, :groups]
            source.normal_().mul_(.25)
            storage = empty_mxfp8_rows_for_dense_gemm(m, span, num_groups=groups, device="cuda")
        source[0].zero_()
        if m > 1:
            source[1].fill_(-0.0)
        positions = torch.arange(-1, m, device="cuda", dtype=positions_dtype or torch.int64)[1:]
        # BF16-valued cache entries make these finite products/sums exact in
        # FP32 for both activation dtypes, including the FP32 cache overload.
        cache = torch.randn(m, 32, device="cuda").to(torch.bfloat16).to(cache_dtype or torch.bfloat16)
        items.append((source, storage, positions, cache))

    def launch(item, stream=None):
        source, storage, positions, cache = item
        m = source.shape[0]
        if mode == "group_major":
            quant.quantize_wo_group_major_rows_cute(source, storage.values, storage.scale_rows,
                storage.scale_mma, m=m, groups=groups, rank=span, stream=stream)
        else:
            quant.quantize_wo_grouped_rows_cute(source, storage.values, storage.scale_rows,
                storage.scale_mma, m=m, groups=groups, group_width=span,
                positions=positions if mode == "rope" else None,
                cos_sin_cache=cache if mode == "rope" else None,
                head_dim=128, nope_dim=96, rope_dim=32, stream=stream)

    def verify(item):
        source, storage, positions, cache = item
        if mode == "group_major":
            grouped = source.permute(0, 2, 1).reshape(1, source.shape[0], -1)
        else:
            x = rotate(source, positions, cache, 96, 32) if mode == "rope" else source
            grouped = x.permute(1, 0, 2)
        assert_rows(storage, grouped)

    launch(items[0])
    misses = quant._get_compiled_wo_quant.cache_info().misses
    with kernel_resolution_guard("WO live counts reuse warmed quantization"):
        for item in items:
            for tensor in (item[1].values, item[1].scale_rows, item[1].scale_mma):
                tensor.view(torch.uint8).fill_(0xA5)
            launch(item)
            verify(item)
        assert quant._get_compiled_wo_quant.cache_info().misses == misses
        item = items[-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(item)
        addresses = [t.data_ptr() for t in (item[0], item[1].values, item[1].scale_rows, item[1].scale_mma)]
        for _ in range(2):
            item[0].normal_().mul_(.25)
            for tensor in (item[1].values, item[1].scale_rows, item[1].scale_mma):
                tensor.view(torch.uint8).fill_(0xFF)
            graph.replay()
            verify(item)
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        launch(item, stream=other)
        torch.cuda.current_stream().wait_stream(other)
        verify(item)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("quantizer allocated"))
            launch(item)
            graph.replay()
            torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
            assert before[key] == after[key]
        assert addresses == [t.data_ptr() for t in (item[0], item[1].values, item[1].scale_rows, item[1].scale_mma)]


def test_inverse_rope_cosine_pool_offset_exceeds_int32():
    # The position itself fits Int32; multiplying it by the cache stride does
    # not. Only the referenced tail rows of this 4 GiB cache are initialized.
    rope, m = 32, 3
    high = 2**31 // rope + 7
    cache = torch.empty(high + m, rope, device="cuda", dtype=torch.bfloat16)
    tail = torch.randn(m, rope, device="cuda", dtype=torch.bfloat16)
    cache[high:].copy_(tail)
    source = torch.randn(m, 2, 128, device="cuda", dtype=torch.bfloat16)
    storage = empty_mxfp8_rows_for_dense_gemm(m, 128, num_groups=2, device="cuda")
    for positions_dtype in (torch.int32, torch.int64):
        positions = torch.arange(high, high + m, device="cuda", dtype=positions_dtype)
        quant.quantize_wo_grouped_rows_cute(source, storage.values, storage.scale_rows,
            storage.scale_mma, m=m, groups=2, group_width=128, positions=positions,
            cos_sin_cache=cache, head_dim=128, nope_dim=96, rope_dim=32)
        assert_rows(storage, rotate(source, positions, cache, 96, 32).permute(1, 0, 2))


def test_grouped_input_row_stride_exceeds_int32():
    row_stride = 2**31 + 8
    pool = torch.empty(row_stride + 256, device="cuda", dtype=torch.bfloat16)
    source = pool.as_strided((2, 2, 128), (row_stride, 128, 1))
    source.copy_(torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16))
    positions = torch.arange(2, device="cuda")
    cache = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    storage = empty_mxfp8_rows_for_dense_gemm(2, 128, num_groups=2, device="cuda")
    for inverse in (False, True):
        quant.quantize_wo_grouped_rows_cute(source, storage.values, storage.scale_rows,
            storage.scale_mma, m=2, groups=2, group_width=128,
            positions=positions if inverse else None, cos_sin_cache=cache if inverse else None,
            head_dim=128, nope_dim=96, rope_dim=32)
        expected = rotate(source, positions, cache, 96, 32) if inverse else source
        assert_rows(storage, expected.permute(1, 0, 2))


@pytest.mark.parametrize("groups,width,rank,hidden", [(1, 128, 128, 136), (3, 512, 512, 2560), (4, 4096, 1024, 4096)])
@pytest.mark.parametrize("inverse", [False, True])
def test_native_wo_planned_stages_frozen_counts_poison_and_graph(groups, width, rank, hidden, inverse, monkeypatch):
    require_native()
    torch.manual_seed(8053)
    capacity = 129
    head, rope = (128, 32) if width == 128 else (512, 64)
    padded = torch.full((capacity, groups * (width // head) + 1, head), float("nan"), device="cuda", dtype=torch.bfloat16)
    source = padded[:, :-1]
    source.normal_().mul_(.25)
    positions = torch.arange(capacity, device="cuda")
    cache = torch.randn(capacity, rope, device="cuda", dtype=torch.bfloat16)
    wa = torch.randn(groups, rank, width, device="cuda", dtype=torch.bfloat16).mul_(.125)
    wb = torch.randn(hidden, groups * rank, device="cuda", dtype=torch.bfloat16).mul_(.125)
    weights = quantize_wo_projection_weights_mxfp8_torch(wa, wb)
    plan = wo.plan(wo.Caps(device=source.device, max_tokens=capacity, groups=groups,
                           group_width=width, rank=rank, hidden=hidden),
                   invocation=FrozenMapping(dict(operation="inv_rope", heads_per_group=width // head,
                       nope_dim=head - rope, rope_dim=rope, dynamic_tokens=True) if inverse
                       else {"operation": "plain", "dynamic_tokens": True}))
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(capacity, hidden, device="cuda", dtype=torch.bfloat16)
    bindings = []
    for m in (1, 3, 8, 9, 16, 127, 128, 129):
        if inverse:
            binding = wo.bind_inv_rope(plan, scratch=scratch, o=source[:m], positions=positions[:m],
                cos_sin_cache=cache, weights=weights, heads_per_group=width // head,
                nope_dim=head - rope, rope_dim=rope, out=output[:m])
        else:
            binding = wo.bind(plan, scratch=scratch, source_tgd=source[:m].reshape(m, groups, width), weights=weights, out=output[:m])
        bindings.append(binding)
    fn = wo.run_inv_rope if inverse else wo.run
    fn(plan=plan, binding=bindings[0])
    frozen_pointers = tuple(t.data_ptr() for t in (source, scratch, weights.wo_a.values, weights.wo_b.values))
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with kernel_resolution_guard("native WO retains geometry and capacity across live rows"):
        try:
            def verify(binding):
                m = binding.output.shape[0]
                x = rotate(source[:m], positions[:m], cache, head - rope, rope) if inverse else source[:m]
                assert_rows(binding.x_q, x.reshape(m, groups, width).permute(1, 0, 2))
                tmp_ref = torch.bmm(decode(binding.x_q), decode(weights.wo_a).transpose(1, 2))
                check(binding.tmp.permute(2, 0, 1), tmp_ref)
                assert_rows(binding.tmp_q, binding.tmp.permute(0, 2, 1).reshape(1, m, groups * rank))
                expected = decode(binding.tmp_q)[0] @ decode(weights.wo_b)[0].T
                check(binding.output[:, :, 0], expected)
            for binding in bindings:
                scratch.fill_(0xFF)
                output.fill_(float("nan"))
                out = fn(plan=plan, binding=binding)
                assert out.data_ptr() == output.data_ptr()
                verify(binding)
            binding = bindings[-1]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(plan=plan, binding=binding)
            for _ in range(2):
                source.normal_().mul_(.25)
                eager = fn(plan=plan, binding=binding).clone()
                scratch.fill_(0xA5)
                output.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(binding.output[:, :, 0], eager, atol=0, rtol=0)
                verify(binding)
            other = torch.cuda.Stream()
            other.wait_stream(torch.cuda.current_stream())
            fn(plan=plan, binding=binding, stream=other)
            torch.cuda.current_stream().wait_stream(other)
            verify(binding)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            with monkeypatch.context() as patch:
                patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("native WO allocated"))
                fn(plan=plan, binding=binding)
                graph.replay()
                torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert frozen_pointers == tuple(t.data_ptr() for t in (source, scratch, weights.wo_a.values, weights.wo_b.values))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("position", [-1, 2, 2**31])
def test_invalid_cosine_position_raises_device_error(position):
    # Invalid positions poison their own CUDA context, so isolate each launch.
    import subprocess
    import sys
    code = f'''
import torch
from b12x.gemm.wo_projection._quant_cute import quantize_wo_grouped_rows_cute
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
source = torch.ones(1, 1, 128, device="cuda", dtype=torch.bfloat16)
positions = torch.zeros(1, device="cuda", dtype=torch.int64)
cache = torch.ones(2, 32, device="cuda", dtype=torch.bfloat16)
out = empty_mxfp8_rows_for_dense_gemm(1, 128, device="cuda")
quantize_wo_grouped_rows_cute(source, out.values, out.scale_rows, out.scale_mma,
    m=1, groups=1, group_width=128, positions=positions, cos_sin_cache=cache,
    head_dim=128, nope_dim=96, rope_dim=32)
torch.cuda.synchronize()
positions.fill_({position})
try:
    quantize_wo_grouped_rows_cute(source, out.values, out.scale_rows, out.scale_mma,
        m=1, groups=1, group_width=128, positions=positions, cos_sin_cache=cache,
        head_dim=128, nope_dim=96, rope_dim=32)
    torch.cuda.synchronize()
except RuntimeError as error:
    if "CUDA" not in str(error) and "cuda" not in str(error):
        raise
else:
    raise AssertionError("invalid WO cosine position did not raise a device error")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
