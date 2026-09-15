"""Block-FP8 checkpoint arithmetic and fixed-capacity serving on physical GPUs."""

import pytest
import torch

from b12x._lib.intrinsics import quant_dequant_mxfp8_torch
from b12x._lib.quant import mxfp8_rows as quant
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import block_fp8_linear as linear
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from tests.gemm.test_sm103_blockscaled import check


@pytest.fixture(autouse=True)
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (12, 0), (12, 1)):
        pytest.skip("requires physical SM103/SM120/SM121")


def reference(source, weight, scale, block):
    x = quant_dequant_mxfp8_torch(source, min_amax=1e-4 if block == 32 else 0.0)
    w = weight.float() * scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:weight.shape[0], :weight.shape[1]]
    return x @ w.T


def assert_quantized(source, storage, floor):
    m, logical_k = source.shape
    physical_k = storage.values.shape[1]
    actual = storage.values[:m].float() * storage.scale_rows.reshape(-1, physical_k // 32)[:m].float().repeat_interleave(32, 1)
    expected = quant_dequant_mxfp8_torch(source, min_amax=floor)
    torch.testing.assert_close(actual[:, :logical_k], expected, rtol=0, atol=0)
    assert torch.all(storage.values[:m, logical_k:].float() == 0)
    rows = torch.arange(m, device=source.device)[:, None]
    groups = torch.arange(physical_k // 32, device=source.device)[None, :]
    mma = storage.scale_mma.view(torch.uint8)
    unpacked = mma[rows % 32, (rows // 32) % 4, rows // 128,
                   groups % 4, groups // 4, 0]
    torch.testing.assert_close(unpacked, storage.scale_rows.view(torch.uint8).reshape(-1, physical_k // 32)[:m], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("floor", [0.0, 1e-4])
@pytest.mark.parametrize("subgroup", [0, 4, 8])
def test_quantizer_lane_contract_with_independent_oracle(dtype, floor, subgroup):
    source = torch.randn(129, 160, device="cuda", dtype=dtype)
    source[0].zero_()
    source[1].fill_(1e-6)
    source[2, :32].fill_(torch.finfo(dtype).max)
    source[3, :32].fill_(-torch.finfo(dtype).max)
    storage = empty_mxfp8_rows_for_dense_gemm(129, 256, device="cuda")
    compiled = quant._get_compiled_mxfp8_rows_quant(256, dtype, subgroup, 128 if subgroup == 8 else 256, "linear", floor)
    compiled(source, storage.values, storage.scale_rows, storage.scale_mma)
    assert_quantized(source, storage, floor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("floor", [0.0, 1e-4])
def test_public_quantizer_counts_and_graph_reuse(dtype, floor, monkeypatch):
    source = torch.randn(129, 160, device="cuda", dtype=dtype)
    storage = empty_mxfp8_rows_for_dense_gemm(129, 256, device="cuda")
    def run(m):
        quant.quantize_mxfp8_rows_cute(source[:m], storage.values, storage.scale_rows,
                                     storage.scale_mma, expected_m=129, physical_k=256, min_amax=floor)
    run(1)
    cached = quant._get_compiled_mxfp8_rows_quant.cache_info()
    with kernel_resolution_guard("MXFP8 live rows reuse one planned callable"):
        for m in (1, 4, 8, 9, 65, 129):
            run(m)
            assert_quantized(source[:m], storage, floor)
        assert quant._get_compiled_mxfp8_rows_quant.cache_info().misses == cached.misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(129)
        tensors = (source, storage.values, storage.scale_rows, storage.scale_mma)
        pointers = [t.data_ptr() for t in tensors]
        for _ in range(2):
            source.normal_().mul_(.25)
            for t in tensors[1:]:
                t.view(torch.uint8).fill_(0xA5)
            graph.replay()
            assert_quantized(source, storage, floor)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("quantizer allocated"))
            run(129)
            graph.replay()
            torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
            assert before[key] == after[key]
        assert pointers == [t.data_ptr() for t in tensors]


@pytest.mark.parametrize("block,n,k", [(128, 256, 384), (32, 136, 160), (32, 256, 576)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_public_plan_counts_bias_mutation_graph_and_stream(block, n, k, dtype, monkeypatch):
    torch.manual_seed(7301)
    source = torch.randn(129, k, device="cuda", dtype=dtype).mul_(.25)
    weight = torch.randn(n, k, device="cuda").mul_(.125).to(torch.float8_e4m3fn)
    scale = torch.randint(123, 129, ((n + block - 1) // block, k // block), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    packed = linear.pack_weight(weight, scale, block_size=(block, block))
    bias = torch.randn(n, device="cuda", dtype=dtype).mul_(.125)
    plan = linear.plan(linear.Caps(device=source.device, max_tokens=129, in_features=k,
                                  out_features=n, source_dtype=dtype, output_dtype=dtype, block_size=(block, block)))
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(129, n, 1, device="cuda", dtype=dtype)
    bindings = [linear.bind(plan, scratch=scratch, source=source[:m], packed_weight=packed,
                            output=output[:m], bias=bias) for m in (1, 4, 8, 9, 65, 129)]
    linear.run(binding=bindings[0])
    cached = quant._get_compiled_mxfp8_rows_quant.cache_info()
    with kernel_resolution_guard("planned block-FP8 fixed capacity"):
        for binding in bindings:
            expected = reference(binding.source, weight, scale, block).to(dtype) + bias
            check(linear.run(binding=binding), expected.float())
        assert quant._get_compiled_mxfp8_rows_quant.cache_info().misses == cached.misses
        binding = bindings[-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            linear.run(binding=binding)
        tensors = (source, scratch, output, packed.weight.values, packed.weight.scale_mma, bias)
        addresses = [t.data_ptr() for t in tensors]
        for _ in range(2):
            source.normal_().mul_(.25)
            bias.mul_(.5)
            output.fill_(float("nan"))
            graph.replay()
            check(output[..., 0], (reference(source, weight, scale, block).to(dtype) + bias).float())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            source.mul_(.5)
        linear.run(binding=binding, stream=stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        check(output[..., 0], (reference(source, weight, scale, block).to(dtype) + bias).float())
        before = torch.cuda.memory_stats()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("bound linear allocated"))
            linear.run(binding=binding)
            graph.replay()
            torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
            assert before[key] == after[key]
        assert addresses == [t.data_ptr() for t in tensors]


@pytest.mark.parametrize("subgroup", [4, 8])
def test_quantizer_source_row_past_int32_elements(subgroup):
    m, k = 65537, 32768
    source = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    source[-1].fill_(1)
    storage = empty_mxfp8_rows_for_dense_gemm(m, k, device="cuda")
    compiled = quant._get_compiled_mxfp8_rows_quant(k, source.dtype, subgroup, 128 if subgroup == 8 else 256, "linear")
    compiled(source, storage.values, storage.scale_rows, storage.scale_mma)
    assert (m - 1) * k == 2**31
    actual = storage.values[-1].float() * storage.scale_rows[0, -1].float().repeat_interleave(32)
    assert torch.all(actual == 1)


@pytest.mark.parametrize("block,k", [(128, 384), (32, 160)])
def test_functional_preparation_compilation_and_graph(block, k):
    n = 256
    source = torch.randn(129, k, device="cuda", dtype=torch.bfloat16).mul_(.25)
    weight = torch.randn(n, k, device="cuda").mul_(.125).to(torch.float8_e4m3fn)
    scale = torch.randint(123, 129, (n // block, k // block), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    packed = linear.pack_weight(weight, scale, block_size=(block, block))
    plan = linear.plan(linear.Caps(device=source.device, max_tokens=129,
        in_features=k, out_features=n, block_size=(block, block), output_mode="functional"))
    from b12x.preparation import require_prepared
    require_prepared(plan, "gemm.block_fp8_linear", source.device)
    def run(x):
        return linear.run(x, packed, plan=plan)
    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    check(compiled(source), reference(source, weight, scale, block))
    with kernel_resolution_guard("functional block-FP8 serving after preparation"):
        for m in (1, 4, 8, 65, 129):
            check(run(source[:m]), reference(source[:m], weight, scale, block))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(source)
        address = output.data_ptr()
        for _ in range(2):
            source.normal_().mul_(.25)
            output.fill_(float("nan"))
            graph.replay()
            check(output, reference(source, weight, scale, block))
        assert output.data_ptr() == address
