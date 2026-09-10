from __future__ import annotations

import pytest
import torch

import b12x._lib.dense_gemm as dense_module
import b12x._lib.quant.mxfp8_rows as quant_module
from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import block_fp8_linear as bfl
from tests._reference.helpers import require_b12x
from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight,
    _reference_from_quantized_operands,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "capacity,expected_m,live_counts",
    [
        (1, None, (1,)),
        (8, None, (8, 1, 7, 2)),
        (129, None, (129, 9, 128, 8, 127, 1)),
        (2048, None, (2048, 129, 9, 1, 1024)),
        (129, 8, (8, 1, 2, 7)),
        (2048, 129, (129, 9, 128, 1)),
    ],
)
def test_block_fp8_linear_scratch_reuse_across_row_tiles(
    dtype: torch.dtype, capacity: int, expected_m: int | None,
    live_counts: tuple[int, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    resolved = []
    quantized = []

    def track(resolve, calls):
        def wrapped(*args, **kwargs):
            compiled = resolve(*args, **kwargs)
            calls.append(compiled)
            return compiled
        return wrapped

    for name in ("_get_compiled_dense_gemm", "_get_compiled_dense_gemm_fused_quant_a"):
        monkeypatch.setattr(dense_module, name, track(getattr(dense_module, name), resolved))
    monkeypatch.setattr(
        quant_module, "_get_compiled_mxfp8_rows_quant",
        track(quant_module._get_compiled_mxfp8_rows_quant, quantized),
    )
    in_features, out_features = 256, 384
    source = torch.randn((capacity, in_features), device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)
    plan = bfl.plan(bfl.Caps(device=source.device, max_tokens=capacity,
                           in_features=in_features, out_features=out_features,
                           output_dtype=dtype))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty((capacity, out_features, 1), dtype=dtype, device=source.device)
    pointers = (source.data_ptr(), scratch.data_ptr(), output.data_ptr())

    def bind(tokens: int):
        return bfl.bind(plan, scratch=scratch, source=source[:tokens],
                        packed_weight=packed, output=output[:tokens],
                        expected_m=expected_m)

    bound = capacity if expected_m is None else expected_m
    # The reference uses standalone quantization even for fused decode plans.
    _reference_from_quantized_operands(source[:1], weight, scale)
    quantized.clear()
    bfl.run(binding=bind(bound))
    assert len(resolved) == 1
    warmed = resolved[0]
    warmed_quant = tuple(quantized)

    def run(binding):
        start = len(quantized)
        result = bfl.run(binding=binding)
        assert tuple(quantized[start:]) == warmed_quant
        return result

    freeze_kernel_resolution("block FP8 scratch reuse across row-tile boundaries")
    try:
        for tokens in live_counts:
            binding = bind(tokens)
            run(binding)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(binding)
            source.normal_().mul_(0.25)
            expected = _reference_from_quantized_operands(source[:tokens], weight, scale)
            binding.x_q.scale_rows.view(torch.uint8).fill_(127)
            binding.x_q.scale_mma.view(torch.uint8).fill_(127)
            initialized = run(binding).clone()
            scratch.fill_(255)
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert pointers == (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
            actual = output[:tokens, :, 0]
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, initialized, rtol=0, atol=0)
            assert all(compiled is warmed for compiled in resolved)
            # Distinct accumulation orders may round to adjacent output values.
            lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
            upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
            assert torch.all((actual >= lower) & (actual <= upper))
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("expected_m", (1, 8, 129, 2048))
def test_block_fp8_linear_fixed_expected_m_functional_capture(
    dtype: torch.dtype, expected_m: int,
) -> None:
    """The functional call used by vLLM retains its fixed capture-row bound."""
    require_b12x()
    source = torch.randn((expected_m, 256), device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(384, 256)
    packed = bfl.pack_weight(weight, scale)

    def run():
        return bfl.run(source, packed, expected_m=expected_m)

    _reference_from_quantized_operands(source, weight, scale)
    run()
    freeze_kernel_resolution("block FP8 functional capture with fixed expected_m")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        pointers = (source.data_ptr(), actual.data_ptr())
        for _ in range(3):
            source.normal_().mul_(0.25)
            expected = _reference_from_quantized_operands(source, weight, scale)
            actual.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert pointers == (source.data_ptr(), actual.data_ptr())
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
            upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
            assert torch.all((actual >= lower) & (actual <= upper))
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,expected_m", ((8, None), (8, 8), (129, None), (129, 129)))
def test_block_fp8_linear_prewarm_covers_bound_and_functional_capture(
    dtype: torch.dtype, tokens: int, expected_m: int | None,
) -> None:
    require_b12x()
    source = torch.randn((tokens, 256), dtype=dtype, device="cuda").mul_(0.25)
    weight, scale = _make_block_fp8_weight(384, 256)
    packed = bfl.pack_weight(weight, scale)
    expected = _reference_from_quantized_operands(source, weight, scale)
    plan = bfl.plan(bfl.Caps(device=source.device, max_tokens=tokens,
                           in_features=256, out_features=384, output_dtype=dtype))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty((tokens, 384, 1), dtype=dtype, device=source.device)
    binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed,
                       output=output, expected_m=expected_m)
    bfl.prewarm(packed, (tokens,), output_dtype=dtype, expected_m=expected_m)
    freeze_kernel_resolution("block FP8 public prewarm covers both serving forms")
    try:
        for bound in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = (bfl.run(binding=binding) if bound else
                          bfl.run(source, packed, expected_m=expected_m))
            graph.replay()
            torch.cuda.synchronize()
            lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
            upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            assert torch.all((actual >= lower) & (actual <= upper))
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("capacity", (8, 129))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_v41_prewarm_reuses_capacity_across_live_rows(capacity, dtype) -> None:
    """A single small live warmup must cover its declared V4.1 serving capacity."""
    require_b12x()
    torch.manual_seed(20260910)
    in_features, out_features = 256, 160
    source = torch.randn((capacity, in_features), device="cuda", dtype=dtype)
    weight, scale = _make_block_fp8_weight(out_features, in_features, 32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    plan = bfl.plan(bfl.Caps(
        device=source.device, max_tokens=capacity, in_features=in_features,
        out_features=out_features, output_dtype=dtype, block_size=(32, 32),
    ))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, device=spec.device, dtype=spec.dtype)
    output = torch.empty((capacity, out_features, 1), device="cuda", dtype=dtype)
    pointers = (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
    bfl.prewarm(packed, (1,), output_dtype=dtype, expected_m=capacity)
    freeze_kernel_resolution("V4.1 format and capacity must own kernel resolution")
    try:
        counts = (1, 7, 8) if capacity == 8 else (1, 8, 9, 127, 128, 129)
        for rows in counts:
            binding = bfl.bind(
                plan, scratch=scratch, source=source[:rows],
                packed_weight=packed, output=output[:rows],
            )
            for bound in (True, False):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = (bfl.run(binding=binding) if bound else
                              bfl.run(source[:rows], packed, expected_m=capacity))
                source.normal_().mul_(0.25)
                source[:, :32].mul_(1e-5)
                scratch.fill_(255)
                actual.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                    assert before[key] == after[key]
                assert pointers == (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scale, actual,
                )
    finally:
        unfreeze_kernel_resolution()


@torch.inference_mode()
def test_v41_large_capacity_respects_narrow_tile_with_automatic_k_depth():
    """A 4096-row plan must not attach BK64 to an explicit 64-row MMA tile."""
    from b12x.policy import BLOCK_FP8_LINEAR, get_auto_policy

    require_b12x()
    torch.manual_seed(40961280)
    capacity, live_capacity, k, n = 4096, 65, 1280, 8192
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randint(-4, 5, (live_capacity, k), device=device).bfloat16()
    weight = torch.randint(-2, 3, (n, k), device=device).to(torch.float8_e4m3fn)
    scales = torch.full((n // 32, k // 32), 127, device=device,
                        dtype=torch.uint8).view(torch.float8_e8m0fnu)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    policy = get_auto_policy(device).with_override(
        BLOCK_FP8_LINEAR, bfl.BlockFp8LinearConfig(
            backend="mxfp8", tile_m=64, tile_n=128))
    plan = bfl.plan(bfl.Caps(
        device=device, max_tokens=capacity, in_features=k, out_features=n,
        block_size=(32, 32)), policy=policy)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    output = torch.empty((live_capacity, n, 1), dtype=torch.bfloat16, device=device)

    def bind(rows):
        return bfl.bind(plan, scratch=scratch, source=source[:rows],
                        packed_weight=packed, output=output[:rows])

    bfl.run(binding=bind(live_capacity))
    freeze_kernel_resolution("large-capacity block32 explicit MMA tile")
    try:
        for rows in (1, live_capacity):
            binding = bind(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bfl.run(binding=binding)
            source.neg_()
            scratch.fill_(255)
            output.fill_(float("nan"))
            graph.replay()
            expected = (source[:rows].float() @ weight.float().T).bfloat16()
            torch.testing.assert_close(output[:rows, :, 0], expected, rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()
