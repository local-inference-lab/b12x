from __future__ import annotations

import pytest
import torch

from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared import block_fp8 as block_impl

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm._shared.block_fp8 import (
    BlockFP8LinearScratchCaps,
    block_fp8_linear_mxfp8,
    pack_block_fp8_linear_weight_mxfp8,
    plan_block_fp8_linear_scratch,
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

from tests._reference.helpers import require_b12x


def _make_block_fp8_weight(
    out_features: int,
    in_features: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    if block_size == 32:
        n_blocks = (out_features + 31) // 32
        k_blocks = in_features // 32
        scale_u8 = (
            (torch.arange(n_blocks, device="cuda")[:, None]
             + 2 * torch.arange(k_blocks, device="cuda")[None, :]) % 7 + 123
        ).to(torch.uint8)
        return weight, scale_u8.view(torch.float8_e8m0fnu)
    scale_u8 = (
        torch.arange(
            (out_features // 128) * (in_features // 128),
            device="cuda",
            dtype=torch.int32,
        )
        % 3
        + 126
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(
        out_features // 128,
        in_features // 128,
    )
    return weight, scale


def _v41_dequantized_operands(x, weight, scale):
    from tests.gemm.test_fp8_quant_deepgemm_parity import (
        _per_token_cast_to_fp8,
    )

    values, scales = _per_token_cast_to_fp8(x, gran_k=32)
    x_deq = values.float() * scales.repeat_interleave(32, dim=1)
    w_deq = weight.float() * (
        scale.float().repeat_interleave(32, dim=0)
        .repeat_interleave(32, dim=1)[:weight.shape[0], :weight.shape[1]]
    )
    return x_deq, w_deq


def _assert_v41_accumulation_matches_reference(source, weight, scale, actual):
    x_deq, w_deq = _v41_dequantized_operands(source, weight, scale)
    a, b = x_deq.double(), w_deq.double()
    exact = a @ b.T
    absolute_products = a.abs() @ b.abs().T
    # Native MXF8 MMA and the source K32 GEMM both accumulate in FP32, not
    # FP64. Their summation orders need not match, particularly when tiny
    # K32 groups precede nearly cancelling larger groups. The standard dot
    # product bound is gamma_K * sum(abs(a_i*b_i)), u = 2**-24. UE8M0
    # scaling is an exact power-of-two operation for these finite operands.
    k_u = source.shape[-1] * 2.0**-24
    accumulation_error = (k_u / (1.0 - k_u)) * absolute_products
    # Final BF16/FP16 round-to-nearest contributes at most half the local ULP.
    actual64 = actual.double()
    below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
    above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
    rounding_error = 0.5 * torch.maximum(actual64 - below, above - actual64)
    assert torch.isfinite(actual).all()
    assert torch.all((actual64 - exact).abs() <= accumulation_error + rounding_error)


def _reference_from_quantized_operands(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if block_size == 32:
        x_deq, w_deq = _v41_dequantized_operands(x, weight, scale)
    else:
        x_q = quantize_block_fp8_linear_input_mxfp8(x)
        w_q = pack_block_fp8_linear_weight_mxfp8(weight, scale)
        x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
        w_deq = dequantize_mxfp8_rows_torch(w_q.weight.values, w_q.weight.scale_rows)
    reference = x_deq.double() @ w_deq.double().T
    rounded = reference.to(x.dtype)
    # Torch's half constructors convert through FP32; correct double rounding.
    for direction in (-float("inf"), float("inf")):
        adjacent = torch.nextafter(rounded, torch.full_like(rounded, direction))
        closer = (reference - adjacent.double()).abs() < (reference - rounded.double()).abs()
        rounded = torch.where(closer, adjacent, rounded)
    return rounded


def test_block_fp8_linear_matches_quantized_reference() -> None:
    require_b12x()
    torch.manual_seed(20260523)

    tokens, in_features, out_features = 7, 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    actual = block_fp8_linear_mxfp8(x, packed)
    expected = _reference_from_quantized_operands(x, weight, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tokens", [8, 9, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_linear_immediate_gemm_skips_padding_initialization(
    tokens: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    torch.manual_seed(20260902)

    in_features, out_features = 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=dtype)
        / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    expected = _reference_from_quantized_operands(x, weight, scale)
    allocate = block_impl.empty_mxfp8_rows_bases
    poison = 0

    def poisoned_allocation(*args, **kwargs):
        bases = allocate(*args, **kwargs)
        if not kwargs.get("initialize_scales", True):
            bases[1].fill_(poison)
            bases[2].fill_(poison)
        return bases

    monkeypatch.setattr(block_impl, "empty_mxfp8_rows_bases", poisoned_allocation)
    first = block_fp8_linear_mxfp8(x, packed)
    poison = 255
    second = block_fp8_linear_mxfp8(x, packed)
    torch.cuda.synchronize()

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        first.float(), expected.to(first.dtype).float(), rtol=0, atol=0
    )


def test_block_fp8_linear_replays_under_cuda_graph() -> None:
    require_b12x()
    torch.manual_seed(20260524)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = plan.bind(
        scratch=scratch,
        source=x,
        packed_weight=packed,
        output=output,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output[:, :, 0], eager, rtol=0, atol=0)


def test_block_fp8_linear_scratch_binding_replays_under_cuda_graph() -> None:
    require_b12x()
    torch.manual_seed(20260526)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = plan.bind(
        scratch=scratch,
        source=x,
        packed_weight=packed,
        output=output,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [8, 9, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_linear_scratch_padding_is_not_observed(
    tokens: int, dtype: torch.dtype,
) -> None:
    """Poisoned M128 scale padding must not affect logical GEMM rows."""

    require_b12x()
    torch.manual_seed(20260901)

    in_features, out_features = 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=dtype)
        / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = plan.bind(
        scratch=scratch,
        source=x,
        packed_weight=packed,
        output=output,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=binding)

    scratch[0].fill_(0)
    zero_poison = run_once().clone()
    scratch[0].fill_(255)
    ff_poison = run_once().clone()
    torch.cuda.synchronize()

    expected = _reference_from_quantized_operands(x, weight, scale)
    torch.testing.assert_close(zero_poison, ff_poison, rtol=0, atol=0)
    torch.testing.assert_close(
        zero_poison.float(), expected.to(zero_poison.dtype).float(), rtol=0, atol=0
    )

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()
    scratch[0].fill_(0)
    graph.replay()
    graph_zero = output[:, :, 0].clone()
    scratch[0].fill_(255)
    graph.replay()
    graph_ff = output[:, :, 0].clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_zero, graph_ff, rtol=0, atol=0)
    torch.testing.assert_close(graph_zero, zero_poison, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_linear_public_quantizer_preserves_scale_padding(
    dtype: torch.dtype,
) -> None:
    require_b12x()
    x = torch.randn((129, 256), device="cuda", dtype=dtype).mul_(0.25)
    rows = bfl.quantize_input(x)
    physical_rows = rows.scale_mma.view(torch.uint8).permute(5, 2, 1, 0, 4, 3)
    logical_rows = physical_rows.reshape(1, 256, 8)
    torch.testing.assert_close(logical_rows[:, :129], rows.scale_rows.view(torch.uint8))
    assert torch.all(logical_rows[:, 129:] == 127)


def test_block_fp8_linear_default_fused_path_captures() -> None:
    require_b12x()
    torch.manual_seed(20260525)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    eager = block_fp8_linear_mxfp8(x, packed).clone()
    torch.cuda.synchronize()

    block_fp8_linear_mxfp8(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = block_fp8_linear_mxfp8(x, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_block_fp8_linear_live_m_does_not_resolve_new_dense_kernel() -> None:
    require_b12x()
    torch.manual_seed(20260528)

    warm_tokens, live_tokens = 4096, 1824
    in_features, out_features = 128, 1536
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    warm_x = (
        torch.randn((warm_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    live_x = (
        torch.randn((live_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()

    block_fp8_linear_mxfp8(warm_x, packed)
    torch.cuda.synchronize()

    freeze_kernel_resolution("block FP8 dense GEMM live M should be runtime")
    try:
        actual = block_fp8_linear_mxfp8(live_x, packed)
        torch.cuda.synchronize()
    finally:
        unfreeze_kernel_resolution()

    expected = _reference_from_quantized_operands(live_x, weight, scale)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=1e-4,
    )


def test_block_fp8_linear_small_live_m_reuses_prefill_dense_kernel() -> None:
    require_b12x()
    torch.manual_seed(20260529)

    warm_tokens = 512
    live_token_counts = (16, 32, 128)
    in_features, out_features = 1024, 8192
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    warm_x = (
        torch.randn((warm_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    live_xs = [
        (
            torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
            / 4
        ).contiguous()
        for tokens in live_token_counts
    ]

    block_fp8_linear_mxfp8(warm_x, packed)
    torch.cuda.synchronize()

    freeze_kernel_resolution("small live M should reuse the prefill dense kernel")
    try:
        for tokens, live_x in zip(live_token_counts, live_xs, strict=True):
            actual = block_fp8_linear_mxfp8(live_x, packed)
            torch.cuda.synchronize()
            assert actual.shape == (tokens, out_features)
    finally:
        unfreeze_kernel_resolution()


def test_block_fp8_linear_expected_m_decode_regime_reuses_kernel() -> None:
    # DeepGEMM-style expected_m hint: a decode-regime kernel (expected_m<=128 ->
    # 32x128 tile) must (a) produce byte-identical output to the default
    # (tile choice does not change the block-scaled MMA result) and (b) be
    # reused for every live M in the regime under frozen resolution.
    require_b12x()
    from b12x._lib.dense_gemm import _select_default_mma_tiler_mn

    torch.manual_seed(20260530)
    in_features, out_features = 1024, 8192  # wide-N (>1536) MXFP8 regime
    expected_m = 64  # decode/small-batch regime
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    assert _select_default_mma_tiler_mn(
        expected_m, out_features, sm, is_mxfp8=True, expected_m=expected_m
    ) == (32, 128)

    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    # (a) tile-independence of numerics: hint (32x128) vs default (64x128).
    x = (
        torch.randn((32, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    default_out = block_fp8_linear_mxfp8(x, packed)
    hinted_out = block_fp8_linear_mxfp8(x, packed, expected_m=expected_m)
    torch.cuda.synchronize()
    torch.testing.assert_close(hinted_out.float(), default_out.float(), rtol=0, atol=0)

    # (b) warm the decode kernel once, freeze, serve a range of live M -> all
    # reuse the same warmed (32x128) kernel (no recompile under frozen
    # resolution). Live M stays in the persistent-scheduler policy class (m>=16),
    # matching the warm M; M==1 / m<16 are separate policy regimes
    # (use_m1_non_tma / direct scheduler) that must be warmed on their own -- a
    # pre-existing dense_gemm constraint independent of the expected_m hint.
    warm_x = (
        torch.randn((256, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    block_fp8_linear_mxfp8(warm_x, packed, expected_m=expected_m)
    torch.cuda.synchronize()

    freeze_kernel_resolution("decode-regime block FP8 reused for all live M")
    try:
        for tokens in (16, 32, 128):
            live_x = (
                torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
                / 4
            ).contiguous()
            out = block_fp8_linear_mxfp8(live_x, packed, expected_m=expected_m)
            torch.cuda.synchronize()
            assert out.shape == (tokens, out_features)
    finally:
        unfreeze_kernel_resolution()


def test_block_fp8_linear_expected_m_short_k_large_n_matches_reference() -> None:
    """Exercise the production expected_m route through 128x128x64."""
    require_b12x()
    torch.manual_seed(20260702)

    tokens, in_features, out_features = 16, 1024, 16384
    expected_m = 4096
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    actual = block_fp8_linear_mxfp8(x, packed, expected_m=expected_m)
    expected = _reference_from_quantized_operands(x, weight, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0,
        atol=1 / 128,
    )


@pytest.mark.parametrize(
    "tokens,in_features,out_features",
    [(1, 5120, 1280), (8, 5120, 512), (9, 256, 96),
     (129, 256, 160), (33, 1280, 8192), (7, 1024, 5120),
     (9, 6144, 25600), (1, 32, 256), (1, 576, 5120),
     (8, 576, 5120), (17, 576, 5120), (4096, 576, 5120)],
)
def test_block_fp8_linear_v41_independent_k32_n32_scales(
    tokens: int, in_features: int, out_features: int,
) -> None:
    """Neither K128 replication nor N128 sharing may corrupt V4.1 scales."""
    require_b12x()
    torch.manual_seed(20260910)
    source = torch.randn(
        (tokens, in_features), device="cuda", dtype=torch.bfloat16,
    ).mul_(0.25)
    # Include tiny groups: unfloored legacy activation quantization differs here.
    source[:, :32].mul_(1e-5)
    weight, scale = _make_block_fp8_weight(out_features, in_features, 32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    torch.testing.assert_close(
        packed.weight.values[:, :in_features].view(torch.uint8), weight.view(torch.uint8),
        rtol=0, atol=0,
    )
    from tests.gemm.test_fp8_quant_deepgemm_parity import (
        _per_token_cast_to_fp8, _sf_fp32_to_e8m0_u8,
    )

    x_values, x_scales = _per_token_cast_to_fp8(source, 32)
    x_q = bfl.quantize_input(source, block_size=(32, 32))
    torch.testing.assert_close(
        x_q.values[:, :in_features].view(torch.uint8), x_values.view(torch.uint8), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        x_q.scale_rows.view(torch.uint8)[0, :, :in_features // 32], _sf_fp32_to_e8m0_u8(x_scales),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0, :, :in_features // 32],
        scale.view(torch.uint8).repeat_interleave(32, dim=0)[:out_features],
        rtol=0, atol=0,
    )
    actual = bfl.run(source, packed, expected_m=tokens)
    _assert_v41_accumulation_matches_reference(source, weight, scale, actual)
    assert packed.in_features == in_features
    assert packed.padded_in_features == (in_features + 127) // 128 * 128
    assert not torch.count_nonzero(packed.weight.values[:, in_features:].view(torch.uint8))
    assert not torch.count_nonzero(x_q.values[:, in_features:].view(torch.uint8))
    assert torch.isfinite(x_q.scale_rows.float()).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_linear_k32_tail_scratch_graph_replay(dtype) -> None:
    """TP shard padding is rewritten in caller scratch on every graph replay."""
    require_b12x()
    torch.manual_seed(1337)
    capacity, in_features, out_features = 128, 576, 5120
    source = torch.randn((capacity, in_features), device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(out_features, in_features, 32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    plan = bfl.plan(bfl.Caps(
        device=source.device, max_tokens=capacity, in_features=in_features,
        out_features=out_features, output_dtype=dtype, block_size=(32, 32),
    ))
    scratch = tuple(torch.empty(shape, dtype=dt, device=source.device)
                    for shape, dt in plan.shapes_and_dtypes())
    output = torch.empty((capacity, out_features, 1), device=source.device, dtype=dtype)
    # Warm both the planned and functional paths at one fixed capacity. Live
    # row counts must not create another compiled specialization afterward.
    plan.bind(scratch=scratch, source=source, packed_weight=packed, output=output).run()
    bfl.run(source, packed, expected_m=capacity)
    torch.cuda.synchronize()
    freeze_kernel_resolution()
    try:
        for tokens in (1, 8, 17, 128):
            live_source, live_output = source[:tokens], output[:tokens]
            binding = plan.bind(scratch=scratch, source=live_source,
                                packed_weight=packed, output=live_output)
            binding.run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.run()
            for poison in (0, 255):
                for storage in scratch:
                    storage.fill_(poison)
                live_source.neg_()
                graph.replay()
                torch.cuda.synchronize()
                _assert_v41_accumulation_matches_reference(
                    live_source, weight, scale, live_output[:, :, 0],
                )
                assert not torch.count_nonzero(binding.x_q.values[:, in_features:].view(torch.uint8))
                assert torch.isfinite(binding.x_q.scale_rows.float()).all()
                functional = bfl.run(live_source, packed, expected_m=capacity)
                _assert_v41_accumulation_matches_reference(live_source, weight, scale, functional)
    finally:
        unfreeze_kernel_resolution()


def test_block_fp8_linear_v41_rejects_lossy_weight_scale_repacking() -> None:
    require_b12x()
    weight = torch.ones((64, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((2, 4), 0.3, device="cuda")
    with pytest.raises(ValueError, match="exact UE8M0"):
        bfl.pack_weight(weight, scales, block_size=(32, 32))
