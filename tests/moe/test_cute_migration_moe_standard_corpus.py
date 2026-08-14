"""Production standard-MoE coverage for the CuTe 4.6 migration corpus.

These tests deliberately enter through the public planned/bound serving API.
They use synthetic checkpoint tensors, a pure-Torch GPU oracle, fixed scratch,
and live-input CUDA-graph replay.  Together they force the production compile
IDs for direct micro, dynamic prefill, and both tiny-decode phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from b12x.moe._shared.kernels.reference import (
    compare_to_reference,
    moe_reference_nvfp4,
    moe_reference_w4a8_mx,
)

from tests._reference.helpers import (
    prepare_tp_moe_fp4_experts,
    require_b12x,
    swizzle_block_scale_reference,
)


_E = 4
_K = 512
_N = 128
_TOPK = 2


@dataclass(frozen=True)
class _Weights:
    w1_fp4: torch.Tensor
    w1_scale: torch.Tensor
    w1_alpha: torch.Tensor
    a1_scale: torch.Tensor
    w2_fp4: torch.Tensor
    w2_scale: torch.Tensor
    w2_alpha: torch.Tensor
    a2_scale: torch.Tensor


@dataclass(frozen=True)
class _Inputs:
    a: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class _BoundCase:
    """Strongly own every allocation used by a captured serving launch."""

    source_weights: _Weights
    experts: object
    scratch_plan: object
    scratch: tuple[torch.Tensor, ...]
    binding: object


def _make_nvfp4_weights(device: torch.device, *, seed: int) -> _Weights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_fp4 = torch.randint(
        0,
        256,
        (_E, 2 * _N, _K // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (_E, _K, _N // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )

    # ModelOpt NVFP4 carries FlashInfer's vec16-swizzled E4M3 block scales.
    # A constant exact power of two keeps the synthetic layer well-conditioned
    # while the random FP4 payload still exercises every nibble value.
    w1_logical_scale = torch.full(
        (_E, 2 * _N, _K // 16),
        2.0**-5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    w2_logical_scale = torch.full(
        (_E, _K, _N // 16),
        2.0**-5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    w1_scale = swizzle_block_scale_reference(w1_logical_scale).contiguous()
    w2_scale = swizzle_block_scale_reference(w2_logical_scale).contiguous()
    w1_alpha = torch.linspace(0.5, 0.8, _E, dtype=torch.float32, device=device)
    w2_alpha = torch.linspace(0.6, 0.9, _E, dtype=torch.float32, device=device)
    unit = torch.ones(1, dtype=torch.float32, device=device)
    return _Weights(
        w1_fp4=w1_fp4,
        w1_scale=w1_scale,
        w1_alpha=w1_alpha,
        a1_scale=unit,
        w2_fp4=w2_fp4,
        w2_scale=w2_scale,
        w2_alpha=w2_alpha,
        a2_scale=unit.clone(),
    )


def _make_mxfp4_weights(device: torch.device, *, seed: int) -> _Weights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_fp4 = torch.randint(
        0,
        256,
        (_E, 2 * _N, _K // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (_E, _K, _N // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    # E8M0 byte 122 is exactly 2^-5.  These are checkpoint-native logical
    # K/32 grids; production preparation repacks them for tiny decode.
    w1_scale = torch.full(
        (_E, 2 * _N, _K // 32), 122, dtype=torch.uint8, device=device
    )
    w2_scale = torch.full(
        (_E, _K, _N // 32), 122, dtype=torch.uint8, device=device
    )
    alpha = torch.ones(_E, dtype=torch.float32, device=device)
    unit = torch.ones(_E, dtype=torch.float32, device=device)
    return _Weights(
        w1_fp4=w1_fp4,
        w1_scale=w1_scale,
        w1_alpha=alpha,
        a1_scale=unit,
        w2_fp4=w2_fp4,
        w2_scale=w2_scale,
        w2_alpha=alpha.clone(),
        a2_scale=unit.clone(),
    )


def _make_inputs(
    device: torch.device,
    *,
    m: int,
    seed: int,
    route_shift: int,
) -> _Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = (
        torch.randn(m, _K, dtype=torch.float32, device=device, generator=generator)
        * 0.35
    ).to(torch.bfloat16)
    token = torch.arange(m, dtype=torch.int32, device=device)
    topk_ids = torch.stack(
        (
            (token + route_shift) % _E,
            (token + route_shift + 1) % _E,
        ),
        dim=1,
    ).contiguous()
    topk_weights = torch.rand(
        m, _TOPK, dtype=torch.float32, device=device, generator=generator
    ).add_(0.25)
    topk_weights.div_(topk_weights.sum(dim=1, keepdim=True))

    assert topk_ids.dtype is torch.int32 and topk_ids.is_contiguous()
    assert bool(((topk_ids >= 0) & (topk_ids < _E)).all().item())
    assert bool((topk_ids[:, 0] != topk_ids[:, 1]).all().item())
    assert bool((topk_weights > 0).all().item())
    torch.testing.assert_close(
        topk_weights.sum(dim=1),
        torch.ones(m, dtype=torch.float32, device=device),
        rtol=0,
        atol=1e-6,
    )
    return _Inputs(a=a, topk_ids=topk_ids, topk_weights=topk_weights)


def _nvfp4_oracle(
    weights: _Weights,
    inputs: _Inputs,
    *,
    quant_scale_math: str = "direct_division",
) -> torch.Tensor:
    # This is the pure-Torch GPU oracle; it does not instantiate or call a CuTe
    # kernel and consumes the original checkpoint layout directly.
    return moe_reference_nvfp4(
        inputs.a,
        weights.w1_fp4,
        weights.w1_scale,
        weights.w1_alpha,
        weights.w2_fp4,
        weights.w2_scale,
        weights.w2_alpha,
        weights.a1_scale,
        weights.a2_scale,
        inputs.topk_ids,
        inputs.topk_weights,
        _E,
        _K,
        _N,
        activation="silu",
        quant_scale_math=quant_scale_math,
    )


def _mxfp4_oracle(weights: _Weights, inputs: _Inputs) -> torch.Tensor:
    # No prepared/repacked tensor participates in this oracle.  It consumes the
    # checkpoint-native FP4 + E8M0 grids and emulates MXFP8 activation rounding.
    return moe_reference_w4a8_mx(
        inputs.a.float(),
        weights.w1_fp4,
        weights.w1_scale,
        None,
        weights.w1_alpha,
        weights.w2_fp4,
        weights.w2_scale,
        None,
        weights.w2_alpha,
        inputs.topk_ids,
        inputs.topk_weights,
        _E,
        _K,
        _N,
        activation="silu",
        w13_layout="w13",
    )


def _prepare_and_bind(
    weights: _Weights,
    inputs: _Inputs,
    *,
    quant_mode: str,
    source_format: str,
) -> _BoundCase:
    from b12x.moe.fused_moe._impl import TPMoEScratchCaps, plan_tp_moe_scratch

    experts = prepare_tp_moe_fp4_experts(
        a=inputs.a,
        a1_gscale=weights.a1_scale,
        w1_fp4=weights.w1_fp4,
        w1_blockscale=weights.w1_scale,
        w1_alphas=weights.w1_alpha,
        a2_gscale=weights.a2_scale,
        w2_fp4=weights.w2_fp4,
        w2_blockscale=weights.w2_scale,
        w2_alphas=weights.w2_alpha,
        activation="silu",
        quant_mode=quant_mode,
        source_format=source_format,
        w13_layout="w13",
    )
    scratch_plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=int(inputs.a.shape[0]),
            num_topk=_TOPK,
            device=inputs.a.device,
            weight_plan=experts.plan,
            core_token_counts=(int(inputs.a.shape[0]),),
            route_num_experts=0,
            quant_mode=quant_mode,
            frozen=True,
        )
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in scratch_plan.scratch_specs()
    )
    output = torch.empty_like(inputs.a)
    binding = scratch_plan.bind(
        scratch=scratch,
        a=inputs.a,
        experts=experts,
        topk_weights=inputs.topk_weights,
        topk_ids=inputs.topk_ids,
        output=output,
        input_scales_static=True,
        fast_math=False,
    )
    assert binding.output is output
    return _BoundCase(
        source_weights=weights,
        experts=experts,
        scratch_plan=scratch_plan,
        scratch=scratch,
        binding=binding,
    )


def _assert_oracle(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    context: str,
    min_cos: float,
    max_normalized_rmse: float,
) -> None:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    assert bool(actual_f32.isfinite().all().item()), (context, "non-finite output")
    actual_rms = actual_f32.square().mean().sqrt().item()
    reference_rms = reference_f32.square().mean().sqrt().item()
    assert actual_rms > 1e-5, (context, "all-zero output")
    assert reference_rms > 1e-5, (context, "all-zero reference")
    metrics = compare_to_reference(actual_f32, reference_f32)
    normalized_rmse = metrics.rmse / reference_rms
    assert metrics.cos >= min_cos, (context, metrics, normalized_rmse)
    assert normalized_rmse <= max_normalized_rmse, (
        context,
        metrics,
        normalized_rmse,
    )


def _run_live_graph_check(
    case: _BoundCase,
    *,
    initial: _Inputs,
    changed: _Inputs,
    initial_reference: torch.Tensor,
    changed_reference: torch.Tensor,
    context: str,
    min_cos: float,
    max_normalized_rmse: float,
) -> None:
    from b12x.moe.fused_moe._impl import b12x_moe_fp4

    binding = case.binding
    output = binding.output
    assert output is not None

    # Eager warmup resolves and compiles the production specialization before
    # capture.  No workspace or output allocation occurs inside the graph.
    b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    _assert_oracle(
        output,
        initial_reference,
        context=f"{context}:eager",
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )
    initial_output = output.clone()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        b12x_moe_fp4(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    _assert_oracle(
        output,
        initial_reference,
        context=f"{context}:capture",
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )

    # Mutate every live serving input in place.  IDs remain in range and each
    # token still selects two distinct experts with positive normalized weights.
    initial.a.copy_(changed.a)
    initial.topk_ids.copy_(changed.topk_ids)
    initial.topk_weights.copy_(changed.topk_weights)
    output.fill_(37.0)  # Poison proves the captured launch owns output reset.
    graph.replay()
    torch.cuda.synchronize()
    _assert_oracle(
        output,
        changed_reference,
        context=f"{context}:live-replay",
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )

    changed_rmse = (
        (output.float() - initial_output.float()).square().mean().sqrt().item()
    )
    initial_rms = initial_output.float().square().mean().sqrt().item()
    assert changed_rmse > max(1e-4, 0.05 * initial_rms), (
        context,
        changed_rmse,
        initial_rms,
    )


def _reset_dispatch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", raising=False)
    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches

    clear_tp_moe_caches()


def test_standard_moe_micro_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach ``integration.tp_moe.micro_direct`` through production dispatch."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=101)
    initial = _make_inputs(device, m=2, seed=102, route_shift=0)
    # Seed 103 intentionally exercises an FP4 RN-even boundary.  For token 0,
    # direct division produces exactly -2.5 while the micro kernel's explicit
    # reciprocal-then-multiply is just below -2.5 and rounds to the adjacent
    # FP4 value.  Preserve that boundary and model the production evaluation
    # order instead of selecting an input that happens not to expose it.
    changed = _make_inputs(device, m=2, seed=103, route_shift=2)
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        quant_scale_math="reciprocal_multiply",
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        quant_scale_math="reciprocal_multiply",
    )
    direct_division_reference = _nvfp4_oracle(weights, changed)
    assert not torch.equal(changed_reference, direct_division_reference), (
        "seed 103 must retain the reciprocal/division FP4 tie boundary"
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.scratch_plan.caps.frozen
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    assert 2 * _TOPK < 64
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-micro",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


def test_standard_moe_dynamic_prefill_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach ``integration.tp_moe.dynamic`` at a prefill-sized standard M."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=201)
    initial = _make_inputs(device, m=128, seed=202, route_shift=0)
    changed = _make_inputs(device, m=128, seed=203, route_shift=2)
    initial_reference = _nvfp4_oracle(weights, initial)
    changed_reference = _nvfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"
    assert launch_plan.execution.tile_m == 64
    assert launch_plan.execution.tile_n == 128
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-dynamic-prefill-m128",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


def test_standard_moe_tiny_decode_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach both ``integration.tp_moe.tiny_decode`` production phases."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_mxfp4_weights(device, seed=301)
    initial = _make_inputs(device, m=2, seed=302, route_shift=0)
    changed = _make_inputs(device, m=2, seed=303, route_shift=2)

    # Preparation destructively transfers/re-packs the source allocation, so
    # calculate both independent checkpoint-layout references first.
    initial_reference = _mxfp4_oracle(weights, initial)
    changed_reference = _mxfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    assert case.binding.quant_mode == "w4a8_mx"
    assert 1 <= int(initial.a.shape[0]) <= 4
    assert _K % 256 == 0 and _N % 32 == 0 and (_K // 128) % 4 == 0
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-tiny-decode-phases-1-2",
        min_cos=0.998,
        max_normalized_rmse=0.05,
    )


# ---------------------------------------------------------------------------
# Regression tests for issue #154: out-of-range expert IDs must be clamped
# in-kernel so they cannot cause OOB GPU writes.  Each test forces a
# specific kernel route (micro.py via nvfp4, tiny_decode via w4a8_mx,
# W4A16 FC2-only capacity), uses actual CUDA graph capture/replay, and
# asserts finite output + all-invalid zero.
# ---------------------------------------------------------------------------


def _make_invalid_inputs(
    device: torch.device,
    *,
    m: int,
    seed: int,
    invalid_ids: list[int],
    dtype: torch.dtype = torch.int32,
) -> _Inputs:
    """Like _make_inputs but injects invalid expert IDs (no validity guard)."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = (
        torch.randn(m, _K, dtype=torch.float32, device=device, generator=generator)
        * 0.35
    ).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [invalid_ids[:_TOPK] for _ in range(m)],
        dtype=dtype,
        device=device,
    ).contiguous()
    topk_weights = torch.rand(
        m, _TOPK, dtype=torch.float32, device=device, generator=generator
    ).add_(0.25)
    topk_weights.div_(topk_weights.sum(dim=1, keepdim=True))
    return _Inputs(a=a, topk_ids=topk_ids, topk_weights=topk_weights)


def _run_invalid_id_safety_check(
    case: _BoundCase,
    initial: _Inputs,
    changed: _Inputs,
    *,
    context: str,
) -> None:
    """Run eager + CUDA-graph capture + mutate + replay; assert finite output."""
    from b12x.moe.fused_moe._impl import b12x_moe_fp4

    binding = case.binding
    output = binding.output
    assert output is not None

    # Eager warmup.
    b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    assert bool(output.float().isfinite().all().item()), (context, "eager non-finite")

    # Graph capture + replay.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        b12x_moe_fp4(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    assert bool(output.float().isfinite().all().item()), (
        context,
        "capture non-finite",
    )

    # Replay with mutated invalid inputs.
    initial.a.copy_(changed.a)
    initial.topk_ids.copy_(changed.topk_ids)
    initial.topk_weights.copy_(changed.topk_weights)
    output.fill_(37.0)
    graph.replay()
    torch.cuda.synchronize()
    assert bool(output.float().isfinite().all().item()), (
        context,
        "replay non-finite",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_micro_nvfp4_negative_expert_id_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real micro.py route (nvfp4): clamp -1 expert ID without OOB writes."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=401)
    initial = _make_invalid_inputs(device, m=2, seed=402, invalid_ids=[-1, 0])
    changed = _make_invalid_inputs(device, m=2, seed=403, invalid_ids=[0, -1])
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.binding.implementation == "micro"
    _run_invalid_id_safety_check(case, initial, changed, context="micro-nvfp4-neg-id")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_micro_nvfp4_equal_num_experts_id_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real micro.py route (nvfp4): clamp ID == num_experts without OOB."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=411)
    initial = _make_invalid_inputs(device, m=2, seed=412, invalid_ids=[_E, 0])
    changed = _make_invalid_inputs(device, m=2, seed=413, invalid_ids=[0, _E])
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.binding.implementation == "micro"
    _run_invalid_id_safety_check(case, initial, changed, context="micro-nvfp4-eq-E-id")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_micro_nvfp4_large_expert_id_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real micro.py route (nvfp4): clamp very large ID without OOB."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=421)
    initial = _make_invalid_inputs(device, m=2, seed=422, invalid_ids=[99999, 0])
    changed = _make_invalid_inputs(device, m=2, seed=423, invalid_ids=[0, 99999])
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.binding.implementation == "micro"
    _run_invalid_id_safety_check(case, initial, changed, context="micro-nvfp4-large-id")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_micro_nvfp4_int64_wraparound_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real micro.py route: Int64 wraparound IDs must be rejected."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=431)
    wrap_val = 2**32 + 1
    initial = _make_invalid_inputs(
        device, m=2, seed=432, invalid_ids=[wrap_val, 0], dtype=torch.int64
    )
    changed = _make_invalid_inputs(
        device, m=2, seed=433, invalid_ids=[0, wrap_val], dtype=torch.int64
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.binding.implementation == "micro"
    _run_invalid_id_safety_check(case, initial, changed, context="micro-nvfp4-int64-wrap")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_decode_negative_expert_id_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny-decode kernel must clamp -1 expert ID without OOB writes."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_mxfp4_weights(device, seed=441)
    initial = _make_invalid_inputs(device, m=2, seed=442, invalid_ids=[-1, 0])
    changed = _make_invalid_inputs(device, m=2, seed=443, invalid_ids=[0, -1])
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
    )
    assert 1 <= int(initial.a.shape[0]) <= 4
    _run_invalid_id_safety_check(case, initial, changed, context="tiny-decode-neg-id")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_decode_large_expert_id_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny-decode kernel must clamp very large ID without OOB writes."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_mxfp4_weights(device, seed=451)
    initial = _make_invalid_inputs(device, m=2, seed=452, invalid_ids=[99999, 0])
    changed = _make_invalid_inputs(device, m=2, seed=453, invalid_ids=[0, 99999])
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
    )
    assert 1 <= int(initial.a.shape[0]) <= 4
    _run_invalid_id_safety_check(case, initial, changed, context="tiny-decode-large-id")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a16_fc2_capacity_mismatch_invalid_id_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W4A16 FC2-only: ID in [num_experts, capacity) must be rejected.

    The compiled capacity bucket may exceed the actual expert count.
    An ID == num_experts must be clamped, not indexed past the allocation.
    """
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    from b12x.moe.fused_moe._impl import (
        prepare_w4a16_fc2_e8m0,
        run_w4a16_fc2_e8m0,
        prewarm_w4a16_fc2_e8m0,
    )

    E_actual = 2
    K = _K
    n = _N
    # API: prepare_w4a16_fc2_e8m0(w2_fp4, w2_e8m0_scale)
    # w2_fp4: [E, H, I/2] uint8, w2_e8m0_scale: [E, H, I/32] uint8
    w2_fp4 = torch.randint(
        0, 256, (E_actual, K, n // 2), dtype=torch.uint8, device=device
    )
    # E8M0 scale bytes: byte 122 = 2^-5
    w2_e8m0_scale = torch.full(
        (E_actual, K, n // 32), 122, dtype=torch.uint8, device=device
    )

    experts = prepare_w4a16_fc2_e8m0(w2_fp4, w2_e8m0_scale)
    prewarm_w4a16_fc2_e8m0(experts, route_ids_dtype=torch.int32)

    m = 4
    intermediate = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    # One route has ID == E_actual (in the capacity bucket but past the allocation).
    route_ids = torch.tensor([0, 1, E_actual, -1], dtype=torch.int32, device=device)
    route_weights = torch.tensor(
        [0.5, 0.3, 0.2, 0.1], dtype=torch.float32, device=device
    )
    run_w4a16_fc2_e8m0(
        intermediate, experts, route_ids, route_weights, output=output
    )
    assert torch.isfinite(output).all(), (
        "FC2 capacity mismatch produced non-finite output"
    )

    # All-invalid IDs must produce zero output.
    route_ids_all_bad = torch.full((m,), -1, dtype=torch.int32, device=device)
    route_weights_all = torch.ones(m, dtype=torch.float32, device=device)
    output_all_bad = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    run_w4a16_fc2_e8m0(
        intermediate, experts, route_ids_all_bad, route_weights_all, output=output_all_bad
    )
    assert output_all_bad.abs().sum().item() == 0.0, (
        "all-invalid FC2 IDs must produce zero"
    )
