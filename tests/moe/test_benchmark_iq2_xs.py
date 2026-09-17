"""IQ2_XS enters the canonical benchmark without FP4 conversion."""

import pytest
import torch

from benchmarks import benchmark_moe as benchmark
from b12x.moe import fused_moe
from .test_iq2_xs_checkpoint import make_snapshot


@pytest.fixture
def snapshot(tmp_path):
    return make_snapshot(tmp_path)


def test_benchmark_profile_and_checkpoint_adapter(snapshot, monkeypatch):
    profile = benchmark.MODEL_PROFILES["qwen36-35b-iq2-xs"]
    assert profile.hf_repo_id == "nvidia/Qwen3.6-35B-A3B-IQ2_XS-NVFP4"
    assert profile.default_quant_mode == "w4a16"
    spec = benchmark.build_model_spec(snapshot, profile, tp_size_override=2, tp_rank=1)
    weights = benchmark.load_iq2_xs_expert_weights(
        snapshot, spec, layer_idx=0, activation="silu", device="cpu"
    )
    assert weights.source_format == "iq2_xs"
    assert weights.w13_weight.shape == (3, 512, 1, 74)
    assert weights.w2_weight.shape == (3, 256, 1, 74)
    assert weights.w13_permuted is None and weights.w13_blockscale_swizzled is None
    params = benchmark.get_quant_mode_params(weights, "per-expert", "w4a16")
    plan = benchmark.plan_b12x_benchmark_weights(
        weights, quant_mode="w4a16", activation="silu"
    )
    assert plan.prepared_format.packing is fused_moe.WeightPacking.IQ2_XS_COMPACT

    sentinel = object()

    def prepare(*, plan, weights):
        assert isinstance(weights, fused_moe.IQ2XSWeights)
        return sentinel

    monkeypatch.setattr(fused_moe, "prepare_weights", prepare)
    actual, actual_params = benchmark.prepare_b12x_benchmark_weights(
        weights, params, quant_mode="w4a16", activation="silu", plan=plan
    )
    assert actual is sentinel and actual_params is params


def test_benchmark_iq2_xs_oracle_and_invalid_precision(snapshot):
    profile = benchmark.MODEL_PROFILES["qwen36-35b-iq2-xs"]
    spec = benchmark.build_model_spec(snapshot, profile)
    weights = benchmark.load_iq2_xs_expert_weights(
        snapshot, spec, layer_idx=0, activation="silu", device="cpu"
    )
    params = benchmark.get_quant_mode_params(weights, "per-expert", "w4a16")
    x = torch.full((2, 256), 0.03125, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    probabilities = torch.full((2, 2), 0.5)
    result = benchmark.make_oracle_reference(
        "w4a16", "w4a16", x, weights, params, ids, probabilities, activation="silu"
    )
    assert result.shape == x.shape and result.dtype == torch.bfloat16
    assert torch.isfinite(result).all() and torch.count_nonzero(result) > 0
    zero = benchmark.make_oracle_reference(
        "w4a16", "w4a16", x * 0, weights, params, ids, probabilities, activation="silu"
    )
    assert torch.count_nonzero(zero) == 0
    with pytest.raises(ValueError):
        benchmark.make_oracle_reference(
            "nvfp4", "w4a16", x, weights, params, ids, probabilities, activation="silu"
        )
    with pytest.raises(ValueError):
        benchmark.plan_b12x_benchmark_weights(
            weights, quant_mode="nvfp4", activation="silu"
        )
