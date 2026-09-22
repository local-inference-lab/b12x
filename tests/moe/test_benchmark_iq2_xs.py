"""IQ2_XS enters the canonical benchmark without FP4 conversion."""

import pytest
import torch
import json
from safetensors.torch import save_file

from benchmarks import benchmark_moe as benchmark
from b12x.moe import fused_moe
from .test_iq2_xs_checkpoint import make_snapshot
from .test_iq2_xs import blocks


def test_puzzle3_layer_geometry_relu2_and_tp_slicing(tmp_path):
    config = dict(
        model_type="nemotron_h_puzzle",
        hidden_size=512,
        num_hidden_layers=3,
        mlp_hidden_act="relu2",
        block_configs=[
            dict(block_type="mamba"),
            dict(
                block_type="moe",
                moe_latent_size=256,
                moe_intermediate_size=512,
                n_routed_experts=3,
                num_experts_per_tok=2,
            ),
            dict(
                block_type="moe",
                moe_latent_size=256,
                moe_intermediate_size=1024,
                n_routed_experts=3,
                num_experts_per_tok=1,
            ),
        ],
    )
    tensors, recipes = {}, {}
    for layer, width in ((1, 512), (2, 1024)):
        for expert in range(3):
            for projection, n, k in (("up", width, 256), ("down", 256, width)):
                name = (
                    f"backbone.layers.{layer}.mixer.experts.{expert}.{projection}_proj"
                )
                tensors[name + ".weight"] = blocks(e=1, n=n, k=k)[0]
                recipes[name] = dict(
                    quant_algo="IQ2_XS",
                    packing="ggml",
                    group_size=256,
                    block_payload_bytes=74,
                )
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(dict(quantization=dict(quantized_layers=recipes)))
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=dict.fromkeys(tensors, "model.safetensors")))
    )
    profile = benchmark.MODEL_PROFILES["puzzle3-iq2-xs"]
    assert profile.hf_repo_id is None
    for layer, width, top_k in ((1, 512, 2), (2, 1024, 1)):
        spec = benchmark.build_model_spec(
            tmp_path, profile, layer_idx=layer, tp_size_override=2, tp_rank=1
        )
        assert (spec.hidden_size, spec.I_tp, spec.top_k) == (256, width // 2, top_k)
        weights = benchmark.load_iq2_xs_expert_weights(
            tmp_path, spec, layer_idx=layer, activation="relu2", device="cpu"
        )
        assert weights.w13_weight.shape == (3, width // 2, 1, 74)
        assert weights.w2_weight.shape == (3, 256, width // 512, 74)
        assert torch.equal(
            weights.w13_weight[0],
            tensors[f"backbone.layers.{layer}.mixer.experts.0.up_proj.weight"][
                width // 2 :
            ],
        )
        params = benchmark.get_quant_mode_params(weights, "per-expert", "w4a16")
        output = benchmark.make_oracle_reference(
            "w4a16",
            "w4a16",
            torch.full((2, 256), 0.03125, dtype=torch.bfloat16),
            weights,
            params,
            torch.arange(top_k, dtype=torch.int32).expand(2, -1),
            torch.full((2, top_k), 1 / top_k),
            activation="relu2",
        )
        assert (
            output.shape == (2, 256)
            and torch.isfinite(output).all()
            and torch.count_nonzero(output)
        )
    with pytest.raises(ValueError, match="MoE layer"):
        benchmark.build_model_spec(tmp_path, profile, layer_idx=0)


@pytest.fixture
def snapshot(tmp_path):
    return make_snapshot(tmp_path)


def test_benchmark_profile_and_checkpoint_adapter(snapshot, monkeypatch):
    profile = benchmark.MODEL_PROFILES["qwen36-35b-iq2-xs"]
    assert profile.hf_repo_id is None
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
