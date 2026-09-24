"""Safetensors slicing, expert mapping and cached-checkpoint byte preservation."""

import json
import os
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from benchmarks.iq2_xs_checkpoint import load_iq2_xs_layer
from b12x.moe._shared.kernels.w4a16.iq2_xs import pack_iq2_xs_matrix
from b12x.testing.iq2_xs_reference import dequantize_blocks
from .test_iq2_xs import blocks, unpack_planes


def make_snapshot(tmp_path, codec="iq2_xs"):
    prefix = "model.language_model.layers.0.mlp.experts"
    tensors = {}
    for e in range(3):
        for name in ("gate", "up", "down"):
            n, k = (256, 512) if name == "down" else (512, 256)
            tensors[f"{prefix}.{e}.{name}_proj.weight"] = blocks(e=1, n=n, k=k, codec=codec)[0]
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(tensors, "model.safetensors"),
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 256,
                    "moe_intermediate_size": 512,
                    "num_experts": 3,
                    "num_experts_per_tok": 2,
                    "num_hidden_layers": 1,
                }
            }
        )
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "quantized_layers": {
                        prefix: {
                            "quant_algo": codec.upper(),
                            "packing": "ggml",
                            "group_size": 256,
                            "block_payload_bytes": 66 if codec == "iq2_xxs" else 74,
                        }
                    },
                }
            }
        )
    )
    return tmp_path


@pytest.fixture(params=["iq2_xs", "iq2_xxs"])
def snapshot(tmp_path, request):
    return make_snapshot(tmp_path, codec=request.param)


def test_tp_reconstruction_and_expert_map(snapshot):
    whole = load_iq2_xs_layer(snapshot, layer=0, expert_ids=(2, 0))
    ranks = [
        load_iq2_xs_layer(snapshot, layer=0, tp_size=2, tp_rank=r, expert_ids=(2, 0))
        for r in (0, 1)
    ]
    for projection in (0, 1):
        joined = torch.cat(
            [
                r.weights.w13[:, projection * 256 : (projection + 1) * 256]
                for r in ranks
            ],
            1,
        )
        assert torch.equal(
            joined, whole.weights.w13[:, projection * 512 : (projection + 1) * 512]
        )
    assert torch.equal(torch.cat([r.weights.w2 for r in ranks], 2), whole.weights.w2)
    assert whole.expert_map("cpu").tolist() == [1, -1, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp_size": 4},
        {"tp_rank": 1},
        {"expert_ids": (0, 0)},
        {"expert_ids": (3,)},
        {"layer": 1},
    ],
)
def test_invalid_selection(snapshot, kwargs):
    with pytest.raises(ValueError):
        load_iq2_xs_layer(snapshot, **{"layer": 0, **kwargs})


def test_super3_nested_config_global_latent_size_and_layer_prefix(tmp_path):
    prefix = "language_model.model.layers.1.mixer.experts"
    w1, w2 = blocks(e=1, n=512, k=256, codec="iq2_xxs")[0], blocks(e=1, n=256, k=512, codec="iq2_xxs")[0]
    tensors = {f"{prefix}.0.up_proj.weight": w1, f"{prefix}.0.down_proj.weight": w2}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    ))
    (tmp_path / "config.json").write_text(json.dumps({"llm_config": {
        "model_type": "nemotron_h_puzzle", "moe_latent_size": 256,
        "mlp_hidden_act": "relu2", "block_configs": [
            {"block_type": "mamba"}, {"block_type": "moe", "moe_intermediate_size": 512,
                                    "n_routed_experts": 1, "num_experts_per_tok": 1},
        ],
    }}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quantized_layers": {
        prefix: {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml", "block_payload_bytes": 66},
    }}}))
    layer = load_iq2_xs_layer(tmp_path, layer=1)
    assert (layer.hidden_size, layer.intermediate_size, layer.activation) == (256, 512, "relu2")
    assert torch.equal(layer.weights.w13[0], w1)
    assert torch.equal(layer.weights.w2[0], w2)


@pytest.mark.parametrize("layer", [0, 20, 39])
def test_cached_projection_reconstruction(layer):
    path = os.environ.get("IQ2_XS_CHECKPOINT")
    if path is None:
        pytest.skip("set IQ2_XS_CHECKPOINT to the local safetensors snapshot")
    assert Path(path).is_dir()
    loaded = load_iq2_xs_layer(path, layer=layer, expert_ids=(0, 127, 255))
    assert (
        loaded.hidden_size,
        loaded.intermediate_size,
        loaded.route_num_experts,
        loaded.top_k,
    ) == (2048, 512, 256, 8)
    for source in (loaded.weights.w13, loaded.weights.w2):
        words, metadata = pack_iq2_xs_matrix(source)
        restored = unpack_planes(words, metadata, source.shape)
        assert torch.equal(restored, source)
        # Bound oracle memory to one expert's projection.
        for actual, expected in zip(restored, source, strict=True):
            decoded = dequantize_blocks(actual)
            assert torch.isfinite(decoded).all() and torch.count_nonzero(decoded) > 0
            assert torch.equal(decoded, dequantize_blocks(expected))
