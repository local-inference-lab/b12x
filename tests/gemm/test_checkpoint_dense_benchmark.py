import json

import pytest
from safetensors.torch import save_file
import torch

from benchmarks.checkpoint_dense import checkpoint_cases


def write_checkpoint(root, tensors, recipes):
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    (root / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {"quantized_layers": recipes},
            }
        )
    )


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs"])
def test_checkpoint_cases_exclude_routed_and_preserve_heterogeneous_dense_roles(
    tmp_path, codec,
):
    block_bytes = 66 if codec == "iq2_xxs" else 74
    iq2 = dict(
        quant_algo=codec.upper(), group_size=256, block_payload_bytes=block_bytes, packing="ggml"
    )
    tensors, recipes = {}, {}
    for name, n in (
        ("backbone.layers.1.mixer.shared_experts.up_proj", 8),
        ("backbone.layers.3.mixer.shared_experts.up_proj", 8),
        ("backbone.layers.5.mixer.shared_experts.up_proj", 16),
        ("backbone.layers.1.mixer.experts.0.up_proj", 8),
    ):
        tensors[name + ".weight"] = torch.zeros(n, 1, block_bytes, dtype=torch.uint8)
        recipes[name] = iq2
    dense = "backbone.layers.0.mixer.in_proj"
    tensors[dense + ".weight"] = torch.zeros(8, 128, dtype=torch.uint8)
    recipes[dense] = dict(quant_algo="W4A16_NVFP4", group_size=16)
    tensors["lm_head.weight"] = torch.zeros(16, 256, dtype=torch.bfloat16)
    write_checkpoint(tmp_path, tensors, recipes)
    _, cases = checkpoint_cases(tmp_path)
    assert sorted(
        (c["recipe"], c["n"], c["k"], len(c["equivalent_weights"])) for c in cases
    ) == [
        (codec, 8, 256, 2),
        (codec, 16, 256, 1),
        ("nvfp4", 8, 256, 1),
    ]
    assert all(
        ".experts." not in c["weight"] and c["weight"] != "lm_head.weight"
        for c in cases
    )


def test_checkpoint_cases_reject_wrong_iq2_block_contract(tmp_path):
    name = "backbone.layers.1.mixer.shared_experts.up_proj"
    write_checkpoint(
        tmp_path,
        {name + ".weight": torch.zeros(8, 1, 74, dtype=torch.uint8)},
        {
            name: dict(
                quant_algo="IQ2_XS",
                group_size=256,
                block_payload_bytes=66,
                packing="ggml",
            ),
        },
    )
    with pytest.raises(ValueError, match="unsupported IQ2_XS payload"):
        checkpoint_cases(tmp_path)
