from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from benchmarks.mhc_profiles import MODEL_PROFILES, load_mhc_profile


def test_vllm_import_preserves_benchmark_discovery_for_spawn(monkeypatch, tmp_path):
    """Compiler children must resolve b12x's benchmarks after importing vLLM."""
    from benchmarks import vllm_mhc

    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    package = SimpleNamespace(__file__=str(tmp_path / "vllm" / "__init__.py"))
    monkeypatch.setattr(vllm_mhc.importlib, "import_module", lambda name: package)
    original = list(sys.path)
    assert vllm_mhc._import_vllm(tmp_path) is package
    assert sys.path == original


def _config(profile_name: str, hidden_size: int) -> dict[str, object]:
    text = {
        "model_type": "deepseek_v41_text",
        "hidden_size": hidden_size,
        "num_hidden_layers": 5,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
    }
    if profile_name == "deepseek-v4.1-flash":
        return {"model_type": "deepseek_v41", "text_config": text}
    text["model_type"] = "deepseek_v4"
    return text


def _checkpoint(
    root: Path,
    profile_name: str,
    *,
    hidden_size: int = 8,
    override: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    root.mkdir()
    layer = 3
    prefix = f"layers.{layer}"
    tensors = {
        f"{prefix}.hc_ffn_fn": torch.arange(24 * 4 * hidden_size, dtype=torch.float32).reshape(24, -1),
        f"{prefix}.hc_ffn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_ffn_base": torch.arange(24, dtype=torch.float32),
        f"{prefix}.hc_attn_fn": torch.full((24, 4 * hidden_size), 11.0),
        f"{prefix}.hc_attn_scale": torch.full((3,), 12.0),
        f"{prefix}.hc_attn_base": torch.full((24,), 13.0),
        f"{prefix}.ffn_norm.weight": torch.arange(hidden_size, dtype=torch.bfloat16),
    }
    if override:
        tensors.update(override)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}}), encoding="utf-8"
    )
    (root / "config.json").write_text(
        json.dumps(_config(profile_name, hidden_size)), encoding="utf-8"
    )
    return tensors


@pytest.mark.parametrize("profile_name", tuple(MODEL_PROFILES))
def test_loads_native_cpu_mhc_parameters(profile_name: str, tmp_path: Path) -> None:
    expected = _checkpoint(tmp_path / "checkpoint", profile_name)

    bundle = load_mhc_profile(profile_name, tmp_path / "checkpoint")

    assert bundle.tensors["fn"].device.type == "cpu"
    assert bundle.tensors["fn"].dtype == torch.float32
    assert bundle.tensors["fn"].shape == (24, 32)
    assert bundle.tensors["norm_weight"].dtype == torch.bfloat16
    assert torch.equal(bundle.tensors["fn"], expected["layers.3.hc_ffn_fn"])
    assert torch.equal(bundle.tensors["prev_fn"], expected["layers.3.hc_attn_fn"])


def test_v41_profile_rejects_root_or_nested_wrong_model_identity(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    _checkpoint(root, "deepseek-v4.1-flash")
    config = _config("deepseek-v4.1-flash", 8)
    config["text_config"]["model_type"] = "deepseek_v4"  # type: ignore[index]
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError):
        load_mhc_profile("deepseek-v4.1-flash", root)


def test_profile_rejects_non_native_weight_abi(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    _checkpoint(
        root,
        "deepseek-v4-flash",
        override={"layers.3.hc_ffn_fn": torch.ones((24, 31), dtype=torch.float32)},
    )

    with pytest.raises(ValueError):
        load_mhc_profile("deepseek-v4-flash", root)


def test_profile_rejects_non_native_weight_dtype(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    _checkpoint(
        root,
        "deepseek-v4-flash",
        override={"layers.3.ffn_norm.weight": torch.ones(8, dtype=torch.float32)},
    )

    with pytest.raises(ValueError):
        load_mhc_profile("deepseek-v4-flash", root)
