from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from b12x.integration.vllm.qsrt_plugin import (
    BITS,
    DESCRIPTOR_SHA256,
    QUANT_NAME,
    RANK,
    _capture_sizes,
    _norm_key,
    _parameter_loader,
    _projection_group,
    _validate_quantization_config,
    _workspace_capacity_rows,
)


def _config() -> dict[str, object]:
    return {
        "quant_method": QUANT_NAME,
        "format": "qwen38_dense_mlp_qsrt_k5_rank16",
        "bits": BITS,
        "codebook": "sqg_fp16",
        "descriptor_sha256": DESCRIPTOR_SHA256,
        "rank": RANK,
        "tensor_parallel_size": 1,
        "workspace_capacity_rows": 8192,
        "modules": [
            f"model.language_model.layers.{layer}.mlp.{projection}"
            for layer in range(64)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ],
    }


def test_quantization_config_claims_only_complete_decoder_mlps() -> None:
    modules = set(_validate_quantization_config(_config()))
    assert len(modules) == 192
    assert _norm_key("language_model.model.layers.7.mlp.down_proj") in modules
    assert (
        _projection_group(
            "language_model.model.layers.7.mlp.gate_up_proj",
            modules,
        )
        == "gate_up"
    )
    assert (
        _projection_group(
            "language_model.model.layers.7.self_attn.qkv_proj",
            modules,
        )
        is None
    )


def test_quantization_config_rejects_a_short_module_inventory() -> None:
    config = _config()
    config["modules"] = config["modules"][:-1]
    with pytest.raises(ValueError, match="incompatible"):
        _validate_quantization_config(config)


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5])
def test_workspace_capacity_requires_a_positive_integer(value: object) -> None:
    config = _config()
    config["workspace_capacity_rows"] = value
    with pytest.raises(ValueError, match="workspace_capacity_rows"):
        _workspace_capacity_rows(config)


def test_stacked_payload_loader_is_exactly_once_and_dtype_strict() -> None:
    loaded: set[tuple[str, int]] = set()
    loader = _parameter_loader(
        component="trellis",
        stacked=True,
        loaded_slots=loaded,
    )
    parameter = torch.nn.Parameter(
        torch.zeros((2, 2, 3), dtype=torch.int16),
        requires_grad=False,
    )
    value = torch.arange(6, dtype=torch.int16).reshape(2, 3)
    loader(parameter, value, 1)
    assert torch.equal(parameter[1], value)
    assert loaded == {("trellis", 1)}
    with pytest.raises(ValueError, match="loaded twice"):
        loader(parameter, value, 1)
    with pytest.raises(TypeError, match="must use"):
        loader(parameter, value.float(), 0)


def _install_fake_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
    sizes: list[int],
) -> None:
    package = ModuleType("vllm")
    config = ModuleType("vllm.config")
    config.get_current_vllm_config = lambda: SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes)
    )
    package.config = config
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.config", config)


def test_capture_sizes_include_vllm_graph_geometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_config(monkeypatch, [32, 64, 128])
    assert _capture_sizes(256) == (1, 2, 4, 8, 16, 32, 64, 128)


def test_capture_sizes_reject_rows_outside_packed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_config(monkeypatch, [512, 1024])
    with pytest.raises(NotImplementedError, match="capacity 512"):
        _capture_sizes(512)
