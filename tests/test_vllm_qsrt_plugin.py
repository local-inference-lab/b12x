from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from b12x.integration.vllm.qsrt_plugin import (
    BITS,
    DESCRIPTOR_SHA256,
    QUALITY_SELECTION_FILENAME,
    QUALITY_SELECTION_KIND,
    QUANT_NAME,
    RANK,
    _capture_sizes,
    _norm_key,
    _parameter_loader,
    _projection_group,
    _require_cudagraphs_disabled,
    _select_linear_method,
    _sha256,
    _validate_quality_selection_receipt,
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


def test_linear_method_selection_preserves_unclaimed_bf16_modules() -> None:
    modules = set(_validate_quantization_config(_config()))
    packed = object()
    unquantized = object()
    assert (
        _select_linear_method(
            "language_model.model.layers.7.mlp.gate_up_proj",
            modules,
            packed_method=packed,
            unquantized_method=unquantized,
        )
        is packed
    )
    assert (
        _select_linear_method(
            "visual.merger.linear_fc1",
            modules,
            packed_method=packed,
            unquantized_method=unquantized,
        )
        is unquantized
    )


def test_quantization_config_rejects_a_short_module_inventory() -> None:
    config = _config()
    config["modules"] = config["modules"][:-1]
    with pytest.raises(ValueError, match="incompatible"):
        _validate_quantization_config(config)


def test_quality_selection_receipt_binds_the_selected_overlay(tmp_path: Path) -> None:
    report = {
        "kind": QUALITY_SELECTION_KIND,
        "schema_version": 1,
        "status": "implemented",
        "classification": "research-only",
        "source_recovery": {
            "complete_report_sha256": "a" * 64,
            "adapter_manifest_sha256": "b" * 64,
        },
        "selected_overlay": {
            "sha256": "c" * 64,
            "step": 100,
            "rank": RANK,
            "variant": "weighted",
            "tensor_count": 384,
        },
        "selection_contract": {
            "metric": "strict-overlap-filtered token mean KLD",
            "partition": "analysis",
            "direction": "lower",
        },
    }
    path = tmp_path / QUALITY_SELECTION_FILENAME
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    digest = _sha256(path)
    config = {
        "quality_selection_report": QUALITY_SELECTION_FILENAME,
        "quality_selection_report_sha256": digest,
    }
    manifest = {
        "build": {
            "quality_selection_report_sha256": digest,
            "recovery_report_sha256": "a" * 64,
            "adapter_manifest_sha256": "b" * 64,
            "recovery_overlay_sha256": "c" * 64,
        },
        "quality_selection": {
            "status": "implemented",
            "classification": "research-only",
            "file": QUALITY_SELECTION_FILENAME,
            "sha256": digest,
            "selected_step": 100,
            "selected_overlay_sha256": "c" * 64,
        },
    }

    _validate_quality_selection_receipt(tmp_path, manifest, config)
    path.write_text('{"status": "tampered"}\n')
    with pytest.raises(ValueError, match="content differs"):
        _validate_quality_selection_receipt(tmp_path, manifest, config)


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


@pytest.mark.parametrize("mode", [0, SimpleNamespace(name="NONE"), "NONE"])
def test_cuda_graph_contract_allows_disabled_modes(mode: object) -> None:
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode)
    )
    _require_cudagraphs_disabled(config)


@pytest.mark.parametrize(
    "mode",
    [None, SimpleNamespace(name="FULL_AND_PIECEWISE"), "PIECEWISE"],
)
def test_cuda_graph_contract_rejects_graph_replay(mode: object) -> None:
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode)
    )
    with pytest.raises(ValueError, match="requires CUDA graphs to be disabled"):
        _require_cudagraphs_disabled(config)
