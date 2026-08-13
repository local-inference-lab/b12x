from __future__ import annotations

import json
import pathlib

import pytest
import torch

from b12x.quantization.mxfp6.model_fp6 import (
    SafetensorsModel,
    convert_dense_model_to_fp6,
    convert_moe_model_to_fp6,
    discover_dense_linears,
    discover_moe_experts,
)
from b12x.quantization.mxfp6.fp6_safetensors_export import (
    export_dense_model_to_fp6_safetensors,
    export_moe_model_to_fp6_safetensors,
)

safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_ckpt(path: pathlib.Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors")
    )
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))


def _fake_moe(path: pathlib.Path, *, e: int, layers: int, k: int, n: int) -> None:
    t: dict[str, torch.Tensor] = {}
    pre = "model.language_model.layers.{L}.mlp"
    for L in range(layers):
        base = pre.format(L=L)
        for ei in range(e):
            t[f"{base}.experts.{ei}.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
        # decoys discovery must ignore:
        t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{L}.linear_attn.in_proj.weight"] = torch.randn(k, k, dtype=torch.bfloat16)
    t["visual.blocks.0.mlp.fc1.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
    _write_ckpt(path, t)


def _fake_moe_packed(path: pathlib.Path, *, e: int, layers: int, k: int, n: int) -> None:
    """Experts stacked in one 3-D tensor: experts.gate_up_proj / experts.down_proj."""
    t: dict[str, torch.Tensor] = {}
    pre = "model.language_model.layers.{L}.mlp"
    for L in range(layers):
        base = pre.format(L=L)
        t[f"{base}.experts.gate_up_proj"] = torch.randn(e, 2 * n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.experts.down_proj"] = torch.randn(e, k, n, dtype=torch.bfloat16) * 0.1
        t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{L}.input_layernorm.weight"] = torch.randn(k, dtype=torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = torch.randn(16, k, dtype=torch.bfloat16)
    _write_ckpt(path, t)


def _fake_dense(path: pathlib.Path, *, layers: int, k: int, n: int, attn_on: set[int]) -> None:
    t: dict[str, torch.Tensor] = {}
    for L in range(layers):
        base = f"model.language_model.layers.{L}"
        t[f"{base}.mlp.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
        if L in attn_on:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                t[f"{base}.self_attn.{p}.weight"] = torch.randn(k, k, dtype=torch.bfloat16) * 0.1
    _write_ckpt(path, t)


def test_discover_moe_experts(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=4, layers=2, k=64, n=32)
    model = SafetensorsModel(tmp_path / "m")
    scheme = discover_moe_experts(model)
    assert scheme is not None
    assert scheme.num_experts == 4
    assert scheme.layers == [0, 1]
    assert scheme.gate_name == "gate_proj" and scheme.up_name == "up_proj"
    assert scheme.down_name == "down_proj" and not scheme.is_fused
    assert scheme.prefix_template == "model.language_model.layers.{L}.mlp"
    assert scheme.expert_key(1, 3, "down_proj") == (
        "model.language_model.layers.1.mlp.experts.3.down_proj.weight"
    )


def test_moe_dryrun_writes_nothing(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=4, layers=2, k=64, n=32)
    report = convert_moe_model_to_fp6(
        tmp_path / "m", tmp_path / "out", dry_run=True, device="cpu", verbose=False
    )
    assert report.arch == "moe"
    assert report.layers == [0, 1]
    assert report.tensors_written == 0
    assert not (tmp_path / "out").exists()


def test_moe_convert_cpu_roundtrip(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    report = convert_moe_model_to_fp6(
        tmp_path / "m",
        tmp_path / "out",
        limit_layers=1,
        device="cpu",
        use_gpu=False,
        verbose=False,
    )
    assert report.tensors_written == 2
    assert (tmp_path / "out" / "manifest.json").is_file()
    art = tmp_path / "out" / "layer_0.moe_fp6.safetensors"
    assert art.is_file()

    from b12x.quantization.mxfp6 import load_fp6_moe_weights

    w = load_fp6_moe_weights(str(art), device="cpu")
    assert w.num_experts == 2 and w.k == 64 and w.n == 32
    assert w.w1_fp6.shape == (2, 64, 48)  # (E, 2N, 3K/4)
    assert w.w2_fp6.shape == (2, 64, 24)  # (E, K, 3N/4)


def test_discover_dense_and_dryrun(tmp_path) -> None:
    _fake_dense(tmp_path / "d", layers=2, k=64, n=128, attn_on={0})
    model = SafetensorsModel(tmp_path / "d")
    scheme = discover_dense_linears(model)
    assert scheme is not None
    assert scheme.layers == [0, 1]
    assert scheme.mlp_gate == "mlp.gate_proj" and scheme.mlp_down == "mlp.down_proj"
    assert scheme.mlp_fused_gate_up is None
    assert scheme.attn_projs == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]

    report = convert_dense_model_to_fp6(
        tmp_path / "d", tmp_path / "out_d", dry_run=True, device="cpu", verbose=False
    )
    assert report.arch == "dense"
    assert report.tensors_written == 0
    assert not (tmp_path / "out_d").exists()


def _read_ckpt(out: pathlib.Path) -> tuple[dict, dict, dict]:
    """Return (state_dict, index, config) for an exported FP6 safetensors dir."""
    index = json.loads((out / "model.safetensors.index.json").read_text())
    config = json.loads((out / "config.json").read_text())
    state: dict[str, torch.Tensor] = {}
    for shard in set(index["weight_map"].values()):
        state.update(safetensors_torch.load_file(str(out / shard)))
    return state, index, config


def test_export_moe_per_expert_safetensors(tmp_path) -> None:
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    out = tmp_path / "out"
    report = export_moe_model_to_fp6_safetensors(
        tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    assert report.arch == "moe"
    # 2 experts x 3 projections (gate/up/down) = 6 quantized linears.
    assert report.quantized_tensors == 6
    state, index, config = _read_ckpt(out)

    base = "model.language_model.layers.0.mlp.experts"
    for proj, (o, i) in {"gate_proj": (32, 64), "up_proj": (32, 64), "down_proj": (64, 32)}.items():
        w = state[f"{base}.0.{proj}.weight"]
        sc = state[f"{base}.0.{proj}.weight_scale"]
        assert w.dtype == torch.uint8 and tuple(w.shape) == (o, i * 3 // 4)
        assert sc.dtype == torch.uint8 and tuple(sc.shape) == (o, i // 32)
        assert f"{base}.0.{proj}.weight_scale_2" in state
        assert f"{base}.0.{proj}.input_scale" in state
    # Non-quantized tensors are copied through (router gate, decoy linear-attn).
    assert "model.language_model.layers.0.mlp.gate.weight" in state
    assert "model.language_model.layers.0.linear_attn.in_proj.weight" in state
    # quantization_config present with the W6A6 contract.
    qc = config["quantization_config"]
    assert qc["quant_method"] == "modelopt" and qc["quant_algo"] == "W6A6"
    assert qc["group_size"] == 32 and qc["exclude_modules"]


def test_export_moe_packed_expands_to_per_expert(tmp_path) -> None:
    _fake_moe_packed(tmp_path / "mp", e=2, layers=1, k=64, n=32)
    out = tmp_path / "outp"
    report = export_moe_model_to_fp6_safetensors(
        tmp_path / "mp", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    assert report.quantized_tensors == 6  # packed source -> per-expert keys
    state, _, _ = _read_ckpt(out)
    base = "model.language_model.layers.0.mlp.experts"
    for e in (0, 1):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"{base}.{e}.{proj}.weight" in state
    # Original packed keys must be gone (replaced by per-expert).
    assert f"{base}.gate_up_proj" not in state
    assert f"{base}.down_proj" not in state
    # Embedding copied through.
    assert "model.language_model.embed_tokens.weight" in state


def test_load_fp6_moe_checkpoint_cpu(tmp_path) -> None:
    """Export -> load round-trip: loader reconstructs FP6MoEWeights from the keys."""
    _fake_moe(tmp_path / "m", e=2, layers=1, k=64, n=32)
    out = tmp_path / "out"
    export_moe_model_to_fp6_safetensors(
        tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False
    )
    from b12x.quantization.mxfp6.fp6_safetensors_load import load_fp6_moe_checkpoint

    layers = load_fp6_moe_checkpoint(str(out), device="cpu")
    assert set(layers) == {0}
    w = layers[0]
    assert w.num_experts == 2 and w.k == 64 and w.n == 32
    assert w.w1_fp6.shape == (2, 64, 48)  # (E, 2N, 3K/4)
    assert w.w2_fp6.shape == (2, 64, 24)  # (E, K, 3N/4)
    # Export default has been mxfp6_w6a8 (E2M3 weights, E4M3 activations)
    # since Jun 13; the loader must report it back verbatim.
    assert w.source_format == "mxfp6_w6a8" and w.weight_fmt == "e2m3"


def test_load_fp6_dense_checkpoint_cpu(tmp_path) -> None:
    """Export -> load round-trip for dense linears (loader reconstructs FP6DenseWeight)."""
    _fake_dense(tmp_path / "d", layers=1, k=128, n=128, attn_on={0})
    out = tmp_path / "outd2"
    export_dense_model_to_fp6_safetensors(
        tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False
    )
    from b12x.quantization.mxfp6.fp6_safetensors_load import load_fp6_dense_checkpoint

    weights = load_fp6_dense_checkpoint(str(out), device="cpu")
    name = "model.language_model.layers.0.mlp.gate_proj"
    assert name in weights
    w = weights[name]
    assert w.out_features == 128 and w.in_features == 128
    assert w.packed.shape == (128, 96)  # (out, 3*in/4)
    assert float(w.global_scale.reshape(-1)[0]) == 1.0
    # Attention projection also quantized + loadable.
    assert "model.language_model.layers.0.self_attn.q_proj" in weights


def test_export_dense_safetensors(tmp_path) -> None:
    _fake_dense(tmp_path / "d", layers=2, k=128, n=128, attn_on={0})
    out = tmp_path / "outd"
    report = export_dense_model_to_fp6_safetensors(
        tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False
    )
    assert report.arch == "dense"
    state, _, config = _read_ckpt(out)
    w = state["model.language_model.layers.0.mlp.gate_proj.weight"]
    assert w.dtype == torch.uint8 and tuple(w.shape) == (128, 128 * 3 // 4)
    assert "model.language_model.layers.0.mlp.gate_proj.weight_scale" in state
    assert config["quantization_config"]["quant_algo"] == "W6A6"


# ---------------------------------------------------------------------------
# Issue #171: safetensors shard path containment contract tests
# ---------------------------------------------------------------------------


def _save_safetensors(path: pathlib.Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write a single safetensors file (no config, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path)
    )


def _make_index(
    root: pathlib.Path, weight_map: dict[str, str], *, with_config: bool = True
) -> None:
    """Write a model.safetensors.index.json with an arbitrary weight_map."""
    root.mkdir(parents=True, exist_ok=True)
    if with_config:
        (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )


# --- accepted layouts -------------------------------------------------------


def test_accept_regular_in_root_shard(tmp_path) -> None:
    """A normal basename index loads its tensor."""
    root = tmp_path / "model"
    shard = root / "model-00001-of-00001.safetensors"
    sentinel = torch.tensor([1.0, 2.0, 3.0])
    _save_safetensors(shard, {"weight": sentinel})
    _make_index(root, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(root)
    assert torch.equal(model.get_tensor("weight"), sentinel)


def test_accept_in_root_symlink(tmp_path) -> None:
    """An in-root basename symlink whose target stays inside root loads."""
    root = tmp_path / "model"
    real = root / "real-00001.safetensors"
    sentinel = torch.tensor([5.0])
    _save_safetensors(real, {"weight": sentinel})
    (root / "model-00001-of-00001.safetensors").symlink_to(real.name)
    _make_index(root, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(root)
    assert torch.equal(model.get_tensor("weight"), sentinel)


def test_accept_symlinked_root(tmp_path) -> None:
    """A symlinked model root is canonicalized and loads normally."""
    real_root = tmp_path / "real_model"
    shard = real_root / "model-00001-of-00001.safetensors"
    sentinel = torch.tensor([8.0])
    _save_safetensors(shard, {"weight": sentinel})
    _make_index(real_root, {"weight": "model-00001-of-00001.safetensors"})
    link = tmp_path / "link_to_model"
    link.symlink_to(real_root)
    model = SafetensorsModel(link)
    assert torch.equal(model.get_tensor("weight"), sentinel)


def test_accept_standard_hf_same_repo_blob_link(tmp_path) -> None:
    """Synthetic models--repo/snapshots/rev/shard -> ../../blobs/digest loads."""
    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    digest = "deadbeef"
    sentinel = torch.tensor([42.0])
    _save_safetensors(blobs / digest, {"weight": sentinel})
    # Standard HF layout: snapshot basename -> ../../blobs/<digest>
    (snap / "model-00001-of-00001.safetensors").symlink_to(
        pathlib.Path("..", "..", "blobs", digest)
    )
    _make_index(snap, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(snap)
    assert torch.equal(model.get_tensor("weight"), sentinel)


def test_accept_single_file_fallback(tmp_path) -> None:
    """Plain model.safetensors (no index) still loads."""
    root = tmp_path / "model"
    root.mkdir(parents=True, exist_ok=True)
    sentinel = torch.tensor([7.0])
    _save_safetensors(root / "model.safetensors", {"weight": sentinel})
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    model = SafetensorsModel(root)
    assert torch.equal(model.get_tensor("weight"), sentinel)


def test_accept_single_file_hf_blob_link(tmp_path) -> None:
    """Single-file fallback with a recognized HF same-repo blob link loads."""
    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    digest = "cafef00d"
    sentinel = torch.tensor([99.0])
    _save_safetensors(blobs / digest, {"weight": sentinel})
    (snap / "model.safetensors").symlink_to(
        pathlib.Path("..", "..", "blobs", digest)
    )
    (snap / "config.json").write_text(json.dumps({"model_type": "test"}))
    model = SafetensorsModel(snap)
    assert torch.equal(model.get_tensor("weight"), sentinel)


# --- lexical rejections (fail at construction, before safe_open) -----------


def test_reject_posix_absolute_shard(tmp_path) -> None:
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    _make_index(root, {"weight": "/etc/passwd.safetensors"})
    with pytest.raises(ValueError, match="invalid shard name"):
        SafetensorsModel(root)


def test_reject_windows_absolute_unc_and_backslash_shards(tmp_path) -> None:
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    for bad in [
        r"C:\Users\secret.safetensors",
        r"\\server\share\secret.safetensors",
        r"subdir\file.safetensors",
    ]:
        _make_index(root, {"weight": bad})
        with pytest.raises(ValueError, match="invalid shard name"):
            SafetensorsModel(root)


def test_reject_windows_drive_separator_free(tmp_path) -> None:
    """Separator-free ``C:`` and ``1:`` drive forms are rejected (ntpath rule)."""
    root = tmp_path / "model"
    _save_safetensors(root / "model.safetensors", {"weight": torch.tensor([1.0])})
    for bad in ["C:model.safetensors", "1:model.safetensors"]:
        _make_index(root, {"weight": bad})
        with pytest.raises(ValueError, match="Windows drive form"):
            SafetensorsModel(root)


def test_reject_parent_traversal_shard(tmp_path) -> None:
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    for bad in ["../outside.safetensors", "../../etc/passwd.safetensors"]:
        _make_index(root, {"weight": bad})
        with pytest.raises(ValueError, match="invalid shard name"):
            SafetensorsModel(root)


def test_reject_nested_component(tmp_path) -> None:
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    _make_index(root, {"weight": "subdir/shard.safetensors"})
    with pytest.raises(ValueError, match="invalid shard name"):
        SafetensorsModel(root)


def test_reject_non_safetensors_suffix(tmp_path) -> None:
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    _make_index(root, {"weight": "model.bin"})
    with pytest.raises(ValueError, match="invalid shard name"):
        SafetensorsModel(root)


# --- type / structure rejections --------------------------------------------


def test_reject_non_dict_weight_map(tmp_path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": [1, 2, 3]})
    )
    with pytest.raises(ValueError, match="weight_map.*not a JSON object"):
        SafetensorsModel(root)


def test_reject_malformed_weight_map_types_eagerly(tmp_path) -> None:
    """Non-string shard values (int, list, null) fail at construction."""
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    for bad_val in [123, ["model.safetensors"], None, True]:
        _make_index(root, {"weight": bad_val})
        with pytest.raises(ValueError, match="expected string"):
            SafetensorsModel(root)


def test_reject_bad_map_entry_eagerly_even_if_unused(tmp_path) -> None:
    """A lexically invalid shard in an unused entry still fails at construction."""
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"good": torch.tensor([1.0])})
    _make_index(
        root,
        {"good": "model.safetensors", "bad": "../escape.safetensors"},
    )
    with pytest.raises(ValueError, match="invalid shard name"):
        SafetensorsModel(root)


# --- symlink / resolution rejections (lazy — triggered on get_tensor) -------


def test_reject_external_leaf_symlink(tmp_path) -> None:
    """An in-root basename symlink to an external safetensors file fails."""
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    _save_safetensors(outside, {"weight": torch.tensor([1.0])})
    (root / "model-00001-of-00001.safetensors").symlink_to(outside)
    _make_index(root, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(root)
    with pytest.raises(ValueError, match="outside the model root"):
        model.get_tensor("weight")


def test_reject_hf_cross_repo_blob_link(tmp_path) -> None:
    """A snapshot basename link resolving outside the same repo blobs fails."""
    repo_a = tmp_path / "models--org--modelA"
    repo_b = tmp_path / "models--org--modelB"
    blobs_a = repo_a / "blobs"
    blobs_b = repo_b / "blobs"
    snap_a = repo_a / "snapshots" / "rev"
    blobs_a.mkdir(parents=True)
    blobs_b.mkdir(parents=True)
    snap_a.mkdir(parents=True)
    digest = "cross1234"
    _save_safetensors(blobs_b / digest, {"weight": torch.tensor([1.0])})
    # Link points to a *different* repo's blobs — outside same-repo boundary.
    (snap_a / "model-00001-of-00001.safetensors").symlink_to(blobs_b / digest)
    _make_index(snap_a, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(snap_a)
    with pytest.raises(ValueError, match="outside the model root"):
        model.get_tensor("weight")


def test_reject_broken_symlink(tmp_path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "model-00001-of-00001.safetensors").symlink_to(
        tmp_path / "nonexistent.safetensors"
    )
    _make_index(root, {"weight": "model-00001-of-00001.safetensors"})
    model = SafetensorsModel(root)
    with pytest.raises(ValueError, match="not found|not a regular file"):
        model.get_tensor("weight")


def test_reject_directory_as_shard(tmp_path) -> None:
    root = tmp_path / "model"
    (root / "subdir.safetensors").mkdir(parents=True)
    _make_index(root, {"weight": "subdir.safetensors"})
    model = SafetensorsModel(root)
    with pytest.raises(ValueError, match="not a regular file"):
        model.get_tensor("weight")


def test_reject_missing_shard_file(tmp_path) -> None:
    root = tmp_path / "model"
    _make_index(root, {"weight": "nonexistent-00001-of-00001.safetensors"})
    model = SafetensorsModel(root)
    with pytest.raises(ValueError, match="not found"):
        model.get_tensor("weight")


# --- single-file fallback symlink escape (construction-time) ----------------


def test_confine_single_file_fallback_symlink_escape(tmp_path) -> None:
    """A model.safetensors symlink escape receives the same rejection."""
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    _save_safetensors(outside, {"weight": torch.tensor([1.0])})
    (root / "model.safetensors").symlink_to(outside)
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    with pytest.raises(ValueError, match="outside the model root"):
        SafetensorsModel(root)


# --- no partial export on invalid index -------------------------------------


def test_no_partial_export_on_invalid_index(tmp_path) -> None:
    """Full exporter raises before creating output or emitting any shard."""
    root = tmp_path / "model"
    shard = root / "model.safetensors"
    _save_safetensors(shard, {"weight": torch.tensor([1.0])})
    _make_index(
        root,
        {"weight": "model.safetensors", "bad": "../escape.safetensors"},
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="invalid shard name"):
        export_dense_model_to_fp6_safetensors(
            root, out, device="cpu", use_gpu=False, verbose=False
        )
    assert not out.exists()


# --- handle caching by validated identity -----------------------------------


def test_cache_validated_identity(tmp_path) -> None:
    """Multiple keys for one validated shard reuse a single handle."""
    root = tmp_path / "model"
    shard = root / "model-00001-of-00001.safetensors"
    _save_safetensors(shard, {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    _make_index(
        root,
        {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"},
    )
    model = SafetensorsModel(root)
    model.get_tensor("a")
    model.get_tensor("b")
    # Both keys should share the same cached handle (one entry).
    assert len(model._handles) == 1
