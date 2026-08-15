"""Focused regression tests for FP6 dense artifact validation (#152).

Primarily CPU-only: these tests exercise validation/rejection paths without a
kernel launch; one mixed-device contract test is CUDA-gated. The pre-fix code
had no geometry validation in ``FP6DenseWeight.__post_init__``, no N-equality
check in ``dense_fp6_linear_expanded``, and no format-tag check in the loader.
The reusable validator and wrapper guard cover those paths.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from b12x.quantization.mxfp6.fp6_dense_weights import (
    FP6DenseWeight,
    dense_fp6_linear_expanded,
    load_fp6_dense_weight,
    save_fp6_dense_weight,
)

_TILE = 128
_SF_VEC = 32
_FMT_TAG = "b12x_fp6_dense_weight_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_numel(out_features: int, in_features: int) -> int:
    """Padded swizzled UE8M0 scale storage size (matches ``align_up``)."""
    rows_pad = ((out_features + _TILE - 1) // _TILE) * _TILE
    cols_pad = ((in_features // _SF_VEC + 3) // 4) * 4
    return rows_pad * cols_pad


def _make_valid_tensors(n: int, k: int, *, device: str = "cpu"):
    """Create schema-valid CPU tensors for an ``(N, K)`` FP6 dense weight."""
    packed = torch.zeros(n, 3 * k // 4, dtype=torch.uint8, device=device)
    scale = torch.zeros(_scale_numel(n, k), dtype=torch.uint8, device=device)
    gs = torch.ones(1, dtype=torch.float32, device=device)
    return packed, scale, gs


def _make_valid_weight(n: int = 128, k: int = 128, **kw) -> FP6DenseWeight:
    """Construct a schema-valid ``FP6DenseWeight`` on CPU (no CUDA needed)."""
    packed, scale, gs = _make_valid_tensors(n, k)
    defaults = dict(fmt="e2m3", act_fmt="e4m3", out_features_unsharded=0)
    defaults.update(kw)
    return FP6DenseWeight(
        packed=packed,
        scale_storage=scale,
        global_scale=gs,
        out_features=n,
        in_features=k,
        **defaults,
    )


def _write_artifact(
    path,
    *,
    out_features: int,
    in_features: int,
    packed: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    gs: torch.Tensor | None = None,
    fmt: str = "e2m3",
    act_fmt: str = "e4m3",
    out_features_unsharded: int = 0,
    format_tag: str = _FMT_TAG,
):
    """Write a standalone FP6 dense-weight safetensors artifact directly."""
    if packed is None:
        packed = torch.zeros(out_features, 3 * in_features // 4, dtype=torch.uint8)
    if scale is None:
        scale = torch.zeros(_scale_numel(out_features, in_features), dtype=torch.uint8)
    if gs is None:
        gs = torch.ones(1, dtype=torch.float32)
    tensors = {"packed": packed, "scale_storage": scale, "global_scale": gs}
    metadata = {
        "__format__": format_tag,
        "out_features": json.dumps(out_features),
        "in_features": json.dumps(in_features),
        "fmt": json.dumps(fmt),
        "act_fmt": json.dumps(act_fmt),
        "out_features_unsharded": json.dumps(out_features_unsharded),
    }
    save_file(tensors, str(path), metadata=metadata)


# ---------------------------------------------------------------------------
# Loader: malformed artifacts reject before device transfer
# ---------------------------------------------------------------------------

def test_loader_rejects_metadata_n_below_packed_rows(tmp_path):
    """Packed has more rows than metadata ``out_features`` → reject on CPU.

    A valid N=256,K=128 artifact's packed tensor has 256 rows, but the JSON
    metadata claims ``out_features=128``. The validator's packed-shape check
    catches the mismatch at dataclass construction — before ``.to(device)``.
    """
    packed, scale, gs = _make_valid_tensors(256, 128)
    path = tmp_path / "below.safetensors"
    _write_artifact(
        path,
        out_features=128,  # metadata claims 128
        in_features=128,
        packed=packed,
        scale=scale,
        gs=gs,
    )
    with pytest.raises(ValueError, match="packed shape"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_metadata_n_above_packed_rows(tmp_path):
    """Packed has fewer rows than metadata ``out_features`` → reject on CPU.

    A valid N=128,K=128 artifact's packed tensor has 128 rows, but the JSON
    metadata claims ``out_features=256``. The validator's packed-shape check
    catches the mismatch at dataclass construction — before ``.to(device)``.
    """
    packed, scale, gs = _make_valid_tensors(128, 128)
    path = tmp_path / "above.safetensors"
    _write_artifact(
        path,
        out_features=256,  # metadata claims 256
        in_features=128,
        packed=packed,
        scale=scale,
        gs=gs,
    )
    with pytest.raises(ValueError, match="packed shape"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_wrong_format_tag(tmp_path):
    """An artifact without the ``b12x_fp6_dense_weight_v1`` tag is rejected."""
    path = tmp_path / "badtag.safetensors"
    _write_artifact(path, out_features=128, in_features=128, format_tag="evil")
    with pytest.raises(ValueError, match="unsupported FP6 dense artifact format"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_unexpected_tensor_key(tmp_path):
    """Only the three tensors in the standalone schema are accepted."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    path = tmp_path / "extra-tensor.safetensors"
    save_file(
        {
            "packed": packed,
            "scale_storage": scale,
            "global_scale": gs,
            "act_fmt": torch.zeros(1, dtype=torch.uint8),
        },
        str(path),
        metadata={
            "__format__": _FMT_TAG,
            "out_features": "128",
            "in_features": "128",
            "fmt": json.dumps("e2m3"),
        },
    )
    with pytest.raises(ValueError, match="tensor keys must be exactly"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_missing_tensor_key(tmp_path):
    """A missing tensor is reported as a schema error, not a constructor error."""
    packed, scale, _ = _make_valid_tensors(128, 128)
    path = tmp_path / "missing-tensor.safetensors"
    save_file(
        {"packed": packed, "scale_storage": scale},
        str(path),
        metadata={
            "__format__": _FMT_TAG,
            "out_features": "128",
            "in_features": "128",
            "fmt": json.dumps("e2m3"),
        },
    )
    with pytest.raises(ValueError, match="tensor keys must be exactly"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_unexpected_metadata_key(tmp_path):
    """Application metadata is allowlisted before dataclass construction."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    path = tmp_path / "extra-metadata.safetensors"
    save_file(
        {"packed": packed, "scale_storage": scale, "global_scale": gs},
        str(path),
        metadata={
            "__format__": _FMT_TAG,
            "out_features": "128",
            "in_features": "128",
            "fmt": json.dumps("e2m3"),
            "evil": json.dumps("value"),
        },
    )
    with pytest.raises(ValueError, match="unexpected metadata keys"):
        load_fp6_dense_weight(str(path), device="cpu")


def test_loader_rejects_metadata_before_reading_tensors(monkeypatch):
    """Metadata schema failure occurs before safetensor materialization."""
    class MetadataOnlyArtifact:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {
                "__format__": _FMT_TAG,
                "out_features": "128",
                "in_features": "128",
                "fmt": json.dumps("e2m3"),
                "evil": json.dumps("value"),
            }

        def keys(self):
            raise AssertionError("tensor keys inspected before metadata rejection")

    monkeypatch.setattr(
        "safetensors.torch.safe_open",
        lambda *_args, **_kwargs: MetadataOnlyArtifact(),
    )
    with pytest.raises(ValueError, match="unexpected metadata keys"):
        load_fp6_dense_weight("unused.safetensors", device="cpu")


def test_loader_rejects_before_device_transfer(tmp_path, monkeypatch):
    """Malformed geometry is rejected before ``FP6DenseWeight.to`` can run."""
    packed, scale, gs = _make_valid_tensors(256, 128)
    path = tmp_path / "pre-transfer.safetensors"
    _write_artifact(
        path,
        out_features=128,
        in_features=128,
        packed=packed,
        scale=scale,
        gs=gs,
    )
    called = False

    def fail_to(self, device):
        nonlocal called
        called = True
        raise AssertionError("malformed artifact reached device transfer")

    monkeypatch.setattr(FP6DenseWeight, "to", fail_to)
    with pytest.raises(ValueError, match="packed shape"):
        load_fp6_dense_weight(str(path), device="cuda")
    assert not called


# ---------------------------------------------------------------------------
# Dataclass: geometry validation on every construction path
# ---------------------------------------------------------------------------

def test_dataclass_rejects_packed_row_mismatch():
    """Direct construction with packed.shape[0] != out_features is rejected."""
    packed, scale, gs = _make_valid_tensors(256, 128)
    with pytest.raises(ValueError, match="packed shape"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=128,  # mismatch with 256 packed rows
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_in_features_not_multiple_of_32():
    """The format requires K to be a positive multiple of the scale group."""
    packed = torch.zeros(64, 36, dtype=torch.uint8)
    scale = torch.zeros(_scale_numel(64, 48), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="in_features must be a positive multiple of 32"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=64,
            in_features=48,
            fmt="e2m3",
        )


def test_dataclass_rejects_wrong_scale_size():
    """Scale storage must match the exact padded size for the declared geometry."""
    packed = torch.zeros(128, 96, dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    # Too few scale elements
    with pytest.raises(ValueError, match="scale_storage must hold"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=torch.zeros(100, dtype=torch.uint8),
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_non_scalar_global_scale():
    """``global_scale`` must have the producer's exact one-element shape."""
    packed, scale, _ = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match=r"global_scale must have shape \(1,\)"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=torch.ones(2, dtype=torch.float32),
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_bool_dimension():
    """A JSON boolean must not pass Python's ``bool``-is-an-``int`` quirk."""
    packed, scale, gs = _make_valid_tensors(1, 128)
    with pytest.raises(ValueError, match="out_features must be int"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=True,
            in_features=128,
            fmt="e2m3",
        )


@pytest.mark.parametrize("bad_dtype", [torch.int8, torch.float32])
def test_dataclass_rejects_wrong_packed_dtype(bad_dtype):
    """Packed code storage is byte-exact uint8."""
    _, scale, gs = _make_valid_tensors(128, 128)
    packed = torch.zeros(128, 96, dtype=bad_dtype)
    with pytest.raises(ValueError, match="packed must be uint8"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_noncontiguous_packed():
    """A shape-compatible strided view cannot be passed to the raw layout."""
    packed = torch.zeros(128, 192, dtype=torch.uint8)[:, ::2]
    assert packed.shape == (128, 96)
    assert not packed.is_contiguous()
    _, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="packed must be contiguous"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_oversized_scale_storage():
    """Scale storage must be exact; extra bytes cannot hide metadata drift."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="scale_storage must hold"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=torch.zeros(scale.numel() + 1, dtype=torch.uint8),
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_bad_fmt():
    """``fmt`` must be on the FP6 sub-format allowlist."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="fmt must be one of"):
        FP6DenseWeight(
            packed=packed, scale_storage=scale, global_scale=gs,
            out_features=128, in_features=128, fmt="e5m2",
        )


def test_dataclass_rejects_bad_act_fmt():
    """``act_fmt`` must be on the activation sub-format allowlist."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="act_fmt must be one of"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
            act_fmt="e5m2",
        )


def test_dataclass_rejects_global_scale_dtype():
    """``global_scale`` must use the producer's float32 representation."""
    packed, scale, _ = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="global_scale must be float32"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=torch.ones(1, dtype=torch.float64),
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_noncontiguous_scale_storage():
    """An exact-size strided scale view is not a canonical kernel buffer."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    strided_scale = torch.zeros(scale.numel() * 2, dtype=torch.uint8)[::2]
    assert strided_scale.numel() == scale.numel()
    assert not strided_scale.is_contiguous()
    with pytest.raises(ValueError, match="scale_storage must be contiguous"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=strided_scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_packed_rank():
    """Packed storage must have the producer's rank-2 layout."""
    _, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="packed must be rank-2"):
        FP6DenseWeight(
            packed=torch.zeros(128, 96, 1, dtype=torch.uint8),
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


def test_dataclass_rejects_negative_out_features_unsharded():
    """The routing hint cannot encode a negative width."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="out_features_unsharded must be >= 0"):
        FP6DenseWeight(
            packed=packed,
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
            out_features_unsharded=-1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dataclass_rejects_mixed_tensor_devices():
    """All raw buffers must name storage on the same device."""
    packed, scale, gs = _make_valid_tensors(128, 128)
    with pytest.raises(ValueError, match="must share one device"):
        FP6DenseWeight(
            packed=packed.to("cuda"),
            scale_storage=scale,
            global_scale=gs,
            out_features=128,
            in_features=128,
            fmt="e2m3",
        )


# ---------------------------------------------------------------------------
# Raw wrapper / custom-op: N-equality before launch
# ---------------------------------------------------------------------------

def test_expanded_wrapper_rejects_weight_rows_below_out_features():
    """``dense_fp6_linear_expanded`` rejects weight.shape[0] < out_features."""
    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    weight = torch.zeros(128, 96, dtype=torch.uint8)  # 128 rows
    scale = torch.zeros(_scale_numel(128, 128), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="out_features mismatch"):
        dense_fp6_linear_expanded(
            x, weight, scale, gs, fmt="e2m3",
            out_features=256,  # metadata claims 256, weight has 128
            in_features=128,
        )


def test_expanded_wrapper_rejects_weight_rows_above_out_features():
    """``dense_fp6_linear_expanded`` rejects weight.shape[0] > out_features."""
    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    weight = torch.zeros(256, 96, dtype=torch.uint8)  # 256 rows
    scale = torch.zeros(_scale_numel(128, 128), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="out_features mismatch"):
        dense_fp6_linear_expanded(
            x, weight, scale, gs, fmt="e2m3",
            out_features=128,  # metadata claims 128, weight has 256
            in_features=128,
        )


def test_expanded_wrapper_rejects_expanded_weight_mismatch():
    """N-equality also holds for the 3-D expanded ``(N, K, 1)`` layout."""
    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    weight = torch.zeros(256, 128, 1, dtype=torch.uint8)  # expanded, 256 rows
    scale = torch.zeros(_scale_numel(128, 128), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="out_features mismatch"):
        dense_fp6_linear_expanded(
            x, weight, scale, gs, fmt="e2m3",
            out_features=128,
            in_features=128,
        )


def test_custom_op_rejects_weight_rows_mismatch():
    """The opaque custom op propagates the N-equality rejection."""
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401

    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    weight = torch.zeros(128, 96, dtype=torch.uint8)
    scale = torch.zeros(_scale_numel(128, 128), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="out_features mismatch"):
        torch.ops.b12x.fp6_dense_linear(
            x, weight, scale, gs, "e2m3",
            256,  # out_features mismatch
            128,
        )


# ---------------------------------------------------------------------------
# Valid artifacts: behavior retained
# ---------------------------------------------------------------------------

def test_valid_artifact_roundtrip_cpu(tmp_path):
    """A valid saver-produced artifact round-trips on CPU validation."""
    w = _make_valid_weight(256, 128, fmt="e2m3", act_fmt="e4m3")
    path = tmp_path / "valid.safetensors"
    save_fp6_dense_weight(w, path)

    loaded = load_fp6_dense_weight(str(path), device="cpu")
    assert isinstance(loaded, FP6DenseWeight)
    assert loaded.out_features == 256
    assert loaded.in_features == 128
    assert loaded.fmt == "e2m3"
    assert loaded.act_fmt == "e4m3"
    torch.testing.assert_close(loaded.packed, w.packed, rtol=0, atol=0)
    torch.testing.assert_close(loaded.scale_storage, w.scale_storage, rtol=0, atol=0)
    torch.testing.assert_close(loaded.global_scale, w.global_scale, rtol=0, atol=0)


def test_valid_w6a6_dataclass_construction():
    """W6A6 (act_fmt == fmt) dataclass construction passes validation."""
    w = _make_valid_weight(128, 128, fmt="e3m2", act_fmt="e3m2")
    assert w.act_fmt == "e3m2"
    assert w.packed.shape == (128, 96)


def test_valid_w6a8_dataclass_construction():
    """W6A8 (act_fmt == e4m3) dataclass construction passes validation."""
    w = _make_valid_weight(128, 128, fmt="e2m3", act_fmt="e4m3")
    assert w.act_fmt == "e4m3"


def test_valid_unaligned_output_width_preserved():
    """N need not be 128-aligned because scale storage pads its row extent."""
    w = _make_valid_weight(64, 128, fmt="e2m3", act_fmt="e4m3")
    assert w.packed.shape == (64, 96)
    assert w.scale_storage.numel() == _scale_numel(64, 128)




def test_to_preserves_geometry():
    """``.to('cpu')`` reconstructs a valid validated dataclass."""
    w = _make_valid_weight(256, 128, fmt="e2m3", act_fmt="e4m3", out_features_unsharded=512)
    w2 = w.to("cpu")
    assert w2.out_features == 256
    assert w2.out_features_unsharded == 512
    assert w2.packed.shape == (256, 96)


# ---------------------------------------------------------------------------
# Custom-op fake shapes: preserved
# ---------------------------------------------------------------------------

def test_custom_op_fake_shape_preserved():
    """Fake and real paths agree on valid metadata/weight geometry."""
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode

    x = torch.zeros(7, 128, dtype=torch.bfloat16)
    weight = torch.zeros(128, 96, dtype=torch.uint8)
    scale = torch.zeros(_scale_numel(128, 128), dtype=torch.uint8)
    gs = torch.ones(1, dtype=torch.float32)
    with FakeTensorMode() as mode:
        fx = mode.from_tensor(x)
        fweight = mode.from_tensor(weight)
        fscale = mode.from_tensor(scale)
        fgs = mode.from_tensor(gs)
        out = torch.ops.b12x.fp6_dense_linear(
            fx, fweight, fscale, fgs, "e2m3", 128, 128
        )
        assert tuple(out.shape) == (7, 128)
        assert out.dtype == torch.bfloat16
