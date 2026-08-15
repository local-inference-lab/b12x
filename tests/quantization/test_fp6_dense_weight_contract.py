from __future__ import annotations

import json

import pytest
import torch

from b12x.quantization.mxfp6.fp6_dense_weights import (
    FP6DenseWeight,
    load_fp6_dense_weight,
)


def _weight(**overrides: object) -> FP6DenseWeight:
    values: dict[str, object] = {
        "packed": torch.empty((128, 96), dtype=torch.uint8),
        "scale_storage": torch.empty(512, dtype=torch.uint8),
        "global_scale": torch.ones(1, dtype=torch.float32),
        "out_features": 128,
        "in_features": 128,
        "fmt": "e2m3",
        "act_fmt": "e4m3",
    }
    values.update(overrides)
    return FP6DenseWeight(**values)


def test_fp6_dense_weight_accepts_canonical_storage() -> None:
    weight = _weight()
    assert weight.packed.shape == (128, 96)
    assert weight.scale_view().shape == (32, 4, 1, 4, 1, 1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"out_features": 64}, "packed weight"),
        ({"in_features": 256}, "packed weight"),
        (
            {"packed": torch.empty((96, 128), dtype=torch.uint8).t()},
            "packed weight",
        ),
        ({"scale_storage": torch.empty(511, dtype=torch.uint8)}, "scale storage"),
        ({"global_scale": torch.ones(1, dtype=torch.bfloat16)}, "global scale"),
        ({"fmt": "e4m3"}, "weight format"),
        ({"act_fmt": "e5m2"}, "activation format"),
    ],
)
def test_fp6_dense_weight_rejects_inconsistent_contract(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _weight(**overrides)


def test_fp6_dense_loader_rejects_metadata_storage_mismatch(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    path = tmp_path / "malformed.safetensors"
    safetensors.save_file(
        {
            "packed": torch.empty((128, 96), dtype=torch.uint8),
            "scale_storage": torch.empty(512, dtype=torch.uint8),
            "global_scale": torch.ones(1, dtype=torch.float32),
        },
        path,
        metadata={
            "__format__": "b12x_fp6_dense_weight_v1",
            "out_features": json.dumps(64),
            "in_features": json.dumps(128),
            "fmt": json.dumps("e2m3"),
            "act_fmt": json.dumps("e4m3"),
            "out_features_unsharded": json.dumps(0),
        },
    )

    with pytest.raises(ValueError, match="packed weight"):
        load_fp6_dense_weight(str(path), device="cpu")
