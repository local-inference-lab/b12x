"""Regression tests for FP6 dequantization geometry and working-set guards."""
from __future__ import annotations

import pytest
import torch

import b12x._lib.fp6 as fp6_impl
import b12x.quantization.mxfp6.fp6_checkpoint as checkpoint


def _artifact(rows: int = 2, k: int = 64):
    torch.manual_seed(0)
    weight = torch.randn(rows, k, dtype=torch.bfloat16) / 8
    return checkpoint.quantize_linear_to_fp6(
        weight,
        source_format="mxfp6_w6a8",
        use_gpu=False,
        block_scale_rule="mse",
    )


def _fail_if_expanded(*_args, **_kwargs):
    raise AssertionError("packed FP6 bytes expanded before validation")


def test_malformed_scale_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    malformed_scale = torch.zeros((2, 1), dtype=torch.uint8)
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="block_scale shape"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            malformed_scale,
            fmt=artifact.fmt,
        )


def test_working_set_budget_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="exceeds the operator budget"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt=artifact.fmt,
            max_working_bytes=1,
        )


def test_invalid_format_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="fmt must be"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt="invalid",
        )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_working_set_budget_rejected(budget) -> None:
    artifact = _artifact()
    expected = TypeError if isinstance(budget, (bool, float)) else ValueError

    with pytest.raises(expected, match="max_working_bytes"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt=artifact.fmt,
            max_working_bytes=budget,
        )


def test_exact_working_set_budget_accepts_valid_producer_artifact() -> None:
    artifact = _artifact(rows=3, k=64)
    rows = int(artifact.weight.shape[0])
    k = int(artifact.weight.shape[1]) * 4 // 3
    budget = checkpoint._estimate_fp6_dequant_working_bytes(rows, k)

    decoded = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
        weight_scale_2=artifact.weight_scale_2,
        max_working_bytes=budget,
    )

    assert decoded.shape == (rows, k)
    assert bool(torch.isfinite(decoded).all())


def test_per_row_global_scale_remains_supported() -> None:
    artifact = _artifact(rows=2, k=64)
    unscaled = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
    )
    row_scales = torch.tensor([2.0, 4.0], dtype=torch.float32)

    scaled = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
        weight_scale_2=row_scales,
    )

    torch.testing.assert_close(scaled[0], unscaled[0] / 2)
    torch.testing.assert_close(scaled[1], unscaled[1] / 4)
