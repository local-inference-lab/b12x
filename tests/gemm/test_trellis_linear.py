from __future__ import annotations

from types import SimpleNamespace

import torch

from sparkinfer.gemm import trellis_linear
from sparkinfer.gemm.trellis_linear import api


def test_prepare_weight_delegates_without_copy(monkeypatch) -> None:
    tensors = tuple(torch.empty(0) for _ in range(3))
    expected = SimpleNamespace()
    seen = {}

    def fake_prepare(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(api, "prepare_trellis256_dense_weight", fake_prepare)
    actual = trellis_linear.prepare_weight(
        *tensors,
        codebook="mcg",
        params_dtype=torch.bfloat16,
    )

    assert actual is expected
    assert seen["args"] == tensors
    assert seen["kwargs"]["codebook"] == "mcg"
    assert seen["kwargs"]["params_dtype"] == torch.bfloat16


def test_run_delegates_caller_owned_capture_storage(monkeypatch) -> None:
    x = torch.empty(0)
    weight = SimpleNamespace()
    output, gemm_output, c_tmp = (torch.empty(0) for _ in range(3))
    seen = {}

    def fake_run(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return output

    monkeypatch.setattr(api, "run_trellis256_dense", fake_run)
    actual = trellis_linear.run(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
    )

    assert actual is output
    assert seen["args"] == (x, weight)
    assert seen["kwargs"]["output"] is output
    assert seen["kwargs"]["gemm_output"] is gemm_output
    assert seen["kwargs"]["c_tmp"] is c_tmp


def test_is_supported_uses_standard_sm12x_gate(monkeypatch) -> None:
    seen = {}

    def fake_gate(device, *, requires):
        seen["device"] = device
        seen["requires"] = requires
        return True

    monkeypatch.setattr(api, "default_is_supported", fake_gate)
    assert trellis_linear.is_supported("cuda:3")
    assert seen == {"device": "cuda:3", "requires": trellis_linear.META.requires}
