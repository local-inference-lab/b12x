from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparkinfer.gemm import trellis_linear
from sparkinfer.gemm.trellis_linear import api


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 12 and minor in (0, 1)


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
    assert all(
        seen_arg is arg for seen_arg, arg in zip(seen["args"], tensors, strict=True)
    )
    assert seen["kwargs"]["codebook"] == "mcg"
    assert seen["kwargs"]["params_dtype"] == torch.bfloat16


def test_run_delegates_caller_owned_capture_storage(monkeypatch) -> None:
    x = torch.empty(0)
    weight = SimpleNamespace()
    buffers = tuple(torch.empty(0) for _ in range(8))
    (
        output,
        gemm_output,
        c_tmp,
        input_f16,
        rotated_f16,
        rotated_compute,
        gemm_output_f16,
        output_f16,
    ) = buffers
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
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
    )

    assert actual is output
    assert seen["args"] == (x, weight)
    assert seen["kwargs"]["output"] is output
    assert seen["kwargs"]["gemm_output"] is gemm_output
    assert seen["kwargs"]["c_tmp"] is c_tmp
    assert seen["kwargs"]["input_f16"] is input_f16
    assert seen["kwargs"]["rotated_f16"] is rotated_f16
    assert seen["kwargs"]["rotated_compute"] is rotated_compute
    assert seen["kwargs"]["gemm_output_f16"] is gemm_output_f16
    assert seen["kwargs"]["output_f16"] is output_f16


def test_is_supported_uses_standard_sm12x_gate(monkeypatch) -> None:
    seen = {}

    def fake_gate(device, *, requires):
        seen["device"] = device
        seen["requires"] = requires
        return True

    monkeypatch.setattr(api, "default_is_supported", fake_gate)
    assert trellis_linear.is_supported("cuda:3")
    assert seen == {"device": "cuda:3", "requires": trellis_linear.META.requires}


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_dense_bf16_reuses_all_scratch_during_cuda_graph_capture() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 48),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones((features,), dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=torch.bfloat16,
    )
    x = torch.randn((m, features), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    input_f16 = torch.empty_like(x, dtype=torch.float16)
    rotated_f16 = torch.empty_like(input_f16)
    rotated_compute = torch.empty_like(x)
    gemm_output_f16 = torch.empty_like(input_f16)
    output_f16 = torch.empty_like(input_f16)

    def hadamard_128(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "c_tmp": c_tmp,
        "input_f16": input_f16,
        "rotated_f16": rotated_f16,
        "rotated_compute": rotated_compute,
        "gemm_output_f16": gemm_output_f16,
        "output_f16": output_f16,
        "hadamard_128": hadamard_128,
    }
    expected = trellis_linear.run(x, weight, **kwargs).clone()
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run(x, weight, **kwargs)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.equal(captured, expected)
