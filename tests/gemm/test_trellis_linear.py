from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparkinfer.gemm import trellis_linear
from sparkinfer.gemm.trellis_linear import api
from sparkinfer.gemm.trellis_linear import _small_m
from sparkinfer.gemm.trellis_linear._small_m import _default_num_sms
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
    _trellis256_dense_launch_geometry,
)


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


@pytest.mark.parametrize(
    ("size_k", "size_n", "available_sms", "expected"),
    [
        (2048, 4096, 188, 128),
        (2048, 4096, 120, 120),
        (6144, 1024, 188, 64),
        (6144, 1024, 48, 48),
        (512, 6144, 188, 96),
        (512, 6144, 80, 80),
        # The unsharded dimensions are deliberately not inferred from TP4.
        (6144, 4096, 188, 188),
        (2048, 6144, 188, 188),
        (3072, 6144, 188, 188),
        (4096, 6144, 188, 188),
        (6144, 6144, 188, 188),
    ],
)
def test_k6_small_m_default_sms_preserves_glm_decode_overlap(
    size_k: int,
    size_n: int,
    available_sms: int,
    expected: int,
) -> None:
    assert _default_num_sms(size_k, size_n, available_sms) == expected


def test_k6_small_m_rejects_unsupported_arch_before_jit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        _small_m,
        "_extension",
        lambda: pytest.fail("extension must not compile for an unsupported GPU"),
    )

    with pytest.raises(NotImplementedError, match="built for sm_120 only"):
        _small_m.run_k6_mcg(*(torch.empty(0) for _ in range(7)))


@pytest.mark.parametrize(
    ("size_m", "size_k", "size_n", "expected"),
    [
        # Narrow profile: avoid a short spill by increasing both M and N work.
        (512, 16384, 6144, (48, (64, 128))),
        (1024, 16384, 6144, (48, (64, 256))),
        # Wide model projection: N already supplies enough work. Splitting N is
        # useful for the second wave, while M48 only adds scheduler overhead.
        (192, 2048, 16384, (64, (64, 128))),
        (384, 2048, 16384, (48, (64, 256))),
        # A deeper K makes M48 scheduler overhead more expensive.
        (384, 6144, 16384, (64, (64, 256))),
        # Full waves retain the throughput-oriented default geometry.
        (2048, 4096, 6144, (64, (64, 256))),
    ],
)
def test_dense_launch_geometry_avoids_short_spill_waves(
    size_m: int,
    size_k: int,
    size_n: int,
    expected: tuple[int, tuple[int, int]],
) -> None:
    assert (
        _trellis256_dense_launch_geometry(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            sms=188,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("m", "size_k", "size_n"),
    [
        (7, 128, 128),
        (1, 2048, 4096),
        (7, 6144, 1024),
        (32, 512, 6144),
    ],
)
@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_k6_small_m_matches_separate_cute_pipeline(
    m: int,
    size_k: int,
    size_n: int,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.randn((size_k,), dtype=torch.float16, device=device)
    svh = torch.randn((size_n,), dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        suh,
        svh,
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=torch.float16,
    )
    x = torch.randn((m, size_k), dtype=torch.float16, device=device)

    def separate_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        left_scale,
        right_scale,
        _scale: float,
    ) -> None:
        _run_trellis_dense_hadamard128(
            source,
            destination,
            left_scale if left_scale is not None else right_scale,
            scale_before=left_scale is not None,
        )

    expected = trellis_linear.run(
        x,
        weight,
        hadamard_128=separate_hadamard,
    ).clone()
    weight.workspace.zero_()
    actual = trellis_linear.run(x, weight).clone()
    torch.cuda.synchronize(device)

    # The cooperative kernel and CuTe fallback use different legal FP16 MMA
    # accumulation orders. Compare their direction and normalized error rather
    # than imposing a large absolute tolerance on values near zero.
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    relative_l2 = delta.norm() / expected_f32.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    max_relative_to_range = delta.abs().max() / expected_f32.abs().max().clamp_min(
        1e-12
    )

    assert float(relative_l2) < 1.5e-3
    assert float(cosine) > 0.999998
    assert float(max_relative_to_range) < 2e-3


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_k6_small_m_cuda_graph_replay_is_stable() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m = 3
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones((features,), dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=torch.float16,
    )
    x = torch.randn((m, features), dtype=torch.float16, device=device)
    output = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    kwargs = {"output": output, "rotated_f16": rotated_f16}

    expected = trellis_linear.run(x, weight, **kwargs).clone()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run(x, weight, **kwargs)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.equal(captured, expected)


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
