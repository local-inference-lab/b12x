from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear import api
from b12x.gemm.trellis_linear import _small_m
from b12x.gemm.trellis_linear._small_m import _default_num_sms
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.quant.sqg_e4m3 import (
    sqg_cheb_normal_e4m3_direct_lut_cpu,
    sqg_xor_cheb_t12_direct_lut_cpu,
)
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.trellis_w4a8_pair import (
    run_trellis_w4a8_fc1_routes,
    run_trellis_w4a8_fc2_routes,
)
from b12x.moe._shared.kernels.trellis_w4a8 import (
    make_trellis_w4a8_moe_scratch,
    run_trellis_w4a8_moe,
)
from b12x.moe._shared.kernels.trellis_w4a8_transform import (
    run_trellis_w4a8_activation_rotation,
    run_trellis_w4a8_input_rotation,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _run_trellis_dense_hadamard128,
    _trellis256_dense_launch_geometry,
    _use_k6_mcg_small,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_qsrt_pair_moe_weights,
)


_MCG = np.uint64(0xCBAC1FED)
_MUL1 = np.uint64(0x83DCD12D)
_MASK = np.uint32(0x8FFF8FFF)
_ORC = np.uint32(0x3B603B60)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 12 and minor in (0, 1)


def _hadamard128_reference(source: torch.Tensor) -> torch.Tensor:
    """Natural-order, normalized H128 with the kernel's fp32 butterflies."""

    original_shape = source.shape
    values = source.float().reshape(-1, 128).clone()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        groups = values.reshape(-1, 128 // (2 * stride), 2, stride)
        low = groups[:, :, 0, :].clone()
        high = groups[:, :, 1, :].clone()
        groups[:, :, 0, :] = low + high
        groups[:, :, 1, :] = low - high
    return (values * (1.0 / np.sqrt(128.0))).reshape(original_shape)


def _decode_3inst_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MASK) ^ _ORC)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_mul1_e4m3_fp16(window: np.ndarray) -> np.ndarray:
    product = ((window.astype(np.uint64) * _MUL1) & np.uint64(0xFFFFFFFF)).astype(
        np.uint32
    )
    byte_sum = (
        (product & np.uint32(0xFF)).astype(np.uint32)
        + ((product >> np.uint32(8)) & np.uint32(0xFF))
        + ((product >> np.uint32(16)) & np.uint32(0xFF))
        + ((product >> np.uint32(24)) & np.uint32(0xFF))
    )
    accumulator = (byte_sum + np.uint32(0x6400)).astype(np.uint16).view(np.float16)
    inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0]
    bias = np.array([0xC931], dtype=np.uint16).view(np.float16)[0]
    reconstructed = (
        accumulator.astype(np.float64) * np.float64(inv) + np.float64(bias)
    ).astype(np.float16)
    return (
        torch.from_numpy(np.asarray(reconstructed))
        .to(torch.float8_e4m3fn)
        .to(torch.float16)
        .numpy()
    )


@lru_cache(maxsize=None)
def _sqg_cheb_normal_e4m3_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported SQG-Cheb test rate K{bits}")
    rate_index = bits - 2
    labels = sqg_cheb_normal_e4m3_direct_lut_cpu()[
        rate_index << 16 : (rate_index + 1) << 16
    ]
    return labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


def _decode_sqg_cheb_normal_e4m3_fp16(window: np.ndarray, bits: int) -> np.ndarray:
    indices = np.asarray(window, dtype=np.uint32) & np.uint32(0xFFFF)
    return _sqg_cheb_normal_e4m3_table(bits)[indices]


@lru_cache(maxsize=None)
def _sqg_xor_cheb_t12_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported SQG-XOR-Cheb-T12 test rate K{bits}")
    rate_index = bits - 2
    labels = sqg_xor_cheb_t12_direct_lut_cpu()[
        rate_index << 16 : (rate_index + 1) << 16
    ]
    return labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


def _decode_sqg_xor_cheb_t12_fp16(window: np.ndarray, bits: int) -> np.ndarray:
    indices = np.asarray(window, dtype=np.uint32) & np.uint32(0xFFFF)
    return _sqg_xor_cheb_t12_table(bits)[indices]


def _decode_lane(
    tile_words: np.ndarray,
    lane: int,
    bits: int,
    *,
    codebook: str = "mcg",
) -> np.ndarray:
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint32)
        if codebook == "mcg":
            values.append(_decode_3inst_fp16(window))
        elif codebook == "mul1-e4m3":
            values.append(_decode_mul1_e4m3_fp16(window))
        elif codebook == "sqg-cheb-normal-e4m3":
            values.append(_decode_sqg_cheb_normal_e4m3_fp16(window, bits))
        elif codebook == "sqg_xor_cheb_t12":
            values.append(_decode_sqg_xor_cheb_t12_fp16(window, bits))
        else:
            raise ValueError(f"unsupported test codebook {codebook!r}")
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(
    trellis: torch.Tensor, *, codebook: str = "mcg"
) -> torch.Tensor:
    native = trellis.detach().cpu().numpy()
    bits = int(native.shape[-1]) // 16
    k_tiles, n_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(k_tiles, n_tiles, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    output = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float16)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            lanes = np.stack(
                [
                    _decode_lane(words[k_tile, n_tile], lane, bits, codebook=codebook)
                    for lane in range(32)
                ]
            )
            block = np.zeros((16, 16), dtype=np.float16)
            for lane in range(32):
                row0 = (lane % 4) * 2
                rows = (row0, row0 + 1, row0 + 8, row0 + 9)
                col0 = lane // 8
                col1 = col0 + 4
                parity = (lane >> 2) & 1
                for weight in range(8):
                    block[
                        rows[weight % 4],
                        2 * (col0 if weight < 4 else col1) + parity,
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)


def _reference_mxfp8_rows(source: torch.Tensor) -> torch.Tensor:
    """Reference the UE8M0/K32 E4M3 activation quantizer in logical K order."""
    m, k = source.shape
    blocks = source.float().reshape(m, k // 32, 32)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, device=source.device), exponent)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    return (blocks / scale).to(torch.float8_e4m3fn).float().mul(scale).reshape(m, k)


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


def test_prepare_pair_weight_delegates_format_metadata(monkeypatch) -> None:
    tensors = tuple(torch.empty(0) for _ in range(3))
    expected = SimpleNamespace()
    seen = {}

    def fake_prepare(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(api, "prepare_trellis256_pair_dense_weight", fake_prepare)
    actual = trellis_linear.prepare_pair_weight(
        *tensors,
        pair_kind="P24",
        rate_axis="n",
        codebook="mcg",
    )

    assert actual is expected
    assert all(
        seen_arg is arg for seen_arg, arg in zip(seen["args"], tensors, strict=True)
    )
    assert seen["kwargs"]["pair_kind"] == "P24"
    assert seen["kwargs"]["rate_axis"] == "n"
    assert seen["kwargs"]["codebook"] == "mcg"


@pytest.mark.parametrize(
    ("pair_kind", "rate_axis", "match"),
    [
        ("P25", "n", "pair_kind must be P24 or P33"),
        ("P24", "x", "rate_axis must be 'k' or 'n'"),
    ],
)
def test_prepare_pair_weight_rejects_malformed_descriptor_before_cuda(
    pair_kind: str,
    rate_axis: str,
    match: str,
) -> None:
    payload = torch.empty(0, dtype=torch.int16)
    scale = torch.empty(0, dtype=torch.float16)

    with pytest.raises(ValueError, match=match):
        trellis_linear.prepare_pair_weight(
            payload,
            scale,
            scale,
            pair_kind=pair_kind,
            rate_axis=rate_axis,
        )


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
    assert seen["kwargs"]["_moe_block_size"] is None
    assert seen["kwargs"]["_force_tile_config"] is None


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
    ("capability", "explicit_launch_config", "expected"),
    [
        ((12, 0), False, True),
        ((12, 0), True, False),
        ((12, 1), False, False),
        ((9, 0), False, False),
    ],
)
def test_k6_small_m_dispatch_requires_compiled_target(
    monkeypatch,
    capability: tuple[int, int],
    explicit_launch_config: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: capability,
    )

    assert (
        _use_k6_mcg_small(
            device=torch.device("cuda"),
            m=128,
            trellis_bits=6,
            trellis_codebook="mcg",
            trellis_pair_kind=None,
            compute_dtype=torch.float16,
            external_hadamard_128=None,
            explicit_launch_config=explicit_launch_config,
        )
        is expected
    )


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
@pytest.mark.parametrize("bits", [2, 3])
def test_dense_bf16_reuses_all_scratch_during_cuda_graph_capture(bits: int) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 16 * bits),
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


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dense_mul1_e4m3_matches_reference(bits: int) -> None:
    torch.manual_seed(0xE4A3 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones(features, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        mul1_e4m3=torch.tensor(0x83DCD12D, dtype=torch.uint32, device=device),
        params_dtype=torch.float16,
    )
    assert weight.trellis_codebook == "mul1-e4m3"
    reference_weight = _reconstruct_native(trellis, codebook="mul1-e4m3").to(device)
    x = (torch.randn((m, features), device=device) * 1.0e-3).to(torch.float16)

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    actual = trellis_linear.run(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
        hadamard_128=identity_hadamard,
    ).clone()
    torch.cuda.synchronize(device)

    expected = (x.float() @ reference_weight.float()).to(torch.float16)
    relative_error = (actual - expected).float().norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dense_sqg_cheb_normal_e4m3_matches_reference(bits: int) -> None:
    """Close the exact SQG labels through the real W4A16 GEMM fragment path."""

    torch.manual_seed(0x535147 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones(features, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        codebook="sqg-cheb-normal-e4m3",
        params_dtype=torch.float16,
    )
    assert weight.trellis_codebook == "sqg-cheb-normal-e4m3"
    reference_weight = _reconstruct_native(trellis, codebook="sqg-cheb-normal-e4m3").to(
        device
    )
    x = (torch.randn((m, features), device=device) * 1.0e-3).to(torch.float16)

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    actual = trellis_linear.run(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
        hadamard_128=identity_hadamard,
    ).clone()
    torch.cuda.synchronize(device)

    expected = (x.float() @ reference_weight.float()).to(torch.float16)
    relative_error = (actual - expected).float().norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(("pair_kind", "bits"), [("P24", (2, 4)), ("P33", (3, 3))])
@pytest.mark.parametrize("rate_axis", ["k", "n"])
@pytest.mark.parametrize(
    "orthogonal_tiles",
    [16, 224],
    ids=["square-proof", "k3-tp12"],
)
@pytest.mark.parametrize("codebook", ["mcg", "sqg-cheb-normal-e4m3"])
def test_dense_pair_matches_independent_reference_and_captures(
    pair_kind: str,
    bits: tuple[int, int],
    rate_axis: str,
    orthogonal_tiles: int,
    codebook: str,
) -> None:
    torch.manual_seed(20260801)
    device = torch.device("cuda", torch.cuda.current_device())
    low_bits, high_bits = bits
    if rate_axis == "n":
        low = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=1,
        ).to(device)
    else:
        low = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=0,
        ).to(device)
    payload = torch.cat((low.reshape(-1), high.reshape(-1))).contiguous()
    suh = torch.ones(reference_weight.shape[0], dtype=torch.float16, device=device)
    svh = torch.ones(reference_weight.shape[1], dtype=torch.float16, device=device)
    codebook_kwargs = (
        {"mcg": torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device)}
        if codebook == "mcg"
        else {"codebook": codebook}
    )
    weight = trellis_linear.prepare_pair_weight(
        payload,
        suh,
        svh,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        params_dtype=torch.float16,
        **codebook_kwargs,
    )
    assert weight.trellis_codebook == codebook
    assert weight.trellis.numel() * weight.trellis.element_size() == (
        payload.numel() * payload.element_size()
    )
    if orthogonal_tiles == 224:
        assert payload.numel() * payload.element_size() == 344064
    if rate_axis == "k":
        assert weight.trellis.data_ptr() == payload.data_ptr()
    else:
        assert weight.trellis.data_ptr() != payload.data_ptr()

    x = (torch.randn((2, reference_weight.shape[0]), device=device) * 1.0e-3).to(
        torch.float16
    )

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty(
        (x.shape[0], reference_weight.shape[1]), dtype=x.dtype, device=device
    )
    gemm_output = torch.empty_like(output)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    actual = trellis_linear.run(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        rotated_f16=rotated_f16,
        c_tmp=c_tmp,
        hadamard_128=identity_hadamard,
    ).clone()
    torch.cuda.synchronize(device)
    expected = (x.float() @ reference_weight.float()).to(torch.float16)
    relative_error = (actual - expected).float().norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run(
            x,
            weight,
            output=output,
            gemm_output=gemm_output,
            rotated_f16=rotated_f16,
            c_tmp=c_tmp,
            hadamard_128=identity_hadamard,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(("pair_kind", "bits"), [("P24", (2, 4)), ("P33", (3, 3))])
@pytest.mark.parametrize("rate_axis", ["k", "n"])
@pytest.mark.parametrize("codebook", ["mul1-e4m3", "sqg-cheb-normal-e4m3"])
def test_dense_pair_e4m3_w4a8_matches_quantized_reference_and_captures(
    pair_kind: str,
    bits: tuple[int, int],
    rate_axis: str,
    codebook: str,
) -> None:
    """Exercise compact P24/P33 addressing through the real E4M3 MMA path."""
    torch.manual_seed(0x4A8 + sum(bits) + (rate_axis == "n"))
    device = torch.device("cuda", torch.cuda.current_device())
    low_bits, high_bits = bits
    orthogonal_tiles = 8
    if rate_axis == "n":
        low = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=1,
        ).to(device)
    else:
        low = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=0,
        ).to(device)

    payload = torch.cat((low.reshape(-1), high.reshape(-1))).contiguous()
    size_k, size_n = reference_weight.shape
    codebook_kwargs = (
        {"mul1_e4m3": torch.tensor(0x83DCD12D, dtype=torch.uint32, device=device)}
        if codebook == "mul1-e4m3"
        else {"codebook": codebook}
    )
    weight = trellis_linear.prepare_pair_weight(
        payload,
        torch.ones(size_k, dtype=torch.float16, device=device),
        torch.ones(size_n, dtype=torch.float16, device=device),
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        params_dtype=torch.float16,
        **codebook_kwargs,
    )
    m = 16
    x = (torch.randn((m, size_k), device=device) * 0.125).to(torch.float16)

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty((m, size_n), dtype=torch.float16, device=device)
    rotated_f16 = torch.empty_like(x)
    quantized = empty_mxfp8_rows_for_dense_gemm(m, size_k, device=device)
    gemm_output_f16 = torch.empty_like(output)
    kwargs = {
        "output": output,
        "rotated_f16": rotated_f16,
        "quantized": quantized,
        "gemm_output_f16": gemm_output_f16,
        "hadamard_128": identity_hadamard,
    }
    actual = trellis_linear.run_w4a8(x, weight, **kwargs).clone()
    torch.cuda.synchronize(device)

    x_quantized = _reference_mxfp8_rows(x)
    expected = (x_quantized @ reference_weight.float()).to(torch.float16)
    torch.testing.assert_close(actual, expected, rtol=2.0e-3, atol=2.0e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trellis_linear.run_w4a8(x, weight, **kwargs)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("broadcast_suh", [False, True])
def test_route_major_w4a8_rotations_match_exl_transform_order_and_capture(
    broadcast_suh: bool,
) -> None:
    torch.manual_seed(20260804)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate, tokens, topk = 3, 256, 256, 2, 2
    routes = tokens * topk
    scale_rows = 1 if broadcast_suh else experts

    prepared = SimpleNamespace(
        hidden_size=hidden,
        num_experts=experts,
        gate_suh=(0.75 + 0.5 * torch.rand((scale_rows, hidden), device=device)).half(),
        up_suh=(0.75 + 0.5 * torch.rand((scale_rows, hidden), device=device)).half(),
        intermediate_rotations=(
            0.75 + 0.5 * torch.rand((experts, 3 * intermediate), device=device)
        ).half(),
    )
    source = torch.randn((tokens, hidden), device=device).bfloat16()
    route_experts = torch.tensor([0, 2, 1, 0], dtype=torch.int32, device=device)
    gate_rot = torch.empty((routes, hidden), dtype=torch.float16, device=device)
    up_rot = torch.empty_like(gate_rot)
    run_trellis_w4a8_input_rotation(
        source,
        route_experts,
        prepared,
        gate_rot,
        up_rot,
        topk=topk,
    )

    source_routes = source.to(torch.float16).repeat_interleave(topk, dim=0)
    scale_indices = (
        torch.zeros_like(route_experts) if broadcast_suh else route_experts
    ).long()
    expected_gate = _hadamard128_reference(
        (source_routes.float() * prepared.gate_suh[scale_indices].float()).half()
    ).half()
    expected_up = _hadamard128_reference(
        (source_routes.float() * prepared.up_suh[scale_indices].float()).half()
    ).half()
    torch.testing.assert_close(gate_rot, expected_gate, rtol=1.0e-3, atol=2.0e-3)
    torch.testing.assert_close(up_rot, expected_up, rtol=1.0e-3, atol=2.0e-3)

    gate = (torch.randn((routes, intermediate), device=device) * 0.25).half()
    up = (torch.randn_like(gate) * 0.25).half()
    activated = torch.empty_like(gate)
    run_trellis_w4a8_activation_rotation(gate, up, route_experts, prepared, activated)
    rotations = prepared.intermediate_rotations[route_experts.long()]
    gate_canonical = _hadamard128_reference(gate) * rotations[:, :intermediate]
    up_canonical = (
        _hadamard128_reference(up) * rotations[:, intermediate : 2 * intermediate]
    )
    situ_gate = (
        SITU_DEFAULT_BETA
        * torch.tanh(gate_canonical / SITU_DEFAULT_BETA)
        * torch.sigmoid(gate_canonical)
    )
    situ_up = SITU_DEFAULT_LINEAR_BETA * torch.tanh(
        up_canonical / SITU_DEFAULT_LINEAR_BETA
    )
    expected_activated = _hadamard128_reference(
        situ_gate * situ_up * rotations[:, 2 * intermediate : 3 * intermediate]
    ).half()
    torch.testing.assert_close(activated, expected_activated, rtol=3.0e-3, atol=3.0e-3)

    gate_before = gate_rot.clone()
    up_before = up_rot.clone()
    activated_before = activated.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_trellis_w4a8_input_rotation(
            source,
            route_experts,
            prepared,
            gate_rot,
            up_rot,
            topk=topk,
        )
        run_trellis_w4a8_activation_rotation(
            gate, up, route_experts, prepared, activated
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(gate_rot, gate_before)
    assert torch.equal(up_rot, up_before)
    assert torch.equal(activated, activated_before)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_route_major_e4m3_w4a8_honors_separate_dynamic_pair_modes() -> None:
    """Mix P33/P24 experts and independently select FC1 versus FC2 modes."""
    codebook = "sqg_xor_cheb_t12"
    torch.manual_seed(20260803)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = 2
    hidden = 128
    intermediate = 256
    hidden_tiles = hidden // 16
    n_tiles = hidden // 16
    fc1_modes = torch.tensor([0, 1], dtype=torch.int32, device=device)
    fc2_modes = torch.tensor([1, 0], dtype=torch.int32, device=device)

    w13_payload = torch.empty(
        (2, experts, hidden_tiles * 8 * 16 * 6),
        dtype=torch.int16,
        device=device,
    )
    w13_reference = torch.empty(
        (2, experts, hidden, intermediate),
        dtype=torch.float16,
        device=device,
    )
    for projection in range(2):
        for expert in range(experts):
            low_bits, high_bits = (2, 4) if int(fc1_modes[expert]) else (3, 3)
            low = torch.randint(
                -32768,
                32767,
                (hidden_tiles, 8, 16 * low_bits),
                dtype=torch.int16,
                device=device,
            )
            high = torch.randint(
                -32768,
                32767,
                (hidden_tiles, 8, 16 * high_bits),
                dtype=torch.int16,
                device=device,
            )
            w13_payload[projection, expert].copy_(
                torch.cat((low.reshape(-1), high.reshape(-1)))
            )
            w13_reference[projection, expert].copy_(
                torch.cat(
                    (
                        _reconstruct_native(low, codebook=codebook),
                        _reconstruct_native(high, codebook=codebook),
                    ),
                    dim=1,
                ).to(device)
            )

    w2_payload = torch.empty(
        (experts, hidden_tiles * 8 * 16 * 6),
        dtype=torch.int16,
        device=device,
    )
    w2_reference = torch.empty(
        (experts, intermediate, hidden), dtype=torch.float16, device=device
    )
    for expert in range(experts):
        low_bits, high_bits = (2, 4) if int(fc2_modes[expert]) else (3, 3)
        low = torch.randint(
            -32768,
            32767,
            (8, n_tiles, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (8, n_tiles, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        w2_payload[expert].copy_(torch.cat((low.reshape(-1), high.reshape(-1))))
        w2_reference[expert].copy_(
            torch.cat(
                (
                    _reconstruct_native(low, codebook=codebook),
                    _reconstruct_native(high, codebook=codebook),
                ),
                dim=0,
            ).to(device)
        )

    prepared = prepare_qsrt_pair_moe_weights(
        w13_payload,
        w2_payload,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        fc1_pair_kind=fc1_modes,
        fc2_pair_kind=fc2_modes,
        gate_suh=torch.ones((1, hidden), dtype=torch.float16, device=device),
        up_suh=torch.ones((1, hidden), dtype=torch.float16, device=device),
        intermediate_rotations=torch.ones(
            (experts, 3 * intermediate), dtype=torch.float16, device=device
        ),
        down_svh=torch.ones((1, hidden), dtype=torch.float16, device=device),
        params_dtype=torch.float16,
        codebook=codebook,
        tile_config=(64, 256, 128, 128),
    )
    route_experts = torch.tensor([0, 1], dtype=torch.int32, device=device)
    routes = int(route_experts.numel())
    decode_kwargs = {"sqg_direct_lut": True}

    gate_x = (torch.randn((routes, hidden), device=device) * 0.125).half()
    up_x = (torch.randn((routes, hidden), device=device) * 0.125).half()
    gate_q = empty_mxfp8_rows_for_dense_gemm(routes, hidden, device=device)
    up_q = empty_mxfp8_rows_for_dense_gemm(routes, hidden, device=device)
    quantize_mxfp8_rows_cute(
        gate_x,
        gate_q.values,
        gate_q.scale_rows,
        gate_q.scale_mma,
        value_order="trellis_native_mma",
    )
    quantize_mxfp8_rows_cute(
        up_x,
        up_q.values,
        up_q.scale_rows,
        up_q.scale_mma,
        value_order="trellis_native_mma",
    )
    gate_out = torch.empty((routes, intermediate), dtype=torch.float16, device=device)
    up_out = torch.empty_like(gate_out)
    run_trellis_w4a8_fc1_routes(
        gate_q,
        up_q,
        prepared,
        route_experts,
        gate_out,
        up_out,
        **decode_kwargs,
    )

    gate_xq = _reference_mxfp8_rows(gate_x)
    up_xq = _reference_mxfp8_rows(up_x)
    expected_gate = torch.stack(
        [
            gate_xq[r] @ w13_reference[0, int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    expected_up = torch.stack(
        [
            up_xq[r] @ w13_reference[1, int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    torch.testing.assert_close(gate_out, expected_gate, rtol=2.0e-3, atol=2.0e-2)
    torch.testing.assert_close(up_out, expected_up, rtol=2.0e-3, atol=2.0e-2)

    down_x = (torch.randn((routes, intermediate), device=device) * 0.125).half()
    down_q = empty_mxfp8_rows_for_dense_gemm(routes, intermediate, device=device)
    quantize_mxfp8_rows_cute(
        down_x,
        down_q.values,
        down_q.scale_rows,
        down_q.scale_mma,
        value_order="trellis_native_mma",
    )
    down_out = torch.empty((routes, hidden), dtype=torch.float16, device=device)
    run_trellis_w4a8_fc2_routes(
        down_q, prepared, route_experts, down_out, **decode_kwargs
    )
    down_xq = _reference_mxfp8_rows(down_x)
    expected_down = torch.stack(
        [
            down_xq[r] @ w2_reference[int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    torch.testing.assert_close(down_out, expected_down, rtol=2.0e-3, atol=2.0e-2)

    fc1_gate_before = gate_out.clone()
    fc1_up_before = up_out.clone()
    fc2_before = down_out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_trellis_w4a8_fc1_routes(
            gate_q,
            up_q,
            prepared,
            route_experts,
            gate_out,
            up_out,
            **decode_kwargs,
        )
        run_trellis_w4a8_fc2_routes(
            down_q, prepared, route_experts, down_out, **decode_kwargs
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(gate_out, fc1_gate_before)
    assert torch.equal(up_out, fc1_up_before)
    assert torch.equal(down_out, fc2_before)

    # Compose the transforms, both MXFP8 quantizers, direct FC1/FC2 decode,
    # and the route-weighted inverse rotation into one graph-safe MoE path.
    source = (torch.randn((1, hidden), device=device) * 0.125).bfloat16()
    topk_ids = route_experts.reshape(1, -1)
    topk_weights = torch.tensor([[0.625, 0.375]], dtype=torch.float32, device=device)
    scratch = make_trellis_w4a8_moe_scratch(
        m=1,
        topk=2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        device=device,
    )
    actual = run_trellis_w4a8_moe(
        source, prepared, topk_weights, topk_ids, scratch
    ).clone()
    torch.cuda.synchronize(device)

    source_routes = source.to(torch.float16).repeat_interleave(2, dim=0)
    gate_rot_ref = _hadamard128_reference(
        (source_routes.float() * prepared.gate_suh[0].float()).half()
    ).half()
    up_rot_ref = _hadamard128_reference(
        (source_routes.float() * prepared.up_suh[0].float()).half()
    ).half()
    gate_quant_ref = _reference_mxfp8_rows(gate_rot_ref)
    up_quant_ref = _reference_mxfp8_rows(up_rot_ref)
    gate_fc1_ref = torch.stack(
        [
            gate_quant_ref[r] @ w13_reference[0, int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    up_fc1_ref = torch.stack(
        [
            up_quant_ref[r] @ w13_reference[1, int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    rotations = prepared.intermediate_rotations[route_experts.long()]
    gate_canonical = (
        _hadamard128_reference(gate_fc1_ref) * rotations[:, :intermediate].float()
    )
    up_canonical = (
        _hadamard128_reference(up_fc1_ref)
        * rotations[:, intermediate : 2 * intermediate].float()
    )
    activated_ref = _hadamard128_reference(
        SITU_DEFAULT_BETA
        * torch.tanh(gate_canonical / SITU_DEFAULT_BETA)
        * torch.sigmoid(gate_canonical)
        * SITU_DEFAULT_LINEAR_BETA
        * torch.tanh(up_canonical / SITU_DEFAULT_LINEAR_BETA)
        * rotations[:, 2 * intermediate : 3 * intermediate].float()
    ).half()
    activated_quant_ref = _reference_mxfp8_rows(activated_ref)
    fc2_ref = torch.stack(
        [
            activated_quant_ref[r] @ w2_reference[int(route_experts[r])].float()
            for r in range(routes)
        ]
    ).half()
    canonical_routes = _hadamard128_reference(fc2_ref)
    expected = (
        canonical_routes * prepared.down_svh[0].float() * topk_weights.reshape(-1, 1)
    ).sum(dim=0, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=4.0e-3, atol=3.0e-2)

    graph_output_before = actual.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run_trellis_w4a8_moe(
            source, prepared, topk_weights, topk_ids, scratch
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(graph_output, graph_output_before)

    global_ids = torch.tensor([[4, 7]], dtype=torch.int32, device=device)
    expert_map = torch.full((8,), -1, dtype=torch.int32, device=device)
    expert_map[4] = 0
    expert_map[7] = 1
    mapped = run_trellis_w4a8_moe(
        source,
        prepared,
        topk_weights,
        global_ids,
        scratch,
        expert_map=expert_map,
    ).clone()
    torch.cuda.synchronize(device)
    assert torch.equal(mapped, actual)

    # A route owned by the other hybrid tier maps to -1 and contributes zero.
    partial_global_ids = torch.tensor([[4, 3]], dtype=torch.int32, device=device)
    partial = run_trellis_w4a8_moe(
        source,
        prepared,
        topk_weights,
        partial_global_ids,
        scratch,
        expert_map=expert_map,
    ).clone()
    partial_expected = (
        canonical_routes[0] * prepared.down_svh[0].float() * topk_weights[0, 0]
    ).reshape(1, -1)
    torch.testing.assert_close(partial, partial_expected, rtol=4.0e-3, atol=3.0e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped_graph_output = run_trellis_w4a8_moe(
            source,
            prepared,
            topk_weights,
            global_ids,
            scratch,
            expert_map=expert_map,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(mapped_graph_output, mapped)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_prepare_pair_weight_rejects_wrong_tp12_payload_length() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.empty(172032 - 1, dtype=torch.int16, device=device)
    suh = torch.ones(3584, dtype=torch.float16, device=device)
    svh = torch.ones(256, dtype=torch.float16, device=device)

    with pytest.raises(ValueError, match="payload length mismatch"):
        trellis_linear.prepare_pair_weight(
            payload,
            suh,
            svh,
            pair_kind="P24",
            rate_axis="n",
        )
