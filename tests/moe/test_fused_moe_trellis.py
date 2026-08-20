from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe import fused_moe
from b12x.moe.fused_moe import META as FUSED_MOE_META
from b12x.moe._shared.kernels.w4a16.host import plan_w4a16_buffers
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.prepare import (
    PreparedW4A16MoeWeights,
    _prepare_qsrt_coupled_h308_atom_v2_moe_weights,
    _prepare_qsrt_p22_atom_v2_moe_weights,
    _prepare_qsrt_p33_p43_moe_weights,
    prepare_qsrt_atom_moe_weights,
    prepare_qsrt_pair_moe_weights,
    prepare_trellis256_moe_weights,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
)
from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_direct_lut_cpu

_MCG = np.uint64(0xCBAC1FED)
_MCG_MASK = np.uint32(0x8FFF8FFF)
_MCG_OR = np.uint32(0x3B603B60)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


def _weight_plan(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    input_dtype: torch.dtype,
    activation: str = "silu",
    trellis_bits: int = 3,
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    storage_format: str | None = None,
    qsrt_profile: str | None = None,
    coupled_hadamard: bool | None = None,
    source_format: str = "exl3_trellis_mcg",
) -> fused_moe.WeightsPlan:
    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format=source_format,
        activation=activation,
        params_dtype=input_dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w13",
        trellis_bits=trellis_bits,
        trellis_tile_config=tile_config,
        qsrt_storage_format=storage_format,
        qsrt_profile=qsrt_profile,
        coupled_hadamard=coupled_hadamard,
    )


def _caps(**overrides) -> fused_moe.Caps:
    values = {
        "max_tokens": 32,
        "num_topk": 8,
        "num_experts": 192,
        "hidden_size": 6144,
        "intermediate_size": 512,
        "route_num_experts": 256,
        "w4a16_block_size_m": 8,
        "input_dtype": torch.bfloat16,
        "device": "cuda:0",
    }
    values.update(overrides)
    weight_plan = _weight_plan(
        num_experts=values.pop("num_experts"),
        hidden_size=values.pop("hidden_size"),
        intermediate_size=values.pop("intermediate_size"),
        input_dtype=values.pop("input_dtype"),
    )
    return fused_moe.Caps(
        weight_plan=weight_plan,
        quant_mode="w4a16",
        **values,
    )


def test_fused_moe_metadata_advertises_trellis_input_dtypes() -> None:
    assert {"bf16", "fp16"}.issubset(FUSED_MOE_META.dtypes)


def test_exl3_trellis_plan_accepts_k3_and_rejects_k2() -> None:
    assert _weight_plan(
        num_experts=1,
        hidden_size=128,
        intermediate_size=128,
        input_dtype=torch.bfloat16,
        trellis_bits=3,
        tile_config=(64, 128, 64, 128),
    ).trellis_bits == 3
    with pytest.raises(ValueError, match=r"\(3, 4, 5, 6\)"):
        _weight_plan(
            num_experts=1,
            hidden_size=128,
            intermediate_size=128,
            input_dtype=torch.bfloat16,
            trellis_bits=2,
            tile_config=(64, 128, 64, 128),
        )


def test_qsrt_atom_plan_is_explicit_and_fail_closed() -> None:
    plan = _weight_plan(
        num_experts=2,
        hidden_size=256,
        intermediate_size=256,
        input_dtype=torch.float16,
        source_format="qsrt_sqg_e4m3",
        storage_format="qsrt_atoms_v1",
    )
    assert plan.qsrt_storage_format == "qsrt_atoms_v1"
    assert plan.trellis_bits == 3

    with pytest.raises(ValueError, match="intermediate_size=256"):
        _weight_plan(
            num_experts=2,
            hidden_size=256,
            intermediate_size=128,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v1",
        )
    with pytest.raises(ValueError, match="trellis_bits=3"):
        _weight_plan(
            num_experts=2,
            hidden_size=256,
            intermediate_size=256,
            input_dtype=torch.float16,
            trellis_bits=4,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v1",
        )

    v2_plan = _weight_plan(
        num_experts=896,
        hidden_size=3584,
        intermediate_size=256,
        input_dtype=torch.float16,
        source_format="qsrt_sqg_e4m3",
        storage_format="qsrt_atoms_v2",
    )
    assert v2_plan.qsrt_storage_format == "qsrt_atoms_v2"


def test_qsrt_coupled_k2_atom_plan_freezes_p22_geometry() -> None:
    plan = _weight_plan(
        num_experts=896,
        hidden_size=3584,
        intermediate_size=384,
        input_dtype=torch.float16,
        source_format="qsrt_sqg_e4m3",
        storage_format="qsrt_atoms_v2",
        qsrt_profile="k2_coupled_h512_h128",
        trellis_bits=2,
        tile_config=(128, 128, 128, 128),
    )
    assert plan.qsrt_profile == "k2_coupled_h512_h128"
    assert plan.trellis_bits == 2
    assert plan.trellis_tile_config == (128, 128, 128, 128)
    assert plan.coupled_hadamard

    with pytest.raises(ValueError, match="requires trellis_bits=2"):
        _weight_plan(
            num_experts=896,
            hidden_size=3584,
            intermediate_size=384,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v2",
            qsrt_profile="k2_coupled_h512_h128",
            trellis_bits=3,
            tile_config=(128, 128, 128, 128),
        )

    with pytest.raises(ValueError, match=r"tile_config=\(128, 128, 128, 128\)"):
        _weight_plan(
            num_experts=896,
            hidden_size=3584,
            intermediate_size=384,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v2",
            qsrt_profile="k2_coupled_h512_h128",
            trellis_bits=2,
            tile_config=(64, 128, 64, 128),
        )


def test_qsrt_coupled_h308_atom_plan_freezes_geometry() -> None:
    plan = _weight_plan(
        num_experts=896,
        hidden_size=3584,
        intermediate_size=256,
        input_dtype=torch.float16,
        source_format="qsrt_sqg_e4m3",
        storage_format="qsrt_atoms_v2",
        qsrt_profile="k3x22_k4x2_coupled_h512_h128",
        trellis_bits=3,
        tile_config=(64, 256, 64, 256),
    )
    assert plan.qsrt_profile == "k3x22_k4x2_coupled_h512_h128"
    assert plan.trellis_bits == 3
    assert plan.trellis_tile_config == (64, 256, 64, 256)
    assert plan.coupled_hadamard

    with pytest.raises(ValueError, match="requires trellis_bits=3"):
        _weight_plan(
            num_experts=896,
            hidden_size=3584,
            intermediate_size=256,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v2",
            qsrt_profile="k3x22_k4x2_coupled_h512_h128",
            trellis_bits=2,
            tile_config=(64, 256, 64, 256),
        )

    with pytest.raises(ValueError, match="coupled_hadamard flag must agree"):
        _weight_plan(
            num_experts=896,
            hidden_size=3584,
            intermediate_size=256,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="qsrt_atoms_v2",
            qsrt_profile="k3x22_k4x2_coupled_h512_h128",
            trellis_bits=3,
            tile_config=(64, 256, 64, 256),
            coupled_hadamard=False,
        )


def test_qsrt_storage_format_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="qsrt_storage_format"):
        _weight_plan(
            num_experts=2,
            hidden_size=256,
            intermediate_size=256,
            input_dtype=torch.float16,
            source_format="qsrt_sqg_e4m3",
            storage_format="not_a_qsrt_format",
        )


def test_rotation_placeholder_must_be_materialized_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    device = torch.device("cuda:127")
    w4a16_kernel._ROT_SCALES_DUMMY.pop(device, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prewarm the planned launch"):
        w4a16_kernel._rot_scales_dummy(device)


def test_dense_scratch_cannot_allocate_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="provide caller-owned storage"):
        w4a16_kernel._trellis_dense_buffer(
            "rotated_f16",
            None,
            shape=(2, 128),
            dtype=torch.float16,
            device=torch.device("cpu"),
        )


def test_caps_preserve_exact_block_m_and_input_dtypes() -> None:
    bf16 = _caps(w4a16_block_size_m=8, input_dtype=torch.bfloat16)
    fp16 = _caps(w4a16_block_size_m=64, input_dtype=torch.float16)

    assert bf16.w4a16_block_size_m == 8
    assert bf16.route_num_experts == 256
    assert bf16.dtype == torch.bfloat16
    assert fp16.w4a16_block_size_m == 64
    assert fp16.dtype == torch.float16


@pytest.mark.parametrize("block_size_m", [0, 4, 12, 128])
def test_caps_reject_unsupported_block_m(block_size_m: int) -> None:
    with pytest.raises(ValueError, match="w4a16_block_size_m"):
        _caps(w4a16_block_size_m=block_size_m)


def _prepare_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    input_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    codebook: str = "mcg",
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
) -> fused_moe.ExpertWeights:
    bits = int(w13.shape[-1]) // 16
    if codebook != "mcg":
        raise ValueError(f"unsupported EXL3 codebook {codebook!r}")
    plan = _weight_plan(
        num_experts=int(w13.shape[1]),
        hidden_size=int(w13.shape[2]) * 16,
        intermediate_size=int(w13.shape[3]) * 16,
        input_dtype=input_dtype,
        activation=activation,
        trellis_bits=bits,
        tile_config=tile_config,
        source_format="exl3_trellis_mcg",
    )
    return fused_moe.prepare_weights(
        plan=plan,
        params_dtype=input_dtype,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        trellis_mcg=0xCBAC1FED,
    )


def _plan(
    weights: fused_moe.ExpertWeights,
    *,
    max_tokens: int,
    num_topk: int,
    route_num_experts: int,
    block_size_m: int,
    device: torch.device | str,
) -> fused_moe.Plan:
    return fused_moe.plan(
        fused_moe.Caps(
            max_tokens=max_tokens,
            num_topk=num_topk,
            route_num_experts=route_num_experts,
            device=device,
            weight_plan=weights.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=block_size_m,
        )
    )


def test_low_level_buffer_plan_honors_explicit_block_m() -> None:
    prepared = SimpleNamespace(
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        is_gated=True,
    )
    automatic = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
    )
    exact = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
        full_rotation=True,
        block_size_m=8,
    )

    assert automatic.block_size_m != 8
    assert exact.block_size_m == 8
    assert exact.rotation_a_elements == 1024 * 8 * 6144
    with pytest.raises(ValueError, match="block_size_m"):
        plan_w4a16_buffers(
            prepared,
            m=1,
            topk=8,
            route_num_experts=256,
            sms=120,
            block_size_m=4,
        )


def test_planned_route_block_overrides_live_batch_heuristic() -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _resolve_route_block_size_m,
    )

    # m=1024/topk=8/E=256 selects 48 heuristically. A caller-owned arena
    # planned for 64 must retain 64 so route packing cannot outgrow its
    # block-expert table.
    assert _resolve_route_block_size_m(
        m=1024,
        topk=8,
        route_num_experts=256,
        planned_block_size_m=None,
        fused_launch=None,
    ) == 48
    assert _resolve_route_block_size_m(
        m=1024,
        topk=8,
        route_num_experts=256,
        planned_block_size_m=64,
        fused_launch=None,
    ) == 64
    with pytest.raises(RuntimeError, match="planned_block_size_m=64"):
        _resolve_route_block_size_m(
            m=1024,
            topk=8,
            route_num_experts=256,
            planned_block_size_m=64,
            fused_launch=SimpleNamespace(moe_block_size=48),
        )


def _cpu_weight_tensors(num_experts: int = 1) -> tuple[torch.Tensor, ...]:
    w13 = torch.zeros((2, num_experts, 8, 8, 48), dtype=torch.int16)
    w2 = torch.zeros((num_experts, 8, 8, 48), dtype=torch.int16)
    edge = torch.ones((1, 128), dtype=torch.float16)
    intermediate = torch.ones((num_experts, 384), dtype=torch.float16)
    return w13, w2, edge, edge.clone(), intermediate, edge.clone()


def test_prepare_weights_rejects_unsupported_codebook_before_cuda_work() -> None:
    w13, w2, gate_suh, up_suh, intermediate, down_svh = _cpu_weight_tensors()
    with pytest.raises(ValueError, match="unsupported EXL3 codebook"):
        _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            codebook="legacy-codebook",
            tile_config=(64, 128, 64, 128),
        )


def test_exl3_prepare_requires_cuda_payload_storage() -> None:
    w13, w2, gate_suh, up_suh, intermediate, down_svh = _cpu_weight_tensors()
    with pytest.raises(ValueError, match="require CUDA storage"):
        _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            tile_config=(64, 128, 64, 128),
        )


def test_prepare_weights_validates_all_rotation_shapes() -> None:
    w13, w2, gate_suh, up_suh, intermediate, down_svh = _cpu_weight_tensors()
    with pytest.raises(ValueError, match="intermediate_rotations must have shape"):
        _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate[:, :-1].contiguous(),
            down_svh=down_svh,
            tile_config=(64, 128, 64, 128),
        )

    w13, w2, gate_suh, _, intermediate, down_svh = _cpu_weight_tensors(2)
    with pytest.raises(ValueError, match="must both be per-expert or both broadcast"):
        _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=gate_suh.expand(2, -1).contiguous(),
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            tile_config=(64, 128, 64, 128),
        )


@lru_cache(maxsize=None)
def _sqg_xor_cheb_t12_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported QSRT test rate K{bits}")
    labels = sqg_xor_cheb_t12_direct_lut_cpu()
    rate_labels = labels[(bits - 2) << 16 : (bits - 1) << 16]
    return rate_labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


@lru_cache(maxsize=None)
def _sqg_fp16_d3l_table(bits: int) -> np.ndarray:
    return sqg_fp16_d3l_direct_lut_cpu(bits).numpy()


def _decode_mcg_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MCG_MASK) ^ _MCG_OR)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_lane(
    tile_words: np.ndarray,
    lane: int,
    bits: int,
    *,
    codebook: str = "mcg",
) -> np.ndarray:
    if codebook not in {"mcg", "sqg_xor_cheb_t12", "sqg_fp16_d3l"}:
        raise ValueError(f"unsupported test codebook {codebook!r}")
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
            values.append(_decode_mcg_fp16(window))
        elif codebook == "sqg_xor_cheb_t12":
            values.append(_sqg_xor_cheb_t12_table(bits)[window])
        else:
            values.append(_sqg_fp16_d3l_table(bits)[window])
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(
    trellis: torch.Tensor,
    *,
    codebook: str = "mcg",
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
                    _decode_lane(
                        words[k_tile, n_tile],
                        lane,
                        bits,
                        codebook=codebook,
                    )
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
                        rows[weight % 4], 2 * (col0 if weight < 4 else col1) + parity
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [5, 6])
def test_sqg_fp16_d3l_uniform_w4a16_matches_reference_and_captures(
    bits: int,
) -> None:
    """Close native K5/K6 payloads through the actual A16 fused kernel."""

    torch.manual_seed(20260808 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = 1
    m = 2
    hidden = intermediate = 128
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((experts, hidden)).contiguous()
    up_suh = scales((experts, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((experts, hidden)).contiguous()
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=128,
        fc2_tile_n=128,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        codebook="sqg_fp16_d3l",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=(64, 128, 64, 128),
    )
    assert prepared.source_format == "sqg_fp16_d3l"
    assert prepared.trellis_codebook == "sqg_fp16_d3l"

    gate_weights = _reconstruct_native(
        w13[0, 0], codebook="sqg_fp16_d3l"
    ).unsqueeze(0).to(device)
    up_weights = _reconstruct_native(
        w13[1, 0], codebook="sqg_fp16_d3l"
    ).unsqueeze(0).to(device)
    down_weights = _reconstruct_native(
        w2[0], codebook="sqg_fp16_d3l"
    ).unsqueeze(0).to(device)
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            router_weights,
            ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=intermediate_rotations,
            full_rotation=True,
            suh_gate_table=gate_suh,
            suh_up_table=up_suh,
            svh_table=down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)

def _hadamard_128(device: torch.device) -> torch.Tensor:
    indices = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(col)).bit_count() & 1 for col in indices],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(device)


def _had128(
    value: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    work = (work.view(rows, width // 128, 128) @ hadamard).view(rows, width)
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


def _coupled_h512(value: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Apply the normalized H512 used at the coupled residual boundary."""

    work = value.float()
    rows, width = work.shape
    work = work.view(rows, width // 512, 4, 128) @ hadamard
    h4 = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=torch.float32,
        device=value.device,
    ) * 0.5
    work = torch.einsum("rbgc,gh->rbhc", work, h4)
    return work.reshape(rows, width)


def _reference_full_rotation(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    device = x.device
    experts = int(w2.shape[0])
    intermediate = int(w2.shape[1]) * 16
    hidden = int(w2.shape[2]) * 16
    hadamard = _hadamard_128(device)
    gate_weights = torch.stack(
        [_reconstruct_native(w13[0, expert]) for expert in range(experts)]
    ).to(device)
    up_weights = torch.stack(
        [_reconstruct_native(w13[1, expert]) for expert in range(experts)]
    ).to(device)
    down_weights = torch.stack(
        [_reconstruct_native(w2[expert]) for expert in range(experts)]
    ).to(device)
    return _reference_full_rotation_decoded(
        x,
        local_ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation=activation,
    )


def _reference_full_rotation_decoded(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    device = x.device
    experts = int(down_weights.shape[0])
    intermediate = int(down_weights.shape[1])
    hidden = int(down_weights.shape[2])
    hadamard = _hadamard_128(device)
    gate_svh = intermediate_rotations[:, :intermediate]
    up_svh = intermediate_rotations[:, intermediate : 2 * intermediate]
    down_suh = intermediate_rotations[:, 2 * intermediate :]
    output = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=device)
    for token in range(int(x.shape[0])):
        for slot in range(int(local_ids.shape[1])):
            expert = int(local_ids[token, slot])
            h_side_expert = 0 if int(gate_suh.shape[0]) == 1 else expert
            source = x[token : token + 1].to(torch.float16)
            gate_a = _had128(
                source,
                hadamard,
                suh=gate_suh[h_side_expert],
                store_fp16=True,
            )
            up_a = _had128(
                source,
                hadamard,
                suh=up_suh[h_side_expert],
                store_fp16=True,
            )
            gate = (gate_a.float() @ gate_weights[expert].float()).to(torch.float16)
            up = (up_a.float() @ up_weights[expert].float()).to(torch.float16)
            gate = _had128(
                gate,
                hadamard,
                svh=gate_svh[expert],
                store_fp16=True,
            )
            up = _had128(
                up,
                hadamard,
                svh=up_svh[expert],
                store_fp16=True,
            )
            if activation == "situ":
                activated = (
                    4.0
                    * torch.tanh(gate.float() / 4.0)
                    * torch.sigmoid(gate.float())
                    * 25.0
                    * torch.tanh(up.float() / 25.0)
                ).to(torch.float16)
            else:
                activated = (
                    gate.float() * torch.sigmoid(gate.float()) * up.float()
                ).to(torch.float16)
            down_a = _had128(
                activated,
                hadamard,
                suh=down_suh[expert],
                store_fp16=True,
            )
            down = (down_a.float() @ down_weights[expert].float()).to(torch.float16)
            route = _had128(
                down,
                hadamard,
                svh=down_svh[0 if int(down_svh.shape[0]) == 1 else expert],
                store_fp16=False,
            )
            output[token] += router_weights[token, slot] * route[0]
    return output


def _reference_coupled_decoded(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    slot0_weights: torch.Tensor,
    slot1_weights: torch.Tensor,
    down_weights: torch.Tensor,
    source_suh: torch.Tensor,
    ordinary_rotations: torch.Tensor,
    coupled_signs: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    quantize_activations: bool = False,
) -> torch.Tensor:
    """Reference the exact transform order of a coupled uniform-rate path."""

    device = x.device
    experts = int(down_weights.shape[0])
    intermediate = int(down_weights.shape[1])
    hidden = int(down_weights.shape[2])
    hadamard = _hadamard_128(device)
    slot0_svh = ordinary_rotations[:, :intermediate]
    slot1_svh = ordinary_rotations[:, intermediate : 2 * intermediate]
    down_suh = ordinary_rotations[:, 2 * intermediate :]
    pre_signs = coupled_signs[:, : 2 * intermediate]
    post_signs = coupled_signs[:, 2 * intermediate :]
    output = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=device)

    for token in range(int(x.shape[0])):
        route_values: list[torch.Tensor] = []
        for slot in range(int(local_ids.shape[1])):
            expert = int(local_ids[token, slot])
            h_side_expert = 0 if int(source_suh.shape[0]) == 1 else expert
            source = _coupled_h512(x[token : token + 1].to(torch.float16), hadamard)
            if quantize_activations:
                source = _had128(
                    source * source_suh[h_side_expert].float(),
                    hadamard,
                    store_fp16=False,
                )
                source = _reference_mxfp8_rows(source)
            else:
                source = (source * source_suh[h_side_expert].float()).to(
                    torch.float16
                )
                source = _had128(source, hadamard, store_fp16=True)

            atoms = intermediate // 32
            physical = [
                (source.float() @ weights[expert].float()).to(torch.float16)
                for weights in (slot0_weights, slot1_weights)
            ]
            raw_pre = torch.stack(
                (
                    physical[0].view(1, atoms, 32),
                    physical[1].view(1, atoms, 32),
                ),
                dim=2,
            ).reshape(1, 2 * intermediate)
            pre_scales = torch.stack(
                (
                    slot0_svh[expert].view(atoms, 32),
                    slot1_svh[expert].view(atoms, 32),
                ),
                dim=1,
            ).reshape(1, 2 * intermediate)
            pre = _had128(raw_pre, hadamard, store_fp16=False)
            pre = _had128(
                pre * pre_scales.float(),
                hadamard,
                store_fp16=False,
            )
            pre = pre * pre_signs[expert].float()
            gate = pre[:, 0::2]
            up = pre[:, 1::2]
            activated = (
                4.0
                * torch.tanh(gate / 4.0)
                * torch.sigmoid(gate)
                * 25.0
                * torch.tanh(up / 25.0)
            )
            activated = activated * post_signs[expert].float()
            activated = _had128(activated, hadamard, store_fp16=False)
            activated = _had128(
                activated * down_suh[expert].float(),
                hadamard,
                store_fp16=not quantize_activations,
            )
            if quantize_activations:
                activated = _reference_mxfp8_rows(activated)
            down = (
                activated.float() @ down_weights[expert].float()
            ).to(torch.float16)
            route = _had128(down, hadamard, store_fp16=False)
            route = route * down_svh[0 if int(down_svh.shape[0]) == 1 else expert]
            route_values.append(route)

        routed = sum(
            router_weights[token, slot] * route_values[slot]
            for slot in range(int(local_ids.shape[1]))
        )
        output[token] = _coupled_h512(routed, hadamard)[0]
    return output


def _reference_mxfp8_rows(source: torch.Tensor) -> torch.Tensor:
    """Quantize logical K32 groups to E4M3 with UE8M0 power-of-two scales."""

    m, k = source.shape
    blocks = source.float().reshape(m, k // 32, 32)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, device=source.device), exponent)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    return (
        (blocks / scale)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scale)
        .reshape(m, k)
    )


def _qsrt_p22_atom_extent_from_payloads(
    w13: torch.Tensor,
    w2: torch.Tensor,
    ordinary_rotations: torch.Tensor,
) -> torch.Tensor:
    """Pack one TP-local pure-K2 extent in canonical atoms-v2 order."""

    projections, experts, hidden_tiles, local_tiles, words = w13.shape
    if projections != 2 or words != 32 or local_tiles % 2:
        raise ValueError("unexpected pure-K2 upstream payload shape")
    if tuple(w2.shape) != (experts, local_tiles, hidden_tiles, words):
        raise ValueError("unexpected pure-K2 down payload shape")
    atoms = local_tiles // 2
    intermediate = local_tiles * 16
    if tuple(ordinary_rotations.shape) != (experts, 3 * intermediate):
        raise ValueError("unexpected pure-K2 local-scale shape")
    matrix_bytes = hidden_tiles * 64 * torch.int16.itemsize
    bundle_bytes = 3 * matrix_bytes + 3 * 64
    output = torch.zeros(
        (atoms, experts, bundle_bytes), dtype=torch.uint8, device=w13.device
    )

    for matrix_index in range(3):
        if matrix_index < 2:
            source = (
                w13[matrix_index]
                .reshape(experts, hidden_tiles, atoms, 2, 32)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, matrix_bytes)
            )
        else:
            source = (
                w2.reshape(experts, atoms, 2, hidden_tiles, 32)
                .permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, matrix_bytes)
            )
        begin = matrix_index * matrix_bytes
        output[:, :, begin : begin + matrix_bytes].copy_(source)

        scale_source = (
            ordinary_rotations[
                :,
                matrix_index * intermediate : (matrix_index + 1) * intermediate,
            ]
            .reshape(experts, atoms, 32)
            .permute(1, 0, 2)
            .contiguous()
            .view(torch.uint8)
            .reshape(atoms, experts, 64)
        )
        scale_begin = 3 * matrix_bytes + matrix_index * 64
        output[:, :, scale_begin : scale_begin + 64].copy_(scale_source)
    return output.reshape(atoms, experts * bundle_bytes).contiguous()


def _qsrt_coupled_h308_atom_extent_from_payloads(
    w13_low: torch.Tensor,
    w13_high: torch.Tensor,
    w2_low: torch.Tensor,
    w2_high: torch.Tensor,
    ordinary_rotations: torch.Tensor,
) -> torch.Tensor:
    """Pack one static coupled-H308 pair in canonical atoms-v2 order."""

    projections, experts, hidden_tiles, atoms, low_words = w13_low.shape
    if projections != 2 or atoms != 8 or low_words % 16:
        raise ValueError("unexpected coupled-H308 upstream payload shape")
    high_words = int(w13_high.shape[-1])
    if tuple(w13_high.shape[:-1]) != tuple(w13_low.shape[:-1]) or high_words % 16:
        raise ValueError("coupled-H308 upstream halves disagree")
    w2_low_words = int(w2_low.shape[-1])
    w2_high_words = int(w2_high.shape[-1])
    if tuple(w2_low.shape[:-1]) != (experts, atoms, hidden_tiles) or w2_low_words % 16:
        raise ValueError("unexpected coupled-H308 down low-half shape")
    if (
        tuple(w2_high.shape[:-1]) != (experts, atoms, hidden_tiles)
        or w2_high_words % 16
    ):
        raise ValueError("unexpected coupled-H308 down high-half shape")
    intermediate = 2 * atoms * 16
    if tuple(ordinary_rotations.shape) != (experts, 3 * intermediate):
        raise ValueError("unexpected coupled-H308 local-scale shape")

    fc1_low_bytes = hidden_tiles * low_words * torch.int16.itemsize
    fc1_high_bytes = hidden_tiles * high_words * torch.int16.itemsize
    fc1_matrix_bytes = fc1_low_bytes + fc1_high_bytes
    fc2_low_bytes = hidden_tiles * w2_low_words * torch.int16.itemsize
    fc2_high_bytes = hidden_tiles * w2_high_words * torch.int16.itemsize
    fc2_matrix_bytes = fc2_low_bytes + fc2_high_bytes
    bundle_bytes = 2 * fc1_matrix_bytes + fc2_matrix_bytes + 3 * 64
    output = torch.empty(
        (atoms, experts, bundle_bytes), dtype=torch.uint8, device=w13_low.device
    )

    cursor = 0
    for matrix_index in range(3):
        if matrix_index < 2:
            low_bytes = fc1_low_bytes
            high_bytes = fc1_high_bytes
            matrix_bytes = fc1_matrix_bytes
            low = (
                w13_low[matrix_index]
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, low_bytes)
            )
            high = (
                w13_high[matrix_index]
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, high_bytes)
            )
        else:
            low_bytes = fc2_low_bytes
            high_bytes = fc2_high_bytes
            matrix_bytes = fc2_matrix_bytes
            low = (
                w2_low.permute(1, 0, 2, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, low_bytes)
            )
            high = (
                w2_high.permute(1, 0, 2, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(atoms, experts, high_bytes)
            )
        output[:, :, cursor : cursor + low_bytes].copy_(low)
        output[
            :, :, cursor + low_bytes : cursor + matrix_bytes
        ].copy_(high)
        cursor += matrix_bytes

    for matrix_index in range(3):
        values = ordinary_rotations[
            :,
            matrix_index * intermediate : (matrix_index + 1) * intermediate,
        ]
        stripes = torch.cat(
            (
                values[:, :128].reshape(experts, atoms, 16),
                values[:, 128:].reshape(experts, atoms, 16),
            ),
            dim=2,
        ).permute(1, 0, 2)
        output[:, :, cursor : cursor + 64].copy_(
            stripes.contiguous().view(torch.uint8).reshape(atoms, experts, 64)
        )
        cursor += 64
    assert cursor == bundle_bytes
    return output.reshape(atoms, experts * bundle_bytes).contiguous()


def _qsrt_coupled_signs_reference(
    length: int, *, draw: int, axis: int, device: torch.device
) -> torch.Tensor:
    if draw == 0:
        return torch.ones(length, dtype=torch.float16, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (
            0x6A09E667F3BCC909 * draw
            + 0xBB67AE8584CAA73B * axis
        )
        & ((1 << 63) - 1)
    )
    return (
        torch.randint(0, 2, (length,), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(device=device, dtype=torch.float16)
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("first_atom_slot", [0, 48])
def test_coupled_k2_fused_moe_matches_transform_reference_and_captures(
    first_atom_slot: int,
) -> None:
    """Close uniform-K2 payloads through every coupled transform boundary."""

    torch.manual_seed(20260810)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 1
    # Keep this closure at Kimi-K3's TP8 serving shape: seven H512 residual
    # blocks and a 3,072 / 8 = 384-wide local intermediate extent.  This makes
    # the coupled H128 boundaries, atom reassembly, and FC2 traversal cross
    # multiple blocks and local tile groups.
    hidden, intermediate, bits = 3584, 384, 2
    tile_config = (128, 128, 128, 128)
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    assert not torch.equal(gate_suh, up_suh)
    source_suh = gate_suh if first_atom_slot < 48 else up_suh
    ordinary_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    draw = 6
    pre_sign_begin = 2 * first_atom_slot * 32
    post_sign_begin = first_atom_slot * 32
    coupled_signs = torch.cat(
        (
            _qsrt_coupled_signs_reference(
                2 * 3072, draw=draw, axis=1, device=device
            )[pre_sign_begin : pre_sign_begin + 2 * intermediate],
            _qsrt_coupled_signs_reference(
                3072, draw=draw, axis=2, device=device
            )[post_sign_begin : post_sign_begin + intermediate],
        )
    ).reshape(experts, 3 * intermediate)
    # Serving keeps the canonical atom extent host-backed and streams bounded
    # expert groups into the rank-local CUDA representation during load.
    atom_payload = _qsrt_p22_atom_extent_from_payloads(
        w13, w2, ordinary_rotations
    ).cpu()
    prepared = _prepare_qsrt_p22_atom_v2_moe_weights(
        atom_payload,
        first_atom_slot=first_atom_slot,
        rotation_draws=torch.tensor([draw], dtype=torch.uint8),
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
        params_dtype=torch.float16,
        codebook="sqg_xor_cheb_t12",
        tile_config=tile_config,
        dummy_scale=None,
        workspace=None,
    )
    torch.testing.assert_close(
        prepared.w13.view(torch.int16).view_as(w13), w13, rtol=0, atol=0
    )
    torch.testing.assert_close(
        prepared.w2.view(torch.int16).view_as(w2), w2, rtol=0, atol=0
    )
    torch.testing.assert_close(prepared.gate_suh, source_suh, rtol=0, atol=0)
    torch.testing.assert_close(prepared.up_suh, source_suh, rtol=0, atol=0)
    torch.testing.assert_close(
        prepared.intermediate_rotations[:, : 3 * intermediate],
        ordinary_rotations,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        prepared.intermediate_rotations[:, 3 * intermediate :],
        coupled_signs,
        rtol=0,
        atol=0,
    )

    slot0_weights = torch.stack(
        [
            _reconstruct_native(w13[0, expert], codebook="sqg_xor_cheb_t12")
            for expert in range(experts)
        ]
    ).to(device)
    slot1_weights = torch.stack(
        [
            _reconstruct_native(w13[1, expert], codebook="sqg_xor_cheb_t12")
            for expert in range(experts)
        ]
    ).to(device)
    down_weights = torch.stack(
        [
            _reconstruct_native(w2[expert], codebook="sqg_xor_cheb_t12")
            for expert in range(experts)
        ]
    ).to(device)
    x = torch.randn((m, hidden), device=device).to(torch.bfloat16)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    assert buffers.rotation_a_gate.data_ptr() == buffers.rotation_a_up.data_ptr()

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            router_weights,
            ids,
            activation="situ",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=prepared.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=prepared.gate_suh,
            suh_up_table=prepared.up_suh,
            svh_table=down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    hadamard = _hadamard_128(device)
    expected_source = _coupled_h512(x.to(torch.float16), hadamard)
    expected_source = (expected_source * source_suh.float()).to(torch.float16)
    expected_source = _had128(
        expected_source, hadamard, store_fp16=True
    )
    torch.testing.assert_close(
        buffers.rotation_a_gate.view(-1, hidden)[:m],
        expected_source,
        rtol=1.0e-2,
        atol=5.0e-7,
    )
    physical0 = (expected_source.float() @ slot0_weights[0].float()).to(
        torch.float16
    )
    physical1 = (expected_source.float() @ slot1_weights[0].float()).to(
        torch.float16
    )
    atoms = intermediate // 32
    raw_pre = torch.stack(
        (
            physical0.view(m, atoms, 32),
            physical1.view(m, atoms, 32),
        ),
        dim=2,
    ).reshape(m, 2 * intermediate)
    pre_scales = torch.stack(
        (
            ordinary_rotations[:, :intermediate].view(m, atoms, 32),
            ordinary_rotations[:, intermediate : 2 * intermediate].view(
                m, atoms, 32
            ),
        ),
        dim=2,
    ).reshape(m, 2 * intermediate)
    expected_pre = _had128(raw_pre, hadamard, store_fp16=False)
    expected_pre = _had128(
        expected_pre * pre_scales.float(),
        hadamard,
        store_fp16=False,
    )
    expected_pre *= coupled_signs[:, : 2 * intermediate].float()
    gate = expected_pre[:, 0::2]
    up = expected_pre[:, 1::2]
    expected_activated = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )
    expected_activated *= coupled_signs[:, 2 * intermediate :].float()
    expected_activated = _had128(
        expected_activated, hadamard, store_fp16=False
    )
    expected_activated = _had128(
        expected_activated * ordinary_rotations[:, 2 * intermediate :].float(),
        hadamard,
        store_fp16=True,
    )
    actual_activated = buffers.intermediate_cache2.view(-1, intermediate)[:m]
    activation_relative_error = (
        (actual_activated - expected_activated).norm()
        / expected_activated.norm().clamp_min(1.0e-9)
    )
    assert float(activation_relative_error) <= 2.0e-2
    reference = _reference_coupled_decoded(
        x,
        ids,
        router_weights,
        slot0_weights,
        slot1_weights,
        down_weights,
        source_suh,
        ordinary_rotations,
        coupled_signs,
        down_svh,
    )
    relative_error = (
        (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.9999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)




@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [3, 4])
def test_coupled_higher_rate_w4a16_matches_transform_reference(
    bits: int,
) -> None:
    """Close uniform K3/K4 payloads through the coupled transform contract."""

    torch.manual_seed(20260810 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 1
    hidden, intermediate = 512, 128
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    source_suh = scales((1, hidden)).contiguous()
    ordinary_rotations = scales((experts, 3 * intermediate)).contiguous()
    coupled_signs = (
        torch.randint(
            0,
            2,
            (experts, 3 * intermediate),
            dtype=torch.int32,
            device=device,
        )
        .mul_(2)
        .sub_(1)
        .to(torch.float16)
    )
    down_svh = scales((1, hidden)).contiguous()
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        params_dtype=torch.float16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        codebook="sqg_xor_cheb_t12",
        gate_suh=source_suh,
        up_suh=source_suh,
        intermediate_rotations=ordinary_rotations,
        down_svh=down_svh,
        tile_config=(128, 128, 128, 128),
    )
    prepared = replace(
        prepared,
        coupled_hadamard=True,
        intermediate_rotations=torch.cat(
            (ordinary_rotations, coupled_signs), dim=1
        ).contiguous(),
    )

    slot0_weights = _reconstruct_native(
        w13[0, 0], codebook="sqg_xor_cheb_t12"
    ).unsqueeze(0).to(device)
    slot1_weights = _reconstruct_native(
        w13[1, 0], codebook="sqg_xor_cheb_t12"
    ).unsqueeze(0).to(device)
    down_weights = _reconstruct_native(
        w2[0], codebook="sqg_xor_cheb_t12"
    ).unsqueeze(0).to(device)
    x = torch.randn((m, hidden), dtype=torch.bfloat16, device=device)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)

    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    actual_a16 = run_w4a16_moe(
        x,
        prepared,
        router_weights,
        ids,
        activation="situ",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        route_block_size_m=8,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=source_suh,
        suh_up_table=source_suh,
        svh_table=down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    ).clone()
    reference_a16 = _reference_coupled_decoded(
        x,
        ids,
        router_weights,
        slot0_weights,
        slot1_weights,
        down_weights,
        source_suh,
        ordinary_rotations,
        coupled_signs,
        down_svh,
    )
    relative_a16 = (
        (actual_a16 - reference_a16).norm()
        / reference_a16.norm().clamp_min(1.0e-9)
    )
    cosine_a16 = torch.nn.functional.cosine_similarity(
        actual_a16.flatten(), reference_a16.flatten(), dim=0
    )
    assert float(relative_a16) <= 2.0e-2
    assert float(cosine_a16) >= 0.9999



def _qsrt_atom_extent_from_pair_payloads(
    w13_payload: torch.Tensor,
    w2_payload: torch.Tensor,
    fc1_modes: torch.Tensor,
    fc2_modes: torch.Tensor,
    intermediate_rotations: torch.Tensor,
) -> torch.Tensor:
    """Build the canonical eight-atom view used by preparation tests."""

    _, experts, pair_words = w13_payload.shape
    hidden_tiles = pair_words // (8 * 16 * 6)
    device = w13_payload.device
    atoms = torch.zeros((8, experts, 129_216), dtype=torch.uint8, device=device)

    def store_matrix(
        payload: torch.Tensor,
        modes: torch.Tensor,
        *,
        offset: int,
        fc1: bool,
    ) -> None:
        matrix_atoms = torch.zeros(
            (8, experts, 21_504), dtype=torch.int16, device=device
        )
        for mode, (low_bits, high_bits) in ((0, (3, 3)), (1, (2, 4))):
            ids = torch.nonzero(modes == mode, as_tuple=False).flatten()
            if int(ids.numel()) == 0:
                continue
            selected = payload.index_select(0, ids)
            low_words = hidden_tiles * 8 * 16 * low_bits
            if fc1:
                low = selected[:, :low_words].reshape(
                    -1, hidden_tiles, 8, 16 * low_bits
                ).permute(2, 0, 1, 3)
                high = selected[:, low_words:].reshape(
                    -1, hidden_tiles, 8, 16 * high_bits
                ).permute(2, 0, 1, 3)
            else:
                low = selected[:, :low_words].reshape(
                    -1, 8, hidden_tiles, 16 * low_bits
                ).permute(1, 0, 2, 3)
                high = selected[:, low_words:].reshape(
                    -1, 8, hidden_tiles, 16 * high_bits
                ).permute(1, 0, 2, 3)
            joined = torch.cat(
                (low.reshape(8, ids.numel(), -1), high.reshape(8, ids.numel(), -1)),
                dim=2,
            )
            selected_atoms = matrix_atoms.index_select(1, ids)
            selected_atoms[..., : joined.shape[-1]].copy_(joined)
            matrix_atoms.index_copy_(1, ids, selected_atoms)
        atoms[:, :, offset : offset + 43_008].copy_(
            matrix_atoms.contiguous().view(torch.uint8).reshape(8, experts, 43_008)
        )

    store_matrix(w13_payload[0], fc1_modes, offset=0, fc1=True)
    store_matrix(w13_payload[1], fc1_modes, offset=43_008, fc1=True)
    store_matrix(w2_payload, fc2_modes, offset=86_016, fc1=False)
    for matrix, offset in enumerate((129_024, 129_088, 129_152)):
        values = intermediate_rotations[:, matrix * 256 : (matrix + 1) * 256]
        scale_atoms = torch.cat(
            (
                values[:, :128].reshape(experts, 8, 16),
                values[:, 128:].reshape(experts, 8, 16),
            ),
            dim=2,
        ).permute(1, 0, 2).contiguous()
        atoms[:, :, offset : offset + 64].copy_(
            scale_atoms.view(torch.uint8).reshape(8, experts, 64)
        )
    return atoms


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("pair_case", ["P24", "P33", "PDYNAMIC"])
def test_qsrt_atom_fused_moe_matches_full_rotation_reference_and_captures(
    pair_case: str,
) -> None:
    """Close compact FC1-N/FC2-K pairs through the actual routed kernel."""
    torch.manual_seed(20260801)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 2
    hidden, intermediate = 256, 256
    hidden_tiles = hidden // 16
    if pair_case == "PDYNAMIC":
        fc1_kinds = ["P24", "P33"]
        fc2_kinds = ["P33", "P24"]
        fc1_pair_spec = torch.tensor(
            [1, 0], dtype=torch.int32, device=device
        )
        fc2_pair_spec = torch.tensor(
            [0, 1], dtype=torch.int32, device=device
        )
    else:
        fc1_kinds = [pair_case] * experts
        fc2_kinds = [pair_case] * experts
        mode = 1 if pair_case == "P24" else 0
        fc1_pair_spec = torch.full(
            (experts,), mode, dtype=torch.int32, device=device
        )
        fc2_pair_spec = fc1_pair_spec.clone()

    def make_pair(
        kind: str, shape: tuple[int, int], *, is_fc2: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_bits, high_bits = (2, 4) if kind == "P24" else (3, 3)
        low = torch.randint(
            -32768,
            32767,
            (*shape, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (*shape, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        payload = torch.cat((low.reshape(-1), high.reshape(-1)))
        decoded = torch.cat(
            (
                _reconstruct_native(
                    low,
                    codebook="sqg_xor_cheb_t12",
                ),
                _reconstruct_native(high, codebook="sqg_xor_cheb_t12"),
            ),
            dim=1 if shape == (hidden_tiles, 8) else 0,
        )
        return payload, decoded

    w13_payloads: list[list[torch.Tensor]] = [[], []]
    w13_decoded: list[list[torch.Tensor]] = [[], []]
    for projection in range(2):
        for expert, kind in enumerate(fc1_kinds):
            payload, decoded = make_pair(kind, (hidden_tiles, 8))
            w13_payloads[projection].append(payload)
            w13_decoded[projection].append(decoded)
    w2_payloads: list[torch.Tensor] = []
    w2_decoded: list[torch.Tensor] = []
    for expert, kind in enumerate(fc2_kinds):
        payload, decoded = make_pair(kind, (8, hidden_tiles), is_fc2=True)
        w2_payloads.append(payload)
        w2_decoded.append(decoded)
    w13_payload = torch.stack(
        [torch.stack(projection) for projection in w13_payloads]
    ).contiguous()
    w2_payload = torch.stack(w2_payloads).contiguous()

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    atom_payload = _qsrt_atom_extent_from_pair_payloads(
        w13_payload,
        w2_payload,
        fc1_pair_spec,
        fc2_pair_spec,
        intermediate_rotations,
    )
    expert_ids = torch.tensor([0, 12], dtype=torch.int32, device=device)
    if pair_case == "P24":
        format_codes = torch.tensor([0x11, 0x11], dtype=torch.uint8, device=device)
    elif pair_case == "P33":
        format_codes = torch.tensor([0x00, 0x00], dtype=torch.uint8, device=device)
    else:
        format_codes = torch.tensor([0x10, 0x01], dtype=torch.uint8, device=device)
    prepared = prepare_qsrt_atom_moe_weights(
        atom_payload,
        first_atom_slot=8,
        layer_index=1,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
        params_dtype=torch.float16,
        tile_config=(64, 256, 64, 256),
        codebook="sqg_xor_cheb_t12",
    )
    pair_prepared = prepare_qsrt_pair_moe_weights(
        w13_payload,
        w2_payload,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        fc1_pair_kind=fc1_pair_spec,
        fc2_pair_kind=fc2_pair_spec,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        params_dtype=torch.float16,
        tile_config=(64, 256, 64, 256),
        codebook="sqg_xor_cheb_t12",
    )
    torch.testing.assert_close(prepared.w13, pair_prepared.w13, rtol=0, atol=0)
    torch.testing.assert_close(prepared.w2, pair_prepared.w2, rtol=0, atol=0)
    torch.testing.assert_close(
        prepared.intermediate_rotations, intermediate_rotations, rtol=0, atol=0
    )
    assert prepared.trellis_codebook == "sqg_xor_cheb_t12"
    assert prepared.fc1_trellis_pair_kind == "PDYNAMIC"
    assert prepared.fc2_trellis_pair_kind == "PDYNAMIC"
    assert torch.equal(prepared.fc1_trellis_pair_modes, fc1_pair_spec)
    assert torch.equal(prepared.fc2_trellis_pair_modes, fc2_pair_spec)
    assert prepared.w13.numel() * prepared.w13.element_size() == w13_payload.numel() * 2
    assert prepared.w2.numel() * prepared.w2.element_size() == w2_payload.numel() * 2

    gate_weights = torch.stack(w13_decoded[0]).to(device)
    up_weights = torch.stack(w13_decoded[1]).to(device)
    down_weights = torch.stack(w2_decoded).to(device)

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    router_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            router_weights,
            ids,
            activation="situ",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=intermediate_rotations,
            full_rotation=True,
            suh_gate_table=gate_suh,
            suh_up_table=up_suh,
            svh_table=down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    public_weight_plan = _weight_plan(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        input_dtype=x.dtype,
        activation="situ",
        storage_format="qsrt_atoms_v1",
        source_format="qsrt_sqg_e4m3",
    )
    public_weights = fused_moe.prepare_weights(
        plan=public_weight_plan,
        params_dtype=x.dtype,
        qsrt_atom_payload=atom_payload,
        qsrt_first_atom_slot=8,
        qsrt_layer_index=1,
        qsrt_expert_ids=expert_ids,
        qsrt_format_codes=format_codes,
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
    )
    public_prepared = public_weights.representation_for("w4a16")
    assert torch.equal(public_prepared.w13, prepared.w13)
    public_plan = _plan(
        public_weights,
        max_tokens=m,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    public_spec = public_plan.scratch_specs()[0]
    public_scratch = torch.empty(
        public_spec.shape, dtype=public_spec.dtype, device=public_spec.device
    )
    public_binding = fused_moe.bind(
        public_plan,
        scratch=public_scratch,
        a=x,
        experts=public_weights,
        topk_weights=router_weights,
        topk_ids=ids,
    )
    public_actual = fused_moe.run(binding=public_binding).clone()
    torch.cuda.synchronize(device)
    assert torch.equal(public_actual, actual)

    public_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(public_graph):
        public_captured = fused_moe.run(binding=public_binding)
    public_graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(public_captured, public_actual)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("first_atom_slot", "fc1_bits", "fc2_bits"),
    [
        (0, (3, 3), (3, 3)),
        (40, (4, 3), (3, 3)),
        (88, (4, 3), (4, 4)),
    ],
)
def test_coupled_h308_fused_moe_matches_transform_reference_and_captures(
    first_atom_slot: int,
    fc1_bits: tuple[int, int],
    fc2_bits: tuple[int, int],
) -> None:
    """Close every static coupled-H308 pair shape through the fused kernel."""

    torch.manual_seed(20260810 + first_atom_slot)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 1
    hidden, intermediate = 3584, 256
    hidden_tiles, atoms = hidden // 16, intermediate // 32

    w13_low = torch.randint(
        -32768,
        32767,
        (2, experts, hidden_tiles, atoms, 16 * fc1_bits[0]),
        dtype=torch.int16,
        device=device,
    )
    w13_high = torch.randint(
        -32768,
        32767,
        (2, experts, hidden_tiles, atoms, 16 * fc1_bits[1]),
        dtype=torch.int16,
        device=device,
    )
    w2_low = torch.randint(
        -32768,
        32767,
        (experts, atoms, hidden_tiles, 16 * fc2_bits[0]),
        dtype=torch.int16,
        device=device,
    )
    w2_high = torch.randint(
        -32768,
        32767,
        (experts, atoms, hidden_tiles, 16 * fc2_bits[1]),
        dtype=torch.int16,
        device=device,
    )
    ordinary_rotations = torch.ones(
        (experts, 3 * intermediate), dtype=torch.float16, device=device
    )
    atom_payload = _qsrt_coupled_h308_atom_extent_from_payloads(
        w13_low,
        w13_high,
        w2_low,
        w2_high,
        ordinary_rotations,
    ).cpu()

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    draw = 5
    prepared = _prepare_qsrt_coupled_h308_atom_v2_moe_weights(
        atom_payload,
        first_atom_slot=first_atom_slot,
        rotation_draws=torch.tensor([draw], dtype=torch.uint8),
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
        params_dtype=torch.float16,
        codebook="sqg_xor_cheb_t12",
        tile_config=(64, 256, 64, 256),
        dummy_scale=None,
        workspace=None,
    )
    expected_w13 = torch.stack(
        tuple(
            torch.cat(
                (
                    w13_low[projection].reshape(experts, hidden_tiles, -1),
                    w13_high[projection].reshape(experts, hidden_tiles, -1),
                ),
                dim=2,
            ).reshape(-1)
            for projection in range(2)
        )
    ).reshape(-1)
    expected_w2 = torch.cat((w2_low.reshape(-1), w2_high.reshape(-1)))
    assert torch.equal(prepared.w13.view(torch.int16), expected_w13)
    assert torch.equal(prepared.w2.view(torch.int16), expected_w2)
    assert prepared.fc1_trellis_pair_kind == (
        "P43" if fc1_bits == (4, 3) else "P33"
    )
    assert prepared.fc2_trellis_pair_kind == (
        "P44" if fc2_bits == (4, 4) else "P33"
    )

    slot_weights: list[torch.Tensor] = []
    for projection in range(2):
        low = _reconstruct_native(
            w13_low[projection, 0], codebook="sqg_xor_cheb_t12"
        )
        high = _reconstruct_native(
            w13_high[projection, 0], codebook="sqg_xor_cheb_t12"
        )
        # P43 serializes the funded K4 record first; the coupled activation
        # restores the logical K3/K4 block order before SiTU.
        slot_weights.append(
            torch.cat((high, low) if fc1_bits == (4, 3) else (low, high), dim=1)
            .unsqueeze(0)
            .to(device)
        )
    down_weights = torch.cat(
        (
            _reconstruct_native(w2_low[0], codebook="sqg_xor_cheb_t12"),
            _reconstruct_native(w2_high[0], codebook="sqg_xor_cheb_t12"),
        ),
        dim=0,
    ).unsqueeze(0).to(device)
    coupled_signs = prepared.intermediate_rotations[:, 3 * intermediate :]
    source_suh = gate_suh if first_atom_slot < 48 else up_suh

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    router_weights = torch.ones((m, topk), dtype=torch.float32, device=device)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            router_weights,
            ids,
            activation="situ",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=prepared.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=prepared.gate_suh,
            suh_up_table=prepared.up_suh,
            svh_table=prepared.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    actual = run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_coupled_decoded(
        x,
        ids,
        router_weights,
        slot_weights[0],
        slot_weights[1],
        down_weights,
        source_suh,
        ordinary_rotations,
        coupled_signs,
        down_svh,
    )
    relative_error = (
        (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.9999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("mode_values", [(0, 0), (1, 1), (0, 1)])
def test_qsrt_atoms_v2_prepared_p33_p43_matches_full_rotation_reference(
    mode_values: tuple[int, int],
) -> None:
    """Close exact compact offsets and seven-bit P43 staging end to end."""

    torch.manual_seed(20260809)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = m = 2
    hidden = intermediate = 256
    hidden_tiles = hidden // 16
    modes = torch.tensor(mode_values, dtype=torch.int64, device=device)

    def make_pair(
        p43: bool, shape: tuple[int, int], *, fc1: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_bits, high_bits = ((4, 3) if p43 else (3, 3))
        low = torch.randint(
            -32768,
            32767,
            (*shape, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (*shape, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        decoded = torch.cat(
            (
                _reconstruct_native(low, codebook="sqg_xor_cheb_t12"),
                _reconstruct_native(high, codebook="sqg_xor_cheb_t12"),
            ),
            dim=1 if fc1 else 0,
        )
        if fc1:
            payload = torch.cat(
                (
                    low.reshape(hidden_tiles, -1),
                    high.reshape(hidden_tiles, -1),
                ),
                dim=-1,
            ).reshape(-1)
        else:
            payload = torch.cat((low.reshape(-1), high.reshape(-1)))
        return payload, decoded

    w13_experts: list[list[torch.Tensor]] = [[], []]
    w13_decoded: list[list[torch.Tensor]] = [[], []]
    for projection in range(2):
        for expert in range(experts):
            payload, decoded = make_pair(
                bool(mode_values[expert]), (hidden_tiles, 8), fc1=True
            )
            w13_experts[projection].append(payload)
            w13_decoded[projection].append(decoded)
    w2_experts: list[torch.Tensor] = []
    w2_decoded: list[torch.Tensor] = []
    for expert in range(experts):
        payload, decoded = make_pair(
            bool(mode_values[expert]), (8, hidden_tiles), fc1=False
        )
        w2_experts.append(payload)
        w2_decoded.append(decoded)

    w13_payload = torch.cat(
        tuple(torch.cat(tuple(plane)) for plane in w13_experts)
    ).contiguous()
    w2_payload = torch.cat(tuple(w2_experts)).contiguous()
    offsets_u32 = torch.tensor(
        [0, w2_experts[0].numel() // 2], dtype=torch.int64, device=device
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    prepared = _prepare_qsrt_p33_p43_moe_weights(
        w13_payload,
        w2_payload,
        pair_offsets_u32=offsets_u32,
        pair_is_p43=modes,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
    )
    expected_descriptors = (offsets_u32 << 1) | modes
    assert prepared.fc1_trellis_pair_kind == "P33_P43"
    assert prepared.fc2_trellis_pair_kind == "P33_P43"
    assert torch.equal(prepared.fc1_trellis_pair_modes, expected_descriptors)
    assert torch.equal(prepared.fc2_trellis_pair_modes, expected_descriptors)
    assert prepared.w13.numel() * 4 == w13_payload.numel() * 2
    assert prepared.w2.numel() * 4 == w2_payload.numel() * 2

    gate_weights = torch.stack(w13_decoded[0]).to(device)
    up_weights = torch.stack(w13_decoded[1]).to(device)
    down_weights = torch.stack(w2_decoded).to(device)
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    router_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    actual = run_w4a16_moe(
        x,
        prepared,
        router_weights,
        ids,
        activation="situ",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        route_block_size_m=8,
        intermediate_rotation_scales=intermediate_rotations,
        full_rotation=True,
        suh_gate_table=gate_suh,
        suh_up_table=up_suh,
        svh_table=down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    ).clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_weights,
        up_weights,
        down_weights,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    if mode_values == (0, 0):
        legacy_modes = torch.zeros(experts, dtype=torch.int32, device=device)
        legacy_prepared = replace(
            prepared,
            fc1_trellis_pair_kind="PDYNAMIC",
            fc2_trellis_pair_kind="PDYNAMIC",
            fc1_trellis_pair_modes=legacy_modes,
            fc2_trellis_pair_modes=legacy_modes,
        )
        legacy_buffers = make_w4a16_packed_buffers(
            legacy_prepared,
            m=m,
            topk=topk,
            dtype=torch.float16,
            device=device,
            full_rotation=True,
            block_size_m=8,
        )
        legacy_actual = run_w4a16_moe(
            x,
            legacy_prepared,
            router_weights,
            ids,
            activation="situ",
            intermediate_cache13=legacy_buffers.intermediate_cache13,
            intermediate_cache2=legacy_buffers.intermediate_cache2,
            output=legacy_buffers.output,
            fc1_c_tmp=legacy_buffers.fc1_c_tmp,
            fc2_c_tmp=legacy_buffers.fc2_c_tmp,
            packed_route_indices=legacy_buffers.packed_route_indices,
            block_expert_ids=legacy_buffers.block_expert_ids,
            packed_route_count=legacy_buffers.packed_route_count,
            expert_offsets=legacy_buffers.expert_offsets,
            expert_counts=legacy_buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=intermediate_rotations,
            full_rotation=True,
            suh_gate_table=gate_suh,
            suh_up_table=up_suh,
            svh_table=down_svh,
            rotation_a_gate=legacy_buffers.rotation_a_gate,
            rotation_a_up=legacy_buffers.rotation_a_up,
        ).clone()
        torch.cuda.synchronize(device)
        legacy_relative_error = (
            (legacy_actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
        )
        assert float(legacy_relative_error) <= 2.0e-2
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA storage")
def test_qsrt_atom_prepare_rejects_malformed_payloads_and_metadata() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    experts = 2
    hidden = intermediate = 256
    atoms = torch.zeros((8, experts, 129_216), dtype=torch.uint8, device=device)
    rotations = torch.ones((1, hidden), dtype=torch.float16, device=device)
    expert_ids = torch.tensor([0, 12], dtype=torch.int32, device=device)
    format_codes = torch.tensor([0x10, 0x01], dtype=torch.uint8, device=device)
    kwargs = dict(
        first_atom_slot=8,
        layer_index=1,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="situ",
        gate_suh=rotations,
        up_suh=rotations,
        down_svh=rotations,
        params_dtype=torch.float16,
        codebook="sqg_xor_cheb_t12",
        tile_config=(64, 256, 64, 256),
    )

    public_plan = _weight_plan(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        input_dtype=torch.float16,
        activation="situ",
        source_format="qsrt_sqg_e4m3",
        storage_format="qsrt_atoms_v1",
    )
    public_weights = fused_moe.prepare_weights(
        plan=public_plan,
        params_dtype=torch.float16,
        qsrt_atom_payload=atoms,
        qsrt_first_atom_slot=8,
        qsrt_layer_index=1,
        qsrt_expert_ids=expert_ids,
        qsrt_format_codes=format_codes,
        gate_suh=rotations,
        up_suh=rotations,
        down_svh=rotations,
    )
    public_prepared = public_weights.representation_for("w4a16")
    assert public_prepared.fc1_trellis_pair_kind == "PDYNAMIC"
    assert public_prepared.fc2_trellis_pair_kind == "PDYNAMIC"

    with pytest.raises(ValueError, match="atom_payload must have shape"):
        prepare_qsrt_atom_moe_weights(atoms[..., :-1].contiguous(), **kwargs)
    with pytest.raises(ValueError, match="pair-aligned"):
        prepare_qsrt_atom_moe_weights(atoms, **(kwargs | {"first_atom_slot": 1}))
    with pytest.raises(ValueError, match="format codes"):
        prepare_qsrt_atom_moe_weights(
            atoms,
            **(
                kwargs
                | {
                    "format_codes": torch.tensor(
                        [0x30, 0x01], dtype=torch.uint8, device=device
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="expert_ids must have shape"):
        prepare_qsrt_atom_moe_weights(
            atoms,
            **(
                kwargs
                | {"expert_ids": torch.tensor([0], dtype=torch.int32, device=device)}
            ),
        )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [3, 4])
def test_full_rotation_topk16_route_parallel_sum_matches_reference(bits: int) -> None:
    """Exercise legacy EXL3 K3/K4 through the eight-warp route partition."""
    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = topk = 16
    hidden = intermediate = 128
    tile_config = (64, 128, 64, 128)
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        activation="situ",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=1,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn((1, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.randperm(experts, device=device).to(torch.int32).view(1, topk)
    router_weights = torch.rand((1, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=ids,
    )
    actual = binding.run().clone()
    torch.cuda.synchronize(device)
    reference = _reference_full_rotation(
        x,
        ids,
        router_weights,
        w13,
        w2,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = binding.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_planned_full_rotation_matches_reference_and_captures(
    input_dtype: torch.dtype,
    activation: str,
) -> None:
    torch.manual_seed(20260721)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 2, 128, 128
    bits = 3
    tile_config = (64, 128, 64, 128)
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((1, hidden)).contiguous()
    up_suh = scales((1, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((1, hidden)).contiguous()
    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
        tile_config=tile_config,
        input_dtype=input_dtype,
        activation=activation,
    )
    prepared = weights.representation_for("w4a16")
    assert prepared.w13.data_ptr() == w13.data_ptr()
    assert prepared.w2.data_ptr() == w2.data_ptr()

    plan = _plan(
        weights,
        max_tokens=2,
        num_topk=2,
        route_num_experts=4,
        block_size_m=8,
        device=device,
    )
    assert plan.caps.w4a16_block_size_m == 8
    assert plan.full_rotation

    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn((2, hidden), device=device) * 1.0e-3).to(input_dtype)
    local_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    global_ids = torch.tensor([[1, 3], [3, 1]], dtype=torch.int64, device=device)
    router_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    route_map = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    output_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device=device)
    external_output = torch.empty((2, hidden), dtype=torch.float32, device=device)

    mapped = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped_output = fused_moe.run(binding=mapped)
    torch.cuda.synchronize(device)
    assert mapped_output.data_ptr() == external_output.data_ptr()
    mapped_eager = mapped_output.clone()

    identity = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=local_ids,
    )
    identity_output = identity.run()
    torch.cuda.synchronize(device)
    assert identity_output.dtype == torch.float32
    assert torch.allclose(identity_output, mapped_eager, rtol=2.0e-3, atol=2.0e-3)

    reference = _reference_full_rotation(
        x,
        local_ids,
        router_weights,
        w13,
        w2,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation=activation,
    )
    relative_error = (mapped_eager - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        mapped_eager.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    mapped = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped.run()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    mapped.run()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = mapped.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_output.data_ptr() == external_output.data_ptr()
    assert torch.allclose(captured_output, mapped_eager, rtol=2.0e-3, atol=2.0e-3)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_full_rotation_reuses_compiled_kernels_across_expert_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact EXL3 layer sizes are runtime data, including the W13 plane stride."""
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = intermediate = 128
    topk = max_tokens = 2
    route_experts = 4
    bits = 3
    tile_config = (64, 128, 64, 128)
    w4a16_kernel.clear_w4a16_kernel_cache()

    compiled_after_first: tuple[dict[tuple, int], dict[tuple, int]] | None = None
    for experts in (2, 3):
        w13 = torch.randint(
            -32768,
            32767,
            (2, experts, hidden // 16, intermediate // 16, 16 * bits),
            dtype=torch.int16,
            device=device,
        )
        w2 = torch.randint(
            -32768,
            32767,
            (experts, intermediate // 16, hidden // 16, 16 * bits),
            dtype=torch.int16,
            device=device,
        )

        def scales(shape: tuple[int, ...]) -> torch.Tensor:
            return (0.875 + 0.25 * torch.rand(shape, device=device)).to(
                torch.float16
            )

        gate_suh = scales((experts, hidden)).contiguous()
        up_suh = scales((experts, hidden)).contiguous()
        intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
        down_svh = scales((experts, hidden)).contiguous()
        weights = _prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
            tile_config=tile_config,
        )
        plan = _plan(
            weights,
            max_tokens=max_tokens,
            num_topk=topk,
            route_num_experts=route_experts,
            block_size_m=8,
            device=device,
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        x = (torch.randn((max_tokens, hidden), device=device) * 1.0e-3).to(
            torch.bfloat16
        )
        global_ids = torch.tensor(
            [[0, 3], [2, 1]], dtype=torch.int32, device=device
        )
        expert_map = (
            torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
            if experts == 2
            else torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=device)
        )
        local_ids = expert_map[global_ids]
        router_weights = torch.tensor(
            [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=router_weights,
            topk_ids=global_ids,
            route_expert_map=expert_map,
            output_expert_map=expert_map,
        )
        if binding.fused_launch is not None:
            assert binding.fused_launch.direct_topk_routes
            assert binding.fused_launch.use_expert_map

        def _route_pack_must_not_run(*args, **kwargs):
            raise AssertionError("mapped decode unexpectedly invoked route packing")

        with monkeypatch.context() as patch:
            patch.setattr(
                w4a16_kernel,
                "pack_topk_routes_by_expert",
                _route_pack_must_not_run,
            )
            actual = fused_moe.run(binding=binding).clone()
        torch.cuda.synchronize(device)
        reference = _reference_full_rotation(
            x,
            local_ids,
            router_weights,
            w13,
            w2,
            gate_suh,
            up_suh,
            intermediate_rotations,
            down_svh,
        )
        relative_error = (actual - reference).norm() / reference.norm().clamp_min(
            1.0e-9
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
        assert float(relative_error) <= 2.0e-2
        assert float(cosine) >= 0.999

        if experts == 2:
            # Unmapped, negative, and out-of-range global ids must be rejected
            # inside the fused kernel before weight access. The top-k sum must
            # skip those stale route rows as well.
            sparse_map = torch.tensor(
                [0, -1, 1, -1], dtype=torch.int32, device=device
            )
            invalid_global_ids = torch.tensor(
                [[0, 1], [route_experts + 3, -1]],
                dtype=torch.int32,
                device=device,
            )
            masked_router_weights = router_weights.clone()
            masked_router_weights[0, 1] = 0.0
            masked_router_weights[1].zero_()
            invalid_binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x,
                experts=weights,
                topk_weights=router_weights,
                topk_ids=invalid_global_ids,
                route_expert_map=sparse_map,
                output_expert_map=sparse_map,
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    w4a16_kernel,
                    "pack_topk_routes_by_expert",
                    _route_pack_must_not_run,
                )
                invalid_actual = invalid_binding.run().clone()
            invalid_reference = _reference_full_rotation(
                x,
                torch.zeros_like(invalid_global_ids),
                masked_router_weights,
                w13,
                w2,
                gate_suh,
                up_suh,
                intermediate_rotations,
                down_svh,
            )
            invalid_relative_error = (
                (invalid_actual - invalid_reference).norm()
                / invalid_reference.norm().clamp_min(1.0e-9)
            )
            assert float(invalid_relative_error) <= 2.0e-2
            assert torch.count_nonzero(invalid_actual[1]) == 0

            graph = torch.cuda.CUDAGraph()
            with monkeypatch.context() as patch:
                patch.setattr(
                    w4a16_kernel,
                    "pack_topk_routes_by_expert",
                    _route_pack_must_not_run,
                )
                with torch.cuda.graph(graph):
                    captured_invalid = invalid_binding.run()
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(captured_invalid, invalid_actual)

        compiled_now = (
            {
                key: id(value.compiled)
                for key, value in w4a16_kernel._FUSED_CACHE.items()
            },
            {
                key: id(value.compiled)
                for key, value in w4a16_kernel._SUM_CACHE.items()
            },
        )
        if compiled_after_first is None:
            compiled_after_first = compiled_now
            assert compiled_after_first[0]
            assert compiled_after_first[1]
        else:
            assert compiled_now == compiled_after_first


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_qsrt_pdynamic_cache_key_ignores_expert_count() -> None:
    """Profile-ID-5 layer populations are runtime data, like mixed tiers."""
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    def make_kernel(
        num_experts: int,
        pair_kind: str = "PDYNAMIC",
        fc2_pair_kind: str | None = None,
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=2,
            hidden_size=256,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="situ",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=256,
            fc1_tile_k=64,
            fc2_tile_n=256,
            fc2_tile_k=64,
            moe_block_size=8,
            max_m_blocks=4,
            element_dtype="fp16",
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis3_t256_proj",
            trellis_bits=3,
            trellis_codebook="sqg_xor_cheb_t12",
            fc1_trellis_pair_kind=pair_kind,
            fc2_trellis_pair_kind=(
                pair_kind if fc2_pair_kind is None else fc2_pair_kind
            ),
            schedule_whole_tiles=True,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            broadcast_suh=True,
        )

    compact = make_kernel(27)
    complete = make_kernel(896)

    assert compact.dynamic_num_experts
    assert complete.dynamic_num_experts
    assert compact.__cache_key__ == complete.__cache_key__
    assert compact.fc1.__cache_key__ == complete.fc1.__cache_key__
    assert compact.fc2.__cache_key__ == complete.fc2.__cache_key__

    make_kernel(27, "P24")
    make_kernel(27, "P43", "P33")
    with pytest.raises(
        ValueError, match="dynamic fused trellis pair kinds must match"
    ):
        make_kernel(27, "PDYNAMIC", "P33_P43")
    with pytest.raises(
        ValueError, match="dynamic fused trellis pair kinds must match"
    ):
        make_kernel(27, "P24", "PDYNAMIC")


def test_x4t_fc2_direct_uses_bounded_expert_capacity() -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _w4a16_fc2_direct_expert_capacity,
    )

    # Every possible Kimi layer endpoint population reuses one specialization.
    assert {
        _w4a16_fc2_direct_expert_capacity(num_experts)
        for num_experts in (1, 27, 192, 700, 896, 1024)
    } == {1024}
    assert _w4a16_fc2_direct_expert_capacity(1025) == 2048
    with pytest.raises(ValueError, match="at least one expert"):
        _w4a16_fc2_direct_expert_capacity(0)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
def test_planned_full_rotation_small_m_partial_blocks(m: int) -> None:
    """Small-m coverage at production geometry, with m < plan capacity.

    Motivated by vLLM issue #183, which reported that widening the Trellis window
    (VLLM_EXL3_TRELLIS_MIN_M=1) so an MTP-N draft's m=1..N GEMMs stay on the fused
    path silently corrupts output. m < moe_block_size means the route packer emits
    mostly-padding blocks (at m=3, topk=8 roughly 1-3 real rows of 8), so padding
    masking is load-bearing and was previously untested: the only numerical test on
    this path ran m == max_tokens == 2 with topk=2 and a non-production tile.

    Three oracles:
      * per-token bitwise equality against an m=1 run of the same token. A token's
        output depends only on its own routes, so batching must not change a bit.
      * the scratch arena pre-filled with 0xFF (NaN in fp32/fp16) before every
        bind, so any read of a padding/uninitialised slot surfaces as NaN.
      * plan capacity 32 vs m, i.e. the window-widening regime itself.
    """
    torch.manual_seed(20260726)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 32, 1024, 1024
    bits, topk, capacity = 3, 8, 32
    tile_config = (64, 256, 64, 256)

    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=scales((experts, hidden)).contiguous(),
        up_suh=scales((experts, hidden)).contiguous(),
        intermediate_rotations=scales((experts, 3 * intermediate)).contiguous(),
        down_svh=scales((experts, hidden)).contiguous(),
        codebook="mcg",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=capacity,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    assert plan.caps.w4a16_block_size_m == 8
    assert plan.full_rotation

    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    router_weights = torch.rand((m, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    def _run(rows: slice) -> torch.Tensor:
        # NaN-poison the arena so a padding-slot read cannot pass silently.
        scratch.fill_(0xFF)
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x[rows].contiguous(),
            experts=weights,
            topk_weights=router_weights[rows].contiguous(),
            topk_ids=ids[rows].contiguous(),
        )
        assert binding.route_block_size_m == plan.caps.w4a16_block_size_m
        out = fused_moe.run(binding=binding)
        torch.cuda.synchronize(device)
        return out.clone()

    batched = _run(slice(0, m))
    assert not torch.isnan(batched).any(), (
        f"NaN in fused output at m={m}: a padding or uninitialised arena slot was read"
    )

    for row in range(m):
        single = _run(slice(row, row + 1))
        assert not torch.isnan(single).any()
        assert torch.equal(batched[row], single[0]), (
            f"row {row} of an m={m} batch differs from its own m=1 run; a token's "
            "output must not depend on how many other tokens share the batch"
        )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [1, 2, 3])
def test_planned_full_rotation_capture_below_capacity(m: int) -> None:
    """CUDA-graph capture and replay at m strictly below plan capacity.

    This is the surface vLLM #183 actually leaves untested. The existing capture
    test captures at ``m == max_tokens``, and a small-m eager check does not cover
    capture: ``bind`` runs inside the captured region, so the route-pack grid, the
    ``kernel_workspace`` zero-fill and the active-m constants are all baked into
    the graph. If any of those were captured for the wrong m, replay would produce
    stale or mismatched results while eager execution stayed correct -- which is
    exactly a "short prompts pass, long context corrupts" signature once an
    MTP-N draft replays its captured graph.

    Asserts replay matches the eager result for the same inputs bit-for-bit, and
    that a second replay is stable (no dependence on scratch left over from the
    capture pass).
    """
    torch.manual_seed(20260726)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 32, 1024, 1024
    bits, topk, capacity = 3, 8, 32
    tile_config = (64, 256, 64, 256)

    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    weights = _prepare_weights(
        w13,
        w2,
        gate_suh=scales((experts, hidden)).contiguous(),
        up_suh=scales((experts, hidden)).contiguous(),
        intermediate_rotations=scales((experts, 3 * intermediate)).contiguous(),
        down_svh=scales((experts, hidden)).contiguous(),
        codebook="mcg",
        tile_config=tile_config,
    )
    plan = _plan(
        weights,
        max_tokens=capacity,
        num_topk=topk,
        route_num_experts=experts,
        block_size_m=8,
        device=device,
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    router_weights = torch.rand((m, topk), dtype=torch.float32, device=device)
    router_weights /= router_weights.sum(dim=1, keepdim=True)

    eager_binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=ids,
    )
    assert eager_binding.route_block_size_m == plan.caps.w4a16_block_size_m
    eager = eager_binding.run().clone()
    torch.cuda.synchronize(device)
    assert not torch.isnan(eager).any()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=router_weights,
            topk_ids=ids,
        )
        captured = binding.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert not torch.isnan(captured).any(), (
        f"NaN after graph replay at m={m} (capacity {capacity})"
    )
    assert torch.equal(captured, eager), (
        f"graph replay at m={m} below capacity {capacity} disagrees with eager; "
        "capture baked constants or a route-pack grid for the wrong m"
    )

    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager), (
        f"second replay at m={m} drifted; replay depends on scratch state left "
        "over from the capture pass"
    )
