from __future__ import annotations

import pytest
import torch

from sparkinfer._lib.intrinsics import (
    FLOAT4_E2M1_MAX,
    fp4_quantize_values_torch,
    pack_grouped_fp4_values,
    swizzle_block_scale,
)
from sparkinfer.moe.fused_moe._impl import (
    plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
)
from sparkinfer.moe._shared.execution import PreparedWeightLayout
from sparkinfer.moe._shared.kernels.reference import (
    moe_reference_nvfp4,
    moe_reference_w4a16_f32,
    moe_reference_w4a16_fp4_e8m0_k32,
)
from sparkinfer.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_pack_capacity,
    select_route_block_size_m,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _DEFAULT_MAX_SHARED_MEM,
    MoEMicroKernelW4A16SmallMDirect,
    _w4a16_stream_is_capturing,
    _small_m_direct_supported,
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers as make_w4a16_buffers,
    prepare_w4a16_e8m0_native_weights,
    prepare_w4a16_modelopt_native_weights,
    prepare_w4a16_modelopt_nvfp4_weights as prepare_w4a16_weights,
    prepare_w4a16_packed_weights,
)
from tests._reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
    run_tp_moe_fp4,
)
from tests._reference.w4a16_reference import compare_to_reference, moe_reference_w4a16


def test_w4a16_fused_compile_rejects_unresolved_capture_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    monkeypatch.setattr(w4a16_kernel, "_FUSED_CACHE", {})
    monkeypatch.setattr(w4a16_kernel.torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeError,
        match="W4A16 fused MoE launch is not resolved for CUDA graph capture",
    ):
        compile_w4a16_fused_moe(
            size_m=16,
            hidden_size=128,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=16,
            sms=188,
            max_shared_mem=_DEFAULT_MAX_SHARED_MEM,
            _require_cached=True,
        )


def test_w4a16_capture_guard_checks_selected_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    selected_stream = 22
    queried: list[object] = []
    monkeypatch.setattr(
        w4a16_kernel.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        w4a16_kernel.cuda,
        "cuStreamIsCapturing",
        lambda stream: (
            queried.append(stream) or w4a16_kernel.cuda.CUresult.CUDA_SUCCESS,
            1,
        ),
    )

    assert _w4a16_stream_is_capturing(selected_stream, current_stream=11)
    assert queried == [selected_stream]


def test_w4a16_capture_guard_preserves_current_stream_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    monkeypatch.setattr(
        w4a16_kernel.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        w4a16_kernel.cuda,
        "cuStreamIsCapturing",
        lambda _stream: (_ for _ in ()).throw(
            AssertionError("current-stream capture queried the selected stream")
        ),
    )

    assert _w4a16_stream_is_capturing(22, current_stream=11)


@pytest.mark.parametrize(
    ("tokens", "bucket_tokens"),
    (
        (1, 1),
        (8, 8),
        (9, 16),
        (3072, 4096),
        (4096, 4096),
        (4097, 8192),
    ),
)
def test_generic_w4a16_planner_runtime_key_agrees_at_bucket_boundaries(
    tokens: int,
    bucket_tokens: int,
) -> None:
    topk = 8
    block_size = 64
    num_experts = 160
    routed_capacity, packed_routes, max_m_blocks = route_pack_capacity(
        tokens * topk,
        block_size,
        num_experts,
        topk=topk,
    )
    expected_routes = bucket_tokens * topk
    expected_packed = max_packed_route_slots(
        expected_routes,
        block_size,
        num_experts,
    )

    assert routed_capacity == expected_routes
    assert packed_routes == expected_packed
    assert max_m_blocks == (expected_packed + block_size - 1) // block_size
    if tokens == 3072:
        assert max_m_blocks == 670


def test_trellis_w4a16_planner_runtime_key_preserves_exact_capacity() -> None:
    topk = 8
    block_size = 64
    num_experts = 160
    routed_capacity, packed_routes, max_m_blocks = route_pack_capacity(
        3072 * topk,
        block_size,
        num_experts,
        topk=topk,
        bucket_tokens=False,
    )

    assert routed_capacity == 3072 * topk
    assert packed_routes == max_packed_route_slots(
        3072 * topk,
        block_size,
        num_experts,
    )
    assert max_m_blocks == 542


def test_trellis_w4a16_capture_prewarm_uses_exact_runtime_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from sparkinfer.moe.fused_moe import _impl as tp_moe_impl
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    workspace = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        full_rotation=True,
        route_block_size_m=64,
        num_topk=8,
        route_E=160,
        weight_E=160,
        k=6144,
        n=2048,
        activation="silu",
        trellis_bits=3,
        trellis_tile_config=None,
    )
    fused_calls: list[dict[str, object]] = []
    resolved_fused = object()
    monkeypatch.setattr(tp_moe_impl.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        tp_moe_impl.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        tp_moe_impl.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=188,
            shared_memory_per_block_optin=_DEFAULT_MAX_SHARED_MEM,
        ),
    )
    monkeypatch.setattr(w4a16_kernel, "_rot_scales_dummy", lambda _device: None)
    monkeypatch.setattr(
        w4a16_kernel,
        "compile_w4a16_fused_moe",
        lambda **kwargs: fused_calls.append(kwargs) or resolved_fused,
    )
    monkeypatch.setattr(
        w4a16_kernel,
        "compile_w4a16_topk_sum",
        lambda **_kwargs: object(),
    )

    tp_moe_impl._prewarm_w4a16_planned_launches(
        workspace,
        token_counts=(3072,),
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=1.0,
        scale_format="e4m3_k32",
        weight_layout="trellis3_t256",
        w13_layout="trellis3_t256_proj",
        collect_activation_amax=False,
    )

    assert len(fused_calls) == 2
    assert {call["size_m"] for call in fused_calls} == {3072}
    assert {call["max_m_blocks"] for call in fused_calls} == {542}
    assert (
        workspace.planned_fused_moe_launches[("trellis3_t256", "e4m3_k32", 3072, False)]
        is resolved_fused
    )


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 0.25 + 0.03125).to(torch.float8_e4m3fn)


def _constant_e8m0(shape: tuple[int, ...], byte: int) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("requires torch.float8_e8m0fnu")
    return torch.full(shape, byte, dtype=torch.uint8, device="cuda").view(e8m0_dtype)


def _pattern_e8m0(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("requires torch.float8_e8m0fnu")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    storage = (torch.arange(numel, device="cuda", dtype=torch.int64) + offset) % 4 + 119
    return storage.to(torch.uint8).reshape(shape).view(e8m0_dtype)


def _quantize_dense_moe_weight_storage(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros(
        (num_groups, rows, cols),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    scales = torch.zeros(
        (num_groups, rows, cols // 16),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        x = input_tensor[group_idx].float()
        sliced = x.view(rows, cols // 16, 16)
        block_max = sliced.abs().amax(dim=-1, keepdim=True)
        scale = (global_scale[group_idx] * (block_max / FLOAT4_E2M1_MAX)).to(
            torch.float8_e4m3fn
        )
        scale = scale.to(torch.float32)
        output_scale = 1.0 / (scale * (1.0 / global_scale[group_idx])).clamp(min=1e-30)
        clipped = torch.clamp(
            sliced * output_scale,
            -FLOAT4_E2M1_MAX,
            FLOAT4_E2M1_MAX,
        ).view(rows, cols)
        quantized[group_idx] = fp4_quantize_values_torch(clipped)
        scales[group_idx] = scale.squeeze(-1)

    packed = pack_grouped_fp4_values(quantized).permute(2, 0, 1).contiguous()
    swizzled = swizzle_block_scale(scales.to(torch.float8_e4m3fn))
    return packed, swizzled


def _make_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    is_gated = activation in {"silu", "situ"}
    w13_rows = intermediate_size * (2 if is_gated else 1)
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((experts, w13_rows, hidden_size // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((experts, hidden_size, intermediate_size // 16))
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(
        torch.float32
    )
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(
        torch.float32
    )
    return w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale


def _run_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    prepared = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
        route_num_experts=None if expert_map is None else int(expert_map.numel()),
    )
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        expert_map=expert_map,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        swiglu_limit=swiglu_limit,
    )


def _reference_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    reference_topk_ids = topk_ids
    reference_topk_weights = topk_weights
    if expert_map is not None:
        reference_topk_ids = expert_map[topk_ids.long()].to(torch.int32)
        active = reference_topk_ids >= 0
        reference_topk_weights = topk_weights.masked_fill(~active, 0.0)
        reference_topk_ids = reference_topk_ids.clamp_min(0)
    return moe_reference_w4a16(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        reference_topk_ids,
        reference_topk_weights,
        w13.shape[0],
        w2.shape[1],
        w2.shape[2] * 2,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def _assert_matches_oracle(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    activation: str,
) -> None:
    metrics = compare_to_reference(actual, expected)
    min_cos = 0.9975 if activation == "silu" else 0.9900
    assert metrics.cos >= min_cos, metrics


@pytest.mark.parametrize("scale_format", ["e4m3_k16", "e8m0_k32"])
@pytest.mark.parametrize("m", [1, 3, 8])
def test_w4a16_packed_weights_do_not_route_to_small_m_direct(
    m: int,
    scale_format: str,
) -> None:
    assert not _small_m_direct_supported(
        m=m,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        topk=2,
        activation="silu",
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="packed",
        w13_layout="packed",
        scale_format=scale_format,
    )


@pytest.mark.parametrize(
    "intermediate_size",
    [16, 144, 352, 464, 480, 496, 512],
)
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
def test_w4a16_modelopt_direct_keeps_non64_intermediate_contract(
    m: int,
    activation: str,
    intermediate_size: int,
) -> None:
    """Native ModelOpt decode supports every 16-aligned intermediate shape."""
    assert _small_m_direct_supported(
        m=m,
        hidden_size=128,
        intermediate_size=intermediate_size,
        num_experts=1,
        topk=1,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w13",
        scale_format="e4m3_k16",
    )
    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation=activation,
        fast_math=True,
        share_input_across_experts=True,
        share_expert_scales=True,
        single_token=m == 1,
    )
    kernel.configure(
        m=m,
        k=128,
        n=intermediate_size,
        num_topk=1,
        weight_E=1,
        max_active_ctas=1,
    )
    assert kernel._cfg is not None
    assert kernel._cfg.n % kernel._cfg.fc1_chunks == 0
    assert kernel._cfg.i_chunk % 16 == 0
    assert kernel._cfg.inter_blocks * 16 == kernel._cfg.i_chunk
    if m > 1:
        assert kernel._cfg.i_chunk <= 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_fp4_e8m0_k32_kernel_matches_raw_e8m0_oracle(
    activation: str,
) -> None:
    experts, hidden_size, intermediate_size = 4, 128, 128
    rows = intermediate_size * (2 if activation == "silu" else 1)
    m, topk = 8, 2
    torch.manual_seed(20260526 + (1000 if activation == "silu" else 0))
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_w4a16_packed_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        source_format="fp4_e8m0_k32",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 0], [3, 2], [0, 2], [1, 3], [2, 0], [3, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            swiglu_limit=10.0 if activation == "silu" else None,
        )

    actual = launch()
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=10.0 if activation == "silu" else None,
        w13_layout="w13",
    )
    torch.cuda.synchronize()

    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)
    if activation == "relu2":
        eager = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        buffers.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(buffers.output).all().item())
        assert torch.equal(buffers.output, eager)
        _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tc_decode", [False, True], ids=["route-packed", "tc-decode"])
def test_w4a16_packed_runtime_expert_count_reuses_compiled_kernel(
    tc_decode: bool,
) -> None:
    """Packed tier size is runtime data, while the launch geometry stays static."""
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    hidden_size = intermediate_size = 128
    activation = "situ"
    rows = 2 * intermediate_size
    m, topk = 4, 2
    props = torch.cuda.get_device_properties(device)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    w4a16_kernel.clear_w4a16_kernel_cache()

    compiled_after_first: dict[tuple, int] | None = None
    for experts in (2, 3):
        w13 = torch.randint(
            0,
            256,
            (experts, rows, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            256,
            (experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w13_scale = _pattern_e8m0(
            (experts, rows, hidden_size // 32), offset=experts
        )
        w2_scale = _pattern_e8m0(
            (experts, hidden_size, intermediate_size // 32),
            offset=experts + 1,
        )
        w13_global_scale = torch.ones(
            experts, dtype=torch.float32, device=device
        )
        w2_global_scale = torch.ones(
            experts, dtype=torch.float32, device=device
        )
        prepared = prepare_w4a16_packed_weights(
            w13,
            w13_scale,
            w13_global_scale,
            w2,
            w2_scale,
            w2_global_scale,
            activation=activation,
            params_dtype=torch.bfloat16,
            source_format="fp4_e8m0_k32",
        )
        buffers = make_w4a16_buffers(
            prepared,
            m=m,
            topk=topk,
            dtype=torch.bfloat16,
            device=device,
        )
        x = (torch.randn(m, hidden_size, device=device) * 0.125).to(
            torch.bfloat16
        )
        topk_ids = torch.tensor(
            [
                [0, experts - 1],
                [experts - 1, 0],
                [experts - 1, experts - 1],
                [0, experts - 1],
            ],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.softmax(
            torch.randn(m, topk, dtype=torch.float32, device=device), dim=-1
        )
        fused_launch = compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=experts,
            top_k=topk,
            activation=activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            # Capacity metadata legitimately changes with the tier's expert
            # count, but it is not device-code geometry either.
            max_m_blocks=m * topk + experts,
            element_dtype="bf16",
            sms=int(props.multi_processor_count),
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e8m0_k32",
            w13_layout="packed",
            direct_topk_routes=tc_decode,
            tc_decode_fused_sum=tc_decode,
            force_tile_config=(64, 128, 64, 128),
        )
        assert fused_launch.num_experts == experts
        topk_sum_launch = (
            None
            if tc_decode
            else compile_w4a16_topk_sum(
                m=m,
                topk=topk,
                hidden_size=hidden_size,
                element_dtype="bf16",
            )
        )
        actual = run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
        ).clone()
        expected = moe_reference_w4a16_fp4_e8m0_k32(
            x,
            w13,
            w13_scale,
            w13_global_scale,
            w2,
            w2_scale,
            w2_global_scale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
            w13_layout="w13",
        )
        torch.cuda.synchronize(device)
        _assert_matches_oracle(actual, expected, activation=activation)

        compiled_now = {
            key: id(value.compiled)
            for key, value in w4a16_kernel._FUSED_CACHE.items()
        }
        if compiled_after_first is None:
            compiled_after_first = compiled_now
            assert len(compiled_after_first) == 1
        else:
            assert compiled_now == compiled_after_first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("intermediate_size", [128, 224])
@pytest.mark.parametrize("w13_layout", ["w13", "w31"])
# 0.5 is small enough to actually engage the SwiGLU clamp at these scales.
@pytest.mark.parametrize("swiglu_limit", [None, 0.5])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_e8m0_native_micro_matches_raw_e8m0_oracle(
    activation: str,
    m: int,
    swiglu_limit: float | None,
    w13_layout: str,
    intermediate_size: int,
) -> None:
    """Small-M micro decode path with native MXFP4 (E8M0 K/32) scales."""
    if swiglu_limit is not None and activation != "silu":
        pytest.skip("swiglu_limit only applies to gated (silu) activation")
    if (activation, m, swiglu_limit, w13_layout) == ("silu", 1, None, "w13"):
        common = dict(
            activation="silu",
            fast_math=True,
            share_input_across_experts=True,
            share_expert_scales=True,
            single_token=True,
        )
        e4m3 = MoEMicroKernelW4A16SmallMDirect(scale_format="e4m3_k16", **common)
        e8m0 = MoEMicroKernelW4A16SmallMDirect(scale_format="e8m0_k32", **common)
        assert e4m3.__cache_key__ != e8m0.__cache_key__
    experts, hidden_size = 4, 128
    rows = intermediate_size * (2 if activation == "silu" else 1)
    topk = 2
    torch.manual_seed(20260601 + (1000 if activation == "silu" else 0) + m)
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    # Confirm the dispatch routes this native E8M0 case to the micro kernel.
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    expected_w13_scale_rows = ((rows + 63) // 64) * 64
    assert tuple(prepared.w13_scale.shape) == (
        experts,
        hidden_size // 32,
        expected_w13_scale_rows,
    )
    assert prepared.micro_w13_scale is not None
    assert int(prepared.micro_w13_scale.numel()) == (
        experts * (hidden_size // 32) * expected_w13_scale_rows
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    # The micro decode path needs m * fc2_n_chunks * 128 * topk u32 of FC1-output
    # scratch, which is larger than make_w4a16_buffers' generic intermediate_cache2.
    fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
    intermediate_cache2 = torch.zeros(
        2 * m * fc2_n_chunks * 128 * topk, dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            swiglu_limit=swiglu_limit,
        )

    actual = launch()
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=swiglu_limit,
        w13_layout=w13_layout,
    )
    torch.cuda.synchronize()

    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)
    if (activation, m, swiglu_limit, w13_layout) == ("silu", 1, None, "w13"):
        eager = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        buffers.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(buffers.output).all().item())
        assert torch.equal(buffers.output, eager)
        _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("w13_layout", ["w13", "w31"])
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_e8m0_native_large_m_uses_main_gemm(
    activation: str,
    w13_layout: str,
) -> None:
    """The native E8M0 object routes med/large M to the main W4A16 GEMM."""
    experts, hidden_size, intermediate_size = 8, 256, 256
    rows = intermediate_size * (2 if activation == "silu" else 1)
    topk, m = 2, 24  # m > _W4A16_SMALL_M_DIRECT_MAX_M -> main W4A16 kernel
    torch.manual_seed(20260607 + (1000 if activation == "silu" else 0))
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    # m=24 must NOT take the small-M micro path.
    assert not _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=None,
        w13_layout=w13_layout,
    )
    torch.cuda.synchronize()
    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("activation", "intermediate_size"),
    [
        ("silu", 312),
        ("relu2", 112),
        # 32-aligned but not 128-aligned: no dividing tile config, so the
        # same logical-tail path must engage (2048/TP6 = 352, 3072/TP16 = 192).
        ("silu", 352),
        ("relu2", 192),
    ],
)
def test_w4a16_e8m0_native_compact_tail_uses_ceil_scale_grid(
    activation: str,
    intermediate_size: int,
) -> None:
    """Native E8M0 supports compact FC2 K tails without padding I to 32."""
    experts, hidden_size = 4, 128
    rows = intermediate_size * (2 if activation == "silu" else 1)
    topk, m = 2, 24
    assert intermediate_size % 8 == 0
    assert intermediate_size % 128 != 0
    torch.manual_seed(20260610)
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale_cols = (intermediate_size + 31) // 32
    w2_scale = _pattern_e8m0((experts, hidden_size, w2_scale_cols), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    assert not _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w31",
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )
    assert prepared.weight_layout == "modelopt"
    assert prepared.scale_format == "e8m0_k32"
    padded_w13_scale_rows = ((rows + 63) // 64) * 64
    assert tuple(prepared.w13_scale.shape) == (
        experts,
        hidden_size // 32,
        padded_w13_scale_rows,
    )
    assert tuple(prepared.w2_scale.shape) == (experts, w2_scale_cols, hidden_size)

    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=None,
        w13_layout="w31",
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_beats_nvfp4_against_true_fp32_oracle_for_odd_shapes(
    activation: str,
) -> None:
    experts, hidden_size, intermediate_size = 8, 128, 128
    cases = [
        (1, 3, 2, torch.int32, 0.50),
        (5, 7, 3, torch.int64, 0.75),
    ]
    for batch_size, seq_len, topk, ids_dtype, input_scale in cases:
        m = batch_size * seq_len
        torch.manual_seed(
            20260525 + m * 31 + topk + (1000 if activation == "silu" else 0)
        )
        rows = intermediate_size * (2 if activation == "silu" else 1)
        w13_dense = torch.randn(experts, rows, hidden_size, device="cuda") * (
            0.18 if activation == "silu" else 0.08
        )
        w2_dense = torch.randn(
            experts, hidden_size, intermediate_size, device="cuda"
        ) * (0.18 if activation == "silu" else 0.08)
        w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
        w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
        w13, w13_blockscale = _quantize_dense_moe_weight_storage(
            w13_dense,
            w13_global_scale,
        )
        w2, w2_blockscale = _quantize_dense_moe_weight_storage(
            w2_dense,
            w2_global_scale,
        )

        x = (torch.randn(m, hidden_size, device="cuda") * input_scale).to(
            torch.bfloat16
        )
        topk_ids = torch.randint(
            0,
            experts,
            (m, topk),
            device="cuda",
            dtype=ids_dtype,
        )
        topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
        a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
        nvfp4_experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=a_gscale,
            w1_fp4=w13,
            w1_blockscale=w13_blockscale,
            w1_alphas=w13_global_scale,
            a2_gscale=a_gscale,
            w2_fp4=w2,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_global_scale,
            activation=activation,
            quant_mode="nvfp4",
        )

        nvfp4 = run_tp_moe_fp4(
            a=x,
            experts=nvfp4_experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_mode="nvfp4",
        )
        w4a16_experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=a_gscale,
            w1_fp4=w13,
            w1_blockscale=w13_blockscale,
            w1_alphas=w13_global_scale,
            a2_gscale=a_gscale,
            w2_fp4=w2,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_global_scale,
            activation=activation,
            quant_mode="w4a16",
        )
        w4a16 = run_tp_moe_fp4(
            a=x,
            experts=w4a16_experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_mode="w4a16",
        )
        nvfp4_reference = moe_reference_nvfp4(
            x,
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            a_gscale,
            a_gscale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )
        true_fp32 = moe_reference_w4a16_f32(
            x,
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )
        torch.cuda.synchronize()

        nvfp4_reference_metrics = compare_to_reference(nvfp4, nvfp4_reference)
        nvfp4_true_metrics = compare_to_reference(nvfp4, true_fp32)
        w4a16_true_metrics = compare_to_reference(w4a16, true_fp32)
        assert nvfp4_reference_metrics.cos > 0.98, nvfp4_reference_metrics
        assert w4a16_true_metrics.cos > 0.999, w4a16_true_metrics
        assert w4a16_true_metrics.cos > nvfp4_true_metrics.cos + 0.003, (
            nvfp4_true_metrics,
            w4a16_true_metrics,
            batch_size,
            seq_len,
            topk,
            ids_dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize(
    "intermediate_size",
    [32, 128, 144, 224, 352, 464, 480, 496, 512],
)
def test_w4a16_modelopt_direct_replay_ignores_stale_swizzle_tail(
    activation: str,
    m: int,
    intermediate_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    experts, hidden_size = 8, 128
    topk = 2
    monkeypatch.setenv("SPARKINFER_W4A16_SMALL_M_DIRECT", "1")
    torch.manual_seed(
        20260730 + (1000 if activation == "silu" else 0) + m + intermediate_size
    )
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat
    direct_launches: list[tuple[int, int]] = []

    def spy_direct_launch(*args, **kwargs) -> None:
        direct_launches.append((int(kwargs["m"]), int(kwargs["intermediate_size"])))
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    rows = intermediate_size * (2 if activation == "silu" else 1)
    w13_dense = torch.randn(experts, rows, hidden_size, device="cuda") * 0.12
    w2_dense = (
        torch.randn(experts, hidden_size, intermediate_size, device="cuda") * 0.12
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w13, w13_blockscale = _quantize_dense_moe_weight_storage(
        w13_dense,
        w13_global_scale,
    )
    w2, w2_blockscale = _quantize_dense_moe_weight_storage(
        w2_dense,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        experts,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w4a16_experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    prepared_w4a16 = w4a16_experts.representation_for("w4a16")
    assert prepared_w4a16.weight_layout == "modelopt"
    micro_w13_scale = prepared_w4a16.micro_w13_scale
    assert micro_w13_scale is not None
    expected_micro_w13_rows = ((rows + 63) // 64) * 64
    assert int(micro_w13_scale.numel()) == (
        experts * (hidden_size // 16) * expected_micro_w13_rows
    )
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout=prepared_w4a16.weight_layout,
        w13_layout="w13",
        scale_format="e4m3_k16",
    )
    binding = make_tp_moe_fp4_binding(
        a=x,
        experts=w4a16_experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty(
            (m, hidden_size),
            dtype=x.dtype,
            device=x.device,
        ),
        quant_mode="w4a16",
    )
    expected = moe_reference_w4a16_f32(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
    )

    first = binding.run().clone()
    torch.cuda.synchronize()
    first_metrics = compare_to_reference(first, expected)
    assert bool(torch.isfinite(first).all().item())
    assert first_metrics.cos > 0.999, first_metrics

    # The arena legitimately preserves the padded half of each 128-u32
    # swizzle block. Poison it all, then prove FC1 rewrites every logical lane
    # and FC2 ignores every inactive lane on the same-binding replay.
    binding.intermediate_cache2.fill_(float("nan"))
    replay = binding.run().clone()
    torch.cuda.synchronize()
    replay_metrics = compare_to_reference(replay, expected)
    assert bool(torch.isfinite(replay).all().item())
    assert replay_metrics.cos > 0.999, replay_metrics
    torch.testing.assert_close(replay, first, atol=0.0, rtol=0.0)

    # Capture the same direct path once, then repeatedly replay it after
    # poisoning the arena tail. Graph replay does not re-enter the Python spy,
    # so three launches prove two eager calls plus one captured custom-op node.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = binding.run()
    for _ in range(16):
        binding.intermediate_cache2.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    graph_metrics = compare_to_reference(graph_output, expected)
    assert bool(torch.isfinite(graph_output).all().item())
    assert graph_metrics.cos > 0.999, graph_metrics
    torch.testing.assert_close(graph_output, first, atol=0.0, rtol=0.0)
    assert direct_launches == [(m, intermediate_size)] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "intermediate_size",
    [144, 352, 464, 480, 496, 512],
)
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
def test_w4a16_modelopt_direct_non64_intermediate_is_bounds_safe(
    m: int,
    activation: str,
    intermediate_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the exact native row tail under CUDA memcheck.

    I=144 exercises the narrow one-chunk kernel: it has 18 logical packed-u32
    values per W2 row while the ModelOpt scale grid pads its control geometry
    to 24. I=352 exercises a partial scale/control tail in the wide kernel.
    I=496 is the boundary case where that padded control grid looks completely
    full even though the final two activation lanes are unwritten. Selecting
    the only expert makes an unmasked W2 tail load cross the tensor allocation
    instead of merely reading the next expert. Run this test under
    compute-sanitizer with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` to audit
    m==1/m>=2 and ungated/gated direct-FC2 implementations.
    """
    import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    experts, hidden_size, topk = 1, 128, 1
    monkeypatch.setenv("SPARKINFER_W4A16_SMALL_M_DIRECT", "1")
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat
    direct_launches: list[tuple[int, int]] = []

    def spy_direct_launch(*args, **kwargs) -> None:
        direct_launches.append((int(kwargs["m"]), int(kwargs["intermediate_size"])))
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    torch.manual_seed(
        20260731 + m + intermediate_size + (1000 if activation == "silu" else 0)
    )

    rows = intermediate_size * (2 if activation == "silu" else 1)
    w13_dense = (
        torch.randn(
            experts,
            rows,
            hidden_size,
            device="cuda",
        )
        * 0.12
    )
    w2_dense = (
        torch.randn(
            experts,
            hidden_size,
            intermediate_size,
            device="cuda",
        )
        * 0.12
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w13, w13_blockscale = _quantize_dense_moe_weight_storage(
        w13_dense,
        w13_global_scale,
    )
    w2, w2_blockscale = _quantize_dense_moe_weight_storage(
        w2_dense,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.ones((m, topk), device="cuda", dtype=torch.float32)
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")

    experts_w4a16 = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    prepared = experts_w4a16.representation_for("w4a16")
    assert prepared.weight_layout == "modelopt"
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout=prepared.weight_layout,
        w13_layout="w13",
        scale_format="e4m3_k16",
    )

    binding = make_tp_moe_fp4_binding(
        a=x,
        experts=experts_w4a16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_mode="w4a16",
    )
    expected = moe_reference_w4a16_f32(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
    )

    first = binding.run().clone()
    binding.intermediate_cache2.fill_(float("nan"))
    replay = binding.run().clone()
    torch.cuda.synchronize()
    for actual in (first, replay):
        metrics = compare_to_reference(actual, expected)
        assert bool(torch.isfinite(actual).all().item())
        assert metrics.cos > 0.999, metrics
    assert direct_launches == [(m, intermediate_size), (m, intermediate_size)]
    torch.testing.assert_close(first, replay, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_w4a16_tc_decode_fused_sum_matches_oracle(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TC-decode folds the top-k sum into the FC2 store via atomic accumulate.
    Validate the epilogue across the whole small-M range, not just powers of
    two, since 3/5/6/7 were never exercised through it before. TC-decode is
    unconditional for packed small-M, so no toggle is needed."""
    import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    # Spy on the fused compile so we can assert the fused-sum path actually engaged
    # (a silent fallback to the packed GEMM would also pass the cosine gate).
    real_compile = w4a16_kernel.compile_w4a16_fused_moe
    saw_tc_decode: list[bool] = []

    def _spy_compile(*args, **kwargs):
        saw_tc_decode.append(bool(kwargs.get("tc_decode_fused_sum", False)))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(w4a16_kernel, "compile_w4a16_fused_moe", _spy_compile)

    torch.manual_seed(20260529 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    assert prepared.weight_layout == "packed"
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    tiny_route_workspace = torch.empty((1,), dtype=torch.int32, device=x.device)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny_route_workspace,
        block_expert_ids=tiny_route_workspace,
        packed_route_count=tiny_route_workspace,
        expert_offsets=tiny_route_workspace,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert any(saw_tc_decode), (
        f"TC-decode fused-sum path did not engage for m={m}; "
        f"compile calls: {saw_tc_decode}"
    )
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 2, 3, 6, 8])
def test_w4a16_tc_decode_preplanned_launch_matches_oracle(m: int) -> None:
    """The vLLM binding path passes a *preplanned* fused launch. Validate that a
    preplanned TC-decode launch (direct_topk_routes + tc_decode_fused_sum) is
    consumed correctly by run_w4a16_moe (contract validation + guard + epilogue)."""
    torch.manual_seed(20260529 + 100 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights, activation=activation, params_dtype=x.dtype
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=x.dtype, device=x.device
    )
    tiny = torch.empty((1,), dtype=torch.int32, device=x.device)

    # Build the preplanned TC-decode launch exactly as the prewarm does.
    props = torch.cuda.get_device_properties(x.device)
    block_size_m = select_route_block_size_m(m, topk, experts)
    tc_launch = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size_m,
        max_m_blocks=m * topk,
        element_dtype="bf16",
        sms=int(props.multi_processor_count),
        max_shared_mem=int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        ),
        weight_layout="packed",
        scale_format="e4m3_k16",
        w13_layout="packed",
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
    )
    assert bool(tc_launch.tc_decode_fused_sum)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny,
        block_expert_ids=tiny,
        packed_route_count=tiny,
        expert_offsets=tiny,
        fused_launch=tc_launch,
    )
    expected = _reference_w4a16(
        x, *weights, topk_ids, topk_weights, activation=activation
    )
    torch.cuda.synchronize()
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 4, 6])
def test_w4a16_small_m_packed_direct_topk_routes_matches_oracle(m: int) -> None:
    torch.manual_seed(20260524 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    tiny_route_workspace = torch.empty((1,), dtype=torch.int32, device=x.device)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny_route_workspace,
        block_expert_ids=tiny_route_workspace,
        packed_route_count=tiny_route_workspace,
        expert_offsets=tiny_route_workspace,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 8])
def test_w4a16_mapped_decode_consumes_global_map_without_route_pack(
    m: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730 + m)
    global_experts, local_experts = 12, 8
    hidden_size, intermediate_size = 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    expert_map = torch.full(
        (global_experts,), -1, dtype=torch.int32, device="cuda"
    )
    mapped_global_ids = torch.tensor(
        [0, 2, 3, 5, 7, 8, 10, 11], dtype=torch.int32, device="cuda"
    )
    expert_map[mapped_global_ids] = torch.arange(
        local_experts, dtype=torch.int32, device="cuda"
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        global_experts,
        (m, topk),
        dtype=torch.int32,
        device="cuda",
    )
    # Exercise both a fully inactive token and mixed valid/inactive routes.
    topk_ids[0] = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    if m > 1:
        topk_ids[1] = torch.tensor([2, 6], dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
        route_num_experts=global_experts,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            expert_map=expert_map,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    def route_pack_must_not_run(*args, **kwargs):
        raise AssertionError("mapped decode unexpectedly invoked route packing")

    compiled_launches = []
    compile_fused = w4a16_kernel.compile_w4a16_fused_moe

    def record_compile(*args, **kwargs):
        launch = compile_fused(*args, **kwargs)
        if kwargs.get("use_expert_map"):
            compiled_launches.append(launch)
        return launch

    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            w4a16_kernel,
            "pack_topk_routes_by_expert",
            route_pack_must_not_run,
        )
        patch.setattr(w4a16_kernel, "compile_w4a16_fused_moe", record_compile)
        eager = run().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(eager, expected, activation=activation)
    assert torch.equal(captured, eager)
    assert any(
        launch.use_expert_map
        and launch.direct_topk_routes
        and launch.tc_decode_fused_sum
        for launch in compiled_launches
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_scratch_plan_mapped_decode_is_graph_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparkinfer.moe import fused_moe
    from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260731)
    global_experts, local_experts = 12, 8
    hidden_size, intermediate_size = 128, 128
    topk, activation = 2, "silu"
    device = torch.device("cuda", torch.cuda.current_device())
    w1 = torch.randint(
        0,
        256,
        (local_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (local_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w1_scale = _pattern_e8m0(
        (local_experts, 2 * intermediate_size, hidden_size // 32)
    )
    w2_scale = _pattern_e8m0(
        (local_experts, hidden_size, intermediate_size // 32),
        offset=1,
    )
    w1_global = torch.ones(local_experts, dtype=torch.float32, device=device)
    w2_global = torch.ones(local_experts, dtype=torch.float32, device=device)
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="fp4_e8m0_k32",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w13",
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w1,
        w1_blockscale=w1_scale,
        w1_global_scale=w1_global,
        a1_gscale=torch.ones_like(w1_global),
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=w2_global,
        a2_gscale=torch.ones_like(w2_global),
    )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=8,
            num_topk=topk,
            route_num_experts=global_experts,
            device=device,
            weight_plan=prepared.plan,
            quant_mode="w4a16",
            core_token_counts=(1, 8),
            frozen=True,
        )
    )
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    x = (torch.randn(1, hidden_size, device=device) * 0.25).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32, device=device)
    expert_map = torch.full(
        (global_experts,), -1, dtype=torch.int32, device=device
    )
    expert_map[::2] = torch.arange(
        local_experts - 2, dtype=torch.int32, device=device
    )
    output = torch.empty_like(x)
    binding = plan.bind(
        scratch=scratch,
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
        unit_scale_contract=True,
        route_expert_map=expert_map,
    )
    legacy_ids = torch.zeros_like(topk_ids)
    legacy_weights = torch.zeros_like(topk_weights)
    legacy_weights[0, 0] = topk_weights[0, 0]
    legacy_output = torch.empty_like(x)
    legacy_binding = plan.bind(
        scratch=scratch,
        a=x,
        experts=prepared,
        topk_weights=legacy_weights,
        topk_ids=legacy_ids,
        output=legacy_output,
        input_scales_static=True,
        unit_scale_contract=True,
    )

    def route_pack_must_not_run(*args, **kwargs):
        raise AssertionError("mapped scratch-plan decode invoked route packing")

    compiled_launches = []
    compile_fused = w4a16_kernel.compile_w4a16_fused_moe

    def record_compile(*args, **kwargs):
        launch = compile_fused(*args, **kwargs)
        if kwargs.get("use_expert_map"):
            compiled_launches.append(launch)
        return launch

    expected = fused_moe.run(binding=legacy_binding).clone()
    with monkeypatch.context() as patch:
        patch.setattr(
            w4a16_kernel,
            "pack_topk_routes_by_expert",
            route_pack_must_not_run,
        )
        patch.setattr(w4a16_kernel, "compile_w4a16_fused_moe", record_compile)
        eager = fused_moe.run(binding=binding).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_moe.run(binding=binding)
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(eager, expected)
    assert torch.equal(captured, eager)
    assert any(
        launch.use_expert_map
        and launch.direct_topk_routes
        and launch.tc_decode_fused_sum
        for launch in compiled_launches
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_activation_amax_calibration_tracks_routed_inputs_and_fc2_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260616)
    experts, hidden_size, intermediate_size = 4, 128, 128
    m, topk, activation = 3, 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    activation_amax = torch.zeros((2, experts, 2), dtype=torch.float32, device="cuda")
    activation_amax[0].fill_(123.0)
    compile_calls: list[dict[str, object]] = []

    def spy_compile_w4a16_fused_moe(**kwargs):
        compile_calls.append(dict(kwargs))
        return compile_w4a16_fused_moe(**kwargs)

    monkeypatch.setattr(
        "sparkinfer.moe._shared.kernels.w4a16.kernel.compile_w4a16_fused_moe",
        spy_compile_w4a16_fused_moe,
    )

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        activation_amax=activation_amax,
        layer_idx=1,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert compile_calls
    assert any(bool(call["collect_activation_amax"]) for call in compile_calls)
    assert not any(bool(call["direct_topk_routes"]) for call in compile_calls)
    assert not any(bool(call["tc_decode_fused_sum"]) for call in compile_calls)
    _assert_matches_oracle(actual, expected, activation=activation)

    expected_amax = torch.zeros((experts, 2), dtype=torch.float32, device="cuda")
    activated_rows = buffers.intermediate_cache2.view(-1, intermediate_size)
    flat_ids = topk_ids.view(-1)
    route_tokens = torch.arange(m, device="cuda").repeat_interleave(topk)
    route_indices = torch.arange(m * topk, device="cuda")
    for expert in range(experts):
        mask = flat_ids == expert
        if bool(mask.any().item()):
            expected_amax[expert, 0] = x[route_tokens[mask]].float().abs().max()
            expected_amax[expert, 1] = (
                activated_rows[route_indices[mask]].float().abs().max()
            )

    torch.testing.assert_close(activation_amax[1], expected_amax, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        activation_amax[0],
        torch.full_like(activation_amax[0], 123.0),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_activation_amax_forces_main_kernel_over_native_small_m_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260617)
    experts, hidden_size, intermediate_size = 4, 128, 128
    rows = intermediate_size * 2
    m, topk, activation = 2, 2, "silu"
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w13",
        scale_format="e8m0_k32",
    )
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    activation_amax = torch.zeros((1, experts, 2), dtype=torch.float32, device="cuda")
    compile_calls: list[dict[str, object]] = []

    def spy_compile_w4a16_fused_moe(**kwargs):
        compile_calls.append(dict(kwargs))
        return compile_w4a16_fused_moe(**kwargs)

    monkeypatch.setattr(
        "sparkinfer.moe._shared.kernels.w4a16.kernel.compile_w4a16_fused_moe",
        spy_compile_w4a16_fused_moe,
    )

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        activation_amax=activation_amax,
        layer_idx=0,
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        w13_layout="w13",
    )
    torch.cuda.synchronize()

    assert compile_calls
    assert any(bool(call["collect_activation_amax"]) for call in compile_calls)
    assert not any(bool(call["direct_topk_routes"]) for call in compile_calls)
    assert not any(bool(call["tc_decode_fused_sum"]) for call in compile_calls)
    assert bool((activation_amax > 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize(
    ("routed_size", "m"),
    [(8, 16), (16, 32), (32, 64), (48, 128), (64, 192)],
)
def test_w4a16_moe_matches_oracle(
    activation: str,
    routed_size: int,
    m: int,
) -> None:
    torch.manual_seed(20260515 + routed_size + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk = 2
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_situ_moe_matches_oracle() -> None:
    torch.manual_seed(20260726)
    experts, hidden_size, intermediate_size = 8, 128, 128
    m, topk = 16, 2
    activation = "situ"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        experts,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual.abs().sum().item() > 0
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_modelopt_nvfp4_prepare_moe_matches_oracle(
    activation: str,
) -> None:
    torch.manual_seed(20260523 + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prepare_native", [False, True])
@pytest.mark.parametrize("w13_layout", ["up_gate", "gate_up"])
def test_w4a16_modelopt_nvfp4_explicit_w13_layout_matches_oracle(
    prepare_native: bool,
    w13_layout: str,
) -> None:
    """W13 physical order is an explicit input, independent of source_format."""
    torch.manual_seed(20260525 + (1000 if prepare_native else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    source_w13 = w13
    source_w13_blockscale = w13_blockscale
    if w13_layout == "gate_up":
        half = intermediate_size
        source_w13 = torch.cat([w13[:, half:], w13[:, :half]], dim=1).contiguous()
        source_w13_blockscale = torch.cat(
            [w13_blockscale[:, half:], w13_blockscale[:, :half]], dim=1
        ).contiguous()

    prepare = (
        prepare_w4a16_modelopt_native_weights
        if prepare_native
        else prepare_w4a16_weights
    )
    kwargs = {"source_format": "modelopt_nvfp4"} if prepare_native else {}
    prepared = prepare(
        source_w13,
        source_w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=x.dtype,
        w13_layout=w13_layout,
        **kwargs,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_tp_moe_w4a16_modelopt_nvfp4_uses_normal_nvfp4_scale_contract(
    activation: str,
) -> None:
    torch.manual_seed(20260524 + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    w13_input_scale = (torch.rand(experts, device="cuda") * 2.0 + 1.5).to(torch.float32)
    w2_input_scale = (torch.rand(experts, device="cuda") * 2.0 + 1.5).to(torch.float32)
    a1_gscale = (1.0 / w13_input_scale).contiguous()
    a2_gscale = (1.0 / w2_input_scale).contiguous()

    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    output = torch.empty_like(x)
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a1_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a2_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    actual = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tp_moe_w4a16_prepared_reuse_path_is_deterministic_under_odd_shape_stress() -> (
    None
):
    torch.manual_seed(20260526)
    experts, hidden_size, intermediate_size = 8, 128, 128
    activation = "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    reference_weights = tuple(t.clone() for t in weights)
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
    weight_plan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    prepared = prepare_sparkinfer_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_global_scale,
        a1_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_global_scale,
        a2_gscale=a_gscale,
        params_dtype=torch.bfloat16,
    )
    assert prepared.representation_for("w4a16") is not None

    cases = [
        (1, 1, 1, torch.int32, 0.25),
        (3, 5, 2, torch.int32, 0.50),
        (5, 7, 3, torch.int64, 0.75),
        (9, 11, 4, torch.int32, 1.00),
    ]
    for case_idx, (batch_size, seq_len, topk, ids_dtype, input_scale) in enumerate(
        cases
    ):
        m = batch_size * seq_len
        torch.manual_seed(2026052600 + case_idx)
        x = (torch.randn(m, hidden_size, device="cuda") * input_scale).to(
            torch.bfloat16
        )
        topk_ids = torch.randint(
            0,
            experts,
            (m, topk),
            device="cuda",
            dtype=ids_dtype,
        )
        topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
        expected = moe_reference_w4a16_f32(
            x,
            *reference_weights,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )

        baseline = None
        for repeat in range(6):
            output = torch.empty_like(x)
            actual = run_tp_moe_fp4(
                a=x,
                experts=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
                quant_mode="w4a16",
            )
            torch.cuda.synchronize()
            assert actual is output
            if baseline is None:
                baseline = output.detach().clone()
                continue
            if not torch.equal(output, baseline):
                max_abs = (output.float() - baseline.float()).abs().max().item()
                raise AssertionError(
                    "W4A16 prepared reuse path changed output for "
                    f"case={case_idx}, repeat={repeat}, m={m}, topk={topk}, "
                    f"ids_dtype={ids_dtype}, max_abs={max_abs}"
                )

        assert baseline is not None
        metrics = compare_to_reference(baseline, expected)
        assert metrics.cos > 0.999, (
            metrics,
            batch_size,
            seq_len,
            topk,
            ids_dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_moe_matches_oracle_with_expert_map(
    activation: str,
) -> None:
    torch.manual_seed(20260516 + (1000 if activation == "silu" else 0))
    global_experts, local_experts = 8, 4
    hidden_size, intermediate_size = 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[::2] = torch.arange(local_experts, dtype=torch.int32, device="cuda")

    valid_global_ids = torch.arange(
        0, global_experts, 2, dtype=torch.int32, device="cuda"
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = valid_global_ids[
        torch.randint(0, local_experts, (m, topk), device="cuda")
    ].to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_moe_swiglu_limit_matches_oracle_under_cuda_graph() -> None:
    torch.manual_seed(20260519)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "silu"
    swiglu_limit = 10.0
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, _, w2, w2_blockscale, w2_global_scale = weights
    w13_global_scale = torch.full((experts,), 8.0, dtype=torch.float32, device="cuda")
    weights = (
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 2.0).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    eager = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        swiglu_limit=swiglu_limit,
    )
    torch.cuda.synchronize()
    _assert_matches_oracle(eager, expected, activation=activation)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            swiglu_limit=swiglu_limit,
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_preplanned_capacity_launch_accepts_smaller_live_m() -> None:
    torch.manual_seed(20260522)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, live_m, capacity_m = 2, 24, 32
    activation = "relu2"
    assert select_route_block_size_m(
        live_m, topk, experts
    ) != select_route_block_size_m(capacity_m, topk, experts)
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(live_m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0, experts, (live_m, topk), device="cuda", dtype=torch.int32
    )
    topk_weights = torch.softmax(torch.randn(live_m, topk, device="cuda"), dim=-1)
    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=capacity_m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    props = torch.cuda.get_device_properties(x.device)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    block_size_m = select_route_block_size_m(capacity_m, topk, experts)
    route_slots = max_packed_route_slots(capacity_m * topk, block_size_m, experts)
    max_m_blocks = (route_slots + block_size_m - 1) // block_size_m
    fused_launch = compile_w4a16_fused_moe(
        size_m=capacity_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size_m,
        max_m_blocks=max_m_blocks,
        element_dtype="bf16",
        sms=int(props.multi_processor_count),
        max_shared_mem=max_shared_mem,
    )
    topk_sum_launch = compile_w4a16_topk_sum(
        m=capacity_m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype="bf16",
    )
    output = torch.empty_like(x)

    def _run(output_buffer: torch.Tensor) -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=output_buffer,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
        )

    actual = _run(output)
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)

    graph_output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        _run(graph_output)
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(graph_output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tp_moe_w4a16_dispatch_uses_native_path() -> None:
    torch.manual_seed(20260518)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "relu2"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights

    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    output = torch.empty_like(x)
    a_gscale = torch.ones((), dtype=torch.float32, device="cuda")
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    actual = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)
