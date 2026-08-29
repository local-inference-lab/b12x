from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import torch

import b12x
from b12x.sequence import gdn_decode as gdn

from ..conftest import require_b12x as require_sm120


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(dtype)
        .contiguous()
    )


def _make_case(
    *,
    device: torch.device,
    query_lengths: tuple[int, ...] = (3, 1),
    heads: int = 2,
    columns: int = 3,
    max_tokens: int = 5,
    state_dtype: torch.dtype = torch.float32,
    null_state_index: int | None = None,
    raw_beta_stride: int = 1,
) -> gdn.KdaBinding:
    max_seqs = len(query_lengths)
    live_tokens = sum(query_lengths)
    state_slots = max_seqs * columns + 1
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=state_slots,
        key_heads=heads,
        value_heads=heads,
        state_index_columns=columns,
        state_dtype=state_dtype,
        gate_activation="sigmoid",
        null_state_index=null_state_index,
    )
    plan = gdn.plan(caps)
    (scratch_spec,) = plan.scratch_specs()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    state_indices = torch.arange(
        max_seqs * columns, dtype=torch.int64, device=device
    ).view(max_seqs, columns)
    raw_beta_storage = _randn(
        (max_tokens, heads * raw_beta_stride), device=device
    )
    raw_beta = raw_beta_storage[:, ::raw_beta_stride]
    return gdn.bind_kda(
        plan,
        scratch=torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        mixed_qkv=_randn((max_tokens, caps.packed_qkv_width), device=device),
        raw_g=_randn((max_tokens, heads, 128), device=device),
        raw_beta=raw_beta,
        z=_randn((max_tokens, heads, 128), device=device),
        A_log=_randn((heads,), device=device, dtype=torch.float32, scale=0.1),
        dt_bias=_randn(
            (heads, 128), device=device, dtype=torch.float32, scale=0.1
        ),
        norm_weight=(
            1.0
            + _randn((128,), device=device, dtype=torch.float32, scale=0.05)
        ).contiguous(),
        recurrent_state=_randn(
            (state_slots, heads, 128, 128),
            device=device,
            dtype=state_dtype,
            scale=0.1,
        ),
        query_start_loc=query_start_loc,
        num_accepted_tokens=torch.tensor(
            [min(length, 2) for length in query_lengths],
            dtype=torch.int32,
            device=device,
        ),
        state_indices=state_indices,
        num_seqs=torch.tensor([max_seqs], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([live_tokens], dtype=torch.int32, device=device),
        output=torch.empty(
            (max_tokens, heads, 128), dtype=torch.bfloat16, device=device
        ),
    )


def _reference(binding: gdn.KdaBinding, state: torch.Tensor) -> torch.Tensor:
    caps = binding.plan.caps
    return gdn.reference.decode_kda(
        binding.mixed_qkv,
        binding.raw_g,
        binding.raw_beta,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        heads=caps.value_heads,
        qk_l2norm=caps.qk_l2norm,
        null_state_index=caps.null_state_index,
    )


def test_reference_applies_per_key_lower_bounded_decay() -> None:
    torch.manual_seed(19)
    device = torch.device("cpu")
    mixed_qkv = _randn((1, 384), device=device)
    raw_g = _randn((1, 1, 128), device=device)
    raw_beta = _randn((1, 1), device=device)
    z = _randn((1, 1, 128), device=device)
    A_log = torch.tensor([0.2], dtype=torch.float32)
    dt_bias = torch.linspace(-0.4, 0.3, 128, dtype=torch.float32).view(1, 128)
    norm_weight = torch.linspace(0.8, 1.2, 128, dtype=torch.float32)
    state = _randn((1, 1, 128, 128), device=device, dtype=torch.float32)
    initial_state = state.clone()
    state_indices = torch.zeros((1, 1), dtype=torch.int64)

    actual = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        state_indices,
        1,
        1,
        heads=1,
    )

    q, k, value = mixed_qkv.float().view(3, 128).unbind(dim=0)
    q *= torch.rsqrt(q.square().sum() + 1e-6)
    k *= torch.rsqrt(k.square().sum() + 1e-6)
    q *= 128**-0.5
    log_decay = -5.0 * torch.sigmoid(
        torch.exp(A_log[0]) * (raw_g[0, 0].float() + dt_bias[0])
    )
    expected_state = initial_state[0, 0] * torch.exp(log_decay).unsqueeze(0)
    delta = value - expected_state.mv(k)
    expected_state += (
        delta * torch.sigmoid(raw_beta[0, 0].float())
    ).unsqueeze(1) * k.unsqueeze(0)
    core = expected_state.mv(q).to(torch.bfloat16)
    normalized = core.float() * torch.rsqrt(core.float().square().mean() + 1e-6)
    expected = (
        normalized * norm_weight * torch.sigmoid(z[0, 0].float())
    ).to(torch.bfloat16)

    torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(state[0, 0], expected_state, rtol=0, atol=0)


def test_reference_keeps_kda_rmsnorm_in_fp32_until_final_store() -> None:
    device = torch.device("cpu")
    core = torch.linspace(-1.3, 1.7, 128, dtype=torch.float32).to(torch.bfloat16)
    q = torch.zeros(128, dtype=torch.bfloat16)
    q[0] = 1
    k = q.clone()
    value = core.clone()
    mixed_qkv = torch.cat((q, k, value)).view(1, -1)
    raw_g = torch.full((1, 1, 128), -100, dtype=torch.bfloat16)
    raw_beta = torch.full((1, 1), 100, dtype=torch.bfloat16)
    z = torch.linspace(-2, 2, 128, dtype=torch.float32).to(torch.bfloat16).view(
        1, 1, 128
    )
    norm_weight = torch.linspace(0.7, 1.4, 128, dtype=torch.float32).to(
        torch.bfloat16
    )
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

    actual = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        torch.zeros(1, dtype=torch.float32),
        torch.zeros((1, 128), dtype=torch.float32),
        norm_weight,
        state,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int64),
        1,
        1,
        heads=1,
    )[0, 0]

    decoded = (core.float() * (128**-0.5)).to(torch.bfloat16)
    normalized = decoded.float() * torch.rsqrt(
        decoded.float().square().mean() + 1e-6
    )
    gate = torch.sigmoid(z[0, 0].float())
    expected = (normalized * norm_weight.float() * gate).to(torch.bfloat16)
    old_intermediate_bf16 = (
        normalized.to(torch.bfloat16).float() * norm_weight.float() * gate
    ).to(torch.bfloat16)

    assert torch.count_nonzero(expected != old_intermediate_bf16) > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_null_state_sentinel_zeroes_requests_and_skips_writes() -> None:
    torch.manual_seed(23)
    device = torch.device("cpu")
    heads = 1
    max_tokens = 2
    mixed_qkv = _randn((max_tokens, heads * 384), device=device)
    raw_g = _randn((max_tokens, heads, 128), device=device)
    raw_beta = _randn((max_tokens, heads), device=device)
    z = _randn((max_tokens, heads, 128), device=device)
    A_log = _randn((heads,), device=device, dtype=torch.float32)
    dt_bias = _randn((heads, 128), device=device, dtype=torch.float32)
    norm_weight = torch.ones(128, dtype=torch.float32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    accepted = torch.full((2,), 2, dtype=torch.int32)
    state = _randn((3, heads, 128, 128), device=device, dtype=torch.float32)
    before = state.clone()

    output = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        query_start_loc,
        accepted,
        torch.tensor([[1, 0], [2, 0]], dtype=torch.int64),
        2,
        2,
        heads=heads,
        null_state_index=0,
    )

    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(state, before, rtol=0, atol=0)

    output = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        query_start_loc,
        torch.full((2,), 2, dtype=torch.int32),
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int64),
        2,
        2,
        heads=heads,
        null_state_index=0,
    )

    assert torch.count_nonzero(output) > 0
    torch.testing.assert_close(state, before, rtol=0, atol=0)


def test_kda_rmsnorm_kernel_avoids_intermediate_bf16_rounding() -> None:
    device = require_sm120()
    from b12x.sequence.gdn_decode._kernels import _gated_rmsnorm_kernel

    eps = 1e-5
    output = (
        torch.linspace(-1.3, 1.7, 128, dtype=torch.float32, device=device)
        .to(torch.bfloat16)
        .view(1, 1, 128)
    )
    core = output.clone()
    z = torch.full_like(output, 100)
    norm_weight = torch.linspace(
        0.7, 1.4, 128, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    error_code = torch.zeros(1, dtype=torch.int32, device=device)

    _gated_rmsnorm_kernel[(1,)](
        output,
        z,
        norm_weight,
        num_tokens,
        error_code,
        eps,
        stride_output_token=output.stride(0),
        stride_output_head=output.stride(1),
        stride_z_token=z.stride(0),
        stride_z_head=z.stride(1),
        MAX_TOKENS=1,
        VALUE_HEADS=1,
        VALUE_HEAD_DIM=128,
        SIGMOID_GATE=True,
        NORM_WEIGHT_FP32=False,
        KDA_NORM_FP32=True,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize(device)

    values = core[0, 0].float()
    normalized = values * torch.rsqrt(values.square().mean() + eps)
    expected = (normalized * norm_weight.float()).to(torch.bfloat16)
    old_intermediate_bf16 = (
        normalized.to(torch.bfloat16) * norm_weight
    ).to(torch.bfloat16)
    expected_delta = torch.count_nonzero(expected != old_intermediate_bf16)
    assert expected_delta >= 16
    new_error = (output[0, 0].float() - expected.float()).abs().sum()
    old_error = (
        output[0, 0].float() - old_intermediate_bf16.float()
    ).abs().sum()
    assert new_error < old_error


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_packed_kda_matches_reference(state_dtype: torch.dtype) -> None:
    device = require_sm120()
    binding = _make_case(device=device, state_dtype=state_dtype)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state,
        state_reference,
        rtol=1e-2 if state_dtype == torch.bfloat16 else 1e-5,
        atol=8e-3 if state_dtype == torch.bfloat16 else 2e-5,
    )
    assert torch.count_nonzero(actual[4:]) == 0


def test_glm53_tp8_head_geometry_matches_reference() -> None:
    device = require_sm120()
    binding = _make_case(device=device, heads=8)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert binding.mixed_qkv.shape == (5, 3072)
    assert binding.raw_g.shape == (5, 8, 128)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_subcapacity_matches_reference_and_replays_cuda_graph() -> None:
    device = require_sm120()
    source = _make_case(
        device=device,
        query_lengths=(2, 1, 1),
        max_tokens=6,
        raw_beta_stride=2,
    )
    plan = gdn.plan(replace(source.plan.caps, max_tokens=8, max_seqs=4))
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape, dtype=scratch_spec.dtype, device=device
    )
    first = gdn.bind_kda(
        plan,
        scratch=scratch,
        mixed_qkv=source.mixed_qkv[:5],
        raw_g=source.raw_g[:5],
        raw_beta=source.raw_beta[:5],
        z=source.z[:5],
        A_log=source.A_log,
        dt_bias=source.dt_bias,
        norm_weight=source.norm_weight,
        recurrent_state=source.recurrent_state,
        query_start_loc=source.query_start_loc[:3],
        num_accepted_tokens=source.num_accepted_tokens[:2],
        state_indices=source.state_indices[:2],
        num_seqs=torch.tensor([2], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([3], dtype=torch.int32, device=device),
        output=source.output[:5],
    )
    state_reference = first.recurrent_state.clone()
    expected = _reference(first, state_reference)

    actual = gdn.run_kda(first)
    torch.cuda.synchronize(device)

    assert first.plan.caps.max_tokens == 8
    assert first.mixed_qkv.shape[0] == 5
    assert first.num_accepted_tokens.shape[0] == 2
    assert not first.raw_beta.is_contiguous()
    assert actual.data_ptr() == source.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        first.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )

    from b12x.sequence.gdn_decode import _kernels

    triton_kernels = (
        _kernels._reset_validation_kernel,
        _kernels._validate_packed_metadata_kernel,
        _kernels._validate_active_state_slots_kernel,
        _kernels._packed_sequential_kda_decode_kernel,
        _kernels._gated_rmsnorm_kernel,
    )
    device_index = torch.cuda.current_device()

    def cache_sizes() -> tuple[int, ...]:
        return tuple(
            len(kernel.device_caches[device_index][0]) for kernel in triton_kernels
        )

    first_cache_sizes = cache_sizes()
    second = gdn.bind_kda(
        plan,
        scratch=scratch,
        mixed_qkv=source.mixed_qkv,
        raw_g=source.raw_g,
        raw_beta=source.raw_beta,
        z=source.z,
        A_log=source.A_log,
        dt_bias=source.dt_bias,
        norm_weight=source.norm_weight,
        recurrent_state=source.recurrent_state,
        query_start_loc=source.query_start_loc,
        num_accepted_tokens=source.num_accepted_tokens,
        state_indices=source.state_indices,
        num_seqs=source.num_seqs,
        num_tokens=source.num_tokens,
        output=source.output,
    )
    state_reference = second.recurrent_state.clone()
    expected = _reference(second, state_reference)

    b12x.freeze_kernel_resolution("reduced-capacity KDA regression")
    try:
        actual = gdn.run_kda(second)
        torch.cuda.synchronize(device)
    finally:
        b12x.unfreeze_kernel_resolution()

    assert second.mixed_qkv.shape[0] == 6
    assert second.num_accepted_tokens.shape[0] == 3
    assert cache_sizes() == first_cache_sizes
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        second.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(second)

    second.mixed_qkv.copy_(torch.randn_like(second.mixed_qkv).mul_(0.2))
    second.raw_g.copy_(torch.randn_like(second.raw_g).mul_(0.2))
    second.raw_beta.copy_(torch.randn_like(second.raw_beta).mul_(0.2))
    second.z.copy_(torch.randn_like(second.z).mul_(0.2))
    second.recurrent_state.copy_(
        torch.randn_like(second.recurrent_state).mul_(0.1)
    )
    state_reference = second.recurrent_state.clone()
    expected = _reference(second, state_reference)

    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        second.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_subcapacity_metadata_errors_are_transactional() -> None:
    device = require_sm120()
    source = _make_case(device=device)
    plan = gdn.plan(replace(source.plan.caps, max_tokens=6, max_seqs=4))
    (scratch_spec,) = plan.scratch_specs()
    binding = gdn.bind_kda(
        plan,
        scratch=torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        mixed_qkv=source.mixed_qkv,
        raw_g=source.raw_g,
        raw_beta=source.raw_beta,
        z=source.z,
        A_log=source.A_log,
        dt_bias=source.dt_bias,
        norm_weight=source.norm_weight,
        recurrent_state=source.recurrent_state,
        query_start_loc=source.query_start_loc,
        num_accepted_tokens=source.num_accepted_tokens,
        state_indices=source.state_indices,
        num_seqs=source.num_seqs,
        num_tokens=source.num_tokens,
        output=source.output,
    )
    before = binding.recurrent_state.clone()

    def assert_rejected() -> None:
        actual = gdn.run_kda(binding)
        torch.cuda.synchronize(device)

        assert binding.error_code.item() & 2
        assert torch.isnan(actual).all()
        torch.testing.assert_close(
            binding.recurrent_state, before, rtol=0, atol=0
        )

    binding.num_seqs.fill_(1)
    assert_rejected()

    binding.num_seqs.fill_(2)
    binding.query_start_loc.copy_(
        torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    )
    assert_rejected()


def test_kda_rejected_draft_restarts_from_accepted_checkpoint() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    binding.num_accepted_tokens[0] = 1
    gdn.run_kda(binding)
    torch.cuda.synchronize(device)
    accepted_checkpoint = binding.recurrent_state[1].clone()

    binding.num_accepted_tokens[0] = 2
    binding.recurrent_state[2].fill_(73.0)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    torch.testing.assert_close(
        state_reference[1], accepted_checkpoint, rtol=0, atol=0
    )
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_duplicate_state_slot_is_transactional() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    binding.state_indices[0, :3].fill_(1)
    before = binding.recurrent_state.clone()

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 1
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_kda_null_state_sentinel_is_graph_safe_and_immutable() -> None:
    device = require_sm120()
    binding = _make_case(device=device, null_state_index=0)
    binding.state_indices.zero_()
    binding.state_indices[:, 0].copy_(
        torch.arange(1, 3, dtype=binding.state_indices.dtype, device=device)
    )
    binding.num_accepted_tokens.fill_(2)
    before = binding.recurrent_state.clone()

    gdn.run_kda(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(binding)

    output_ptr = captured_output.data_ptr()
    state_ptr = binding.recurrent_state.data_ptr()
    captured_output.fill_(float("nan"))
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    graph.replay()
    torch.cuda.synchronize(device)

    assert binding.error_code.item() == 0
    assert captured_output.data_ptr() == output_ptr
    assert binding.recurrent_state.data_ptr() == state_ptr
    assert torch.count_nonzero(captured_output) == 0
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_kda_cuda_graph_replay_preserves_addresses() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    gdn.run_kda(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(binding)

    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    output_ptr = captured_output.data_ptr()
    state_ptr = binding.recurrent_state.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)

    assert captured_output.data_ptr() == output_ptr
    assert binding.recurrent_state.data_ptr() == state_ptr
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_cuda_graph_replay_accepts_multiple_live_counts() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    gdn.run_kda(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(binding)

    binding.query_start_loc.copy_(
        torch.tensor([0, 2, 2], dtype=torch.int32, device=device)
    )
    binding.num_accepted_tokens.fill_(1)
    binding.num_seqs.fill_(1)
    binding.num_tokens.fill_(2)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    binding.recurrent_state.copy_(
        torch.randn_like(binding.recurrent_state).mul_(0.1)
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_torch_compile_fullgraph_keeps_outer_op_opaque() -> None:
    device = require_sm120()
    binding = _make_case(device=device)

    def launch() -> torch.Tensor:
        return gdn.run_kda(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.recurrent_state.copy_(
        torch.randn_like(binding.recurrent_state).mul_(0.1)
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = compiled()
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_padded_state_slot_past_int32_element_boundary() -> None:
    device = require_sm120()
    heads = 1
    slot_elements = heads * 128 * 128
    slot_stride = slot_elements + 2_048
    tail_slot = (1 << 31) // slot_stride + 1
    assert tail_slot * slot_stride > 1 << 31
    caps = gdn.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        max_state_slots=tail_slot + 1,
        key_heads=heads,
        value_heads=heads,
        state_dtype=torch.bfloat16,
        gate_activation="sigmoid",
    )
    state_storage = torch.empty(
        tail_slot * slot_stride + slot_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(tail_slot + 1, heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
    )
    recurrent_state[tail_slot].copy_(
        _randn((heads, 128, 128), device=device, scale=0.1)
    )
    compact_state = recurrent_state[tail_slot : tail_slot + 1].clone()
    mixed_qkv = _randn((1, caps.packed_qkv_width), device=device)
    raw_g = _randn((1, heads, 128), device=device)
    raw_beta = _randn((1, heads), device=device)
    z = _randn((1, heads, 128), device=device)
    A_log = _randn((heads,), device=device, dtype=torch.float32, scale=0.1)
    dt_bias = _randn(
        (heads, 128), device=device, dtype=torch.float32, scale=0.1
    )
    norm_weight = (
        1.0 + _randn((128,), device=device, dtype=torch.float32, scale=0.05)
    ).contiguous()
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    state_indices = torch.tensor([[tail_slot]], dtype=torch.int64, device=device)
    compact_indices = torch.zeros((1, 1), dtype=torch.int64, device=device)
    num_seqs = torch.ones(1, dtype=torch.int32, device=device)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.empty((1, heads, 128), dtype=torch.bfloat16, device=device)
    planned = gdn.plan(caps)
    (scratch_spec,) = planned.scratch_specs()
    binding = gdn.bind_kda(
        planned,
        scratch=torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        state_indices=state_indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )
    expected = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        compact_state,
        query_start_loc,
        accepted,
        compact_indices,
        num_seqs,
        num_tokens,
        heads=heads,
    )

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        recurrent_state[tail_slot], compact_state[0], rtol=1e-2, atol=8e-3
    )

    del binding, recurrent_state, state_storage
    gc.collect()
    torch.cuda.empty_cache()
